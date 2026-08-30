"""Continuous, graph-observable safety constraints for learned safety filters.

The public seam is :func:`safety_constraint`.  Environment-specific geometry
stays behind that function so algorithms and data collectors do not need to
know how a particular graph encodes agents and obstacles.
"""

import jax.numpy as jnp

from .base import MultiAgentEnv
from .lidar_env.base import LidarEnv
from .vmas.vmas_navigation import VMASNavigation
from .vmas.vmas_navigation_obs import VMASNavigationObs
from ..utils.graph import GraphsTuple
from ..utils.typing import Array


def _velocity(state: Array) -> Array:
    if state.shape[-1] == 5:
        return state[..., 4:5] * state[..., 2:4]
    return state[..., 2:4]


def _braking_margin(relative_position: Array, relative_velocity: Array, braking_accel: float) -> Array:
    distance = jnp.linalg.norm(relative_position, axis=-1)
    direction = relative_position / jnp.maximum(distance[..., None], 1e-6)
    closing_speed = jnp.maximum(-jnp.sum(relative_velocity * direction, axis=-1), 0.0)
    return closing_speed ** 2 / (2.0 * braking_accel)


def _maximum_speed(env: MultiAgentEnv) -> Array:
    lower, upper = env.state_lim()
    if env.state_dim == 5:
        return jnp.maximum(jnp.abs(lower[4]), jnp.abs(upper[4]))
    velocity_limit = jnp.maximum(jnp.abs(lower[2:4]), jnp.abs(upper[2:4]))
    return jnp.linalg.norm(velocity_limit)


def _agent_constraint(
        *,
        positions: Array,
        velocities: Array,
        num_agents: int,
        collision_distance: float,
        sensing_range: float,
        maximum_margin: float,
        braking_accel: float | None,
        max_speed: Array | None,
) -> Array:
    """Minimum locally observable inter-agent clearance for every agent."""
    if num_agents == 1:
        return jnp.full((1,), maximum_margin, dtype=positions.dtype)

    relative_position = positions[:, None, :] - positions[None, :, :]
    distance = jnp.linalg.norm(relative_position, axis=-1)
    relative_velocity = velocities[:, None, :] - velocities[None, :, :]
    braking = 0.0 if braking_accel is None else _braking_margin(
        relative_position, relative_velocity, braking_accel
    )
    worst_braking = (
        0.0
        if braking_accel is None
        else (2.0 * max_speed) ** 2 / (2.0 * braking_accel)
    )
    ceiling = jnp.minimum(
        maximum_margin,
        sensing_range - collision_distance - worst_braking,
    )
    clearance = jnp.minimum(distance - collision_distance - braking, ceiling)
    observable = (distance < sensing_range) & (~jnp.eye(num_agents, dtype=bool))
    clearance = jnp.where(observable, clearance, ceiling)
    return jnp.min(clearance, axis=-1)


def lidar_safety_constraint(
        env: LidarEnv,
        graph: GraphsTuple,
        *,
        agent_margin: float = 0.02,
        obstacle_margin: float = 0.02,
        braking_accel: float | None = 10.0,
        maximum_margin: float | None = None,
) -> Array:
    """Return one continuous safety margin per agent; non-negative is safe.

    Only entities inside the environment's communication/lidar range influence
    the result.  Far or padded entities contribute ``maximum_margin``.  This is
    the simulation adapter for the local-observation interface used by the
    learned filter; it deliberately does not read ``graph.env_states``.
    """
    if maximum_margin is None:
        maximum_margin = env.params["comm_radius"]
    if braking_accel is not None and braking_accel <= 0:
        raise ValueError("braking_accel must be positive or None")

    agent_states = graph.type_states(type_idx=env.AGENT, n_type=env.num_agents)
    positions = agent_states[:, :2]
    velocities = _velocity(agent_states)
    max_speed = None if braking_accel is None else _maximum_speed(env)
    agent_constraint = _agent_constraint(
        positions=positions,
        velocities=velocities,
        num_agents=env.num_agents,
        collision_distance=2.0 * env.params["car_radius"] + agent_margin,
        sensing_range=env.params["comm_radius"],
        maximum_margin=maximum_margin,
        braking_accel=braking_accel,
        max_speed=max_speed,
    )

    if env.params["n_obs"] > 0:
        n_hits = env.params["top_k_rays"] * env.num_agents
        obstacle_states = graph.type_states(type_idx=env.OBS, n_type=n_hits)
        obstacle_states = obstacle_states.reshape(env.num_agents, env.params["top_k_rays"], env.state_dim)
        relative_position = positions[:, None, :] - obstacle_states[..., :2]
        distance = jnp.linalg.norm(relative_position, axis=-1)
        relative_velocity = velocities[:, None, :]
        braking = 0.0 if braking_accel is None else _braking_margin(
            relative_position, relative_velocity, braking_accel
        )
        worst_obstacle_braking = (
            0.0 if braking_accel is None else max_speed ** 2 / (2.0 * braking_accel)
        )
        obstacle_sense_range = env.params["comm_radius"] - 0.1
        obstacle_ceiling = jnp.minimum(
            maximum_margin,
            obstacle_sense_range
            - (env.params["car_radius"] + obstacle_margin)
            - worst_obstacle_braking,
        )
        obstacle_clearance = distance - (env.params["car_radius"] + obstacle_margin) - braking
        obstacle_clearance = jnp.minimum(obstacle_clearance, obstacle_ceiling)
        observable = distance < obstacle_sense_range
        obstacle_clearance = jnp.where(observable, obstacle_clearance, obstacle_ceiling)
        obstacle_constraint = jnp.min(obstacle_clearance, axis=-1)
    else:
        obstacle_constraint = jnp.full((env.num_agents,), maximum_margin, dtype=agent_states.dtype)

    constraint = jnp.minimum(agent_constraint, obstacle_constraint)
    return jnp.minimum(constraint, maximum_margin)


def vmas_navigation_safety_constraint(
        env: VMASNavigation,
        graph: GraphsTuple,
        *,
        agent_margin: float = 0.02,
        obstacle_margin: float = 0.02,
        braking_accel: float | None = None,
        maximum_margin: float | None = None,
) -> Array:
    """Return continuous clearance for the VMAS navigation family.

    The implementation reads only graph state and static environment geometry;
    it does not call the simulator or an explicit dynamics function.  Positive
    values are safe.  When ``braking_accel`` is ``None`` (the default), the
    signal is the instantaneous signed clearance used by the HJ objective.
    """
    if maximum_margin is None:
        maximum_margin = env.params["comm_radius"]
    if braking_accel is not None and braking_accel <= 0:
        raise ValueError("braking_accel must be positive or None")

    agent_states = graph.type_states(type_idx=env.AGENT, n_type=env.num_agents)
    positions = agent_states[:, :2]
    velocities = agent_states[:, 2:4]
    max_speed = None if braking_accel is None else _maximum_speed(env)
    agent_constraint = _agent_constraint(
        positions=positions,
        velocities=velocities,
        num_agents=env.num_agents,
        collision_distance=2.0 * env.agent_radius + agent_margin,
        sensing_range=env.params["comm_radius"],
        maximum_margin=maximum_margin,
        braking_accel=braking_accel,
        max_speed=max_speed,
    )

    if isinstance(env, VMASNavigationObs) and env.params["n_obs"] > 0:
        obstacle_states = graph.type_states(
            type_idx=env.OBS, n_type=env.params["n_obs"]
        )
        relative_position = positions[:, None, :] - obstacle_states[None, :, :2]
        distance = jnp.linalg.norm(relative_position, axis=-1)
        relative_velocity = velocities[:, None, :]
        braking = 0.0 if braking_accel is None else _braking_margin(
            relative_position, relative_velocity, braking_accel
        )
        worst_braking = (
            0.0
            if braking_accel is None
            else max_speed ** 2 / (2.0 * braking_accel)
        )
        collision_distance = (
            env.agent_radius + env.params["obstacle_radius"] + obstacle_margin
        )
        ceiling = jnp.minimum(
            maximum_margin,
            env.params["comm_radius"] - collision_distance - worst_braking,
        )
        clearance = jnp.minimum(distance - collision_distance - braking, ceiling)
        clearance = jnp.where(distance < env.params["comm_radius"], clearance, ceiling)
        obstacle_constraint = jnp.min(clearance, axis=-1)
    else:
        obstacle_constraint = jnp.full(
            (env.num_agents,), maximum_margin, dtype=agent_states.dtype
        )

    return jnp.minimum(jnp.minimum(agent_constraint, obstacle_constraint), maximum_margin)


def safety_constraint(
        env: MultiAgentEnv,
        graph: GraphsTuple,
        *,
        agent_margin: float = 0.02,
        obstacle_margin: float = 0.02,
        braking_accel: float | None = None,
        maximum_margin: float | None = None,
) -> Array:
    """Dispatch to the continuous safety-constraint adapter for ``env``."""
    kwargs = dict(
        agent_margin=agent_margin,
        obstacle_margin=obstacle_margin,
        braking_accel=braking_accel,
        maximum_margin=maximum_margin,
    )
    if isinstance(env, LidarEnv):
        return lidar_safety_constraint(env, graph, **kwargs)
    if isinstance(env, VMASNavigation):
        return vmas_navigation_safety_constraint(env, graph, **kwargs)
    raise TypeError(
        "Deep-QP safety constraints support LidarEnv and the "
        "VMASNavigation family; got " + type(env).__name__
    )


def safety_constraint_metadata(
        env: MultiAgentEnv,
        *,
        agent_margin: float,
        obstacle_margin: float,
        braking_accel: float | None,
) -> dict:
    """Checkpoint identity for the selected environment adapter."""
    if isinstance(env, LidarEnv):
        adapter = "lidar_clearance_v1"
        geometry = {
            "car_radius": env.params["car_radius"],
            "n_obs": env.params["n_obs"],
            "top_k_rays": env.params["top_k_rays"],
        }
    elif isinstance(env, VMASNavigation):
        adapter = "vmas_navigation_clearance_v1"
        geometry = {
            "agent_radius": env.agent_radius,
            "n_obs": env.params.get("n_obs", 0),
            "obstacle_radius": env.params.get("obstacle_radius"),
        }
    else:
        raise TypeError(f"No Deep-QP constraint adapter for {type(env).__name__}")
    action_lower, action_upper = env.action_lim()
    return {
        "constraint_adapter": adapter,
        "env_class": type(env).__name__,
        "state_dim": env.state_dim,
        "node_dim": env.node_dim,
        "edge_dim": env.edge_dim,
        "action_dim": env.action_dim,
        "n_agents": env.num_agents,
        "dt": env.dt,
        "comm_radius": env.params["comm_radius"],
        "action_lower": tuple(float(x) for x in action_lower.tolist()),
        "action_upper": tuple(float(x) for x in action_upper.tolist()),
        "agent_margin": agent_margin,
        "obstacle_margin": obstacle_margin,
        "braking_accel": braking_accel,
    } | geometry


def safety_node_feature_mask(env: MultiAgentEnv) -> Array:
    """Select graph node features that belong to the safety state.

    VMAS navigation stores relative task-goal position in columns 4:6 of each
    agent node.  The safety critic must not condition its HJ value on that task
    variable, while the actor still receives the original full graph.
    """
    mask = jnp.ones((env.node_dim,), dtype=bool)
    if isinstance(env, VMASNavigation):
        mask = mask.at[4:6].set(False)
    return mask
