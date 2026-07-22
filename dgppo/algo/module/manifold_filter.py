import functools as ft

import jax
import jax.numpy as jnp

from ...env.lidar_env.base import LidarEnv
from ...utils.graph import GraphsTuple
from ...utils.typing import Action, AgentState, Array


def _agent_position(state: Array) -> Array:
    return state[:2]


def _agent_velocity(state: Array) -> Array:
    if state.shape[0] == 5:
        return state[4] * state[2:4]
    return state[2:4]


def _entity_drift(entity_states: AgentState, state_dim: int) -> AgentState:
    if state_dim == 5:
        speed = entity_states[:, 4]
        drift = jnp.stack(
            [
                speed * entity_states[:, 2],
                speed * entity_states[:, 3],
                jnp.zeros_like(speed),
                jnp.zeros_like(speed),
                jnp.zeros_like(speed),
            ],
            axis=1,
        )
    else:
        drift = jnp.concatenate(
            [entity_states[:, 2:4], jnp.zeros((entity_states.shape[0], 2), dtype=entity_states.dtype)],
            axis=1,
        )
    return drift


def _pairwise_collision_h(
        own_state: Array,
        entity_states: AgentState,
        active: Array,
        safe_radius: Array,
        braking_accel: float,
        velocity_margin: float,
) -> Array:
    own_pos = _agent_position(own_state)
    own_vel = _agent_velocity(own_state)
    entity_pos = entity_states[:, :2]
    if own_state.shape[0] == 5:
        entity_vel = entity_states[:, 4:5] * entity_states[:, 2:4]
    else:
        entity_vel = entity_states[:, 2:4]

    rel_pos = own_pos[None, :] - entity_pos
    rel_vel = own_vel[None, :] - entity_vel
    dist = jnp.linalg.norm(rel_pos, axis=-1)
    direction = rel_pos / jnp.maximum(dist[:, None], 1e-6)
    approach_speed = jnp.maximum(0.0, -jnp.sum(rel_vel * direction, axis=-1))
    braking_margin = velocity_margin + approach_speed ** 2 / (2.0 * braking_accel)
    h = (safe_radius + braking_margin) ** 2 - dist ** 2
    return jnp.where(active, h, -1.0)


def _single_agent_project(
        env: LidarEnv,
        own_state: Array,
        own_drift: Array,
        own_control_matrix: Array,
        entity_states: AgentState,
        entity_drift: AgentState,
        active: Array,
        safe_radius: Array,
        u_ref: Array,
        braking_accel: float,
        velocity_margin: float,
        contraction_gain: float,
        slack_min: float,
        slack_beta: float,
        slack_weight: float,
        reg: float,
) -> Array:
    h_fn = ft.partial(
        _pairwise_collision_h,
        active=active,
        safe_radius=safe_radius,
        braking_accel=braking_accel,
        velocity_margin=velocity_margin,
    )
    h = h_fn(own_state, entity_states)
    c = h + jnp.maximum(-h, slack_min)
    dh_down = jax.jacfwd(h_fn, argnums=0)(own_state, entity_states)
    dh_dentities = jax.jacfwd(h_fn, argnums=1)(own_state, entity_states)

    psi_own = dh_down @ own_drift
    psi_entities = jnp.einsum("mns,ns->m", dh_dentities, entity_drift)
    psi = psi_own + psi_entities

    action_jac = dh_down @ own_control_matrix
    slack = jnp.maximum(-h, slack_min)
    slack_actuation = jnp.exp(slack_beta * slack) - 1.0

    # Use scaled slack coordinates so large slack_weight prefers correcting the
    # physical action before using virtual slack control.
    j_slack = jnp.diag(slack_actuation / slack_weight)
    j_aug = jnp.concatenate([action_jac, j_slack], axis=1)
    u_ref_aug = jnp.concatenate([u_ref, jnp.zeros_like(h)])
    residual = psi + j_aug @ u_ref_aug + contraction_gain * c
    gram = j_aug @ j_aug.T + reg * jnp.eye(j_aug.shape[0], dtype=j_aug.dtype)
    correction = j_aug.T @ (jnp.linalg.pinv(gram) @ residual)
    u_safe_aug = u_ref_aug - correction
    return u_safe_aug[:env.action_dim]


def lidar_manifold_project(
        env: LidarEnv,
        graph: GraphsTuple,
        action_ref: Action,
        *,
        top_k_obs: int = 3,
        safety_margin: float = 0.02,
        braking_accel: float = 1.0,
        velocity_margin: float = 0.02,
        contraction_gain: float = 30.0,
        slack_min: float = 0.1,
        slack_beta: float = 1.0,
        slack_weight: float = 10.0,
        reg: float = 1e-5,
) -> Action:
    """Project LidarEnv nominal actions through a one-step manifold filter.

    This is a single-step safety filter. It treats the policy action as the
    nominal reference action and projects it using local pairwise constraints
    against other agents and per-agent lidar hit points.
    """
    agent_states = graph.type_states(type_idx=env.AGENT, n_type=env.num_agents)
    dynamics = env.agent_control_affine_dynamics(agent_states)
    action_ref = env.clip_action(action_ref)

    n_agent_constraints = env.num_agents - 1
    n_obs_constraints = min(top_k_obs, env.params["top_k_rays"]) if env.params["n_obs"] > 0 else 0
    n_constraints = n_agent_constraints + n_obs_constraints

    if n_constraints == 0:
        return action_ref

    obs_states_all = None
    if env.params["n_obs"] > 0:
        n_hits = env.params["top_k_rays"] * env.num_agents
        obs_states_all = graph.type_states(type_idx=env.OBS, n_type=n_hits)
        obs_states_all = obs_states_all.reshape(env.num_agents, env.params["top_k_rays"], env.state_dim)

    safe_agent = 2.0 * env.params["car_radius"] + safety_margin
    safe_obs = env.params["car_radius"] + safety_margin
    projected_actions = []

    for i_agent in range(env.num_agents):
        other_ids = [idx for idx in range(env.num_agents) if idx != i_agent]
        entities = []
        entity_drifts = []
        active = []
        safe_radius = []
        if n_agent_constraints > 0:
            other_states = agent_states[jnp.array(other_ids)]
            entities.append(other_states)
            entity_drifts.append(_entity_drift(other_states, env.state_dim))
            active.append(jnp.ones((n_agent_constraints,), dtype=bool))
            safe_radius.append(jnp.ones((n_agent_constraints,), dtype=agent_states.dtype) * safe_agent)

        if n_obs_constraints > 0:
            obs_states = obs_states_all[i_agent, :top_k_obs]
            obs_dist = jnp.linalg.norm(obs_states[:, :2] - agent_states[i_agent, None, :2], axis=-1)
            obs_active = obs_dist < env.params["comm_radius"]
            entities.append(obs_states)
            entity_drifts.append(jnp.zeros_like(obs_states))
            active.append(obs_active)
            safe_radius.append(jnp.ones((n_obs_constraints,), dtype=agent_states.dtype) * safe_obs)

        entity_states = jnp.concatenate(entities, axis=0)
        drift_entities = jnp.concatenate(entity_drifts, axis=0)
        active_constraints = jnp.concatenate(active, axis=0)
        safe_radius_constraints = jnp.concatenate(safe_radius, axis=0)

        action_safe_i = _single_agent_project(
            env=env,
            own_state=agent_states[i_agent],
            own_drift=dynamics.drift[i_agent],
            own_control_matrix=dynamics.control_matrix[i_agent],
            entity_states=entity_states,
            entity_drift=drift_entities,
            active=active_constraints,
            safe_radius=safe_radius_constraints,
            u_ref=action_ref[i_agent],
            braking_accel=braking_accel,
            velocity_margin=velocity_margin,
            contraction_gain=contraction_gain,
            slack_min=slack_min,
            slack_beta=slack_beta,
            slack_weight=slack_weight,
            reg=reg,
        )
        projected_actions.append(action_safe_i)

    action_safe = jnp.stack(projected_actions, axis=0)
    assert action_safe.shape == action_ref.shape
    return env.clip_action(action_safe)
