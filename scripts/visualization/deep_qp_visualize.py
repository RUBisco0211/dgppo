"""Visualize a pretrained multi-agent Deep-QP HJ value as ego-centric GIFs.

For every selected ego agent, this script rolls out an arbitrary policy and
freezes that ego's graph neighborhood and LiDAR returns at each sampled frame.
It then moves only the ego state over the global ``x-y`` grid, updates features
on the fixed graph, and evaluates the frozen Graph-HJ critic.  Objects outside
the sensing neighborhood remain visible but cannot enter the certificate
input during the sweep.

The policy is used only to produce changing scene snapshots.  Its safety is not
evaluated and does not affect how the HJ value checkpoint is loaded.

Example
-------
python deep_qp_visualize.py \
    --deep-qp-checkpoint logs/LidarSpread/deepqp \
    --policy-mode random \
    --output-dir 'figures/deep-qp-contour/train3->eval3'
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Polygon
from PIL import Image

from dgppo.algo import make_algo
from dgppo.algo.module.deep_qp_safety import (
    DeepQPSafetyConfig,
    GraphHJSafetyCritic,
    environment_cost_metadata,
    graph_hj_node_feature_mask,
    safety_lambda_at,
)
from dgppo.env import make_env
from dgppo.env.lidar_env.base import LidarEnv
from dgppo.env.plot import get_BuRd
from dgppo.env.safety_constraint import (
    safety_constraint_metadata,
    safety_node_feature_mask,
)


DEFAULT_DEEP_QP_CHECKPOINT = Path("logs/LidarSpread/deepqp")
DEFAULT_POLICY_DIR = Path("logs/LidarSpread/dgppo/seed0_707102621_YGIV")
DEFAULT_OUTPUT_DIR = Path("outputs/deep_qp_visualization")


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _checkpoint_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Deep-QP checkpoint not found: {path}")

    # Accept either a checkpoint directory, a complete training-run directory,
    # or the common parent containing copied/archived runs.
    direct_candidates = (
        path / "deep_qp_safety.pkl",
        path / "models" / "latest" / "deep_qp_safety.pkl",
    )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    candidates = list(path.glob("**/models/latest/deep_qp_safety.pkl"))
    if not candidates:
        candidates = list(path.glob("**/deep_qp_safety.pkl"))
    if not candidates:
        raise FileNotFoundError(
            "Deep-QP checkpoint not found under directory: " f"{path}"
        )
    checkpoint = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    if len(candidates) > 1:
        print(
            f"found {len(candidates)} Deep-QP checkpoints under {path}; "
            f"using the most recently modified: {checkpoint}",
            flush=True,
        )
    return checkpoint


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        payload = pickle.load(file)
    if payload.get("format_version") != 2:
        raise ValueError("expected a format_version=2 Deep-QP checkpoint")
    if payload.get("derivative_head") != "local_joint_pair":
        raise ValueError("expected the local_joint_pair derivative head")
    return payload


def _config_from_payload(payload: dict[str, Any]) -> DeepQPSafetyConfig:
    valid_names = {field.name for field in fields(DeepQPSafetyConfig)}
    saved = payload.get("config", {})
    return DeepQPSafetyConfig(**{key: value for key, value in saved.items() if key in valid_names})


def _make_checkpoint_env(
    payload: dict[str, Any],
    max_step: int,
    num_agents: int | None = None,
    num_obs: int | None = None,
) -> LidarEnv:
    metadata = payload.get("metadata", {})
    env_name = metadata.get("env_class", "LidarSpread")
    env = make_env(
        env_id=env_name,
        num_agents=(
            int(payload["n_agents"])
            if num_agents is None
            else num_agents
        ),
        num_obs=(metadata.get("n_obs") if num_obs is None else num_obs),
        n_rays=metadata.get("n_rays"),
        max_step=max_step,
        full_observation=False,
    )
    if not isinstance(env, LidarEnv):
        raise TypeError(
            "deep_qp_visualize.py currently supports the two-dimensional "
            f"LidarEnv family, got {type(env).__name__}"
        )
    return env


def _load_safety_critic(
    checkpoint: Path,
    payload: dict[str, Any],
    env: LidarEnv,
    init_graph,
) -> tuple[GraphHJSafetyCritic, Any, float]:
    config = _config_from_payload(payload)
    metadata = payload["metadata"]
    action_lower, action_upper = env.action_lim()
    is_environment_cost = metadata.get("cost_source") == "env.get_cost"
    is_local_adapter = metadata.get("constraint_adapter") == "lidar_clearance_v1"
    if not (is_environment_cost or is_local_adapter):
        raise ValueError(
            "unsupported Deep-QP constraint source: "
            f"{metadata.get('cost_source') or metadata.get('constraint_adapter')}"
        )
    critic = GraphHJSafetyCritic(
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        action_lower=action_lower,
        action_upper=action_upper,
        config=config,
        node_feature_mask=(
            graph_hj_node_feature_mask(env)
            if is_environment_cost
            else safety_node_feature_mask(env)
        ),
    )
    state = critic.initialize(jr.PRNGKey(1), init_graph)
    if is_environment_cost:
        expected_metadata = environment_cost_metadata(env)
    else:
        expected_metadata = safety_constraint_metadata(
            env,
            agent_margin=float(metadata["agent_margin"]),
            obstacle_margin=float(metadata["obstacle_margin"]),
            braking_accel=metadata.get("braking_accel"),
        )
    state = critic.load_checkpoint(
        state,
        checkpoint,
        expected_metadata=expected_metadata,
        allow_agent_count_transfer=(
            env.num_agents != int(payload["n_agents"])
        ),
        allow_obstacle_count_transfer=(
            metadata.get("n_obs") is not None
            and env.params["n_obs"] != metadata["n_obs"]
        ),
    )
    safety_lambda = float(
        np.asarray(safety_lambda_at(config, int(np.asarray(payload["step"]))))
    )
    return critic, state, safety_lambda


def _latest_policy_step(models_dir: Path) -> int:
    steps = sorted(
        int(path.name)
        for path in models_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not steps:
        raise FileNotFoundError(f"no numeric policy checkpoints found in {models_dir}")
    return steps[-1]


def _load_policy(
    policy_dir: Path,
    policy_step: int | None,
    env: LidarEnv,
) -> tuple[Callable, Any, str]:
    policy_dir = policy_dir.expanduser().resolve()
    config_path = policy_dir / "config.yaml"
    models_dir = policy_dir / "models"
    if not config_path.is_file() or not models_dir.is_dir():
        raise FileNotFoundError(
            "policy-dir must contain config.yaml and models/: " f"{policy_dir}"
        )
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.UnsafeLoader)

    expected = {
        "env": type(env).__name__,
    }
    for name, value in expected.items():
        configured = _cfg_get(config, name)
        if configured is not None and configured != value:
            raise ValueError(
                f"policy {name}={configured!r} does not match Deep-QP "
                f"checkpoint environment value {value!r}"
            )

    algo = make_algo(
        algo=_cfg_get(config, "algo"),
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=_cfg_get(config, "cost_weight", 0.0),
        actor_gnn_layers=_cfg_get(config, "actor_gnn_layers", 2),
        Vl_gnn_layers=_cfg_get(config, "Vl_gnn_layers", 2),
        Vh_gnn_layers=_cfg_get(config, "Vh_gnn_layers", 1),
        gamma=_cfg_get(config, "gamma", 0.99),
        lr_actor=_cfg_get(config, "lr_actor", 3e-4),
        lr_Vl=_cfg_get(config, "lr_Vl", 1e-3),
        lr_Vh=_cfg_get(config, "lr_Vh", 1e-3),
        batch_size=_cfg_get(config, "batch_size", 8192),
        epoch_ppo=_cfg_get(config, "epoch_ppo", 1),
        clip_eps=_cfg_get(config, "clip_eps", 0.25),
        gae_lambda=_cfg_get(config, "gae_lambda", 0.95),
        coef_ent=_cfg_get(config, "coef_ent", 1e-2),
        max_grad_norm=_cfg_get(config, "max_grad_norm", 2.0),
        seed=_cfg_get(config, "seed", 0),
        use_rnn=_cfg_get(config, "use_rnn", not _cfg_get(config, "no_rnn", False)),
        rnn_layers=_cfg_get(config, "rnn_layers", 1),
        rnn_step=_cfg_get(config, "rnn_step", 16),
        use_lstm=_cfg_get(config, "use_lstm", False),
        alpha=_cfg_get(config, "alpha", 10.0),
        cbf_eps=_cfg_get(config, "cbf_eps", 1e-2),
        cbf_weight=_cfg_get(config, "cbf_weight", 1.0),
        train_steps=_cfg_get(config, "steps", 100_000),
        cbf_schedule=_cfg_get(config, "cbf_schedule", True),
    )
    step = _latest_policy_step(models_dir) if policy_step is None else policy_step
    actor_path = models_dir / str(step) / "actor.pkl"
    if not actor_path.is_file():
        raise FileNotFoundError(f"policy actor checkpoint not found: {actor_path}")
    with actor_path.open("rb") as file:
        actor_params = pickle.load(file)
    algo.policy_train_state = algo.policy_train_state.replace(
        params=actor_params
    )
    act = jax.jit(algo.act)
    label = f"{_cfg_get(config, 'algo')}:{policy_dir.name}/step={step}"
    return act, algo.init_rnn_state, label


def _make_action_source(
    mode: str,
    policy_dir: Path,
    policy_step: int | None,
    env: LidarEnv,
    seed: int,
) -> tuple[Callable, Any, str]:
    if mode == "checkpoint":
        return _load_policy(policy_dir, policy_step, env)

    if mode == "zero":
        def zero_action(_graph, state):
            return jnp.zeros((env.num_agents, env.action_dim)), state

        return zero_action, None, "zero policy"

    # Match gcbfplus_visualize.py exactly: the first split is reserved for
    # environment reset and subsequent splits generate rollout actions.
    _, random_key = jr.split(jr.PRNGKey(seed))
    lower, upper = env.action_lim()

    def random_action(_graph, state):
        nonlocal random_key
        action_key, random_key = jr.split(random_key)
        action = jr.uniform(
            action_key,
            (env.num_agents, env.action_dim),
            minval=lower,
            maxval=upper,
        )
        return action, state

    return random_action, None, "uniform random policy"


def _collect_scene_snapshots(
    env: LidarEnv,
    act: Callable,
    rnn_state: Any,
    *,
    seed: int,
    frames: int,
    frame_stride: int,
    rollout_start: int,
) -> list[Any]:
    reset_key, _ = jr.split(jr.PRNGKey(seed))
    graph = env.reset(reset_key)
    snapshots = []
    total_steps = rollout_start + (frames - 1) * frame_stride + 1
    capture_steps = {
        rollout_start + frame * frame_stride: frame for frame in range(frames)
    }
    for step in range(total_steps):
        if step in capture_steps:
            snapshots.append(jax.device_get(graph))
        action, rnn_state = act(graph, rnn_state)
        graph, _, _, _, _ = env.step(graph, action)
    if len(snapshots) != frames:
        raise RuntimeError(f"collected {len(snapshots)} snapshots, expected {frames}")
    return snapshots


def _parse_ego_agents(spec: str, n_agents: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(n_agents))
    result = []
    for token in spec.split(","):
        ego = int(token.strip())
        if ego < 0 or ego >= n_agents:
            raise ValueError(f"ego agent {ego} is outside [0, {n_agents - 1}]")
        if ego not in result:
            result.append(ego)
    if not result:
        raise ValueError("at least one ego agent is required")
    return result


def _move_ego_in_fixed_graph(
    graph, env: LidarEnv, ego_agent: int, xy: jax.Array
):
    """Move ego while preserving snapshot senders, receivers, and LiDAR nodes."""

    states = graph.states.at[ego_agent, :2].set(xy)
    nodes = graph.nodes.at[ego_agent, : env.state_dim].set(states[ego_agent])
    edges = states[graph.receivers] - states[graph.senders]
    agents = graph.env_states.agent.at[ego_agent, :2].set(xy)
    env_state = graph.env_states._replace(agent=agents)
    return graph._replace(
        nodes=nodes,
        edges=edges,
        states=states,
        env_states=env_state,
    )


def _encode_environment_cost(raw_cost: jax.Array) -> jax.Array:
    cost = jnp.where(raw_cost <= 0.0, raw_cost - 0.5, raw_cost + 0.5)
    return jnp.clip(cost, a_min=-1.0, a_max=1.0)


def _fixed_ego_cost(env: LidarEnv, graph, ego_agent: int) -> jax.Array:
    """Apply env.get_cost geometry to the ego's frozen input contributors."""

    n_agents = env.num_agents
    positions = graph.states[:n_agents, :2]
    senders = graph.senders
    receivers = graph.receivers
    safe_senders = jnp.clip(senders, 0, graph.states.shape[0] - 1)

    agent_visible = (
        (receivers == ego_agent)
        & (senders < n_agents)
        & (senders != ego_agent)
    )
    agent_distances = jnp.linalg.norm(
        positions[ego_agent] - graph.states[safe_senders, :2], axis=-1
    )
    nearest_agent = jnp.min(
        jnp.where(agent_visible, agent_distances, env.params["comm_radius"])
    )
    raw_agent_cost = 2.0 * env.params["car_radius"] - nearest_agent

    if env.params["n_obs"] > 0:
        n_rays = int(env.params["top_k_rays"])
        lidar_start = n_agents + env.num_goals + ego_agent * n_rays
        lidar_stop = lidar_start + n_rays
        lidar_visible = (
            (receivers == ego_agent)
            & (senders >= lidar_start)
            & (senders < lidar_stop)
        )
        lidar_distances = jnp.linalg.norm(
            positions[ego_agent] - graph.states[safe_senders, :2], axis=-1
        )
        nearest_lidar = jnp.min(
            jnp.where(
                lidar_visible,
                lidar_distances,
                env.params["comm_radius"] - 0.1,
            )
        )
        raw_obstacle_cost = env.params["car_radius"] - nearest_lidar
    else:
        raw_obstacle_cost = jnp.asarray(0.0, dtype=positions.dtype)

    return _encode_environment_cost(
        jnp.stack([raw_agent_cost, raw_obstacle_cost])
    )


def _fixed_ego_clearance(
    env: LidarEnv,
    graph,
    ego_agent: int,
    metadata: dict[str, Any],
    maximum_margin: float,
) -> jax.Array:
    """Evaluate the legacy clearance adapter on frozen contributor identities."""

    positions = graph.states[: env.num_agents, :2]
    velocities = graph.states[: env.num_agents, 2:4]
    senders = graph.senders
    receivers = graph.receivers
    safe_senders = jnp.clip(senders, 0, graph.states.shape[0] - 1)
    braking_accel = metadata.get("braking_accel")

    def braking(relative_position, relative_velocity):
        if braking_accel is None:
            return jnp.zeros(relative_position.shape[:-1], dtype=positions.dtype)
        distance = jnp.linalg.norm(relative_position, axis=-1)
        direction = relative_position / jnp.maximum(distance[..., None], 1e-6)
        closing_speed = jnp.maximum(
            -jnp.sum(relative_velocity * direction, axis=-1), 0.0
        )
        return closing_speed**2 / (2.0 * float(braking_accel))

    if braking_accel is None:
        maximum_speed = 0.0
    else:
        lower, upper = env.state_lim()
        velocity_limit = jnp.maximum(jnp.abs(lower[2:4]), jnp.abs(upper[2:4]))
        maximum_speed = jnp.linalg.norm(velocity_limit)

    agent_collision = 2.0 * env.params["car_radius"] + float(
        metadata["agent_margin"]
    )
    worst_agent_braking = (
        0.0
        if braking_accel is None
        else (2.0 * maximum_speed) ** 2 / (2.0 * float(braking_accel))
    )
    agent_ceiling = jnp.minimum(
        maximum_margin,
        env.params["comm_radius"] - agent_collision - worst_agent_braking,
    )
    agent_visible = (
        (receivers == ego_agent)
        & (senders < env.num_agents)
        & (senders != ego_agent)
    )
    agent_relative_position = (
        positions[ego_agent] - graph.states[safe_senders, :2]
    )
    agent_relative_velocity = (
        velocities[ego_agent] - graph.states[safe_senders, 2:4]
    )
    agent_clearance = (
        jnp.linalg.norm(agent_relative_position, axis=-1)
        - agent_collision
        - braking(agent_relative_position, agent_relative_velocity)
    )
    agent_clearance = jnp.min(
        jnp.where(agent_visible, jnp.minimum(agent_clearance, agent_ceiling), agent_ceiling)
    )

    if env.params["n_obs"] == 0:
        return jnp.minimum(agent_clearance, maximum_margin)

    obstacle_collision = env.params["car_radius"] + float(
        metadata["obstacle_margin"]
    )
    obstacle_sense_range = env.params["comm_radius"] - 0.1
    worst_obstacle_braking = (
        0.0
        if braking_accel is None
        else maximum_speed**2 / (2.0 * float(braking_accel))
    )
    obstacle_ceiling = jnp.minimum(
        maximum_margin,
        obstacle_sense_range - obstacle_collision - worst_obstacle_braking,
    )
    n_rays = int(env.params["top_k_rays"])
    lidar_start = env.num_agents + env.num_goals + ego_agent * n_rays
    lidar_stop = lidar_start + n_rays
    lidar_visible = (
        (receivers == ego_agent)
        & (senders >= lidar_start)
        & (senders < lidar_stop)
    )
    obstacle_relative_position = (
        positions[ego_agent] - graph.states[safe_senders, :2]
    )
    obstacle_relative_velocity = jnp.broadcast_to(
        velocities[ego_agent], obstacle_relative_position.shape
    )
    obstacle_clearance = (
        jnp.linalg.norm(obstacle_relative_position, axis=-1)
        - obstacle_collision
        - braking(obstacle_relative_position, obstacle_relative_velocity)
    )
    obstacle_clearance = jnp.min(
        jnp.where(
            lidar_visible,
            jnp.minimum(obstacle_clearance, obstacle_ceiling),
            obstacle_ceiling,
        )
    )
    return jnp.minimum(
        jnp.minimum(agent_clearance, obstacle_clearance), maximum_margin
    )


def _make_grid_evaluator(
    env: LidarEnv,
    critic: GraphHJSafetyCritic,
    params,
    safety_lambda: float,
    checkpoint_metadata: dict[str, Any],
    ego_agent: int,
) -> Callable:
    config = critic.config

    def evaluate_one(
        xy: jax.Array,
        base_graph,
    ) -> jax.Array:
        graph = _move_ego_in_fixed_graph(base_graph, env, ego_agent, xy)
        if checkpoint_metadata.get("cost_source") == "env.get_cost":
            fixed_constraint = -jnp.max(_fixed_ego_cost(env, graph, ego_agent))
        else:
            fixed_constraint = _fixed_ego_clearance(
                env,
                graph,
                ego_agent,
                checkpoint_metadata,
                config.constraint_scale,
            )
        constraint = jnp.full(
            (env.num_agents,), config.constraint_scale, dtype=graph.nodes.dtype
        )
        constraint = constraint.at[ego_agent].set(fixed_constraint)
        certificate = critic.certify(params, graph, constraint, safety_lambda)
        return jnp.stack(
            [
                certificate.value[ego_agent] * config.constraint_scale,
                constraint[ego_agent],
            ]
        )

    return jax.jit(jax.vmap(evaluate_one, in_axes=(0, None)))


def _evaluate_contours(
    snapshots: Sequence[Any],
    evaluators: dict[int, Callable],
    grid_points: jax.Array,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    values = {ego: [] for ego in evaluators}
    constraints = {ego: [] for ego in evaluators}
    for frame_idx, graph in enumerate(snapshots):
        for ego, evaluator in evaluators.items():
            evaluated = np.asarray(
                evaluator(
                    grid_points,
                    jax.tree.map(jnp.asarray, graph),
                )
            )
            values[ego].append(evaluated[:, 0])
            constraints[ego].append(evaluated[:, 1])
        print(f"evaluated contour frame {frame_idx + 1}/{len(snapshots)}", flush=True)
    return values, constraints


def _symmetric_value_limit(
    values: dict[int, list[np.ndarray]],
    requested_limit: float | None,
) -> float:
    if requested_limit is not None:
        if requested_limit <= 0.0:
            raise ValueError("value-limit must be positive")
        return requested_limit
    flattened = np.concatenate(
        [np.ravel(frame) for ego_frames in values.values() for frame in ego_frames]
    )
    finite = np.abs(flattened[np.isfinite(flattened)])
    if finite.size == 0:
        raise ValueError("Deep-QP evaluation produced no finite values")
    return max(float(np.percentile(finite, 99.5)), 0.025)


def _draw_obstacles(ax, obstacles) -> None:
    if obstacles is None:
        return
    points = np.asarray(obstacles.points)
    for polygon in points:
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="#970b07",
                edgecolor="#4d0503",
                linewidth=1.4,
                alpha=0.96,
                zorder=8,
            )
        )


def _draw_zero_contour(ax, x_grid, y_grid, field, **kwargs) -> None:
    finite = field[np.isfinite(field)]
    if finite.size and finite.min() <= 0.0 <= finite.max():
        ax.contour(x_grid, y_grid, field, levels=[0.0], **kwargs)


def _fixed_input_geometry(
    graph, env: LidarEnv, ego_agent: int
) -> tuple[list[tuple[int, int]], np.ndarray]:
    senders = np.asarray(graph.senders)
    receivers = np.asarray(graph.receivers)
    n_agents = env.num_agents
    direct_neighbors = set(
        senders[(receivers == ego_agent) & (senders < n_agents)].tolist()
    )
    local_agents = direct_neighbors | {ego_agent}
    edge_pairs: set[tuple[int, int]] = set()
    for sender, receiver in zip(senders, receivers):
        if sender in local_agents and receiver in local_agents and sender != receiver:
            edge_pairs.add(tuple(sorted((int(sender), int(receiver)))))

    n_rays = int(env.params["top_k_rays"])
    lidar_start = n_agents + env.num_goals + ego_agent * n_rays
    lidar_stop = lidar_start + n_rays
    lidar_ids = senders[
        (receivers == ego_agent)
        & (senders >= lidar_start)
        & (senders < lidar_stop)
    ]
    lidar_hits = np.asarray(graph.states)[np.unique(lidar_ids.astype(int)), :2]
    return sorted(edge_pairs), lidar_hits


def _render_frame(
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    value_grid: np.ndarray,
    constraint_grid: np.ndarray,
    graph,
    ego_agent: int,
    frame_idx: int,
    value_limit: float,
    env: LidarEnv,
    show_goals: bool,
    show_clearance: bool,
    dpi: int,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=dpi)
    fig.subplots_adjust(left=0.025, right=0.90, bottom=0.025, top=0.975)
    levels = np.linspace(-value_limit, value_limit, 17)
    norm = TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit)
    contour = ax.contourf(
        x_grid,
        y_grid,
        np.clip(value_grid, -value_limit, value_limit),
        levels=levels,
        cmap=get_BuRd().reversed(),
        norm=norm,
        extend="both",
        alpha=0.88,
        zorder=0,
    )
    _draw_zero_contour(
        ax,
        x_grid,
        y_grid,
        value_grid,
        colors="black",
        linewidths=1.7,
        zorder=5,
    )
    if show_clearance:
        finite_constraint = constraint_grid[np.isfinite(constraint_grid)]
        if (
            finite_constraint.size
            and finite_constraint.min() <= 0.0 <= finite_constraint.max()
        ):
            _draw_zero_contour(
                ax,
                x_grid,
                y_grid,
                constraint_grid,
                colors="#3f3f3f",
                linestyles="--",
                linewidths=1.2,
                zorder=4,
            )

    state = graph.env_states
    agents = np.asarray(state.agent)
    positions = agents[:, :2]
    _draw_obstacles(ax, state.obstacle)

    ego_position = positions[ego_agent]
    agent_edges, lidar_hits = _fixed_input_geometry(graph, env, ego_agent)
    for sender, receiver in agent_edges:
        ax.plot(
            [positions[sender, 0], positions[receiver, 0]],
            [positions[sender, 1], positions[receiver, 1]],
            color="#707070",
            linewidth=0.9,
            alpha=0.72,
            zorder=7,
        )
    for hit in lidar_hits:
        ax.plot(
            [ego_position[0], hit[0]],
            [ego_position[1], hit[1]],
            color="#777777",
            linewidth=0.7,
            alpha=0.68,
            zorder=6,
        )
    if len(lidar_hits):
        ax.scatter(
            lidar_hits[:, 0],
            lidar_hits[:, 1],
            s=5,
            color="#555555",
            alpha=0.7,
            zorder=7,
        )
    ax.add_patch(
        Circle(
            ego_position,
            env.params["comm_radius"],
            fill=False,
            linestyle=(0, (3, 3)),
            linewidth=1.8,
            edgecolor="#6f6f6f",
            alpha=0.9,
            zorder=6,
        )
    )

    if show_goals:
        goals = np.asarray(state.goal)[:, :2]
        ax.scatter(
            goals[:, 0],
            goals[:, 1],
            marker="x",
            s=45,
            color="#525252",
            alpha=0.55,
            linewidths=1.2,
            label="policy goals (masked from HJ critic)",
            zorder=7,
        )

    radius = env.params["car_radius"]
    for agent, position in enumerate(positions):
        is_ego = agent == ego_agent
        ax.add_patch(
            Circle(
                position,
                radius * (1.28 if is_ego else 1.0),
                facecolor="#0068ff" if is_ego else "#75a9f9",
                edgecolor="#001b52",
                linewidth=2.0 if is_ego else 1.0,
                zorder=10,
            )
        )
        ax.text(
            position[0],
            position[1],
            str(agent),
            ha="center",
            va="center",
            color="white" if is_ego else "#06285c",
            fontsize=10,
            fontweight="bold",
            zorder=11,
        )

    nearest_x = np.abs(x_grid[0] - ego_position[0]).argmin()
    nearest_y = np.abs(y_grid[:, 0] - ego_position[1]).argmin()
    actual_value = value_grid[nearest_y, nearest_x]
    actual_constraint = constraint_grid[nearest_y, nearest_x]
    ax.text(
        0.015,
        0.985,
        (
            f"ego agent {ego_agent} | frame {frame_idx:02d}\n"
            f"h={actual_value:+.4f}, fixed constraint={actual_constraint:+.4f}\n"
            f"method=Deep-QP, graph=frozen"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#777777"},
        zorder=20,
    )
    ax.set_xlim(0.0, env.area_size)
    ax.set_ylim(0.0, env.area_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.text(
        0.97,
        0.97,
        f"Sensing radius\nR={env.params['comm_radius']:g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#202020",
        zorder=20,
    )
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label("h (blue=safe, red=unsafe)")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _save_gif(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("cannot save an empty GIF")
    duration_ms = max(1, round(1000.0 / fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def visualize(args: argparse.Namespace) -> list[Path]:
    if args.frames <= 0 or args.frame_stride <= 0 or args.grid_size < 3:
        raise ValueError("frames/frame-stride must be positive and grid-size >= 3")
    if args.fps <= 0.0 or args.dpi <= 0:
        raise ValueError("fps and dpi must be positive")
    if args.num_agents is not None and args.num_agents <= 0:
        raise ValueError("num-agents must be positive")
    if args.num_obs is not None and args.num_obs < 0:
        raise ValueError("num-obs must be non-negative")

    checkpoint = _checkpoint_file(args.deep_qp_checkpoint)
    payload = _load_checkpoint_payload(checkpoint)
    env = _make_checkpoint_env(
        payload,
        max_step=args.rollout_start + args.frames * args.frame_stride + 1,
        num_agents=args.num_agents,
        num_obs=args.num_obs,
    )
    init_graph = env.reset(jr.PRNGKey(args.seed))
    critic, safety_state, safety_lambda = _load_safety_critic(
        checkpoint, payload, env, init_graph
    )
    params = (
        safety_state.online.params if args.online_params else safety_state.target_params
    )
    act, rnn_state, _ = _make_action_source(
        args.policy_mode,
        args.policy_dir,
        args.policy_step,
        env,
        args.seed,
    )
    snapshots = _collect_scene_snapshots(
        env,
        act,
        rnn_state,
        seed=args.seed,
        frames=args.frames,
        frame_stride=args.frame_stride,
        rollout_start=args.rollout_start,
    )

    ego_agents = _parse_ego_agents(args.ego_agents, env.num_agents)
    axis = np.linspace(0.0, env.area_size, args.grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(axis, axis)
    grid_points = jnp.asarray(np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1))
    evaluators = {
        ego: _make_grid_evaluator(
            env,
            critic,
            params,
            safety_lambda,
            payload["metadata"],
            ego,
        )
        for ego in ego_agents
    }
    values, constraints = _evaluate_contours(snapshots, evaluators, grid_points)
    value_limit = _symmetric_value_limit(values, args.value_limit)

    output_dir = args.output_dir.expanduser().resolve()
    written = []
    for ego in ego_agents:
        gif_frames = []
        for frame_idx, graph in enumerate(snapshots):
            gif_frames.append(
                _render_frame(
                    x_grid=x_grid,
                    y_grid=y_grid,
                    value_grid=values[ego][frame_idx].reshape(x_grid.shape),
                    constraint_grid=constraints[ego][frame_idx].reshape(x_grid.shape),
                    graph=graph,
                    ego_agent=ego,
                    frame_idx=frame_idx,
                    value_limit=value_limit,
                    env=env,
                    show_goals=args.show_goals,
                    show_clearance=args.show_clearance,
                    dpi=args.dpi,
                )
            )
        output_path = output_dir / f"deep_qp_ego_agent_{ego}.gif"
        _save_gif(gif_frames, output_path, args.fps)
        written.append(output_path)
        print(f"wrote {output_path}", flush=True)

    print(
        "checkpoint="
        f"{checkpoint}, step={int(np.asarray(payload['step']))}, "
        f"source_agents={int(payload['n_agents'])}, "
        f"eval_agents={env.num_agents}, "
        f"source_obs={payload['metadata'].get('n_obs')}, "
        f"eval_obs={env.params['n_obs']}, "
        "constraint_source="
        f"{payload['metadata'].get('cost_source') or payload['metadata'].get('constraint_adapter')}, "
        f"lambda={safety_lambda:.8f}, value_limit=±{value_limit:.5f}",
        flush=True,
    )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed-neighborhood Deep-QP HJ contours on the global plane."
        )
    )
    parser.add_argument(
        "--deep-qp-checkpoint",
        type=Path,
        default=DEFAULT_DEEP_QP_CHECKPOINT,
        help="deep_qp_safety.pkl or the directory containing it",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("checkpoint", "random", "zero"),
        default="random",
        help="policy used only for scene snapshots; defaults to the shared random rollout",
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=DEFAULT_POLICY_DIR,
        help="repository policy run containing config.yaml and models/",
    )
    parser.add_argument(
        "--policy-step",
        type=int,
        default=None,
        help="numeric models/<step> checkpoint; defaults to the latest",
    )
    parser.add_argument(
        "-n",
        "--num-agents",
        type=int,
        default=None,
        help=(
            "target evaluation agent count; defaults to the Graph-HJ "
            "checkpoint count"
        ),
    )
    parser.add_argument(
        "--num-obs",
        "--obs",
        dest="num_obs",
        type=int,
        default=None,
        help=(
            "target physical obstacle count; defaults to the Graph-HJ "
            "checkpoint count"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--ego-agents",
        default="all",
        help="'all' or a comma-separated list such as 0,2",
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--rollout-start", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument(
        "--value-limit",
        type=float,
        default=None,
        help="symmetric color limit; defaults to the 99.5th percentile",
    )
    parser.add_argument(
        "--online-params",
        action="store_true",
        help="evaluate online rather than the default target-network params",
    )
    parser.add_argument("--show-goals", action="store_true")
    parser.add_argument(
        "--show-clearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw the fixed-neighborhood environment boundary as a dashed line",
    )
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
