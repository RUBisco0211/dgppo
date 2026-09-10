"""Render fixed-neighborhood contours for a trained GCBF/GCBF+ certificate.

Blue/positive values are predicted safe and red/negative values are predicted
unsafe.  For LidarEnv, every snapshot's ego neighborhood and LiDAR returns are
frozen while only the ego position is swept over the global plane.  Objects
outside the sensing neighborhood remain visible without entering the current
certificate input.

Example
-------
python gcbfplus_visualize.py \
    --gcbfplus-dir logs/LidarSpread/gcbf+/seed0_... \
    --ego-agents all
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from PIL import Image

from dgppo.algo import make_algo
from dgppo.algo.gcbf_plus_adapter import make_gcbf_plus_env_adapter
from dgppo.env import make_env
from dgppo.env.lidar_env.base import LidarEnv
from dgppo.env.lidar_env.lidar_line import LidarLine
from dgppo.env.plot import get_BuRd
from dgppo.env.vmas import VMASNavigationObs, VMASReverseTransport, VMASWheel


DEFAULT_OUTPUT_DIR = Path("outputs/gcbfplus_visualization")


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _load_config(run_dir: Path) -> Any:
    path = run_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"GCBF+ config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.UnsafeLoader)
    if _cfg_get(config, "algo") not in ("gcbf", "gcbf+", "gcbfplus"):
        raise ValueError(
            "gcbfplus_visualize.py requires a GCBF/GCBF+ run, got "
            f"algo={_cfg_get(config, 'algo')!r}"
        )
    return config


def _resolve_step(models_dir: Path, requested: str | None) -> str:
    if requested is not None:
        if not (models_dir / requested).is_dir():
            raise FileNotFoundError(f"checkpoint not found: {models_dir / requested}")
        return requested
    if (models_dir / "latest").is_dir():
        return "latest"
    numeric = sorted(
        int(path.name)
        for path in models_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not numeric:
        raise FileNotFoundError(f"no checkpoints found in {models_dir}")
    return str(numeric[-1])


def _make_env(config: Any, args: argparse.Namespace):
    configured_obs = _cfg_get(config, "obs", 0)
    env = make_env(
        env_id=_cfg_get(config, "env"),
        num_agents=(
            int(_cfg_get(config, "num_agents"))
            if args.num_agents is None
            else args.num_agents
        ),
        num_obs=(configured_obs if args.num_obs is None else args.num_obs),
        n_rays=_cfg_get(config, "n_rays", 32),
        max_step=args.rollout_start + args.frames * args.frame_stride + 1,
        full_observation=_cfg_get(config, "full_observation", False),
    )
    make_gcbf_plus_env_adapter(env)  # validate support before loading weights
    return env


def _make_algo(config: Any, env):
    return make_algo(
        algo="gcbf+",
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gcbf_gnn_layers=_cfg_get(config, "gcbf_gnn_layers", 1),
        gcbf_batch_size=_cfg_get(config, "gcbf_batch_size", 256),
        gcbf_buffer_size=max(
            _cfg_get(config, "gcbf_buffer_size", 65_536),
            _cfg_get(config, "gcbf_batch_size", 256),
        ),
        gcbf_horizon=_cfg_get(config, "gcbf_horizon", 32),
        gcbf_inner_epoch=_cfg_get(config, "gcbf_inner_epoch", 8),
        gcbf_lr_actor=_cfg_get(config, "gcbf_lr_actor", 3e-5),
        gcbf_lr_cbf=_cfg_get(config, "gcbf_lr_cbf", 3e-5),
        gcbf_alpha=_cfg_get(config, "gcbf_alpha", 1.0),
        gcbf_eps=_cfg_get(config, "gcbf_eps", 0.02),
        gcbf_loss_action_coef=_cfg_get(config, "gcbf_loss_action_coef", 1e-4),
        gcbf_loss_unsafe_coef=_cfg_get(config, "gcbf_loss_unsafe_coef", 1.0),
        gcbf_loss_safe_coef=_cfg_get(config, "gcbf_loss_safe_coef", 1.0),
        gcbf_loss_h_dot_coef=_cfg_get(config, "gcbf_loss_h_dot_coef", 0.01),
        gcbf_target_tau=_cfg_get(config, "gcbf_target_tau", 0.5),
        gcbf_qp_relax_penalty=_cfg_get(config, "gcbf_qp_relax_penalty", 1e3),
        gcbf_qp_chunk_size=_cfg_get(config, "gcbf_qp_chunk_size", 32),
        gcbf_unsafe_fraction=_cfg_get(config, "gcbf_unsafe_fraction", 0.5),
        max_grad_norm=_cfg_get(config, "max_grad_norm", 2.0),
        seed=_cfg_get(config, "seed", 0),
    )


def _parse_ego_agents(spec: str, n_agents: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(n_agents))
    result = []
    for token in spec.split(","):
        agent = int(token.strip())
        if not 0 <= agent < n_agents:
            raise ValueError(f"ego agent {agent} is outside [0, {n_agents - 1}]")
        if agent not in result:
            result.append(agent)
    if not result:
        raise ValueError("at least one ego agent is required")
    return result


def _move_ego_in_fixed_lidar_graph(
    graph, env: LidarEnv, ego_agent: int, position: jax.Array
):
    """Move ego while preserving snapshot senders, receivers, and LiDAR nodes."""

    states = graph.states.at[ego_agent, :2].set(position)
    nodes = graph.nodes.at[ego_agent, : env.state_dim].set(states[ego_agent])
    edge_states = jax.vmap(env.state2feat)(states)
    edges = edge_states[graph.receivers] - edge_states[graph.senders]
    agents = graph.env_states.agent.at[ego_agent, :2].set(position)
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

    positions = graph.states[: env.num_agents, :2]
    senders = graph.senders
    receivers = graph.receivers
    safe_senders = jnp.clip(senders, 0, graph.states.shape[0] - 1)
    agent_visible = (
        (receivers == ego_agent)
        & (senders < env.num_agents)
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
        lidar_start = env.num_agents + env.num_goals + ego_agent * n_rays
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


def _make_action_source(mode: str, algo, env, seed: int) -> tuple[Callable, str]:
    if mode == "checkpoint":
        return jax.jit(lambda graph: algo.act(graph, algo.init_rnn_state)[0]), "GCBF+ actor"
    if mode == "nominal":
        return jax.jit(algo.adapter.nominal_action), "adapter nominal controller"
    if mode == "zero":
        return lambda _graph: jnp.zeros((env.num_agents, env.action_dim)), "zero policy"
    lower, upper = env.action_lim()

    def random_action(_graph, key):
        return jr.uniform(key, (env.num_agents, env.action_dim), minval=lower, maxval=upper)

    return jax.jit(random_action), f"uniform random policy (seed={seed})"


def _collect_snapshots(env, action_fn: Callable, mode: str, args) -> list[Any]:
    key = jr.PRNGKey(args.seed)
    reset_key, key = jr.split(key)
    graph = env.reset(reset_key)
    snapshots = []
    capture = {
        args.rollout_start + frame * args.frame_stride for frame in range(args.frames)
    }
    total_steps = args.rollout_start + (args.frames - 1) * args.frame_stride + 1
    for step in range(total_steps):
        if step in capture:
            snapshots.append(jax.device_get(graph))
        if mode == "random":
            action_key, key = jr.split(key)
            action = action_fn(graph, action_key)
        else:
            action = action_fn(graph)
        graph, _, _, _, _ = env.step(graph, action)
    return snapshots


def _make_grid_evaluator(algo, ego_agent: int) -> Callable:
    adapter = algo.adapter
    cbf_params = algo.cbf_train_state.params

    def evaluate(position, graph):
        if isinstance(algo._env, LidarEnv):
            moved_graph = _move_ego_in_fixed_lidar_graph(
                graph, algo._env, ego_agent, position
            )
            native_cost = _fixed_ego_cost(algo._env, moved_graph, ego_agent)
        else:
            moved_graph = adapter.with_agent_position(graph, ego_agent, position)
            native_cost = algo._env.get_cost(moved_graph)[ego_agent]
        value = algo.get_cbf(moved_graph, cbf_params)[ego_agent, 0]
        clearance = -jnp.max(native_cost)
        return jnp.stack([value, clearance])

    return jax.jit(jax.vmap(evaluate, in_axes=(0, None)))


def _evaluate_grid_in_chunks(
    evaluator: Callable, points: jax.Array, graph, batch_size: int
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("grid-batch-size must be positive")
    chunks = []
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        valid = len(batch)
        if valid < batch_size:
            batch = jnp.concatenate(
                [batch, jnp.repeat(batch[-1:], batch_size - valid, axis=0)], axis=0
            )
        chunks.append(np.asarray(evaluator(batch, graph))[:valid])
    return np.concatenate(chunks, axis=0)


def _fixed_lidar_input_geometry(
    graph, env: LidarEnv, ego_agent: int
) -> tuple[list[tuple[int, int]], np.ndarray]:
    senders = np.asarray(graph.senders)
    receivers = np.asarray(graph.receivers)
    direct_neighbors = set(
        senders[
            (receivers == ego_agent) & (senders < env.num_agents)
        ].tolist()
    )
    local_agents = direct_neighbors | {ego_agent}
    edge_pairs: set[tuple[int, int]] = set()
    for sender, receiver in zip(senders, receivers):
        if sender in local_agents and receiver in local_agents and sender != receiver:
            edge_pairs.add(tuple(sorted((int(sender), int(receiver)))))

    n_rays = int(env.params["top_k_rays"])
    lidar_start = env.num_agents + env.num_goals + ego_agent * n_rays
    lidar_stop = lidar_start + n_rays
    lidar_ids = senders[
        (receivers == ego_agent)
        & (senders >= lidar_start)
        & (senders < lidar_stop)
    ]
    lidar_hits = np.asarray(graph.states)[np.unique(lidar_ids.astype(int)), :2]
    return sorted(edge_pairs), lidar_hits


def _draw_scene(ax, env, graph, adapter, ego_agent: int, show_goals: bool) -> None:
    state = graph.env_states
    positions = np.asarray(adapter.agent_positions(graph))
    ego_position = positions[ego_agent]

    if isinstance(env, LidarEnv):
        agent_edges, lidar_hits = _fixed_lidar_input_geometry(
            graph, env, ego_agent
        )
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

    if isinstance(env, LidarEnv) and state.obstacle is not None:
        for polygon in np.asarray(state.obstacle.points):
            ax.add_patch(
                Polygon(polygon, facecolor="#970b07", edgecolor="#4d0503", alpha=0.9)
            )
    elif hasattr(state, "o_pos"):
        obstacle_radius = float(
            getattr(env, "obs_radius", env.params.get("obstacle_radius", 0.1))
        )
        for position in np.asarray(state.o_pos):
            ax.add_patch(Circle(position, obstacle_radius, color="#970b07", alpha=0.8))

    if isinstance(env, VMASReverseTransport):
        box_xy = np.asarray(state.box_pos) - np.array(
            [env.package_length / 2, env.package_width / 2]
        )
        ax.add_patch(
            Rectangle(
                box_xy,
                env.package_length,
                env.package_width,
                fill=False,
                edgecolor="#5d2516",
                linewidth=2,
            )
        )
    elif isinstance(env, VMASWheel):
        angle = float(state.line_angle)
        half = env.line_length / 2
        direction = np.array([np.cos(angle), np.sin(angle)]) * half
        ax.plot([-direction[0], direction[0]], [-direction[1], direction[1]], color="#5d2516", linewidth=5)
        avoid = float(state.avoid_angle)
        half_angle = float(env.obs_halfwidth_rad)
        ax.add_patch(
            Wedge(
                (0, 0),
                env.half_width,
                np.rad2deg(avoid - half_angle),
                np.rad2deg(avoid + half_angle),
                color="#970b07",
                alpha=0.18,
            )
        )

    if show_goals:
        if hasattr(state, "goal"):
            goals = np.asarray(state.goal)[:, :2]
            if isinstance(env, LidarLine):
                goals = np.asarray(env.landmark2goal(jnp.asarray(goals)))
        elif hasattr(state, "goal_pos") and np.asarray(state.goal_pos).ndim == 2:
            goals = np.asarray(state.goal_pos)
        elif hasattr(state, "goal_pos"):
            goals = np.asarray(state.goal_pos)[None]
        else:
            goals = np.empty((0, 2))
        if len(goals):
            ax.scatter(goals[:, 0], goals[:, 1], marker="x", color="#315b20", zorder=9)

    radius = adapter.agent_radius
    for index, position in enumerate(positions):
        is_ego = index == ego_agent
        ax.add_patch(
            Circle(
                position,
                radius * (1.3 if is_ego else 1.0),
                facecolor="#0068ff" if is_ego else "#75a9f9",
                edgecolor="#001b52",
                linewidth=2 if is_ego else 1,
                zorder=10,
            )
        )
        ax.text(*position, str(index), ha="center", va="center", fontsize=8, zorder=11)

    if isinstance(env, LidarEnv):
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


def _zero_contour(ax, x_grid, y_grid, values, **kwargs) -> None:
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() <= 0.0 <= finite.max():
        ax.contour(x_grid, y_grid, values, levels=[0.0], **kwargs)


def _render_frame(
    x_grid,
    y_grid,
    value_grid,
    clearance_grid,
    graph,
    algo,
    ego_agent,
    frame_index,
    value_limit,
    certificate_label,
    args,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=args.dpi)
    fig.subplots_adjust(left=0.025, right=0.90, bottom=0.025, top=0.975)
    levels = np.linspace(-value_limit, value_limit, 17)
    contour = ax.contourf(
        x_grid,
        y_grid,
        np.clip(value_grid, -value_limit, value_limit),
        levels=levels,
        cmap=get_BuRd().reversed(),
        norm=TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit),
        extend="both",
        alpha=0.88,
    )
    _zero_contour(ax, x_grid, y_grid, value_grid, colors="black", linewidths=1.7)
    if args.show_clearance:
        _zero_contour(
            ax,
            x_grid,
            y_grid,
            clearance_grid,
            colors="#3f3f3f",
            linestyles="--",
            linewidths=1.2,
        )
    _draw_scene(ax, algo._env, graph, algo.adapter, ego_agent, args.show_goals)
    xmin, xmax, ymin, ymax = algo.adapter.plot_bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        f"{certificate_label} · {type(algo._env).__name__}\nego {ego_agent} · frame {frame_index:02d}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#303030",
        zorder=20,
    )
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label("GCBF h (positive=safe)")
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), mode="RGBA")
    plt.close(fig)
    return image.convert("RGB")


def _save_gif(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=max(1, round(1000 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )


def visualize(args: argparse.Namespace) -> list[Path]:
    if args.frames <= 0 or args.frame_stride <= 0 or args.grid_size < 3:
        raise ValueError("frames/frame-stride must be positive and grid-size >= 3")
    if args.rollout_start < 0 or args.grid_batch_size <= 0:
        raise ValueError("rollout-start must be non-negative and grid-batch-size positive")
    if args.fps <= 0 or args.dpi <= 0:
        raise ValueError("fps and dpi must be positive")
    run_dir = args.gcbfplus_dir.expanduser().resolve()
    config = _load_config(run_dir)
    env = _make_env(config, args)
    algo = _make_algo(config, env)
    models_dir = run_dir / "models"
    step = _resolve_step(models_dir, args.step)
    checkpoint_dir = models_dir / step
    cbf_only = _cfg_get(config, "training_mode") == "cbf_only"
    certificate_label = "GCBF" if cbf_only else "GCBF+"
    policy_mode = args.policy_mode
    if cbf_only:
        with (checkpoint_dir / "cbf.pkl").open("rb") as file:
            cbf_params = pickle.load(file)
        algo.cbf_train_state = algo.cbf_train_state.replace(params=cbf_params)
        algo.cbf_target_params = jax.tree.map(jnp.copy, cbf_params)
        if policy_mode == "checkpoint":
            policy_mode = "nominal"
            print(
                "CBF-only checkpoint has no actor; using the nominal controller "
                "for scene rollout.",
                flush=True,
            )
    else:
        algo.load(str(models_dir), step)
    action_fn, _ = _make_action_source(policy_mode, algo, env, args.seed)
    snapshots = _collect_snapshots(env, action_fn, policy_mode, args)

    xmin, xmax, ymin, ymax = algo.adapter.plot_bounds
    x_axis = np.linspace(xmin, xmax, args.grid_size, dtype=np.float32)
    y_axis = np.linspace(ymin, ymax, args.grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    points = jnp.asarray(np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1))
    ego_agents = _parse_ego_agents(args.ego_agents, env.num_agents)
    evaluators = {agent: _make_grid_evaluator(algo, agent) for agent in ego_agents}
    values = {agent: [] for agent in ego_agents}
    clearances = {agent: [] for agent in ego_agents}
    for frame_index, graph in enumerate(snapshots):
        for agent in ego_agents:
            result = _evaluate_grid_in_chunks(
                evaluators[agent], points, graph, args.grid_batch_size
            )
            values[agent].append(result[:, 0].reshape(x_grid.shape))
            clearances[agent].append(result[:, 1].reshape(x_grid.shape))
        print(f"evaluated contour frame {frame_index + 1}/{len(snapshots)}", flush=True)

    if args.value_limit is None:
        finite = np.abs(
            np.concatenate([frame.ravel() for series in values.values() for frame in series])
        )
        finite = finite[np.isfinite(finite)]
        value_limit = max(float(np.percentile(finite, 99.5)), 0.025)
    elif args.value_limit <= 0:
        raise ValueError("value-limit must be positive")
    else:
        value_limit = args.value_limit

    written = []
    output_dir = args.output_dir.expanduser().resolve()
    for agent in ego_agents:
        frames = [
            _render_frame(
                x_grid,
                y_grid,
                values[agent][frame_index],
                clearances[agent][frame_index],
                graph,
                algo,
                agent,
                frame_index,
                value_limit,
                certificate_label,
                args,
            )
            for frame_index, graph in enumerate(snapshots)
        ]
        path = output_dir / f"gcbfplus_ego_agent_{agent}.gif"
        _save_gif(frames, path, args.fps)
        written.append(path)
        print(f"wrote {path}", flush=True)
    print(
        f"checkpoint={models_dir / step}, env={type(env).__name__}, "
        f"source_agents={_cfg_get(config, 'num_agents')}, "
        f"eval_agents={env.num_agents}, "
        f"source_obs={_cfg_get(config, 'obs')}, eval_obs={env.params.get('n_obs')}",
        flush=True,
    )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GCBF/GCBF+ certificate contours")
    parser.add_argument("--gcbfplus-dir", type=Path, required=True)
    parser.add_argument("--step", default=None, help="checkpoint directory name; defaults to latest")
    parser.add_argument(
        "--policy-mode",
        choices=("checkpoint", "nominal", "random", "zero"),
        default="random",
        help="policy used only for scene rollout; defaults to the shared random rollout",
    )
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--num-obs", "--obs", dest="num_obs", type=int, default=None)
    parser.add_argument("--ego-agents", default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--rollout-start", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--grid-batch-size", type=int, default=512)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--value-limit", type=float, default=None)
    parser.add_argument(
        "--show-clearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw the fixed-neighborhood environment boundary as a dashed line",
    )
    parser.add_argument("--show-goals", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
