"""Render ego-centric contours for a trained GCBF+ certificate.

Blue/positive values are predicted safe and red/negative values are predicted
unsafe.  The solid black curve is the learned ``h=0`` contour; the dashed grey
curve is the environment's native collision boundary.

Example
-------
python gcbfplus_visualize.py \
    --gcbfplus-dir logs/LidarSpread/gcbf+/seed0_... \
    --ego-agents all
"""

from __future__ import annotations

import argparse
import os
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
    if _cfg_get(config, "algo") not in ("gcbf+", "gcbfplus"):
        raise ValueError(
            "gcbfplus_visualize.py requires a GCBF+ run, got "
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
        moved_graph = adapter.with_agent_position(graph, ego_agent, position)
        value = algo.get_cbf(moved_graph, cbf_params)[ego_agent, 0]
        clearance = -jnp.max(algo._env.get_cost(moved_graph)[ego_agent])
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


def _draw_scene(ax, env, graph, adapter, ego_agent: int, show_goals: bool) -> None:
    state = graph.env_states
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

    positions = np.asarray(adapter.agent_positions(graph))
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
    policy_label,
    args,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=args.dpi, constrained_layout=True)
    levels = np.linspace(-value_limit, value_limit, 17)
    contour = ax.contourf(
        x_grid,
        y_grid,
        np.clip(value_grid, -value_limit, value_limit),
        levels=levels,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit),
        extend="both",
        alpha=0.88,
    )
    _zero_contour(ax, x_grid, y_grid, value_grid, colors="black", linewidths=1.7)
    _zero_contour(
        ax,
        x_grid,
        y_grid,
        clearance_grid,
        colors="#5a5a5a",
        linestyles="--",
        linewidths=1.0,
    )
    _draw_scene(ax, algo._env, graph, algo.adapter, ego_agent, args.show_goals)
    xmin, xmax, ymin, ymax = algo.adapter.plot_bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"GCBF+ certificate | ego {ego_agent} | frame {frame_index:02d}")
    ax.text(
        0.5,
        -0.10,
        f"scene policy: {policy_label}",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#444444",
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
    algo.load(str(models_dir), step)
    action_fn, policy_label = _make_action_source(args.policy_mode, algo, env, args.seed)
    snapshots = _collect_snapshots(env, action_fn, args.policy_mode, args)

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
                policy_label,
                args,
            )
            for frame_index, graph in enumerate(snapshots)
        ]
        path = output_dir / f"gcbfplus_ego_agent_{agent}.gif"
        _save_gif(frames, path, args.fps)
        written.append(path)
        print(f"wrote {path}", flush=True)
    print(f"checkpoint={models_dir / step}, env={type(env).__name__}", flush=True)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GCBF+ certificate contours")
    parser.add_argument("--gcbfplus-dir", type=Path, required=True)
    parser.add_argument("--step", default=None, help="checkpoint directory name; defaults to latest")
    parser.add_argument("--policy-mode", choices=("checkpoint", "nominal", "random", "zero"), default="checkpoint")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-limit", type=float, default=None)
    parser.add_argument("--show-goals", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
