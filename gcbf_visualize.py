"""Visualize a trained DGPPO safety value as ego-centric contour GIFs.

DGPPO learns one ``V_h`` channel per environment cost.  Its training convention
is unsafe-positive: ``dot(V_h) + alpha * V_h <= 0`` is treated as safe.  To make
the figures directly comparable with safe-positive HJ/GCBF plots, this script
renders ``h = -V_h`` for one channel, or ``h = -max_k V_h,k`` for the default
worst-channel view.  Thus blue/positive is predicted safe and red/negative is
predicted unsafe.

At each rollout snapshot the other agents, goals, obstacles, velocities, and
RNN history are fixed while the selected ego position is swept over an x-y
grid.  One GIF is written for every selected ego agent.

Example
-------
python gcbf_visualize.py \
    --dgppo-dir logs/LidarSpread/dgppo/seed0_707102621_YGIV \
    --cost-channel worst --ego-agents all
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Polygon
from PIL import Image

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState
from dgppo.env.safety_constraint import safety_constraint


DEFAULT_DGPPO_DIR = Path("logs/LidarSpread/dgppo/seed0_707102621_YGIV")
DEFAULT_OUTPUT_DIR = Path("outputs/gcbf_visualization")


class Snapshot(NamedTuple):
    graph: Any
    rnn_state: Any


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _load_config(run_dir: Path) -> Any:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"DGPPO config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.UnsafeLoader)
    if _cfg_get(config, "algo") != "dgppo":
        raise ValueError(
            "gcbf_visualize.py requires a DGPPO run, got "
            f"algo={_cfg_get(config, 'algo')!r}"
        )
    return config


def _latest_step(models_dir: Path) -> int:
    steps = sorted(
        int(path.name)
        for path in models_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoints found in {models_dir}")
    return steps[-1]


def _make_env(config: Any, args: argparse.Namespace) -> LidarEnv:
    env = make_env(
        env_id=_cfg_get(config, "env"),
        num_agents=(
            int(_cfg_get(config, "num_agents"))
            if args.num_agents is None
            else args.num_agents
        ),
        num_obs=(
            int(_cfg_get(config, "obs"))
            if args.num_obs is None
            else args.num_obs
        ),
        n_rays=_cfg_get(config, "n_rays"),
        max_step=args.rollout_start + args.frames * args.frame_stride + 1,
        full_observation=_cfg_get(config, "full_observation", False),
    )
    if not isinstance(env, LidarEnv):
        raise TypeError(
            "gcbf_visualize.py currently supports the two-dimensional "
            f"LidarEnv family, got {type(env).__name__}"
        )
    return env


def _make_dgppo(config: Any, env: LidarEnv):
    return make_algo(
        algo="dgppo",
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
        use_rnn=_cfg_get(
            config, "use_rnn", not _cfg_get(config, "no_rnn", False)
        ),
        rnn_layers=_cfg_get(config, "rnn_layers", 1),
        rnn_step=_cfg_get(config, "rnn_step", 16),
        use_lstm=_cfg_get(config, "use_lstm", False),
        alpha=_cfg_get(config, "alpha", 10.0),
        cbf_eps=_cfg_get(config, "cbf_eps", 1e-2),
        cbf_weight=_cfg_get(config, "cbf_weight", 1.0),
        train_steps=_cfg_get(config, "steps", 100_000),
        cbf_schedule=_cfg_get(config, "cbf_schedule", True),
    )


def _load_weights(run_dir: Path, step: int | None, algo) -> tuple[int, Any, Any]:
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"DGPPO models directory not found: {models_dir}")
    step = _latest_step(models_dir) if step is None else step
    checkpoint_dir = models_dir / str(step)
    actor_path = checkpoint_dir / "actor.pkl"
    vh_path = checkpoint_dir / "Vh.pkl"
    for path in (actor_path, vh_path):
        if not path.is_file():
            raise FileNotFoundError(f"DGPPO checkpoint file not found: {path}")
    with actor_path.open("rb") as file:
        actor_params = pickle.load(file)
    with vh_path.open("rb") as file:
        vh_params = pickle.load(file)

    # Initializing at the target cardinality changes only runtime array shapes;
    # actor and decomposed V_h parameter trees remain shared across agents.
    algo.policy_train_state = algo.policy_train_state.replace(params=actor_params)
    algo.Vh_train_state = algo.Vh_train_state.replace(params=vh_params)
    return step, actor_params, vh_params


def _action_source(
    mode: str,
    algo,
    actor_params,
    env: LidarEnv,
    seed: int,
) -> tuple[Callable, Any, str]:
    initial_state = algo.init_rnn_state
    if mode == "checkpoint":
        def checkpoint_action(graph, rnn_state):
            return algo.act(graph, rnn_state, {"policy": actor_params})

        return jax.jit(checkpoint_action), initial_state, "DGPPO actor"

    if mode == "zero":
        def zero_action(_graph, rnn_state):
            return jnp.zeros((env.num_agents, env.action_dim)), rnn_state

        return zero_action, initial_state, "zero policy"

    rng = np.random.default_rng(seed)

    def random_action(_graph, rnn_state):
        action = rng.uniform(-1.0, 1.0, (env.num_agents, env.action_dim))
        return jnp.asarray(action, dtype=jnp.float32), rnn_state

    return random_action, initial_state, "uniform random policy"


def _collect_snapshots(
    env: LidarEnv,
    act: Callable,
    rnn_state: Any,
    *,
    seed: int,
    frames: int,
    frame_stride: int,
    rollout_start: int,
) -> list[Snapshot]:
    graph = env.reset(jr.PRNGKey(seed))
    snapshots = []
    total_steps = rollout_start + (frames - 1) * frame_stride + 1
    capture_steps = {
        rollout_start + frame * frame_stride for frame in range(frames)
    }
    for step in range(total_steps):
        if step in capture_steps:
            snapshots.append(
                Snapshot(
                    graph=jax.device_get(graph),
                    rnn_state=jax.device_get(rnn_state),
                )
            )
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


def _safe_value(raw_vh: jax.Array, channel: str) -> jax.Array:
    """Convert DGPPO's unsafe-positive V_h to a safe-positive scalar."""
    if channel == "worst":
        return -jnp.max(raw_vh)
    if channel == "agent":
        return -raw_vh[0]
    if channel == "obstacle":
        return -raw_vh[1]
    raise ValueError(f"unsupported cost channel: {channel}")


def _make_grid_evaluator(
    env: LidarEnv,
    algo,
    vh_params,
    ego_agent: int,
    cost_channel: str,
) -> Callable:
    def evaluate_one(
        xy: jax.Array,
        base_agents: jax.Array,
        goals: jax.Array,
        obstacles,
        rnn_state: jax.Array,
    ) -> jax.Array:
        agents = base_agents.at[ego_agent, :2].set(xy)
        env_state = LidarEnvState(agents, goals, obstacles)
        lidar_data = env.get_lidar_data(agents, obstacles)
        graph = env.get_graph(env_state, lidar_data)
        raw_vh, _ = algo.Vh.get_value(vh_params, graph, rnn_state)
        clearance = safety_constraint(
            env,
            graph,
            agent_margin=0.0,
            obstacle_margin=0.0,
            braking_accel=None,
            maximum_margin=env.params["comm_radius"],
        )
        return jnp.stack(
            [
                _safe_value(raw_vh[ego_agent], cost_channel),
                clearance[ego_agent],
                raw_vh[ego_agent, 0],
                raw_vh[ego_agent, 1],
            ]
        )

    return jax.jit(
        jax.vmap(evaluate_one, in_axes=(0, None, None, None, None))
    )


def _evaluate_contours(
    snapshots: Sequence[Snapshot],
    evaluators: dict[int, Callable],
    grid_points: jax.Array,
    zero_rnn_state: Any,
    rnn_state_mode: str,
) -> tuple[
    dict[int, list[np.ndarray]],
    dict[int, list[np.ndarray]],
    dict[int, list[np.ndarray]],
]:
    values = {ego: [] for ego in evaluators}
    clearances = {ego: [] for ego in evaluators}
    channels = {ego: [] for ego in evaluators}
    for frame_idx, snapshot in enumerate(snapshots):
        env_state = snapshot.graph.env_states
        rnn_state = (
            zero_rnn_state
            if rnn_state_mode == "zero"
            else snapshot.rnn_state
        )
        obstacles = jax.tree.map(jnp.asarray, env_state.obstacle)
        for ego, evaluator in evaluators.items():
            evaluated = np.asarray(
                evaluator(
                    grid_points,
                    jnp.asarray(env_state.agent),
                    jnp.asarray(env_state.goal),
                    obstacles,
                    jnp.asarray(rnn_state),
                )
            )
            values[ego].append(evaluated[:, 0])
            clearances[ego].append(evaluated[:, 1])
            channels[ego].append(evaluated[:, 2:4])
        print(f"evaluated contour frame {frame_idx + 1}/{len(snapshots)}", flush=True)
    return values, clearances, channels


def _symmetric_value_limit(
    values: dict[int, list[np.ndarray]], requested_limit: float | None
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
        raise ValueError("DGPPO V_h evaluation produced no finite values")
    return max(float(np.percentile(finite, 99.5)), 0.025)


def _draw_obstacles(ax, obstacles) -> None:
    if obstacles is None:
        return
    for polygon in np.asarray(obstacles.points):
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
    if finite.size > 0 and finite.min() <= 0.0 <= finite.max():
        ax.contour(x_grid, y_grid, field, levels=[0.0], **kwargs)


def _render_frame(
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    value_grid: np.ndarray,
    clearance_grid: np.ndarray,
    channel_grid: np.ndarray,
    snapshot: Snapshot,
    ego_agent: int,
    frame_idx: int,
    value_limit: float,
    env: LidarEnv,
    policy_label: str,
    cost_channel: str,
    rnn_state_mode: str,
    show_goals: bool,
    dpi: int,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=dpi, constrained_layout=True)
    levels = np.linspace(-value_limit, value_limit, 17)
    norm = TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit)
    contour = ax.contourf(
        x_grid,
        y_grid,
        np.clip(value_grid, -value_limit, value_limit),
        levels=levels,
        cmap="RdBu",
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
    _draw_zero_contour(
        ax,
        x_grid,
        y_grid,
        clearance_grid,
        colors="#5a5a5a",
        linestyles="--",
        linewidths=1.0,
        zorder=4,
    )

    state = snapshot.graph.env_states
    agents = np.asarray(state.agent)
    positions = agents[:, :2]
    _draw_obstacles(ax, state.obstacle)

    ego_position = positions[ego_agent]
    distances = np.linalg.norm(positions - ego_position, axis=-1)
    neighbors = (
        (distances < env.params["comm_radius"])
        & (np.arange(env.num_agents) != ego_agent)
    )
    for neighbor in np.flatnonzero(neighbors):
        ax.plot(
            [ego_position[0], positions[neighbor, 0]],
            [ego_position[1], positions[neighbor, 1]],
            color="#707070",
            linewidth=1.1,
            alpha=0.85,
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
            alpha=0.65,
            linewidths=1.2,
            label="task goals (visible to DGPPO Vh)",
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
    actual_clearance = clearance_grid[nearest_y, nearest_x]
    actual_channels = channel_grid[nearest_y, nearest_x]
    ax.text(
        0.015,
        0.985,
        (
            f"ego agent {ego_agent} | frame {frame_idx:02d}\n"
            f"h={actual_value:+.4f}, clearance={actual_clearance:+.4f}\n"
            f"raw Vh(agent/obs)={actual_channels[0]:+.3f}/"
            f"{actual_channels[1]:+.3f}\n"
            f"channel={cost_channel}, RNN={rnn_state_mode}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#777777"},
        zorder=20,
    )
    ax.set_title("DGPPO Learned Safety Value (ego position projected to x-y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, env.area_size)
    ax.set_ylim(0.0, env.area_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", linewidth=0.5, alpha=0.25)
    ax.text(
        0.5,
        -0.10,
        (
            f"scene policy: {policy_label}; black: learned h=0; "
            "gray dashed: geometric clearance=0"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label(
        "safe-positive DGPPO value h "
        f"({_channel_formula(cost_channel)})"
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _channel_formula(channel: str) -> str:
    if channel == "worst":
        return "-max(Vh_agent, Vh_obstacle)"
    if channel == "agent":
        return "-Vh_agent"
    return "-Vh_obstacle"


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

    run_dir = args.dgppo_dir.expanduser().resolve()
    config = _load_config(run_dir)
    env = _make_env(config, args)
    algo = _make_dgppo(config, env)
    step, actor_params, vh_params = _load_weights(run_dir, args.step, algo)
    act, rnn_state, policy_label = _action_source(
        args.policy_mode, algo, actor_params, env, args.seed
    )
    snapshots = _collect_snapshots(
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
            env, algo, vh_params, ego, args.cost_channel
        )
        for ego in ego_agents
    }
    values, clearances, channels = _evaluate_contours(
        snapshots,
        evaluators,
        grid_points,
        algo.init_rnn_state,
        args.rnn_state_mode,
    )
    value_limit = _symmetric_value_limit(values, args.value_limit)

    output_dir = args.output_dir.expanduser().resolve()
    written = []
    for ego in ego_agents:
        gif_frames = []
        for frame_idx, snapshot in enumerate(snapshots):
            gif_frames.append(
                _render_frame(
                    x_grid=x_grid,
                    y_grid=y_grid,
                    value_grid=values[ego][frame_idx].reshape(x_grid.shape),
                    clearance_grid=clearances[ego][frame_idx].reshape(x_grid.shape),
                    channel_grid=channels[ego][frame_idx].reshape(
                        x_grid.shape + (2,)
                    ),
                    snapshot=snapshot,
                    ego_agent=ego,
                    frame_idx=frame_idx,
                    value_limit=value_limit,
                    env=env,
                    policy_label=policy_label,
                    cost_channel=args.cost_channel,
                    rnn_state_mode=args.rnn_state_mode,
                    show_goals=args.show_goals,
                    dpi=args.dpi,
                )
            )
        output_path = output_dir / f"dgppo_cbf_ego_agent_{ego}.gif"
        _save_gif(gif_frames, output_path, args.fps)
        written.append(output_path)
        print(f"wrote {output_path}", flush=True)

    print(
        f"run={run_dir}, step={step}, "
        f"source_agents={_cfg_get(config, 'num_agents')}, "
        f"eval_agents={env.num_agents}, "
        f"source_obs={_cfg_get(config, 'obs')}, eval_obs={env.params['n_obs']}, "
        f"channel={args.cost_channel}, RNN={args.rnn_state_mode}, "
        f"value_limit=±{value_limit:.5f}",
        flush=True,
    )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ego-centric contours of a trained DGPPO V_h network."
    )
    parser.add_argument(
        "--dgppo-dir",
        type=Path,
        default=DEFAULT_DGPPO_DIR,
        help="DGPPO run containing config.yaml and models/",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="numeric models/<step> checkpoint; defaults to the latest",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("checkpoint", "random", "zero"),
        default="checkpoint",
        help="policy used only to produce scene snapshots",
    )
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument(
        "--num-obs", "--obs", dest="num_obs", type=int, default=None
    )
    parser.add_argument(
        "--cost-channel",
        choices=("worst", "agent", "obstacle"),
        default="worst",
        help="DGPPO V_h channel to render; worst combines both costs",
    )
    parser.add_argument(
        "--rnn-state-mode",
        choices=("rollout", "zero"),
        default="rollout",
        help="use captured rollout history or a zero hidden state for contours",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--value-limit",
        type=float,
        default=None,
        help="symmetric color limit; defaults to the 99.5th percentile",
    )
    parser.add_argument("--show-goals", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
