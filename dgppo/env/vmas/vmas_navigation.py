import pathlib
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from dgppo.env.base import MultiAgentEnv
from dgppo.env.utils import get_node_goal_rng
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State
from dgppo.utils.utils import save_anim, tree_index


class VMASNavigationState(NamedTuple):
    a_pos: Array
    a_vel: Array
    goal_pos: Array


class VMASNavigationObsState(NamedTuple):
    a_pos: Array
    a_vel: Array
    goal_pos: Array
    o_pos: Array



class VMASNavigation(MultiAgentEnv):
    AGENT = 0

    PARAMS = {
        "comm_radius": 10.0,
        "default_area_size": 2.0,
        "dist2goal": 0.1,
        "agent_radius": 0.1,
        "world_spawning_x": 1.0,
        "world_spawning_y": 1.0,
        "pos_shaping_factor": 1.0,
        "final_reward": 0.01,
        "u_multiplier": 0.5,
        "damping": 0.75,
    }

    def __init__(
        self,
        num_agents: int,
        area_size: Optional[float] = None,
        max_step: int = 200,
        dt: float = 0.03,
        params: dict = None,
    ):
        params = self.PARAMS.copy() if params is None else params
        half_width = max(params["world_spawning_x"], params["world_spawning_y"])
        area_size = 2 * half_width
        super().__init__(num_agents, area_size, max_step, dt, params)
        self.half_width = half_width
        self.agent_radius = params["agent_radius"]

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def node_dim(self) -> int:
        return 6

    @property
    def edge_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def n_cost(self) -> int:
        return 1

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return ("agent collisions",)

    def reset(self, key: Array) -> GraphsTuple:
        agent_key, vel_key = jax.random.split(key)
        a_pos, goal_pos = get_node_goal_rng(
            agent_key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.agent_radius + 0.05,
            None,
            side_length_y=self.area_size,
        )
        offset = jnp.array([self.params["world_spawning_x"], self.params["world_spawning_y"]])
        a_pos = a_pos - offset
        goal_pos = goal_pos - offset
        a_vel = jax.random.uniform(
            vel_key, shape=(self.num_agents, 2), minval=-0.01, maxval=0.01
        )
        return self.get_graph(VMASNavigationState(a_pos, a_vel, goal_pos))

    def step(
        self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        action = self.clip_action(action)
        env_state: VMASNavigationState = graph.env_states

        u = action * self.params["u_multiplier"]
        a_vel = self.params["damping"] * env_state.a_vel + u * self.dt
        a_pos = env_state.a_pos + a_vel * self.dt
        a_pos = jnp.clip(a_pos, -self.half_width, self.half_width)

        env_state_new = env_state._replace(a_pos=a_pos, a_vel=a_vel)
        reward = self.get_reward(graph, env_state_new, action)
        cost = self.get_cost(graph)
        goal_reached = (
            jnp.linalg.norm(env_state_new.a_pos - env_state_new.goal_pos, axis=-1)
            < self.params["dist2goal"]
        )
        done = goal_reached.all()
        info = {}
        return self.get_graph(env_state_new), reward, cost, done, info

    def get_reward(
        self, graph: GraphsTuple, next_state: VMASNavigationState, action: Action
    ) -> Reward:
        env_state: VMASNavigationState = graph.env_states
        prev_dist = jnp.linalg.norm(env_state.a_pos - env_state.goal_pos, axis=-1)
        next_dist = jnp.linalg.norm(next_state.a_pos - next_state.goal_pos, axis=-1)
        progress = (prev_dist - next_dist).mean() * self.params["pos_shaping_factor"]
        final_rew = jnp.where(
            (next_dist < self.params["dist2goal"]).all(),
            self.params["final_reward"],
            0.0,
        )
        action_penalty = 0.001 * jnp.square(action).sum(axis=-1).mean()
        return progress + final_rew - action_penalty

    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASNavigationState = graph.env_states
        agent_cost = self._agent_collision_cost(env_state.a_pos)
        cost = agent_cost[:, None]
        assert cost.shape == (self.num_agents, self.n_cost)
        return jnp.clip(cost, a_min=-1.0, a_max=1.0)

    def _agent_collision_cost(self, a_pos: Array) -> Array:
        dist = jnp.linalg.norm(a_pos[:, None, :] - a_pos[None, :, :], axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        return 2 * self.agent_radius - min_dist

    def get_graph(self, env_state: VMASNavigationState) -> GraphsTuple:
        rel_goal_pos = env_state.goal_pos - env_state.a_pos
        node_feats = jnp.zeros((self.num_agents, self.node_dim))
        node_feats = node_feats.at[:, :2].set(env_state.a_pos)
        node_feats = node_feats.at[:, 2:4].set(env_state.a_vel)
        node_feats = node_feats.at[:, 4:6].set(rel_goal_pos)

        node_type = jnp.full(self.num_agents, VMASNavigation.AGENT)
        n_state_vec = jnp.concatenate([env_state.a_pos, env_state.a_vel], axis=-1)
        return GetGraph(
            node_feats, node_type, self.edge_blocks(env_state), env_state, n_state_vec
        ).to_padded()

    def edge_blocks(self, env_state: VMASNavigationState) -> list[EdgeBlock]:
        agent_states = jnp.concatenate([env_state.a_pos, env_state.a_vel], axis=-1)
        state_diff = agent_states[:, None, :] - agent_states[None, :, :]
        dist = jnp.linalg.norm(env_state.a_pos[:, None, :] - env_state.a_pos[None, :, :], axis=-1)
        mask = (jnp.eye(self.num_agents) == 0) & (dist <= self.params["comm_radius"])
        ids = jnp.arange(self.num_agents)
        return [EdgeBlock(state_diff, mask, ids, ids)]

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([-self.half_width, -self.half_width, -1.0, -1.0])
        upper_lim = jnp.array([self.half_width, self.half_width, 1.0, 1.0])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        return -jnp.ones(2), jnp.ones(2)

    def render_video(
        self,
        rollout: Rollout,
        video_path: pathlib.Path,
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 200,
        **kwargs,
    ) -> None:
        T_env_states = rollout.graph.env_states
        T_costs = rollout.costs

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
        ax.set_xlim(-1.05 * self.half_width, 1.05 * self.half_width)
        ax.set_ylim(-1.05 * self.half_width, 1.05 * self.half_width)
        ax.set_aspect("equal")
        ax.add_patch(
            plt.Rectangle(
                (-self.half_width, -self.half_width),
                2 * self.half_width,
                2 * self.half_width,
                fc="none",
                ec="0.4",
            )
        )

        goal_patches = [
            plt.Circle((0, 0), self.params["dist2goal"], color=f"C{ii}", alpha=0.25)
            for ii in range(self.num_agents)
        ]
        agent_patches = [
            plt.Circle((0, 0), self.agent_radius, color=f"C{ii}", zorder=5)
            for ii in range(self.num_agents)
        ]
        [ax.add_patch(patch) for patch in goal_patches + agent_patches]

        text_opts = dict(size=12, color="k", transform=ax.transAxes)
        step_text = ax.text(0.99, 1.01, "kk=0", va="bottom", ha="right", **text_opts)
        cost_text = ax.text(0.99, 1.05, "cost=0", va="bottom", ha="right", **text_opts)

        def init_fn():
            return [*goal_patches, *agent_patches, step_text, cost_text]

        def update(kk: int):
            env_state = tree_index(T_env_states, kk)
            for ii in range(self.num_agents):
                goal_patches[ii].set_center(np.asarray(env_state.goal_pos[ii]))
                agent_patches[ii].set_center(np.asarray(env_state.a_pos[ii]))
            step_text.set_text(f"kk={kk:04}")
            cost_text.set_text("cost={:+.3f}".format(float(T_costs[kk].max())))
            return [*goal_patches, *agent_patches, step_text, cost_text]

        ani = FuncAnimation(
            fig,
            update,
            frames=len(rollout.graph.n_node),
            init_func=init_fn,
            interval=33,
            blit=True,
        )
        save_anim(ani, video_path)


class VMASNavigationObs(VMASNavigation):
    PARAMS = VMASNavigation.PARAMS | {
        "n_obs": 3,
        "obstacle_radius": 0.1,
        "lidar_range": 0.35,
    }

    @property
    def node_dim(self) -> int:
        return 9

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return ("agent collisions", "obstacle collisions")

    def reset(self, key: Array) -> GraphsTuple:
        agent_key, obs_key, vel_key = jax.random.split(key, 3)
        a_pos, goal_pos = get_node_goal_rng(
            agent_key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.agent_radius + 0.05,
            None,
            side_length_y=self.area_size,
        )
        o_pos, _ = get_node_goal_rng(
            obs_key,
            self.area_size,
            2,
            self.params["n_obs"],
            2 * self.params["obstacle_radius"] + 0.05,
            None,
            side_length_y=self.area_size,
        )
        offset = jnp.array([self.params["world_spawning_x"], self.params["world_spawning_y"]])
        a_pos = a_pos - offset
        goal_pos = goal_pos - offset
        o_pos = o_pos - offset
        a_vel = jax.random.uniform(
            vel_key, shape=(self.num_agents, 2), minval=-0.01, maxval=0.01
        )
        return self.get_graph(VMASNavigationObsState(a_pos, a_vel, goal_pos, o_pos))

    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASNavigationObsState = graph.env_states
        agent_cost = self._agent_collision_cost(env_state.a_pos)
        obs_cost = self._obstacle_collision_cost(env_state.a_pos, env_state.o_pos)
        cost = jnp.stack([agent_cost, obs_cost], axis=-1)
        assert cost.shape == (self.num_agents, self.n_cost)
        return jnp.clip(cost, a_min=-1.0, a_max=1.0)

    def _obstacle_collision_cost(self, a_pos: Array, o_pos: Array) -> Array:
        dist = jnp.linalg.norm(a_pos[:, None, :] - o_pos[None, :, :], axis=-1)
        min_dist = jnp.min(dist, axis=1)
        return self.agent_radius + self.params["obstacle_radius"] - min_dist

    def get_graph(self, env_state: VMASNavigationObsState) -> GraphsTuple:
        rel_goal_pos = env_state.goal_pos - env_state.a_pos
        rel_obs = env_state.o_pos[None, :, :] - env_state.a_pos[:, None, :]
        obs_dist = jnp.sqrt(jnp.sum(rel_obs**2, axis=-1) + 1e-6)
        nearest_idx = jnp.argmin(obs_dist, axis=1)
        nearest_vec = jnp.take_along_axis(
            rel_obs, nearest_idx[:, None, None], axis=1
        ).squeeze(1)
        nearest_dist = jnp.take_along_axis(obs_dist, nearest_idx[:, None], axis=1)

        node_feats = jnp.zeros((self.num_agents, self.node_dim))
        node_feats = node_feats.at[:, :2].set(env_state.a_pos)
        node_feats = node_feats.at[:, 2:4].set(env_state.a_vel)
        node_feats = node_feats.at[:, 4:6].set(rel_goal_pos)
        node_feats = node_feats.at[:, 6:8].set(nearest_vec)
        node_feats = node_feats.at[:, 8:9].set(nearest_dist)

        node_type = jnp.full(self.num_agents, VMASNavigation.AGENT)
        n_state_vec = jnp.concatenate([env_state.a_pos, env_state.a_vel], axis=-1)
        return GetGraph(
            node_feats, node_type, self.edge_blocks(env_state), env_state, n_state_vec
        ).to_padded()

    def render_video(
        self,
        rollout: Rollout,
        video_path: pathlib.Path,
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 200,
        **kwargs,
    ) -> None:
        T_env_states = rollout.graph.env_states
        T_costs = rollout.costs

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
        ax.set_xlim(-1.05 * self.half_width, 1.05 * self.half_width)
        ax.set_ylim(-1.05 * self.half_width, 1.05 * self.half_width)
        ax.set_aspect("equal")
        ax.add_patch(
            plt.Rectangle(
                (-self.half_width, -self.half_width),
                2 * self.half_width,
                2 * self.half_width,
                fc="none",
                ec="0.4",
            )
        )

        obs_patches = [
            plt.Circle((0, 0), self.params["obstacle_radius"], color="0.5", alpha=0.65)
            for _ in range(self.params["n_obs"])
        ]
        goal_patches = [
            plt.Circle((0, 0), self.params["dist2goal"], color=f"C{ii}", alpha=0.25)
            for ii in range(self.num_agents)
        ]
        agent_patches = [
            plt.Circle((0, 0), self.agent_radius, color=f"C{ii}", zorder=5)
            for ii in range(self.num_agents)
        ]
        [ax.add_patch(patch) for patch in obs_patches + goal_patches + agent_patches]

        text_opts = dict(size=12, color="k", transform=ax.transAxes)
        step_text = ax.text(0.99, 1.01, "kk=0", va="bottom", ha="right", **text_opts)
        cost_text = ax.text(0.99, 1.05, "cost=0", va="bottom", ha="right", **text_opts)

        def init_fn():
            return [*obs_patches, *goal_patches, *agent_patches, step_text, cost_text]

        def update(kk: int):
            env_state = tree_index(T_env_states, kk)
            for oo in range(self.params["n_obs"]):
                obs_patches[oo].set_center(np.asarray(env_state.o_pos[oo]))
            for ii in range(self.num_agents):
                goal_patches[ii].set_center(np.asarray(env_state.goal_pos[ii]))
                agent_patches[ii].set_center(np.asarray(env_state.a_pos[ii]))
            step_text.set_text(f"kk={kk:04}")
            cost_text.set_text(
                "cost={}".format(
                    ", ".join([f"{float(v):+.3f}" for v in T_costs[kk].max(axis=0)])
                )
            )
            return [*obs_patches, *goal_patches, *agent_patches, step_text, cost_text]

        ani = FuncAnimation(
            fig,
            update,
            frames=len(rollout.graph.n_node),
            init_func=init_fn,
            interval=33,
            blit=True,
        )
        save_anim(ani, video_path)

