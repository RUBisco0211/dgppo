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
from .physax.entity import Agent
from .physax.shapes import Sphere
from .physax.world import World


class VMASNavigationState(NamedTuple):
    a_pos: Array
    a_vel: Array
    goal_pos: Array


class VMASNavigation(MultiAgentEnv):
    AGENT = 0

    PARAMS = {
        "comm_radius": 10.0,
        "default_area_size": 2.0,
        "dist2goal": 0.1,
        "agent_radius": 0.1,
        "world_spawning_x": 1.0,
        "world_spawning_y": 1.0,
        "enforce_bounds": False,
        "collisions": True,
        "shared_rew": True,
        "pos_shaping_factor": 1.0,
        "final_reward": 0.01,
        "agent_collision_penalty": 0.0,
        "min_collision_distance": 0.005,
        "u_multiplier": 1.0,
        "drag": 0.25,
        "substeps": 2,
        "collision_force": 100.0,
        "contact_margin": 1e-3,
        "cost_margin": 0.5,
        "dt": 0.1,
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
        super().__init__(num_agents, area_size, max_step, params.get("dt", dt), params)
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
        offset = jnp.array(
            [self.params["world_spawning_x"], self.params["world_spawning_y"]]
        )
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
        assert action.shape == (self.num_agents, self.action_dim)
        env_state: VMASNavigationState = graph.env_states

        x_semidim = self.params["world_spawning_x"] if self.params["enforce_bounds"] else None
        y_semidim = self.params["world_spawning_y"] if self.params["enforce_bounds"] else None
        world = World(
            dt=self.dt,
            substeps=self.params["substeps"],
            x_semidim=x_semidim,
            y_semidim=y_semidim,
            collision_force=self.params["collision_force"],
            contact_margin=self.params["contact_margin"],
        )

        agents = self._make_agents(env_state, action)
        agents, _ = world.step(agents)

        a_pos = jnp.stack([agent.state.pos for agent in agents], axis=0)
        a_vel = jnp.stack([agent.state.vel for agent in agents], axis=0)
        assert a_pos.shape == (self.num_agents, 2)
        assert a_vel.shape == (self.num_agents, 2)

        env_state_new = env_state._replace(a_pos=a_pos, a_vel=a_vel)
        next_graph = self.get_graph(env_state_new)
        reward = self.get_reward(graph, env_state_new)
        cost = self.get_cost(graph)
        done = jnp.array(False)
        info = {}
        return next_graph, reward, cost, done, info

    def _make_agents(
        self, env_state: VMASNavigationState, action: Action
    ) -> list[Agent]:
        agents = []
        for ii in range(self.num_agents):
            agent = Agent.create(
                f"agent_{ii}",
                shape=Sphere(self.agent_radius),
                collide=self.params["collisions"],
                rotatable=False,
                u_multiplier=self.params["u_multiplier"],
                drag=self.params["drag"],
            )
            agent = agent.withstate(pos=env_state.a_pos[ii], vel=env_state.a_vel[ii])
            agent = agent.withforce(force=action[ii] * agent.u_multiplier)
            agents.append(agent)
        return agents

    def get_reward(
        self, graph: GraphsTuple, next_state: VMASNavigationState
    ) -> Reward:
        env_state: VMASNavigationState = graph.env_states
        prev_dist = jnp.linalg.norm(env_state.a_pos - env_state.goal_pos, axis=-1)
        next_dist = jnp.linalg.norm(next_state.a_pos - next_state.goal_pos, axis=-1)
        pos_rew = (prev_dist - next_dist) * self.params["pos_shaping_factor"]
        progress = jnp.where(
            self.params["shared_rew"],
            pos_rew.sum(),
            pos_rew.mean(),
        )
        final_rew = jnp.where(
            (next_dist < self.params["dist2goal"]).all(),
            self.params["final_reward"],
            0.0,
        )
        collision_rew = self._agent_collision_reward(
            self._transition_agent_collision_cost(env_state, next_state)
        ).sum()
        return progress + final_rew + collision_rew

    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASNavigationState = graph.env_states
        agent_cost = self._agent_collision_cost(env_state.a_pos)
        cost = agent_cost[:, None]
        margin = self.params["cost_margin"]
        cost = jnp.where(cost <= 0.0, cost - margin, cost + margin)
        assert cost.shape == (self.num_agents, self.n_cost)
        return jnp.clip(cost, a_min=-1.0, a_max=1.0)

    def get_transition_cost(
        self, env_state: VMASNavigationState, next_state: VMASNavigationState
    ) -> Cost:
        agent_cost = self._transition_agent_collision_cost(env_state, next_state)
        cost = agent_cost[:, None]
        margin = self.params["cost_margin"]
        cost = jnp.where(cost <= 0.0, cost - margin, cost + margin)
        assert cost.shape == (self.num_agents, self.n_cost)
        return jnp.clip(cost, a_min=-1.0, a_max=1.0)

    def _transition_agent_collision_cost(
        self, env_state: VMASNavigationState, next_state: VMASNavigationState
    ) -> Array:
        return jnp.maximum(
            self._agent_collision_cost(env_state.a_pos),
            self._agent_collision_cost(next_state.a_pos),
        )

    def _agent_collision_cost(self, a_pos: Array) -> Array:
        dist = jnp.linalg.norm(a_pos[:, None, :] - a_pos[None, :, :], axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        return 2 * self.agent_radius - min_dist

    def _agent_collision_reward(self, signed_dist: Array) -> Array:
        collides = signed_dist >= -self.params["min_collision_distance"]
        return jnp.where(collides, self.params["agent_collision_penalty"], 0.0)

    def get_graph(self, env_state: VMASNavigationState) -> GraphsTuple:
        rel_goal_pos = env_state.a_pos - env_state.goal_pos
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
        dist = jnp.linalg.norm(
            env_state.a_pos[:, None, :] - env_state.a_pos[None, :, :], axis=-1
        )
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
