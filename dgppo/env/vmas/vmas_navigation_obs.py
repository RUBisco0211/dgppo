import pathlib
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from dgppo.env.utils import get_node_goal_rng
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward
from dgppo.utils.utils import save_anim, tree_index
from .physax.entity import Entity
from .physax.shapes import Sphere
from .physax.world import World
from .vmas_navigation import VMASNavigation, VMASNavigationState


class VMASNavigationObsState(NamedTuple):
    a_pos: Array
    a_vel: Array
    goal_pos: Array
    o_pos: Array


class VMASNavigationObs(VMASNavigation):
    OBS = 2

    PARAMS = VMASNavigation.PARAMS | {
        "n_obs": 3,
        "obstacle_radius": 0.1,
        "obstacle_collision_penalty": 0.0,
    }

    @property
    def node_dim(self) -> int:
        # [pos(2), vel(2), rel_goal(2), is_obs(1), is_agent(1)]
        return 8

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return ("agent collisions", "obstacle collisions")

    def reset(self, key: Array) -> GraphsTuple:
        entity_key, goal_key, vel_key = jax.random.split(key, 3)
        entity_pos, _ = get_node_goal_rng(
            entity_key,
            self.area_size,
            2,
            self.num_agents + self.params["n_obs"],
            self.agent_radius + self.params["obstacle_radius"] + 0.05,
            None,
            side_length_y=self.area_size,
        )
        _, goal_pos = get_node_goal_rng(
            goal_key,
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
        a_pos = entity_pos[: self.num_agents] - offset
        goal_pos = goal_pos - offset
        o_pos = entity_pos[self.num_agents :] - offset
        a_vel = jax.random.uniform(
            vel_key, shape=(self.num_agents, 2), minval=-0.01, maxval=0.01
        )
        env_state = VMASNavigationObsState(a_pos, a_vel, goal_pos, o_pos)
        return self.get_graph(env_state)

    def step(
        self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        action = self.clip_action(action)
        assert action.shape == (self.num_agents, self.action_dim)
        env_state: VMASNavigationObsState = graph.env_states

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

        agents = self._make_agents(
            VMASNavigationState(env_state.a_pos, env_state.a_vel, env_state.goal_pos),
            action,
        )
        obstacles = self._make_obstacles(env_state)
        entities, _ = world.step([*agents, *obstacles])
        agents = entities[: self.num_agents]

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

    def _make_obstacles(self, env_state: VMASNavigationObsState) -> list[Entity]:
        obstacles = []
        for ii in range(self.params["n_obs"]):
            obstacle = Entity.create(
                f"obstacle_{ii}",
                movable=False,
                rotatable=False,
                collide=True,
                shape=Sphere(self.params["obstacle_radius"]),
            )
            obstacle = obstacle.withstate(pos=env_state.o_pos[ii], vel=jnp.zeros(2))
            obstacles.append(obstacle)
        return obstacles

    def get_reward(
        self, graph: GraphsTuple, next_state: VMASNavigationObsState
    ) -> Reward:
        env_state: VMASNavigationObsState = graph.env_states
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
        agent_collision_rew = self._agent_collision_reward(
            self._transition_agent_collision_cost(env_state, next_state)
        ).sum()
        obs_collision_rew = self._obstacle_collision_reward(
            self._transition_obstacle_collision_cost(env_state, next_state)
        ).sum()
        return progress + final_rew + agent_collision_rew + obs_collision_rew

    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASNavigationObsState = graph.env_states
        cost = jnp.stack(
            [
                self._agent_collision_cost(env_state.a_pos),
                self._obstacle_collision_cost(env_state.a_pos, env_state.o_pos),
            ],
            axis=-1,
        )
        margin = self.params["cost_margin"]
        cost = jnp.where(cost <= 0.0, cost - margin, cost + margin)
        assert cost.shape == (self.num_agents, self.n_cost)
        return jnp.clip(cost, a_min=-1.0, a_max=1.0)

    def get_transition_cost(
        self, env_state: VMASNavigationObsState, next_state: VMASNavigationObsState
    ) -> Cost:
        cost = jnp.stack(
            [
                self._transition_agent_collision_cost(env_state, next_state),
                self._transition_obstacle_collision_cost(env_state, next_state),
            ],
            axis=-1,
        )
        margin = self.params["cost_margin"]
        cost = jnp.where(cost <= 0.0, cost - margin, cost + margin)
        assert cost.shape == (self.num_agents, self.n_cost)
        return jnp.clip(cost, a_min=-1.0, a_max=1.0)

    def _transition_obstacle_collision_cost(
        self, env_state: VMASNavigationObsState, next_state: VMASNavigationObsState
    ) -> Array:
        return jnp.maximum(
            self._obstacle_collision_cost(env_state.a_pos, env_state.o_pos),
            self._obstacle_collision_cost(next_state.a_pos, next_state.o_pos),
        )

    def _obstacle_collision_cost(self, a_pos: Array, o_pos: Array) -> Array:
        dist = jnp.linalg.norm(a_pos[:, None, :] - o_pos[None, :, :], axis=-1)
        min_dist = jnp.min(dist, axis=1)
        return self.agent_radius + self.params["obstacle_radius"] - min_dist

    def _obstacle_collision_reward(self, signed_dist: Array) -> Array:
        collides = signed_dist >= -self.params["min_collision_distance"]
        return jnp.where(collides, self.params["obstacle_collision_penalty"], 0.0)

    def get_graph(self, env_state: VMASNavigationObsState) -> GraphsTuple:
        rel_goal_pos = env_state.a_pos - env_state.goal_pos
        n_nodes = self.num_agents + self.params["n_obs"]

        node_feats = jnp.zeros((n_nodes, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :2].set(env_state.a_pos)
        node_feats = node_feats.at[: self.num_agents, 2:4].set(env_state.a_vel)
        node_feats = node_feats.at[: self.num_agents, 4:6].set(rel_goal_pos)
        node_feats = node_feats.at[: self.num_agents, 7].set(1.0)
        node_feats = node_feats.at[self.num_agents :, :2].set(env_state.o_pos)
        node_feats = node_feats.at[self.num_agents :, 6].set(1.0)

        node_type = jnp.full(n_nodes, -1, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(VMASNavigation.AGENT)
        node_type = node_type.at[self.num_agents :].set(VMASNavigationObs.OBS)

        a_state = jnp.concatenate([env_state.a_pos, env_state.a_vel], axis=-1)
        o_state = jnp.concatenate(
            [env_state.o_pos, jnp.zeros((self.params["n_obs"], 2))], axis=-1
        )
        n_state_vec = jnp.concatenate([a_state, o_state], axis=0)
        return GetGraph(
            node_feats, node_type, self.edge_blocks(env_state), env_state, n_state_vec
        ).to_padded()

    def edge_blocks(self, env_state: VMASNavigationObsState) -> list[EdgeBlock]:
        agent_states = jnp.concatenate([env_state.a_pos, env_state.a_vel], axis=-1)
        agent_diff = agent_states[:, None, :] - agent_states[None, :, :]
        agent_dist = jnp.linalg.norm(
            env_state.a_pos[:, None, :] - env_state.a_pos[None, :, :], axis=-1
        )
        agent_mask = (
            (jnp.eye(self.num_agents) == 0)
            & (agent_dist <= self.params["comm_radius"])
        )
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(agent_diff, agent_mask, id_agent, id_agent)

        o_state = jnp.concatenate(
            [env_state.o_pos, jnp.zeros((self.params["n_obs"], 2))], axis=-1
        )
        obs_diff = agent_states[:, None, :] - o_state[None, :, :]
        obs_dist = jnp.linalg.norm(
            env_state.a_pos[:, None, :] - env_state.o_pos[None, :, :], axis=-1
        )
        obs_mask = obs_dist <= self.params["comm_radius"]
        id_obs = jnp.arange(self.params["n_obs"]) + self.num_agents
        agent_obs_edges = EdgeBlock(obs_diff, obs_mask, id_agent, id_obs)

        return [agent_agent_edges, agent_obs_edges]

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
            plt.Circle(
                (0, 0),
                self.params["obstacle_radius"],
                facecolor="0.35",
                edgecolor="0.05",
                linewidth=1.5,
                alpha=0.85,
                zorder=3,
            )
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
