"""Environment seam used by the GCBF+ implementation.

The upstream GCBF+ environments expose controller-specific helpers such as
``u_ref`` and ``forward_graph``.  DGPPO environments intentionally expose the
smaller ``reset/step/get_cost`` interface instead.  The adapters in this file
keep the GCBF+ assumptions local and leave the environment classes unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp

from ..env.base import MultiAgentEnv
from ..env.lidar_env.base import LidarEnv, LidarEnvState
from ..env.lidar_env.lidar_bicycle_target import LidarBicycleTarget
from ..env.lidar_env.lidar_line import LidarLine
from ..env.vmas import (
    VMASNavigation,
    VMASNavigationObs,
    VMASReverseTransport,
    VMASWheel,
)
from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array


def _angle_difference(target: Array, current: Array) -> Array:
    return jnp.arctan2(jnp.sin(target - current), jnp.cos(target - current))


class GCBFPlusEnvAdapter(ABC):
    """The environment behaviour required by GCBF+ training and plotting."""

    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        """Return one environment-native collision label per agent."""
        cost = self.env.get_cost(graph)
        return jnp.max(cost, axis=-1) > 0.0

    @abstractmethod
    def nominal_action(self, graph: GraphsTuple) -> Action:
        """Return a task-directed action around which the safety QP operates."""

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        """Apply the real differentiable one-step environment transition."""
        next_graph, _, _, _, _ = self.env.step(graph, action)
        return next_graph

    @abstractmethod
    def agent_positions(self, graph: GraphsTuple) -> Array:
        """Return the two-dimensional positions used by contour plots."""

    @abstractmethod
    def with_agent_position(
        self, graph: GraphsTuple, agent: int, position: Array
    ) -> GraphsTuple:
        """Rebuild a graph after moving one agent in a frozen scene."""

    @property
    @abstractmethod
    def plot_bounds(self) -> tuple[float, float, float, float]:
        """Return ``xmin, xmax, ymin, ymax`` for contour evaluation."""

    @property
    def agent_radius(self) -> float:
        if hasattr(self.env, "agent_radius"):
            return float(self.env.agent_radius)
        return float(self.env.params["car_radius"])


class LidarGCBFPlusAdapter(GCBFPlusEnvAdapter):
    env: LidarEnv

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        """Differentiable GCBF+ prediction with the current LiDAR hits frozen.

        Ray/rectangle intersection contains discrete hit selection and unstable
        derivatives at corners.  Upstream GCBF+ likewise keeps sensed obstacle
        points fixed during its one-step actor/CBF update.
        """
        state: LidarEnvState = graph.env_states
        action = self.env.clip_action(action)
        agents = self.env.agent_step_euler(state.agent, action)
        new_state = state._replace(agent=agents)
        if self.env.params["n_obs"] > 0:
            n_hits = self.env.params["top_k_rays"] * self.env.num_agents
            lidar_data = graph.type_states(type_idx=self.env.OBS, n_type=n_hits)[:, :2]
            lidar_data = lidar_data.reshape(
                self.env.num_agents, self.env.params["top_k_rays"], 2
            )
        else:
            lidar_data = None
        return self.env.get_graph(new_state, lidar_data)

    def nominal_action(self, graph: GraphsTuple) -> Action:
        state: LidarEnvState = graph.env_states
        goal_positions = state.goal[:, :2]
        if isinstance(self.env, LidarLine):
            goal_positions = self.env.landmark2goal(goal_positions)
        position_error = goal_positions - state.agent[:, :2]

        if isinstance(self.env, LidarBicycleTarget):
            heading = jnp.arctan2(state.agent[:, 3], state.agent[:, 2])
            desired_heading = jnp.arctan2(position_error[:, 1], position_error[:, 0])
            heading_error = _angle_difference(desired_heading, heading)
            speed = state.agent[:, 4]
            target_speed = jnp.minimum(jnp.linalg.norm(position_error, axis=-1), 0.5)
            turn_denominator = 10.0 * jnp.maximum(jnp.abs(speed), 0.1) * self.env.dt
            steering = heading_error / turn_denominator
            acceleration = (target_speed - speed) / (10.0 * self.env.dt)
            action = jnp.stack([steering, acceleration], axis=-1)
        else:
            velocity = state.agent[:, 2:4]
            # LidarEnv scales actions by ten before Euler integration.
            action = (2.0 * position_error - 1.0 * velocity) / 10.0
        return self.env.clip_action(action)

    def agent_positions(self, graph: GraphsTuple) -> Array:
        return graph.env_states.agent[:, :2]

    def with_agent_position(
        self, graph: GraphsTuple, agent: int, position: Array
    ) -> GraphsTuple:
        state: LidarEnvState = graph.env_states
        agents = state.agent.at[agent, :2].set(position)
        new_state = state._replace(agent=agents)
        lidar_data = self.env.get_lidar_data(agents, state.obstacle)
        return self.env.get_graph(new_state, lidar_data)

    @property
    def plot_bounds(self) -> tuple[float, float, float, float]:
        return 0.0, self.env.area_size, 0.0, self.env.area_size


class VMASGCBFPlusAdapter(GCBFPlusEnvAdapter):
    env: VMASNavigation | VMASNavigationObs | VMASReverseTransport | VMASWheel

    def nominal_action(self, graph: GraphsTuple) -> Action:
        state = graph.env_states
        if isinstance(self.env, VMASWheel):
            angle_error = _angle_difference(state.goal_angle, state.line_angle)
            radius = jnp.linalg.norm(state.a_pos, axis=-1, keepdims=True)
            radial = state.a_pos / jnp.maximum(radius, 1e-3)
            tangent = jnp.stack([-radial[:, 1], radial[:, 0]], axis=-1)
            action = jnp.sign(angle_error) * tangent - 0.25 * state.a_vel
        elif isinstance(self.env, VMASReverseTransport):
            direction = state.goal_pos - state.box_pos
            direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-3)
            action = jnp.broadcast_to(direction, state.a_pos.shape) - 0.25 * state.a_vel
        else:
            position_error = state.goal_pos - state.a_pos
            action = 1.5 * position_error - 0.75 * state.a_vel
        return self.env.clip_action(action)

    def agent_positions(self, graph: GraphsTuple) -> Array:
        return graph.env_states.a_pos

    def with_agent_position(
        self, graph: GraphsTuple, agent: int, position: Array
    ) -> GraphsTuple:
        state = graph.env_states
        new_state = state._replace(a_pos=state.a_pos.at[agent].set(position))
        return self.env.get_graph(new_state)

    @property
    def plot_bounds(self) -> tuple[float, float, float, float]:
        half_width = float(self.env.half_width)
        return -half_width, half_width, -half_width, half_width


def make_gcbf_plus_env_adapter(env: MultiAgentEnv) -> GCBFPlusEnvAdapter:
    if isinstance(env, LidarEnv):
        return LidarGCBFPlusAdapter(env)
    if isinstance(
        env,
        (VMASNavigation, VMASNavigationObs, VMASReverseTransport, VMASWheel),
    ):
        return VMASGCBFPlusAdapter(env)
    raise ValueError(
        "GCBF+ supports the LidarEnv and VMAS environment families; "
        f"got {type(env).__name__}"
    )
