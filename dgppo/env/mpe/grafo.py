"""Graph-based formation tasks for continual MARL.

The task graph is fixed for an episode. Only the reference landmark is an
observable goal node; target offsets are used by the reward and evaluator.
"""

from dataclasses import dataclass
from itertools import permutations
from typing import NamedTuple, Optional

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.env.mpe.base import MPE, MPEEnvGraphsTuple
from dgppo.utils.graph import EdgeBlock, GraphsTuple
from dgppo.utils.typing import Action, Array, Pos2d, Reward
from dgppo.utils.utils import tree_index


@dataclass(frozen=True)
class GrafoTask:
    name: str
    adjacency: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[float, float], ...]
    equivalent: tuple[tuple[int, ...], ...] = ()
    edge_target_distance: bool = False

    def __post_init__(self):
        n = len(self.offsets)
        adjacency = np.asarray(self.adjacency)
        if adjacency.shape != (n, n) or not np.array_equal(adjacency, adjacency.T):
            raise ValueError("adjacency must be a symmetric N x N matrix")
        if np.any(np.diag(adjacency)) or not np.isin(adjacency, (0, 1)).all():
            raise ValueError("adjacency must be binary without self edges")
        if sum(tuple(x) == (0.0, 0.0) for x in self.offsets) != 1:
            raise ValueError("exactly one reference offset must be (0, 0)")
        members = [i for group in self.equivalent for i in group]
        if len(set(members)) != len(members) or any(i < 0 or i >= n for i in members):
            raise ValueError("equivalent groups must contain distinct valid indices")


def _task(name, offsets, edges, equivalent=(), edge_target_distance=False):
    n = len(offsets)
    adjacency = [[0] * n for _ in range(n)]
    for i, j in edges:
        adjacency[i - 1][j - 1] = adjacency[j - 1][i - 1] = 1
    return GrafoTask(name, tuple(map(tuple, adjacency)), tuple(offsets),
                     tuple(equivalent), edge_target_distance)


# Graphs and offsets transcribed from Figure 9 in arXiv:2610.10302v1 source.
GRAFO_3 = GrafoTask(
    "GRAFO-3", ((0, 1, 0), (1, 0, 1), (0, 1, 0)),
    ((-1., 0.), (0., 0.), (1., 0.)), ((0, 2),),
)
GRAFO_4_1 = GrafoTask(
    "GRAFO-4(1)",
    ((0, 1, 0, 0), (1, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)),
    ((0., 0.), (1., 0.), (2., .5), (2., -.5)), ((2, 3),), True,
)
GRAFO_4_2 = GrafoTask(
    "GRAFO-4(2)", GRAFO_4_1.adjacency,
    ((0., 0.), (1., 0.), (2., -1.5), (2., 1.5)), ((2, 3),), True,
)
GRAFO_4_3 = GrafoTask(
    "GRAFO-4(3)", GRAFO_4_1.adjacency,
    ((0., 0.), (2., 0.), (4., -.5), (4., .5)), ((2, 3),), True,
)
GRAFO_4_4 = GrafoTask(
    "GRAFO-4(4)", GRAFO_4_1.adjacency,
    ((0., 0.), (.5, 0.), (1., -.5), (1., .5)), ((2, 3),), True,
)
GRAFO_5 = _task(
    "GRAFO-5", ((0., 0.), (1., 0.), (1., .5), (2., 0.), (2., .5)),
    ((1, 2), (2, 3), (2, 4), (3, 5), (4, 5)), ((2, 3),),
)
GRAFO_6_1 = _task(
    "GRAFO-6(1)", ((0., 0.), (1., 0.), (2., 0.), (3., 0.),
                    (4., -.5), (4., .5)),
    ((1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)), ((4, 5),),
)
GRAFO_6_2 = _task(
    "GRAFO-6(2)", ((0., 0.), (1., 0.), (2., 0.), (3., .5),
                    (4., 0.), (4., .5)),
    ((1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)), ((3, 4),),
)
GRAFO_6_3 = _task(
    "GRAFO-6(3)", ((0., 0.), (1., .5), (1., 0.), (1., -.5),
                    (2., -.5), (2., 0.)),
    ((1, 2), (1, 3), (2, 3), (3, 4), (3, 6), (4, 5), (5, 6)),
)
GRAFO_6_4 = _task(
    "GRAFO-6(4)", ((0., 0.), (1., 0.), (2., 0.), (3., .5),
                    (4., 0.), (5., 0.)),
    ((1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)),
)
GRAFO_6_5 = _task(
    "GRAFO-6(5)", ((0., 0.), (1., 0.), (1., .5), (2., 0.),
                    (2., .5), (3., 0.)),
    ((1, 2), (2, 3), (3, 5), (2, 4), (4, 5), (4, 6), (5, 6)),
)
GRAFO_7 = _task(
    "GRAFO-7", ((0., 0.), (1., 0.), (1., .5), (2., 0.),
                (2., -.5), (3., 0.), (4., 0.)),
    ((1, 2), (1, 3), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7)),
)
GRAFO_4 = GrafoTask("GRAFO-4", GRAFO_4_1.adjacency,
                    GRAFO_4_1.offsets, GRAFO_4_1.equivalent, False)
GRAFO_TASKS = {task.name: task for task in (
    GRAFO_3, GRAFO_4, GRAFO_4_1, GRAFO_4_2, GRAFO_4_3, GRAFO_4_4,
    GRAFO_5, GRAFO_6_1, GRAFO_6_2, GRAFO_6_3, GRAFO_6_4,
    GRAFO_6_5, GRAFO_7)}

GRAFO_SEQUENCES = {
    "TS": ("GRAFO-5", "GRAFO-6(1)", "GRAFO-7", "GRAFO-4",
           "GRAFO-3", "GRAFO-6(3)"),
    "TS-inc": ("GRAFO-3", "GRAFO-4", "GRAFO-5", "GRAFO-6(1)",
               "GRAFO-6(3)", "GRAFO-7"),
    "TS-dec": ("GRAFO-7", "GRAFO-6(3)", "GRAFO-6(1)", "GRAFO-5",
               "GRAFO-4", "GRAFO-3"),
    "CS": tuple(f"GRAFO-6({i})" for i in range(1, 6)),
    "GFS": tuple(f"GRAFO-4({i})" for i in range(1, 5)),
}


class GrafoEnvState(NamedTuple):
    agent: Array
    goal: Array
    obs: Array
    reached: Array


class GRAFO(MPE):
    """MPE dynamics with graph-defined, unobserved formation targets."""

    PARAMS = {
        "car_radius": 0.05, "comm_radius": 0.5, "n_obs": 0,
        "obs_radius": 0.05, "default_area_size": 8.0,
        "dist2goal": 0.05, "goal_scale": 1.0, "reach_bonus": 1.0,
        "initial_agent_radius": 1.0, "initial_landmark_radius": 0.5,
        "acceleration": 5.0, "damping": 0.25,
    }

    def __init__(self, task: GrafoTask = GRAFO_3, num_agents: Optional[int] = None,
                 area_size: Optional[float] = None, max_step: int = 50,
                 dt: float = 0.1, params: Optional[dict] = None,
                 task_index: Optional[int] = None, n_tasks: Optional[int] = None,
                 action_encoding: str = "continuous",
                 reach_threshold: Optional[float] = None):
        if num_agents is not None and num_agents != len(task.offsets):
            raise ValueError("num_agents must match the task graph")
        if (task_index is None) != (n_tasks is None):
            raise ValueError("task_index and n_tasks must be provided together")
        if n_tasks is not None and (n_tasks < 2 or not 0 <= task_index < n_tasks):
            raise ValueError("task identity requires at least two tasks and a valid index")
        self.task = task
        if action_encoding not in ("continuous", "discrete"):
            raise ValueError("action_encoding must be continuous or discrete")
        self.action_encoding = action_encoding
        self.task_index = task_index
        self.n_tasks = n_tasks or 0
        config = self.PARAMS.copy()
        if params is not None:
            config.update(params)
        if reach_threshold is not None:
            if reach_threshold <= 0:
                raise ValueError("reach_threshold must be positive")
            config["dist2goal"] = reach_threshold
        if config["n_obs"] != 0:
            raise ValueError("paper GRAFO has no obstacles")
        super().__init__(len(task.offsets), area_size or config["default_area_size"],
                         max_step, dt, config)
        self.num_goals = 1
        self._adjacency = jnp.asarray(task.adjacency, dtype=jnp.bool_)
        self._offsets = jnp.asarray(task.offsets, dtype=jnp.float32) * config["goal_scale"]
        self._permutations = self._make_permutations()

    @property
    def edge_dim(self) -> int:
        # Keep model parameter shapes identical across all task sequences.
        return 5

    @property
    def action_dim(self) -> int:
        return 5 if self.action_encoding == "discrete" else 2

    def action_to_force(self, action: Action) -> Array:
        if self.action_encoding == "continuous":
            return self.clip_action(action)
        # JaxInforMARL Formation: 0 stay, 1 left, 2 right, 3 down, 4 up.
        directions = jnp.asarray(((0., 0.), (-1., 0.), (1., 0.),
                                  (0., -1.), (0., 1.)), dtype=jnp.float32)
        return directions[jnp.argmax(action, axis=-1)]

    @property
    def node_dim(self) -> int:
        return super().node_dim + self.n_tasks

    def _make_permutations(self) -> Array:
        assignments = [tuple(range(self.num_agents))]
        for group in self.task.equivalent:
            assignments = [tuple(p[i] if i not in group else order[group.index(i)]
                                 for i in range(self.num_agents))
                           for p in assignments for order in permutations(group)]
        return jnp.asarray(assignments, dtype=jnp.int32)

    def reset(self, key: Array) -> GraphsTuple:
        agent_key, landmark_key = jr.split(key)
        # Formation reference sampling: agents in [-1, 1]^2 and landmark in
        # [-0.5, 0.5]^2. Target offsets may extend beyond the initial region.
        landmark_radius = self.params["initial_landmark_radius"]
        landmark = jr.uniform(landmark_key, (2,), minval=-landmark_radius,
                              maxval=landmark_radius)
        agent_radius = self.params["initial_agent_radius"]
        positions = jr.uniform(agent_key, (self.num_agents, 2),
                               minval=-agent_radius, maxval=agent_radius)
        agents = jnp.concatenate((positions, jnp.zeros_like(positions)), axis=-1)
        goal = jnp.concatenate((landmark[None], jnp.zeros((1, 2))), axis=-1)
        return self.get_graph(GrafoEnvState(
            agents, goal, jnp.zeros((0, 4)), jnp.zeros((self.num_agents,), dtype=bool)))

    def agent_step_euler(self, agent_states: Array, action: Action) -> Array:
        """Cited JaxInforMARL integrator: move, then damp and apply force."""
        position = agent_states[:, :2] + agent_states[:, 2:] * self.dt
        velocity = (agent_states[:, 2:] * (1.0 - self.params["damping"]) +
                    action * self.params["acceleration"] * self.dt)
        return jnp.concatenate((position, velocity), axis=-1)

    def render_plot_bounds(self, rollout):
        positions = np.asarray(rollout.graph.states[..., :2])
        goals = np.asarray(self.render_task_goal_positions(
            tree_index(rollout.graph, 0)))
        extent = np.concatenate((positions.reshape(-1, 2), goals), axis=0)
        low = extent.min(axis=0) - 0.3
        high = extent.max(axis=0) + 0.3
        return float(low[0]), float(high[0]), float(low[1]), float(high[1])

    def _assigned_targets(self, graph: MPEEnvGraphsTuple) -> Array:
        positions = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        landmark = graph.type_states(type_idx=1, n_type=1)[0, :2]
        targets = landmark + self._offsets
        candidates = targets[self._permutations]
        distances = jnp.linalg.norm(positions[None] - candidates, axis=-1)
        best = jnp.argmin(distances.sum(axis=-1))
        return candidates[best]

    def _assigned_distances(self, graph: MPEEnvGraphsTuple) -> Array:
        positions = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        return jnp.linalg.norm(positions - self._assigned_targets(graph), axis=-1)

    def success(self, graph: MPEEnvGraphsTuple) -> Array:
        return jnp.all(self._assigned_distances(graph) <= self.params["dist2goal"])

    def get_reward(self, graph: MPEEnvGraphsTuple, action: Action) -> Reward:
        distances = self._assigned_distances(graph)
        newly_reached = ((distances <= self.params["dist2goal"]) &
                         ~graph.env_states.reached)
        return -distances.mean() + self.params["reach_bonus"] * newly_reached.mean()

    def step(self, graph: MPEEnvGraphsTuple, action: Action, get_eval_info: bool = False):
        state = graph.env_states
        moved = self.agent_step_euler(state.agent, self.action_to_force(action))
        agents = jnp.where(state.reached[:, None], state.agent, moved)
        next_state = GrafoEnvState(agents, state.goal, state.obs, state.reached)
        next_graph = self.get_graph(next_state)
        distances = self._assigned_distances(next_graph)
        newly_reached = (distances <= self.params["dist2goal"]) & ~state.reached
        reached = state.reached | newly_reached
        # Match the cited JaxInforMARL target environment: a reaching agent is
        # placed on its assigned target and cannot drift away on later steps.
        targets = self._assigned_targets(next_graph)
        settled_agents = jnp.concatenate(
            (jnp.where(reached[:, None], targets, agents[:, :2]),
             jnp.where(reached[:, None], 0.0, agents[:, 2:])), axis=-1)
        next_state = next_state._replace(agent=settled_agents, reached=reached)
        next_graph = self.get_graph(next_state)
        reward = (-self._assigned_distances(next_graph).mean() +
                  self.params["reach_bonus"] * newly_reached.mean())
        return next_graph, reward, self.get_cost(graph), jnp.all(reached), {}

    def get_cost(self, graph: MPEEnvGraphsTuple) -> Array:
        # Collisions are disabled in the paper's GRAFO experiments.
        return jnp.zeros((self.num_agents, self.n_cost), dtype=jnp.float32)

    def edge_blocks(self, state: GrafoEnvState) -> list[EdgeBlock]:
        agents = state.agent
        diff = agents[:, None, :] - agents[None, :, :]
        if self.task.edge_target_distance:
            target_distance = jnp.linalg.norm(
                self._offsets[:, None, :] - self._offsets[None, :, :], axis=-1)
            diff = jnp.concatenate((diff, target_distance[..., None]), axis=-1)
        else:
            diff = jnp.pad(diff, ((0, 0), (0, 0), (0, 1)))
        ids = jnp.arange(self.num_agents)
        agent_edges = EdgeBlock(diff, self._adjacency, ids, ids)
        goal_diff = agents[:, None, :] - state.goal[None, :, :]
        goal_diff = jnp.pad(goal_diff, ((0, 0), (0, 0), (0, 1)))
        goal_edges = EdgeBlock(goal_diff, jnp.ones((self.num_agents, 1), dtype=bool),
                               ids, jnp.array([self.num_agents]))
        return [agent_edges, goal_edges]

    def get_graph(self, env_state: GrafoEnvState) -> MPEEnvGraphsTuple:
        # InforMARL uses an agent-relative frame. Keep absolute coordinates in
        # states for physics/rewards, but expose only local coordinates to PPO.
        graph = super().get_graph(env_state)
        nodes = graph.nodes
        relative_goal = env_state.agent[:, :2] - env_state.goal[0, :2]
        nodes = nodes.at[:self.num_agents, :2].set(relative_goal)
        nodes = nodes.at[self.num_agents, :2].set(0.0)
        if self.n_tasks:
            nodes = nodes.at[:self.num_agents, 7 + self.task_index].set(1.0)
        return graph._replace(nodes=nodes)

    def render_task_goal_positions(self, graph: MPEEnvGraphsTuple) -> Pos2d:
        landmark = graph.type_states(type_idx=1, n_type=1)[0, :2]
        return landmark + self._offsets
