import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jax.random as jr

from dgppo.env import make_env


class MPERenderTest(unittest.TestCase):
    @staticmethod
    def _single_graph_rollout(graph):
        return SimpleNamespace(
            graph=jax.tree.map(lambda value: value[None], graph)
        )

    def test_line_render_exposes_derived_agent_goals(self):
        env = make_env("MPELine", 3, num_obs=1, max_step=1)
        graph = env.reset(jr.PRNGKey(0))

        task_goals = env.render_task_goal_positions(graph)
        landmarks = graph.type_states(type_idx=1, n_type=env.num_goals)[:, :2]

        self.assertEqual(task_goals.shape, (env.num_agents, 2))
        self.assertTrue(bool(jnp.allclose(task_goals, env.landmark2goal(landmarks))))

    def test_formation_render_exposes_derived_agent_goals(self):
        env = make_env("MPEFormation", 3, num_obs=1, max_step=1)
        graph = env.reset(jr.PRNGKey(1))

        task_goals = env.render_task_goal_positions(graph)
        landmark = graph.type_states(type_idx=1, n_type=env.num_goals)[:, :2]

        self.assertEqual(task_goals.shape, (env.num_agents, 2))
        self.assertTrue(bool(jnp.allclose(
            task_goals,
            env.landmark2goal(landmark, env.params["comm_radius"]),
        )))

    def test_corridor_render_expands_to_cover_the_rollout(self):
        env = make_env("MPECorridor", 3, num_obs=2, max_step=1)
        graph = env.reset(jr.PRNGKey(2))
        agents = graph.env_states.agent.at[0, 1].set(1.5)
        graph = env.get_graph(graph.env_states._replace(agent=agents))
        rollout = self._single_graph_rollout(graph)

        with patch("dgppo.env.mpe.base.render_mpe") as render_mpe:
            env.render_video(rollout, "unused.gif")

        self.assertEqual(
            render_mpe.call_args.kwargs["plot_bounds"],
            (0.0, env.area_size, 0.0, 1.5 + env.params["car_radius"]),
        )


if __name__ == "__main__":
    unittest.main()
