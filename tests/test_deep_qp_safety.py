import unittest

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

from dgppo.algo.module.deep_qp_safety import DeepQPSafetyConfig, GraphHJSafetyCritic
from dgppo.env.safety_constraint import (
    safety_constraint,
    safety_node_feature_mask,
    vmas_navigation_safety_constraint,
)
from dgppo.env.lidar_env.lidar_target import LidarTarget
from dgppo.env.vmas.vmas_navigation import VMASNavigation, VMASNavigationState
from dgppo.env.vmas.vmas_navigation_obs import (
    VMASNavigationObs,
    VMASNavigationObsState,
)
from dgppo.trainer.data import SafetyBatch
from dgppo.trainer.safety_buffer import SafetyReplayBuffer


def _env_and_graph():
    params = VMASNavigation.PARAMS.copy()
    env = VMASNavigation(num_agents=2, max_step=4, params=params)
    positions = jnp.array([[0.20, 0.20], [0.50, 0.20]])
    velocities = jnp.zeros((2, 2))
    goals = jnp.array([[0.80, 0.80], [-0.80, -0.80]])
    graph = env.get_graph(VMASNavigationState(positions, velocities, goals))
    return env, graph


class SafetyConstraintTest(unittest.TestCase):
    def test_lidar_constraint_uses_only_graph_observation(self):
        params = LidarTarget.PARAMS.copy()
        params["n_obs"] = 1
        params["n_rays"] = 8
        env = LidarTarget(num_agents=2, max_step=4, params=params)
        graph = env.reset(jr.PRNGKey(4))._replace(env_states=None)
        constraint = safety_constraint(env, graph, braking_accel=None)
        self.assertEqual(constraint.shape, (env.num_agents,))
        self.assertTrue(np.isfinite(np.asarray(constraint)).all())

    def test_continuous_agent_clearance(self):
        env, graph = _env_and_graph()
        constraint = vmas_navigation_safety_constraint(
            env, graph, agent_margin=0.02, obstacle_margin=0.02,
            braking_accel=None,
        )
        np.testing.assert_allclose(constraint, np.array([0.08, 0.08]), atol=1e-6)

    def test_obstacle_clearance_uses_graph_state(self):
        params = VMASNavigationObs.PARAMS.copy()
        params["n_obs"] = 1
        env = VMASNavigationObs(num_agents=2, max_step=4, params=params)
        state = VMASNavigationObsState(
            a_pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]),
            a_vel=jnp.zeros((2, 2)),
            goal_pos=jnp.array([[0.8, 0.8], [-0.8, -0.8]]),
            o_pos=jnp.array([[0.25, 0.0]]),
        )
        graph = env.get_graph(state)._replace(env_states=None)
        constraint = safety_constraint(env, graph, braking_accel=None)
        np.testing.assert_allclose(constraint, np.array([0.03, 0.33]), atol=1e-6)


class GraphHJSafetyCriticTest(unittest.TestCase):
    def setUp(self):
        self.env, self.graph = _env_and_graph()
        self.config = DeepQPSafetyConfig(
            hidden_dim=16,
            hidden_layers=1,
            gnn_out_dim=8,
            dt=self.env.dt,
            lambda_decay_steps=10,
        )
        lower, upper = self.env.action_lim()
        self.critic = GraphHJSafetyCritic(
            self.env.action_dim,
            self.env.num_agents,
            lower,
            upper,
            self.config,
            node_feature_mask=safety_node_feature_mask(self.env),
        )
        self.state = self.critic.initialize(jr.PRNGKey(0), self.graph)

    def test_joint_direction_coefficients_are_local_and_goal_invariant(self):
        constraint = safety_constraint(self.env, self.graph, braking_accel=None)
        certificate = self.critic.certify(
            self.state.target_params, self.graph, constraint,
            self.config.lambda_init,
        )
        self.assertEqual(
            certificate.coefficient.shape,
            (self.env.num_agents, self.env.num_agents, self.env.action_dim),
        )
        self.assertEqual(certificate.action_mask.shape, (2, 2))
        np.testing.assert_array_equal(np.diag(certificate.action_mask), np.ones(2))
        np.testing.assert_allclose(
            np.asarray(certificate.coefficient)[~np.asarray(certificate.action_mask)],
            0.0,
            atol=1e-7,
        )

        moved_goals = self.graph._replace(
            nodes=self.graph.nodes.at[:2, 4:6].set(
                jnp.array([[0.1, 1.3], [1.4, 0.1]])
            )
        )
        moved = self.critic.certify(
            self.state.target_params, moved_goals, constraint,
            self.config.lambda_init,
        )
        np.testing.assert_allclose(certificate.value, moved.value, atol=1e-6)
        np.testing.assert_allclose(
            certificate.coefficient, moved.coefficient, atol=1e-6
        )

    def test_hj_update_and_joint_action_residual_are_finite(self):
        action = jnp.array([[0.2, -0.1], [-0.3, 0.1]])
        next_graph, _, _, done, _ = self.env.step(self.graph, action)
        constraint = safety_constraint(self.env, self.graph, braking_accel=None)
        next_constraint = safety_constraint(self.env, next_graph, braking_accel=None)
        batch = SafetyBatch(
            graph=jtu.tree_map(
                lambda x: x[None], self.graph._replace(env_states=None)
            ),
            actions=action[None],
            constraints=constraint[None],
            next_graph=jtu.tree_map(
                lambda x: x[None], next_graph._replace(env_states=None)
            ),
            next_constraints=next_constraint[None],
            dones=done[None],
        )
        new_state, info = self.critic.update(self.state, batch)
        self.assertEqual(int(new_state.online.step), 1)
        self.assertTrue(np.isfinite(np.asarray(info["safety/loss"])))

        certificate = self.critic.certify(
            new_state.target_params, self.graph, constraint, info["safety/lambda"]
        )
        residual = self.critic.cbf_residual(certificate, action, alpha=1.0)
        self.assertEqual(residual.shape, (self.env.num_agents,))
        self.assertTrue(np.isfinite(np.asarray(residual)).all())

        replay = SafetyReplayBuffer(size=3, seed=0)
        for _ in range(4):
            replay.append(batch)
        self.assertEqual(replay.length, 3)
        self.assertEqual(
            replay.sample(2).actions.shape,
            (2, self.env.num_agents, self.env.action_dim),
        )


if __name__ == "__main__":
    unittest.main()
