import tempfile
import unittest

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

from dgppo.algo.informarl_deep_qp import (
    InforMARLHJCRPO,
    aggregate_owner_violation,
    mix_hj_advantages,
)
from dgppo.env.vmas.vmas_navigation import VMASNavigation
from dgppo.env.lidar_env.lidar_target import LidarTarget
from dgppo.trainer.data import Rollout


class InforMARLHJCRPOTest(unittest.TestCase):
    def assert_tree_equal(self, left, right):
        left_leaves = jtu.tree_leaves(left)
        right_leaves = jtu.tree_leaves(right)
        self.assertEqual(len(left_leaves), len(right_leaves))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves):
            np.testing.assert_array_equal(left_leaf, right_leaf)

    def test_violation_is_attributed_through_local_action_edges(self):
        violation = jnp.array([[1.0, 2.0], [3.0, 0.0]])
        action_mask = jnp.array([
            [[True, True], [False, True]],
            [[True, True], [False, True]],
        ])
        owner_violation = aggregate_owner_violation(violation, action_mask)
        np.testing.assert_allclose(owner_violation, np.array([[1.0, 2.0], [3.0, 3.0]]))

    def test_advantage_mixing_matches_dgppo_rule(self):
        task_advantage = jnp.array([[2.0, -1.0], [4.0, 3.0]])
        owner_violation = jnp.array([[0.0, 0.5], [1.5, 0.0]])
        mixed = mix_hj_advantages(task_advantage, owner_violation, cbf_weight=2.0)
        np.testing.assert_allclose(mixed, np.array([[2.0, -1.0], [-3.0, 3.0]]))

    def test_execution_keeps_standard_rollout_semantics(self):
        env = VMASNavigation(
            num_agents=2,
            max_step=4,
            params=VMASNavigation.PARAMS.copy(),
        )
        algo = InforMARLHJCRPO(
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            use_rnn=False,
            batch_size=env.max_episode_steps,
            rnn_step=env.max_episode_steps,
            deep_qp_gnn_out_dim=8,
            deep_qp_hidden_dim=16,
            deep_qp_hidden_layers=1,
        )
        rollout = algo.collect(algo.params, jr.split(jr.PRNGKey(2), 1))
        self.assertIsInstance(rollout, Rollout)
        self.assertFalse(hasattr(rollout, "executed_actions"))
        self.assertEqual(
            rollout.actions.shape,
            (1, env.max_episode_steps, env.num_agents, env.action_dim),
        )
        info = algo.update(rollout, step=0)
        for key in (
            "Vl/loss",
            "policy/loss",
            "hj_crpo/constraint_estimate",
            "hj_crpo/safe_data",
            "hj_crpo/cbf_weight",
            "hj_crpo/violation_mean",
        ):
            self.assertIn(key, info)
            self.assertTrue(np.isfinite(np.asarray(info[key])).all())

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            algo.save(checkpoint_dir, "latest")
            restored = InforMARLHJCRPO(
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                use_rnn=False,
                batch_size=env.max_episode_steps,
                rnn_step=env.max_episode_steps,
                deep_qp_gnn_out_dim=8,
                deep_qp_hidden_dim=16,
                deep_qp_hidden_layers=1,
            )
            restored.load(checkpoint_dir, "latest")
            self.assert_tree_equal(
                algo.policy_train_state, restored.policy_train_state
            )
            self.assert_tree_equal(algo.Vl_train_state, restored.Vl_train_state)
            self.assert_tree_equal(algo.key, restored.key)
            self.assert_tree_equal(
                algo.safety_train_state, restored.safety_train_state
            )

    def test_lidar_env_rollout_and_update_are_supported(self):
        params = LidarTarget.PARAMS.copy()
        params["n_obs"] = 1
        params["n_rays"] = 8
        env = LidarTarget(num_agents=2, max_step=4, params=params)
        algo = InforMARLHJCRPO(
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            use_rnn=False,
            batch_size=env.max_episode_steps,
            rnn_step=env.max_episode_steps,
            deep_qp_gnn_out_dim=8,
            deep_qp_hidden_dim=16,
            deep_qp_hidden_layers=1,
        )
        rollout = algo.collect(algo.params, jr.split(jr.PRNGKey(5), 1))
        info = algo.update(rollout, step=0)
        self.assertEqual(rollout.actions.shape, (1, 4, 2, 2))
        self.assertTrue(np.isfinite(np.asarray(info["policy/loss"])))
        self.assertTrue(np.isfinite(np.asarray(info["hj_crpo/violation_mean"])))


if __name__ == "__main__":
    unittest.main()
