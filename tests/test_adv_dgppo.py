import tempfile
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.algo import make_algo
from dgppo.algo.adv_dgppo import (
    compute_adv_dgcbf_target,
    mix_adv_dgcbf_advantages,
)
from dgppo.env import make_env


class AdversarialDGPPOMathTest(unittest.TestCase):
    def test_discounted_target_and_undiscounted_reference_use_safe_positive_minimum(self):
        constraint = jnp.array([0.6, -0.2, 0.3])
        next_value = jnp.array([0.2, 0.4, -0.5])

        discounted, undiscounted = compute_adv_dgcbf_target(
            constraint, next_value, safety_gamma=0.5
        )

        np.testing.assert_allclose(discounted, np.array([0.4, -0.2, -0.1]))
        np.testing.assert_allclose(undiscounted, np.array([0.2, -0.2, -0.5]))

    def test_robust_violation_replaces_task_advantage_only_when_needed(self):
        task_advantage = jnp.array([2.0, 4.0, 3.0])
        value = jnp.array([0.5, 0.5, -0.2])
        task_action_value = jnp.array([0.46, 0.2, -0.1])

        mixed, violation, safe = mix_adv_dgcbf_advantages(
            task_advantage,
            value,
            task_action_value,
            cbf_weight=2.0,
            cbf_kappa=0.2,
            cbf_eps=0.0,
        )

        np.testing.assert_allclose(violation, np.array([0.0, 0.2, 0.1]))
        np.testing.assert_allclose(mixed, np.array([2.0, -0.4, -0.2]))
        np.testing.assert_array_equal(safe, np.array([True, False, False]))


class AdversarialDGPPOIntegrationTest(unittest.TestCase):
    @staticmethod
    def _make_algo(env, **kwargs):
        config = dict(
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            use_rnn=False,
            batch_size=env.max_episode_steps,
            rnn_step=env.max_episode_steps,
            adv_gnn_layers=1,
            adv_hidden_dim=16,
            adv_inner_steps=1,
            adv_batch_size=1,
            adv_task_sample_ratio=1.0,
            adv_actor_egos_per_sample=1,
        )
        config.update(kwargs)
        return make_algo("adversarial_dgppo", **config)

    def test_factory_rollout_update_and_metric_namespace(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=2)
        algo = self._make_algo(env)
        self.assertEqual(algo.config["adv_batch_size"], 1)
        self.assertEqual(algo.config["adv_task_sample_ratio"], 1.0)
        self.assertEqual(algo.config["adv_actor_egos_per_sample"], 1)
        rollout = algo.collect(algo.params, jr.split(jr.PRNGKey(7), 1))
        safety_before = algo.adv_q_train_state.params

        info = algo.update(rollout, step=0)

        safety_change = max(
            float(jnp.max(jnp.abs(before - after)))
            for before, after in zip(
                jax.tree.leaves(safety_before),
                jax.tree.leaves(algo.adv_q_train_state.params),
            )
        )
        self.assertGreater(safety_change, 0.0)
        self.assertIn("Vl/loss", info)
        self.assertIn("policy/loss", info)
        self.assertIn("adv_dgcbf/q/loss", info)
        self.assertIn("adv_dgcbf/vh/loss", info)
        self.assertIn("adv_dgcbf/actor/safe_ratio", info)
        self.assertIn("adv_dgcbf/data/task_action_fraction", info)
        self.assertEqual(
            int(info["adv_dgcbf/data/adversarial_rollout_samples"]), 2
        )
        self.assertEqual(int(info["adv_dgcbf/data/adversarial_samples"]), 1)
        self.assertEqual(int(info["adv_dgcbf/data/task_action_samples"]), 1)
        self.assertEqual(int(info["adv_dgcbf/data/actor_ego_evaluations"]), 1)
        self.assertTrue(
            all(
                not key.startswith("Vh/")
                for key in info
                if key not in {"Vl/loss", "policy/loss"}
            )
        )
        for value in info.values():
            self.assertTrue(np.isfinite(np.asarray(value)).all())

        target_after_first_update = jax.tree.map(
            jnp.copy, algo.adv_q_target_params
        )
        rollout = algo.collect(algo.params, jr.split(jr.PRNGKey(8), 1))
        second_info = algo.update(rollout, step=1)
        self.assertEqual(int(algo.adv_q_train_state.step), 2)
        self.assertTrue(
            any(
                bool(jnp.any(left != right))
                for left, right in zip(
                    jax.tree.leaves(target_after_first_update),
                    jax.tree.leaves(algo.adv_q_target_params),
                )
            )
        )
        self.assertTrue(
            all(np.isfinite(np.asarray(value)).all() for value in second_info.values())
        )

    def test_mpe_rollout_and_update_are_supported(self):
        env = make_env("MPESpread", 2, num_obs=1, max_step=1)
        algo = self._make_algo(env)
        rollout = algo.collect(algo.params, jr.split(jr.PRNGKey(17), 1))

        info = algo.update(rollout, step=0)

        self.assertEqual(rollout.actions.shape, (1, 1, 2, 2))
        self.assertTrue(
            all(np.isfinite(np.asarray(value)).all() for value in info.values())
        )

    def test_checkpoint_restores_all_adversarial_training_state(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=1)
        source = self._make_algo(env)
        with tempfile.TemporaryDirectory() as directory:
            source.save(directory, "latest")
            restored = self._make_algo(env, seed=7)
            restored.load(directory, "latest")

            for left, right in zip(
                jax.tree.leaves(source.params), jax.tree.leaves(restored.params)
            ):
                np.testing.assert_array_equal(left, right)
            for left, right in zip(
                jax.tree.leaves(source.adv_q_target_params),
                jax.tree.leaves(restored.adv_q_target_params),
            ):
                np.testing.assert_array_equal(left, right)

    def test_actor_ego_budget_controls_public_update_work(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=1)
        algo = self._make_algo(env, adv_actor_egos_per_sample=2)
        rollout = algo.collect(algo.params, jr.split(jr.PRNGKey(9), 1))

        info = algo.update(rollout, step=0)

        self.assertEqual(
            int(info["adv_dgcbf/data/actor_ego_evaluations"]), 2
        )
        self.assertEqual(float(info["adv_dgcbf/data/actor_ego_fraction"]), 1.0)


if __name__ == "__main__":
    unittest.main()
