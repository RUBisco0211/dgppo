import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from dgppo.env import make_env
from dgppo.trainer.gcbf_buffer import GCBFReplayBuffer
from train_gcbf import (
    GCBFCertificate,
    _load_checkpoint,
    _make_collector,
    _save_checkpoint,
    parse_args,
)


class GCBFOnlyTrainingTest(unittest.TestCase):
    def test_defaults_follow_original_gcbfplus_entrypoint(self):
        args = parse_args([])
        self.assertEqual(args.steps, 1000)
        self.assertEqual(args.gnn_layers, 1)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.buffer_size, 512)
        self.assertEqual(args.horizon, 32)
        self.assertEqual(args.inner_epoch, 8)
        self.assertEqual(args.n_env_train, 16)
        self.assertEqual(args.n_env_test, 32)
        self.assertAlmostEqual(args.lr_cbf, 3e-5)
        self.assertAlmostEqual(args.alpha, 1.0)
        self.assertAlmostEqual(args.eps, 0.02)
        self.assertAlmostEqual(args.loss_h_dot_coef, 0.01)

    def test_update_changes_only_a_cbf_model(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=2)
        certificate = GCBFCertificate(env, seed=0)
        collector = _make_collector(
            env,
            n_env=1,
            rollout_steps=2,
            horizon=32,
            exploration_probability=0.5,
            exploration_scale=0.5,
        )
        batch = collector(jr.PRNGKey(1))
        before = certificate.state.params
        certificate.state, info = certificate.update(certificate.state, batch)

        change = max(
            float(jnp.max(jnp.abs(old - new)))
            for old, new in zip(
                jax.tree.leaves(before), jax.tree.leaves(certificate.state.params)
            )
        )
        self.assertGreater(change, 0.0)
        self.assertFalse(hasattr(certificate, "actor"))
        self.assertTrue(bool(jnp.isfinite(info["loss/total"])))
        self.assertEqual(batch.safe_mask.shape, (2, 2))

    def test_collector_supports_every_lidar_environment(self):
        cases = (
            ("LidarTarget", 2),
            ("LidarSpread", 3),
            ("LidarLine", 3),
            ("LidarBicycleTarget", 2),
        )
        for index, (name, n_agents) in enumerate(cases):
            with self.subTest(name=name):
                env = make_env(name, n_agents, num_obs=1, max_step=1)
                collector = _make_collector(env, 1, 1, 32, 0.5, 0.5)
                batch = collector(jr.PRNGKey(index))
                self.assertEqual(batch.safe_mask.shape, (1, n_agents))
                self.assertTrue(bool(jnp.isfinite(batch.graph.edges).all()))
                self.assertTrue(bool(jnp.isfinite(batch.next_graph.edges).all()))

    def test_checkpoint_restores_optimizer_replay_and_iteration(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=2)
        certificate = GCBFCertificate(env, seed=0)
        collector = _make_collector(env, 1, 2, 32, 0.5, 0.5)
        batch = collector(jr.PRNGKey(2))
        replay = GCBFReplayBuffer(8, seed=1)
        replay.append(batch)
        certificate.state, _ = certificate.update(certificate.state, batch)
        metadata = {
            "env": "LidarTarget",
            "num_agents": 2,
            "obs": 0,
            "n_rays": 32,
            "gnn_layers": 1,
            "seed": 0,
        }

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _save_checkpoint(
                run_dir, certificate, replay, jr.PRNGKey(9), 3, metadata
            )
            restored = GCBFCertificate(env, seed=7)
            restored_replay = GCBFReplayBuffer(8, seed=8)
            key, iteration = _load_checkpoint(
                run_dir / "models" / "latest",
                restored,
                restored_replay,
                metadata,
            )

            self.assertEqual(iteration, 3)
            self.assertTrue(bool(jnp.array_equal(key, jr.PRNGKey(9))))
            self.assertEqual(restored_replay.length, replay.length)
            self.assertEqual(restored.state.step, certificate.state.step)
            for expected, actual in zip(
                jax.tree.leaves(certificate.state.params),
                jax.tree.leaves(restored.state.params),
            ):
                self.assertTrue(bool(jnp.array_equal(expected, actual)))


if __name__ == "__main__":
    unittest.main()
