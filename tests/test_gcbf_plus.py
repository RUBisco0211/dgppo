import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from dgppo.algo.gcbf_plus import GCBFPlus
from dgppo.algo.gcbf_plus_adapter import make_gcbf_plus_env_adapter
from dgppo.env import make_env
from dgppo.trainer.data import GCBFTransitionBatch
from dgppo.trainer.gcbf_buffer import GCBFReplayBuffer


def _make_algo(env, **kwargs):
    return GCBFPlus(
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gcbf_batch_size=2,
        gcbf_buffer_size=8,
        gcbf_inner_epoch=1,
        gcbf_qp_chunk_size=1,
        **kwargs,
    )


class GCBFPlusTest(unittest.TestCase):
    def test_replay_prioritization_ignores_unwritten_capacity(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=1)
        graph = env.reset(jr.PRNGKey(0))
        batch = GCBFTransitionBatch(
            graph=jax.tree.map(lambda value: jnp.repeat(value[None], 2, axis=0), graph),
            next_graph=jax.tree.map(
                lambda value: jnp.repeat(value[None], 2, axis=0), graph
            ),
            safe_mask=jnp.array([[True, True], [True, True]]),
            unsafe_mask=jnp.array([[True, False], [False, False]]),
        )
        replay = GCBFReplayBuffer(size=8)
        replay.append(batch)

        for _ in range(8):
            sampled = replay.sample(2, unsafe_fraction=1.0)
            self.assertTrue(bool(jnp.isfinite(sampled.graph.edges).all()))
            self.assertTrue(bool(sampled.unsafe_mask[:, 0].all()))

    def test_adapters_cover_lidar_and_vmas_families(self):
        cases = (
            ("LidarTarget", 2, 1),
            ("LidarSpread", 3, 1),
            ("LidarLine", 3, 1),
            ("LidarBicycleTarget", 2, 1),
            ("VMASNavigation", 2, 0),
            ("VMASNavigationObs", 2, 1),
            ("VMASReverseTransport", 3, 0),
            ("VMASWheel", 3, 0),
        )
        for name, n_agents, n_obs in cases:
            with self.subTest(name=name):
                env = make_env(name, n_agents, num_obs=n_obs, max_step=1)
                graph = env.reset(jr.PRNGKey(0))
                adapter = make_gcbf_plus_env_adapter(env)
                action = adapter.nominal_action(graph)
                next_graph = adapter.forward_graph(graph, action)
                self.assertEqual(action.shape, (n_agents, env.action_dim))
                self.assertEqual(adapter.unsafe_mask(graph).shape, (n_agents,))
                self.assertEqual(next_graph.nodes.shape, graph.nodes.shape)

    def test_horizon_safe_mask_matches_upstream_definition(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=2)
        algo = _make_algo(env)
        unsafe = jnp.array(
            [[[False, False], [False, True], [False, False], [True, False]]]
        )
        safe = algo.safe_mask(unsafe)
        expected = jnp.array(
            [[[True, True], [False, False], [False, True], [False, True]]]
        )
        self.assertTrue(bool(jnp.array_equal(safe, expected)))

    def test_lidar_collect_qp_and_joint_update(self):
        env = make_env("LidarTarget", 2, num_obs=1, max_step=2)
        algo = _make_algo(env)
        rollout = algo.collect(algo.params, jr.split(jr.PRNGKey(3), 1))
        actor_before = algo.actor_train_state.params
        cbf_before = algo.cbf_train_state.params
        info = algo.update(rollout, 0)

        actor_change = max(
            float(jnp.max(jnp.abs(before - after)))
            for before, after in zip(
                jax.tree.leaves(actor_before),
                jax.tree.leaves(algo.actor_train_state.params),
            )
        )
        cbf_change = max(
            float(jnp.max(jnp.abs(before - after)))
            for before, after in zip(
                jax.tree.leaves(cbf_before),
                jax.tree.leaves(algo.cbf_train_state.params),
            )
        )
        self.assertGreater(actor_change, 0.0)
        self.assertGreater(cbf_change, 0.0)
        self.assertTrue(bool(jnp.isfinite(info["loss/total"])))

    def test_vmas_qp_is_finite(self):
        env = make_env("VMASNavigation", 2, num_obs=0, max_step=1)
        algo = _make_algo(env)
        graph = env.reset(jr.PRNGKey(4))
        action = algo.get_qp_action(graph, algo.cbf_target_params)
        self.assertEqual(action.shape, (2, 2))
        self.assertTrue(bool(jnp.isfinite(action).all()))

    def test_checkpoint_round_trip(self):
        env = make_env("LidarTarget", 2, num_obs=0, max_step=1)
        source = _make_algo(env)
        with tempfile.TemporaryDirectory() as directory:
            source.save(directory, 7)
            restored = _make_algo(env, seed=9)
            restored.load(directory, 7)
            for expected, actual in zip(
                jax.tree.leaves(source.params), jax.tree.leaves(restored.params)
            ):
                self.assertTrue(bool(jnp.array_equal(expected, actual)))
            self.assertTrue((Path(directory) / "7" / "cbf.pkl").is_file())
            self.assertEqual(source.replay.length, restored.replay.length)
            self.assertTrue((Path(directory) / "7" / "replay.pkl").is_file())


if __name__ == "__main__":
    unittest.main()
