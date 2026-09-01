import unittest

import jax.numpy as jnp
import numpy as np

import dgcbf_visualize
from dgcbf_visualize import _parse_ego_agents, _safe_value


class GCBFVisualizeTest(unittest.TestCase):
    def test_safe_value_uses_dgppo_unsafe_positive_convention(self):
        raw_vh = jnp.array([0.2, -0.3])
        self.assertAlmostEqual(float(_safe_value(raw_vh, "worst")), -0.2)
        self.assertAlmostEqual(float(_safe_value(raw_vh, "agent")), -0.2)
        self.assertAlmostEqual(float(_safe_value(raw_vh, "obstacle")), 0.3)

    def test_all_ego_agents_follow_target_team_size(self):
        np.testing.assert_array_equal(
            _parse_ego_agents("all", 5), np.arange(5)
        )
        self.assertEqual(_parse_ego_agents("3,1,3", 5), [3, 1])

    def test_grid_evaluation_is_chunked_to_bound_peak_memory(self):
        points = jnp.arange(103 * 2, dtype=jnp.float32).reshape(103, 2)
        observed_batch_sizes = []

        def evaluator(batch, offset):
            observed_batch_sizes.append(batch.shape[0])
            if batch.shape[0] > 32:
                raise MemoryError("simulated accelerator memory limit")
            return batch + offset

        evaluated = dgcbf_visualize._evaluate_grid_in_chunks(
            evaluator,
            points,
            jnp.array(1.0),
            batch_size=32,
        )
        self.assertEqual(observed_batch_sizes, [32, 32, 32, 32])
        np.testing.assert_array_equal(
            evaluated, np.asarray(points + 1.0)
        )


if __name__ == "__main__":
    unittest.main()
