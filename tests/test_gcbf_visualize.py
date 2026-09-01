import unittest

import jax.numpy as jnp
import numpy as np

from gcbf_visualize import _parse_ego_agents, _safe_value


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


if __name__ == "__main__":
    unittest.main()
