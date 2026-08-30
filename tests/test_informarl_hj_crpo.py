import unittest

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.algo.informarl_deep_qp import (
    InforMARLHJCRPO,
    aggregate_owner_violation,
)
from dgppo.env.vmas.vmas_navigation import VMASNavigation
from dgppo.trainer.data import Rollout


class InforMARLHJCRPOTest(unittest.TestCase):
    def test_violation_is_attributed_through_local_action_edges(self):
        violation = jnp.array([[1.0, 2.0], [3.0, 0.0]])
        action_mask = jnp.array([
            [[True, True], [False, True]],
            [[True, True], [False, True]],
        ])
        owner_violation = aggregate_owner_violation(violation, action_mask)
        np.testing.assert_allclose(owner_violation, np.array([[1.0, 2.0], [3.0, 3.0]]))

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


if __name__ == "__main__":
    unittest.main()
