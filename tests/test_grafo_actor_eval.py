"""The feedforward actor fast path must reproduce sequential evaluation."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from dgppo.algo.grafo_continual import GrafoContinualInforMARL
from dgppo.env.mpe.grafo import GRAFO, GRAFO_3


def test_feedforward_actor_vectorization_matches_scan():
    env = GRAFO(GRAFO_3)
    algo = GrafoContinualInforMARL(
        env, env.node_dim, env.edge_dim, env.state_dim,
        env.action_dim, env.num_agents, seed=3, batch_size=128,
        epoch_ppo=1, gnn_arch="unimp", hidden_dim=128,
        actor_gnn_layers=4, Vl_gnn_layers=4,
    )
    graph = env.reset(jax.random.PRNGKey(5))._replace(env_states=None)
    graphs = jtu.tree_map(lambda x: jnp.stack([x] * 3), graph)
    actions = jnp.zeros((3, env.num_agents, env.action_dim))
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    carry = algo.init_rnn_state
    params = algo.policy_train_state.params

    def sequential(state, data):
        one_graph, one_action, one_key = data
        logp, entropy, next_state = algo.policy.eval_action(
            params, one_graph, one_action, state, one_key)
        return next_state, (logp, entropy, state)

    old_carry, old_outputs = jax.lax.scan(
        sequential, carry, (graphs, actions, keys))
    new_outputs = algo.scan_eval_action(graphs, actions, carry, keys, params)
    for expected, actual in zip((*old_outputs, old_carry), new_outputs):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
