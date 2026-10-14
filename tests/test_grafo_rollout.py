import jax
import jax.numpy as jnp
import numpy as np

from dgppo.env.mpe.grafo import GRAFO, GRAFO_3
from dgppo.trainer.utils import episodic_rollout


def test_fixed_horizon_crosses_episode_boundaries():
    env = GRAFO(GRAFO_3, max_step=50)

    def stationary_actor(graph, carry, key):
        del graph, key
        return jnp.zeros((env.num_agents, env.action_dim)), jnp.zeros(
            (env.num_agents,)), carry + 1

    rollout = jax.jit(lambda key: episodic_rollout(
        env, stationary_actor, jnp.array(0), key, horizon=128))(
            jax.random.PRNGKey(7))
    assert rollout.rewards.shape == (128,)
    np.testing.assert_array_equal(np.flatnonzero(np.asarray(rollout.dones)),
                                  [49, 99])
    np.testing.assert_array_equal(np.asarray(rollout.rnn_states)[[0, 50, 100]],
                                  [0, 0, 0])
    assert not np.array_equal(np.asarray(rollout.graph.states[50]),
                              np.asarray(rollout.next_graph.states[49]))
