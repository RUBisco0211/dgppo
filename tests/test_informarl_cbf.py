"""PPO probability bookkeeping, QP supervision and optimizer integration."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dgppo.algo.informarl import InforMARL
from dgppo.algo.informarl_cbf import InforMARLCBF, cbf_action_imitation_loss
from dgppo.env import make_env
from dgppo.trainer.data import CBFRollout, Rollout
from dgppo.utils.utils import tree_index


@pytest.fixture(scope="module")
def algorithm():
    env = make_env("MPETarget", 3, num_obs=3, max_step=8)
    return InforMARLCBF(env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
                        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=3,
                        actor_gnn_layers=1, Vl_gnn_layers=1, batch_size=16, rnn_step=8,
                        seed=7, cbf_action_loss_coef=1.0)


def test_supervision_is_masked_and_stops_target_gradients():
    predicted = jnp.zeros((2, 2))
    target = jnp.array([[1.0, -1.0], [10.0, 10.0]])
    mask = jnp.array([True, False])
    assert float(cbf_action_imitation_loss(predicted, target, mask)) == 1.0
    np.testing.assert_allclose(jax.grad(cbf_action_imitation_loss, argnums=0)(predicted, target, mask),
                               [[-1, 1], [0, 0]])
    np.testing.assert_array_equal(jax.grad(cbf_action_imitation_loss, argnums=1)(predicted, target, mask),
                                  np.zeros((2, 2)))
    assert float(cbf_action_imitation_loss(predicted, target, jnp.zeros(2, bool))) == 0


def test_training_retains_nominal_density_but_executes_qp(algorithm):
    rollout = algorithm.collect(algorithm.params, jax.random.split(jax.random.PRNGKey(10), 2))
    assert isinstance(rollout, CBFRollout)
    assert rollout.actions.shape == rollout.executed_actions.shape == (2, 8, 3, 2)
    assert np.asarray(rollout.intervention_mask).any()
    np.testing.assert_allclose(rollout.executed_actions, algorithm.last_executed_actions)
    for env_id, t in [(0, 0), (0, 3), (1, 6)]:
        graph = tree_index(tree_index(rollout.graph, env_id), t)
        expected_next, expected_reward, *_ = algorithm._env.step(graph, rollout.executed_actions[env_id, t])
        actual_next = tree_index(tree_index(rollout.next_graph, env_id), t)
        np.testing.assert_allclose(expected_next.states, actual_next.states, atol=1e-6)
        np.testing.assert_allclose(expected_reward, rollout.rewards[env_id, t], atol=1e-6)
        log_pi, *_ = algorithm.policy.eval_action(
            algorithm.params["policy"], graph, rollout.actions[env_id, t],
            rollout.rnn_states[env_id, t], jax.random.PRNGKey(0))
        np.testing.assert_allclose(log_pi, rollout.log_pis[env_id, t], atol=3e-5)


def test_qp_target_changes_actor_gradient(algorithm):
    rollout = algorithm.collect(algorithm.params, jax.random.split(jax.random.PRNGKey(11), 2))
    chunks = jnp.arange(8)[None]
    graphs = jax.tree_util.tree_map(lambda x: x[:, chunks], rollout.graph)
    states = rollout.rnn_states[:, chunks]
    # Force supervision to isolate its gradient from the clipped PPO objective.
    rollout = rollout._replace(intervention_mask=jnp.ones_like(rollout.intervention_mask))
    def loss(params):
        return algorithm.actor_auxiliary_loss(params, rollout, chunks, graphs, states)[0]
    value, grad = jax.value_and_grad(loss)(algorithm.params["policy"])
    assert np.isfinite(float(value)) and float(value) > 0
    assert sum(float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(grad)) > 0


def test_update_save_load_and_eval_isolation(algorithm, tmp_path):
    rollout = algorithm.collect(algorithm.params, jax.random.split(jax.random.PRNGKey(12), 2))
    info = algorithm.update(rollout, 0)
    for key in ("policy/loss", "policy/ppo_loss", "Vl/loss", "policy/cbf_action_loss", "policy/cbf_action_loss_weighted"):
        assert np.isfinite(float(info[key]))
    np.testing.assert_allclose(info["policy/loss"],
                               info["policy/ppo_loss"] + info["policy/cbf_action_loss_weighted"],
                               atol=1e-6)
    assert int(algorithm.policy_train_state.step) == 1
    assert int(algorithm.Vl_train_state.step) == 1
    algorithm.save(str(tmp_path), "latest")
    assert (tmp_path / "latest" / "algo_training_state.pkl").is_file()
    key = np.asarray(algorithm.key).copy()
    before = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(algorithm.params)]
    eval_rollout = algorithm.eval_rollouts_host(algorithm._env, algorithm.params,
                                               jax.random.split(jax.random.PRNGKey(100), 1))
    assert isinstance(eval_rollout, Rollout) and eval_rollout.log_pis is None
    assert "eval/task/reach_ever_rate" in algorithm.diagnostic_metrics(eval_rollout)
    np.testing.assert_array_equal(algorithm.key, key)
    for x, y in zip(before, jax.tree_util.tree_leaves(algorithm.params)):
        np.testing.assert_array_equal(x, y)
    algorithm.load(str(tmp_path), "latest")
    assert int(algorithm.policy_train_state.step) == 1
    np.testing.assert_array_equal(algorithm.key, key)


def test_plain_informarl_has_no_auxiliary_objective():
    assert not hasattr(InforMARL, "actor_auxiliary_loss")
