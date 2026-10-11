"""Eval-only CBF must leave InforMARL's training stream and objective intact."""
import jax
import numpy as np
import pytest

from dgppo.algo.informarl import InforMARL
from dgppo.algo.informarl_eval_cbf import InforMARLEvalCBF
from dgppo.env import make_env


@pytest.mark.parametrize("mode", ["qp", "certificate"])
def test_eval_only_filter_keeps_native_training(mode):
    env = make_env("MPETarget", 3, num_obs=3, max_step=8)
    kwargs = dict(env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
                  state_dim=env.state_dim, action_dim=env.action_dim, n_agents=3,
                  actor_gnn_layers=1, Vl_gnn_layers=1, batch_size=16,
                  rnn_step=8, seed=7)
    plain = InforMARL(**kwargs)
    filtered = InforMARLEvalCBF(**kwargs, pairwise_cbf_mode=mode)
    assert not hasattr(filtered, "actor_auxiliary_loss")
    keys = jax.random.split(jax.random.PRNGKey(10), 2)
    base_rollout = plain.collect(plain.params, keys)
    filtered_rollout = filtered.collect(filtered.params, keys)
    np.testing.assert_allclose(base_rollout.actions, filtered_rollout.actions)
    np.testing.assert_allclose(base_rollout.rewards, filtered_rollout.rewards)
    numpy_rng = np.random.get_state()
    base_update = plain.update(base_rollout, 0)
    np.random.set_state(numpy_rng)
    filtered_update = filtered.update(filtered_rollout, 0)
    for key in ("policy/loss", "Vl/loss", "policy/entropy"):
        np.testing.assert_allclose(base_update[key], filtered_update[key], atol=1e-6)
    assert "policy/cbf_action_loss" not in filtered_update
    rng_before = np.asarray(filtered.key).copy()
    eval_rollout = filtered.eval_rollouts_host(env, filtered.params,
                                               jax.random.split(jax.random.PRNGKey(11), 2))
    assert eval_rollout.actions.shape == (2, 8, 3, 2)
    assert "eval/task/reach_final_rate" in filtered.diagnostic_metrics(eval_rollout)
    np.testing.assert_array_equal(filtered.key, rng_before)


@pytest.mark.parametrize("mode", ["qp", "certificate"])
def test_mpe_line_eval_filter(mode):
    env = make_env("MPELine", 3, num_obs=3, max_step=8)
    algo = InforMARLEvalCBF(
        env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=3,
        actor_gnn_layers=1, Vl_gnn_layers=1, batch_size=8,
        rnn_step=8, seed=7, pairwise_cbf_mode=mode,
    )
    rollout = algo.eval_rollouts_host(env, algo.params,
                                      jax.random.split(jax.random.PRNGKey(12), 1))
    assert rollout.actions.shape == (1, 8, 3, 2)
    assert np.isfinite(np.asarray(rollout.rewards)).all()
    assert np.isfinite(algo.diagnostic_metrics(rollout)["eval/task/final_goal_distance"])
