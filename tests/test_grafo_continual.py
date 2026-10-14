"""Small checks of the GRAFO task identity and CL regularizers."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from dgppo.algo.grafo_continual import (
    GrafoContinualInforMARL, _actor_mask, output_head, with_output_head,
)
from train_grafo_continual import make_environment


def test_task_identity_is_one_hot_and_shared_shape():
    tasks = ("GRAFO-3", "GRAFO-5")
    for index, name in enumerate(tasks):
        env = make_environment(name, tasks)
        graph = env.reset(jax.random.PRNGKey(index))
        assert env.node_dim == 7 + len(tasks)
        expected = np.eye(len(tasks))[index]
        np.testing.assert_array_equal(
            np.asarray(graph.nodes[:env.num_agents, 7:]),
            np.repeat(expected[None], env.num_agents, axis=0))


def test_grafo_action_module_has_two_hidden_layers_and_output_only_task_head():
    env = make_environment("GRAFO-3")
    algo = GrafoContinualInforMARL(
        env, env.node_dim, env.edge_dim, env.state_dim,
        env.action_dim, env.num_agents, gnn_arch="unimp", hidden_dim=128)
    params = algo.policy_train_state.params
    assert "ScaleHid" not in params["params"]
    assert set(output_head(params)) == {"OutputDenseMean", "OutputDenseStdTrans"}
    head_layers = params["params"]["PolicyNet_0"]["PolicyGNNHead"]
    assert sum(name.startswith("Dense_") for name in head_layers) == 2
    restored = with_output_head(params, output_head(params))
    np.testing.assert_array_equal(
        restored["params"]["OutputDenseMean"]["kernel"],
        params["params"]["OutputDenseMean"]["kernel"])


def test_discrete_grafo_policy_uses_one_hot_actions():
    env = make_environment("GRAFO-3", action_encoding="discrete")
    algo = GrafoContinualInforMARL(
        env, env.node_dim, env.edge_dim, env.state_dim,
        env.action_dim, env.num_agents, gnn_arch="unimp", hidden_dim=128)
    graph = env.reset(jax.random.PRNGKey(44))
    action, log_prob, _ = algo.policy.sample_action(
        algo.policy_train_state.params, graph, algo.init_rnn_state,
        jax.random.PRNGKey(45))
    assert action.shape == (env.num_agents, 5)
    np.testing.assert_allclose(np.asarray(action.sum(axis=-1)), 1.)
    assert log_prob.shape == (env.num_agents,)
    assert set(output_head(algo.policy_train_state.params)) == {"OutputDenseLogits"}


def test_ewc_mas_l2_and_frog_masks_are_active():
    env = make_environment("GRAFO-3")
    kwargs = dict(gnn_arch="unimp", hidden_dim=128,
                  actor_gnn_layers=4, Vl_gnn_layers=4, batch_size=128)
    base = GrafoContinualInforMARL(
        env, env.node_dim, env.edge_dim, env.state_dim,
        env.action_dim, env.num_agents, **kwargs)
    anchor = base.policy_train_state.params
    changed = jtu.tree_map(lambda x: x + 0.01, anchor)
    rollout = base.collect_importance(
        jax.random.split(jax.random.PRNGKey(23), 2), horizon=10)

    for method in ("ewc", "mas"):
        importance = base.estimate_importance(
            rollout, method=method, batch_size=10)
        values = jtu.tree_leaves(importance)
        assert all(bool(jnp.isfinite(x).all()) for x in values)
        assert sum(float(jnp.abs(x).sum()) for x in values) > 0
        model = GrafoContinualInforMARL(
            env, env.node_dim, env.edge_dim, env.state_dim,
            env.action_dim, env.num_agents, regularization=method,
            regularization_coefficient=1.0, anchors=[anchor],
            importances=[importance], **kwargs)
        loss, _ = model.actor_auxiliary_loss(changed, None, None, None, None)
        assert float(loss) > 0

    l2 = GrafoContinualInforMARL(
        env, env.node_dim, env.edge_dim, env.state_dim,
        env.action_dim, env.num_agents, regularization="l2",
        regularization_coefficient=1.0, anchors=[anchor],
        importances=[None], **kwargs)
    loss, _ = l2.actor_auxiliary_loss(changed, None, None, None, None)
    assert float(loss) > 0

    frozen = _actor_mask(anchor, freeze_encoder=True)
    paths = [(jtu.keystr(path), bool(value)) for path, value in
             jtu.tree_leaves_with_path(frozen)]
    assert any("UniMPGNN" in path for path, _ in paths)
    assert all(not selected for path, selected in paths if "UniMPGNN" in path)
    assert any(selected for path, selected in paths if "PolicyGNNHead" in path)
