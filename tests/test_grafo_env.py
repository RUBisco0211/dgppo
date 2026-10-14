import jax
import jax.numpy as jnp
import numpy as np

from dgppo.env.mpe.grafo import (
    GRAFO, GRAFO_3, GRAFO_4_1, GRAFO_4_2, GRAFO_TASKS, GRAFO_SEQUENCES,
)


def test_task_graph_and_hidden_targets():
    env = GRAFO(GRAFO_3)
    graph = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert graph.nodes.shape[0] == env.num_agents + 1 + 1  # padding node
    assert graph.env_states.goal.shape == (1, 4)
    assert graph.env_states.obs.shape == (0, 4)
    assert env.max_episode_steps == 50
    assert env.edge_dim == 5


def test_equivalent_agents_can_exchange_targets():
    env = GRAFO(GRAFO_3)
    graph = env.reset(jax.random.PRNGKey(1))
    landmark = graph.env_states.goal[0, :2]
    targets = landmark + env._offsets
    positions = targets[jnp.array([2, 1, 0])]
    agents = graph.env_states.agent.at[:, :2].set(positions)
    graph = env.get_graph(graph.env_states._replace(agent=agents))
    np.testing.assert_allclose(env._assigned_distances(graph), 0, atol=1e-6)
    assert bool(env.success(graph))


def test_edge_feature_switch_and_jitted_step():
    for task in (GRAFO_4_1, GRAFO_4_2):
        env = GRAFO(task)
        graph = jax.jit(env.reset)(jax.random.PRNGKey(2))
        next_graph, reward, cost, done, _ = jax.jit(env.step)(
            graph, jnp.zeros((env.num_agents, env.action_dim)))
        assert next_graph.edges.shape[-1] == env.edge_dim
        assert jnp.isfinite(reward)
        assert cost.shape == (env.num_agents, env.n_cost)
        assert not bool(done)


def test_paper_sequences_share_model_dimensions():
    for sequence in GRAFO_SEQUENCES.values():
        for name in sequence:
            task = GRAFO_TASKS[name]
            env = GRAFO(task)
            assert len(task.offsets) == env.num_agents
            assert env.node_dim == 7
            assert env.edge_dim == 5


def test_reaching_agents_stay_at_assigned_targets():
    env = GRAFO(GRAFO_3)
    graph = env.reset(jax.random.PRNGKey(11))
    landmark = graph.env_states.goal[0, :2]
    targets = landmark + env._offsets
    agents = graph.env_states.agent.at[:, :2].set(targets)
    graph = env.get_graph(graph.env_states._replace(agent=agents))
    action = jnp.ones((env.num_agents, env.action_dim))
    reached_graph, reward, _, done, _ = env.step(graph, action)
    assert bool(done)
    assert bool(jnp.all(reached_graph.env_states.reached))
    np.testing.assert_allclose(reached_graph.env_states.agent[:, :2], targets,
                               atol=1e-6)
    again, second_reward, _, second_done, _ = env.step(reached_graph, action)
    np.testing.assert_allclose(again.env_states.agent, reached_graph.env_states.agent,
                               atol=1e-6)
    assert bool(second_done)
    assert second_reward < reward


def test_policy_graph_is_translation_invariant():
    env = GRAFO(GRAFO_3)
    graph = env.reset(jax.random.PRNGKey(13))
    shift = jnp.array([2.0, -3.0])
    state = graph.env_states
    shifted = env.get_graph(state._replace(
        agent=state.agent.at[:, :2].add(shift),
        goal=state.goal.at[:, :2].add(shift)))
    np.testing.assert_allclose(graph.nodes, shifted.nodes, atol=1e-6)
    np.testing.assert_allclose(graph.edges, shifted.edges, atol=1e-6)
    np.testing.assert_allclose(env._assigned_distances(graph),
                               env._assigned_distances(shifted), atol=1e-6)


def test_discrete_actions_match_cited_formation_controls():
    env = GRAFO(GRAFO_3, action_encoding="discrete")
    assert env.action_dim == 5
    one_hot = jnp.eye(5, dtype=jnp.float32)
    actions = one_hot[jnp.asarray([1, 2, 4])]
    np.testing.assert_array_equal(env.action_to_force(actions),
                                  np.asarray([[-1., 0.], [1., 0.], [0., 1.]]))
    graph = env.reset(jax.random.PRNGKey(42))
    next_graph, reward, _, _, _ = jax.jit(env.step)(graph, actions)
    assert next_graph.env_states.agent.shape == (3, 4)
    assert bool(jnp.isfinite(reward))
