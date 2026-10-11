from typing import NamedTuple
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from dgppo.algo.gpudrive_gnn_ippo import GPUDriveGNNIPPO
from dgppo.utils.graph import GraphsTuple
from dgppo.env.gpudrive.config import GPUDriveDGPPOConfig
from dgppo.env.gpudrive.graph import build_ippo_scene_graph


class MiniRollout(NamedTuple):
    graph: GraphsTuple
    next_graph: GraphsTuple
    actions: jax.Array
    rewards: jax.Array
    alive_mask: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    log_pis: jax.Array


def _graph(n_agents=4):
    pad = n_agents
    senders, receivers = [], []
    for recv in range(n_agents):
        for send in range(n_agents):
            if recv != send:
                senders.append(send); receivers.append(recv)
    return GraphsTuple(
        n_node=jnp.asarray(n_agents + 1), n_edge=jnp.asarray(len(senders)),
        nodes=jnp.zeros((n_agents + 1, 18 + 32 * 8), jnp.float32),
        edges=jnp.zeros((len(senders), 7), jnp.float32),
        states=jnp.zeros((n_agents + 1, 14), jnp.float32),
        receivers=jnp.asarray(receivers, jnp.int32),
        senders=jnp.asarray(senders, jnp.int32),
        node_type=jnp.asarray([0] * n_agents + [-1], jnp.int32),
        env_states=None,
    )


def test_both_action_modes_sample_and_update():
    graph = _graph()
    graph_batch = jax.tree_util.tree_map(
        lambda x: None if x is None else x[None], graph,
        is_leaf=lambda x: x is None)
    graph_rollout = jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.repeat(x[:, None], 2, axis=1),
        graph_batch, is_leaf=lambda x: x is None)
    for mode in ("discrete", "continuous"):
        algo = GPUDriveGNNIPPO(
            graph, action_mode=mode, n_agents=4, minibatch_worlds=1,
            epoch_ppo=1, road_slots=32, road_feature_dim=8)
        actions, log_pis = algo.sample_actions(graph_batch, jax.random.PRNGKey(1))
        assert actions.shape == (1, 4, 2)
        assert log_pis.shape == (1, 4)
        np.testing.assert_equal(np.isfinite(np.asarray(log_pis)).all(), True)
        rollout = MiniRollout(
            graph_rollout, graph_rollout,
            jnp.repeat(actions[:, None], 2, axis=1),
            jnp.zeros((1, 2, 4)), jnp.ones((1, 2, 4), dtype=bool),
            jnp.zeros((1, 2, 4), dtype=bool),
            jnp.zeros((1, 2, 4), dtype=bool),
            jnp.repeat(log_pis[:, None], 2, axis=1),
        )
        info = algo.update(rollout)
        assert np.isfinite(float(info["policy/loss"]))
        assert np.isfinite(float(info["Vl/loss"]))
        with tempfile.TemporaryDirectory() as directory:
            algo.save(directory, 1, {"iteration": 1})
            restored = GPUDriveGNNIPPO(
                graph, action_mode=mode, n_agents=4, minibatch_worlds=1,
                epoch_ppo=1, road_slots=32, road_feature_dim=8)
            counters = restored.load(directory, "1")
            assert counters["iteration"] == 1


def test_padded_64_slot_discrete_update_is_finite():
    config = GPUDriveDGPPOConfig(
        num_controlled=64, num_background=0, control_all_vehicles=True,
        keep_all_scene_vehicles=True, num_road_segments=200,
        road_feature_dim=14,
    )
    vehicles = jnp.zeros((64, 14), dtype=jnp.float32)
    vehicles = vehicles.at[:8, 0].set(jnp.arange(8, dtype=jnp.float32) * 5)
    vehicles = vehicles.at[:8, 8].set(40.0)
    vehicles = vehicles.at[:8, 10:13].set(jnp.array([4.5, 2.0, 1.5]))
    # Madrona/GPUDrive may expose NaNs in physical-state fields of unused
    # slots.  Those values must never survive in padding edge attributes.
    vehicles = vehicles.at[8:, :].set(jnp.nan)
    valid = jnp.arange(64) < 8
    roads = jnp.zeros((64, 200, 9), dtype=jnp.float32)
    roads = roads.at[:8, :16, 0].set(jnp.arange(1, 17, dtype=jnp.float32))
    roads = roads.at[:8, :16, 2].set(2.0)
    roads = roads.at[:8, :16, 6].set(3.0)
    graph = build_ippo_scene_graph(
        vehicles, valid, roads,
        jnp.concatenate([jnp.zeros(8), jnp.full(56, jnp.nan)]), config).graph
    assert bool(jnp.all(jnp.isfinite(graph.nodes)))
    assert bool(jnp.all(jnp.isfinite(graph.edges)))
    batch = jax.tree_util.tree_map(
        lambda x: None if x is None else x[None], graph,
        is_leaf=lambda x: x is None)
    sequence = jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.repeat(x[:, None], 2, axis=1),
        batch, is_leaf=lambda x: x is None)
    algo = GPUDriveGNNIPPO(
        graph, action_mode="discrete", n_agents=64, minibatch_worlds=1,
        epoch_ppo=1, road_slots=200, road_feature_dim=14)
    actions, log_pis = algo.sample_actions(batch, jax.random.PRNGKey(9))
    mask = jnp.broadcast_to(valid, (1, 2, 64))
    rollout = MiniRollout(
        sequence, sequence, jnp.repeat(actions[:, None], 2, axis=1),
        jnp.zeros((1, 2, 64)), mask,
        jnp.zeros((1, 2, 64), dtype=bool),
        jnp.zeros((1, 2, 64), dtype=bool),
        jnp.repeat(log_pis[:, None], 2, axis=1),
    )
    info = algo.update(rollout)
    assert np.isfinite(float(info["policy/loss"]))
    assert np.isfinite(float(info["Vl/loss"]))
