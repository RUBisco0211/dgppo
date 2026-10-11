import jax.numpy as jnp
import numpy as np

from dgppo.env.gpudrive import (
    GPUDriveDGPPOConfig,
    build_ego_graph,
    build_ippo_scene_graph,
    build_scene_graph,
    continuous_collision_cost,
    road_edge_collision_cost,
)
from dgppo.env.gpudrive.graph import BASE_NODE_DIM, EDGE_DIM


def _sample_inputs():
    config = GPUDriveDGPPOConfig()
    ego = jnp.zeros((5, 8), dtype=jnp.float32)
    ego = ego.at[:, 0].set(jnp.arange(4.0, 9.0))
    ego = ego.at[:, 1:4].set(jnp.array([4.5, 2.0, 1.5]))
    ego = ego.at[:, 4:6].set(jnp.array([10.0, 0.0]))

    partner = jnp.zeros((5, 10, 9), dtype=jnp.float32)
    partner = partner.at[:, :, 0].set(3.0)
    partner = partner.at[:, :, 1].set(jnp.arange(1, 11) * 4.0)
    partner = partner.at[:, :, 4:7].set(jnp.array([4.5, 2.0, 1.5]))
    partner_valid = jnp.array(
        [[True] * 10, [True, True] + [False] * 8,
         [True] + [False] * 9, [False] * 10, [True] * 5 + [False] * 5]
    )
    partner_controlled = jnp.array(
        [[True, True] + [False] * 8] * 5
    )

    road = jnp.zeros((5, 32, 9), dtype=jnp.float32)
    road = road.at[:, :8, 0].set(jnp.arange(1.0, 9.0))
    road = road.at[:, :8, 2].set(2.0)
    road = road.at[:, :8, 6].set(3.0)
    return config, ego, partner, partner_valid, partner_controlled, road


def test_ego_graph_has_fixed_shape_and_only_ego_roots():
    args = _sample_inputs()
    out = build_ego_graph(*args[1:], config=args[0])
    graph = out.graph

    node_dim = BASE_NODE_DIM + args[0].num_road_segments * args[0].road_feature_dim
    assert graph.nodes.shape == (56, node_dim)
    assert graph.edges.shape == (100, EDGE_DIM)
    np.testing.assert_array_equal(np.asarray(graph.node_type[:5]), np.zeros(5, dtype=np.int32))
    assert int((np.asarray(graph.node_type) == 0).sum()) == 5
    assert int(np.asarray(out.node_valid_mask).sum()) == 5 + 18
    # All 18 visible partners send to ego. Only the seven visible,
    # controlled partners receive the cooperative return edge.
    assert int(np.asarray(out.edge_valid_mask).sum()) == 18 + 7


def test_ego_graph_background_edges_are_incoming_only():
    args = _sample_inputs()
    out = build_ego_graph(*args[1:], config=args[0])
    valid = np.asarray(out.edge_valid_mask).reshape(5, 10, 2)
    partner_valid = np.asarray(args[3])
    partner_controlled = np.asarray(args[4])
    np.testing.assert_array_equal(valid[..., 0], partner_valid)
    np.testing.assert_array_equal(valid[..., 1], partner_valid & partner_controlled)


def test_ego_graph_routes_invalid_edges_to_padding_node():
    args = _sample_inputs()
    out = build_ego_graph(*args[1:], config=args[0])
    graph = out.graph
    pad_id = graph.nodes.shape[0] - 1
    invalid = ~np.asarray(out.edge_valid_mask)
    assert np.all(np.asarray(graph.senders)[invalid] == pad_id)
    assert np.all(np.asarray(graph.receivers)[invalid] == pad_id)


def test_continuous_cost_sign_and_margin_for_axis_aligned_boxes():
    common = dict(
        ego_position=jnp.array([[0.0, 0.0]]),
        ego_heading=jnp.array([0.0]),
        ego_length=jnp.array([4.0]),
        ego_width=jnp.array([2.0]),
        other_heading=jnp.array([[0.0]]),
        other_length=jnp.array([[4.0]]),
        other_width=jnp.array([[2.0]]),
        valid_mask=jnp.array([[True]]),
        margin_m=0.5,
    )
    safe_cost, safe_clearance = continuous_collision_cost(
        other_position=jnp.array([[[5.0, 0.0]]]), mode="oriented_box", **common
    )
    boundary_cost, boundary_clearance = continuous_collision_cost(
        other_position=jnp.array([[[4.5, 0.0]]]), mode="oriented_box", **common
    )
    overlap_cost, overlap_clearance = continuous_collision_cost(
        other_position=jnp.array([[[3.5, 0.0]]]), mode="oriented_box", **common
    )
    np.testing.assert_allclose(safe_clearance, [1.0], atol=1e-6)
    np.testing.assert_allclose(safe_cost, [[-0.5]], atol=1e-6)
    np.testing.assert_allclose(boundary_clearance, [0.5], atol=1e-6)
    np.testing.assert_allclose(boundary_cost, [[0.0]], atol=1e-6)
    assert float(overlap_clearance[0]) < 0.0
    assert float(overlap_cost[0, 0]) > 0.5


def test_ellipse_mode_uses_same_cost_convention():
    cost, clearance = continuous_collision_cost(
        ego_position=jnp.array([[0.0, 0.0]]),
        ego_heading=jnp.array([0.0]),
        ego_length=jnp.array([4.0]),
        ego_width=jnp.array([2.0]),
        other_position=jnp.array([[[5.0, 0.0]]]),
        other_heading=jnp.array([[0.0]]),
        other_length=jnp.array([[4.0]]),
        other_width=jnp.array([[2.0]]),
        valid_mask=jnp.array([[True]]),
        margin_m=0.5,
        mode="ellipse",
    )
    np.testing.assert_allclose(clearance, [1.0], atol=1e-6)
    np.testing.assert_allclose(cost, [[-0.5]], atol=1e-6)


def test_continuous_cost_is_bounded_like_original_dgppo_environments():
    cost, _ = continuous_collision_cost(
        ego_position=jnp.array([[0.0, 0.0]]), ego_heading=jnp.array([0.0]),
        ego_length=jnp.array([4.0]), ego_width=jnp.array([2.0]),
        other_position=jnp.array([[[100.0, 0.0]]]), other_heading=jnp.array([[0.0]]),
        other_length=jnp.array([[4.0]]), other_width=jnp.array([[2.0]]),
        valid_mask=jnp.array([[True]]), margin_m=0.5, mode="oriented_box",
    )
    np.testing.assert_allclose(cost, [[-1.0]], atol=1e-6)


def test_road_edge_cost_uses_vehicle_body_clearance_and_ignores_road_lines():
    samples = jnp.zeros((2, 4, 4), dtype=jnp.float32)
    # Samples use (right, forward). Ego 0 sees a RoadEdge 3 m ahead: with a
    # 4 m long body the body-to-edge clearance is 1 m and cost is -0.5.
    samples = samples.at[0, 0].set(jnp.array([3.0, 1.0, 0.0, 3.0]))
    # Entity type 2 is RoadLine and must not become an obstacle cost.
    samples = samples.at[1, 0].set(jnp.array([0.25, 2.0, 0.25, 0.0]))
    cost, clearance = road_edge_collision_cost(
        samples, ego_length=jnp.array([4.0, 4.0]),
        ego_width=jnp.array([2.0, 2.0]), margin_m=0.5,
        mode="oriented_box",
    )
    np.testing.assert_allclose(cost[0], [-0.5], atol=1e-6)
    np.testing.assert_allclose(clearance[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(cost[1], [-1.0], atol=1e-6)


def test_scene_graph_exposes_five_controlled_value_roots():
    config = GPUDriveDGPPOConfig()
    vehicles = jnp.zeros((64, 14), dtype=jnp.float32)
    vehicles = vehicles.at[:, 10:13].set(jnp.array([4.5, 2.0, 1.5]))
    valid = jnp.arange(64) < 15
    controlled = jnp.arange(64) < 5
    roads = jnp.zeros((5, 32, 9), dtype=jnp.float32)
    roads = roads.at[:, :4, 6].set(3.0)
    graph = build_scene_graph(
        vehicles, valid, controlled, roads, jnp.arange(64, dtype=jnp.float32), config,
    )
    assert graph.nodes.shape == (65, BASE_NODE_DIM + 32 * 8)
    assert int((np.asarray(graph.node_type) == 0).sum()) == 5
    assert int((np.asarray(graph.node_type) == 1).sum()) == 10


def test_ippo_graph_keeps_fixed_slots_and_all_radius_edges():
    config = GPUDriveDGPPOConfig(
        num_controlled=64, num_background=0, control_all_vehicles=True,
        keep_all_scene_vehicles=True,
    )
    vehicles = jnp.zeros((64, 14), dtype=jnp.float32)
    vehicles = vehicles.at[:4, 0].set(jnp.array([0.0, 10.0, 60.0, 20.0]))
    vehicles = vehicles.at[:4, 10:13].set(jnp.array([4.5, 2.0, 1.5]))
    valid = jnp.arange(64) < 4
    roads = jnp.zeros((64, 32, 9), dtype=jnp.float32)
    out = build_ippo_scene_graph(
        vehicles, valid, roads, jnp.zeros(64, dtype=jnp.float32), config)
    graph = out.graph
    assert graph.nodes.shape == (65, BASE_NODE_DIM + 32 * 8)
    assert graph.edges.shape == (64 * 63, EDGE_DIM)
    # Five unordered pairs are within the inclusive 50 m radius, and every
    # visible pair contributes both directed message edges.
    assert int(np.asarray(out.edge_valid_mask).sum()) == 10
    assert int((np.asarray(graph.node_type) == 0).sum()) == 4
    invalid = ~np.asarray(out.edge_valid_mask)
    assert np.all(np.asarray(graph.senders)[invalid] == 64)
    assert np.all(np.asarray(graph.receivers)[invalid] == 64)
