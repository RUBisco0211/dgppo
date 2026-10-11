"""Mathematical and simulator-interface checks for the MPE CBF migration."""
import importlib.util
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest

from dgppo.algo.module.cbf_allocation import (
    AllocatedCBFFilter, CBFConstraints, allocate_responsibility, feasibility_reserve,
)
from dgppo.algo.module.mpe_cbf import MPECBFConfig, MPEPairwiseCBF, build_mpe_cbf


def test_hocbf_matches_directional_derivative():
    states = np.array([[0.2, 0.3, 0.4, -0.1], [0.7, 0.6, -0.3, 0.2]])
    obstacles = np.array([[0.4, 0.8]])
    config = MPECBFConfig(alpha1=2, alpha2=4, safety_margin=0.02)
    u = np.array([[0.3, -0.1], [-0.2, 0.4]])
    rows = build_mpe_cbf(states, obstacles, 0.05, 0.08, config)
    derivative = np.concatenate([states[:, 2:], 10 * u], axis=1)
    eps = 1e-6
    plus = build_mpe_cbf(states + eps * derivative, obstacles, 0.05, 0.08, config)
    minus = build_mpe_cbf(states - eps * derivative, obstacles, 0.05, 0.08, config)
    expected = (plus.psi1 - minus.psi1) / (2 * eps) + config.alpha2 * rows.psi1
    actual = rows.constraints.b + np.einsum("knd,nd->k", rows.constraints.G, u)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_array_equal(rows.entities, [[0, 1], [0, -1], [1, -1]])
    assert rows.constraints.owners.sum(1).tolist() == [2, 1, 1]


@pytest.mark.parametrize("seed", range(8))
def test_vector_reserve_matches_independent_primal(seed):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(5, 3, 2))
    owners = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1]], bool)
    G[~owners] = 0
    rows = CBFConstraints(G, rng.normal(size=5), owners)
    lower, upper = rng.uniform(-2, -0.1, (3, 2)), rng.uniform(0.1, 1, (3, 2))
    reserve, multiplier = feasibility_reserve(rows, lower, upper)
    u, t = cp.Variable(6), cp.Variable()
    problem = cp.Problem(cp.Maximize(t), [G.reshape(5, 6) @ u + rows.b >= t,
                                         u >= lower.ravel(), u <= upper.ravel()])
    problem.solve(solver=cp.CLARABEL)
    assert abs(reserve - problem.value) < 1e-6
    assert np.min(multiplier) >= -1e-9
    assert abs(multiplier.sum() - 1) < 1e-9
    shares, margin, witness = allocate_responsibility(rows, lower, upper)
    assert margin >= allocate_responsibility(rows, lower, upper, "uniform")[1] - 1e-8
    assert margin >= allocate_responsibility(rows, lower, upper, "capability")[1] - 1e-8
    np.testing.assert_allclose(shares.sum(1), 1, atol=1e-9)
    assert np.all(shares[~owners] == 0)
    residual = np.einsum("knd,nd->kn", G, witness) + shares * rows.b[:, None]
    assert abs(residual[owners].min() - margin) < 1e-8


def obstacle_conflict():
    # Pair: u0_x+u1_x>=0.8. Obstacle: u0_x<=0.
    # Equal shares require u0_x>=0.4, but allocating the pair to agent1 works.
    G = np.zeros((2, 2, 2))
    G[0, :, 0] = 1
    G[1, 0, 0] = -1
    return CBFConstraints(G, [-0.8, 0], [[1, 1], [1, 0]])


def test_allocation_rescues_uniform_obstacle_conflict():
    rows = obstacle_conflict()
    assert allocate_responsibility(rows, -1, 1, "uniform")[1] < -0.1
    shares, margin, _ = allocate_responsibility(rows, -1, 1)
    assert margin >= -1e-9
    np.testing.assert_allclose(shares, [[0, 1], [1, 0]], atol=1e-8)
    action, info = AllocatedCBFFilter().project(rows, np.zeros((2, 2)), -1, 1)
    assert not info["infeasible"] and not info["solver_failed"]
    assert info["max_violation"] < 1e-4
    assert action[1, 0] > 0.79


def test_plain_qp_uses_joint_constraints_without_allocation_lps(monkeypatch):
    import dgppo.algo.module.cbf_allocation as allocation

    def forbidden(*args, **kwargs):
        raise AssertionError("plain QP called a paper LP")

    monkeypatch.setattr(allocation, "feasibility_reserve", forbidden)
    monkeypatch.setattr(allocation, "allocate_responsibility", forbidden)
    rows = obstacle_conflict()
    action, info = AllocatedCBFFilter("qp").project(rows, np.zeros((2, 2)), -1, 1)
    assert not info["solver_failed"] and info["shares"] is None
    assert np.isnan(info["reserve"])
    assert info["max_violation"] < 1e-4
    assert action[1, 0] > 0.79


def test_joint_feasibility_does_not_imply_allocatable_feasibility():
    rows = obstacle_conflict()
    rows.b[:] = [0, -0.5]  # joint: u0+u1>=0, u0<=-0.5; local pair requires u0>=0
    assert feasibility_reserve(rows, -1, 1)[0] > 0
    assert allocate_responsibility(rows, -1, 1)[1] < 0


@pytest.mark.parametrize("mode", AllocatedCBFFilter.MODES)
def test_empty_constraints_preserve_bounded_nominal(mode):
    rows = CBFConstraints(np.zeros((0, 1, 2)), np.empty(0), np.zeros((0, 1), bool))
    action, info = AllocatedCBFFilter(mode).project(rows, [[2, -0.3]], -1, 1)
    np.testing.assert_allclose(action, [[1, -0.3]])
    assert (np.isnan(info["reserve"]) if mode == "qp" else np.isinf(info["reserve"]))
    assert not info["infeasible"]


def test_infeasible_witness_and_solver_failure_are_explicit(monkeypatch):
    rows = obstacle_conflict()
    rows.b[0] = -3
    solver = AllocatedCBFFilter(witness_fallback=True)
    action, info = solver.project(rows, np.zeros((2, 2)), -1, 1)
    assert info["infeasible"] and info["fallback"] and info["max_violation"] > 0
    assert np.max(np.abs(action)) <= 1

    def fail(*args, **kwargs):
        raise cp.error.SolverError("intentional test failure")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    nominal = np.full((2, 2), 0.2)
    action, info = AllocatedCBFFilter().project(rows, nominal, -1, 1)
    np.testing.assert_allclose(action, nominal)
    assert info["solver_failed"]


def test_scalar_limit_matches_upstream():
    path = Path(__file__).resolve().parents[1] / "multi-cbf/src/mrcbf/certificate.py"
    if not path.is_file():
        pytest.skip("Optional upstream checkout is absent")
    spec = importlib.util.spec_from_file_location("upstream_certificate", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rng = np.random.default_rng(19)
    edges = [(0, 1), (0, 2), (1, 2)]
    for _ in range(10):
        G = rng.normal(size=(3, 3))
        owners = np.zeros((3, 3), bool)
        for k, (i, j) in enumerate(edges):
            owners[k, [i, j]] = True
        G[~owners] = 0
        b, bound = rng.normal(size=3), rng.uniform(0.5, 2, size=3)
        rows = CBFConstraints(G[..., None], b, owners)
        value, multiplier = feasibility_reserve(rows, -bound[:, None], bound[:, None])
        ref, _, _ = upstream.reserve(G, b, bound)
        assert abs(value - ref) < 1e-8
        shares, margin, _ = allocate_responsibility(rows, -bound[:, None], bound[:, None])
        _, ref_margin, _ = upstream.alloc_certificate(G, b, edges, list(range(3)), 3, bound)
        assert abs(margin - ref_margin) < 1e-8


def test_solver_failure_retries_identical_qp_from_cold_start(monkeypatch):
    original = cp.Problem.solve
    calls = []

    def fail_once(problem, *args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise cp.error.SolverError("warm-start test failure")
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_once)
    action, info = AllocatedCBFFilter().project(obstacle_conflict(), np.zeros((2, 2)), -1, 1)
    assert info["solver_retries"] == 1 and not info["solver_failed"]
    assert calls[1]["warm_start"] is False
    assert action[1, 0] > 0.79


@pytest.mark.parametrize("n,obs", [(1, 0), (3, 0), (3, 2)])
def test_mpe_graph_adapter_and_step(n, obs):
    import jax
    import jax.numpy as jnp
    from dgppo.env import make_env

    env = make_env("MPETarget", n, num_obs=obs)
    graph = env.reset(jax.random.PRNGKey(4))
    filter_ = MPEPairwiseCBF(env)
    rows = filter_.rows(graph)
    assert len(rows.h) == n * (n - 1) // 2 + n * obs
    np.testing.assert_array_equal(rows.constraints.owners.sum(0), n - 1 + obs)
    for _ in range(2):
        action, info = filter_(graph, np.zeros((n, 2)))
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
        graph, *_ = env.step(graph, jnp.asarray(action))
        assert not info["solver_failed"]


def test_sensing_mask_is_symmetric_and_lidar_is_rejected():
    from dgppo.env import make_env

    states = np.array([[0, 0, 0, 0], [0.2, 0, 0, 0], [1, 0, 0, 0]])
    rows = build_mpe_cbf(states, np.empty((0, 2)), 0.05, 0.05,
                         MPECBFConfig(sensing_radius=0.3))
    np.testing.assert_array_equal(rows.entities, [[0, 1]])
    with pytest.raises(ValueError, match="only MPE"):
        MPEPairwiseCBF(make_env("LidarTarget", 3, num_obs=0))
    with pytest.raises(ValueError, match="connectivity"):
        MPEPairwiseCBF(make_env("MPEConnectSpread", 3))
