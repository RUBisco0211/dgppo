"""CPU reference for Sah & Keshavan (2026) feasibility/allocation LPs.

Adapted from multi-cbf/src/mrcbf/certificate.py and filters.py for vector
controls, asymmetric boxes and single-owner obstacle rows. Attribution and
the upstream MIT notice are in docs/licenses/multi-cbf.txt.
This module is not JAX-jittable or differentiable.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class CBFConstraints:
    """b[k] + sum_i G[k,i] @ u[i] >= 0, with one/two owners per row."""

    G: np.ndarray                 # (constraints, agents, action_dim)
    b: np.ndarray                 # (constraints,)
    owners: np.ndarray            # (constraints, agents), boolean

    def __post_init__(self):
        self.G = np.asarray(self.G, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.owners = np.asarray(self.owners, dtype=bool)
        if self.G.ndim != 3 or min(self.G.shape[1:]) < 1:
            raise ValueError("G must have shape (constraints, agents, action_dim)")
        if self.b.shape != self.G.shape[:1] or self.owners.shape != self.G.shape[:2]:
            raise ValueError("b/owners do not match G")
        if not np.isfinite(self.G).all() or not np.isfinite(self.b).all():
            raise ValueError("CBF rows must be finite")
        if np.any((self.owners.sum(1) < 1) | (self.owners.sum(1) > 2)):
            raise ValueError("Each constraint needs one or two owners")
        if np.any(self.G[~self.owners] != 0):
            raise ValueError("Nonowners cannot contribute control to a row")


def action_bounds(rows, lower, upper):
    shape = rows.G.shape[1:]
    lower = np.broadcast_to(np.asarray(lower, dtype=float), shape).copy()
    upper = np.broadcast_to(np.asarray(upper, dtype=float), shape).copy()
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()) or np.any(lower > upper):
        raise ValueError("Action bounds must be finite and ordered")
    return lower, upper


def _lp(*args, **kwargs):
    result = linprog(*args, method="highs", **kwargs)
    if not result.success:
        # With finite boxes the margin LP is feasible, even if its optimum is
        # negative. Solver failure must not masquerade as a certificate.
        raise RuntimeError(f"CBF LP failed: {result.message}")
    return result


def feasibility_reserve(rows: CBFConstraints, lower, upper):
    """Return joint feasibility reserve M and its simplex multiplier.

    M>=0 iff the affine constraints admit a bounded action (up to numerical
    tolerance). This is not a certificate of future/discrete-time safety.
    """
    lower, upper = action_bounds(rows, lower, upper)
    k = len(rows.b)
    if not k:
        return np.inf, np.empty(0)
    G = rows.G.reshape(k, -1)
    center, radius = ((lower + upper) / 2).ravel(), ((upper - lower) / 2).ravel()
    d = len(center)
    # sigma_box(G.T lambda) = center @ G.T lambda + radius @ |G.T lambda|.
    result = _lp(
        np.concatenate([rows.b + G @ center, radius]),
        A_ub=np.block([[G.T, -np.eye(d)], [-G.T, -np.eye(d)]]),
        b_ub=np.zeros(2 * d),
        A_eq=np.concatenate([np.ones(k), np.zeros(d)])[None], b_eq=[1.0],
        bounds=[(0, None)] * (k + d),
    )
    return float(result.fun), result.x[:k]


def allocate_responsibility(rows: CBFConstraints, lower, upper, mode="certificate"):
    """Return shares (K,N), worst local margin t, and witness actions (N,D).

    Single-owner rows receive share 1. Uniform/capability use fixed shares
    and solve the same max-margin problem for comparable diagnostics.
    """
    if mode not in ("uniform", "capability", "certificate"):
        raise ValueError(f"Unknown allocation mode: {mode}")
    lower, upper = action_bounds(rows, lower, upper)
    k, n, d = rows.G.shape
    shares = rows.owners.astype(float) / np.maximum(rows.owners.sum(1, keepdims=True), 1)
    if not k:
        return shares, np.inf, np.clip(np.zeros((n, d)), lower, upper)
    pair_ids = np.flatnonzero(rows.owners.sum(1) == 2)
    pair_slot = {int(row): slot for slot, row in enumerate(pair_ids)}
    if mode == "capability":
        authority = np.sum(np.abs(rows.G) * ((upper - lower) / 2)[None], axis=-1)
        for row in pair_ids:
            if authority[row].sum() > 1e-12:
                shares[row] = authority[row] / authority[row].sum()
    optimize = mode == "certificate"
    p = len(pair_ids) if optimize else 0
    # Variables: t, theta (one per shared row), all agents' vector inputs.
    A, rhs = [], []
    for row in range(k):
        for endpoint, agent in enumerate(np.flatnonzero(rows.owners[row])):
            coeff = np.zeros(1 + p + n * d)
            coeff[0] = 1.0
            coeff[1 + p + agent * d:1 + p + (agent + 1) * d] = -rows.G[row, agent]
            offset = shares[row, agent] * rows.b[row]
            if optimize and row in pair_slot:
                coeff[1 + pair_slot[row]] = (-1 if endpoint == 0 else 1) * rows.b[row]
                offset = 0.0 if endpoint == 0 else rows.b[row]
            A.append(coeff)
            rhs.append(offset)
    cost = np.zeros(1 + p + n * d)
    cost[0] = -1.0
    result = _lp(cost, A_ub=np.asarray(A), b_ub=np.asarray(rhs),
                 bounds=[(None, None)] + [(0, 1)] * p + list(zip(lower.ravel(), upper.ravel())))
    if optimize:
        for slot, row in enumerate(pair_ids):
            i, j = np.flatnonzero(rows.owners[row])
            shares[row, i] = result.x[1 + slot]
            shares[row, j] = 1.0 - shares[row, i]
    return shares, float(result.x[0]), result.x[1 + p:].reshape(n, d)


class AllocatedCBFFilter:
    """LP allocation followed by local soft QPs (or a joint reference QP).

    Matches upstream's squared slack penalty, with optional max-margin
    witness fallback for certificate mode. Solver errors use bounded nominal
    controls and are reported explicitly; they do not imply safety.
    """

    MODES = ("uniform", "capability", "certificate", "centralized", "qp")

    def __init__(self, mode="certificate", *, slack_penalty=1e6,
                 witness_fallback=False, tolerance=1e-4):
        if mode not in self.MODES:
            raise ValueError(f"Unknown filter mode: {mode}")
        if not np.isfinite([slack_penalty, tolerance]).all() or min(slack_penalty, tolerance) <= 0:
            raise ValueError("Slack penalty and tolerance must be finite and positive")
        self.mode, self.slack_penalty = mode, slack_penalty
        self.witness_fallback, self.tolerance = witness_fallback, tolerance
        self._qp_cache = {}
        self.qp_retries = 0

    def _qp(self, G, b, nominal, lower, upper):
        k, d = G.shape
        if not k:
            return np.clip(nominal, lower, upper), False
        import cvxpy as cp

        key = (k, d)
        if key not in self._qp_cache:
            u, slack = cp.Variable(d), cp.Variable(k, nonneg=True)
            matrix, offset, ref = cp.Parameter((k, d)), cp.Parameter(k), cp.Parameter(d)
            lb, ub = cp.Parameter(d), cp.Parameter(d)
            problem = cp.Problem(
                cp.Minimize(cp.sum_squares(u - ref) + self.slack_penalty * cp.sum_squares(slack)),
                [matrix @ u + offset + slack >= 0, u >= lb, u <= ub],
            )
            self._qp_cache[key] = (problem, u, matrix, offset, ref, lb, ub)
        problem, u, matrix, offset, ref, lb, ub = self._qp_cache[key]
        matrix.value, offset.value, ref.value = G, b, nominal
        lb.value, ub.value = lower, upper
        try:
            problem.solve(solver=cp.CLARABEL)
            failed = problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        except cp.error.SolverError:
            failed = True
        failed = failed or u.value is None or not np.isfinite(u.value).all()
        if failed:
            # Parameter updates / warm-started factorizations can fail near a
            # tight constraint. Retry the identical QP from a fresh solver.
            self.qp_retries += 1
            try:
                problem.solve(solver=cp.CLARABEL, warm_start=False, max_iter=500)
                failed = problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            except cp.error.SolverError:
                failed = True
            failed = failed or u.value is None or not np.isfinite(u.value).all()
        return np.clip(nominal if failed else u.value, lower, upper), failed

    def project(self, rows: CBFConstraints, nominal, lower, upper):
        """Return executed bounded actions and independent feasibility diagnostics."""
        lower, upper = action_bounds(rows, lower, upper)
        nominal = np.asarray(nominal, dtype=float)
        if nominal.shape != rows.G.shape[1:] or not np.isfinite(nominal).all():
            raise ValueError("Nominal action must be finite with shape (agents, action_dim)")
        nominal = np.clip(nominal, lower, upper)
        retries_before = self.qp_retries
        k, n, d = rows.G.shape
        # The plain joint QP is a control baseline: it does not run either
        # feasibility/allocation LP from the paper.
        if self.mode == "qp":
            reserve, multiplier = np.nan, None
        else:
            reserve, multiplier = feasibility_reserve(rows, lower, upper)
        shares, margin = None, reserve
        failed, fallback = False, False
        if self.mode in ("centralized", "qp"):
            action, failed = self._qp(rows.G.reshape(k, n * d), rows.b,
                                      nominal.ravel(), lower.ravel(), upper.ravel())
            action = action.reshape(n, d)
        else:
            shares, margin, witness = allocate_responsibility(rows, lower, upper, self.mode)
            fallback = self.mode == "certificate" and self.witness_fallback and margin < -self.tolerance
            if fallback:
                action = witness
            else:
                action = nominal.copy()
                for i in range(n):
                    mask = rows.owners[:, i]
                    action[i], error = self._qp(rows.G[mask, i], shares[mask, i] * rows.b[mask],
                                                nominal[i], lower[i], upper[i])
                    failed = failed or error
        joint_residual = rows.b + np.einsum("knd,nd->k", rows.G, action)
        local_residual = joint_residual if shares is None else (
            np.einsum("knd,nd->kn", rows.G, action) + shares * rows.b[:, None]
        )[rows.owners]
        # Measure actual bounded action residuals, not the solver's slack.
        violation = float(np.maximum(-local_residual, 0).max(initial=0))
        return action, {
            "reserve": reserve, "allocation_margin": margin, "shares": shares,
            "multiplier": multiplier, "max_violation": violation,
            "joint_min_residual": float(joint_residual.min(initial=np.inf)),
            "infeasible": bool(margin < -self.tolerance), "solver_failed": failed,
            "solver_retries": self.qp_retries - retries_before,
            "fallback": fallback, "n_constraints": k,
        }
