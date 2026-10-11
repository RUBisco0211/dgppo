"""Handcrafted pairwise second-order CBFs for DGPPO's MPE environments.

Uses x_dot=[v,10*u] before the simulator's state clipping. Safe is h>=0,
opposite to environment cost. Does not use learned/manifold CBFs or another
environment repository. All pairs and all obstacles are retained by default.
"""
from dataclasses import dataclass

import numpy as np

from .cbf_allocation import AllocatedCBFFilter, CBFConstraints


@dataclass(frozen=True)
class MPECBFConfig:
    alpha1: float = 3.0
    alpha2: float = 3.0
    safety_margin: float = 0.0
    # Optional experimental distance mask; None means complete geometry.
    sensing_radius: float | None = None

    def __post_init__(self):
        if not np.isfinite([self.alpha1, self.alpha2, self.safety_margin]).all():
            raise ValueError("CBF parameters must be finite")
        if min(self.alpha1, self.alpha2) <= 0 or self.safety_margin < 0:
            raise ValueError("Gains must be positive and safety margin nonnegative")
        if self.sensing_radius is not None and (
                not np.isfinite(self.sensing_radius) or self.sensing_radius <= 0):
            raise ValueError("Sensing radius must be positive and finite")


@dataclass
class MPECBFRows:
    constraints: CBFConstraints
    h: np.ndarray
    psi1: np.ndarray
    # Shared rows use (i,j), obstacle rows use (i,-1-obstacle_id).
    entities: np.ndarray


def build_mpe_cbf(states, obstacles, agent_radius, obstacle_radius, config=None):
    """Build b+Gu>=0 from positions/velocities and static circular obstacles.

    h=||p||²-d_safe²; psi1=h_dot+alpha1*h;
    b=2||v||²+2(alpha1+alpha2)p.v+alpha1*alpha2*h;
    G_i=20*p.T and G_j=-20*p.T for shared robot-pair rows.
    """
    config = MPECBFConfig() if config is None else config
    states, obstacles = np.asarray(states, dtype=float), np.asarray(obstacles, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4 or len(states) < 1:
        raise ValueError("Expected states with shape (agents,4): x,y,vx,vy")
    if obstacles.ndim != 2 or obstacles.shape[1] != 2:
        raise ValueError("Expected obstacle centers with shape (obstacles,2)")
    if not np.isfinite(states).all() or not np.isfinite(obstacles).all():
        raise ValueError("States and obstacles must be finite")
    if not np.isfinite([agent_radius, obstacle_radius]).all() or min(agent_radius, obstacle_radius) < 0:
        raise ValueError("Radii must be finite and nonnegative")
    n = len(states)
    G, b, owners, hs, psis, entities = [], [], [], [], [], []

    def append(i, other_id, position, velocity, safe_distance):
        p, v = states[i, :2] - position, states[i, 2:] - velocity
        if config.sensing_radius is not None and np.linalg.norm(p) > config.sensing_radius:
            return
        row, owner = np.zeros((n, 2)), np.zeros(n, dtype=bool)
        row[i], owner[i] = 20.0 * p, True
        if other_id >= 0:
            row[other_id], owner[other_id] = -row[i], True
        h = p @ p - safe_distance ** 2
        h_dot = 2.0 * (p @ v)
        psi1 = h_dot + config.alpha1 * h
        drift = 2.0 * (v @ v) + config.alpha1 * h_dot + config.alpha2 * psi1
        G.append(row)
        b.append(drift)
        owners.append(owner)
        hs.append(h)
        psis.append(psi1)
        entities.append((i, other_id))

    for i in range(n):
        for j in range(i + 1, n):
            append(i, j, states[j, :2], states[j, 2:], 2 * agent_radius + config.safety_margin)
        for o, center in enumerate(obstacles):
            append(i, -1 - o, center, np.zeros(2), agent_radius + obstacle_radius + config.safety_margin)
    constraints = CBFConstraints(np.asarray(G).reshape(-1, n, 2), np.asarray(b),
                                 np.asarray(owners, dtype=bool).reshape(-1, n))
    return MPECBFRows(constraints, np.asarray(hs), np.asarray(psis),
                      np.asarray(entities, dtype=int).reshape(-1, 2))


class MPEPairwiseCBF:
    """Evaluation-time wrapper: policy action -> LP/QP -> env.step.

    Reads full geometric state; centralized allocation plus local QPs is not
    fully decentralized execution. Call outside jax.jit/vmap/lax.scan.
    """

    def __init__(self, env, mode="certificate", config=None, **solver_kwargs):
        from ...env.mpe.base import MPE
        from ...env.mpe.mpe_connect_spread import MPEConnectSpread

        if not isinstance(env, MPE) or env.state_dim != 4 or env.action_dim != 2:
            raise ValueError("MPEPairwiseCBF supports only MPE double integrators")
        if isinstance(env, MPEConnectSpread):
            raise ValueError("MPEConnectSpread's additional connectivity constraint is not implemented")
        if type(env).agent_step_euler is not MPE.agent_step_euler:
            raise ValueError("This MPE overrides the supported double-integrator dynamics")
        self.env = env
        self.config = MPECBFConfig() if config is None else config
        self.solver = AllocatedCBFFilter(mode, **solver_kwargs)

    def rows(self, graph):
        env = self.env
        states = np.asarray(graph.type_states(env.AGENT, env.num_agents))
        obstacles = (np.asarray(graph.type_states(env.OBS, env.params["n_obs"]))[:, :2]
                     if env.params["n_obs"] else np.empty((0, 2)))
        return build_mpe_cbf(states, obstacles, env.params["car_radius"],
                             env.params["obs_radius"], self.config)

    def __call__(self, graph, nominal):
        rows = self.rows(graph)
        action, info = self.solver.project(rows.constraints, nominal, *self.env.action_lim())
        info.update(min_h=float(rows.h.min(initial=np.inf)),
                    min_psi1=float(rows.psi1.min(initial=np.inf)), entities=rows.entities)
        return action, info
