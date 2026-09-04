"""GCBF+ actor/CBF joint training for DGPPO environments.

This is a native port of the upstream training objective.  The main adaptation
is a discrete-time QP teacher: it linearizes the real one-step environment
transition with respect to the joint action.  This preserves the GCBF+ training
loop for nonlinear VMAS physics without requiring every environment to expose
continuous control-affine dynamics.
"""

from __future__ import annotations

import functools as ft
import pickle
from pathlib import Path
from typing import NamedTuple, Optional

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP

from .base import Algorithm
from .gcbf_plus_adapter import GCBFPlusEnvAdapter, make_gcbf_plus_env_adapter
from .module.gcbf_plus import DeterministicGCBFPolicy, GCBFNetwork
from ..env.base import MultiAgentEnv
from ..trainer.data import GCBFTransitionBatch, Rollout
from ..trainer.gcbf_buffer import GCBFReplayBuffer
from ..trainer.utils import compute_norm_and_clip, has_any_nan_or_inf
from ..trainer.utils import rollout as rollout_fn
from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Params, PRNGKey
from ..utils.utils import jax_vmap, merge01


class GCBFTrainBatch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    qp_action: Action


class GCBFPlus(Algorithm):
    def __init__(
        self,
        env: MultiAgentEnv,
        node_dim: int,
        edge_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        gcbf_gnn_layers: int = 1,
        gcbf_batch_size: int = 256,
        gcbf_buffer_size: int = 65_536,
        gcbf_horizon: int = 32,
        gcbf_inner_epoch: int = 8,
        gcbf_lr_actor: float = 3e-5,
        gcbf_lr_cbf: float = 3e-5,
        gcbf_alpha: float = 1.0,
        gcbf_eps: float = 0.02,
        gcbf_loss_action_coef: float = 1e-4,
        gcbf_loss_unsafe_coef: float = 1.0,
        gcbf_loss_safe_coef: float = 1.0,
        gcbf_loss_h_dot_coef: float = 0.01,
        gcbf_target_tau: float = 0.5,
        gcbf_qp_relax_penalty: float = 1e3,
        gcbf_qp_chunk_size: int = 32,
        gcbf_unsafe_fraction: float = 0.5,
        max_grad_norm: float = 2.0,
        seed: int = 0,
        **kwargs,
    ):
        del kwargs
        super().__init__(env, node_dim, edge_dim, action_dim, n_agents)
        if gcbf_batch_size <= 0 or gcbf_buffer_size < gcbf_batch_size:
            raise ValueError("GCBF+ buffer size must be at least its positive batch size")
        if gcbf_horizon <= 0 or gcbf_inner_epoch <= 0:
            raise ValueError("GCBF+ horizon and inner epoch must be positive")
        if gcbf_qp_chunk_size <= 0:
            raise ValueError("GCBF+ QP chunk size must be positive")
        if not 0.0 <= gcbf_unsafe_fraction <= 1.0:
            raise ValueError("GCBF+ unsafe fraction must be in [0, 1]")
        if not 0.0 <= gcbf_target_tau <= 1.0:
            raise ValueError("GCBF+ target tau must be in [0, 1]")

        self.adapter: GCBFPlusEnvAdapter = make_gcbf_plus_env_adapter(env)
        self.state_dim = state_dim
        self.gnn_layers = gcbf_gnn_layers
        self.batch_size = gcbf_batch_size
        self.buffer_size = gcbf_buffer_size
        self.horizon = gcbf_horizon
        self.inner_epoch = gcbf_inner_epoch
        self.lr_actor = gcbf_lr_actor
        self.lr_cbf = gcbf_lr_cbf
        self.alpha = gcbf_alpha
        self.eps = gcbf_eps
        self.loss_action_coef = gcbf_loss_action_coef
        self.loss_unsafe_coef = gcbf_loss_unsafe_coef
        self.loss_safe_coef = gcbf_loss_safe_coef
        self.loss_h_dot_coef = gcbf_loss_h_dot_coef
        self.target_tau = gcbf_target_tau
        self.qp_relax_penalty = gcbf_qp_relax_penalty
        self.qp_chunk_size = gcbf_qp_chunk_size
        self.unsafe_fraction = gcbf_unsafe_fraction
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        key = jr.PRNGKey(seed)
        graph_key, cbf_key, actor_key, key = jr.split(key, 4)
        initial_graph = env.reset(graph_key)

        self.cbf = GCBFNetwork(n_agents, gcbf_gnn_layers)
        cbf_params = self.cbf.initialize(cbf_key, initial_graph)
        cbf_optimizer = optax.apply_if_finite(
            optax.adamw(gcbf_lr_cbf, weight_decay=1e-3), 1_000_000
        )
        self.cbf_train_state = TrainState.create(
            apply_fn=self.cbf.get_cbf,
            params=cbf_params,
            tx=cbf_optimizer,
        )
        self.cbf_target_params = jtu.tree_map(jnp.copy, cbf_params)

        self.actor = DeterministicGCBFPolicy(action_dim, n_agents, gcbf_gnn_layers)
        actor_params = self.actor.initialize(actor_key, initial_graph)
        actor_optimizer = optax.apply_if_finite(
            optax.adamw(gcbf_lr_actor, weight_decay=1e-3), 1_000_000
        )
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.get_action,
            params=actor_params,
            tx=actor_optimizer,
        )

        # GCBF+ is feed-forward; this placeholder satisfies the repository's
        # common rollout interface and is carried through unchanged.
        self.init_rnn_state = jnp.zeros((1, n_agents, 1, 1), dtype=jnp.float32)
        self.key = key
        self.replay = GCBFReplayBuffer(gcbf_buffer_size, seed=seed + 1)
        self._qp_batch_fn = jax.jit(
            jax.vmap(self.get_qp_action, in_axes=(0, None))
        )

        def collect_one(params, collect_key):
            return rollout_fn(
                self._env,
                ft.partial(self.step, params=params),
                self.init_rnn_state,
                collect_key,
            )

        self.rollout_fn = jax.jit(jax.vmap(collect_one, in_axes=(None, 0)))

    @property
    def config(self) -> dict:
        return {
            "gcbf_gnn_layers": self.gnn_layers,
            "gcbf_batch_size": self.batch_size,
            "gcbf_buffer_size": self.buffer_size,
            "gcbf_horizon": self.horizon,
            "gcbf_inner_epoch": self.inner_epoch,
            "gcbf_lr_actor": self.lr_actor,
            "gcbf_lr_cbf": self.lr_cbf,
            "gcbf_alpha": self.alpha,
            "gcbf_eps": self.eps,
            "gcbf_loss_action_coef": self.loss_action_coef,
            "gcbf_loss_unsafe_coef": self.loss_unsafe_coef,
            "gcbf_loss_safe_coef": self.loss_safe_coef,
            "gcbf_loss_h_dot_coef": self.loss_h_dot_coef,
            "gcbf_target_tau": self.target_tau,
            "gcbf_qp_relax_penalty": self.qp_relax_penalty,
            "gcbf_qp_chunk_size": self.qp_chunk_size,
            "gcbf_unsafe_fraction": self.unsafe_fraction,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "use_rnn": False,
        }

    @property
    def params(self) -> Params:
        return {
            "actor": self.actor_train_state.params,
            "cbf": self.cbf_train_state.params,
        }

    def get_cbf(self, graph: GraphsTuple, params: Optional[Params] = None) -> Array:
        cbf_params = self.cbf_train_state.params if params is None else params
        return self.cbf.get_cbf(cbf_params, graph)

    def _action(self, actor_params: Params, graph: GraphsTuple) -> Action:
        residual = self.actor.get_action(actor_params, graph)
        return 2.0 * residual + self.adapter.nominal_action(graph)

    def act(
        self,
        graph: GraphsTuple,
        rnn_state: Array,
        params: Optional[Params] = None,
    ) -> tuple[Action, Array]:
        params = self.params if params is None else params
        return self._action(params["actor"], graph), rnn_state

    def step(
        self,
        graph: GraphsTuple,
        rnn_state: Array,
        key: PRNGKey,
        params: Optional[Params] = None,
    ) -> tuple[Action, Array, Array]:
        del key
        action, rnn_state = self.act(graph, rnn_state, params)
        log_pi = jnp.zeros((self.n_agents,), dtype=action.dtype)
        return action, log_pi, rnn_state

    def collect(self, params: Params, b_key: PRNGKey) -> Rollout:
        return self.rollout_fn(params, b_key)

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, unsafe_mask: Array) -> Array:
        """Label states whose following horizon contains no unsafe state."""
        safe = jnp.ones_like(unsafe_mask, dtype=jnp.bool_)
        time_steps = unsafe_mask.shape[1]
        for time_idx in range(time_steps):
            start = max(0, time_idx - self.horizon)
            window = safe[:, start : time_idx + 1]
            window = window & ~unsafe_mask[:, time_idx : time_idx + 1]
            safe = safe.at[:, start : time_idx + 1].set(window)
        return safe.at[:, 0].set(True)

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_qp_action(self, graph: GraphsTuple, cbf_params: Params) -> Action:
        """Solve the discrete one-step, locally linearized GCBF+ teacher QP."""
        nominal = self.adapter.nominal_action(graph)
        h = self.cbf.get_cbf(cbf_params, graph).squeeze(-1)

        def next_h(action):
            next_graph = self.adapter.forward_graph(graph, action)
            return self.cbf.get_cbf(cbf_params, next_graph).squeeze(-1)

        h_next_nominal = next_h(nominal)
        jacobian = jax.jacobian(next_h)(nominal)
        jacobian_flat = ei.rearrange(jacobian, "ai aj u -> ai (aj u)")
        nominal_flat = nominal.reshape(-1)

        # h(x+) ~= h_nominal + J (u-u_nominal).  Enforce
        # h(x+) - h(x) + alpha*dt*h(x) + relaxation >= 0.
        rhs = (
            h_next_nominal
            - jacobian_flat @ nominal_flat
            + (self.alpha * self._env.dt - 1.0) * h
        )
        n_actions = self.n_agents * self.action_dim
        hessian = jnp.eye(n_actions + self.n_agents, dtype=nominal.dtype)
        hessian = hessian.at[n_actions:, n_actions:].multiply(10.0)
        gradient = jnp.concatenate(
            [-nominal_flat, self.qp_relax_penalty * jnp.ones(self.n_agents)]
        )
        constraint = -jnp.concatenate(
            [jacobian_flat, jnp.eye(self.n_agents)], axis=1
        )
        lower_action, upper_action = self._env.action_lim()
        lower = jnp.concatenate(
            [jnp.tile(lower_action, self.n_agents), jnp.zeros(self.n_agents)]
        )
        upper = jnp.concatenate(
            [
                jnp.tile(upper_action, self.n_agents),
                jnp.full((self.n_agents,), jnp.inf),
            ]
        )
        model = JaxProxQP.QPModel.create(
            hessian, gradient, constraint, rhs, lower, upper
        )
        solution = JaxProxQP(model, JaxProxQP.Settings.default()).solve()
        qp_action = solution.x[:n_actions].reshape(self.n_agents, self.action_dim)
        # A failed numerical solve must not poison replay training.  The
        # nominal controller is always a valid bounded fallback because the QP
        # relaxation variables make the safety constraints soft.
        qp_is_finite = jnp.all(jnp.isfinite(qp_action))
        return jnp.where(qp_is_finite, qp_action, nominal)

    def _batch_qp_actions(self, graph: GraphsTuple) -> Action:
        n_rows = int(graph.n_node.shape[0])
        chunks = []
        for start in range(0, n_rows, self.qp_chunk_size):
            stop = min(start + self.qp_chunk_size, n_rows)
            chunk = jtu.tree_map(lambda value: value[start:stop], graph)
            valid = stop - start
            if valid < self.qp_chunk_size:
                chunk = jtu.tree_map(
                    lambda value: jnp.concatenate(
                        [value, jnp.repeat(value[-1:], self.qp_chunk_size - valid, axis=0)],
                        axis=0,
                    ),
                    chunk,
                )
            chunks.append(self._qp_batch_fn(chunk, self.cbf_target_params)[:valid])
        return jnp.concatenate(chunks, axis=0)

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_batch(
        self,
        cbf_state: TrainState,
        actor_state: TrainState,
        batch: GCBFTrainBatch,
    ) -> tuple[TrainState, TrainState, dict]:
        safe_flat = merge01(batch.safe_mask)
        unsafe_flat = merge01(batch.unsafe_mask)

        def loss_fn(cbf_params, actor_params):
            cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf_params))
            cbf_no_grad_fn = jax_vmap(
                ft.partial(self.cbf.get_cbf, jax.lax.stop_gradient(cbf_params))
            )
            h = merge01(cbf_fn(batch.graph).squeeze(-1))

            unsafe_ratio = jnp.mean(unsafe_flat)
            h_unsafe = jnp.where(unsafe_flat, h, -2.0 * self.eps)
            loss_unsafe = jnp.sum(jax.nn.relu(h_unsafe + self.eps)) / (
                jnp.count_nonzero(unsafe_flat) + 1e-6
            )
            h_safe = jnp.where(safe_flat, h, 2.0 * self.eps)
            loss_safe = jnp.sum(jax.nn.relu(-h_safe + self.eps)) / (
                jnp.count_nonzero(safe_flat) + 1e-6
            )

            actions = jax_vmap(ft.partial(self._action, actor_params))(batch.graph)
            next_graph = jax_vmap(self.adapter.forward_graph)(batch.graph, actions)
            h_next = merge01(cbf_fn(next_graph).squeeze(-1))
            h_dot = (h_next - h) / self._env.dt

            h_stopped = jax.lax.stop_gradient(h)
            h_next_cbf_stopped = merge01(cbf_no_grad_fn(next_graph).squeeze(-1))
            h_dot_cbf_stopped = (h_next_cbf_stopped - h_stopped) / self._env.dt
            labeled = safe_flat | unsafe_flat
            violation = jax.nn.relu(-h_dot - self.alpha * h + self.eps)
            next_cbf_frozen_violation = jax.nn.relu(
                -h_dot_cbf_stopped - self.alpha * h + self.eps
            )
            loss_h_dot = jnp.mean(
                jnp.where(labeled, violation, next_cbf_frozen_violation)
            )
            loss_action = jnp.mean(jnp.sum(jnp.square(actions - batch.qp_action), axis=-1))

            total = (
                self.loss_action_coef * loss_action
                + self.loss_unsafe_coef * loss_unsafe
                + self.loss_safe_coef * loss_safe
                + self.loss_h_dot_coef * loss_h_dot
            )
            info = {
                "loss/action": loss_action,
                "loss/unsafe": loss_unsafe,
                "loss/safe": loss_safe,
                "loss/h_dot": loss_h_dot,
                "loss/total": total,
                "acc/unsafe": jnp.sum(jnp.where(unsafe_flat, h < 0.0, False))
                / (jnp.count_nonzero(unsafe_flat) + 1e-6),
                "acc/safe": jnp.sum(jnp.where(safe_flat, h > 0.0, False))
                / (jnp.count_nonzero(safe_flat) + 1e-6),
                "acc/h_dot": jnp.mean(h_dot + self.alpha * h > 0.0),
                "data/unsafe_ratio": unsafe_ratio,
            }
            return total, info

        (loss, info), (cbf_grad, actor_grad) = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 1)
        )(cbf_state.params, actor_state.params)
        del loss
        cbf_has_nan = has_any_nan_or_inf(cbf_grad).astype(jnp.float32)
        actor_has_nan = has_any_nan_or_inf(actor_grad).astype(jnp.float32)
        cbf_grad, cbf_norm = compute_norm_and_clip(cbf_grad, self.max_grad_norm)
        actor_grad, actor_norm = compute_norm_and_clip(actor_grad, self.max_grad_norm)
        cbf_state = cbf_state.apply_gradients(grads=cbf_grad)
        actor_state = actor_state.apply_gradients(grads=actor_grad)
        return cbf_state, actor_state, info | {
            "grad_norm/cbf": cbf_norm,
            "grad_norm/actor": actor_norm,
            "grad_has_nan/cbf": cbf_has_nan,
            "grad_has_nan/actor": actor_has_nan,
            "policy/loss": info["loss/action"],
        }

    def update(self, rollout: Rollout, step: int) -> dict:
        del step
        unsafe = jnp.max(rollout.costs, axis=-1) > 0.0
        safe = self.safe_mask(unsafe)
        transitions = GCBFTransitionBatch(
            graph=jtu.tree_map(merge01, rollout.graph),
            next_graph=jtu.tree_map(merge01, rollout.next_graph),
            safe_mask=merge01(safe),
            unsafe_mask=merge01(unsafe),
        )
        self.replay.append(transitions)

        update_info = None
        current_batch_size = min(self.batch_size, self.replay.length)
        for _ in range(self.inner_epoch):
            sampled = self.replay.sample(current_batch_size, self.unsafe_fraction)
            qp_action = self._batch_qp_actions(sampled.graph)
            batch = GCBFTrainBatch(
                graph=sampled.graph,
                safe_mask=sampled.safe_mask,
                unsafe_mask=sampled.unsafe_mask,
                qp_action=qp_action,
            )
            self.cbf_train_state, self.actor_train_state, update_info = self._update_batch(
                self.cbf_train_state, self.actor_train_state, batch
            )

        self.cbf_target_params = optax.incremental_update(
            self.cbf_train_state.params,
            self.cbf_target_params,
            self.target_tau,
        )
        assert update_info is not None
        return update_info | {"data/replay_size": jnp.asarray(self.replay.length)}

    def save(self, save_dir: str, step: int | str):
        model_dir = Path(save_dir) / str(step)
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "actor.pkl").open("wb") as file:
            pickle.dump(self.actor_train_state.params, file)
        with (model_dir / "cbf.pkl").open("wb") as file:
            pickle.dump(self.cbf_train_state.params, file)
        with (model_dir / "algo_training_state.pkl").open("wb") as file:
            pickle.dump(
                {
                    "format_version": 1,
                    "actor_step": self.actor_train_state.step,
                    "actor_opt_state": self.actor_train_state.opt_state,
                    "cbf_step": self.cbf_train_state.step,
                    "cbf_opt_state": self.cbf_train_state.opt_state,
                    "cbf_target_params": self.cbf_target_params,
                    "algo_key": np.asarray(self.key),
                },
                file,
            )
        self.replay.save(model_dir / "replay.pkl")

    def load(self, load_dir: str, step: int | str):
        model_dir = Path(load_dir) / str(step)
        with (model_dir / "actor.pkl").open("rb") as file:
            actor_params = pickle.load(file)
        with (model_dir / "cbf.pkl").open("rb") as file:
            cbf_params = pickle.load(file)
        self.actor_train_state = self.actor_train_state.replace(params=actor_params)
        self.cbf_train_state = self.cbf_train_state.replace(params=cbf_params)
        self.cbf_target_params = jtu.tree_map(jnp.copy, cbf_params)

        state_path = model_dir / "algo_training_state.pkl"
        if state_path.exists():
            with state_path.open("rb") as file:
                state = pickle.load(file)
            if state.get("format_version") != 1:
                raise ValueError("unsupported GCBF+ training checkpoint format")
            self.actor_train_state = self.actor_train_state.replace(
                step=state["actor_step"], opt_state=state["actor_opt_state"]
            )
            self.cbf_train_state = self.cbf_train_state.replace(
                step=state["cbf_step"], opt_state=state["cbf_opt_state"]
            )
            self.cbf_target_params = state["cbf_target_params"]
            self.key = jnp.asarray(state["algo_key"])

        replay_path = model_dir / "replay.pkl"
        if replay_path.exists():
            self.replay.load(replay_path)
