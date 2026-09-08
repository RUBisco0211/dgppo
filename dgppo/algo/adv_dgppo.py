"""Adversarial DGCBF training integrated with DGPPO-style PPO updates."""

from __future__ import annotations

import functools as ft
import os
import pickle
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import lax

from .informarl import InforMARL
from .module.adv_dgcbf import (
    ActionConditionedSafetyQ,
    SharedAdversary,
    SharedSafetyActor,
    SharedSafetyValue,
)
from .utils import compute_dec_ocp_gae
from ..trainer.data import Rollout
from ..trainer.utils import compute_norm_and_clip, has_any_nan_or_inf
from ..trainer.utils import test_rollout as deterministic_rollout
from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Params
from ..utils.utils import merge01, tree_index


def compute_adv_dgcbf_target(
    constraint: Array,
    next_value: Array,
    safety_gamma: float,
) -> tuple[Array, Array]:
    """Return discounted and exact undiscounted safe-positive Bellman targets."""
    discounted_continuation = (
        (1.0 - safety_gamma) * constraint + safety_gamma * next_value
    )
    discounted = jnp.minimum(constraint, discounted_continuation)
    undiscounted = jnp.minimum(constraint, next_value)
    return discounted, undiscounted


def mix_adv_dgcbf_advantages(
    task_advantage: Array,
    value: Array,
    task_action_value: Array,
    cbf_weight: float,
    cbf_kappa: float,
    cbf_eps: float,
) -> tuple[Array, Array, Array]:
    """Apply DGPPO's per-sample switch using a robust safe-positive residual."""
    target_floor = jnp.where(value >= 0.0, (1.0 - cbf_kappa) * value, 0.0)
    violation = jax_relu(target_floor - task_action_value + cbf_eps)
    safe = (value >= 0.0) & (violation <= 0.0)
    mixed = jnp.where(safe, task_advantage, -cbf_weight * violation)
    return mixed, violation, safe


def jax_relu(value: Array) -> Array:
    return jnp.maximum(value, 0.0)


def sample_transition_indices(
    key: Array,
    population_size: int,
    sample_size: int,
) -> Array:
    """Sample a minibatch without shuffling the full rollout index array."""
    if sample_size == population_size:
        return jnp.arange(population_size, dtype=jnp.int32)
    return jr.randint(
        key,
        shape=(sample_size,),
        minval=0,
        maxval=population_size,
        dtype=jnp.int32,
    )


class AdversarialDGPPO(InforMARL):
    """InforMARL PPO with a separately sampled adversarial DGCBF safety game."""

    def __init__(
        self,
        *args,
        Vh_gnn_layers: int = 1,
        lr_Vh: float = 1e-3,
        alpha: float = 10.0,
        cbf_eps: float = 1e-2,
        cbf_weight: float = 1.0,
        train_steps: int = 100_000,
        cbf_schedule: bool = True,
        adv_gnn_layers: Optional[int] = None,
        adv_hidden_dim: int = 64,
        adv_inner_steps: int = 1,
        adv_target_tau: float = 0.005,
        adv_batch_size: int = 4096,
        adv_task_sample_ratio: float = 0.25,
        adv_actor_egos_per_sample: int = 1,
        safety_gamma: Optional[float] = None,
        cbf_kappa: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if adv_inner_steps <= 0:
            raise ValueError("adv_inner_steps must be positive")
        if not 0.0 < adv_target_tau <= 1.0:
            raise ValueError("adv_target_tau must be in (0, 1]")
        if adv_batch_size <= 0:
            raise ValueError("adv_batch_size must be positive")
        if not 0.0 < adv_task_sample_ratio <= 1.0:
            raise ValueError("adv_task_sample_ratio must be in (0, 1]")
        if not 1 <= adv_actor_egos_per_sample <= self.n_agents:
            raise ValueError(
                "adv_actor_egos_per_sample must be between 1 and n_agents"
            )
        self.adv_gnn_layers = (
            Vh_gnn_layers if adv_gnn_layers is None else adv_gnn_layers
        )
        self.adv_hidden_dim = adv_hidden_dim
        self.adv_inner_steps = adv_inner_steps
        self.adv_target_tau = adv_target_tau
        self.adv_batch_size = adv_batch_size
        self.adv_task_sample_ratio = adv_task_sample_ratio
        self.adv_actor_egos_per_sample = adv_actor_egos_per_sample
        self.safety_gamma = self.gamma if safety_gamma is None else safety_gamma
        if not 0.0 < self.safety_gamma <= 1.0:
            raise ValueError("safety_gamma must be in (0, 1]")
        default_kappa = float(np.clip(alpha * self._env.dt, 0.0, 1.0))
        self.cbf_kappa = default_kappa if cbf_kappa is None else cbf_kappa
        if not 0.0 < self.cbf_kappa <= 1.0:
            raise ValueError("cbf_kappa must be in (0, 1]")
        self.cbf_eps = cbf_eps
        self.cbf_weight = cbf_weight
        self.cbf_schedule = cbf_schedule
        if cbf_schedule:
            self.cbf_schedule_fn = optax.piecewise_constant_schedule(
                init_value=cbf_weight,
                boundaries_and_scales={
                    int(train_steps * 0.5): 2.0,
                    int(train_steps * 0.75): 2.0,
                },
            )
        else:
            self.cbf_schedule_fn = optax.constant_schedule(cbf_weight)

        graph_key, q_key, vh_key, mu_key, beta_key, self.key = jr.split(
            self.key, 6
        )
        initial_graph = self._env.reset(graph_key)
        zero_action = jnp.zeros((self.n_agents, self.action_dim))
        ego_id = jnp.array(0, dtype=jnp.int32)
        neighbor_mask = self._neighbor_mask(initial_graph, ego_id)

        self.adv_q = ActionConditionedSafetyQ(
            self.action_dim,
            self.n_agents,
            self.adv_gnn_layers,
            adv_hidden_dim,
        )
        self.adv_vh = SharedSafetyValue(
            self.n_agents, self.adv_gnn_layers, adv_hidden_dim
        )
        self.adv_mu = SharedSafetyActor(
            self.action_dim,
            self.n_agents,
            self.adv_gnn_layers,
            adv_hidden_dim,
        )
        self.adv_beta = SharedAdversary(
            self.action_dim,
            self.n_agents,
            self.adv_gnn_layers,
            adv_hidden_dim,
        )

        q_params = self.adv_q.initialize(
            q_key, initial_graph, zero_action, ego_id, neighbor_mask
        )
        vh_params = self.adv_vh.initialize(vh_key, initial_graph)
        mu_params = self.adv_mu.initialize(mu_key, initial_graph)
        beta_params = self.adv_beta.initialize(
            beta_key,
            initial_graph,
            zero_action[0],
            ego_id,
            neighbor_mask,
        )
        critic_optim = optax.apply_if_finite(optax.adam(lr_Vh), 1_000_000)
        actor_optim = optax.apply_if_finite(
            optax.adam(self.lr_actor), 1_000_000
        )
        self.adv_q_train_state = TrainState.create(
            apply_fn=self.adv_q.get_value, params=q_params, tx=critic_optim
        )
        self.adv_vh_train_state = TrainState.create(
            apply_fn=self.adv_vh.get_value, params=vh_params, tx=critic_optim
        )
        self.adv_mu_train_state = TrainState.create(
            apply_fn=self.adv_mu.get_action, params=mu_params, tx=actor_optim
        )
        self.adv_beta_train_state = TrainState.create(
            apply_fn=self.adv_beta.get_action, params=beta_params, tx=actor_optim
        )
        self.adv_q_target_params = jtu.tree_map(jnp.copy, q_params)
        self.adv_mu_target_params = jtu.tree_map(jnp.copy, mu_params)
        self.adv_beta_target_params = jtu.tree_map(jnp.copy, beta_params)

        def collect_one(params, collect_key, collect_ego_id):
            def game_actor(graph, rnn_state):
                task_action, next_rnn_state = self.act(graph, rnn_state, params)
                game_action = self._game_action(
                    graph,
                    collect_ego_id,
                    task_action,
                    params["adv_mu"],
                    params["adv_beta"],
                )
                return game_action, next_rnn_state

            return deterministic_rollout(
                self._env, game_actor, self.init_rnn_state, collect_key
            )

        self.adv_rollout_fn = jax.jit(
            jax.vmap(collect_one, in_axes=(None, 0, 0))
        )

    @property
    def config(self) -> dict:
        return super().config | {
            "Vh_gnn_layers": self.adv_gnn_layers,
            "adv_gnn_layers": self.adv_gnn_layers,
            "adv_hidden_dim": self.adv_hidden_dim,
            "adv_inner_steps": self.adv_inner_steps,
            "adv_target_tau": self.adv_target_tau,
            "adv_batch_size": self.adv_batch_size,
            "adv_task_sample_ratio": self.adv_task_sample_ratio,
            "adv_actor_egos_per_sample": self.adv_actor_egos_per_sample,
            "safety_gamma": self.safety_gamma,
            "cbf_kappa": self.cbf_kappa,
            "cbf_eps": self.cbf_eps,
            "cbf_weight": self.cbf_weight,
            "cbf_schedule": self.cbf_schedule,
            "adv_safety_use_rnn": False,
        }

    @property
    def params(self) -> Params:
        return super().params | {
            "adv_q": self.adv_q_train_state.params,
            "adv_vh": self.adv_vh_train_state.params,
            "adv_mu": self.adv_mu_train_state.params,
            "adv_beta": self.adv_beta_train_state.params,
        }

    def _neighbor_mask(self, graph: GraphsTuple, ego_id: Array) -> Array:
        valid = (
            (graph.receivers == ego_id)
            & (graph.senders < self.n_agents)
            & (graph.senders != ego_id)
        )
        sender = jnp.minimum(graph.senders, self.n_agents)
        counts = jnp.zeros((self.n_agents + 1,), dtype=jnp.int32)
        counts = counts.at[sender].max(valid.astype(jnp.int32))
        return counts[: self.n_agents].astype(jnp.bool_)

    def _bounded_action(self, raw_action: Action) -> Action:
        lower, upper = self._env.action_lim()
        lower = jnp.asarray(lower)
        upper = jnp.asarray(upper)
        return lower + 0.5 * (raw_action + 1.0) * (upper - lower)

    def _game_action(
        self,
        graph: GraphsTuple,
        ego_id: Array,
        background_action: Action,
        mu_params: Params,
        beta_params: Params,
        stop_beta: bool = False,
    ) -> Action:
        neighbor_mask = self._neighbor_mask(graph, ego_id)
        mu_actions = self._bounded_action(
            self.adv_mu.get_action(mu_params, graph)
        )
        ego_action = mu_actions[ego_id]
        beta_actions = self._bounded_action(
            self.adv_beta.get_action(
                beta_params, graph, ego_action, ego_id, neighbor_mask
            )
        )
        if stop_beta:
            beta_actions = lax.stop_gradient(beta_actions)
        action = jnp.where(neighbor_mask[:, None], beta_actions, background_action)
        return action.at[ego_id].set(ego_action)

    def _task_game_action(
        self,
        graph: GraphsTuple,
        task_action: Action,
        ego_id: Array,
        beta_params: Params,
    ) -> Action:
        neighbor_mask = self._neighbor_mask(graph, ego_id)
        ego_action = task_action[ego_id]
        beta_actions = self._bounded_action(
            self.adv_beta.get_action(
                beta_params, graph, ego_action, ego_id, neighbor_mask
            )
        )
        action = jnp.where(neighbor_mask[:, None], beta_actions, task_action)
        return action.at[ego_id].set(ego_action)

    def _q_value(
        self,
        q_params: Params,
        graph: GraphsTuple,
        action: Action,
        ego_id: Array,
    ) -> Array:
        return self.adv_q.get_value(
            q_params,
            graph,
            action,
            ego_id,
            self._neighbor_mask(graph, ego_id),
        )

    @ft.partial(jax.jit, static_argnums=(0,))
    def _counterfactual_task_transitions(
        self,
        beta_params: Params,
        graph: GraphsTuple,
        task_action: Action,
        ego_id: Array,
    ) -> tuple[Action, GraphsTuple]:
        game_action = jax.vmap(
            ft.partial(self._task_game_action, beta_params=beta_params)
        )(graph, task_action, ego_id)

        def next_graph(this_graph, this_action):
            next_state, _, _, _, _ = self._env.step(this_graph, this_action)
            return next_state

        return game_action, jax.vmap(next_graph)(graph, game_action)

    def _saddle_value(
        self,
        q_params: Params,
        mu_params: Params,
        beta_params: Params,
        graph: GraphsTuple,
        ego_id: Array,
    ) -> Array:
        zero_background = jnp.zeros((self.n_agents, self.action_dim))
        action = self._game_action(
            graph, ego_id, zero_background, mu_params, beta_params
        )
        return self._q_value(q_params, graph, action, ego_id)

    @staticmethod
    def _future_minimum(values: Array) -> Array:
        def body(carry, value):
            minimum = jnp.minimum(carry, value)
            return minimum, minimum

        _, minima = lax.scan(body, jnp.inf, values, reverse=True)
        return minima

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_q(
        self,
        state: TrainState,
        target_q_params: Params,
        target_mu_params: Params,
        target_beta_params: Params,
        graph: GraphsTuple,
        action: Action,
        constraint: Array,
        next_graph: GraphsTuple,
        ego_id: Array,
    ) -> tuple[TrainState, dict]:
        next_value = jax.vmap(
            ft.partial(
                self._saddle_value,
                target_q_params,
                target_mu_params,
                target_beta_params,
            )
        )(next_graph, ego_id)
        target, undiscounted = compute_adv_dgcbf_target(
            constraint, next_value, self.safety_gamma
        )
        target = lax.stop_gradient(target)
        undiscounted = lax.stop_gradient(undiscounted)

        def loss_fn(params):
            prediction = jax.vmap(ft.partial(self._q_value, params))(
                graph, action, ego_id
            )
            loss = optax.l2_loss(prediction, target).mean()
            info = {
                "adv_dgcbf/q/loss": loss,
                "adv_dgcbf/q/value_mean": prediction.mean(),
                "adv_dgcbf/q/target_mean": target.mean(),
                "adv_dgcbf/q/discounted_residual": jnp.abs(
                    prediction - target
                ).mean(),
                "adv_dgcbf/q/undiscounted_residual": jnp.abs(
                    prediction - undiscounted
                ).mean(),
            }
            return loss, info

        grad, info = jax.grad(loss_fn, has_aux=True)(state.params)
        has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        state = state.apply_gradients(grads=grad)
        return state, info | {
            "adv_dgcbf/q/grad_norm": grad_norm,
            "adv_dgcbf/q/grad_has_nan": has_nan,
        }

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_vh(
        self,
        state: TrainState,
        q_params: Params,
        mu_params: Params,
        beta_params: Params,
        graph: GraphsTuple,
        constraint: Array,
        trajectory_minimum: Array,
        ego_id: Array,
    ) -> tuple[TrainState, dict]:
        saddle = lax.stop_gradient(
            jax.vmap(
                ft.partial(
                    self._saddle_value,
                    q_params,
                    mu_params,
                    beta_params,
                )
            )(graph, ego_id)
        )

        def loss_fn(params):
            all_values = jax.vmap(ft.partial(self.adv_vh.get_value, params))(graph)
            value = jnp.take_along_axis(
                all_values, ego_id[:, None], axis=1
            ).squeeze(1)
            rollout_loss = optax.l2_loss(value, trajectory_minimum).mean()
            alignment_loss = optax.l2_loss(value, saddle).mean()
            dominance_loss = jnp.square(jax_relu(value - constraint)).mean()
            loss = rollout_loss + alignment_loss + dominance_loss
            info = {
                "adv_dgcbf/vh/loss": loss,
                "adv_dgcbf/vh/rollout_loss": rollout_loss,
                "adv_dgcbf/vh/alignment_loss": alignment_loss,
                "adv_dgcbf/vh/dominance_loss": dominance_loss,
                "adv_dgcbf/vh/value_min": value.min(),
                "adv_dgcbf/vh/value_mean": value.mean(),
                "adv_dgcbf/vh/value_max": value.max(),
            }
            return loss, info

        grad, info = jax.grad(loss_fn, has_aux=True)(state.params)
        has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        state = state.apply_gradients(grads=grad)
        return state, info | {
            "adv_dgcbf/vh/grad_norm": grad_norm,
            "adv_dgcbf/vh/grad_has_nan": has_nan,
        }

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_beta(
        self,
        state: TrainState,
        q_params: Params,
        mu_params: Params,
        graph: GraphsTuple,
        ego_id: Array,
    ) -> tuple[TrainState, dict]:
        mu_params = lax.stop_gradient(mu_params)
        q_params = lax.stop_gradient(q_params)

        def loss_fn(params):
            def value_one(this_graph, this_ego):
                zero_background = jnp.zeros((self.n_agents, self.action_dim))
                action = self._game_action(
                    this_graph,
                    this_ego,
                    zero_background,
                    mu_params,
                    params,
                )
                return self._q_value(q_params, this_graph, action, this_ego)

            value = jax.vmap(value_one)(graph, ego_id)
            return value.mean(), value.mean()

        grad, value = jax.grad(loss_fn, has_aux=True)(state.params)
        has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        state = state.apply_gradients(grads=grad)
        return state, {
            "adv_dgcbf/game/beta_value": value,
            "adv_dgcbf/game/beta_grad_norm": grad_norm,
            "adv_dgcbf/game/beta_grad_has_nan": has_nan,
        }

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_mu(
        self,
        state: TrainState,
        q_params: Params,
        beta_params: Params,
        graph: GraphsTuple,
        ego_id: Array,
    ) -> tuple[TrainState, dict]:
        beta_params = lax.stop_gradient(beta_params)
        q_params = lax.stop_gradient(q_params)

        def loss_fn(params):
            def value_one(this_graph, this_ego):
                zero_background = jnp.zeros((self.n_agents, self.action_dim))
                action = self._game_action(
                    this_graph,
                    this_ego,
                    zero_background,
                    params,
                    beta_params,
                    stop_beta=True,
                )
                return self._q_value(q_params, this_graph, action, this_ego)

            value = jax.vmap(value_one)(graph, ego_id)
            return -value.mean(), value.mean()

        grad, value = jax.grad(loss_fn, has_aux=True)(state.params)
        has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        state = state.apply_gradients(grads=grad)
        return state, {
            "adv_dgcbf/game/mu_value": value,
            "adv_dgcbf/game/mu_grad_norm": grad_norm,
            "adv_dgcbf/game/mu_grad_has_nan": has_nan,
        }

    def _update_safety(
        self,
        rollout: Rollout,
        ego_ids: Array,
        adv_sample_idx: Array,
        task_graph: GraphsTuple,
        task_game_action: Action,
        task_constraint: Array,
        task_next_graph: GraphsTuple,
        task_ego: Array,
    ) -> dict:
        b, t = rollout.dones.shape[:2]
        all_graph = jtu.tree_map(merge01, rollout.graph)
        all_next_graph = jtu.tree_map(merge01, rollout.next_graph)
        all_action = merge01(rollout.actions)
        all_ego = jnp.repeat(ego_ids, t)
        constraints = -jnp.max(rollout.costs, axis=-1)
        ego_constraints = jnp.take_along_axis(
            constraints, ego_ids[:, None, None], axis=2
        ).squeeze(2)
        trajectory_minimum = jax.vmap(self._future_minimum)(ego_constraints)
        all_constraint = merge01(ego_constraints)
        all_trajectory_minimum = merge01(trajectory_minimum)
        assert all_constraint.shape == (b * t,)

        flat_graph = jtu.tree_map(lambda value: value[adv_sample_idx], all_graph)
        flat_next_graph = jtu.tree_map(
            lambda value: value[adv_sample_idx], all_next_graph
        )
        flat_action = all_action[adv_sample_idx]
        flat_ego = all_ego[adv_sample_idx]
        flat_constraint = all_constraint[adv_sample_idx]
        flat_trajectory_minimum = all_trajectory_minimum[adv_sample_idx]

        q_graph = jtu.tree_map(
            lambda adv, task: jnp.concatenate([adv, task], axis=0),
            flat_graph,
            task_graph,
        )
        q_action = jnp.concatenate([flat_action, task_game_action], axis=0)
        q_constraint = jnp.concatenate([flat_constraint, task_constraint], axis=0)
        q_next_graph = jtu.tree_map(
            lambda adv, task: jnp.concatenate([adv, task], axis=0),
            flat_next_graph,
            task_next_graph,
        )
        q_ego = jnp.concatenate([flat_ego, task_ego], axis=0)

        self.adv_q_train_state, q_info = self._update_q(
            self.adv_q_train_state,
            self.adv_q_target_params,
            self.adv_mu_target_params,
            self.adv_beta_target_params,
            q_graph,
            q_action,
            q_constraint,
            q_next_graph,
            q_ego,
        )
        self.adv_vh_train_state, vh_info = self._update_vh(
            self.adv_vh_train_state,
            self.adv_q_train_state.params,
            self.adv_mu_train_state.params,
            self.adv_beta_train_state.params,
            flat_graph,
            flat_constraint,
            flat_trajectory_minimum,
            flat_ego,
        )
        game_info = {}
        for _ in range(self.adv_inner_steps):
            self.adv_beta_train_state, beta_info = self._update_beta(
                self.adv_beta_train_state,
                self.adv_q_train_state.params,
                self.adv_mu_train_state.params,
                flat_graph,
                flat_ego,
            )
            self.adv_mu_train_state, mu_info = self._update_mu(
                self.adv_mu_train_state,
                self.adv_q_train_state.params,
                self.adv_beta_train_state.params,
                flat_graph,
                flat_ego,
            )
            game_info = beta_info | mu_info
        data_info = {
            "adv_dgcbf/data/adversarial_rollout_samples": jnp.asarray(b * t),
            "adv_dgcbf/data/adversarial_samples": jnp.asarray(flat_constraint.size),
            "adv_dgcbf/data/task_action_samples": jnp.asarray(task_constraint.size),
            "adv_dgcbf/data/task_action_fraction": task_constraint.size
            / (flat_constraint.size + task_constraint.size),
        }
        return q_info | vh_info | game_info | data_info

    def _all_actor_safety_values(
        self,
        graph: GraphsTuple,
        task_action: Action,
        ego_ids: Array,
        q_params: Params,
        vh_params: Params,
        beta_params: Params,
    ) -> tuple[Array, Array]:
        all_values = self.adv_vh.get_value(vh_params, graph)

        def one_ego(ego_id):
            task_game_action = self._task_game_action(
                graph, task_action, ego_id, beta_params
            )
            return self._q_value(
                q_params, graph, task_game_action, ego_id
            )

        task_value = jax.vmap(one_ego)(ego_ids)
        return all_values[ego_ids], task_value

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_task(
        self,
        Vl_train_state: TrainState,
        policy_train_state: TrainState,
        q_params: Params,
        vh_params: Params,
        beta_params: Params,
        rollout: Rollout,
        actor_sample_idx: Array,
        batch_idx: Array,
        rnn_chunk_ids: Array,
        step: Array,
    ) -> tuple[TrainState, TrainState, dict]:
        b, t, _, _ = rollout.actions.shape
        bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
            ft.partial(
                self.scan_Vl,
                init_Vl_rnn_state=self.init_Vl_rnn_state,
                Vl_params=Vl_train_state.params,
            )
        )(rollout)

        def final_value(graph, rnn_state):
            value, _ = self.Vl.get_value(
                Vl_train_state.params, tree_index(graph, -1), rnn_state
            )
            return value.squeeze(0).squeeze(0)

        final_vl = jax.vmap(final_value)(rollout.next_graph, final_Vl_rnn_states)
        values = jnp.concatenate([bT_Vl, final_vl[:, None]], axis=1)
        safety_placeholder = values[:, :, None, None].repeat(
            self.n_agents, axis=-2
        ).repeat(rollout.costs.shape[-1], axis=-1)
        _, task_q = jax.vmap(
            ft.partial(
                compute_dec_ocp_gae,
                disc_gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        )(
            Tah_hs=rollout.costs,
            T_l=-rollout.rewards,
            Tp1ah_Vh=safety_placeholder,
            Tp1_Vl=values,
        )
        task_advantage = task_q - bT_Vl
        task_advantage = (
            task_advantage - task_advantage.mean(axis=1, keepdims=True)
        ) / (task_advantage.std(axis=1, keepdims=True) + 1e-8)
        task_advantage = -task_advantage[:, :, None].repeat(
            self.n_agents, axis=-1
        )

        flat_graph = jtu.tree_map(merge01, rollout.graph)
        flat_action = merge01(rollout.actions)
        n_transition = b * t
        actor_graph = jtu.tree_map(
            lambda value: value[actor_sample_idx], flat_graph
        )
        actor_action = flat_action[actor_sample_idx]
        actor_ego_ids = (
            actor_sample_idx[:, None]
            + jnp.arange(self.adv_actor_egos_per_sample, dtype=jnp.int32)[
                None, :
            ]
            + step
        ) % self.n_agents
        flat_value, flat_task_value = jax.vmap(
            ft.partial(
                self._all_actor_safety_values,
                q_params=lax.stop_gradient(q_params),
                vh_params=lax.stop_gradient(vh_params),
                beta_params=lax.stop_gradient(beta_params),
            )
        )(actor_graph, actor_action, actor_ego_ids)
        flat_task_advantage = task_advantage.reshape(
            (n_transition, self.n_agents)
        )
        actor_task_advantage = flat_task_advantage[actor_sample_idx]
        selected_task_advantage = jnp.take_along_axis(
            actor_task_advantage, actor_ego_ids, axis=1
        )
        selected_advantage, violation, safe = mix_adv_dgcbf_advantages(
            selected_task_advantage,
            flat_value,
            flat_task_value,
            self.cbf_schedule_fn(step),
            self.cbf_kappa,
            self.cbf_eps,
        )
        row_ids = actor_sample_idx[:, None]
        mixed_advantage = flat_task_advantage.at[
            row_ids, actor_ego_ids
        ].set(selected_advantage)
        mixed_advantage = mixed_advantage.reshape(
            (b, t, self.n_agents)
        )

        def update_fn(carry, idx):
            vl_state, actor_state = carry
            rollout_batch = jtu.tree_map(lambda value: value[idx], rollout)
            vl_state, vl_info = self.update_Vl(
                vl_state,
                rollout_batch,
                task_q[idx],
                bT_Vl_rnn_states[idx],
                rnn_chunk_ids,
            )
            actor_state, actor_info = self.update_policy(
                actor_state,
                rollout_batch,
                mixed_advantage[idx],
                rnn_chunk_ids,
            )
            return (vl_state, actor_state), vl_info | actor_info

        (Vl_train_state, policy_train_state), info = lax.scan(
            update_fn,
            (Vl_train_state, policy_train_state),
            batch_idx,
        )
        info = jtu.tree_map(lambda value: value[-1], info)
        safety_update_ratio = (~safe).sum() / (n_transition * self.n_agents)
        info |= {
            "adv_dgcbf/actor/residual_mean": violation.mean(),
            "adv_dgcbf/actor/residual_max": violation.max(),
            "adv_dgcbf/actor/safe_ratio": safe.mean(),
            "adv_dgcbf/actor/task_update_ratio": 1.0 - safety_update_ratio,
            "adv_dgcbf/actor/safety_update_ratio": safety_update_ratio,
            "adv_dgcbf/actor/value_mean": flat_value.mean(),
            "adv_dgcbf/actor/task_action_value_mean": flat_task_value.mean(),
            "adv_dgcbf/actor/cbf_weight": self.cbf_schedule_fn(step),
            "adv_dgcbf/data/actor_ego_evaluations": jnp.asarray(safe.size),
            "adv_dgcbf/data/actor_ego_fraction": safe.size
            / (n_transition * self.n_agents),
        }
        return Vl_train_state, policy_train_state, info

    def _update_targets(self):
        self.adv_q_target_params = optax.incremental_update(
            self.adv_q_train_state.params,
            self.adv_q_target_params,
            self.adv_target_tau,
        )
        self.adv_mu_target_params = optax.incremental_update(
            self.adv_mu_train_state.params,
            self.adv_mu_target_params,
            self.adv_target_tau,
        )
        self.adv_beta_target_params = optax.incremental_update(
            self.adv_beta_train_state.params,
            self.adv_beta_target_params,
            self.adv_target_tau,
        )

    def update(self, rollout: Rollout, step: int) -> dict:
        (
            rollout_key,
            adv_sample_key,
            task_sample_key,
            actor_sample_key,
            self.key,
        ) = jr.split(self.key, 5)
        n_env = rollout.dones.shape[0]
        time_horizon = rollout.dones.shape[1]
        n_transition = n_env * time_horizon
        adv_n_env = min(
            n_env,
            max(1, (self.adv_batch_size + time_horizon - 1) // time_horizon),
        )
        adv_ego_ids = (
            jnp.arange(adv_n_env, dtype=jnp.int32) + jnp.asarray(step)
        ) % self.n_agents
        task_ego_ids = (
            jnp.arange(n_env, dtype=jnp.int32) + jnp.asarray(step)
        ) % self.n_agents
        adv_rollout = self.adv_rollout_fn(
            self.params, jr.split(rollout_key, adv_n_env), adv_ego_ids
        )

        n_adv_transition = adv_n_env * time_horizon
        adv_sample_size = min(self.adv_batch_size, n_adv_transition)
        actor_sample_size = min(self.adv_batch_size, n_transition)
        task_sample_size = min(
            n_transition,
            max(1, round(adv_sample_size * self.adv_task_sample_ratio)),
        )
        adv_sample_idx = sample_transition_indices(
            adv_sample_key, n_adv_transition, adv_sample_size
        )
        actor_sample_idx = sample_transition_indices(
            actor_sample_key, n_transition, actor_sample_size
        )
        task_sample_idx = sample_transition_indices(
            task_sample_key, n_transition, task_sample_size
        )

        all_task_graph = jtu.tree_map(merge01, rollout.graph)
        all_task_action = merge01(rollout.actions)
        all_task_ego = jnp.repeat(task_ego_ids, time_horizon)
        task_graph = jtu.tree_map(
            lambda value: value[task_sample_idx], all_task_graph
        )
        task_action = all_task_action[task_sample_idx]
        task_ego = all_task_ego[task_sample_idx]
        task_game_action, task_next_graph = self._counterfactual_task_transitions(
            self.adv_beta_train_state.params,
            task_graph,
            task_action,
            task_ego,
        )
        task_constraints = -jnp.max(rollout.costs, axis=-1)
        task_constraint = jnp.take_along_axis(
            task_constraints, task_ego_ids[:, None, None], axis=2
        ).squeeze(2)
        task_constraint = merge01(task_constraint)[task_sample_idx]

        rollout = rollout._replace(
            graph=rollout.graph._replace(env_states=None),
            next_graph=rollout.next_graph._replace(env_states=None),
        )
        adv_rollout = adv_rollout._replace(
            graph=adv_rollout.graph._replace(env_states=None),
            next_graph=adv_rollout.next_graph._replace(env_states=None),
        )
        task_graph = task_graph._replace(env_states=None)
        task_next_graph = task_next_graph._replace(env_states=None)
        assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
        update_info = {}
        for _ in range(self.epoch_ppo):
            idx = np.arange(rollout.dones.shape[0])
            np.random.shuffle(idx)
            rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
            rnn_chunk_ids = jnp.array(
                jnp.array_split(
                    rnn_chunk_ids,
                    rollout.dones.shape[1] // self.rnn_step,
                )
            )
            batch_idx = jnp.array(
                jnp.array_split(
                    idx,
                    idx.shape[0]
                    // (self.batch_size // rollout.dones.shape[1]),
                )
            )
            safety_info = self._update_safety(
                adv_rollout,
                adv_ego_ids,
                adv_sample_idx,
                task_graph,
                task_game_action,
                task_constraint,
                task_next_graph,
                task_ego,
            )
            (
                self.Vl_train_state,
                self.policy_train_state,
                task_info,
            ) = self._update_task(
                self.Vl_train_state,
                self.policy_train_state,
                self.adv_q_train_state.params,
                self.adv_vh_train_state.params,
                self.adv_beta_train_state.params,
                rollout,
                actor_sample_idx,
                batch_idx,
                rnn_chunk_ids,
                jnp.asarray(step),
            )
            update_info = task_info | safety_info
        self._update_targets()
        return update_info

    def save(self, save_dir: str, step: int | str):
        super().save(save_dir, step)
        model_dir = os.path.join(save_dir, str(step))

        def train_state_payload(state):
            return {
                "params": state.params,
                "step": state.step,
                "opt_state": state.opt_state,
            }

        with open(os.path.join(model_dir, "adv_dgcbf.pkl"), "wb") as file:
            pickle.dump(
                {
                    "format_version": 1,
                    "q": train_state_payload(self.adv_q_train_state),
                    "vh": train_state_payload(self.adv_vh_train_state),
                    "mu": train_state_payload(self.adv_mu_train_state),
                    "beta": train_state_payload(self.adv_beta_train_state),
                    "q_target": self.adv_q_target_params,
                    "mu_target": self.adv_mu_target_params,
                    "beta_target": self.adv_beta_target_params,
                },
                file,
            )

    def load(self, load_dir: str, step: int | str):
        super().load(load_dir, step)
        path = os.path.join(load_dir, str(step), "adv_dgcbf.pkl")
        with open(path, "rb") as file:
            state = pickle.load(file)
        if state.get("format_version") != 1:
            raise ValueError("unsupported adversarial DGCBF checkpoint format")

        def restore_train_state(current, payload):
            return current.replace(
                params=payload["params"],
                step=payload["step"],
                opt_state=payload["opt_state"],
            )

        self.adv_q_train_state = restore_train_state(
            self.adv_q_train_state, state["q"]
        )
        self.adv_vh_train_state = restore_train_state(
            self.adv_vh_train_state, state["vh"]
        )
        self.adv_mu_train_state = restore_train_state(
            self.adv_mu_train_state, state["mu"]
        )
        self.adv_beta_train_state = restore_train_state(
            self.adv_beta_train_state, state["beta"]
        )
        self.adv_q_target_params = state["q_target"]
        self.adv_mu_target_params = state["mu_target"]
        self.adv_beta_target_params = state["beta_target"]
