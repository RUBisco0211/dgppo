"""Distributed Graph-HJ critic trained with the Deep-QP loss."""

import functools as ft
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.training.train_state import TrainState

from ...nn.gnn import GraphTransformerGNN
from ...nn.mlp import MLP
from ...nn.utils import default_nn_init
from ...trainer.data import SafetyBatch
from ...utils.graph import GraphsTuple
from ...utils.typing import Action, Array, Params, PRNGKey


@dataclass(frozen=True)
class DeepQPSafetyConfig:
    gnn_layers: int = 1
    gnn_out_dim: int = 64
    hidden_dim: int = 256
    hidden_layers: int = 2
    n_heads: int = 3
    learning_rate: float = 3e-4
    learning_rate_final: float = 3e-6
    max_grad_norm: float = 2.0
    target_tau: float = 0.005
    dt: float = 0.03
    lambda_init: float = 0.1 / 0.03
    lambda_final: float = 0.0001 / 0.03
    lambda_decay_steps: int = 1_000_000
    lambda_decay_start: int = 0
    terminal_value: float = -0.05
    value_offset: float = 0.05
    constraint_scale: float = 0.5
    value_loss_weight: float = 1.0
    derivative_loss_weight: float = 1.0

    def __post_init__(self):
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.constraint_scale <= 0.0:
            raise ValueError("constraint_scale must be positive")
        if self.learning_rate <= 0.0 or self.learning_rate_final <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        if self.lambda_decay_steps <= 0:
            raise ValueError("lambda_decay_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyNetworkOutput(NamedTuple):
    value1: Array
    value2: Array
    coefficient: Array
    scalar_deviation: Array
    action_mask: Array


class SafetyCertificate(NamedTuple):
    value: Array
    value1: Array
    value2: Array
    coefficient: Array
    max_derivative: Array
    constraint: Array
    action_mask: Array


class DeepQPSafetyTrainState(NamedTuple):
    online: TrainState
    target_params: Params


class _SafetyHead(nn.Module):
    hidden_dim: int
    hidden_layers: int
    output_dim: int

    @nn.compact
    def __call__(self, embedding: Array) -> Array:
        hidden_sizes = (self.hidden_dim,) * self.hidden_layers
        x = MLP(
            hid_sizes=hidden_sizes,
            act=nn.elu,
            act_final=True,
            use_layernorm=True,
        )(embedding)
        return nn.Dense(self.output_dim, kernel_init=default_nn_init())(x)


class DeepQPSafetyNet(nn.Module):
    """DGPPO-style shared local GNN with Deep-QP structured output heads."""

    action_dim: int
    gnn_layers: int = 1
    gnn_out_dim: int = 64
    hidden_dim: int = 256
    hidden_layers: int = 2
    n_heads: int = 3
    terminal_value: float = -0.05
    value_offset: float = 0.05

    @nn.compact
    def __call__(
            self,
            graph: GraphsTuple,
            constraint: Array,
            n_agents: int,
            clip_value: bool = True,
    ) -> SafetyNetworkOutput:
        embedding = GraphTransformerGNN(
            msg_dim=32,
            out_dim=self.gnn_out_dim,
            n_heads=self.n_heads,
            n_layers=self.gnn_layers,
            name="LocalSafetyGNN",
        )(graph, node_type=0, n_type=n_agents)
        assert embedding.shape[0] == n_agents

        receiver_embedding = jnp.broadcast_to(
            embedding[:, None, :], (n_agents, n_agents, embedding.shape[-1])
        )
        owner_embedding = jnp.broadcast_to(
            embedding[None, :, :], (n_agents, n_agents, embedding.shape[-1])
        )
        pair_embedding = jnp.concatenate(
            [receiver_embedding, owner_embedding], axis=-1
        )

        valid_edge = (graph.receivers < n_agents) & (graph.senders < n_agents)
        receiver = jnp.clip(graph.receivers, 0, n_agents - 1)
        sender = jnp.clip(graph.senders, 0, n_agents - 1)
        edge_count = jnp.zeros((n_agents, n_agents), dtype=jnp.int32)
        edge_count = edge_count.at[receiver, sender].add(valid_edge.astype(jnp.int32))
        action_mask = edge_count > 0
        action_mask = action_mask | jnp.eye(n_agents, dtype=bool)

        value_raw1 = _SafetyHead(
            self.hidden_dim, self.hidden_layers, 1, name="ValueHead1"
        )(embedding).squeeze(-1)
        value_raw2 = _SafetyHead(
            self.hidden_dim, self.hidden_layers, 1, name="ValueHead2"
        )(embedding).squeeze(-1)
        coefficient = _SafetyHead(
            self.hidden_dim, self.hidden_layers, self.action_dim, name="PairCoefficientHead"
        )(pair_embedding)
        coefficient = jnp.where(action_mask[..., None], coefficient, 0.0)
        scalar_raw = _SafetyHead(
            self.hidden_dim, self.hidden_layers, 1, name="ScalarHead"
        )(embedding).squeeze(-1)

        value1 = constraint - (nn.softplus(value_raw1) - self.value_offset)
        value2 = constraint - (nn.softplus(value_raw2) - self.value_offset)
        scalar_deviation = nn.softplus(scalar_raw) - self.value_offset
        if clip_value:
            value1 = jnp.minimum(jnp.maximum(value1, self.terminal_value), constraint)
            value2 = jnp.minimum(jnp.maximum(value2, self.terminal_value), constraint)
            scalar_deviation = jnp.maximum(scalar_deviation, 0.0)

        return SafetyNetworkOutput(
            value1, value2, coefficient, scalar_deviation, action_mask
        )


def safety_lambda_at(config: DeepQPSafetyConfig, step: Array | int) -> Array:
    elapsed = jnp.maximum(jnp.asarray(step) - config.lambda_decay_start, 0)
    ratio = jnp.minimum(elapsed / config.lambda_decay_steps, 1.0)
    decay = (1.0 - ratio) ** 5
    return config.lambda_final + (config.lambda_init - config.lambda_final) * decay


def _polyak_update(online: Params, target: Params, tau: float) -> Params:
    return jtu.tree_map(lambda new, old: tau * new + (1.0 - tau) * old, online, target)


class GraphHJSafetyCritic:
    """Pretrained local HJ value and local-joint-action derivative model."""

    def __init__(
            self,
            action_dim: int,
            n_agents: int,
            action_lower: Array | float,
            action_upper: Array | float,
            config: DeepQPSafetyConfig | None = None,
            node_feature_mask: Array | None = None,
    ):
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.action_lower = jnp.asarray(action_lower)
        self.action_upper = jnp.asarray(action_upper)
        self.node_feature_mask = (
            None if node_feature_mask is None else jnp.asarray(node_feature_mask, dtype=bool)
        )
        self.config = DeepQPSafetyConfig() if config is None else config
        self.network = DeepQPSafetyNet(
            action_dim=action_dim,
            gnn_layers=self.config.gnn_layers,
            gnn_out_dim=self.config.gnn_out_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            n_heads=self.config.n_heads,
            terminal_value=self.config.terminal_value,
            value_offset=self.config.value_offset,
        )

    def initialize(self, key: PRNGKey, graph: GraphsTuple) -> DeepQPSafetyTrainState:
        constraint = jnp.zeros((self.n_agents,), dtype=graph.nodes.dtype)
        params = self.network.init(key, graph, constraint, self.n_agents, True)
        learning_rate = optax.polynomial_schedule(
            init_value=self.config.learning_rate,
            end_value=self.config.learning_rate_final,
            power=5.0,
            transition_steps=self.config.lambda_decay_steps,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(learning_rate),
        )
        online = TrainState.create(apply_fn=self.network.apply, params=params, tx=optimizer)
        return DeepQPSafetyTrainState(online=online, target_params=params)

    def save_checkpoint(
            self,
            state: DeepQPSafetyTrainState,
            path: str | Path,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "format_version": 2,
            "derivative_head": "local_joint_pair",
            "online_params": state.online.params,
            "target_params": state.target_params,
            "opt_state": state.online.opt_state,
            "step": state.online.step,
            "config": self.config.to_dict(),
            "n_agents": self.n_agents,
            "action_dim": self.action_dim,
            "metadata": {} if metadata is None else metadata,
        }
        with Path(path).open("wb") as file:
            pickle.dump(payload, file)

    def load_checkpoint(
            self,
            state: DeepQPSafetyTrainState,
            path: str | Path,
            expected_metadata: dict[str, Any] | None = None,
    ) -> DeepQPSafetyTrainState:
        path = Path(path)
        if path.is_dir():
            path = path / "deep_qp_safety.pkl"
        with path.open("rb") as file:
            payload = pickle.load(file)
        if payload.get("format_version") != 2:
            raise ValueError(
                "legacy safety checkpoint uses the ego-only derivative head; "
                "retrain the Graph-HJ critic with the local-joint pair head"
            )
        if payload.get("derivative_head") != "local_joint_pair":
            raise ValueError("unsupported safety checkpoint derivative head")
        if payload.get("action_dim", self.action_dim) != self.action_dim:
            raise ValueError("safety checkpoint action_dim does not match the critic")
        saved_config = payload.get("config", {})
        saved_scale = saved_config.get("constraint_scale")
        if saved_scale is not None and saved_scale != self.config.constraint_scale:
            raise ValueError("safety checkpoint constraint_scale does not match the critic")
        structural_fields = (
            "gnn_layers", "gnn_out_dim", "hidden_dim", "hidden_layers", "n_heads",
            "terminal_value", "value_offset", "dt", "lambda_init", "lambda_final",
        )
        for field in structural_fields:
            if field in saved_config and saved_config[field] != getattr(self.config, field):
                raise ValueError(f"safety checkpoint {field} does not match the critic")
        saved_metadata = payload.get("metadata", {})
        if expected_metadata is not None:
            for field, expected in expected_metadata.items():
                if field not in saved_metadata:
                    raise ValueError(f"safety checkpoint metadata {field} is missing")
                if saved_metadata[field] != expected:
                    raise ValueError(f"safety checkpoint metadata {field} does not match")
        online = state.online.replace(
            params=payload["online_params"],
            opt_state=payload.get("opt_state", state.online.opt_state),
            step=payload.get("step", state.online.step),
        )
        return DeepQPSafetyTrainState(online=online, target_params=payload["target_params"])

    def _network_output(
            self,
            params: Params,
            graph: GraphsTuple,
            normalized_constraint: Array,
            *,
            clip_value: bool,
    ) -> SafetyNetworkOutput:
        graph = self._local_safety_graph(graph)
        return self.network.apply(params, graph, normalized_constraint, self.n_agents, clip_value)

    def _local_safety_graph(self, graph: GraphsTuple) -> GraphsTuple:
        """Remove task-goal messages and centralized state from a single graph."""
        goal_node = graph.node_type == 1
        pad_id = graph.nodes.shape[0] - 1
        touches_goal = goal_node[graph.senders] | goal_node[graph.receivers]
        nodes = jnp.where(goal_node[:, None], 0.0, graph.nodes)
        if self.node_feature_mask is not None:
            if self.node_feature_mask.shape != (graph.nodes.shape[-1],):
                raise ValueError("node_feature_mask does not match graph node features")
            nodes = jnp.where(self.node_feature_mask[None, :], nodes, 0.0)
        edges = jnp.where(touches_goal[:, None], 0.0, graph.edges)
        senders = jnp.where(touches_goal, pad_id, graph.senders)
        receivers = jnp.where(touches_goal, pad_id, graph.receivers)
        return graph._replace(
            nodes=nodes,
            edges=edges,
            states=jnp.zeros_like(graph.states),
            senders=senders,
            receivers=receivers,
            env_states=None,
            connectivity=None,
        )

    def certify(
            self,
            params: Params,
            graph: GraphsTuple,
            constraint: Array,
            safety_lambda: Array | float,
    ) -> SafetyCertificate:
        normalized_constraint = constraint / self.config.constraint_scale
        output = self._network_output(
            params, graph, normalized_constraint, clip_value=True
        )
        value = jnp.minimum(output.value1, output.value2)
        max_derivative = output.scalar_deviation - safety_lambda * (normalized_constraint - value)
        return SafetyCertificate(
            value=value,
            value1=output.value1,
            value2=output.value2,
            coefficient=output.coefficient,
            max_derivative=max_derivative,
            constraint=normalized_constraint,
            action_mask=output.action_mask,
        )

    def _support(self, coefficient: Array) -> Array:
        return jnp.sum(
            jnp.where(
                coefficient >= 0.0,
                coefficient * self.action_upper,
                coefficient * self.action_lower,
            ),
            axis=(-2, -1),
        )

    def cbf_residual(
            self,
            certificate: SafetyCertificate,
            joint_action: Action,
            alpha: float,
            margin: float = 0.0,
    ) -> Array:
        """Evaluate ``dot(V_i) + alpha * (V_i - margin)`` for every owner."""
        action_term = jnp.einsum(
            "...ijd,...jd->...i", certificate.coefficient, joint_action
        )
        derivative = (
            action_term
            - self._support(certificate.coefficient)
            + certificate.max_derivative
        )
        normalized_margin = margin / self.config.constraint_scale
        return derivative + alpha * (certificate.value - normalized_margin)

    def _batched_output(
            self,
            params: Params,
            graphs: GraphsTuple,
            normalized_constraints: Array,
            *,
            clip_value: bool,
    ) -> SafetyNetworkOutput:
        apply_one = ft.partial(self._network_output, params, clip_value=clip_value)
        return jax.vmap(apply_one)(graphs, normalized_constraints)

    def loss(
            self,
            online_params: Params,
            target_params: Params,
            batch: SafetyBatch,
            safety_lambda: Array,
    ) -> tuple[Array, dict[str, Array]]:
        config = self.config
        constraint = batch.constraints / config.constraint_scale
        next_constraint = batch.next_constraints / config.constraint_scale
        next_constraint = jnp.maximum(next_constraint, config.terminal_value)

        online = self._batched_output(
            online_params, batch.graph, constraint, clip_value=False
        )
        target = self._batched_output(
            target_params, batch.graph, constraint, clip_value=True
        )
        target_next = self._batched_output(
            target_params, batch.next_graph, next_constraint, clip_value=True
        )

        target_value = target.value1
        target_next_value = jnp.minimum(target_next.value1, target_next.value2)
        gap = constraint - target_value
        next_gap = next_constraint - target_next_value
        scalar_lower_bound = safety_lambda * gap
        next_scalar_lower_bound = safety_lambda * next_gap

        next_close_enough = (next_gap <= 0.0) & (next_constraint > config.terminal_value)
        target_next_max_derivative = jnp.where(
            next_close_enough,
            target_next.scalar_deviation - next_scalar_lower_bound,
            -next_scalar_lower_bound,
        )
        lambda_dt = safety_lambda * config.dt
        discount = jnp.exp(-lambda_dt)
        target_next_bellman_value = jnp.minimum(
            next_constraint,
            discount * (target_next_value + config.dt * target_next_max_derivative)
            + lambda_dt * next_constraint,
        )
        target_next_bellman_value = jnp.minimum(
            jnp.maximum(target_next_bellman_value, config.terminal_value), next_constraint
        )
        done = batch.dones[..., None]
        target_next_bellman_value = jnp.where(done, next_constraint, target_next_bellman_value)

        target_coefficient_term = (
            jnp.einsum("...ijd,...jd->...i", target.coefficient, batch.actions)
            - self._support(target.coefficient)
        )
        target_max_derivative = target.scalar_deviation - scalar_lower_bound
        rhs_derivative = (target_next_bellman_value - target_value) / config.dt
        integral_exp_constraint = lambda_dt * constraint
        derivative_hj_term = (
            config.dt * (
                target_coefficient_term + target_max_derivative - safety_lambda * target_value
            )
            + integral_exp_constraint
        )
        q_part_for_value = jnp.minimum(derivative_hj_term, 0.0)
        rhs_value = (
            jnp.minimum(constraint, discount * target_next_value + integral_exp_constraint)
            - q_part_for_value
        )
        rhs_value = jax.lax.stop_gradient(rhs_value)
        rhs_derivative = jax.lax.stop_gradient(rhs_derivative)
        target_coefficient_term = jax.lax.stop_gradient(target_coefficient_term)
        target_max_derivative = jax.lax.stop_gradient(target_max_derivative)

        value_loss = (
            optax.l2_loss(online.value1, rhs_value).mean()
            + optax.l2_loss(online.value2, rhs_value).mean()
        ) / config.dt

        online_coefficient_term = (
            jnp.einsum("...ijd,...jd->...i", online.coefficient, batch.actions)
            - self._support(online.coefficient)
        )
        online_max_derivative = online.scalar_deviation - jax.lax.stop_gradient(scalar_lower_bound)
        rhs_scalar = rhs_derivative - target_coefficient_term
        coefficient_loss = optax.l2_loss(
            online_coefficient_term + target_max_derivative, rhs_derivative
        ).mean() * config.dt
        scalar_loss = optax.l2_loss(online_max_derivative, rhs_scalar).mean() * config.dt
        derivative_loss = coefficient_loss + scalar_loss
        total_loss = (
            config.value_loss_weight * value_loss
            + config.derivative_loss_weight * derivative_loss
        )

        min_online_value = jnp.minimum(online.value1, online.value2)
        info = {
            "safety/loss": total_loss,
            "safety/value_loss": value_loss,
            "safety/derivative_loss": derivative_loss,
            "safety/coefficient_loss": coefficient_loss,
            "safety/scalar_loss": scalar_loss,
            "safety/value_bound_violation": jnp.mean(min_online_value > constraint),
            "safety/value_mean": jnp.mean(min_online_value),
            "safety/target_online_gap": jnp.mean(jnp.abs(min_online_value - target_value)),
            "safety/derivative_residual": jnp.mean(jnp.abs(
                online_coefficient_term + online_max_derivative - rhs_derivative
            )),
            "safety/coefficient_norm": jnp.linalg.norm(
                online.coefficient, axis=(-2, -1)
            ).mean(),
            "safety/lambda": safety_lambda,
        }
        return total_loss, info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update(
            self,
            state: DeepQPSafetyTrainState,
            batch: SafetyBatch,
    ) -> tuple[DeepQPSafetyTrainState, dict[str, Array]]:
        safety_lambda = safety_lambda_at(self.config, state.online.step)

        def loss_fn(params):
            return self.loss(params, state.target_params, batch, safety_lambda)

        (loss, info), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.online.params)
        gradients_finite = jnp.stack([
            jnp.all(jnp.isfinite(x)) for x in jtu.tree_leaves(gradients)
        ]).all()
        update_finite = gradients_finite & jnp.isfinite(loss)
        candidate_online = state.online.apply_gradients(grads=gradients)
        online = jax.lax.cond(
            update_finite,
            lambda _: candidate_online,
            lambda _: state.online,
            operand=None,
        )
        candidate_target = _polyak_update(
            online.params, state.target_params, self.config.target_tau
        )
        target_params = jax.lax.cond(
            update_finite,
            lambda _: candidate_target,
            lambda _: state.target_params,
            operand=None,
        )
        grad_norm = optax.global_norm(gradients)
        info = info | {
            "safety/grad_norm": grad_norm,
            "safety/has_nan": (~update_finite).astype(jnp.float32),
            "safety/update_applied": update_finite.astype(jnp.float32),
        }
        return DeepQPSafetyTrainState(online=online, target_params=target_params), info
