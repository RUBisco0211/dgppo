"""Feed-forward shared networks for the adversarial DGCBF safety game."""

from __future__ import annotations

import functools as ft

import flax.linen as nn
import jax.numpy as jnp

from ...nn.gnn import GraphTransformerGNN
from ...nn.mlp import MLP
from ...nn.utils import default_nn_init
from ...utils.graph import GraphsTuple
from ...utils.typing import Action, Array, Params


def _backbone(layers: int, hidden_dim: int):
    return ft.partial(
        GraphTransformerGNN,
        msg_dim=32,
        out_dim=hidden_dim,
        n_heads=3,
        n_layers=layers,
    )


def _augment_agent_nodes(
    graph: GraphsTuple,
    n_agents: int,
    features: Array,
) -> GraphsTuple:
    padding = jnp.zeros(
        (graph.nodes.shape[0] - n_agents, features.shape[-1]),
        dtype=features.dtype,
    )
    node_features = jnp.concatenate([features, padding], axis=0)
    return graph._replace(nodes=jnp.concatenate([graph.nodes, node_features], axis=-1))


class _SharedSafetyActorNet(nn.Module):
    action_dim: int
    gnn_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, n_agents: int) -> Action:
        embedding = _backbone(self.gnn_layers, self.hidden_dim)()(
            graph, node_type=0, n_type=n_agents
        )
        hidden = MLP(
            hid_sizes=(self.hidden_dim, self.hidden_dim),
            act=nn.relu,
            act_final=True,
            name="SafetyActorHead",
        )(embedding)
        return jnp.tanh(
            nn.Dense(
                self.action_dim,
                kernel_init=default_nn_init(),
                name="SafetyActorOutput",
            )(hidden)
        )


class SharedSafetyActor:
    def __init__(
        self,
        action_dim: int,
        n_agents: int,
        gnn_layers: int,
        hidden_dim: int,
    ):
        self.n_agents = n_agents
        self.net = _SharedSafetyActorNet(action_dim, gnn_layers, hidden_dim)

    def initialize(self, key: Array, graph: GraphsTuple) -> Params:
        return self.net.init(key, graph, self.n_agents)

    def get_action(self, params: Params, graph: GraphsTuple) -> Action:
        return self.net.apply(params, graph, self.n_agents)


class _AdversaryNet(nn.Module):
    action_dim: int
    gnn_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(
        self,
        graph: GraphsTuple,
        ego_action: Action,
        ego_id: Array,
        neighbor_mask: Array,
        n_agents: int,
    ) -> Action:
        ego_mask = jnp.arange(n_agents) == ego_id
        action_context = jnp.zeros((n_agents, self.action_dim), dtype=ego_action.dtype)
        action_context = action_context.at[ego_id].set(ego_action)
        roles = jnp.stack([ego_mask, neighbor_mask], axis=-1).astype(
            graph.nodes.dtype
        )
        augmented = _augment_agent_nodes(
            graph,
            n_agents,
            jnp.concatenate([action_context, roles], axis=-1),
        )
        embedding = _backbone(self.gnn_layers, self.hidden_dim)()(
            augmented, node_type=0, n_type=n_agents
        )
        hidden = MLP(
            hid_sizes=(self.hidden_dim, self.hidden_dim),
            act=nn.relu,
            act_final=True,
            name="AdversaryHead",
        )(embedding)
        return jnp.tanh(
            nn.Dense(
                self.action_dim,
                kernel_init=default_nn_init(),
                name="AdversaryOutput",
            )(hidden)
        )


class SharedAdversary:
    def __init__(
        self,
        action_dim: int,
        n_agents: int,
        gnn_layers: int,
        hidden_dim: int,
    ):
        self.n_agents = n_agents
        self.net = _AdversaryNet(action_dim, gnn_layers, hidden_dim)

    def initialize(
        self,
        key: Array,
        graph: GraphsTuple,
        ego_action: Action,
        ego_id: Array,
        neighbor_mask: Array,
    ) -> Params:
        return self.net.init(
            key, graph, ego_action, ego_id, neighbor_mask, self.n_agents
        )

    def get_action(
        self,
        params: Params,
        graph: GraphsTuple,
        ego_action: Action,
        ego_id: Array,
        neighbor_mask: Array,
    ) -> Action:
        return self.net.apply(
            params, graph, ego_action, ego_id, neighbor_mask, self.n_agents
        )


class _ActionConditionedSafetyQNet(nn.Module):
    action_dim: int
    gnn_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(
        self,
        graph: GraphsTuple,
        joint_action: Action,
        ego_id: Array,
        neighbor_mask: Array,
        n_agents: int,
    ) -> Array:
        ego_mask = jnp.arange(n_agents) == ego_id
        relevant = ego_mask | neighbor_mask
        action_features = joint_action * relevant[:, None]
        roles = jnp.stack([ego_mask, neighbor_mask], axis=-1).astype(
            graph.nodes.dtype
        )
        augmented = _augment_agent_nodes(
            graph,
            n_agents,
            jnp.concatenate([action_features, roles], axis=-1),
        )
        embedding = _backbone(self.gnn_layers, self.hidden_dim)()(
            augmented, node_type=0, n_type=n_agents
        )
        ego_embedding = jnp.sum(embedding * ego_mask[:, None], axis=0)
        hidden = MLP(
            hid_sizes=(self.hidden_dim, self.hidden_dim),
            act=nn.relu,
            act_final=True,
            name="SafetyQHead",
        )(ego_embedding)
        return nn.Dense(1, kernel_init=default_nn_init(), name="SafetyQOutput")(
            hidden
        ).squeeze(-1)


class ActionConditionedSafetyQ:
    def __init__(
        self,
        action_dim: int,
        n_agents: int,
        gnn_layers: int,
        hidden_dim: int,
    ):
        self.n_agents = n_agents
        self.net = _ActionConditionedSafetyQNet(
            action_dim, gnn_layers, hidden_dim
        )

    def initialize(
        self,
        key: Array,
        graph: GraphsTuple,
        joint_action: Action,
        ego_id: Array,
        neighbor_mask: Array,
    ) -> Params:
        return self.net.init(
            key, graph, joint_action, ego_id, neighbor_mask, self.n_agents
        )

    def get_value(
        self,
        params: Params,
        graph: GraphsTuple,
        joint_action: Action,
        ego_id: Array,
        neighbor_mask: Array,
    ) -> Array:
        return self.net.apply(
            params, graph, joint_action, ego_id, neighbor_mask, self.n_agents
        )


class _SharedSafetyValueNet(nn.Module):
    gnn_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, n_agents: int) -> Array:
        embedding = _backbone(self.gnn_layers, self.hidden_dim)()(
            graph, node_type=0, n_type=n_agents
        )
        hidden = MLP(
            hid_sizes=(self.hidden_dim, self.hidden_dim),
            act=nn.relu,
            act_final=True,
            name="SafetyValueHead",
        )(embedding)
        return nn.Dense(
            1, kernel_init=default_nn_init(), name="SafetyValueOutput"
        )(hidden).squeeze(-1)


class SharedSafetyValue:
    def __init__(self, n_agents: int, gnn_layers: int, hidden_dim: int):
        self.n_agents = n_agents
        self.net = _SharedSafetyValueNet(gnn_layers, hidden_dim)

    def initialize(self, key: Array, graph: GraphsTuple) -> Params:
        return self.net.init(key, graph, self.n_agents)

    def get_value(self, params: Params, graph: GraphsTuple) -> Array:
        return self.net.apply(params, graph, self.n_agents)
