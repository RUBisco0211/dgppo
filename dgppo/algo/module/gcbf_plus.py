"""Neural modules used by the native DGPPO GCBF+ port."""

from __future__ import annotations

import functools as ft

import flax.linen as nn
import jax.numpy as jnp

from ...nn.gnn import AttentionAggregationGNN
from ...nn.mlp import MLP
from ...nn.utils import default_nn_init
from ...utils.graph import GraphsTuple
from ...utils.typing import Action, Array, Params


def _backbone(layers: int):
    return ft.partial(
        AttentionAggregationGNN,
        msg_dim=128,
        hid_size_msg=(256, 256),
        hid_size_aggr=(128, 128),
        hid_size_update=(256, 256),
        out_dim=128,
        n_layers=layers,
    )


class _GCBFNet(nn.Module):
    gnn_layers: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, n_agents: int) -> Array:
        embedding = _backbone(self.gnn_layers)()(graph, node_type=0, n_type=n_agents)
        hidden = MLP(
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name="CBFHead",
        )(embedding)
        return jnp.tanh(nn.Dense(1, kernel_init=default_nn_init())(hidden))


class GCBFNetwork:
    def __init__(self, n_agents: int, gnn_layers: int = 1):
        self.n_agents = n_agents
        self.net = _GCBFNet(gnn_layers=gnn_layers)

    def initialize(self, key: Array, graph: GraphsTuple) -> Params:
        return self.net.init(key, graph, self.n_agents)

    def get_cbf(self, params: Params, graph: GraphsTuple) -> Array:
        return self.net.apply(params, graph, self.n_agents)


class _DeterministicActorNet(nn.Module):
    action_dim: int
    gnn_layers: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, n_agents: int) -> Action:
        embedding = _backbone(self.gnn_layers)()(graph, node_type=0, n_type=n_agents)
        hidden = MLP(
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name="PolicyHead",
        )(embedding)
        return jnp.tanh(
            nn.Dense(
                self.action_dim,
                kernel_init=default_nn_init(),
                name="OutputDense",
            )(hidden)
        )


class DeterministicGCBFPolicy:
    def __init__(self, action_dim: int, n_agents: int, gnn_layers: int = 1):
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.net = _DeterministicActorNet(action_dim, gnn_layers)

    def initialize(self, key: Array, graph: GraphsTuple) -> Params:
        return self.net.init(key, graph, self.n_agents)

    def get_action(self, params: Params, graph: GraphsTuple) -> Action:
        return self.net.apply(params, graph, self.n_agents)
