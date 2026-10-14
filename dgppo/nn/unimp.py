"""UniMP-style graph attention used by the GRAFO reproduction."""

import flax.linen as nn
import jax.numpy as jnp
import jraph

from dgppo.utils.graph import GraphsTuple
from dgppo.nn.utils import default_nn_init


class UniMPLayer(nn.Module):
    hidden_dim: int
    n_heads: int
    key_dim: int

    @nn.compact
    def __call__(self, nodes, edges, senders, receivers):
        query = nn.Dense(self.n_heads * self.key_dim,
                         kernel_init=default_nn_init(), name="query")(nodes)
        key = nn.Dense(self.n_heads * self.key_dim,
                       kernel_init=default_nn_init(), name="key")(nodes)
        edge_key = nn.Dense(self.n_heads * self.key_dim, use_bias=False,
                            kernel_init=default_nn_init(), name="edge_key")(edges)
        value = nn.Dense(self.n_heads * self.hidden_dim,
                         kernel_init=default_nn_init(), name="value")(nodes)
        residual = nn.Dense(self.hidden_dim,
                            kernel_init=default_nn_init(), name="residual")(nodes)
        query = query[receivers].reshape((-1, self.n_heads, self.key_dim))
        key = key[senders].reshape((-1, self.n_heads, self.key_dim))
        edge_key = edge_key.reshape((-1, self.n_heads, self.key_dim))
        logits = jnp.sum(query * (key + edge_key), axis=-1) / jnp.sqrt(self.key_dim)
        attention = jraph.segment_softmax(logits, receivers, num_segments=nodes.shape[0])
        values = value[senders].reshape((-1, self.n_heads, self.hidden_dim))
        messages = attention[..., None] * values
        summed = jraph.segment_sum(messages, receivers, num_segments=nodes.shape[0])
        return nn.relu(residual + summed.mean(axis=1))


class UniMPGNN(nn.Module):
    hidden_dim: int = 128
    n_heads: int = 3
    key_dim: int = 16
    n_layers: int = 4

    @nn.compact
    def __call__(self, graph: GraphsTuple, node_type: int = None,
                 n_type: int = None):
        # The paper specifies two hidden GNN layers before four UniMP rounds.
        nodes = graph.nodes
        edges = graph.edges
        for i in range(2):
            nodes = nn.relu(nn.Dense(self.hidden_dim, name=f"node_embed_{i}")(nodes))
            edges = nn.relu(nn.Dense(self.hidden_dim, name=f"edge_embed_{i}")(edges))
        scales = []
        for layer in range(self.n_layers):
            nodes = UniMPLayer(self.hidden_dim, self.n_heads, self.key_dim,
                               name=f"layer_{layer}")(
                                   nodes, edges, graph.senders,
                                   graph.receivers)
            scales.append(nodes)
        result = jnp.concatenate(scales, axis=-1)
        if node_type is None:
            return result
        return graph._replace(nodes=result).type_nodes(node_type, n_type)
