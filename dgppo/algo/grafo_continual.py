"""Continual-learning extensions around the repository's InforMARL PPO."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import functools as ft

from .informarl import InforMARL
from ..trainer.utils import episodic_rollout
from ..trainer.data import Rollout


def _actor_mask(params, *, freeze_encoder=False, regularize=True):
    def selected(path, leaf):
        name = jtu.keystr(path)
        is_encoder = "GraphTransformerGNN" in name or "UniMPGNN" in name
        is_output = any(x in name for x in (
            "OutputDenseMean", "OutputDenseStdTrans", "OutputDenseLogits"))
        keep = not is_encoder if freeze_encoder else (not is_output if regularize else True)
        return jnp.asarray(keep)
    return jtu.tree_map_with_path(selected, params)


def output_head(params):
    """Return the output layers that are kept separately for each task."""
    return {name: params["params"][name] for name in
            ("OutputDenseMean", "OutputDenseStdTrans", "OutputDenseLogits")
            if name in params["params"]}


def with_output_head(params, head):
    from flax.core import freeze, unfreeze
    result = unfreeze(params)
    for name, value in head.items():
        result["params"][name] = value
    return result if isinstance(params, dict) else freeze(result)


class GrafoContinualInforMARL(InforMARL):
    def __init__(self, *args, freeze_encoder=False, regularization="none",
                 regularization_coefficient=0.0, anchor=None,
                 importance=None, anchors=None, importances=None,
                 rollout_horizon=128, shuffle_rng=None, **kwargs):
        kwargs.setdefault("actor_use_rnn", False)
        kwargs.setdefault("critic_use_rnn", True)
        kwargs.setdefault("policy_extra_hidden", False)
        kwargs.setdefault("discrete_actions", getattr(args[0], "action_encoding", "continuous") == "discrete")
        super().__init__(*args, **kwargs)
        self.shuffle_rng = shuffle_rng
        self.episodic_rollout = True
        self.rollout_horizon = rollout_horizon

        def collect_one(params, key):
            return episodic_rollout(
                self._env, ft.partial(self.step, params=params),
                self.init_rnn_state, key, horizon=rollout_horizon)

        self.rollout_fn = jax.jit(
            lambda params, keys: jax.vmap(ft.partial(collect_one, params))(keys))
        self.actor_grad_mask = (_actor_mask(self.policy_train_state.params,
                                            freeze_encoder=True)
                                if freeze_encoder else None)
        self.regularization = regularization
        self.regularization_coefficient = regularization_coefficient
        self.anchor = anchor
        self.importance = importance
        self.anchors = anchors if anchors is not None else ([anchor] if anchor is not None else [])
        self.importances = (importances if importances is not None else
                            ([importance] if importance is not None else []))
        self.regularization_mask = _actor_mask(self.policy_train_state.params)

    def actor_auxiliary_loss(self, params, rollout, rnn_chunk_ids,
                             bcT_graph, bcT_rnn_states):
        if not self.anchors or self.regularization == "none":
            return jnp.array(0., dtype=jnp.float32), {}
        if self.regularization not in ("l2", "ewc", "mas"):
            raise ValueError(f"unknown continual regularizer: {self.regularization}")
        penalty = jnp.array(0., dtype=jnp.float32)
        for index, anchor in enumerate(self.anchors):
            weights = (self.importances[index] if index < len(self.importances)
                       and self.importances[index] is not None else
                       jtu.tree_map(jnp.ones_like, params))
            terms = jtu.tree_leaves(jtu.tree_map(
                lambda p, old, weight, selected: jnp.sum(
                    jnp.where(selected, weight * jnp.square(p - old), 0.)),
                params, anchor, weights, self.regularization_mask))
            penalty = penalty + self.regularization_coefficient * 0.5 * sum(terms)
        return penalty, {"policy/continual_penalty": penalty}

    def estimate_importance(self, rollout, method="ewc", batch_size=16):
        """Estimate diagonal Fisher (EWC) or MAS importance on JAX arrays."""
        if self.actor_use_rnn:
            raise ValueError("importance estimation requires a feedforward actor")
        graph = jtu.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]),
                             rollout.graph._replace(env_states=None))
        actions = rollout.actions.reshape((-1,) + rollout.actions.shape[2:])
        count = actions.shape[0]
        # Each importance trajectory is one episode. Ignore fixed-horizon
        # padding after its first terminal transition.
        earlier_done = jnp.cumsum(rollout.dones.astype(jnp.int32), axis=1)
        valid = jnp.concatenate(
            (jnp.ones_like(rollout.dones[:, :1], dtype=bool),
             earlier_done[:, :-1] == 0), axis=1).reshape((-1,))
        valid_count = jnp.maximum(valid.sum(), 1)
        params = self.policy_train_state.params

        def score(p, g, a):
            dist, _ = self.policy.dist.apply(p, g, self.init_rnn_state,
                                             n_agents=self.n_agents)
            if method == "ewc":
                return jnp.mean(dist.log_prob(a))
            if method == "mas":
                action_signal = (dist.probs_parameter()
                                 if self.discrete_actions else dist.mode())
                return jnp.mean(jnp.square(action_signal))
            raise ValueError(method)

        per_sample_grad = jax.jit(jax.vmap(jax.grad(score), in_axes=(None, 0, 0)))
        importance = jtu.tree_map(jnp.zeros_like, params)
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            small_graph = jtu.tree_map(lambda x: x[start:stop], graph)
            grad = per_sample_grad(params, small_graph, actions[start:stop])
            sample_valid = valid[start:stop]
            importance = jtu.tree_map(
                lambda old, values: old + jnp.sum(
                    (jnp.square(values) if method == "ewc" else jnp.abs(values)) *
                    sample_valid.reshape((-1,) + (1,) * (values.ndim - 1)),
                    axis=0), importance, grad)
        return jtu.tree_map(lambda x: x / valid_count, importance)

    def collect_importance(self, keys, horizon=50):
        """Collect the paper's 50-step importance episodes."""
        def collect_one(key):
            reset_key, key = jax.random.split(key)
            graph = self._env.reset(reset_key)

            def step(carry, action_key):
                graph, rnn_state = carry
                action, log_pi, next_rnn_state = self.step(
                    graph, rnn_state, action_key, params=self.params)
                next_graph, reward, cost, done, _ = self._env.step(graph, action)
                return (next_graph, next_rnn_state), (
                    graph, action, rnn_state, reward, cost, done,
                    log_pi, next_graph)

            _, data = jax.lax.scan(
                step, (graph, self.init_rnn_state),
                jax.random.split(key, horizon))
            return Rollout(*data)
        return jax.jit(jax.vmap(collect_one))(keys)
