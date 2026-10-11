"""JAX GNN-IPPO for fixed-slot GPUDrive vehicle graphs."""

import os
import pickle
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax.training.train_state import TrainState

from ..nn.gnn import GraphTransformerGNN


def _masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape).astype(x.dtype)
    return jnp.sum(jnp.where(mask.astype(bool), x, 0.0)) / jnp.maximum(mask.sum(), 1.0)


def _tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


class _Actor(nn.Module):
    n_agents: int
    action_mode: Literal["discrete", "continuous"]
    gnn_layers: int
    base_node_dim: int
    road_slots: int
    road_feature_dim: int

    @nn.compact
    def __call__(self, graph):
        x = GraphTransformerGNN(
            msg_dim=32, out_dim=64, n_heads=3, n_layers=self.gnn_layers,
            base_node_dim=self.base_node_dim, road_slots=self.road_slots,
            road_feature_dim=self.road_feature_dim,
        )(graph, node_type=0, n_type=self.n_agents)
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        if self.action_mode == "discrete":
            return nn.Dense(91)(x)
        mean = nn.Dense(2)(x)
        log_std = jnp.clip(nn.Dense(2)(x), -5.0, 2.0)
        return mean, log_std


class _Critic(nn.Module):
    n_agents: int
    gnn_layers: int
    base_node_dim: int
    road_slots: int
    road_feature_dim: int

    @nn.compact
    def __call__(self, graph):
        x = GraphTransformerGNN(
            msg_dim=32, out_dim=64, n_heads=3, n_layers=self.gnn_layers,
            base_node_dim=self.base_node_dim, road_slots=self.road_slots,
            road_feature_dim=self.road_feature_dim,
        )(graph, node_type=0, n_type=self.n_agents)
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(1)(x).squeeze(-1)


class GPUDriveGNNIPPO:
    """Independent PPO with shared graph actor and per-agent graph values."""

    def __init__(self, nominal_graph, *, action_mode="discrete", n_agents=64,
                 actor_gnn_layers=1, critic_gnn_layers=1, lr_actor=3e-4,
                 lr_critic=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, coef_ent=1e-4, max_grad_norm=0.5,
                 minibatch_worlds=4, epoch_ppo=4, seed=0,
                 base_node_dim=18, road_slots=32, road_feature_dim=8):
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"unsupported action_mode: {action_mode}")
        self.action_mode = action_mode
        self.n_agents = int(n_agents)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.clip_eps, self.coef_ent = float(clip_eps), float(coef_ent)
        self.lr_actor, self.lr_critic = float(lr_actor), float(lr_critic)
        self.minibatch_worlds = int(minibatch_worlds)
        self.epoch_ppo = int(epoch_ppo)
        self.last_update_count = 0
        self.key = jr.PRNGKey(seed)
        self.actor = _Actor(self.n_agents, action_mode, actor_gnn_layers,
                            base_node_dim, road_slots, road_feature_dim)
        self.critic = _Critic(self.n_agents, critic_gnn_layers, base_node_dim,
                              road_slots, road_feature_dim)
        k1, k2, self.key = jr.split(self.key, 3)
        actor_params = self.actor.init(k1, nominal_graph)
        critic_params = self.critic.init(k2, nominal_graph)
        tx_actor = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_actor))
        tx_critic = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_critic))
        self.actor_state = TrainState.create(apply_fn=self.actor.apply, params=actor_params, tx=tx_actor)
        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=critic_params, tx=tx_critic)
        self._sample_jit = jax.jit(self._sample)
        self._deterministic_jit = jax.jit(self._deterministic)
        self._update_jit = jax.jit(self._update_step)

    @staticmethod
    def _decode_discrete(index):
        steer_idx, accel_idx = index // 7, index % 7
        steer = -1.0 + 2.0 * steer_idx.astype(jnp.float32) / 12.0
        accel = -1.0 + 2.0 * accel_idx.astype(jnp.float32) / 6.0
        return jnp.stack([accel, steer], axis=-1)

    @staticmethod
    def _encode_discrete(action):
        accel_idx = jnp.clip(jnp.rint((action[..., 0] + 1.0) * 3.0), 0, 6).astype(jnp.int32)
        steer_idx = jnp.clip(jnp.rint((action[..., 1] + 1.0) * 6.0), 0, 12).astype(jnp.int32)
        return steer_idx * 7 + accel_idx

    def _distribution_stats(self, params, graph, action=None, key=None, deterministic=False):
        output = self.actor.apply(params, graph)
        if self.action_mode == "discrete":
            logits = output
            index = jnp.argmax(logits, axis=-1) if deterministic else jr.categorical(key, logits)
            if action is not None:
                index = self._encode_discrete(action)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            probs = jax.nn.softmax(logits, axis=-1)
            logp = jnp.take_along_axis(log_probs, index[..., None], axis=-1).squeeze(-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1)
            return self._decode_discrete(index), logp, entropy
        mean, log_std = output
        if action is None:
            raw = mean if deterministic else mean + jnp.exp(log_std) * jr.normal(key, mean.shape)
            action = jnp.tanh(raw)
        clipped = jnp.clip(action, -0.999, 0.999)
        raw = jnp.arctanh(clipped)
        logp = -0.5 * jnp.square((raw - mean) / jnp.exp(log_std)) - log_std - 0.5 * jnp.log(2*jnp.pi)
        logp = jnp.sum(logp - jnp.log(1.0 - jnp.square(clipped) + 1e-6), axis=-1)
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2*jnp.pi*jnp.e), axis=-1)
        return action, logp, entropy

    def _sample(self, params, graphs, key):
        keys = jr.split(key, graphs.n_node.shape[0])
        return jax.vmap(lambda g, k: self._distribution_stats(params, g, key=k)[:2])(graphs, keys)

    def sample_actions(self, graphs, key):
        return self._sample_jit(self.actor_state.params, graphs, key)

    def _deterministic(self, params, graphs):
        return jax.vmap(lambda g: self._distribution_stats(
            params, g, deterministic=True, key=jr.PRNGKey(0))[0])(graphs)

    def deterministic_actions(self, graphs):
        return self._deterministic_jit(self.actor_state.params, graphs)

    def _values(self, params, graphs):
        return jax.vmap(jax.vmap(lambda g: self.critic.apply(params, g)))(graphs)

    def _update_step(self, actor_state, critic_state, key, rollout):
        key, eval_key = jr.split(key)
        values = self._values(critic_state.params, rollout.graph)
        next_values = self._values(critic_state.params, rollout.next_graph)
        valid = rollout.alive_mask
        terminal = rollout.terminated | rollout.truncated
        bootstrap = jnp.where(terminal, 0.0, next_values)
        delta = rollout.rewards + self.gamma * bootstrap - values

        def gae_step(carry, xs):
            d, term, mask = xs
            carry = d + self.gamma * self.gae_lambda * carry * (~term)
            carry = jnp.where(mask, carry, 0.0)
            return carry, carry
        _, reversed_adv = jax.lax.scan(
            gae_step, jnp.zeros((delta.shape[0], delta.shape[2]), dtype=delta.dtype),
            (jnp.swapaxes(delta, 0, 1)[::-1],
             jnp.swapaxes(terminal, 0, 1)[::-1],
             jnp.swapaxes(valid, 0, 1)[::-1]))
        advantage = jnp.swapaxes(reversed_adv[::-1], 0, 1)
        target = advantage + values
        adv_mean = _masked_mean(advantage, valid)
        adv_var = _masked_mean(jnp.square(advantage - adv_mean), valid)
        norm_adv = (advantage - adv_mean) / jnp.sqrt(adv_var + 1e-8)

        def critic_loss(params):
            pred = self._values(params, rollout.graph)
            return _masked_mean(jnp.square(pred - target), valid)
        critic_value, critic_grad = jax.value_and_grad(critic_loss)(critic_state.params)
        critic_grad_norm = optax.global_norm(critic_grad)
        critic_state = critic_state.apply_gradients(grads=critic_grad)

        keys = jr.split(eval_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        def actor_loss(params):
            def evaluate(g, a, k):
                return self._distribution_stats(params, g, action=a, key=k)[1:]
            logp, entropy = jax.vmap(jax.vmap(evaluate))(rollout.graph, rollout.actions, keys)
            ratio = jnp.exp(logp - rollout.log_pis)
            clipped = jnp.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)
            objective = jnp.minimum(ratio * norm_adv, clipped * norm_adv)
            loss = -_masked_mean(objective, valid) - self.coef_ent * _masked_mean(entropy, valid)
            return loss, (entropy, ratio)
        (actor_value, (entropy, ratio)), actor_grad = jax.value_and_grad(
            actor_loss, has_aux=True)(actor_state.params)
        actor_grad_norm = optax.global_norm(actor_grad)
        actor_state = actor_state.apply_gradients(grads=actor_grad)
        info = {
            "policy/loss": actor_value,
            "policy/entropy": _masked_mean(entropy, valid),
            "policy/clip_fraction": _masked_mean(jnp.abs(ratio - 1) > self.clip_eps, valid),
            "policy/grad_norm": actor_grad_norm,
            "Vl/loss": critic_value,
            "Vl/grad_norm": critic_grad_norm,
            "Vl/value_mean": _masked_mean(values, valid),
            "Vl/target_mean": _masked_mean(target, valid),
            "adv/task_abs_mean": _masked_mean(jnp.abs(advantage), valid),
            "learning_rate/actor": self.lr_actor,
            "learning_rate/Vl": self.lr_critic,
            "diagnostics/actor_grad_finite": _tree_all_finite(actor_grad),
            "diagnostics/critic_grad_finite": _tree_all_finite(critic_grad),
            "diagnostics/actor_params_finite": _tree_all_finite(actor_state.params),
            "diagnostics/critic_params_finite": _tree_all_finite(critic_state.params),
            "diagnostics/value_finite": jnp.all(jnp.isfinite(values)),
            "diagnostics/next_value_finite": jnp.all(jnp.isfinite(next_values)),
            "diagnostics/advantage_finite": jnp.all(jnp.isfinite(advantage)),
            "diagnostics/target_finite": jnp.all(jnp.isfinite(target)),
            "diagnostics/ratio_finite": jnp.all(jnp.isfinite(ratio)),
            "diagnostics/value_abs_max": jnp.max(jnp.abs(values)),
            "diagnostics/target_abs_max": jnp.max(jnp.abs(target)),
            "diagnostics/advantage_abs_max": jnp.max(jnp.abs(advantage)),
            "diagnostics/ratio_max": jnp.max(ratio),
        }
        return actor_state, critic_state, key, info

    def update(self, rollout, safety_rollout=None):
        input_checks = {
            "graph.nodes": rollout.graph.nodes,
            "graph.edges": rollout.graph.edges,
            "next_graph.nodes": rollout.next_graph.nodes,
            "next_graph.edges": rollout.next_graph.edges,
            "actions": rollout.actions,
            "rewards": rollout.rewards,
            "log_pis": rollout.log_pis,
        }
        invalid_inputs = [
            name for name, value in input_checks.items()
            if not bool(np.isfinite(np.asarray(value)).all())
        ]
        if invalid_inputs:
            raise FloatingPointError(
                f"non-finite GNN-IPPO rollout inputs: {invalid_inputs}")
        n_worlds = int(rollout.actions.shape[0])
        batch = min(self.minibatch_worlds, n_worlds)
        if n_worlds % batch:
            raise ValueError("num_worlds must be divisible by minibatch_worlds")
        infos = []
        self.last_update_count = 0
        for epoch in range(self.epoch_ppo):
            perm_key, self.key = jr.split(self.key)
            order = np.asarray(jax.device_get(jr.permutation(perm_key, n_worlds)))
            for start in range(0, n_worlds, batch):
                sample = jax.tree_util.tree_map(
                    lambda x: x[order[start:start+batch]], rollout)
                self.actor_state, self.critic_state, self.key, info = self._update_jit(
                    self.actor_state, self.critic_state, self.key, sample)
                host_info = jax.device_get(info)
                failed = [
                    name for name in (
                        "diagnostics/actor_grad_finite",
                        "diagnostics/critic_grad_finite",
                        "diagnostics/actor_params_finite",
                        "diagnostics/critic_params_finite",
                        "diagnostics/value_finite",
                        "diagnostics/next_value_finite",
                        "diagnostics/advantage_finite",
                        "diagnostics/target_finite",
                        "diagnostics/ratio_finite",
                    )
                    if not bool(host_info[name])
                ]
                if failed:
                    stats = {
                        name: float(host_info[name]) for name in (
                            "policy/loss", "Vl/loss", "policy/grad_norm",
                            "Vl/grad_norm", "diagnostics/value_abs_max",
                            "diagnostics/target_abs_max",
                            "diagnostics/advantage_abs_max",
                            "diagnostics/ratio_max",
                        )
                    }
                    raise FloatingPointError(
                        "non-finite GNN-IPPO update at "
                        f"epoch={epoch}, minibatch={start // batch}: "
                        f"{failed}; stats={stats}")
                infos.append(info)
                self.last_update_count += 1
        return jax.tree_util.tree_map(lambda *x: jnp.mean(jnp.stack(x)), *infos)

    def save(self, directory, step, counters=None):
        path = os.path.join(directory, str(step)); os.makedirs(path, exist_ok=True)
        def state_payload(state):
            return {"step": state.step, "params": state.params,
                    "opt_state": state.opt_state}
        payload = {"actor": state_payload(self.actor_state),
                   "critic": state_payload(self.critic_state),
                   "key": self.key, "counters": counters or {},
                   "action_mode": self.action_mode}
        target = os.path.join(path, "training_state.pkl")
        with open(target + ".tmp", "wb") as f:
            pickle.dump(jax.device_get(payload), f); f.flush(); os.fsync(f.fileno())
        os.replace(target + ".tmp", target)

    def load(self, directory, step):
        with open(os.path.join(directory, str(step), "training_state.pkl"), "rb") as f:
            payload = pickle.load(f)
        if payload.get("action_mode") != self.action_mode:
            raise ValueError("checkpoint action_mode does not match requested mode")
        def restore(state, saved):
            return state.replace(step=saved["step"], params=saved["params"],
                                 opt_state=saved["opt_state"])
        self.actor_state = restore(self.actor_state, payload["actor"])
        self.critic_state = restore(self.critic_state, payload["critic"])
        self.key = jnp.asarray(payload["key"])
        return payload.get("counters", {})
