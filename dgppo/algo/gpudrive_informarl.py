"""InforMARL PPO specialized for externally collected GPUDrive rollouts."""

import os
import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training.train_state import TrainState

from .module.policy import PPOPolicy
from .module.value import ValueNet
from ..utils.graph import GraphsTuple


def _masked_mean(value, mask):
    mask = jnp.broadcast_to(mask, value.shape).astype(value.dtype)
    return jnp.sum(jnp.where(mask.astype(bool), value, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)


class GPUDriveInforMARL:
    """InforMARL actor and task critic, without Vh/CBF safety updates."""

    def __init__(self, actor_graph: GraphsTuple, critic_graph: GraphsTuple, *,
                 actor_gnn_layers=1, Vl_gnn_layers=1, lr_actor=3e-4,
                 lr_Vl=1e-3, gamma=0.99, gae_lambda=0.95, clip_eps=0.25,
                 coef_ent=1e-2, max_grad_norm=2.0, seed=0,
                 minibatch_worlds=4, epoch_ppo=1, n_agents=5,
                 base_node_dim=18, road_slots=32, road_feature_dim=8,
                 behavior_cloning=False, bc_checkpoint=None, bc_steps=0):
        self.n_agents, self.action_dim = int(n_agents), 2
        self.behavior_cloning = bool(behavior_cloning)
        self.bc_checkpoint, self.bc_steps = bc_checkpoint, int(bc_steps)
        if self.behavior_cloning and not self.bc_checkpoint:
            raise ValueError("behavior cloning requires --bc-checkpoint")
        self.gamma, self.gae_lambda, self.clip_eps = gamma, gae_lambda, clip_eps
        self.coef_ent, self.lr_actor, self.lr_Vl = coef_ent, lr_actor, lr_Vl
        self.minibatch_worlds, self.epoch_ppo = int(minibatch_worlds), int(epoch_ppo)
        self.last_update_count = 0
        self.key = jr.PRNGKey(seed)
        self.rnn_state = jnp.zeros((1, self.n_agents, 1, 64), dtype=jnp.float32)
        self.Vl_rnn_state = jnp.zeros((1, 1, 1, 64), dtype=jnp.float32)
        self.policy = PPOPolicy(actor_graph.nodes.shape[-1], actor_graph.edges.shape[-1],
                                self.n_agents, self.action_dim, use_rnn=False,
                                gnn_layers=actor_gnn_layers, gnn_out_dim=64,
                                base_node_dim=base_node_dim, road_slots=road_slots,
                                road_feature_dim=road_feature_dim)
        self.Vl = ValueNet(critic_graph.nodes.shape[-1], critic_graph.edges.shape[-1],
                           self.n_agents, use_rnn=False, gnn_layers=Vl_gnn_layers,
                           gnn_out_dim=64, decompose=False,
                           base_node_dim=base_node_dim, road_slots=road_slots,
                           road_feature_dim=road_feature_dim)
        k1, k2, self.key = jr.split(self.key, 3)
        pp = self.policy.dist.init(k1, actor_graph, self.rnn_state, self.n_agents)
        vp = self.Vl.net.init(k2, critic_graph, self.Vl_rnn_state, self.n_agents)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.sample_action, params=pp,
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_actor)))
        self.Vl_state = TrainState.create(
            apply_fn=self.Vl.get_value, params=vp,
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_Vl)))
        self._sample_actions_jit = jax.jit(self._sample_actions)
        self._deterministic_actions_jit = jax.jit(self._deterministic_actions)
        self._update_jit = jax.jit(self._update_step)

    def _sample_actions(self, params, graphs, key):
        keys = jr.split(key, graphs.n_node.shape[0])
        return jax.vmap(lambda g, k: self.policy.sample_action(
            params, g, self.rnn_state, k)[:2])(graphs, keys)

    def sample_actions(self, graphs, key):
        return self._sample_actions_jit(self.policy_state.params, graphs, key)

    def _deterministic_actions(self, params, graphs):
        return jax.vmap(lambda g: self.policy.get_action(
            params, g, self.rnn_state)[0])(graphs)

    def deterministic_actions(self, graphs):
        return self._deterministic_actions_jit(self.policy_state.params, graphs)

    def _value(self, params, graph):
        return self.Vl.get_value(params, graph, self.Vl_rnn_state)[0].squeeze()

    def _update_step(self, policy_state, Vl_state, key, rollout):
        values = jax.vmap(jax.vmap(lambda g: self._value(Vl_state.params, g)))(rollout.critic_graph)
        next_values = jax.vmap(jax.vmap(lambda g: self._value(Vl_state.params, g)))(rollout.next_critic_graph)
        alive = rollout.alive_mask
        world_alive = jnp.any(alive, axis=-1)
        alive_agents = jnp.sum(alive, axis=-1)
        team_reward = (
            jnp.sum(jnp.where(alive, rollout.rewards, 0.0), axis=-1)
            / jnp.maximum(alive_agents, 1)
        )
        world_terminal = jnp.all((~alive) | rollout.terminated, axis=-1)
        bootstrap = jnp.where(
            world_terminal | rollout.truncated.any(axis=-1), 0.0, next_values
        )
        deltas = team_reward + self.gamma * bootstrap - values
        def gae_scan(carry, xs):
            delta, valid = xs
            carry = delta + self.gamma * self.gae_lambda * carry * valid
            return carry, carry
        _, adv_rev = jax.lax.scan(gae_scan, jnp.zeros(values.shape[0]),
                                  (jnp.swapaxes(deltas, 0, 1)[::-1],
                                   jnp.swapaxes(world_alive, 0, 1)[::-1]))
        advantage = jnp.swapaxes(adv_rev[::-1], 0, 1)
        target = advantage + values
        adv_mean = _masked_mean(advantage, world_alive)
        adv_var = _masked_mean(jnp.square(advantage - adv_mean), world_alive)
        advantage = (advantage - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
        agent_advantage = jnp.repeat(advantage[..., None], self.n_agents, axis=-1)

        def vl_loss(params):
            pred = jax.vmap(jax.vmap(lambda g: self._value(params, g)))(rollout.critic_graph)
            return _masked_mean(optax.l2_loss(pred, target), world_alive)
        vl_value, vl_grad = jax.value_and_grad(vl_loss)(Vl_state.params)
        vl_grad_norm = optax.global_norm(vl_grad)
        Vl_state = Vl_state.apply_gradients(grads=vl_grad)
        key, entropy_key = jr.split(key)
        keys = jr.split(entropy_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        def policy_loss(params):
            def evaluate(g, a, k):
                return self.policy.eval_action(params, g, a, self.rnn_state, k)[:2]
            logp, entropy = jax.vmap(jax.vmap(evaluate))(rollout.graph, rollout.actions, keys)
            ratio = jnp.exp(logp - rollout.log_pis)
            objective = jnp.minimum(ratio * agent_advantage,
                                    jnp.clip(ratio, 1-self.clip_eps, 1+self.clip_eps) * agent_advantage)
            loss = -_masked_mean(objective, alive) - self.coef_ent * _masked_mean(entropy, alive)
            return loss, (entropy, _masked_mean(jnp.abs(ratio - 1) > self.clip_eps, alive))
        (policy_value, (entropy, clip_fraction)), policy_grad = jax.value_and_grad(
            policy_loss, has_aux=True)(policy_state.params)
        policy_grad_norm = optax.global_norm(policy_grad)
        policy_state = policy_state.apply_gradients(grads=policy_grad)
        info = {"policy/loss": policy_value, "policy/entropy": _masked_mean(entropy, alive),
                "policy/clip_fraction": clip_fraction, "policy/grad_norm": policy_grad_norm,
                "Vl/loss": vl_value, "Vl/grad_norm": vl_grad_norm,
                "Vl/target_mean": _masked_mean(target, world_alive),
                "Vl/team_reward_mean": _masked_mean(team_reward, world_alive),
                "adv/task_abs_mean": _masked_mean(jnp.abs(advantage), world_alive),
                "learning_rate/actor": self.lr_actor, "learning_rate/Vl": self.lr_Vl}
        return policy_state, Vl_state, key, info

    def update(self, rollout, safety_rollout=None):
        n_worlds = int(rollout.actions.shape[0])
        minibatch = min(self.minibatch_worlds, n_worlds)
        if n_worlds % minibatch:
            raise ValueError("num_worlds must be divisible by minibatch_worlds")
        infos, self.last_update_count = [], 0
        for _ in range(self.epoch_ppo):
            permutation_key, self.key = jr.split(self.key)
            permutation = jr.permutation(permutation_key, n_worlds)
            for start in range(0, n_worlds, minibatch):
                batch = jax.tree_util.tree_map(lambda x: x[permutation[start:start+minibatch]], rollout)
                self.policy_state, self.Vl_state, self.key, info = self._update_jit(
                    self.policy_state, self.Vl_state, self.key, batch)
                infos.append(info); self.last_update_count += 1
        return jax.tree_util.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *infos)

    def save(self, directory, step, counters=None):
        path = os.path.join(directory, str(step)); os.makedirs(path, exist_ok=True)
        def state_payload(state):
            return {"step": state.step, "params": state.params,
                    "opt_state": state.opt_state}
        payload = {"policy": state_payload(self.policy_state),
                   "Vl": state_payload(self.Vl_state),
                   "key": self.key, "counters": counters or {}}
        target = os.path.join(path, "training_state.pkl")
        with open(target + ".tmp", "wb") as handle:
            pickle.dump(jax.device_get(payload), handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(target + ".tmp", target)

    def load(self, directory, step):
        with open(os.path.join(directory, str(step), "training_state.pkl"), "rb") as handle:
            payload = pickle.load(handle)
        def restore(state, saved):
            return state.replace(step=saved["step"], params=saved["params"],
                                 opt_state=saved["opt_state"])
        self.policy_state = restore(self.policy_state, payload["policy"])
        self.Vl_state = restore(self.Vl_state, payload["Vl"])
        self.key = jnp.asarray(payload["key"])
        return payload.get("counters", {})
