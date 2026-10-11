"""DGPPO update specialized for externally collected GPUDrive rollouts."""

import os
import pickle
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training.train_state import TrainState

from .utils import compute_dec_ocp_gae
from .module.policy import PPOPolicy
from .module.value import ValueNet
from ..trainer.gpudrive_runner import GPUDriveRollout
from ..utils.graph import GraphsTuple


def _masked_mean(value, mask):
    mask = jnp.broadcast_to(mask, value.shape).astype(value.dtype)
    return jnp.sum(jnp.where(mask.astype(bool), value, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)


class GPUDriveDGPPO:
    def __init__(self, actor_graph: GraphsTuple, critic_graph: GraphsTuple, *,
                 actor_gnn_layers=1, Vl_gnn_layers=1, Vh_gnn_layers=1,
                 lr_actor=3e-4, lr_Vl=1e-3, lr_Vh=1e-3, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.25, coef_ent=1e-2,
                 cbf_alpha=10.0, cbf_eps=1e-2, cbf_weight=1.0,
                 max_grad_norm=2.0, dt=0.04, seed=0,
                 minibatch_worlds=4, epoch_ppo=1, n_cost=2):
        self.n_agents, self.action_dim, self.n_cost = 3, 2, int(n_cost)
        self.gamma, self.gae_lambda, self.clip_eps = gamma, gae_lambda, clip_eps
        self.coef_ent, self.cbf_alpha, self.cbf_eps = coef_ent, cbf_alpha, cbf_eps
        self.cbf_weight, self.max_grad_norm, self.dt = cbf_weight, max_grad_norm, dt
        self.lr_actor, self.lr_Vl, self.lr_Vh = lr_actor, lr_Vl, lr_Vh
        self.minibatch_worlds = int(minibatch_worlds)
        self.epoch_ppo = int(epoch_ppo)
        if self.minibatch_worlds <= 0 or self.epoch_ppo <= 0:
            raise ValueError("minibatch_worlds and epoch_ppo must be positive")
        self.last_update_count = 0
        self.key = jr.PRNGKey(seed)
        self.rnn_state = jnp.zeros((1, self.n_agents, 1, 64), dtype=jnp.float32)
        self.Vl_rnn_state = jnp.zeros((1, 1, 1, 64), dtype=jnp.float32)

        self.policy = PPOPolicy(actor_graph.nodes.shape[-1], actor_graph.edges.shape[-1],
                                self.n_agents, self.action_dim, use_rnn=False,
                                gnn_layers=actor_gnn_layers, gnn_out_dim=64)
        self.Vl = ValueNet(critic_graph.nodes.shape[-1], critic_graph.edges.shape[-1],
                           self.n_agents, use_rnn=False, gnn_layers=Vl_gnn_layers,
                           gnn_out_dim=64, decompose=False)
        self.Vh = ValueNet(actor_graph.nodes.shape[-1], actor_graph.edges.shape[-1],
                           self.n_agents, n_out=self.n_cost, use_rnn=False,
                           gnn_layers=Vh_gnn_layers, gnn_out_dim=64, decompose=True)
        k1, k2, k3, self.key = jr.split(self.key, 4)
        pp = self.policy.dist.init(k1, actor_graph, self.rnn_state, self.n_agents)
        lp = self.Vl.net.init(k2, critic_graph, self.Vl_rnn_state, self.n_agents)
        hp = self.Vh.net.init(k3, actor_graph, self.rnn_state, self.n_agents)
        self.policy_state = TrainState.create(apply_fn=self.policy.sample_action, params=pp,
                                               tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_actor)))
        self.Vl_state = TrainState.create(apply_fn=self.Vl.get_value, params=lp,
                                           tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_Vl)))
        self.Vh_state = TrainState.create(apply_fn=self.Vh.get_value, params=hp,
                                           tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_Vh)))
        self._sample_actions_jit = jax.jit(self._sample_actions)
        self._deterministic_actions_jit = jax.jit(self._deterministic_actions)
        self._update_jit = jax.jit(self._update_step)

    def _sample_actions(self, params, graphs: GraphsTuple, key):
        keys = jr.split(key, graphs.n_node.shape[0])
        def one(graph, sample_key):
            action, log_pi, _ = self.policy.sample_action(
                params, graph, self.rnn_state, sample_key)
            return action, log_pi
        return jax.vmap(one)(graphs, keys)

    def sample_actions(self, graphs: GraphsTuple, key):
        return self._sample_actions_jit(self.policy_state.params, graphs, key)

    def _deterministic_actions(self, params, graphs: GraphsTuple):
        return jax.vmap(lambda graph: self.policy.get_action(
            params, graph, self.rnn_state)[0])(graphs)

    def deterministic_actions(self, graphs: GraphsTuple):
        return self._deterministic_actions_jit(self.policy_state.params, graphs)

    def _values(self, rollout, Vl_params, Vh_params):
        vl_fn = lambda graph: self.Vl.get_value(Vl_params, graph, self.Vl_rnn_state)[0].squeeze()
        vh_fn = lambda graph: self.Vh.get_value(Vh_params, graph, self.rnn_state)[0]
        Vl = jax.vmap(jax.vmap(vl_fn))(rollout.critic_graph)
        next_Vl = jax.vmap(jax.vmap(vl_fn))(rollout.next_critic_graph)
        Vh = jax.vmap(jax.vmap(vh_fn))(rollout.graph)
        next_Vh = jax.vmap(jax.vmap(vh_fn))(rollout.next_graph)
        return Vl, next_Vl, Vh, next_Vh

    def _update_step(self, policy_state, Vl_state, Vh_state, key,
                     rollout: GPUDriveRollout, safety_rollout: GPUDriveRollout):
        Vl, next_Vl, Vh, next_Vh = self._values(
            rollout, Vl_state.params, Vh_state.params)
        alive = rollout.alive_mask
        world_terminal = jnp.all(rollout.terminated | ~alive, axis=-1)
        team_reward = jnp.sum(rollout.rewards * alive, axis=-1) / jnp.maximum(jnp.sum(alive, axis=-1), 1)
        valid_world = jnp.any(alive, axis=-1)

        _, _, safety_Vh, safety_next_Vh = self._values(
            safety_rollout, Vl_state.params, Vh_state.params)
        safety_cost = safety_rollout.costs
        safety_next_cost = safety_rollout.next_costs

        # Make terminal states absorbing before applying the same Dec-EFOCP
        # GAE target used by the original DGPPO implementation. GPUDrive can
        # terminate agents independently, whereas the original JAX tasks do
        # not terminate inside a rollout.
        safety_terminal_value = jnp.maximum(safety_cost, safety_next_cost)
        safety_done_before = jnp.concatenate(
            [jnp.zeros_like(safety_rollout.terminated[:, :1]),
             jnp.cumsum(safety_rollout.terminated[:, :-1], axis=1) > 0], axis=1)
        safety_absorbing = jnp.sum(
            jnp.where(safety_rollout.terminated[..., None], safety_terminal_value, 0.0), axis=1)
        safety_cost_eff = jnp.where(
            safety_done_before[..., None], safety_absorbing[:, None, :, :],
            jnp.where(safety_rollout.terminated[..., None], safety_terminal_value, safety_cost))
        safety_Vh_eff = jnp.where(
            safety_done_before[..., None], safety_absorbing[:, None, :, :], safety_Vh)
        safety_final_Vh = jnp.where(
            jnp.any(safety_rollout.terminated, axis=1)[..., None],
            safety_absorbing, safety_next_Vh[:, -1])
        safety_Tp1_Vh = jnp.concatenate(
            [safety_Vh_eff, safety_final_Vh[:, None, :, :]], axis=1)

        world_done_before = jnp.concatenate(
            [jnp.zeros_like(world_terminal[:, :1]),
             jnp.cumsum(world_terminal[:, :-1], axis=1) > 0], axis=1)
        task_loss = jnp.where(world_done_before, 0.0, -team_reward)
        Vl_eff = jnp.where(world_done_before, 0.0, Vl)
        final_Vl = jnp.where(jnp.any(world_terminal, axis=1), 0.0, next_Vl[:, -1])
        task_Tp1_Vl = jnp.concatenate([Vl_eff, final_Vl[:, None]], axis=1)

        safety_target, task_target = jax.vmap(
            lambda hs, loss, vhs, vls: compute_dec_ocp_gae(
                Tah_hs=hs, T_l=loss, Tp1ah_Vh=vhs,
                Tp1_Vl=vls, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(safety_cost_eff, task_loss, safety_Tp1_Vh, task_Tp1_Vl)

        task_cost_adv = task_target - Vl
        world_count = jnp.maximum(jnp.sum(valid_world, axis=1, keepdims=True), 1.0)
        task_mean = jnp.sum(jnp.where(valid_world, task_cost_adv, 0.0), axis=1, keepdims=True) / world_count
        task_var = jnp.sum(
            jnp.where(valid_world, jnp.square(task_cost_adv - task_mean), 0.0),
            axis=1, keepdims=True) / world_count
        task_cost_adv = (task_cost_adv - task_mean) / jnp.sqrt(task_var + 1e-8)

        terminal_Vh = jnp.maximum(rollout.costs, rollout.next_costs)
        actor_next_Vh = jnp.where(rollout.terminated[..., None], terminal_Vh, next_Vh)
        deriv = (actor_next_Vh - Vh) / self.dt + self.cbf_alpha * Vh
        constraint_violation = jnp.maximum(deriv + self.cbf_eps, 0.0)
        violation = jnp.max(constraint_violation, axis=-1)
        safe_derivative = jnp.all(deriv <= 0, axis=-1)
        policy_adv = jnp.where(
            safe_derivative, -task_cost_adv[..., None],
            -self.cbf_weight * violation)

        def vl_loss(params):
            pred = jax.vmap(jax.vmap(lambda g: self.Vl.get_value(
                params, g, self.Vl_rnn_state)[0].squeeze()))(rollout.critic_graph)
            return _masked_mean(optax.l2_loss(pred, task_target), valid_world)
        vl_value, vl_grad = jax.value_and_grad(vl_loss)(Vl_state.params)
        vl_grad_norm = optax.global_norm(vl_grad)
        Vl_state = Vl_state.apply_gradients(grads=vl_grad)

        def vh_loss(params):
            pred = jax.vmap(jax.vmap(lambda g: self.Vh.get_value(
                params, g, self.rnn_state)[0]))(safety_rollout.graph)
            return _masked_mean(
                optax.l2_loss(pred, safety_target), safety_rollout.alive_mask[..., None])
        vh_value, vh_grad = jax.value_and_grad(vh_loss)(Vh_state.params)
        vh_grad_norm = optax.global_norm(vh_grad)
        Vh_state = Vh_state.apply_gradients(grads=vh_grad)

        key, entropy_key = jr.split(key)
        keys = jr.split(entropy_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        def policy_loss(params):
            def eval_one(graph, action, key):
                return self.policy.eval_action(params, graph, action, self.rnn_state, key)[:2]
            logp, entropy = jax.vmap(jax.vmap(eval_one))(
                rollout.graph, rollout.actions, keys)
            ratio = jnp.exp(logp - rollout.log_pis)
            unclipped = ratio * policy_adv
            clipped = jnp.clip(ratio, 1-self.clip_eps, 1+self.clip_eps) * policy_adv
            objective = jnp.minimum(unclipped, clipped)
            loss = -_masked_mean(objective, alive) - self.coef_ent * _masked_mean(entropy, alive)
            clip_fraction = _masked_mean(jnp.abs(ratio - 1) > self.clip_eps, alive)
            return loss, (entropy, clip_fraction)
        (policy_value, (entropy, clip_fraction)), policy_grad = jax.value_and_grad(
            policy_loss, has_aux=True)(policy_state.params)
        policy_grad_norm = optax.global_norm(policy_grad)
        policy_state = policy_state.apply_gradients(grads=policy_grad)
        info = {
            "policy/loss": policy_value, "policy/entropy": _masked_mean(entropy, alive),
            "policy/clip_fraction": clip_fraction, "Vl/loss": vl_value, "Vh/loss": vh_value,
            "policy/grad_norm": policy_grad_norm, "Vl/grad_norm": vl_grad_norm,
            "Vh/grad_norm": vh_grad_norm,
            "Vh/pred_mean": _masked_mean(safety_Vh, safety_rollout.alive_mask[..., None]),
            "Vh/pred_min": jnp.min(jnp.where(safety_rollout.alive_mask[..., None], safety_Vh, jnp.inf)),
            "Vh/pred_max": jnp.max(jnp.where(safety_rollout.alive_mask[..., None], safety_Vh, -jnp.inf)),
            "Vh/target_mean": _masked_mean(safety_target, safety_rollout.alive_mask[..., None]),
            "Vh/target_min": jnp.min(jnp.where(safety_rollout.alive_mask[..., None], safety_target, jnp.inf)),
            "Vh/target_max": jnp.max(jnp.where(safety_rollout.alive_mask[..., None], safety_target, -jnp.inf)),
            "Vh/vehicle_pred_mean": _masked_mean(safety_Vh[..., 0], safety_rollout.alive_mask),
            "Vh/road_pred_mean": _masked_mean(safety_Vh[..., 1], safety_rollout.alive_mask),
            "Vh/vehicle_target_mean": _masked_mean(safety_target[..., 0], safety_rollout.alive_mask),
            "Vh/road_target_mean": _masked_mean(safety_target[..., 1], safety_rollout.alive_mask),
            "learning_rate/actor": self.lr_actor, "learning_rate/Vl": self.lr_Vl,
            "learning_rate/Vh": self.lr_Vh,
            "safety/unsafe_rate": _masked_mean(jnp.any(rollout.costs > 0, axis=-1), alive),
            "safety/cbf_violation": _masked_mean(violation, alive),
            "safety/safe_derivative_rate": _masked_mean(safe_derivative, alive),
            "adv/task_abs_mean": _masked_mean(jnp.abs(task_cost_adv), valid_world),
            "adv/cbf_abs_mean": _masked_mean(violation, alive),
            "adv/cbf_branch_rate": _masked_mean(~safe_derivative, alive),
        }
        return policy_state, Vl_state, Vh_state, key, info

    def update(self, rollout: GPUDriveRollout, safety_rollout: Optional[GPUDriveRollout] = None):
        safety_rollout = rollout if safety_rollout is None else safety_rollout
        n_worlds = int(rollout.actions.shape[0])
        if n_worlds % min(self.minibatch_worlds, n_worlds) != 0:
            raise ValueError(
                f"num_worlds={n_worlds} must be divisible by "
                f"minibatch_worlds={self.minibatch_worlds}")
        minibatch = min(self.minibatch_worlds, n_worlds)
        infos = []
        self.last_update_count = 0
        for _ in range(self.epoch_ppo):
            permutation_key, self.key = jr.split(self.key)
            permutation = jr.permutation(permutation_key, n_worlds)
            for start in range(0, n_worlds, minibatch):
                indices = permutation[start:start + minibatch]
                rollout_batch = jax.tree_util.tree_map(lambda value: value[indices], rollout)
                safety_batch = jax.tree_util.tree_map(lambda value: value[indices], safety_rollout)
                (self.policy_state, self.Vl_state, self.Vh_state,
                 self.key, info) = self._update_jit(
                    self.policy_state, self.Vl_state, self.Vh_state, self.key,
                    rollout_batch, safety_batch)
                infos.append(info)
                self.last_update_count += 1
        return jax.tree_util.tree_map(lambda *values: jnp.mean(jnp.stack(values)), *infos)

    def save(self, directory, step, counters=None):
        path = os.path.join(directory, str(step)); os.makedirs(path, exist_ok=True)
        def state_payload(state):
            return {"step": state.step, "params": state.params, "opt_state": state.opt_state}
        payload = {"policy": state_payload(self.policy_state),
                   "Vl": state_payload(self.Vl_state), "Vh": state_payload(self.Vh_state),
                   "key": self.key, "counters": counters or {}}
        target = os.path.join(path, "training_state.pkl")
        temporary = target + ".tmp"
        with open(temporary, "wb") as handle:
            pickle.dump(jax.device_get(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def load(self, directory, step):
        with open(os.path.join(directory, str(step), "training_state.pkl"), "rb") as handle:
            payload = pickle.load(handle)
        def restore(state, saved):
            expected = jax.tree_util.tree_map(lambda value: value.shape, state.params)
            found = jax.tree_util.tree_map(lambda value: value.shape, saved["params"])
            if jax.tree_util.tree_structure(expected) != jax.tree_util.tree_structure(found) or \
                    jax.tree_util.tree_leaves(expected) != jax.tree_util.tree_leaves(found):
                raise ValueError("checkpoint parameter structure is incompatible with the configured model")
            return state.replace(step=saved["step"], params=saved["params"], opt_state=saved["opt_state"])
        self.policy_state = restore(self.policy_state, payload["policy"])
        self.Vl_state = restore(self.Vl_state, payload["Vl"])
        self.Vh_state = restore(self.Vh_state, payload["Vh"])
        self.key = jnp.asarray(payload["key"])
        return payload.get("counters", {})
