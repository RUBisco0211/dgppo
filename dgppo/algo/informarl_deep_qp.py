"""InforMARL with a frozen Graph-HJ critic and CRPO-style policy updates."""

import functools as ft
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from flax.training.train_state import TrainState
from jax import lax

from .informarl import InforMARL
from .module.deep_qp_safety import (
    DeepQPSafetyConfig,
    GraphHJSafetyCritic,
    safety_lambda_at,
)
from .utils import compute_dec_ocp_gae
from ..env.safety_constraint import (
    safety_constraint,
    safety_constraint_metadata,
    safety_node_feature_mask,
)
from ..env.vmas.vmas_navigation import VMASNavigation
from ..trainer.data import Rollout
from ..utils.graph import GraphsTuple
from ..utils.typing import Array, Params
from ..utils.utils import tree_index


def aggregate_owner_violation(violation: Array, action_mask: Array) -> Array:
    """Attribute each local constraint violation to participating action owners."""
    return jnp.max(
        jnp.where(action_mask, violation[..., :, None], 0.0),
        axis=-2,
    )


class InforMARLHJCRPO(InforMARL):
    """CTDE PPO with CRPO switching driven by a pretrained Graph-HJ critic.

    Execution is the ordinary decentralized InforMARL actor. During centralized
    training, the frozen critic evaluates each local HJ constraint using the
    complete joint action of the corresponding graph neighborhood.
    """

    def __init__(
            self,
            *args,
            deep_qp_checkpoint: Optional[str] = None,
            deep_qp_gnn_layers: int = 1,
            deep_qp_gnn_out_dim: int = 64,
            deep_qp_hidden_dim: int = 256,
            deep_qp_hidden_layers: int = 2,
            deep_qp_lr: float = 3e-4,
            deep_qp_lr_final: float = 3e-6,
            deep_qp_tau: float = 0.005,
            deep_qp_lambda_init: Optional[float] = None,
            deep_qp_lambda_final: Optional[float] = None,
            deep_qp_lambda_decay_steps: int = 1_000_000,
            deep_qp_constraint_scale: float = 0.5,
            deep_qp_agent_margin: float = 0.02,
            deep_qp_obstacle_margin: float = 0.02,
            deep_qp_braking_accel: Optional[float] = None,
            hj_cbf_alpha: float = 1.0,
            hj_cbf_margin: float = 0.0,
            hj_cbf_eps: float = 0.0,
            crpo_threshold: float = 0.0,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self._env, VMASNavigation):
            raise ValueError(
                "InforMARLHJCRPO currently targets VMASNavigation and "
                "VMASNavigationObs scenarios."
            )
        if hj_cbf_alpha < 0.0:
            raise ValueError("hj_cbf_alpha must be non-negative")
        if hj_cbf_eps < 0.0 or crpo_threshold < 0.0:
            raise ValueError("hj_cbf_eps and crpo_threshold must be non-negative")

        dt = self._env.dt
        critic_config = DeepQPSafetyConfig(
            gnn_layers=deep_qp_gnn_layers,
            gnn_out_dim=deep_qp_gnn_out_dim,
            hidden_dim=deep_qp_hidden_dim,
            hidden_layers=deep_qp_hidden_layers,
            learning_rate=deep_qp_lr,
            learning_rate_final=deep_qp_lr_final,
            target_tau=deep_qp_tau,
            dt=dt,
            lambda_init=0.1 / dt if deep_qp_lambda_init is None else deep_qp_lambda_init,
            lambda_final=0.0001 / dt if deep_qp_lambda_final is None else deep_qp_lambda_final,
            lambda_decay_steps=deep_qp_lambda_decay_steps,
            constraint_scale=deep_qp_constraint_scale,
            max_grad_norm=self.max_grad_norm,
        )
        action_lower, action_upper = self._env.action_lim()
        self.safety_critic = GraphHJSafetyCritic(
            action_dim=self.action_dim,
            n_agents=self.n_agents,
            action_lower=action_lower,
            action_upper=action_upper,
            node_feature_mask=safety_node_feature_mask(self._env),
            config=critic_config,
        )
        safety_key, self.key = jr.split(self.key)
        self.safety_train_state = self.safety_critic.initialize(
            safety_key, self.nominal_graph
        )

        self.deep_qp_agent_margin = deep_qp_agent_margin
        self.deep_qp_obstacle_margin = deep_qp_obstacle_margin
        self.deep_qp_braking_accel = deep_qp_braking_accel
        self.hj_cbf_alpha = hj_cbf_alpha
        self.hj_cbf_margin = hj_cbf_margin
        self.hj_cbf_eps = hj_cbf_eps
        self.crpo_threshold = crpo_threshold

        if deep_qp_checkpoint is not None:
            self.load_safety_checkpoint(deep_qp_checkpoint)

    @property
    def config(self) -> dict:
        return super().config | {
            "deep_qp_agent_margin": self.deep_qp_agent_margin,
            "deep_qp_obstacle_margin": self.deep_qp_obstacle_margin,
            "deep_qp_braking_accel": self.deep_qp_braking_accel,
            "hj_cbf_alpha": self.hj_cbf_alpha,
            "hj_cbf_margin": self.hj_cbf_margin,
            "hj_cbf_eps": self.hj_cbf_eps,
            "crpo_threshold": self.crpo_threshold,
            **{
                f"deep_qp_{key}": value
                for key, value in self.safety_critic.config.to_dict().items()
            },
        }

    def safety_constraint(self, graph: GraphsTuple) -> Array:
        return safety_constraint(
            self._env,
            graph,
            agent_margin=self.deep_qp_agent_margin,
            obstacle_margin=self.deep_qp_obstacle_margin,
            braking_accel=self.deep_qp_braking_accel,
            maximum_margin=self.safety_critic.config.constraint_scale,
        )

    def update(self, rollout: Rollout, step: int) -> dict:
        del step
        graph_clean = rollout.graph._replace(env_states=None)
        next_graph_clean = rollout.next_graph._replace(env_states=None)
        rollout = rollout._replace(graph=graph_clean, next_graph=next_graph_clean)

        update_info = {}
        assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
        for _ in range(self.epoch_ppo):
            idx = np.arange(rollout.dones.shape[0])
            np.random.shuffle(idx)
            rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
            rnn_chunk_ids = jnp.array(jnp.array_split(
                rnn_chunk_ids, rollout.dones.shape[1] // self.rnn_step
            ))
            batch_idx = jnp.array(jnp.array_split(
                idx, idx.shape[0] // (self.batch_size // rollout.dones.shape[1])
            ))
            self.Vl_train_state, self.policy_train_state, update_info = self.update_inner(
                self.Vl_train_state,
                self.policy_train_state,
                self.safety_train_state.target_params,
                rollout,
                batch_idx,
                rnn_chunk_ids,
            )
        return update_info

    @ft.partial(
        jax.jit,
        static_argnums=(0,),
        donate_argnames=("Vl_train_state", "policy_train_state"),
    )
    def update_inner(
            self,
            Vl_train_state: TrainState,
            policy_train_state: TrainState,
            safety_params: Params,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
    ) -> Tuple[TrainState, TrainState, dict]:
        b, T, a, _ = rollout.actions.shape

        bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
            ft.partial(
                self.scan_Vl,
                init_Vl_rnn_state=self.init_Vl_rnn_state,
                Vl_params=Vl_train_state.params,
            )
        )(rollout)

        def final_Vl_fn(graph, rnn_state):
            value, _ = self.Vl.get_value(
                Vl_train_state.params, tree_index(graph, -1), rnn_state
            )
            return value.squeeze(0).squeeze(0)

        b_final_Vl = jax.vmap(final_Vl_fn)(
            rollout.next_graph, final_Vl_rnn_states
        )
        bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape == (b, T + 1)

        bTp1ah_Vh = bTp1_Vl[:, :, None, None].repeat(
            self.n_agents, axis=-2
        ).repeat(rollout.costs.shape[-1], axis=-1)
        _, bT_Ql = jax.vmap(ft.partial(
            compute_dec_ocp_gae,
            disc_gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        ))(
            Tah_hs=rollout.costs,
            T_l=-rollout.rewards,
            Tp1ah_Vh=bTp1ah_Vh,
            Tp1_Vl=bTp1_Vl,
        )
        bT_Al = bT_Ql - bT_Vl
        bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (
            bT_Al.std(axis=1, keepdims=True) + 1e-8
        )
        bTa_task_A = -bT_Al[:, :, None].repeat(self.n_agents, axis=-1)

        bTa_constraint = jax.vmap(jax.vmap(self.safety_constraint))(rollout.graph)
        safety_lambda = safety_lambda_at(
            self.safety_critic.config, self.safety_train_state.online.step
        )

        def certify_one(graph, constraint):
            return self.safety_critic.certify(
                safety_params, graph, constraint, safety_lambda
            )

        certificate = jax.vmap(jax.vmap(certify_one))(
            rollout.graph, bTa_constraint
        )
        bTa_residual = self.safety_critic.cbf_residual(
            certificate,
            rollout.actions,
            alpha=self.hj_cbf_alpha,
            margin=self.hj_cbf_margin,
        )
        bTa_violation = jax.nn.relu(self.hj_cbf_eps - bTa_residual)

        # Action owner j receives the worst local violation among all V_i that
        # explicitly depend on u_j. This coupling exists only during CTDE.
        bTa_owner_violation = aggregate_owner_violation(
            bTa_violation, certificate.action_mask
        )
        owner_mean = bTa_owner_violation.mean(axis=1, keepdims=True)
        owner_std = bTa_owner_violation.std(axis=1, keepdims=True)
        bTa_safety_A = -(bTa_owner_violation - owner_mean) / (owner_std + 1e-8)

        constraint_estimate = bTa_violation.max(axis=-1).mean()
        use_safety_update = constraint_estimate > self.crpo_threshold
        bTa_A = jnp.where(use_safety_update, bTa_safety_A, bTa_task_A)
        assert bTa_A.shape == (b, T, a)

        def update_fn(carry, idx):
            Vl_model, policy_model = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            Vl_model, Vl_info = self.update_Vl(
                Vl_model,
                rollout_batch,
                bT_Ql[idx],
                bT_Vl_rnn_states[idx],
                rnn_chunk_ids,
            )
            policy_model, policy_info = self.update_policy(
                policy_model, rollout_batch, bTa_A[idx], rnn_chunk_ids
            )
            return (Vl_model, policy_model), (Vl_info | policy_info)

        (Vl_train_state, policy_train_state), info = lax.scan(
            update_fn, (Vl_train_state, policy_train_state), batch_idx
        )
        info = jtu.tree_map(lambda x: x[-1], info) | {
            "hj_crpo/constraint_estimate": constraint_estimate,
            "hj_crpo/safety_update": use_safety_update.astype(jnp.float32),
            "hj_crpo/violation_mean": bTa_violation.mean(),
            "hj_crpo/violation_max": bTa_violation.max(),
            "hj_crpo/residual_min": bTa_residual.min(),
            "hj_crpo/value_min": certificate.value.min(),
            "hj_crpo/neighborhood_density": certificate.action_mask.mean(),
        }
        return Vl_train_state, policy_train_state, info

    def _checkpoint_metadata(self) -> dict:
        return safety_constraint_metadata(
            self._env,
            agent_margin=self.deep_qp_agent_margin,
            obstacle_margin=self.deep_qp_obstacle_margin,
            braking_accel=self.deep_qp_braking_accel,
        )

    def save_safety_checkpoint(self, path: str | Path) -> None:
        self.safety_critic.save_checkpoint(
            self.safety_train_state,
            path,
            metadata=self._checkpoint_metadata(),
        )

    def load_safety_checkpoint(self, path: str | Path) -> None:
        self.safety_train_state = self.safety_critic.load_checkpoint(
            self.safety_train_state,
            path,
            expected_metadata=self._checkpoint_metadata(),
        )

    def save(self, save_dir: str, step: int):
        super().save(save_dir, step)
        self.save_safety_checkpoint(
            Path(save_dir) / str(step) / "deep_qp_safety.pkl"
        )

    def load(self, load_dir: str, step: int):
        super().load(load_dir, step)
        safety_path = Path(load_dir) / str(step) / "deep_qp_safety.pkl"
        if not safety_path.exists():
            raise FileNotFoundError(
                f"HJ-CRPO checkpoint is missing frozen critic: {safety_path}"
            )
        self.load_safety_checkpoint(safety_path)


# Preserve the experimental name used by early local configs.
InforMARLDeepQP = InforMARLHJCRPO
