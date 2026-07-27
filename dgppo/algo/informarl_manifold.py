import functools as ft
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .informarl import InforMARL
from .module.manifold_filter import (
    lidar_manifold_init_slack,
    lidar_manifold_project,
    lidar_manifold_project_with_slack,
)
from ..env.lidar_env.base import LidarEnv
from ..trainer.data import Rollout
from ..utils.graph import GraphsTuple
from ..utils.utils import jax_vmap
from ..utils.typing import Action, Array, PRNGKey, Params


class InforMARLManifold(InforMARL):
    """InforMARL with a one-step constraint-manifold safety filter.

    The actor still proposes a nominal single-step action. Before execution,
    the nominal action is projected through a model-based LidarEnv manifold
    filter. The rollout stores the executed action so PPO optimizes the action
    that actually produced the observed reward.
    """

    def __init__(
            self,
            *args,
            manifold_top_k_obs: int = 3,
            manifold_safety_margin: float = 0.02,
            manifold_braking_accel: float = 1.0,
            manifold_velocity_margin: float = 0.02,
            manifold_contraction_gain: float = 30.0,
            manifold_slack_min: float = 0.1,
            manifold_slack_beta: float = 1.0,
            manifold_slack_weight: float = 10.0,
            manifold_reg: float = 1e-5,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self._env, LidarEnv):
            raise ValueError("InforMARLManifold currently supports only LidarEnv scenarios.")
        self.manifold_top_k_obs = manifold_top_k_obs
        self.manifold_safety_margin = manifold_safety_margin
        self.manifold_braking_accel = manifold_braking_accel
        self.manifold_velocity_margin = manifold_velocity_margin
        self.manifold_contraction_gain = manifold_contraction_gain
        self.manifold_slack_min = manifold_slack_min
        self.manifold_slack_beta = manifold_slack_beta
        self.manifold_slack_weight = manifold_slack_weight
        self.manifold_reg = manifold_reg

        def rollout_fn_single_(cur_params, cur_key):
            return self.rollout_with_filter(cur_params, cur_key)

        def rollout_fn_(cur_params, cur_keys):
            return jax.vmap(ft.partial(rollout_fn_single_, cur_params))(cur_keys)

        self.rollout_fn = jax.jit(rollout_fn_)

    @property
    def config(self) -> dict:
        return super().config | {
            "manifold_top_k_obs": self.manifold_top_k_obs,
            "manifold_safety_margin": self.manifold_safety_margin,
            "manifold_braking_accel": self.manifold_braking_accel,
            "manifold_velocity_margin": self.manifold_velocity_margin,
            "manifold_contraction_gain": self.manifold_contraction_gain,
            "manifold_slack_min": self.manifold_slack_min,
            "manifold_slack_beta": self.manifold_slack_beta,
            "manifold_slack_weight": self.manifold_slack_weight,
            "manifold_reg": self.manifold_reg,
        }

    def project_action(self, graph: GraphsTuple, action_ref: Action) -> Action:
        action_safe = lidar_manifold_project(
            env=self._env,
            graph=graph,
            action_ref=action_ref,
            top_k_obs=self.manifold_top_k_obs,
            safety_margin=self.manifold_safety_margin,
            braking_accel=self.manifold_braking_accel,
            velocity_margin=self.manifold_velocity_margin,
            contraction_gain=self.manifold_contraction_gain,
            slack_min=self.manifold_slack_min,
            slack_beta=self.manifold_slack_beta,
            slack_weight=self.manifold_slack_weight,
            reg=self.manifold_reg,
        )
        assert action_safe.shape == (self.n_agents, self.action_dim)
        return action_safe

    def init_filter_state(self, graph: GraphsTuple, env: Optional[LidarEnv] = None) -> Array:
        if env is None:
            env = self._env
        return lidar_manifold_init_slack(
            env=env,
            graph=graph,
            top_k_obs=self.manifold_top_k_obs,
            safety_margin=self.manifold_safety_margin,
            braking_accel=self.manifold_braking_accel,
            velocity_margin=self.manifold_velocity_margin,
            slack_min=self.manifold_slack_min,
        )

    def project_action_with_state(
            self,
            graph: GraphsTuple,
            action_ref: Action,
            filter_state: Array,
            env: Optional[LidarEnv] = None,
    ) -> Tuple[Action, Array]:
        if env is None:
            env = self._env
        action_safe, next_filter_state = lidar_manifold_project_with_slack(
            env=env,
            graph=graph,
            action_ref=action_ref,
            slack_state=filter_state,
            top_k_obs=self.manifold_top_k_obs,
            safety_margin=self.manifold_safety_margin,
            braking_accel=self.manifold_braking_accel,
            velocity_margin=self.manifold_velocity_margin,
            contraction_gain=self.manifold_contraction_gain,
            slack_min=self.manifold_slack_min,
            slack_beta=self.manifold_slack_beta,
            slack_weight=self.manifold_slack_weight,
            reg=self.manifold_reg,
        )
        assert action_safe.shape == (self.n_agents, self.action_dim)
        return action_safe, next_filter_state

    def rollout_with_filter(self, params: Params, key: PRNGKey) -> Rollout:
        key_x0, _, key = jax.random.split(key, 3)
        init_graph = self._env.reset(key_x0)
        init_filter_state = self.init_filter_state(init_graph)

        def body(data, key_):
            graph, rnn_state, filter_state = data
            action_ref, _, new_rnn_state = self.policy_train_state.apply_fn(
                params["policy"], graph, rnn_state, key_
            )
            action_safe, next_filter_state = self.project_action_with_state(graph, action_ref, filter_state)
            log_pi_safe, _, _ = self.policy.eval_action(
                params["policy"], graph, action_safe, rnn_state, key_
            )
            log_pi_safe = jnp.nan_to_num(log_pi_safe)
            next_graph, reward, cost, done, info = self._env.step(graph, action_safe)
            return (
                (next_graph, new_rnn_state, next_filter_state),
                (graph, action_safe, rnn_state, reward, cost, done, log_pi_safe, next_graph),
            )

        keys = jax.random.split(key, self._env.max_episode_steps)
        _, (graphs, actions_safe, rnn_states, rewards, costs, dones, log_pis, next_graphs) = jax.lax.scan(
            body,
            (init_graph, self.init_rnn_state, init_filter_state),
            keys,
            length=self._env.max_episode_steps,
        )
        return Rollout(graphs, actions_safe, rnn_states, rewards, costs, dones, log_pis, next_graphs)

    def eval_rollout_with_filter(self, env: LidarEnv, params: Params, key: PRNGKey) -> Rollout:
        key_x0, key = jax.random.split(key)
        init_graph = env.reset(key_x0)
        init_filter_state = self.init_filter_state(init_graph, env=env)

        def body(data, key_):
            graph, rnn_state, filter_state = data
            action_ref, new_rnn_state = self.policy.get_action(params["policy"], graph, rnn_state)
            action_safe, next_filter_state = self.project_action_with_state(
                graph, action_ref, filter_state, env=env
            )
            next_graph, reward, cost, done, info = env.step(graph, action_safe)
            return (
                (next_graph, new_rnn_state, next_filter_state),
                (graph, action_safe, rnn_state, reward, cost, done, None, next_graph),
            )

        keys = jax.random.split(key, env.max_episode_steps)
        _, (graphs, actions_safe, rnn_states, rewards, costs, dones, log_pis, next_graphs) = jax.lax.scan(
            body,
            (init_graph, self.init_rnn_state, init_filter_state),
            keys,
            length=env.max_episode_steps,
        )
        return Rollout(graphs, actions_safe, rnn_states, rewards, costs, dones, log_pis, next_graphs)

    def act(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            params: Optional[Params] = None,
    ) -> Tuple[Action, Array]:
        action_ref, rnn_state = super().act(graph, rnn_state, params)
        action_safe = self.project_action(graph, action_ref)
        return action_safe, rnn_state

    def step(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            key: PRNGKey,
            params: Optional[Params] = None,
    ) -> Tuple[Action, Array, Array]:
        if params is None:
            params = self.params
        rnn_state_in = rnn_state
        action_ref, _, rnn_state = self.policy_train_state.apply_fn(params["policy"], graph, rnn_state, key)
        assert action_ref.shape == (self.n_agents, self.action_dim)
        action_safe = self.project_action(graph, action_ref)
        log_pi_safe, _, _ = self.policy.eval_action(params["policy"], graph, action_safe, rnn_state_in, key)
        log_pi_safe = jnp.nan_to_num(log_pi_safe)
        return action_safe, log_pi_safe, rnn_state

    def update(self, rollout: Rollout, step: int) -> dict:
        action_reprojected = jax_vmap(jax_vmap(self.project_action))(rollout.graph, rollout.actions)
        action_delta = action_reprojected - rollout.actions
        filter_info = {
            "manifold/executed_action_norm": jnp.linalg.norm(rollout.actions, axis=-1).mean(),
            "manifold/reprojected_action_norm": jnp.linalg.norm(action_reprojected, axis=-1).mean(),
            "manifold/action_delta_norm": jnp.linalg.norm(action_delta, axis=-1).mean(),
            "manifold/action_clip_frac": (jnp.abs(rollout.actions) >= 1.0 - 1e-6).mean(),
            "manifold/action_nan_frac": jnp.isnan(rollout.actions).mean(),
            "manifold/reward_nan_frac": jnp.isnan(rollout.rewards).mean(),
        }
        return super().update(rollout, step) | filter_info
