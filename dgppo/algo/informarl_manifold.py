import functools as ft
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .informarl import InforMARL
from .module.manifold_filter import lidar_manifold_project
from ..env.lidar_env.base import LidarEnv
from ..trainer.data import Rollout
from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, PRNGKey, Params


class InforMARLManifold(InforMARL):
    """InforMARL with a one-step constraint-manifold safety filter.

    The actor still learns a nominal single-step action. Before execution, the
    nominal action is projected through a model-based LidarEnv manifold filter.
    The PPO log probability remains the nominal action log probability.
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

    def rollout_with_filter(self, params: Params, key: PRNGKey) -> Rollout:
        key_x0, _, key = jax.random.split(key, 3)
        init_graph = self._env.reset(key_x0)

        def body(data, key_):
            graph, rnn_state = data
            action_ref, log_pi, new_rnn_state = self.policy_train_state.apply_fn(
                params["policy"], graph, rnn_state, key_
            )
            action_safe = self.project_action(graph, action_ref)
            next_graph, reward, cost, done, info = self._env.step(graph, action_safe)
            return (
                (next_graph, new_rnn_state),
                (graph, action_ref, rnn_state, reward, cost, done, log_pi, next_graph),
            )

        keys = jax.random.split(key, self._env.max_episode_steps)
        _, (graphs, actions_ref, rnn_states, rewards, costs, dones, log_pis, next_graphs) = jax.lax.scan(
            body,
            (init_graph, self.init_rnn_state),
            keys,
            length=self._env.max_episode_steps,
        )
        return Rollout(graphs, actions_ref, rnn_states, rewards, costs, dones, log_pis, next_graphs)

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
        action_ref, log_pi, rnn_state = self.policy_train_state.apply_fn(params["policy"], graph, rnn_state, key)
        assert action_ref.shape == (self.n_agents, self.action_dim)
        action_safe = self.project_action(graph, action_ref)
        log_pi = jnp.nan_to_num(log_pi)
        return action_safe, log_pi, rnn_state
