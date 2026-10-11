"""InforMARL trained through a model-based, certificate-allocated MPE filter.

PPO samples nominal actions. The environment transition includes the fixed
CBF filter, so Rollout.actions/log_pis retain the *nominal* policy sample.
Only executed actions produce rewards and next states. No derivative through
the LP/QP or fictitious log-probability of the projected action is used.
"""
import functools as ft
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .informarl import InforMARL
from .module.mpe_cbf import MPECBFConfig, MPEPairwiseCBF
from ..trainer.data import CBFRollout, Rollout


def cbf_action_imitation_loss(predicted, executed, intervention_mask):
    """Masked mean squared error, with no gradient into the QP target."""
    target = jax.lax.stop_gradient(executed)
    mask = jax.lax.stop_gradient(intervention_mask).astype(predicted.dtype)
    error = jnp.mean(jnp.square(predicted - target), axis=-1)
    return jnp.sum(mask * error) / jnp.maximum(jnp.sum(mask), 1.0)


def _stack(trees):
    return jtu.tree_map(lambda *leaves: jnp.stack(leaves), *trees)


def _summarize(records, prefix):
    result = {}
    for key in records[0]:
        values = np.asarray([record[key] for record in records], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            result[f"{prefix}/{key}"] = float(values.mean())
    return result


class InforMARLCBF(InforMARL):
    """InforMARL PPO plus supervised CBF actions and CPU-filtered rollouts."""

    def __init__(self, *args, pairwise_cbf_mode="certificate",
                 pairwise_cbf_alpha1=3.0, pairwise_cbf_alpha2=3.0,
                 pairwise_cbf_safety_margin=0.0, pairwise_cbf_sensing_radius=None,
                 pairwise_cbf_slack_penalty=1e6, pairwise_cbf_witness_fallback=False,
                 cbf_action_loss_coef=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(cbf_action_loss_coef) or cbf_action_loss_coef < 0:
            raise ValueError("cbf_action_loss_coef must be finite and nonnegative")
        self.cbf_action_loss_coef = cbf_action_loss_coef
        self.cbf_config = MPECBFConfig(pairwise_cbf_alpha1, pairwise_cbf_alpha2,
                                      pairwise_cbf_safety_margin, pairwise_cbf_sensing_radius)
        self.cbf_mode = pairwise_cbf_mode
        self.cbf_solver_kwargs = dict(slack_penalty=pairwise_cbf_slack_penalty,
                                      witness_fallback=pairwise_cbf_witness_fallback)
        self._env_functions = {}
        self._filters = {}
        self._environment(self._env)  # validate MPE support before training
        self._sample_nominal = jax.jit(self.policy.sample_action)
        self._act_nominal = jax.jit(self.policy.get_action)
        self.collection_filter_metrics = {}
        self.eval_filter_metrics = {}
        self.last_executed_actions = None
        # Base class creates a jitted scan using self.step. The new collector
        # deliberately uses a host loop because SciPy/CVXPY cannot be traced.

    @property
    def config(self):
        return super().config | {
            "pairwise_cbf_mode": self.cbf_mode,
            "pairwise_cbf_alpha1": self.cbf_config.alpha1,
            "pairwise_cbf_alpha2": self.cbf_config.alpha2,
            "pairwise_cbf_safety_margin": self.cbf_config.safety_margin,
            "pairwise_cbf_sensing_radius": self.cbf_config.sensing_radius,
            "pairwise_cbf_slack_penalty": self.cbf_solver_kwargs["slack_penalty"],
            "pairwise_cbf_witness_fallback": self.cbf_solver_kwargs["witness_fallback"],
            "cbf_action_loss_coef": self.cbf_action_loss_coef,
        }

    def _environment(self, env):
        if env not in self._filters:
            self._filters[env] = MPEPairwiseCBF(env, self.cbf_mode, self.cbf_config,
                                               **self.cbf_solver_kwargs)
            self._env_functions[env] = (jax.jit(env.reset), jax.jit(env.step))
        return self._filters[env], self._env_functions[env]

    def act(self, graph, rnn_state, params=None):
        """Host-only deterministic execution; use the host eval rollout hook."""
        params = self.params if params is None else params
        nominal, state = self._act_nominal(params["policy"], graph, rnn_state)
        action, _ = self._filters[self._env](graph, np.asarray(nominal))
        return jnp.asarray(action, dtype=nominal.dtype), state

    def _rollout(self, env, params, key, *, training, stochastic=False):
        filter_, (reset, step_env) = self._environment(env)
        if training:
            key_x0, _, key = jax.random.split(key, 3)  # match InforMARL collection
        else:
            key_x0, key = jax.random.split(key)
        graph = reset(key_x0)
        if env.params["n_obs"] == 0:
            # MPE.reset stores an empty array; MPE.step stores None. Keep the
            # metadata tree consistent for stacked zero-obstacle rollouts.
            graph = graph._replace(env_states=graph.env_states._replace(obs=None))
        carry = self.init_rnn_state
        transitions, executed, metrics = [], [], []
        for action_key in jax.random.split(key, env.max_episode_steps):
            if training or stochastic:
                nominal, log_pi, new_carry = self._sample_nominal(params["policy"], graph, carry, action_key)
            else:
                nominal, new_carry = self._act_nominal(params["policy"], graph, carry)
                log_pi = None
            nominal_host = np.asarray(nominal)
            start = perf_counter()
            safe, info = filter_(graph, nominal_host)
            elapsed = perf_counter() - start
            safe = jnp.asarray(safe, dtype=nominal.dtype)
            next_graph, reward, cost, done, _ = step_env(graph, safe)
            # Training: policy density belongs to nominal action. Evaluation:
            # executed action is retained for rendering/replaying the episode.
            transitions.append(Rollout(graph, nominal if training else safe, carry,
                                       reward, cost, done, log_pi if training else None, next_graph))
            executed.append(safe)
            physical = np.asarray(graph.type_states(env.AGENT, env.num_agents))
            predicted = physical + env.dt * np.concatenate([physical[:, 2:], 10 * np.asarray(safe)], axis=-1)
            actual = np.asarray(next_graph.type_states(env.AGENT, env.num_agents))
            delta = np.linalg.norm(np.asarray(safe) - nominal_host, axis=-1)
            metrics.append({
                "action_correction_mean": delta.mean(),
                "modified_agent_fraction": (delta > 1e-4).mean(),
                "infeasible_step_rate": info["infeasible"],
                "solver_failed_step_rate": info["solver_failed"],
                "solver_retry_step_rate": info["solver_retries"] > 0,
                "fallback_step_rate": info["fallback"],
                "reserve_mean": info["reserve"],
                "allocation_margin_mean": info["allocation_margin"],
                "constraint_violation_mean": info["max_violation"],
                "state_clipped_step_rate": not np.allclose(actual, predicted, atol=1e-6, rtol=0),
                "filter_ms_mean": elapsed * 1000,
            })
            graph, carry = next_graph, new_carry
        return _stack(transitions), jnp.stack(executed), metrics

    def collect(self, params, b_key):
        rollouts, executed, metrics = [], [], []
        for key in b_key:
            rollout, actions, diagnostics = self._rollout(self._env, params, key, training=True)
            rollouts.append(rollout)
            executed.append(actions)
            metrics.extend(diagnostics)
        self.last_executed_actions = jnp.stack(executed)
        self.collection_filter_metrics = _summarize(metrics, "cbf/collection")
        nominal_rollout = _stack(rollouts)
        mask = jnp.linalg.norm(self.last_executed_actions - nominal_rollout.actions, axis=-1) > 1e-4
        return CBFRollout(*nominal_rollout, self.last_executed_actions, mask)

    def update(self, rollout, step):
        # The parent invokes the opt-in actor auxiliary loss below. GAE,
        # clipped PPO objective, optimizer and checkpoint format are retained.
        return super().update(rollout, step) | self.collection_filter_metrics

    def actor_auxiliary_loss(self, params, rollout, rnn_chunk_ids, graphs, rnn_states):
        # Use the same current-parameter RNN histories as the PPO forward pass,
        # not stale behavior-policy hidden states. get_action returns tanh mean.
        predict = ft.partial(self.policy.get_action, params)
        predicted, _ = jax.vmap(jax.vmap(jax.vmap(predict)))(graphs, rnn_states)
        targets = rollout.executed_actions[:, rnn_chunk_ids]
        mask = rollout.intervention_mask[:, rnn_chunk_ids]
        loss = cbf_action_imitation_loss(predicted, targets, mask)
        weighted = self.cbf_action_loss_coef * loss
        return weighted, {
            "policy/cbf_action_loss": loss,
            "policy/cbf_action_loss_weighted": weighted,
            "policy/cbf_supervised_fraction": mask.mean(),
        }

    def eval_rollout_host(self, env, params, key, stochastic=False):
        rollout, _, _ = self._rollout(env, params, key, training=False, stochastic=stochastic)
        return rollout

    def eval_rollouts_host(self, env, params, keys):
        rollouts, metrics = [], []
        for key in keys:
            rollout, _, diagnostics = self._rollout(env, params, key, training=False)
            rollouts.append(rollout)
            metrics.extend(diagnostics)
        result = _stack(rollouts)
        self.eval_filter_metrics = _summarize(metrics, "cbf/eval")

        def goal_distance(graph):
            positions = graph.type_states(env.AGENT, env.num_agents)[:, :2]
            goals = env.render_task_goal_positions(graph)
            if goals is None:
                goals = graph.type_states(env.GOAL, env.num_goals)[:, :2]
            if type(env).__name__ == "MPETarget":
                return jnp.linalg.norm(positions - goals, axis=-1)
            return jnp.linalg.norm(goals[:, None] - positions[None], axis=-1).min(axis=1)

        distance = jax.jit(jax.vmap(jax.vmap(goal_distance)))(result.next_graph)
        reached = np.asarray(distance) <= env.params["dist2goal"]
        self.eval_filter_metrics |= {
            "eval/task/reach_ever_rate": float(reached.any(axis=1).mean()),
            "eval/task/reach_final_rate": float(reached[:, -1].mean()),
            "eval/task/final_goal_distance": float(np.asarray(distance)[:, -1].mean()),
        }
        return result

    def diagnostic_metrics(self, rollout):
        return dict(self.eval_filter_metrics)
