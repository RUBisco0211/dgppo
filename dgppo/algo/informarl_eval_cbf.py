"""Native InforMARL training with a pairwise CBF filter only at evaluation."""
import jax

from .informarl import InforMARL
from .informarl_cbf import InforMARLCBF
from .module.mpe_cbf import MPECBFConfig, MPEPairwiseCBF


class InforMARLEvalCBF(InforMARL):
    """Keep InforMARL collection/update untouched; filter deterministic eval."""

    # Reuse the previously tested host evaluation and its diagnostics. Its
    # training collector and auxiliary actor objective are deliberately absent.
    _environment = InforMARLCBF._environment
    _rollout = InforMARLCBF._rollout
    eval_rollout_host = InforMARLCBF.eval_rollout_host
    eval_rollouts_host = InforMARLCBF.eval_rollouts_host
    diagnostic_metrics = InforMARLCBF.diagnostic_metrics

    def __init__(self, *args, pairwise_cbf_mode="qp",
                 pairwise_cbf_alpha1=3.0, pairwise_cbf_alpha2=3.0,
                 pairwise_cbf_safety_margin=0.0, pairwise_cbf_sensing_radius=None,
                 pairwise_cbf_slack_penalty=1e6, pairwise_cbf_witness_fallback=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if pairwise_cbf_mode not in ("qp", "certificate"):
            raise ValueError("Eval-only comparison supports qp or certificate")
        self.cbf_mode = pairwise_cbf_mode
        self.cbf_config = MPECBFConfig(pairwise_cbf_alpha1, pairwise_cbf_alpha2,
                                       pairwise_cbf_safety_margin, pairwise_cbf_sensing_radius)
        self.cbf_solver_kwargs = dict(slack_penalty=pairwise_cbf_slack_penalty,
                                      witness_fallback=pairwise_cbf_witness_fallback)
        self._env_functions = {}
        self._filters = {}
        self._environment(self._env)
        self._sample_nominal = jax.jit(self.policy.sample_action)
        self._act_nominal = jax.jit(self.policy.get_action)
        self.eval_filter_metrics = {}

    @property
    def config(self):
        return super().config | {
            "eval_pairwise_cbf_mode": self.cbf_mode,
            "eval_pairwise_cbf_alpha1": self.cbf_config.alpha1,
            "eval_pairwise_cbf_alpha2": self.cbf_config.alpha2,
            "eval_pairwise_cbf_safety_margin": self.cbf_config.safety_margin,
            "eval_pairwise_cbf_sensing_radius": self.cbf_config.sensing_radius,
            "eval_pairwise_cbf_slack_penalty": self.cbf_solver_kwargs["slack_penalty"],
            "eval_pairwise_cbf_witness_fallback": self.cbf_solver_kwargs["witness_fallback"],
        }
