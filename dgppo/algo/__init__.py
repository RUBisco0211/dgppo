from .base import Algorithm

_LAZY_ALGORITHMS = {
    "InforMARL": (".informarl", "InforMARL"),
    "InforMARLCBF": (".informarl_cbf", "InforMARLCBF"),
    "InforMARLEvalCBF": (".informarl_eval_cbf", "InforMARLEvalCBF"),
    "InforMARLManifold": (".informarl_manifold", "InforMARLManifold"),
    "InforMARLDeepQP": (".informarl_deep_qp", "InforMARLDeepQP"),
    "InforMARLLagr": (".informarl_lagr", "InforMARLLagr"),
    "DGPPO": (".dgppo", "DGPPO"),
    "HCBFCRPO": (".hcbfcrpo", "HCBFCRPO"),
    "GCBFPlus": (".gcbf_plus", "GCBFPlus"),
    "AdversarialDGPPO": (".adv_dgppo", "AdversarialDGPPO"),
    "GPUDriveDGPPO": (".gpudrive_dgppo", "GPUDriveDGPPO"),
}


def __getattr__(name):
    """Preserve the package's direct-import API without eager optional imports."""
    if name not in _LAZY_ALGORITHMS:
        raise AttributeError(name)
    from importlib import import_module
    module_name, class_name = _LAZY_ALGORITHMS[name]
    value = getattr(import_module(module_name, __name__), class_name)
    globals()[name] = value
    return value


def make_algo(algo: str, **kwargs) -> Algorithm:
    if algo == 'informarl':
        from .informarl import InforMARL
        return InforMARL(**kwargs)
    elif algo == 'informarl_cbf':
        from .informarl_cbf import InforMARLCBF
        return InforMARLCBF(**kwargs)
    elif algo == 'informarl_eval_cbf':
        from .informarl_eval_cbf import InforMARLEvalCBF
        return InforMARLEvalCBF(**kwargs)
    elif algo == 'informarl_manifold':
        from .informarl_manifold import InforMARLManifold
        return InforMARLManifold(**kwargs)
    elif algo == 'informarl_deep_qp':
        from .informarl_deep_qp import InforMARLDeepQP
        return InforMARLDeepQP(**kwargs)
    elif algo == 'informarl_lagr':
        from .informarl_lagr import InforMARLLagr
        return InforMARLLagr(**kwargs)
    elif algo == 'dgppo':
        from .dgppo import DGPPO
        return DGPPO(**kwargs)
    elif algo == 'hcbfcrpo':
        from .hcbfcrpo import HCBFCRPO
        return HCBFCRPO(**kwargs)
    elif algo in ('gcbf+', 'gcbfplus'):
        from .gcbf_plus import GCBFPlus
        return GCBFPlus(**kwargs)
    elif algo in ('adversarial_dgppo', 'adv_dgppo'):
        from .adv_dgppo import AdversarialDGPPO
        return AdversarialDGPPO(**kwargs)
    elif algo == 'gpudrive_dgppo':
        from .gpudrive_dgppo import GPUDriveDGPPO
        return GPUDriveDGPPO(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
