from .base import Algorithm
from .informarl import InforMARL
from .informarl_manifold import InforMARLManifold
from .informarl_deep_qp import InforMARLDeepQP
from .informarl_lagr import InforMARLLagr
from .dgppo import DGPPO
from .hcbfcrpo import HCBFCRPO


def make_algo(algo: str, **kwargs) -> Algorithm:
    if algo == 'informarl':
        return InforMARL(**kwargs)
    elif algo == 'informarl_manifold':
        return InforMARLManifold(**kwargs)
    elif algo == 'informarl_deep_qp':
        return InforMARLDeepQP(**kwargs)
    elif algo == 'informarl_lagr':
        return InforMARLLagr(**kwargs)
    elif algo == 'dgppo':
        return DGPPO(**kwargs)
    elif algo == 'hcbfcrpo':
        return HCBFCRPO(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
