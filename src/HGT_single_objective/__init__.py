"""Single-objective (makespan-only) HGT package.

Mirrors ``src.HGT`` but with TWT removed: job features drop from 7 to 4,
the default reward is the DP lower-bound reward on the makespan lower bound,
and there are no dynamic events or curriculum learning.
"""

from .hgt_env_so import FJSPEnvHGTSingleObj, HGTEnvStateSingleObj
from .hgt_network_so import HGTSingleObj
from .hgt_trainer_so import HGTSingleObjTrainer

__all__ = [
    'FJSPEnvHGTSingleObj',
    'HGTEnvStateSingleObj',
    'HGTSingleObj',
    'HGTSingleObjTrainer',
]
