"""Single-objective HGT policy network.

Identical to ``HGT`` except ``job_feature_dim`` defaults to 4 (was 7).
The trainer overrides ``config.job_feature_dim`` from
``env.job_fea_dim``, so this default only matters when the config
doesn't carry that attribute (e.g. when loading a checkpoint).
"""

from src.HGT.hgt_network import HGT


class HGTSingleObj(HGT):
    """HGT with 4-dim job features (drops weight / slack / due_date)."""

    def __init__(self, config):
        # Set job_feature_dim default to 4 if not explicitly provided.
        # ``HGT.__init__`` reads via ``getattr(config, "job_feature_dim", 7)``,
        # so we just ensure the attribute is present.
        if not hasattr(config, 'job_feature_dim'):
            config.job_feature_dim = 4
        super().__init__(config)


__all__ = ['HGTSingleObj']
