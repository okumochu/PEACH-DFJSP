"""Single-objective (makespan-only) FJSP environment.

Subclass of ``FJSPEnvHGT`` that strips TWT from the feature/reward path and
uses the OLD/STATIC design (``use_dynamic_event_features=False`` ->
``mch_fea_dim = 6``, no ``is_broken`` column), mirroring
``train_graph_rl_static.py``:
- ``job_fea_dim = 4`` with DP lower-bound features
- ``_combine_objective`` returns the makespan LB only
- ``set_initial_data_static`` reads only ``job_length`` and ``op_pt``
- ``construct_job_features`` writes single-objective job features instead of 7

The parent env's TWT state (``job_weight=1``, ``job_due_date=inf`` set in
``initial_vars``) is left intact and harmless: with ``twt_weight=0`` and
infinite due dates, ``tardiness=0`` and the unchanged step() TWT update
contributes nothing.
"""

import copy
import numpy as np

from src.HGT.hgt_env import FJSPEnvHGT, HGTEnvState
from src.utils.event_utils import init_event_queues, merge_and_sort_events


HGTEnvStateSingleObj = HGTEnvState


class FJSPEnvHGTSingleObj(FJSPEnvHGT):
    """Single-objective FJSP environment (makespan only).

    Differences from the multi-objective parent:
    - OLD/STATIC design (``use_dynamic_event_features=False``): single-obj is
      always static, so the dynamic-event observability columns are dropped —
      machine ``is_broken`` (``mch_fea_dim = 6``, not 7). This mirrors
      ``train_graph_rl_static.py``.
    - ``job_fea_dim = 4`` with DP lower-bound features
    - DP lower-bound reward differences ``lb_makespan`` (no TWT term)
    - ``set_initial_data_static`` ignores ``job_weight`` / ``job_due_date``
    - TWT job features are not populated
    """

    def __init__(
        self,
        n_j,
        n_m,
        device,
    ):
        # Single-objective is ALWAYS the old/static design (mirrors
        # train_graph_rl_static.py): no dynamic events ever fire, so the
        # dynamic-event observability columns are dropped. With
        # use_dynamic_event_features=False the parent sets mch_fea_dim=6 (no
        # is_broken column) and job_fea_dim=7 (no released_time column); we then
        # override job_fea_dim=4 below for the makespan-only DANIEL schema.
        super().__init__(
            n_j, n_m, device,
            use_lowerbound_differencing=True,
            use_dynamic_event_features=False)
        self.use_lowerbound_differencing = True
        # Override feature dims — feature arrays are allocated to these sizes
        # in ``initial_vars`` which the parent calls during set_initial_data.
        # (mch_fea_dim stays at the parent's static value 6; not overridden.)
        self.op_fea_dim = 9
        self.job_fea_dim = 4
        # Make the weights explicit even though _combine_objective ignores
        # them now — callers may still inspect these.
        self.makespan_weight = 1.0
        self.twt_weight = 0.0

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def set_initial_data_static(self, static_data):
        """Init from a single-obj data dict.

        Reads ``static_data['static']['job_length']`` and ``['op_pt']``.
        Ignores ``job_weight``, ``job_due_date``, and any dynamic_events
        keys (single-obj is always static).
        """
        data = static_data['static']
        self.set_initial_data(data['job_length'], data['op_pt'])
        # NB: parent ``initial_vars`` already set self.job_weight=1 and
        # self.job_due_date=inf — leaving the TWT compute harmless.

        self.init_quality = self._compute_objective_from_lb()
        self._init_reward_tracking()
        self.initial_quality = np.copy(self.init_quality)

        # Empty event queue keeps _post_step_hook a no-op without touching
        # the parent's hook implementation.
        self.initial_event_queue = list(merge_and_sort_events({}))
        self.event_queues = init_event_queues(
            self.initial_event_queue, self.number_of_envs)

        return self._finalize_state_initialization()

    # ------------------------------------------------------------------
    # Reward / objective
    # ------------------------------------------------------------------
    def _combine_objective(self, lb_makespan, J=None):
        """Return makespan LB; no TWT term in single-objective mode.

        ``J`` is accepted (and ignored) so the parent's per-step callers can
        pass the precomputed active-job bound without a signature mismatch.
        """
        return lb_makespan

    def compute_objective_lower_bound(self):
        """Compute the DP makespan lower bound."""
        return super().compute_objective_lower_bound()

    def _compute_step_reward(self):
        """Use DP lower-bound reward over makespan lower bounds."""
        return super()._compute_step_reward()

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    def _normalize_op_features(self):
        n_ops = getattr(self, 'initial_number_of_ops', self.number_of_ops)
        mean_fea = np.mean(self.fea_op[:, :n_ops, :], axis=1, keepdims=True)
        std_fea = np.std(self.fea_op[:, :n_ops, :], axis=1, keepdims=True)
        np.subtract(self.fea_op, mean_fea, out=self.fea_op)
        np.divide(self.fea_op, std_fea + 1e-8, out=self.fea_op)
        np.nan_to_num(self.fea_op, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def construct_op_features(self):
        """Populate operation features with DP lower-bound observations."""
        # Match the parent observation contract: scheduled completion times
        # are known exactly, including when a non-minimum-time machine won.
        scheduled = self.op_scheduled_flag == 1
        self.op_ct_lb[scheduled] = self.op_ct[scheduled]
        self.fea_op[:, :, 0] = self.op_min_pt
        self.fea_op[:, :, 1] = self.pt_span
        self.fea_op[:, :, 2] = self.op_mean_pt
        self.fea_op[:, :, 3] = self.op_scheduled_flag
        self.fea_op[:, :, 4] = self.op_ct_lb
        self.fea_op[:, :, 5] = self.op_earliest_start_time
        self.fea_op[:, :, 6] = self.op_available_mch_nums / self.number_of_machines
        self.fea_op[:, :, 7] = self.op_start_time
        self.fea_op[:, :, 8] = self.op_progress

        self._normalize_op_features()

    def construct_job_features(self):
        """Populate makespan-relevant job features (Z-score normalized).

        Multi-objective/TWT features are always dropped:
            [4] job_weight / 5.0
            [5] job_slack_time
            [6] normalized_due_date

        Features:
            [0] remain_op_nums / max_ops_per_job
            [1] remain_sum_pt
            [2] job_ct_lb
            [3] job_idle_time
        """
        self.fea_job[:, :, 0] = self.job_remain_op_nums / self.max_number_of_ops_per_job
        self.fea_job[:, :, 1] = self.job_remain_sum_pt
        self.fea_job[:, :, 2] = self.job_ct_lb
        self.fea_job[:, :, 3] = self.job_idle_time

        np.nan_to_num(self.fea_job, nan=0.0, posinf=1.0, neginf=-1.0, copy=False)
        n_jobs = getattr(self, 'initial_number_of_jobs', self.number_of_jobs)
        mean_fea = np.mean(self.fea_job[:, :n_jobs, :], axis=1, keepdims=True)
        std_fea = np.std(self.fea_job[:, :n_jobs, :], axis=1, keepdims=True)
        np.subtract(self.fea_job, mean_fea, out=self.fea_job)
        np.divide(self.fea_job, std_fea + 1e-8, out=self.fea_job)
        np.nan_to_num(self.fea_job, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
