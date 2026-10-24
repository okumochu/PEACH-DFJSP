import numpy as np
import numpy.ma as ma
import copy
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch_geometric.data import HeteroData

from src.utils.event_utils import (
    fire_due_events,
    init_event_queues,
    merge_and_sort_events,
    reset_event_counters,
)
# Constant-column guard for Z-scoring; see FJSPEnvHGT's module for why.
# hgt_env does not import this module, so there is no cycle.
from src.HGT.hgt_env import _zscore_scale


class HGTEnvStateNoPT:
    """Environment state for the Without PT nodes ablation."""

    def __init__(self):
        self.candidate_mask_tensor = None  # Candidate mask [sz_b, J, M]
        self.candidate_tensor = None  # Candidate operations [sz_b, J]
        self.candidate_pt_indices = None  # Placeholder for shared PPO memory API
        self.graph_tensor = None  # HeteroData graph for HGT processing

    def update(self, candidate_mask, candidate, graph, device):
        """Update state tensors from numpy arrays.

        Only transfers candidate_mask, candidate to GPU.
        Op/machine features are already embedded in the graph's HeteroData nodes.
        """
        # No CPU fast path — see FJSPEnvHGT's HGTEnvState.update. torch.from_numpy
        # shares storage and .to('cpu')/.long()-on-int64 are no-ops, so the
        # stored state aliased buffers step() rewrites in place.
        self.candidate_mask_tensor = torch.tensor(
            candidate_mask, dtype=torch.float32, device=device)
        self.candidate_tensor = torch.tensor(
            candidate, dtype=torch.long, device=device)
        self.candidate_pt_indices = torch.full(
            candidate_mask.shape, -1, dtype=torch.long, device=device)
        self.graph_tensor = graph


class FJSPEnvHGTNoPT:
    """Multi-Criteria FJSP Environment for the Without PT nodes ablation.

    Single class that handles both training and inference — the split between
    a base env and a specialized training env was removed. The
    ``InferenceEnv`` iterator in ``inference_env.py`` wraps this class to
    handle dynamic events as a sequence of static solves.

    Handles multiple environments simultaneously with heterogeneous graphs.
    Supports pre-allocation for dynamic job arrivals.

    Optimizes: Makespan + Total Weighted Tardiness (TWT)

    Ablation: Without PT nodes. Operations and machines are connected by direct
    bidirectional edges wherever compatible. PT statistics are encoded in
    op features (min/span/mean) and machine features (min/mean/max).
    """

    def __init__(self, n_j, n_m, device, use_lowerbound_differencing=True,
                 use_dynamic_event_features=True):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.device = device
        self.use_lowerbound_differencing = bool(use_lowerbound_differencing)
        self.use_pt_nodes = False
        # Dynamic-event observability features (matches the PT design). When
        # True the env without PT nodes exposes the same two columns as the
        # Full model: machine ``is_broken`` and job ``released_time``.
        # The Without PT nodes variant
        # already uses machine idx 6 for ``mch_max_pt`` (a packed PT statistic,
        # since there are no PT nodes) and job idx 6 for ``normalized_due_date``,
        # so the dynamic columns go to a NEW index 7 in each (dim 7 -> 8).
        self.use_dynamic_event_features = bool(use_dynamic_event_features)

        self.op_fea_dim = 9
        # +1 (index 7): is_broken flag (machine_breakdown observability).
        self.mch_fea_dim = 8 if self.use_dynamic_event_features else 7
        # +1 (index 7): normalized release time (new-job arrivals).
        self.job_fea_dim = 8 if self.use_dynamic_event_features else 7
        self.pt_fea_dim = 0

        # Multi-criteria objective weights
        self.makespan_weight = 1.0
        self.twt_weight = 1.0

        # Graph structure caching
        self.cached_graph_structure = None
        self.graph_structure_initialized = False

        # Initialize state
        self.initial_state = HGTEnvStateNoPT()

        # Will be set when data is loaded
        self.number_of_envs = 0
        self.number_of_ops = 0
        self.job_length = None
        self.op_pt = None
        self.normalized_op_pt = None
        self.process_relation = None
        self.compatible_op = None
        self.compatible_mch = None
        self.job_first_op_id = None
        self.job_last_op_id = None
        self.candidate = None

        # Parallel processing variables
        self.done_flag = None
        self.current_makespan = None
        self.current_twt = None
        self.step_count = None  # initialized as [num_envs] array in initial_vars()

    def set_initial_data(self, job_length_list, op_pt_list):
        """Set initial data and construct static features.

        Supports pre-allocation: if alloc_* attributes are set (by subclass),
        arrays are sized to max expected values for dynamic job arrivals.
        """
        self._seeded_reset_spec = None
        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.number_of_jobs = job_length_list[0].shape[0]
        self.number_of_ops = op_pt_list[0].shape[0]
        self.number_of_machines = op_pt_list[0].shape[1]
        self.max_number_of_ops_per_job = np.max(self.job_length)

        # Determine allocation sizes (pre-allocated for dynamic mode).
        # alloc_op_mch_size = max op-machine compatible pairs = max op-machine edges.
        if hasattr(self, 'alloc_jobs') and self.alloc_jobs is not None:
            self.alloc_jobs_size = self.alloc_jobs
            self.alloc_ops_size = self.alloc_ops
            self.alloc_op_mch_size = self.alloc_pt_nodes
        else:
            self.alloc_jobs_size = self.number_of_jobs
            self.alloc_ops_size = self.number_of_ops
            self.alloc_op_mch_size = self.number_of_ops * self.number_of_machines

        # Static properties for parallel processing
        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(
            self.number_of_envs, self.number_of_jobs)

        # Job first/last operation indices
        head_op_id = np.zeros((self.number_of_envs, 1))
        self.job_first_op_id = np.concatenate(
            [head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype('int')
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        # Pad for dynamic mode
        if self.alloc_jobs_size > self.number_of_jobs:
            temp_first = np.full((self.number_of_envs, self.alloc_jobs_size), -1, dtype='int')
            temp_last = np.full((self.number_of_envs, self.alloc_jobs_size), -1, dtype='int')
            temp_first[:, :self.number_of_jobs] = self.job_first_op_id
            temp_last[:, :self.number_of_jobs] = self.job_last_op_id
            self.job_first_op_id = temp_first
            self.job_last_op_id = temp_last

            temp_job_length = np.zeros((self.number_of_envs, self.alloc_jobs_size), dtype='int')
            temp_job_length[:, :self.number_of_jobs] = self.job_length
            self.job_length = temp_job_length

        # Process times (keep original + normalized)
        if self.alloc_ops_size > self.number_of_ops:
            temp_op_pt = np.zeros((self.number_of_envs, self.alloc_ops_size,
                                   self.number_of_machines), dtype=np.float32)
            for env_idx in range(self.number_of_envs):
                temp_op_pt[env_idx, :self.number_of_ops, :] = op_pt_list[env_idx]
            self.op_pt = temp_op_pt
        else:
            self.op_pt = np.array(op_pt_list).astype(np.float32)

        # PER-ENV normaliser, matching FJSPEnvHGT (audit finding L). A
        # batch-global np.max over [E, n_ops, M] makes an instance's result
        # depend on which other instances shared its batch. The PT env was
        # fixed and this one was not, which left the two ARMS OF THE SAME
        # ABLATION with different normalisers during training — a confound in
        # the one comparison the file exists to support. Identical to the old
        # scalar at batch size 1, i.e. every reported experiment.
        _pt_max = np.max(self.op_pt, axis=(1, 2))                      # [E]
        self.max_pt = np.where(_pt_max > 0, _pt_max, 1.0).astype(np.float32)
        self.normalized_op_pt = self.op_pt / self.max_pt[:, None, None]

        # Process relation (compatibility)
        self.process_relation = (self.normalized_op_pt != 0)
        self.reverse_process_relation = ~self.process_relation
        self.compatible_mch = np.sum(self.process_relation[:, :self.number_of_ops, :], 1)
        self.compatible_op = np.sum(self.process_relation, 2)

        # Op-machine edge counts (one bidirectional edge per compatible pair)
        self.num_op_mch_edges_per_env = np.sum(
            self.process_relation[:, :self.number_of_ops, :], axis=(1, 2))
        self.max_op_mch_edges = self.alloc_op_mch_size

        alloc_ops = getattr(self, 'alloc_ops_size', self.number_of_ops)

        # Initialize tracking variables
        self.initial_vars()

        # Build process time statistics (masked for compatibility)
        self.normalized_op_pt = ma.array(self.normalized_op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.normalized_op_pt, axis=2).data.astype(np.float32)
        self.op_min_pt = np.min(self.normalized_op_pt, axis=-1).data
        self.op_max_pt = np.max(self.normalized_op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt

        # op_progress and job-level features
        for env_idx in range(self.number_of_envs):
            for job_idx in range(self.number_of_jobs):
                first_op = self.job_first_op_id[env_idx][job_idx]
                last_op = self.job_last_op_id[env_idx][job_idx]
                job_len = self.job_length[env_idx, job_idx]

                for op_idx in range(first_op, last_op + 1):
                    self.op_progress[env_idx, op_idx] = (op_idx - first_op + 1) / job_len

                self.job_remain_sum_pt[env_idx, job_idx] = np.sum(
                    self.op_mean_pt[env_idx][first_op:last_op + 1])
                self.job_remain_op_nums[env_idx, job_idx] = job_len

        # Build op_to_job reverse lookup (bug fix: avoids linear search in unschedule)
        self.op_to_job = np.full((self.number_of_envs, alloc_ops), -1, dtype=int)
        for env_idx in range(self.number_of_envs):
            for job_idx in range(self.number_of_jobs):
                first_op = self.job_first_op_id[env_idx, job_idx]
                last_op = self.job_last_op_id[env_idx, job_idx]
                self.op_to_job[env_idx, first_op:last_op + 1] = job_idx

        # Compute initial lower bound
        self.compute_objective_lower_bound()

        self.construct_op_features()

        # An unused machine has no PT statistic; never expose masked fill
        # sentinels to the float32 feature standardization.
        self.mch_min_pt = np.min(self.normalized_op_pt, axis=1).filled(0)
        self.mch_mean_pt = np.mean(self.normalized_op_pt, axis=1).filled(0).astype(np.float32)
        self.mch_max_pt = np.max(self.normalized_op_pt, axis=1).filled(0)
        self.mch_total_compatible_op_nums = np.copy(self.compatible_mch)
        self.candidate_pt = np.array(
            [self.normalized_op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])
        if self.alloc_jobs_size > self.number_of_jobs:
            self.candidate_pt[:, self.number_of_jobs:] = 0
        self.candidate_mask = (self.candidate_pt == 0)
        self.mch_candidate_compatible_op_nums = np.sum(~self.candidate_mask, axis=1)
        self.construct_mch_features()

        # Initialize graph structure (edges)
        self._initialize_graph_structure()

        # Save initial state for reset (graph + features built by _finalize_state_initialization)
        self.initial_op_ct_lb = np.copy(self.op_ct_lb)
        self.initial_candidate_pt = np.copy(self.candidate_pt)
        self.initial_candidate_mask = np.copy(self.candidate_mask)
        self.initial_mch_total_compatible_op_nums = np.copy(self.mch_total_compatible_op_nums)
        self.initial_mch_candidate_compatible_op_nums = np.copy(
            self.mch_candidate_compatible_op_nums)
        self.initial_job_remain_sum_pt = np.copy(self.job_remain_sum_pt)
        self.initial_job_remain_op_nums = np.copy(self.job_remain_op_nums)
        self.initial_job_ct_lb = np.copy(self.job_ct_lb)

    # ------------------------------------------------------------------
    # High-level entrypoints (training + inference)
    # ------------------------------------------------------------------

    def _realised_objective(self):
        """Objective already realised by the CURRENT schedule, in reward units.
        Mirrors FJSPEnvHGT._realised_objective; see there for why."""
        return ((self.makespan_weight * np.maximum(self.current_makespan, 0.0)
                 + self.twt_weight * self.current_twt)
                / self.max_pt).astype(np.float32)

    def _init_reward_tracking(self):
        """Anchor the reward tracker to the CURRENT state, in either mode.

        See FJSPEnvHGT._init_reward_tracking: hard-zeroing the
        partial-objective reward anchor charges the first step after a seeded
        prefix the entire prefix objective. Kept identical here so the
        Without PT nodes ablation is not compared against a differently-rewarded arm.
        """
        if self.use_lowerbound_differencing:
            self.max_endTime = np.copy(self.init_quality)
        else:
            self.max_endTime = self._realised_objective()

    def set_initial_data_static(self, static_data: Dict[str, Any]):
        """Set initial data from a static-shape dict.

        ``static_data['dynamic_events']`` optional — when non-empty, events
        fire mid-rollout via ``_post_step_hook``.
        """
        # Guard for the DEPRECATED inline-event path only.
        #
        # This class has no inline job-arrival machinery: it never sets
        # ``alloc_jobs``/``dynamic_jobs``/``real_total_ops`` the way the PT env
        # does, so ``event_utils.apply_event_to_env`` no-ops the job_arrival
        # branch (it is gated on ``env.dynamic_jobs is not None``) while STILL
        # popping the event and incrementing fired_event_counts — the arrivals
        # would vanish invisibly and the ablation would be scored on a strictly
        # easier problem than the PT arm (audit finding D).
        #
        # The ablation itself is NOT blocked by this. Both live paths realize
        # arrivals by REBUILDING the env from a static dict grown to include the
        # arrived job (``HGTTrainer``'s anchored-segment loop and
        # ``InferenceEnv.run``, both via ``_wrap_as_static_data``, which carries
        # no ``dynamic_jobs`` key), so this branch never fires there. Verified
        # end to end by ``bench/check_objective_invariants.py --no_pt``: 20+8
        # jobs, full operation coverage, reported objectives matching a
        # recomputation from the schedule.
        #
        # Reject direct callers that pass raw ``sample_env`` output and expect
        # this environment to apply job arrivals without segment rebuilding.
        arriving = (static_data.get('dynamic_jobs') or {}).get('job_length')
        if arriving is not None and len(np.asarray(arriving)) > 0:
            raise NotImplementedError(
                f'FJSPEnvHGTNoPT has no INLINE job-arrival machinery '
                f'({len(np.asarray(arriving))} arrivals supplied), and firing '
                f'them through apply_event_to_env would drop them silently. '
                f'Use the segmented path (InferenceEnv.run, or the trainer\'s '
                f'anchored-segment rollout), which realizes arrivals by growing '
                f'the static problem and works with this env today.')

        data = static_data['static']
        self.set_initial_data(data['job_length'], data['op_pt'])
        self._set_twt_data(data['job_weight'], data['job_due_date'])

        self.init_quality = self._compute_objective_from_lb()
        self._init_reward_tracking()
        self.initial_quality = np.copy(self.init_quality)
        self._save_twt_initial_data()

        events = static_data.get('dynamic_events', {}) or {}
        self.initial_event_queue = list(merge_and_sort_events(events))
        self.event_queues = init_event_queues(
            self.initial_event_queue, self.number_of_envs)
        reset_event_counters(self)

        return self._finalize_state_initialization()

    def set_initial_data_with_fixed_schedule(
        self,
        static_data: Dict[str, Any],
        fixed_ops: Optional[List[List[Dict[str, Any]]]] = None,
        mch_unavailable_until: Optional[np.ndarray] = None,
        job_release_times: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ):
        """Set initial data with a subset of operations already scheduled.

        Used by InferenceEnv to start a static inference segment from a
        partially-scheduled state (ops frozen by prior events).
        A subsequent reset restarts this same segment, including its prefix
        and availability/release floors.
        """
        state = self.set_initial_data_static(static_data)
        if (fixed_ops is None or all(len(fo) == 0 for fo in (fixed_ops or [[]]))) \
                and mch_unavailable_until is None and job_release_times is None \
                and current_time == 0.0:
            return state
        self._seed_fixed_schedule(
            fixed_ops_per_env=fixed_ops,
            mch_unavailable_until=mch_unavailable_until,
            job_release_times=job_release_times,
            current_time=current_time,
        )
        self.init_quality = self._compute_objective_from_lb()
        self._init_reward_tracking()
        self.initial_quality = np.copy(self.init_quality)
        self._save_twt_initial_data()
        # Rebuild the complete seeded state on reset; a clean-array reset
        # would discard the frozen prefix but retain its reward anchor.
        self._seeded_reset_spec = copy.deepcopy(dict(
            static_data=static_data, fixed_ops=fixed_ops,
            mch_unavailable_until=mch_unavailable_until,
            job_release_times=job_release_times, current_time=current_time))
        return self.state

    def initial_vars(self):
        """Initialize tracking variables for parallel processing.

        Uses alloc_*_size for array dimensions to support dynamic mode.
        """
        self.step_count = np.zeros(self.number_of_envs, dtype=np.int64)
        self.done_flag = np.full(shape=(self.number_of_envs,), fill_value=0, dtype=bool)
        self.current_makespan = np.full(self.number_of_envs, float("-inf"), dtype=np.float32)
        self.current_twt = np.zeros(self.number_of_envs, dtype=np.float32)

        alloc_ops = getattr(self, 'alloc_ops_size', self.number_of_ops)
        alloc_jobs = getattr(self, 'alloc_jobs_size', self.number_of_jobs)

        self.mch_queue = np.full(
            shape=[self.number_of_envs, self.number_of_machines, alloc_ops + 1],
            fill_value=-99, dtype=int)
        self.mch_queue_len = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.op_ct = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)
        self.op_ct_lb = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)

        # Machine features
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines), dtype=np.float32)
        self.mch_earliest_start_time = np.zeros((self.number_of_envs, self.number_of_machines), dtype=np.float32)
        self.mch_idle_time = np.zeros((self.number_of_envs, self.number_of_machines), dtype=np.float32)

        self.next_schedule_time = np.zeros(self.number_of_envs, dtype=np.float32)
        self.candidate_free_time = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)

        self.unnorm_op_ct = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)
        self.unnorm_candidate_free_time = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)
        self.unnorm_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines), dtype=np.float32)

        # Candidate (first op per job)
        self.candidate = np.full((self.number_of_envs, alloc_jobs), -1, dtype='int')
        self.candidate[:, :self.number_of_jobs] = self.job_first_op_id[:, :self.number_of_jobs]

        self.job_finished_mask = np.full(
            shape=(self.number_of_envs, alloc_jobs), fill_value=0, dtype=bool)

        # Operation features
        self.op_scheduled_flag = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)
        self.op_start_time = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)
        self.op_earliest_start_time = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)

        self.op_available_mch_nums = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)
        self.op_available_mch_nums[:, :self.compatible_op.shape[1]] = self.compatible_op
        self.op_progress = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)

        # Job features
        self.job_remain_sum_pt = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)
        self.job_remain_op_nums = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)
        self.job_ct_lb = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)

        # TWT-related job features
        self.job_weight = np.ones((self.number_of_envs, alloc_jobs), dtype=np.float32)
        self.job_due_date = np.full((self.number_of_envs, alloc_jobs), np.inf, dtype=np.float32)
        self.job_release_time = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)
        self.normalized_job_due_date = np.full((self.number_of_envs, alloc_jobs), np.inf, dtype=np.float32)
        self.normalized_job_release_time = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)

        self.job_idle_time = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)

        # Operation-machine assignments (for rescheduling)
        self.op_assigned_machine = np.full((self.number_of_envs, alloc_ops), -1, dtype=int)
        self.unnorm_op_start_time = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)

        # Dynamic event tracking (populated by subclass)
        self.breakdown_periods = []
        self.interrupted_ops = []
        # Per-machine unnormalized breakdown-end time (0 = never broken). Drives
        # the is_broken machine feature; reset here every episode. Populated by
        # the shared event_utils.apply_event_to_env (machine_breakdown branch,
        # gated on hasattr) and by _seed_fixed_schedule for seeded segments.
        self.mch_broken_until = np.zeros(
            (self.number_of_envs, self.number_of_machines), dtype=np.float32)

        # Pre-allocate feature arrays (avoid np.stack allocation every step)
        self.fea_op = np.zeros((self.number_of_envs, alloc_ops, self.op_fea_dim),
                               dtype=np.float32)
        self.fea_job = np.zeros((self.number_of_envs, alloc_jobs, self.job_fea_dim),
                                dtype=np.float32)
        self.fea_m = np.zeros((self.number_of_envs, self.number_of_machines, self.mch_fea_dim),
                              dtype=np.float32)

    def construct_op_features(self):
        """Construct operation features (9 dims) with Z-score normalization.

        Pins a scheduled operation's completion lower bound to its known
        completion time — audit O4, see
        ``FJSPEnvHGT.construct_op_features`` for why it belongs here and not
        inside the DP.
        """
        _sched = self.op_scheduled_flag == 1
        self.op_ct_lb[_sched] = self.op_ct[_sched]

        self.fea_op[:, :, 0] = self.op_min_pt
        self.fea_op[:, :, 1] = self.pt_span
        self.fea_op[:, :, 2] = self.op_mean_pt
        self.fea_op[:, :, 3] = self.op_scheduled_flag
        self.fea_op[:, :, 4] = self.op_ct_lb
        self.fea_op[:, :, 5] = self.op_earliest_start_time
        self.fea_op[:, :, 6] = self.op_available_mch_nums / self.number_of_machines
        self.fea_op[:, :, 7] = self.op_start_time
        self.fea_op[:, :, 8] = self.op_progress

        # Z-score with frozen reference count (stable when jobs arrive)
        n_ops = getattr(self, 'initial_number_of_ops', self.number_of_ops)
        mean_fea = np.mean(self.fea_op[:, :n_ops, :], axis=1, keepdims=True)
        std_fea = np.std(self.fea_op[:, :n_ops, :], axis=1, keepdims=True)
        np.subtract(self.fea_op, mean_fea, out=self.fea_op)
        np.divide(self.fea_op, _zscore_scale(std_fea), out=self.fea_op)
        np.nan_to_num(self.fea_op, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def construct_job_features(self):
        """Construct job features (7 or 8 dims) with Z-score normalization.

        The 8th column (index 7, ``released_time``) is only present in the
        dynamic-event design (``use_dynamic_event_features=True``).
        """
        job_slack_time = np.where(
            self.normalized_job_due_date < np.inf,
            self.normalized_job_due_date - self.job_ct_lb,
            1.0)

        finite_due = np.isfinite(self.normalized_job_due_date)
        max_due_date = np.max(
            np.where(finite_due, self.normalized_job_due_date, -np.inf),
            axis=1, keepdims=True)
        # Keep due-date scaling local to each instance, as in the PT env.
        max_due_date = np.where(
            np.isfinite(max_due_date) & (max_due_date > 0), max_due_date, 1.0)
        normalized_due_date = np.where(
            finite_due,
            self.normalized_job_due_date / max_due_date, 1.0)

        self.fea_job[:, :, 0] = self.job_remain_op_nums / self.max_number_of_ops_per_job
        self.fea_job[:, :, 1] = self.job_remain_sum_pt
        self.fea_job[:, :, 2] = self.job_ct_lb
        self.fea_job[:, :, 3] = self.job_idle_time
        self.fea_job[:, :, 4] = self.job_weight / 5.0
        self.fea_job[:, :, 5] = job_slack_time
        self.fea_job[:, :, 6] = normalized_due_date
        # Release time (index 7, dynamic-event design only): 0 for initially
        # present jobs; >0 for jobs that arrived (seeded via job_release_times).
        # Inert (Z-scored to 0) when arrivals are disabled.
        if self.use_dynamic_event_features:
            self.fea_job[:, :, 7] = self.normalized_job_release_time

        np.nan_to_num(self.fea_job, nan=0.0, posinf=1.0, neginf=-1.0, copy=False)
        n_jobs = getattr(self, 'initial_number_of_jobs', self.number_of_jobs)
        mean_fea = np.mean(self.fea_job[:, :n_jobs, :], axis=1, keepdims=True)
        std_fea = np.std(self.fea_job[:, :n_jobs, :], axis=1, keepdims=True)
        np.subtract(self.fea_job, mean_fea, out=self.fea_job)
        np.divide(self.fea_job, _zscore_scale(std_fea), out=self.fea_job)
        np.nan_to_num(self.fea_job, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def construct_mch_features(self):
        """Construct machine features (7 or 8 dims) with Z-score normalization.

        The 8th column (index 7, ``is_broken``) is only present in the
        dynamic-event design (``use_dynamic_event_features=True``). Index 6 stays
        ``mch_max_pt`` (a packed PT statistic), so is_broken is a NEW column.
        """
        n_ref_jobs = getattr(self, 'initial_number_of_jobs', self.number_of_jobs)
        n_ref_ops = getattr(self, 'initial_number_of_ops', self.number_of_ops)
        self.fea_m[:, :, 0] = self.mch_candidate_compatible_op_nums / n_ref_jobs
        self.fea_m[:, :, 1] = self.mch_total_compatible_op_nums / n_ref_ops
        self.fea_m[:, :, 2] = self.mch_min_pt
        self.fea_m[:, :, 3] = self.mch_mean_pt
        self.fea_m[:, :, 4] = self.mch_earliest_start_time
        self.fea_m[:, :, 5] = self.mch_idle_time
        self.fea_m[:, :, 6] = self.mch_max_pt
        # is_broken (index 7, dynamic-event design only): 1 while a fired
        # machine_breakdown's repair window is still ahead of the schedule
        # frontier (current_makespan), else 0. Mirrors the PT env (there at
        # idx 6, here idx 7 because idx 6 carries mch_max_pt).
        if self.use_dynamic_event_features:
            sim_t = np.maximum(self.current_makespan, 0.0)[:, None]
            self.fea_m[:, :, 7] = ((self.mch_broken_until > sim_t)
                                   & (self.mch_broken_until > 0.0)).astype(np.float32)

        mean_fea = np.mean(self.fea_m, axis=1, keepdims=True)
        std_fea = np.std(self.fea_m, axis=1, keepdims=True)
        np.subtract(self.fea_m, mean_fea, out=self.fea_m)
        np.divide(self.fea_m, _zscore_scale(std_fea), out=self.fea_m)
        np.nan_to_num(self.fea_m, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _get_valid_edges(self, cached):
        """Return edge tensors sliced to valid portion (handles pre-allocated mode)."""
        if 'num_precedence_edges' in cached:
            # Pre-allocated dynamic mode: slice to valid edges
            n_prec = cached['num_precedence_edges']
            n_om = cached['num_op_mch_edges']
            n_job = cached['num_op_job_edges']
            return (
                cached['precedence_edges'][:, :n_prec],
                cached['op_to_job_edges'][:, :n_job],
                cached['op_to_machine_edges'][:, :n_om],
                cached['machine_to_op_edges'][:, :n_om],
            )
        else:
            # Static mode: exact-fit, use full tensors
            return (
                cached['precedence_edges'],
                cached['op_to_job_edges'],
                cached['op_to_machine_edges'],
                cached['machine_to_op_edges'],
            )

    def _build_batch_hetero_graphs(self):
        """Build per-env HeteroData graphs from the current node features."""
        fea_op_gpu = torch.from_numpy(self.fea_op).to(self.device, non_blocking=True, copy=True)
        fea_job_gpu = torch.from_numpy(self.fea_job).to(self.device, non_blocking=True, copy=True)
        fea_m_gpu = torch.from_numpy(self.fea_m).to(self.device, non_blocking=True, copy=True)

        batch_graphs = []
        for env_idx in range(self.number_of_envs):
            cached = self.cached_graph_structure[env_idx]
            num_actual_ops = cached['num_actual_ops']

            (prec_e, job_e, op_mch_e, mch_op_e) = self._get_valid_edges(cached)

            data = HeteroData()
            data['op'].x = fea_op_gpu[env_idx, :num_actual_ops]
            data['job'].x = fea_job_gpu[env_idx, :self.number_of_jobs]
            data['machine'].x = fea_m_gpu[env_idx]

            data['op', 'precedes', 'op'].edge_index = prec_e
            data['op', 'belongs_to', 'job'].edge_index = job_e
            data['op', 'processed_by', 'machine'].edge_index = op_mch_e
            data['machine', 'processes', 'op'].edge_index = mch_op_e

            batch_graphs.append(data)

        return batch_graphs

    def _get_edges(self, env_idx):
        """Get all edge types for specific environment in a single pass.

        Ablation: direct op<->machine edges for every compatible pair
        (wherever process_relation is True).
        """
        precedence_src = []
        precedence_dst = []
        job_ids = []
        op_ids = []
        op_mch_op_list = []
        op_mch_m_list = []

        for job_idx in range(self.number_of_jobs):
            first_op = self.job_first_op_id[env_idx, job_idx]
            last_op = self.job_last_op_id[env_idx, job_idx]

            for op_idx in range(first_op, last_op + 1):
                job_ids.append(job_idx)
                op_ids.append(op_idx)

                if op_idx < last_op:
                    precedence_src.append(op_idx)
                    precedence_dst.append(op_idx + 1)

                for m_idx in range(self.number_of_machines):
                    if self.process_relation[env_idx, op_idx, m_idx]:
                        op_mch_op_list.append(op_idx)
                        op_mch_m_list.append(m_idx)

        precedence_edges = [precedence_src, precedence_dst]
        job_op_edges = [job_ids, op_ids]
        op_to_machine = [op_mch_op_list, op_mch_m_list]
        machine_to_op = [op_mch_m_list, op_mch_op_list]

        return precedence_edges, op_to_machine, machine_to_op, job_op_edges

    def _make_edge_tensor(self, data, max_size=None):
        """Create an edge tensor, optionally pre-allocated to max_size."""
        n = len(data[0])
        if max_size is not None:
            t = torch.full((2, max_size), -1, dtype=torch.long, device=self.device)
            if n > 0:
                t[:, :n] = torch.tensor(data, dtype=torch.long, device=self.device)
            return t
        if n > 0:
            return torch.tensor(data, dtype=torch.long, device=self.device)
        return torch.zeros((2, 0), dtype=torch.long, device=self.device)

    def _initialize_graph_structure(self):
        """Cache all static edge indices on device.

        In dynamic mode (alloc_ops_size > number_of_ops), pre-allocates edge
        tensors to max capacity and tracks valid edge counts. This avoids
        torch.cat when new jobs arrive.
        """
        self.cached_graph_structure = []
        is_dynamic = getattr(self, 'alloc_ops_size', self.number_of_ops) > self.number_of_ops

        for env_idx in range(self.number_of_envs):
            (precedence_edges, op_to_machine,
             machine_to_op, job_op_edges) = self._get_edges(env_idx)

            n_prec = len(precedence_edges[0])
            n_om = len(op_to_machine[0])
            n_job = len(job_op_edges[0])
            op_job_data = [job_op_edges[1], job_op_edges[0]]  # reversed

            if is_dynamic:
                max_prec = self.alloc_ops_size
                max_om = self.max_op_mch_edges
                max_job_op = self.alloc_ops_size

                cached = {
                    'precedence_edges': self._make_edge_tensor(precedence_edges, max_prec),
                    'op_to_machine_edges': self._make_edge_tensor(op_to_machine, max_om),
                    'machine_to_op_edges': self._make_edge_tensor(machine_to_op, max_om),
                    'op_to_job_edges': self._make_edge_tensor(op_job_data, max_job_op),
                    'num_actual_ops': self.number_of_ops,
                    'num_precedence_edges': n_prec,
                    'num_op_mch_edges': n_om,
                    'num_op_job_edges': n_job,
                }
            else:
                cached = {
                    'precedence_edges': self._make_edge_tensor(precedence_edges),
                    'op_to_machine_edges': self._make_edge_tensor(op_to_machine),
                    'machine_to_op_edges': self._make_edge_tensor(machine_to_op),
                    'op_to_job_edges': self._make_edge_tensor(op_job_data),
                    'num_actual_ops': self.number_of_ops,
                }

            self.cached_graph_structure.append(cached)

        self.graph_structure_initialized = True

    def compute_objective_lower_bound(self) -> np.ndarray:
        """Compute multi-criteria objective lower bound using DP (vectorized).

        Processes all [E, J] pairs simultaneously with NumPy broadcasting.
        Only the sequential DP over operation offsets (max ~10) is looped.

        Side Effects:
            Updates self.op_ct_lb and self.job_ct_lb

        Returns:
            lb_objective: [E] array - weighted sum of makespan LB and TWT LB
        """
        E = self.number_of_envs
        J = self.number_of_jobs

        candidate = self.candidate[:, :J]          # [E, J]
        first_op = self.job_first_op_id[:, :J]     # [E, J]
        last_op = self.job_last_op_id[:, :J]        # [E, J]
        finished = self.job_finished_mask[:, :J]     # [E, J] bool
        mch_avail = self.mch_free_time               # [E, M]

        op_pt_data = self.normalized_op_pt.data      # [E, N_ops, M] underlying ndarray
        proc_rel = self.process_relation              # [E, N_ops, M] bool

        env_idx = np.arange(E)[:, None]  # [E, 1] for broadcasting

        # Initialize current_ect for all [E, J] pairs
        # Default: op_ct of predecessor (candidate - 1)
        prev_op = np.maximum(candidate - 1, 0)  # clamp for gather safety
        current_ect = self.op_ct[env_idx, prev_op]  # [E, J]

        # Override: where candidate == first_op, use job release time
        at_first = (candidate == first_op)
        current_ect = np.where(at_first,
                               self.normalized_job_release_time[:, :J],
                               current_ect)

        # Override: finished jobs get their actual completion time
        finished_ect = self.op_ct[env_idx, last_op]
        current_ect = np.where(finished, finished_ect, current_ect)

        # DP loop over operation offsets (max ~10 iterations)
        remaining_ops = np.where(finished, 0, last_op - candidate + 1)  # [E, J]
        max_remaining = int(np.max(remaining_ops)) if E > 0 else 0
        active = ~finished  # [E, J]

        for k in range(max_remaining):
            step_active = active & (k < remaining_ops)  # [E, J]
            if not step_active.any():
                break

            abs_op = candidate + k                                    # [E, J]
            abs_op_safe = np.clip(abs_op, 0, self.op_ct_lb.shape[1] - 1)

            # Gather [E, J, M] slices
            op_pts = op_pt_data[env_idx, abs_op_safe, :]              # [E, J, M]
            compat = proc_rel[env_idx, abs_op_safe, :]                # [E, J, M]

            # DP step: start_times = max(current_ect, machine_availability)
            start_times = np.maximum(current_ect[:, :, None],
                                     mch_avail[:, None, :])           # [E, J, M]
            ects = np.where(compat, start_times + op_pts, np.inf)     # [E, J, M]
            min_ect = np.min(ects, axis=2)                            # [E, J]

            current_ect = np.where(step_active, min_ect, current_ect)

            # Scatter to op_ct_lb, ACTIVE ENTRIES ONLY — see FJSPEnvHGT for why:
            # an unarrived job slot has candidate == -1, clips to index 0, and
            # (last-write-wins) reverts job 0's freshly computed lower bound.
            rows, cols = np.nonzero(step_active)
            if rows.size:
                self.op_ct_lb[rows, abs_op_safe[rows, cols]] = current_ect[rows, cols]

        # Write job-level results
        self.job_ct_lb[:, :J] = current_ect

        lb_makespan = np.max(current_ect, axis=1)
        return self._combine_objective(lb_makespan)

    def _combine_objective(self, lb_makespan: np.ndarray) -> np.ndarray:
        """Compute weighted objective from makespan LB and TWT LB."""
        J = self.number_of_jobs
        tardiness = np.maximum(0, self.job_ct_lb[:, :J] - self.normalized_job_due_date[:, :J])
        weighted_tardiness = tardiness * self.job_weight[:, :J]
        lb_twt = np.sum(weighted_tardiness, axis=1)
        return self.makespan_weight * lb_makespan + self.twt_weight * lb_twt

    def _compute_objective_from_lb(self) -> np.ndarray:
        """Compute combined objective from already-computed job_ct_lb (no DP)."""
        J = self.number_of_jobs
        lb_makespan = np.max(self.job_ct_lb[:, :J], axis=1)
        return self._combine_objective(lb_makespan)

    def _set_twt_data(self, job_weight_list, job_due_date_list):
        """Set TWT-specific data (weights, due dates, release times)."""
        for env_idx in range(self.number_of_envs):
            self.job_weight[env_idx, :self.number_of_jobs] = job_weight_list[env_idx]
            self.job_due_date[env_idx, :self.number_of_jobs] = job_due_date_list[env_idx]
            self.normalized_job_due_date[env_idx, :self.number_of_jobs] = (
                job_due_date_list[env_idx] / self.max_pt[env_idx])
            self.job_release_time[env_idx, :self.number_of_jobs] = 0
            self.normalized_job_release_time[env_idx, :self.number_of_jobs] = 0

    def _save_twt_initial_data(self):
        """Save TWT data for reset."""
        self.initial_job_weight = np.copy(self.job_weight)
        self.initial_job_due_date = np.copy(self.job_due_date)
        self.initial_job_release_time = np.copy(self.job_release_time)
        self.initial_normalized_job_due_date = np.copy(self.normalized_job_due_date)
        self.initial_normalized_job_release_time = np.copy(self.normalized_job_release_time)

    def _restore_twt_initial_data(self):
        """Restore TWT data on reset."""
        self.job_weight = np.copy(self.initial_job_weight)
        self.job_due_date = np.copy(self.initial_job_due_date)
        self.job_release_time = np.copy(self.initial_job_release_time)
        self.normalized_job_due_date = np.copy(self.initial_normalized_job_due_date)
        self.normalized_job_release_time = np.copy(self.initial_normalized_job_release_time)

    def _finalize_state_initialization(self):
        """Rebuild features/graphs and finalize initial state after TWT setup."""
        self.construct_job_features()
        self.initial_state.update(self.candidate_mask, self.candidate,
                                  self._build_batch_hetero_graphs(), self.device)
        # Shallow copy: HGTEnvState only holds tensor/graph references that
        # state.update() reassigns on every step, so deepcopy is unnecessary.
        self.state = copy.copy(self.initial_state)
        return self.state

    def reset(self):
        """Restart the most recently initialized static problem or segment."""
        if getattr(self, '_seeded_reset_spec', None) is not None:
            return self.set_initial_data_with_fixed_schedule(**self._seeded_reset_spec)
        self.initial_vars()
        self.op_ct_lb = np.copy(self.initial_op_ct_lb)
        self.candidate_pt = np.copy(self.initial_candidate_pt)
        self.candidate_mask = np.copy(self.initial_candidate_mask)
        self.mch_total_compatible_op_nums = np.copy(self.initial_mch_total_compatible_op_nums)
        self.mch_candidate_compatible_op_nums = np.copy(
            self.initial_mch_candidate_compatible_op_nums)

        self.job_remain_sum_pt = np.copy(self.initial_job_remain_sum_pt)
        self.job_remain_op_nums = np.copy(self.initial_job_remain_op_nums)
        self.job_ct_lb = np.copy(self.initial_job_ct_lb)

        # Subclass hook for additional reset logic
        self._reset_hook()

        # Shallow copy: HGTEnvState only holds tensor/graph references that
        # state.update() reassigns on every step, so deepcopy is unnecessary.
        self.state = copy.copy(self.initial_state)
        return self.state

    def _reset_hook(self):
        """Restore reward tracking, TWT data, and event queues on reset."""
        if hasattr(self, 'initial_quality'):
            self.init_quality = np.copy(self.initial_quality)
            self._init_reward_tracking()
        if hasattr(self, 'initial_job_weight'):
            self._restore_twt_initial_data()
        if hasattr(self, 'initial_event_queue'):
            self.event_queues = init_event_queues(
                self.initial_event_queue, self.number_of_envs)
            reset_event_counters(self)

    def _compute_step_reward(self) -> np.ndarray:
        """Compute step reward.

        DP lower-bound reward (default):
            reward = previous_lb - current_lb
        partial-objective reward:
            reward = previous_obj - current_obj, normalized by max_pt.
        """
        current_lb = self.compute_objective_lower_bound()
        if not hasattr(self, 'max_endTime'):
            return np.zeros(self.number_of_envs, dtype=np.float32)
        if self.use_lowerbound_differencing:
            reward = self.max_endTime - current_lb
            self.max_endTime = current_lb
        else:
            current_obj = (self.makespan_weight * np.maximum(self.current_makespan, 0.0)
                           + self.twt_weight * self.current_twt) / self.max_pt
            reward = self.max_endTime - current_obj
            self.max_endTime = current_obj
        return reward

    def _post_step_hook(self):
        """Fire due events, then re-anchor the reward tracker for fired envs.

        LB is recomputed unconditionally because ``op_ct_lb`` / ``job_ct_lb``
        feed features. Reset of ``max_endTime`` only happens in
        DP lower-bound reward mode; in partial-objective reward mode,
        ``current_obj`` is invariant under events so ``max_endTime`` is left
        alone.
        """
        events_fired = fire_due_events(self)
        if events_fired is None or not events_fired.any():
            return
        new_lb = self.compute_objective_lower_bound()
        if self.use_lowerbound_differencing:
            self.max_endTime = np.where(events_fired, new_lb, self.max_endTime)
        self.construct_op_features()
        self.construct_job_features()
        # Rebuild machine features so a fired breakdown refreshes the is_broken
        # column (and bumped earliest start) with no lag, mirroring the PT env.
        self.construct_mch_features()

    def step(self, actions):
        """Execute one scheduling step across all environments.

        Uses vectorized operations (no Python loop over environments).

        Only *active* (not-yet-done) envs are advanced; see the PT env
        (``hgt_env.py``) step() for the rationale. ``active`` is all-True in the
        rectangular rollout, so this is a no-op there; it only matters for the
        ragged anchored-segment rollout.
        """
        actions = np.asarray(actions)
        active = ~self.done()
        eidx = self.env_idxs[active]
        actions = actions[active]

        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[eidx, chosen_job]

        self.step_count[eidx] += 1

        # Check if the job is unfinished
        job_unfinished_flag = (chosen_op != self.job_last_op_id[eidx, chosen_job])
        self.candidate[eidx, chosen_job] += job_unfinished_flag
        self.job_finished_mask[eidx, chosen_job] = (1 - job_unfinished_flag)

        # Update machine queue
        self.mch_queue[
            eidx, chosen_mch,
            self.mch_queue_len[eidx, chosen_mch]] = chosen_op
        self.mch_queue_len[eidx, chosen_mch] += 1

        # Calculate timing
        chosen_op_st = np.maximum(
            self.candidate_free_time[eidx, chosen_job],
            self.mch_free_time[eidx, chosen_mch])
        # Capture idle gap before mch_free_time is overwritten below.
        mch_idle_gap = chosen_op_st - self.mch_free_time[eidx, chosen_mch]
        self.op_ct[eidx, chosen_op] = chosen_op_st + self.normalized_op_pt[
            eidx, chosen_op, chosen_mch]

        self.candidate_free_time[eidx, chosen_job] = self.op_ct[
            eidx, chosen_op]
        self.mch_free_time[eidx, chosen_mch] = self.op_ct[
            eidx, chosen_op]

        # Unnormalized timing (for Gantt chart / wandb)
        unnorm_chosen_op_st = np.maximum(
            self.unnorm_candidate_free_time[eidx, chosen_job],
            self.unnorm_mch_free_time[eidx, chosen_mch])
        self.unnorm_op_ct[eidx, chosen_op] = unnorm_chosen_op_st + self.op_pt[
            eidx, chosen_op, chosen_mch]
        self.unnorm_candidate_free_time[eidx, chosen_job] = self.unnorm_op_ct[
            eidx, chosen_op]
        self.unnorm_mch_free_time[eidx, chosen_mch] = self.unnorm_op_ct[
            eidx, chosen_op]

        # Track assignments for rescheduling
        self.unnorm_op_start_time[eidx, chosen_op] = unnorm_chosen_op_st
        self.op_assigned_machine[eidx, chosen_op] = chosen_mch

        # Update makespan (active envs only; done envs keep their final value)
        self.current_makespan[eidx] = np.maximum(
            self.current_makespan[eidx], self.unnorm_op_ct[eidx, chosen_op])

        # --- Vectorized per-environment updates (no Python loop) ---
        unfinished = job_unfinished_flag
        finished = ~unfinished
        next_ops = chosen_op + 1

        # 1. candidate_pt and candidate_mask
        if unfinished.any():
            uf_envs = eidx[unfinished]
            uf_jobs = chosen_job[unfinished]
            uf_next = next_ops[unfinished]
            self.candidate_pt[uf_envs, uf_jobs] = self.normalized_op_pt[uf_envs, uf_next]
            self.candidate_mask[uf_envs, uf_jobs] = self.reverse_process_relation[
                uf_envs, uf_next]

        if finished.any():
            fi_envs = eidx[finished]
            fi_jobs = chosen_job[finished]
            fi_ops = chosen_op[finished]
            self.candidate_mask[fi_envs, fi_jobs] = 1
            # TWT update for finished jobs
            job_ct = self.unnorm_op_ct[fi_envs, fi_ops]
            due_date = self.job_due_date[fi_envs, fi_jobs]
            weight = self.job_weight[fi_envs, fi_jobs]
            tardiness = np.maximum(0.0, job_ct - due_date)
            self.current_twt[fi_envs] += tardiness * weight

        # 2. op_earliest_start_time for successor
        if unfinished.any():
            self.op_earliest_start_time[uf_envs, uf_next] = self.op_ct[
                uf_envs, chosen_op[unfinished]]

        # Compute reward (overridden by training env)
        reward = self._compute_step_reward()

        # Update operation features
        self.op_scheduled_flag[eidx, chosen_op] = 1
        self.op_start_time[eidx, chosen_op] = chosen_op_st
        self.construct_op_features()

        # Update job features
        self.job_remain_op_nums[eidx, chosen_job] -= 1
        self.job_remain_sum_pt[eidx, chosen_job] -= self.op_mean_pt[
            eidx, chosen_op]

        job_ready_time = np.where(
            chosen_op == self.job_first_op_id[eidx, chosen_job],
            self.normalized_job_release_time[eidx, chosen_job],
            self.op_ct[eidx, np.maximum(chosen_op - 1, 0)])
        job_idle_gap = np.maximum(chosen_op_st - job_ready_time, 0)
        self.job_idle_time[eidx, chosen_job] += job_idle_gap
        self.construct_job_features()

        # Update machine features
        self.mch_candidate_compatible_op_nums = np.sum(~self.candidate_mask, axis=1)
        self.mch_total_compatible_op_nums[eidx] -= self.process_relation[eidx, chosen_op]
        self.mch_earliest_start_time[eidx, chosen_mch] = self.op_ct[
            eidx, chosen_op]

        self.mch_idle_time[eidx, chosen_mch] += mch_idle_gap
        self.construct_mch_features()

        self.done_flag = self.done()

        # Post-step hook (dynamic events)
        self._post_step_hook()

        self.state.update(self.candidate_mask, self.candidate,
                          self._build_batch_hetero_graphs(), self.device)
        self.done_flag = self.done()

        return self.state, np.array(reward), self.done_flag

    def done(self):
        """Check if all environments are done (per-env).

        Mirrors the PT env: use ``real_total_ops`` (initial + any inline
        arriving ops) when present, falling back to ``number_of_ops``. The
        anchored-segment driver seeds each segment with arrivals already
        materialized as regular jobs (no inline-arrival pre-allocation), so the
        fallback is exact per segment.
        """
        real_total = getattr(self, 'real_total_ops', self.number_of_ops)
        return (self.step_count >= real_total).astype(bool)

    def _recompute_objectives(self):
        """Recompute current_makespan and current_twt from scratch.

        Slices to the ALLOCATED width for the same reason as the PT env — see
        ``FJSPEnvHGT._recompute_objectives``.
        """
        n = self.op_scheduled_flag.shape[1]
        J = self.job_finished_mask.shape[1]

        scheduled = (self.op_scheduled_flag[:, :n] == 1)
        masked_ct = np.where(scheduled, self.unnorm_op_ct[:, :n], 0.0)
        has_any = scheduled.any(axis=1)
        self.current_makespan[:] = np.where(
            has_any, np.max(masked_ct, axis=1), 0.0)

        finished = self.job_finished_mask[:, :J]
        env_idx = np.arange(self.number_of_envs)[:, None]
        last_ops = self.job_last_op_id[:, :J]
        cts = self.unnorm_op_ct[env_idx, last_ops]
        tardiness = np.maximum(0.0, cts - self.job_due_date[:, :J]) * self.job_weight[:, :J]
        self.current_twt[:] = np.sum(np.where(finished, tardiness, 0.0), axis=1)

    # ------------------------------------------------------------------
    # Fixed-schedule seeding (used by InferenceEnv)
    # ------------------------------------------------------------------

    def _seed_earliest_start_times(self, n: int, J: int):
        """Rebuild ``op_earliest_start_time`` the way ``step()`` maintains it.

        Audit O4 — see ``FJSPEnvHGT._seed_earliest_start_times`` for the full
        argument. ``normalized_job_release_time`` is already divided by the
        per-env ``max_pt``, so no further scaling is needed here.
        """
        for env_idx in range(self.number_of_envs):
            jl = np.asarray(self.job_length[env_idx][:J], dtype=np.int64)
            if jl.sum() == 0:
                continue
            op_job = np.repeat(np.arange(len(jl)), jl)
            m = min(len(op_job), n)
            if m == 0:
                continue
            ops = np.arange(m)
            jobs = op_job[:m]
            is_first = ops == self.job_first_op_id[env_idx, jobs]
            pred_ct = self.op_ct[env_idx, np.maximum(ops - 1, 0)]
            release = self.normalized_job_release_time[env_idx, jobs]
            self.op_earliest_start_time[env_idx, ops] = np.where(
                is_first, release, pred_ct)

    def _seed_fixed_schedule(self, fixed_ops_per_env: List[List[Dict[str, Any]]],
                              mch_unavailable_until: Optional[np.ndarray] = None,
                              job_release_times: Optional[np.ndarray] = None,
                              current_time: float = 0.0):
        """Seed scheduling state from a list of fixed ops + per-machine
        unavailable-until floors + per-job release times.

        Call after set_initial_data_static (clean env). Allows starting a
        static inference from a partially-scheduled state — used by the
        InferenceEnv iterator to carry frozen ops across event boundaries.
        """
        E = self.number_of_envs
        M = self.number_of_machines
        n = self.number_of_ops
        J = self.number_of_jobs

        # --- 0. Release times ---
        if job_release_times is not None:
            jrt = np.asarray(job_release_times, dtype=np.float32)
            if jrt.ndim == 1:
                jrt = np.tile(jrt[None, :], (E, 1))
            nj = min(jrt.shape[1], self.job_release_time.shape[1])
            self.job_release_time[:, :nj] = jrt[:, :nj]
            self.normalized_job_release_time[:, :nj] = (
                jrt[:, :nj] / self.max_pt[:, None])

        # --- 0b. Global current-time floor (event-boundary state-of-world) ---
        unnorm_current_time = float(current_time)
        norm_current_time = unnorm_current_time / self.max_pt   # [E] now

        # --- 1. Normalize fixed_ops_per_env shape ---
        if fixed_ops_per_env is None:
            fixed_ops_per_env = [[] for _ in range(E)]
        elif len(fixed_ops_per_env) == 1 and E > 1:
            fixed_ops_per_env = [fixed_ops_per_env[0]] * E
        elif len(fixed_ops_per_env) != E:
            raise ValueError(
                f"fixed_ops_per_env length {len(fixed_ops_per_env)} != num_envs {E}")

        # --- 2. Write per-op fields (flags, times, machine) ---
        for env_idx, env_fixed_ops in enumerate(fixed_ops_per_env):
            for op in env_fixed_ops:
                op_id = int(op['op_id'])
                mch_id = int(op['machine_id'])
                start = float(op['start'])
                end = float(op['end'])
                self.op_scheduled_flag[env_idx, op_id] = 1
                self.op_assigned_machine[env_idx, op_id] = mch_id
                self.unnorm_op_start_time[env_idx, op_id] = start
                self.unnorm_op_ct[env_idx, op_id] = end
                self.op_start_time[env_idx, op_id] = start / self.max_pt[env_idx]
                self.op_ct[env_idx, op_id] = end / self.max_pt[env_idx]

        self._seed_earliest_start_times(n, J)

        # --- 3. Seed machine queues/free-time/idle from scheduled ops ---
        self.mch_queue[:] = -99
        self.mch_queue_len[:] = 0
        self.mch_free_time[:] = 0.0
        self.unnorm_mch_free_time[:] = 0.0
        self.mch_earliest_start_time[:] = 0.0
        self.mch_idle_time[:] = 0.0

        if mch_unavailable_until is not None:
            mu = np.asarray(mch_unavailable_until, dtype=np.float32)
            if mu.ndim == 1:
                mu = np.tile(mu[None, :], (E, 1))
        else:
            mu = np.zeros((E, M), dtype=np.float32)
        # is_broken feature consistency: a machine unavailable beyond the current
        # clock is "broken" until then. In inline training this is set by
        # apply_event_to_env (machine_breakdown); the anchored-segment driver and
        # InferenceEnv pass the same unavailable-until floor here. Mirrors the PT
        # env so the is_broken machine feature is populated in seeded segments.
        if hasattr(self, 'mch_broken_until'):
            self.mch_broken_until[:] = mu
        unnorm_floors = np.maximum(mu, unnorm_current_time)
        self.unnorm_mch_free_time[:] = np.maximum(
            self.unnorm_mch_free_time, unnorm_floors)
        self.mch_free_time[:] = self.unnorm_mch_free_time / self.max_pt[:, None]
        # Match the PT environment when the retained prefix is empty: repair
        # and current-time floors must still enter the machine observation.
        self.mch_earliest_start_time[:] = self.mch_free_time
        norm_floors_per_env = unnorm_floors / self.max_pt[:, None]

        for env_idx in range(E):
            scheduled_ops = np.where(
                (self.op_scheduled_flag[env_idx, :n] == 1) &
                (self.op_assigned_machine[env_idx, :n] >= 0)
            )[0]
            if len(scheduled_ops) == 0:
                continue

            mchs = self.op_assigned_machine[env_idx, scheduled_ops]
            cts = self.unnorm_op_ct[env_idx, scheduled_ops]
            norm_cts = self.op_ct[env_idx, scheduled_ops]

            order = np.lexsort((cts, mchs))
            sorted_ops = scheduled_ops[order]
            sorted_mchs = mchs[order]
            sorted_cts = cts[order]
            sorted_norm_cts = norm_cts[order]

            mch_counts = np.bincount(sorted_mchs, minlength=M)
            self.mch_queue_len[env_idx] = mch_counts
            pos = np.zeros(M, dtype=int)
            for i in range(len(sorted_ops)):
                m = int(sorted_mchs[i])
                self.mch_queue[env_idx, m, pos[m]] = sorted_ops[i]
                pos[m] += 1

            np.maximum.at(self.mch_free_time[env_idx], sorted_mchs, sorted_norm_cts)
            np.maximum.at(self.unnorm_mch_free_time[env_idx], sorted_mchs, sorted_cts)
            self.mch_earliest_start_time[env_idx] = self.mch_free_time[env_idx]

            sorted_norm_starts = self.op_start_time[env_idx, sorted_ops]
            prev_norm_ct = np.zeros_like(sorted_norm_cts)
            prev_norm_ct[1:] = np.where(
                sorted_mchs[1:] == sorted_mchs[:-1], sorted_norm_cts[:-1], 0.0)
            seen = set()
            for i in range(len(sorted_mchs)):
                m = int(sorted_mchs[i])
                if m not in seen:
                    seen.add(m)
                    prev_norm_ct[i] = norm_floors_per_env[env_idx, m]
            gaps = np.maximum(sorted_norm_starts - prev_norm_ct, 0.0)
            np.add.at(self.mch_idle_time[env_idx], sorted_mchs, gaps)

            self.mch_total_compatible_op_nums[env_idx] -= np.sum(
                self.process_relation[env_idx, sorted_ops], axis=0)

        # --- 4. Per-job candidate, finished mask, remain counts ---
        for env_idx in range(E):
            for j in range(J):
                first_op = int(self.job_first_op_id[env_idx, j])
                last_op = int(self.job_last_op_id[env_idx, j])
                if first_op < 0 or last_op < 0:
                    continue
                ops = np.arange(first_op, last_op + 1)
                sched = self.op_scheduled_flag[env_idx, ops] == 1
                unsched = ~sched
                if unsched.any():
                    first_unsched_rel = int(np.argmax(unsched))
                    self.candidate[env_idx, j] = first_op + first_unsched_rel
                    self.job_finished_mask[env_idx, j] = False
                    unsched_abs = ops[unsched]
                    self.job_remain_op_nums[env_idx, j] = int(unsched.sum())
                    self.job_remain_sum_pt[env_idx, j] = float(
                        np.sum(self.op_mean_pt[env_idx, unsched_abs]))
                else:
                    self.candidate[env_idx, j] = last_op
                    self.job_finished_mask[env_idx, j] = True
                    self.job_remain_op_nums[env_idx, j] = 0
                    self.job_remain_sum_pt[env_idx, j] = 0.0

        # --- 5. Candidate times + PT/mask (vectorized) ---
        candidates = self.candidate[:, :J]
        first_ops = self.job_first_op_id[:, :J]
        env_bcast = np.arange(E)[:, None]
        has_pred = candidates > first_ops
        prev_ops = np.clip(candidates - 1, 0, n - 1)

        pred_end_norm = np.where(
            has_pred, self.op_ct[env_bcast, prev_ops],
            self.normalized_job_release_time[:, :J])
        pred_end_unnorm = np.where(
            has_pred, self.unnorm_op_ct[env_bcast, prev_ops],
            self.job_release_time[:, :J])
        self.candidate_free_time[:, :J] = np.maximum(
            pred_end_norm, norm_current_time[:, None])
        self.unnorm_candidate_free_time[:, :J] = np.maximum(
            pred_end_unnorm, unnorm_current_time)

        valid_cand = (candidates >= 0) & (candidates < n)
        vc_env, vc_job = np.where(valid_cand)
        vc_ops = candidates[vc_env, vc_job]
        self.candidate_pt[vc_env, vc_job] = self.normalized_op_pt[vc_env, vc_ops]
        self.candidate_mask[vc_env, vc_job] = self.reverse_process_relation[vc_env, vc_ops]

        active_cand = valid_cand & ~self.job_finished_mask[:, :J]
        if active_cand.any():
            a_env, a_job = np.where(active_cand)
            a_cand = candidates[a_env, a_job]
            self.op_earliest_start_time[a_env, a_cand] = self.candidate_free_time[a_env, a_job]

        finished = self.job_finished_mask[:, :J]
        fin_env, fin_job = np.where(finished)
        if len(fin_env) > 0:
            self.candidate_mask[fin_env, fin_job] = 1

        # --- 6. Recompute job_idle_time from scheduled ops ---
        self.job_idle_time[:, :J] = 0.0
        max_ops_per_job = int(np.max(self.job_length[:, :J])) if J > 0 else 0
        for offset in range(max_ops_per_job):
            op_abs = self.job_first_op_id[:, :J] + offset
            last_ops_arr = self.job_last_op_id[:, :J]
            valid = (op_abs <= last_ops_arr) & (op_abs < n)
            if not valid.any():
                continue
            v_env, v_job = np.where(valid)
            v_op = op_abs[v_env, v_job]
            sched = self.op_scheduled_flag[v_env, v_op] == 1
            if not sched.any():
                continue
            s_env = v_env[sched]
            s_job = v_job[sched]
            s_op = v_op[sched]
            if offset == 0:
                ready = self.normalized_job_release_time[s_env, s_job]
            else:
                ready = self.op_ct[s_env, s_op - 1]
            idle = np.maximum(0.0, self.op_start_time[s_env, s_op] - ready)
            np.add.at(self.job_idle_time, (s_env, s_job), idle)

        # --- 7. step_count, objectives, lower bound, features ---
        # Allocated width, not the frozen op count — same reason as
        # _recompute_objectives; padded slots have op_scheduled_flag == 0.
        self.step_count = np.sum(
            self.op_scheduled_flag == 1, axis=1).astype(np.int64)
        self._recompute_objectives()
        self.compute_objective_lower_bound()
        self.mch_candidate_compatible_op_nums = np.sum(~self.candidate_mask, axis=1)
        self.construct_op_features()
        self.construct_job_features()
        self.construct_mch_features()
        self.state.update(
            self.candidate_mask, self.candidate,
            self._build_batch_hetero_graphs(),
            self.device)


__all__ = ["FJSPEnvHGTNoPT", "HGTEnvStateNoPT"]
