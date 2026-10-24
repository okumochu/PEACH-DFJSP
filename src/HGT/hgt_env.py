import numpy as np
import numpy.ma as ma
import copy
from typing import List, Dict, Any, Optional
import torch
from torch_geometric.data import HeteroData

from src.utils.event_utils import (
    fire_due_events,
    init_event_queues,
    merge_and_sort_events,
    reset_event_counters,
)


def _zscore_scale(std, eps: float = 1e-8, floor: float = 1e-6):
    """Divisor for feature Z-scoring, with a constant-column guard.

    ``std + eps`` alone turns a numerically CONSTANT column into pure noise:
    features are normalised to O(1), so a column whose spread is float32
    accumulation error (``std`` ~ 1e-7) gets divided by ~1e-7 and the residual
    lands at O(1) z-units. It is reachable in normal use — ``job_remain_sum_pt``
    is maintained by repeated subtraction in ``step()`` but summed afresh when a
    segment is seeded, and once every job is finished the column is all-zero up
    to that error. Measured: raw agreement to <1e-5, z-scored disagreement of
    2.26.

    Above ``floor`` this is exactly the old ``std + eps``; at or below it the
    divisor is infinite, so a constant column reads 0 — which is what "carries
    no information" should look like.
    """
    return np.where(std > floor, std + eps, np.inf)


class HGTEnvState:
    """Environment state for HGT environment."""

    def __init__(self):
        self.candidate_mask_tensor = None  # Candidate mask [sz_b, J, M]
        self.candidate_tensor = None  # Candidate operations [sz_b, J]
        self.candidate_pt_indices = None  # [sz_b, J, M] indices into PT nodes
        self.graph_tensor = None  # HeteroData graph for HGT processing

    def update(self, candidate_mask, candidate, candidate_pt_indices, graph, device):
        """Update state tensors from numpy arrays.

        Only transfers candidate_mask, candidate, candidate_pt_indices to GPU.
        Op/machine features are already embedded in the graph's HeteroData nodes.
        """
        # No CPU fast path. torch.from_numpy SHARES storage, and both .to('cpu')
        # and .long() on an already-int64 array are no-ops that return the same
        # tensor -- so on --device cpu every state pushed into PPO's Memory was
        # a live VIEW of self.candidate / self.candidate_pt_indices, which
        # step() rewrites in place. Every stored observation then became the
        # segment's terminal state and the update trained on it. Invisible on
        # CUDA, where .to(device) copies, which is why it never showed up.
        self.candidate_mask_tensor = torch.tensor(
            candidate_mask, dtype=torch.float32, device=device)
        self.candidate_tensor = torch.tensor(
            candidate, dtype=torch.long, device=device)
        self.candidate_pt_indices = torch.tensor(
            candidate_pt_indices, dtype=torch.long, device=device)
        self.graph_tensor = graph


class FJSPEnvHGT:
    """Multi-Criteria FJSP Environment with HGT (training + inference).

    Fires dynamic events mid-rollout whenever ``static_data['dynamic_events']``
    is non-empty so PPO trains on the same distribution it faces at inference.
    ``InferenceEnv`` wraps this class differently — it carves dynamic events
    into static segments instead of in-lining them.

    Optimizes: Makespan + Total Weighted Tardiness (TWT).

    Process-time (PT) nodes: one PT node per (op, compatible machine), with
    bidirectional PT-op / PT-machine edges; pt features = [normalized_pt,
    assignment_flag].
    """

    def __new__(cls, n_j, n_m, device, use_lowerbound_differencing=True,
                use_pt_nodes=True, use_dynamic_event_features=True,
                reward_measure='dp',
                *args, **kwargs):
        if cls is FJSPEnvHGT and not use_pt_nodes:
            if str(reward_measure) != 'dp':
                # The chain measure is implemented in THIS class only. Finding L
                # happened because a fix was never mirrored into the env
                # without PT nodes
                # and the two ablation arms silently diverged; refuse loudly
                # rather than let the Without PT nodes arm run a different reward.
                raise ValueError(
                    "reward_measure='chain' is not implemented for the env "
                    "without PT nodes. The Job chain lower-bound reward arm must "
                    "run with PT nodes on.")
            from src.HGT.hgt_env_no_pt import FJSPEnvHGTNoPT
            return FJSPEnvHGTNoPT(
                n_j, n_m, device,
                use_lowerbound_differencing=use_lowerbound_differencing,
                use_dynamic_event_features=use_dynamic_event_features)
        return super().__new__(cls)

    def __init__(self, n_j, n_m, device, use_lowerbound_differencing=True,
                 use_pt_nodes=True, use_dynamic_event_features=True,
                 reward_measure='dp'):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.device = device
        self.use_lowerbound_differencing = bool(use_lowerbound_differencing)
        # WHICH admissible measurement the reward differences, when
        # use_lowerbound_differencing is True:
        #   'dp'    -- the DP bound, which consults machine free times (default)
        #   'chain' -- the job-chain lower bound used by prior lower-bound rewards,
        #              which charges each remaining operation only its minimum
        #              compatible process time and never asks whether that
        #              machine is free.
        # Ignored when use_lowerbound_differencing is False (partial-objective
        # reward). This exists so the reward ablation can test the middle
        # term of the tightness chain obj_partial <= LB_chain <= LB_DP, i.e. the
        # measurement prior work actually uses, not only the weakest one.
        if str(reward_measure) not in ('dp', 'chain'):
            raise ValueError(
                f"reward_measure must be 'dp' or 'chain', got {reward_measure!r}")
        self.reward_measure = str(reward_measure)
        self.use_pt_nodes = True
        # Dynamic-event observability features (new design, default). When
        # False (old/static design) the two extra columns are dropped:
        # machine ``is_broken`` (mch_fea_dim 7->6) and job ``released_time``
        # (job_fea_dim 8->7). The network reads these dims back from the env,
        # so its input projections shrink to match automatically; the flag is
        # persisted in the checkpoint (hgt_hyperparams) so inference rebuilds
        # the matching env. See train_graph_rl_static.py / config.static_design.
        self.use_dynamic_event_features = bool(use_dynamic_event_features)

        # Feature dimensions
        self.op_fea_dim = 9
        # +1 (index 6): is_broken flag (machine_breakdown observability)
        self.mch_fea_dim = 7 if self.use_dynamic_event_features else 6
        # +1 (index 7): normalized release time (new-job arrivals)
        self.job_fea_dim = 8 if self.use_dynamic_event_features else 7
        self.pt_fea_dim = 2   # [normalized_pt, assignment_flag]

        # Multi-criteria objective weights
        self.makespan_weight = 1.0
        self.twt_weight = 1.0

        # Graph structure caching
        self.cached_graph_structure = None
        self.graph_structure_initialized = False

        # Initialize state
        self.initial_state = HGTEnvState()

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

        # PT node related
        self.pt_node_idx = None
        self.num_pt_nodes_per_env = None
        self.pt_assignment_flag = None
        self.candidate_pt_indices = None

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
        # Frozen reference counts. number_of_jobs / number_of_ops stay at these
        # INITIAL values for the whole episode; per-env growth from dynamic
        # arrivals is tracked in n_active_* (see initial_vars). Used as the
        # Z-score normalization reference in construct_*_features.
        self.initial_number_of_jobs = self.number_of_jobs
        self.initial_number_of_ops = self.number_of_ops

        # Determine allocation sizes (pre-allocated for dynamic mode)
        if hasattr(self, 'alloc_jobs') and self.alloc_jobs is not None:
            self.alloc_jobs_size = self.alloc_jobs
            self.alloc_ops_size = self.alloc_ops
            self.alloc_pt_size = self.alloc_pt_nodes
        else:
            self.alloc_jobs_size = self.number_of_jobs
            self.alloc_ops_size = self.number_of_ops
            self.alloc_pt_size = self.number_of_ops * self.number_of_machines

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

        # PER-ENV, not a batch scalar. np.max over the whole [E, n_ops, n_m]
        # batch made an instance's normalised features -- and pt_features[...,0]
        # in particular, which is fed to the network WITHOUT z-scoring -- depend
        # on which other instances happened to share the batch. Greedy inference
        # must be a function of the instance alone. At batch size 1 (every
        # reported experiment) this is numerically identical to the old scalar,
        # so existing checkpoints are unaffected there.
        _pt_max = np.max(self.op_pt, axis=(1, 2))                      # [E]
        self.max_pt = np.where(_pt_max > 0, _pt_max, 1.0).astype(np.float32)
        self.normalized_op_pt = self.op_pt / self.max_pt[:, None, None]

        # Process relation (compatibility)
        self.process_relation = (self.normalized_op_pt != 0)
        self.reverse_process_relation = ~self.process_relation
        self.compatible_mch = np.sum(self.process_relation[:, :self.number_of_ops, :], 1)
        self.compatible_op = np.sum(self.process_relation, 2)

        # PT node counts
        self.num_pt_nodes_per_env = np.sum(
            self.process_relation[:, :self.number_of_ops, :], axis=(1, 2))
        self.max_pt_nodes = self.alloc_pt_size

        # PT node index mapping
        alloc_ops = getattr(self, 'alloc_ops_size', self.number_of_ops)
        self.pt_node_idx = np.full(
            (self.number_of_envs, alloc_ops, self.number_of_machines), -1, dtype=np.int64)
        self.pt_features = np.zeros(
            (self.number_of_envs, self.max_pt_nodes, self.pt_fea_dim), dtype=np.float32)
        self.pt_assignment_flag = np.zeros(
            (self.number_of_envs, self.max_pt_nodes), dtype=np.float32)
        self.pt_to_op = np.full(
            (self.number_of_envs, self.max_pt_nodes), -1, dtype=np.int64)
        self.pt_to_machine = np.full(
            (self.number_of_envs, self.max_pt_nodes), -1, dtype=np.int64)

        # Initialize tracking variables
        self.initial_vars()

        # Build process time statistics (masked for compatibility)
        self.normalized_op_pt = ma.array(self.normalized_op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.normalized_op_pt, axis=2).data.astype(np.float32)
        self.op_min_pt = np.min(self.normalized_op_pt, axis=-1).data
        self.op_max_pt = np.max(self.normalized_op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt

        # Build PT nodes, op_progress, job features, candidate_pt_indices
        for env_idx in range(self.number_of_envs):
            # PT node index mapping and features
            pt_counter = 0
            for op_idx in range(self.number_of_ops):
                for m_idx in range(self.number_of_machines):
                    if self.process_relation[env_idx, op_idx, m_idx]:
                        self.pt_node_idx[env_idx, op_idx, m_idx] = pt_counter
                        self.pt_features[env_idx, pt_counter, 0] = self.normalized_op_pt[
                            env_idx, op_idx, m_idx]
                        self.pt_features[env_idx, pt_counter, 1] = 0.0
                        self.pt_to_op[env_idx, pt_counter] = op_idx
                        self.pt_to_machine[env_idx, pt_counter] = m_idx
                        pt_counter += 1

            # op_progress and job-level features
            for job_idx in range(self.number_of_jobs):
                first_op = self.job_first_op_id[env_idx][job_idx]
                last_op = self.job_last_op_id[env_idx][job_idx]
                job_len = self.job_length[env_idx, job_idx]

                for op_idx in range(first_op, last_op + 1):
                    self.op_progress[env_idx, op_idx] = (op_idx - first_op + 1) / job_len

                self.job_remain_sum_pt[env_idx, job_idx] = np.sum(
                    self.op_mean_pt[env_idx][first_op:last_op + 1])
                self.job_remain_op_nums[env_idx, job_idx] = job_len

                # candidate_pt_indices (vectorized over machines)
                candidate_op = self.candidate[env_idx, job_idx]
                self.candidate_pt_indices[env_idx, job_idx] = self.pt_node_idx[
                    env_idx, candidate_op]

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

        # Machine static features
        # An unused machine has no PT statistic; never expose masked fill
        # sentinels to the float32 feature standardization.
        self.mch_min_pt = np.min(self.normalized_op_pt, axis=1).filled(0)
        self.mch_mean_pt = np.mean(self.normalized_op_pt, axis=1).filled(0).astype(np.float32)
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
        self.initial_pt_features = np.copy(self.pt_features)
        self.initial_candidate_pt_indices = np.copy(self.candidate_pt_indices)
        # Machine pt aggregates: mutated in place by _activate_arriving_job, so
        # snapshot for reset() (no-op for static / breakdown-only runs).
        self.initial_mch_min_pt = np.copy(self.mch_min_pt)
        self.initial_mch_mean_pt = np.copy(self.mch_mean_pt)

    # ------------------------------------------------------------------
    # High-level entrypoints (training + inference)
    # ------------------------------------------------------------------

    def _realised_objective(self):
        """Objective already realised by the CURRENT schedule, in reward units.

        Must match ``_compute_step_reward``'s partial-objective reward branch
        exactly, or the first step after an anchor is charged the difference.
        """
        return ((self.makespan_weight * np.maximum(self.current_makespan, 0.0)
                 + self.twt_weight * self.current_twt)
                / self.max_pt).astype(np.float32)

    def _init_reward_tracking(self):
        """Anchor the reward tracker to the CURRENT state, in either mode.

        The partial-objective reward used to hard-zero this. That is right from an
        empty schedule but wrong after ``set_initial_data_with_fixed_schedule``
        seeds a frozen prefix, which is what every anchored-segment boundary
        does: ``current_makespan``/``current_twt`` already carry the prefix, so
        the segment's first step was charged ``0 - obj(after step)`` instead of
        ``obj(seed) - obj(after step)`` — a spurious penalty equal to the whole
        prefix objective, once per segment. Only the ``dp_lb_diff.enabled=False``
        arm was affected (the LB branch already re-anchors via init_quality),
        i.e. exactly the Contribution-2 ablation baseline.

        From an empty schedule ``current_makespan`` is -inf and ``current_twt``
        is 0, so this still yields 0 and the static path is unchanged.
        """
        if self.use_lowerbound_differencing:
            if self.reward_measure == 'chain':
                # init_quality caches the DP bound, so the chain arm has to
                # recompute its own anchor from the current state. It matters
                # only at an anchored-segment boundary: from an empty schedule
                # every machine is free and the two bounds coincide.
                self.max_endTime = self.compute_objective_lower_bound_chain()
            else:
                self.max_endTime = np.copy(self.init_quality)
        else:
            self.max_endTime = self._realised_objective()

    def set_initial_data_static(self, static_data: Dict[str, Any]):
        """Set initial data from a static-shape dict.

        ``static_data['dynamic_events']`` is optional — when non-empty,
        events fire mid-rollout via ``_post_step_hook``.
        """
        data = static_data['static']
        # Pre-allocate to max sizes ONLY when arrivals exist, so breakdown-only
        # / static runs keep the exact static code path (no-op). Must be set
        # BEFORE set_initial_data so its padding branches activate.
        dj = static_data.get('dynamic_jobs', None)
        n_arrivals = (len(np.asarray(dj.get('job_length', []))) if dj is not None else 0)
        if n_arrivals > 0:
            self.alloc_jobs = int(static_data['max_total_jobs'])
            self.alloc_ops = int(static_data['max_total_ops'])
            self.alloc_pt_nodes = int(static_data['max_total_pt_nodes'])
        else:
            self.alloc_jobs = None
            self.alloc_ops = None
            self.alloc_pt_nodes = None
        self.set_initial_data(data['job_length'], data['op_pt'])
        self._set_twt_data(data['job_weight'], data['job_due_date'])

        # Dynamic-arrival bookkeeping: store the shared arriving-job template and
        # the a-priori real total op count (initial + all arriving). real_total_ops
        # is identical across envs (arrivals are shared) -> all envs finish on the
        # same step. Empty arrivals -> real_total_ops == number_of_ops (no-op).
        self.dynamic_jobs = dj if n_arrivals > 0 else None
        if n_arrivals > 0:
            _jl = np.asarray(dj['job_length'], dtype=np.int64)
            self._arrival_op_offsets = np.concatenate([[0], np.cumsum(_jl)]).astype(np.int64)
            self.real_total_ops = int(self.number_of_ops + _jl.sum())
        else:
            self._arrival_op_offsets = None
            self.real_total_ops = int(self.number_of_ops)

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
        # Re-capture the baseline quality/reward after seeding so step rewards
        # compute against the post-seed initial LB.
        self.init_quality = self._compute_objective_from_lb()
        self._init_reward_tracking()
        self.initial_quality = np.copy(self.init_quality)
        self._save_twt_initial_data()
        # Keep inputs rather than another full graph/environment snapshot.
        # Rebuilding on reset restores every seeded array, graph, and reward
        # anchor together. Copy because segment callers grow these inputs.
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
        # Per-env active prefix counts (how many jobs/ops/pt-nodes have arrived).
        # Start at the initial counts; dynamic arrivals extend them contiguously.
        # The graph is sliced to these so unarrived nodes never reach the policy.
        self.n_active_jobs = np.full(self.number_of_envs, self.number_of_jobs, dtype=np.int64)
        self.n_active_ops = np.full(self.number_of_envs, self.number_of_ops, dtype=np.int64)
        _npt = getattr(self, 'num_pt_nodes_per_env', None)
        self.n_active_pt = (np.asarray(_npt, dtype=np.int64).copy() if _npt is not None
                            else np.zeros(self.number_of_envs, dtype=np.int64))
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

        # PT assignment flags
        self.pt_assignment_flag = np.zeros(
            (self.number_of_envs, self.max_pt_nodes), dtype=np.float32)
        self.job_idle_time = np.zeros((self.number_of_envs, alloc_jobs), dtype=np.float32)
        self.candidate_pt_indices = np.full(
            (self.number_of_envs, alloc_jobs, self.number_of_machines), -1, dtype=np.int64)

        # Operation-machine assignments (for rescheduling)
        self.op_assigned_machine = np.full((self.number_of_envs, alloc_ops), -1, dtype=int)
        self.unnorm_op_start_time = np.zeros((self.number_of_envs, alloc_ops), dtype=np.float32)

        # Dynamic event tracking (populated by subclass)
        self.breakdown_periods = []
        self.interrupted_ops = []
        # Per-machine unnormalized breakdown end time (0 = never broken).
        # Drives the is_broken machine feature; reset here every episode.
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

        A SCHEDULED operation's completion time is known exactly, so that IS
        its lower bound. ``compute_objective_lower_bound``'s DP only walks
        forward from each job's candidate and never revisits an operation once
        it is scheduled, so without the pin below a scheduled operation keeps
        whatever the DP last wrote while it was still a candidate — and that is
        a DIFFERENT moment on the two paths: the step at which the policy chose
        it (stepping) versus the initial DP before anything was scheduled
        (seeding a frozen prefix). Audit O4; measured at up to 1.04 z-units on
        ``fea_op[:,:,4]`` at 20x10.

        It has to live here rather than inside the DP because ``step()``
        computes the lower bound (for the reward) BEFORE it sets
        ``op_scheduled_flag`` for the operation it just placed, so a pin inside
        the DP would always miss that one operation — which is exactly the
        residual the equivalence check caught. ``construct_op_features`` runs
        after every state change on both paths, so pinning here is evaluated
        against current flags in both. Idempotent, and ``job_ct_lb`` is
        computed from ``current_ect`` and unaffected.
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
        # Each instance owns its scale. A batch-wide maximum can suppress
        # another instance's due-date spread below the constant-column floor.
        # Zero/all-infinite due dates need a positive finite divisor as well.
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
        # Release time (index 7, dynamic-event design only): 0 for all
        # initially-present jobs; >0 only for jobs that arrive mid-episode
        # (dynamic new-job arrivals). Inert (constant 0 -> Z-scored to 0) when
        # arrivals are disabled. The old/static design has no release-time
        # column (job_fea_dim == 7), so it is skipped entirely there.
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
        """Construct machine features (6 or 7 dims) with Z-score normalization.

        The 7th column (index 6, ``is_broken``) is only present in the
        dynamic-event design (``use_dynamic_event_features=True``).
        """
        n_ref_jobs = getattr(self, 'initial_number_of_jobs', self.number_of_jobs)
        n_ref_ops = getattr(self, 'initial_number_of_ops', self.number_of_ops)
        self.fea_m[:, :, 0] = self.mch_candidate_compatible_op_nums / n_ref_jobs
        self.fea_m[:, :, 1] = self.mch_total_compatible_op_nums / n_ref_ops
        self.fea_m[:, :, 2] = self.mch_min_pt
        self.fea_m[:, :, 3] = self.mch_mean_pt
        self.fea_m[:, :, 4] = self.mch_earliest_start_time
        self.fea_m[:, :, 5] = self.mch_idle_time
        # is_broken (index 6, dynamic-event design only): 1 while a fired
        # machine_breakdown's repair window is still ahead of the schedule
        # frontier (current_makespan), else 0. Complements
        # mch_earliest_start_time (which encodes *until when*) by flagging
        # *why*. The old/static design has no is_broken column
        # (mch_fea_dim == 6), so it is skipped entirely there.
        if self.use_dynamic_event_features:
            sim_t = np.maximum(self.current_makespan, 0.0)[:, None]
            self.fea_m[:, :, 6] = ((self.mch_broken_until > sim_t)
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
            n_pt = cached['num_pt_edges']
            n_job = cached['num_op_job_edges']
            return (
                cached['precedence_edges'][:, :n_prec],
                cached['op_to_job_edges'][:, :n_job],
                cached['pt_to_op_edges'][:, :n_pt],
                cached['op_to_pt_edges'][:, :n_pt],
                cached['pt_to_machine_edges'][:, :n_pt],
                cached['machine_to_pt_edges'][:, :n_pt],
            )
        else:
            # Static mode: exact-fit, use full tensors
            return (
                cached['precedence_edges'],
                cached['op_to_job_edges'],
                cached['pt_to_op_edges'],
                cached['op_to_pt_edges'],
                cached['pt_to_machine_edges'],
                cached['machine_to_pt_edges'],
            )

    def _build_batch_hetero_graphs(self):
        """Build per-env HeteroData graphs from the current node features.

        Each env's graph is sliced to its ARRIVED prefix (n_active_*) so no
        unarrived node ever enters the graph (no future peeking); the slice
        equals the static counts when no arrivals have fired.
        """
        # copy=True is REQUIRED, not an optimisation detail: .to('cpu') on a
        # from_numpy tensor returns the SAME tensor, so without it every
        # HeteroData .x aliased self.fea_* -- which construct_*_features
        # rewrites in place each step -- and the graphs stored in Memory all
        # showed the terminal state. On CUDA the transfer already copies, so
        # this costs nothing there.
        # Batch convert all features to GPU tensors at once.
        # non_blocking=True allows H2D overlap; the next CUDA op syncs implicitly.
        fea_op_gpu = torch.from_numpy(self.fea_op).to(self.device, non_blocking=True, copy=True)
        fea_job_gpu = torch.from_numpy(self.fea_job).to(self.device, non_blocking=True, copy=True)
        fea_m_gpu = torch.from_numpy(self.fea_m).to(self.device, non_blocking=True, copy=True)
        pt_fea_gpu = torch.from_numpy(self.pt_features).to(self.device, non_blocking=True, copy=True)

        batch_graphs = []
        for env_idx in range(self.number_of_envs):
            cached = self.cached_graph_structure[env_idx]
            num_actual_ops = int(self.n_active_ops[env_idx])
            num_jobs = int(self.n_active_jobs[env_idx])
            num_pt_nodes = int(self.n_active_pt[env_idx])

            (prec_e, job_e, pt_op_e, op_pt_e,
             pt_mch_e, mch_pt_e) = self._get_valid_edges(cached)

            data = HeteroData()
            data['op'].x = fea_op_gpu[env_idx, :num_actual_ops]
            data['job'].x = fea_job_gpu[env_idx, :num_jobs]
            data['machine'].x = fea_m_gpu[env_idx]
            data['pt'].x = pt_fea_gpu[env_idx, :num_pt_nodes]

            data['op', 'precedes', 'op'].edge_index = prec_e
            data['op', 'belongs_to', 'job'].edge_index = job_e
            data['pt', 'connects_op', 'op'].edge_index = pt_op_e
            data['op', 'has_pt', 'pt'].edge_index = op_pt_e
            data['pt', 'connects_machine', 'machine'].edge_index = pt_mch_e
            data['machine', 'has_pt', 'pt'].edge_index = mch_pt_e

            batch_graphs.append(data)

        return batch_graphs

    def _get_edges(self, env_idx):
        """Get all edge types for specific environment in a single pass."""
        precedence_src = []
        precedence_dst = []
        job_ids = []
        op_ids = []
        pt_indices = []
        pt_op_indices = []
        pt_machine_indices = []

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
                    pt_idx = self.pt_node_idx[env_idx, op_idx, m_idx]
                    if pt_idx >= 0:
                        pt_indices.append(pt_idx)
                        pt_op_indices.append(op_idx)
                        pt_machine_indices.append(m_idx)

        precedence_edges = [precedence_src, precedence_dst]
        job_op_edges = [job_ids, op_ids]
        pt_to_op = [pt_indices, pt_op_indices]
        op_to_pt = [pt_op_indices, pt_indices]
        pt_to_machine = [pt_indices, pt_machine_indices]
        machine_to_pt = [pt_machine_indices, pt_indices]

        return precedence_edges, pt_to_op, op_to_pt, pt_to_machine, machine_to_pt, job_op_edges

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
            (precedence_edges, pt_to_op, op_to_pt,
             pt_to_machine, machine_to_pt, job_op_edges) = self._get_edges(env_idx)

            n_prec = len(precedence_edges[0])
            n_pt = len(pt_to_op[0])
            n_job = len(job_op_edges[0])
            op_job_data = [job_op_edges[1], job_op_edges[0]]  # reversed

            if is_dynamic:
                max_prec = self.alloc_ops_size
                max_pt = self.max_pt_nodes
                max_job_op = self.alloc_ops_size

                cached = {
                    'precedence_edges': self._make_edge_tensor(precedence_edges, max_prec),
                    'pt_to_op_edges': self._make_edge_tensor(pt_to_op, max_pt),
                    'op_to_pt_edges': self._make_edge_tensor(op_to_pt, max_pt),
                    'pt_to_machine_edges': self._make_edge_tensor(pt_to_machine, max_pt),
                    'machine_to_pt_edges': self._make_edge_tensor(machine_to_pt, max_pt),
                    'op_to_job_edges': self._make_edge_tensor(op_job_data, max_job_op),
                    'num_actual_ops': self.number_of_ops,
                    'num_precedence_edges': n_prec,
                    'num_pt_edges': n_pt,
                    'num_op_job_edges': n_job,
                }
            else:
                cached = {
                    'precedence_edges': self._make_edge_tensor(precedence_edges),
                    'pt_to_op_edges': self._make_edge_tensor(pt_to_op),
                    'op_to_pt_edges': self._make_edge_tensor(op_to_pt),
                    'pt_to_machine_edges': self._make_edge_tensor(pt_to_machine),
                    'machine_to_pt_edges': self._make_edge_tensor(machine_to_pt),
                    'op_to_job_edges': self._make_edge_tensor(op_job_data),
                    'num_actual_ops': self.number_of_ops,
                }

            self.cached_graph_structure.append(cached)

        # Snapshot initial valid-edge counts so reset() can revert edges that
        # dynamic arrivals appended (node arrays revert implicitly via n_active_*).
        self._initial_cached_counts = [
            (c.get('num_precedence_edges'), c.get('num_pt_edges'),
             c.get('num_op_job_edges'), c.get('num_actual_ops'))
            for c in self.cached_graph_structure]

        self.graph_structure_initialized = True

    def _active_job_bounds(self):
        """(J, job_active) for the LB / objective loops.

        J = max arrived jobs across envs; job_active[e, j] marks slots that have
        actually arrived in env e. Equals (number_of_jobs, all-True) when no
        arrivals have fired, so the static path is bit-identical.

        ``n_active_jobs`` is always set in ``initial_vars`` (run by every env,
        including the single-objective subclass) before any objective call.
        """
        n_active = self.n_active_jobs
        J = int(n_active.max()) if self.number_of_envs > 0 else self.number_of_jobs
        job_active = np.arange(J)[None, :] < n_active[:, None]
        return J, job_active

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
        J, job_active = self._active_job_bounds()

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

        # DP loop over operation offsets (max ~10 iterations). job_active gates
        # out unarrived (padded) job slots so their garbage indices never enter
        # the DP -- they contribute 0 to makespan and (via inf due date) 0 to TWT.
        remaining_ops = np.where(finished | ~job_active, 0, last_op - candidate + 1)  # [E, J]
        max_remaining = int(np.max(remaining_ops)) if E > 0 else 0
        active = (~finished) & job_active  # [E, J]

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

            # Scatter to op_ct_lb, ACTIVE ENTRIES ONLY.
            #
            # The np.where form below wrote every (e, j) pair, including
            # unarrived job slots whose candidate is -1 and whose abs_op_safe
            # therefore clips to 0 — the same index job 0 writes. Fancy-index
            # assignment is last-write-wins and the padded slot sits later in
            # row-major order, so it wrote back the value gathered BEFORE the DP
            # and silently reverted job 0's freshly computed lower bound.
            # Targets are unique among active entries (step_active implies
            # k <= last_op - candidate, and per-job op ranges are contiguous and
            # disjoint), so a plain indexed assignment is well defined.
            rows, cols = np.nonzero(step_active)
            if rows.size:
                self.op_ct_lb[rows, abs_op_safe[rows, cols]] = current_ect[rows, cols]

        # Write job-level results
        self.job_ct_lb[:, :J] = current_ect

        lb_makespan = np.max(current_ect, axis=1)
        return self._combine_objective(lb_makespan, J)

    def compute_objective_lower_bound_chain(self) -> np.ndarray:
        """Composite objective under the JOB-CHAIN relaxation.

        This is the measurement prior lower-bound rewards difference (Zhang
        2020; DANIEL), raised from makespan to the composite objective: a
        remaining operation is charged only its minimum compatible processing
        time, and machine availability is never consulted. It is therefore no
        tighter than ``compute_objective_lower_bound`` at any state, with
        equality when every machine is free.

        PURE READ. Unlike the DP version this does NOT write ``op_ct_lb`` /
        ``job_ct_lb``, which feed the observation. That is deliberate: the
        chain arm is an ablation of the REWARD only, so the policy must see
        exactly the same features as the DP arm and the single changed variable
        is which bound the reward differences.

        Returns:
            [E] array, the job-chain lower bound on the composite objective.
        """
        E = self.number_of_envs
        J, job_active = self._active_job_bounds()

        candidate = self.candidate[:, :J]
        first_op = self.job_first_op_id[:, :J]
        last_op = self.job_last_op_id[:, :J]
        finished = self.job_finished_mask[:, :J]

        op_pt_data = self.normalized_op_pt.data
        proc_rel = self.process_relation
        env_idx = np.arange(E)[:, None]

        # Same starting readiness as the DP sweep: predecessor completion, or
        # the release time at a job's first operation, or the realized
        # completion for a finished job.
        prev_op = np.maximum(candidate - 1, 0)
        current_ect = self.op_ct[env_idx, prev_op]
        current_ect = np.where(candidate == first_op,
                               self.normalized_job_release_time[:, :J],
                               current_ect)
        current_ect = np.where(finished, self.op_ct[env_idx, last_op], current_ect)

        remaining_ops = np.where(finished | ~job_active, 0, last_op - candidate + 1)
        max_remaining = int(np.max(remaining_ops)) if E > 0 else 0
        active = (~finished) & job_active

        for k in range(max_remaining):
            step_active = active & (k < remaining_ops)
            if not step_active.any():
                break

            abs_op = candidate + k
            abs_op_safe = np.clip(abs_op, 0, self.op_ct_lb.shape[1] - 1)

            op_pts = op_pt_data[env_idx, abs_op_safe, :]      # [E, J, M]
            compat = proc_rel[env_idx, abs_op_safe, :]        # [E, J, M]

            # The one line that differs from the DP sweep: no max() against
            # machine free time, so the job-chain lower bound stays optimistic
            # exactly where the shop is congested.
            ects = np.where(compat, current_ect[:, :, None] + op_pts, np.inf)
            min_ect = np.min(ects, axis=2)                    # [E, J]

            current_ect = np.where(step_active, min_ect, current_ect)

        lb_makespan = np.max(current_ect, axis=1)
        # Unarrived slots carry due_date = inf, so they contribute 0 tardiness.
        tardiness = np.maximum(
            0, current_ect - self.normalized_job_due_date[:, :J])
        lb_twt = np.sum(tardiness * self.job_weight[:, :J], axis=1)
        return self.makespan_weight * lb_makespan + self.twt_weight * lb_twt

    def _current_reward_measure(self) -> np.ndarray:
        """The bound the reward differences, per ``self.reward_measure``.

        Callers must have already invoked ``compute_objective_lower_bound()``
        for its feature side effects; this only selects the scalar.
        """
        if self.reward_measure == 'chain':
            return self.compute_objective_lower_bound_chain()
        return self._compute_objective_from_lb()

    def _combine_objective(self, lb_makespan: np.ndarray, J: int = None) -> np.ndarray:
        """Compute weighted objective from makespan LB and TWT LB.

        ``J`` (active-job bound) is passed by callers that already computed it
        to avoid rebuilding it on the per-step reward path; falls back to
        ``_active_job_bounds`` when omitted.
        """
        if J is None:
            J, _ = self._active_job_bounds()
        # Unarrived job slots have due_date = inf -> tardiness 0, contributing
        # nothing even though they fall within [:, :J].
        tardiness = np.maximum(0, self.job_ct_lb[:, :J] - self.normalized_job_due_date[:, :J])
        weighted_tardiness = tardiness * self.job_weight[:, :J]
        lb_twt = np.sum(weighted_tardiness, axis=1)
        return self.makespan_weight * lb_makespan + self.twt_weight * lb_twt

    def _compute_objective_from_lb(self) -> np.ndarray:
        """Compute combined objective from already-computed job_ct_lb (no DP)."""
        J, _ = self._active_job_bounds()
        lb_makespan = np.max(self.job_ct_lb[:, :J], axis=1)
        return self._combine_objective(lb_makespan, J)

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
                                  self.candidate_pt_indices,
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

        self.pt_features = np.copy(self.initial_pt_features)
        self.candidate_pt_indices = np.copy(self.initial_candidate_pt_indices)
        # Machine pt aggregates mutated by _activate_arriving_job (read every
        # step over ALL machines, so must revert even though node arrays slice
        # back to the initial prefix via n_active_*).
        self.mch_min_pt = np.copy(self.initial_mch_min_pt)
        self.mch_mean_pt = np.copy(self.initial_mch_mean_pt)

        # Revert dynamic-arrival edge growth (n_active_* already reset by
        # initial_vars, so node features/LB slice back to the initial prefix).
        cc = getattr(self, '_initial_cached_counts', None)
        if cc is not None:
            for c, (n_prec, n_pt, n_job, n_act) in zip(self.cached_graph_structure, cc):
                if n_prec is not None:
                    c['num_precedence_edges'] = n_prec
                    c['num_pt_edges'] = n_pt
                    c['num_op_job_edges'] = n_job
                c['num_actual_ops'] = n_act

        # Subclass hook for additional reset logic
        self._reset_hook()

        # Shallow copy: HGTEnvState only holds tensor/graph references that
        # state.update() reassigns on every step, so deepcopy is unnecessary.
        self.state = copy.copy(self.initial_state)
        return self.state

    def _reset_hook(self):
        """Restore reward tracking, TWT data, and event queues on reset."""
        if hasattr(self, 'initial_job_weight'):
            self._restore_twt_initial_data()
        if hasattr(self, 'initial_quality'):
            self.init_quality = np.copy(self.initial_quality)
            # The chain reward recomputes its anchor from current due dates
            # and weights, so restore those before initializing the tracker.
            self._init_reward_tracking()
        if hasattr(self, 'initial_event_queue'):
            self.event_queues = init_event_queues(
                self.initial_event_queue, self.number_of_envs)
            reset_event_counters(self)
            # mch_broken_until is freshly zero-allocated by initial_vars() at the
            # top of reset(); no separate clear needed here.

    def _compute_step_reward(self) -> np.ndarray:
        """Compute step reward.

        DP lower-bound reward (default):
            reward = previous_lb - current_lb
            With gamma=1, cumulative reward = initial_lb - final_obj
        partial-objective reward:
            reward = previous_obj - current_obj
            obj is makespan_weight*makespan + twt_weight*twt, normalized by
            max_pt to match the lower-bound scale.
        """
        current_lb = self.compute_objective_lower_bound()
        if not hasattr(self, 'max_endTime'):
            # Safety: caller built the env without set_initial_data_static
            # (e.g. via _create_batched_env). LB is still computed for
            # features; reward is discarded at inference, so zero is fine.
            return np.zeros(self.number_of_envs, dtype=np.float32)
        if self.use_lowerbound_differencing:
            # current_lb is computed unconditionally above for its feature side
            # effects; the chain arm differences a different scalar from the
            # SAME observation.
            measure = (self.compute_objective_lower_bound_chain()
                       if self.reward_measure == 'chain' else current_lb)
            reward = self.max_endTime - measure
            self.max_endTime = measure
        else:
            current_obj = (self.makespan_weight * np.maximum(self.current_makespan, 0.0)
                           + self.twt_weight * self.current_twt) / self.max_pt
            reward = self.max_endTime - current_obj
            self.max_endTime = current_obj
        return reward

    def _post_step_hook(self):
        """Fire due events, then re-anchor the reward tracker for fired envs.

        Runs AFTER ``_compute_step_reward`` so the just-computed step reward
        never includes the event effect. LB is recomputed unconditionally
        because ``op_ct_lb`` / ``job_ct_lb`` feed downstream features —
        breakdowns mutate ``mch_free_time`` and priority flips mutate
        ``job_weight``, both inputs to the DP.

        Reward-tracker reset (piecewise-stationary MDP) depends on mode:
        - DP lower-bound reward: ``max_endTime`` tracks LB. Re-anchor
          to the post-event LB so the next step's reward excludes the
          event-induced LB jump.
        - partial-objective reward: ``max_endTime`` tracks
          ``current_obj`` (makespan + TWT of already-completed work), which
          is invariant under events — breakdowns don't unschedule, and
          priority flips don't touch the already-accumulated ``current_twt``.
          Leaving ``max_endTime`` alone keeps the reward measuring the
          true per-step delta.
        """
        real_total = getattr(self, 'real_total_ops', self.number_of_ops)
        any_fired = np.zeros(self.number_of_envs, dtype=bool)
        # Time-advance-on-starvation only exists to realize new-job arrivals;
        # without them no env can be starved (candidates never depend on a
        # future event), so the loop fires once and the static / breakdown-only
        # / single-objective paths skip the per-step candidate_mask reduction.
        has_arrivals = getattr(self, 'dynamic_jobs', None) is not None
        # Bounded loop: each pass fires due events, then time-advances any env
        # that has unscheduled REAL ops but no schedulable candidate (its next
        # arrival has not fired yet) to that arrival's time. This keeps exactly
        # one op scheduled per env per step -> step counts stay synced ->
        # rectangular [T, num_envs] rollouts. Terminates when no env is starved
        # or no clock can advance (finite event queues).
        for _ in range(2 * self.number_of_envs + 8):
            events_fired = fire_due_events(self)
            if events_fired is not None and events_fired.any():
                any_fired |= events_fired
                new_lb = self.compute_objective_lower_bound()
                if self.use_lowerbound_differencing:
                    new_measure = (self.compute_objective_lower_bound_chain()
                                   if self.reward_measure == 'chain' else new_lb)
                    self.max_endTime = np.where(
                        events_fired, new_measure, self.max_endTime)
                self.construct_op_features()
                self.construct_job_features()
                # Rebuild machine features so the post-event graph reflects the
                # breakdown (is_broken + bumped earliest start) with no lag.
                self.construct_mch_features()

            queues = getattr(self, 'event_queues', None)
            if queues is None or not has_arrivals:
                break
            no_candidate = self.candidate_mask.reshape(
                self.number_of_envs, -1).all(axis=1)
            starved = (self.step_count < real_total) & no_candidate
            if not starved.any():
                break
            advanced = False
            for e in np.where(starved)[0]:
                q = queues[e]
                if not q:
                    continue
                nt = float(q[0]['time'])
                if nt > float(self.current_makespan[e]):
                    # current_makespan is float32; assigning the float64 event
                    # time truncates BELOW it, so fire_due_events (event_time <=
                    # makespan) would never fire the event and the handler would
                    # spin until the loop bound -> starved env -> garbage steps.
                    # Bump to the next float32 >= event_time so it reliably fires.
                    self.current_makespan[e] = np.nextafter(
                        np.float32(nt), np.float32(np.inf))
                    advanced = True
            if not advanced:
                break
        # Per-step boundary mask for GAE truncation (OR of all fires this step).
        self.last_events_fired = any_fired

    def step(self, actions):
        """Execute one scheduling step across all environments.

        Uses vectorized operations (no Python loop over environments).

        Only *active* (not-yet-done) envs are advanced. In the rectangular
        rollout (all envs share problem size and finish on the same step)
        ``active`` is all-True every step, so this is a no-op. In a ragged
        segment rollout (anchored-branching: envs seeded from different-size
        partial schedules finish at different steps) the guard prevents a
        finished env from re-appending to its machine queue or inflating its
        makespan while other envs keep scheduling. Inactive envs keep their
        terminal state untouched; the caller masks their transitions.
        """
        actions = np.asarray(actions)
        active = ~self.done()
        eidx = self.env_idxs[active]
        actions = actions[active]

        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[eidx, chosen_job]

        # step_count only advances for active envs (see method docstring).
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
            np.maximum(
                self.candidate_free_time[eidx, chosen_job],
                self.mch_free_time[eidx, chosen_mch]),
            self.normalized_job_release_time[eidx, chosen_job])
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
            np.maximum(
                self.unnorm_candidate_free_time[eidx, chosen_job],
                self.unnorm_mch_free_time[eidx, chosen_mch]),
            self.job_release_time[eidx, chosen_job])
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

        # 3. candidate_pt_indices (vectorized over machines)
        new_candidate_ops = self.candidate[eidx, chosen_job]
        self.candidate_pt_indices[eidx, chosen_job] = \
            self.pt_node_idx[eidx, new_candidate_ops]

        # 4. PT node assignment flag
        pt_idx = self.pt_node_idx[eidx, chosen_op, chosen_mch]
        valid = pt_idx >= 0
        if valid.any():
            v_envs = eidx[valid]
            v_pts = pt_idx[valid]
            self.pt_features[v_envs, v_pts, 1] = 1.0
            self.pt_assignment_flag[v_envs, v_pts] = 1.0

        # Compute reward (overridden by training env). Computed over all envs;
        # inactive (done) envs have unchanged state so their reward is ~0.
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

        # Post-step hook (dynamic events). Inert when no event queue is
        # attached (the segmented rollout strips events and applies them at
        # segment boundaries instead).
        self._post_step_hook()

        self.state.update(self.candidate_mask, self.candidate,
                          self.candidate_pt_indices,
                          self._build_batch_hetero_graphs(), self.device)
        self.done_flag = self.done()

        return self.state, np.array(reward), self.done_flag

    def done(self):
        """Check if all environments are done (per-env).

        Threshold is the REAL total op count (initial + all arriving ops),
        identical across envs since arrivals are shared, so every env finishes
        on the same step -> rectangular [T, num_envs] rollouts.
        """
        real_total = getattr(self, 'real_total_ops', self.number_of_ops)
        return (self.step_count >= real_total).astype(bool)

    # ------------------------------------------------------------------
    # Dynamic new-job arrivals (training)
    # ------------------------------------------------------------------

    def _activate_arriving_job(self, env_idx, job_idx, arrival_time):
        """Realize one dynamic new-job arrival in ``env_idx`` by activating the
        next contiguous job/op/pt slots from the shared ``dynamic_jobs`` template.

        Called from ``apply_event_to_env`` when a ``job_arrival`` event fires —
        i.e. AFTER the current step's reward/observation — so the policy first
        sees the new job on the NEXT step (it never peeks at a future arrival).
        Operates entirely on pre-allocated arrays and advances ``n_active_*``.
        """
        dj = self.dynamic_jobs
        if dj is None:
            return
        e = int(env_idx)
        L = int(np.asarray(dj['job_length'])[job_idx])
        if L <= 0:
            return
        M = self.number_of_machines
        j_slot = int(self.n_active_jobs[e])
        op0 = int(self.n_active_ops[e])
        op1 = op0 + L
        if j_slot >= self.alloc_jobs_size or op1 > self.alloc_ops_size:
            raise RuntimeError(
                f"arrival overflow env {e}: j_slot {j_slot}/{self.alloc_jobs_size}, "
                f"op1 {op1}/{self.alloc_ops_size}")

        off0 = int(self._arrival_op_offsets[job_idx])
        new_pt = np.asarray(dj['op_pt'], dtype=np.float32)[off0:off0 + L]   # [L, M]
        norm_new = new_pt / self.max_pt[e]
        comp = (norm_new != 0)                                             # [L, M] bool
        n_new_pt = int(comp.sum())
        if int(self.n_active_pt[e]) + n_new_pt > self.alloc_pt_size:
            raise RuntimeError(
                f"arrival pt overflow env {e}: {int(self.n_active_pt[e])}+"
                f"{n_new_pt}/{self.alloc_pt_size}")
        norm_release = float(arrival_time) / self.max_pt[e]

        # --- process times + compatibility ---
        self.op_pt[e, op0:op1, :] = new_pt
        self.normalized_op_pt.data[e, op0:op1, :] = norm_new
        self.normalized_op_pt.mask[e, op0:op1, :] = ~comp
        self.process_relation[e, op0:op1, :] = comp
        self.reverse_process_relation[e, op0:op1, :] = ~comp

        # --- per-op statistics (plain arrays; mirror set_initial_data) ---
        cnt = comp.sum(axis=1)                                             # [L]
        self.op_min_pt[e, op0:op1] = np.min(np.where(comp, norm_new, np.inf), axis=1)
        self.op_max_pt[e, op0:op1] = np.max(np.where(comp, norm_new, -np.inf), axis=1)
        self.op_mean_pt[e, op0:op1] = (np.where(comp, norm_new, 0.0).sum(axis=1)
                                       / np.maximum(cnt, 1))
        self.pt_span[e, op0:op1] = self.op_max_pt[e, op0:op1] - self.op_min_pt[e, op0:op1]
        self.compatible_op[e, op0:op1] = cnt

        # --- op-level state ---
        self.op_progress[e, op0:op1] = np.arange(1, L + 1, dtype=np.float32) / L
        self.op_available_mch_nums[e, op0:op1] = cnt
        self.op_earliest_start_time[e, op0] = norm_release
        self.op_to_job[e, op0:op1] = j_slot

        # --- job-level state ---
        self.job_length[e, j_slot] = L
        self.job_first_op_id[e, j_slot] = op0
        self.job_last_op_id[e, j_slot] = op1 - 1
        self.job_weight[e, j_slot] = float(np.asarray(dj['job_weight'])[job_idx])
        dd = float(np.asarray(dj['job_due_date'])[job_idx])
        self.job_due_date[e, j_slot] = dd
        self.normalized_job_due_date[e, j_slot] = dd / self.max_pt[e]
        self.job_release_time[e, j_slot] = float(arrival_time)
        self.normalized_job_release_time[e, j_slot] = norm_release
        self.job_remain_op_nums[e, j_slot] = L
        self.job_remain_sum_pt[e, j_slot] = float(self.op_mean_pt[e, op0:op1].sum())
        self.job_finished_mask[e, j_slot] = False
        self.job_idle_time[e, j_slot] = 0.0

        # --- candidate (first op of the arriving job); unmask compatible mch ---
        self.candidate[e, j_slot] = op0
        self.candidate_free_time[e, j_slot] = norm_release
        self.unnorm_candidate_free_time[e, j_slot] = float(arrival_time)
        self.candidate_pt[e, j_slot, :] = norm_new[0]
        self.candidate_mask[e, j_slot, :] = ~comp[0]

        # --- PT nodes for the new ops ---
        pt_counter = int(self.n_active_pt[e])
        new_pt_ids, new_pt_ops, new_pt_machines = [], [], []
        for k in range(L):
            abs_op = op0 + k
            for m in range(M):
                if comp[k, m]:
                    self.pt_node_idx[e, abs_op, m] = pt_counter
                    self.pt_features[e, pt_counter, 0] = norm_new[k, m]
                    self.pt_features[e, pt_counter, 1] = 0.0
                    self.pt_assignment_flag[e, pt_counter] = 0.0
                    self.pt_to_op[e, pt_counter] = abs_op
                    self.pt_to_machine[e, pt_counter] = m
                    new_pt_ids.append(pt_counter)
                    new_pt_ops.append(abs_op)
                    new_pt_machines.append(m)
                    pt_counter += 1
        self.candidate_pt_indices[e, j_slot, :] = self.pt_node_idx[e, op0]

        # --- machine aggregates ---
        self.mch_total_compatible_op_nums[e] += comp.sum(axis=0)
        comp_e = self.process_relation[e, :op1, :]
        norm_e = self.normalized_op_pt.data[e, :op1, :]
        m_cnt = comp_e.sum(axis=0)
        new_min = np.min(np.where(comp_e, norm_e, np.inf), axis=0)
        new_mean = np.where(comp_e, norm_e, 0.0).sum(axis=0) / np.maximum(m_cnt, 1)
        self.mch_min_pt[e] = np.where(m_cnt > 0, new_min, self.mch_min_pt[e])
        self.mch_mean_pt[e] = np.where(m_cnt > 0, new_mean,
                                       self.mch_mean_pt[e]).astype(np.float32)

        # --- extend cached graph edges (pre-allocated dynamic tensors) ---
        self._append_job_edges(e, op0, L, j_slot, new_pt_ids, new_pt_ops, new_pt_machines)

        # --- advance per-env active counts ---
        self.n_active_jobs[e] = j_slot + 1
        self.n_active_ops[e] = op1
        self.n_active_pt[e] = pt_counter

    def _append_job_edges(self, e, op0, L, j_slot, pt_ids, pt_ops, pt_machines):
        """Append an arrived job's edges into env e's pre-allocated cached edge
        tensors (dynamic mode) and bump the per-env valid-edge counts."""
        cached = self.cached_graph_structure[e]
        dev = self.device

        # precedence: op0->op0+1 ... within the job (L-1 edges)
        if L > 1:
            n = cached['num_precedence_edges']
            cached['precedence_edges'][0, n:n + L - 1] = torch.arange(op0, op0 + L - 1, device=dev)
            cached['precedence_edges'][1, n:n + L - 1] = torch.arange(op0 + 1, op0 + L, device=dev)
            cached['num_precedence_edges'] = n + (L - 1)

        # op -> job  (op_to_job_edges store [op, job]); one per op
        n = cached['num_op_job_edges']
        cached['op_to_job_edges'][0, n:n + L] = torch.arange(op0, op0 + L, device=dev)
        cached['op_to_job_edges'][1, n:n + L] = j_slot
        cached['num_op_job_edges'] = n + L

        # pt <-> op and pt <-> machine
        npt = len(pt_ids)
        if npt > 0:
            n = cached['num_pt_edges']
            pid = torch.tensor(pt_ids, dtype=torch.long, device=dev)
            pop = torch.tensor(pt_ops, dtype=torch.long, device=dev)
            pm = torch.tensor(pt_machines, dtype=torch.long, device=dev)
            cached['pt_to_op_edges'][0, n:n + npt] = pid
            cached['pt_to_op_edges'][1, n:n + npt] = pop
            cached['op_to_pt_edges'][0, n:n + npt] = pop
            cached['op_to_pt_edges'][1, n:n + npt] = pid
            cached['pt_to_machine_edges'][0, n:n + npt] = pid
            cached['pt_to_machine_edges'][1, n:n + npt] = pm
            cached['machine_to_pt_edges'][0, n:n + npt] = pm
            cached['machine_to_pt_edges'][1, n:n + npt] = pid
            cached['num_pt_edges'] = n + npt

        cached['num_actual_ops'] = op0 + L

    def _recompute_objectives(self):
        """Recompute current_makespan and current_twt from scratch (vectorized).

        Must be called after changing which operations are scheduled to avoid
        double-counting tardiness or retaining stale makespan values.

        Slices to the ALLOCATED width, not to ``number_of_ops`` /
        ``number_of_jobs``: those two are frozen at their initial values for
        the whole episode (see ``set_initial_data``) while an inline arrival
        grows ``n_active_*``, so slicing by them would silently drop every
        arrived job's completion time from the makespan and its tardiness from
        the TWT. Padded slots are safe to include — ``op_scheduled_flag`` is 0
        and ``job_finished_mask`` is False there by construction, so both masks
        already exclude them, and this is numerically identical to the old
        slice whenever nothing arrived inline (which is every live path today:
        training and ``InferenceEnv`` both rebuild the env from a static dict
        already grown to include the arrived jobs).
        ``bench/check_objective_invariants.py`` asserts the result against the
        schedule actually produced.
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

        Audit O4. ``step()`` only ever writes this field for the SUCCESSOR of
        the operation it just scheduled (``= that operation's completion``), so
        an already-scheduled operation keeps the value it carried as a
        candidate: its predecessor's completion time, or the job's release time
        if it is the job's first operation. Seeding used to overwrite it with
        the operation's own ACTUAL start instead.

        Those two disagree whenever an operation waited for a machine after its
        predecessor finished — and because an anchored segment mixes a seeded
        frozen prefix with operations scheduled inside the segment, the SAME
        feature column carried two different meanings in one graph. Measured at
        20x10: ``fea_op[:,:,5]`` differed by up to 2.6 z-units, and ``op_ct_lb``
        (which is derived from it) inherited the error exactly.

        One rule everywhere: earliest start = predecessor completion, or the
        release time for a job's first operation. The ``current_time`` floor is
        applied afterwards, to candidates only, via ``candidate_free_time`` —
        for an operation scheduled inside a segment the floor can never bind
        (it finished after ``current_time`` by construction), so the rule is
        the same on both paths.

        Verified by ``bench/check_seed_step_equivalence.py``: seed(S) must
        reproduce step-to(S) element-wise.
        """
        for env_idx in range(self.number_of_envs):
            jl = np.asarray(self.job_length[env_idx][:J], dtype=np.int64)
            if jl.sum() == 0:
                continue
            op_job = np.repeat(np.arange(len(jl)), jl)          # op -> job
            m = min(len(op_job), n)
            if m == 0:
                continue
            ops = np.arange(m)
            jobs = op_job[:m]
            is_first = ops == self.job_first_op_id[env_idx, jobs]
            # op_ct is 0 for an unscheduled predecessor, which is exactly what
            # step() leaves behind for an operation that is not yet reachable.
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
        dynamic orchestrator to carry frozen ops across event boundaries.

        Args:
            fixed_ops_per_env: per-env list of dicts with keys
                {op_id, machine_id, start, end}. A single list is
                broadcast to all envs.
            mch_unavailable_until: [E, n_m] or [n_m]; 0 means free.
            job_release_times: [E, n_j] or [n_j]; 0 by default.
            current_time: scalar; "we are now at this simulation time".
                Floors mch_free_time and candidate_free_time so every
                unscheduled op starts at or after this time. 0.0 (the
                static default) is a no-op floor.
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

        # --- 2. Write per-op fields (flags, times, machine, PT flags) ---
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
                pt_idx = self.pt_node_idx[env_idx, op_id, mch_id]
                if pt_idx >= 0:
                    self.pt_features[env_idx, pt_idx, 1] = 1.0
                    self.pt_assignment_flag[env_idx, pt_idx] = 1.0

        self._seed_earliest_start_times(n, J)

        # --- 3. Seed machine queues/free-time/idle from scheduled ops ---
        self.mch_queue[:] = -99
        self.mch_queue_len[:] = 0
        self.mch_free_time[:] = 0.0
        self.unnorm_mch_free_time[:] = 0.0
        self.mch_earliest_start_time[:] = 0.0
        self.mch_idle_time[:] = 0.0

        # Apply max(mch_unavailable_until, current_time) as the per-machine floor.
        if mch_unavailable_until is not None:
            mu = np.asarray(mch_unavailable_until, dtype=np.float32)
            if mu.ndim == 1:
                mu = np.tile(mu[None, :], (E, 1))
        else:
            mu = np.zeros((E, M), dtype=np.float32)
        # is_broken feature consistency with training: a machine unavailable
        # beyond the current clock is "broken" until then. In training this is
        # set by apply_event_to_env (machine_breakdown); here the inference
        # orchestrator passes the same unavailable-until floor. Without this the
        # is_broken machine feature (fea_m[:,:,6]) would be stuck at 0 at
        # inference even when a machine is down -> train/inference mismatch.
        if hasattr(self, 'mch_broken_until'):
            self.mch_broken_until[:] = mu
        unnorm_floors = np.maximum(mu, unnorm_current_time)
        self.unnorm_mch_free_time[:] = np.maximum(self.unnorm_mch_free_time, unnorm_floors)
        self.mch_free_time[:] = self.unnorm_mch_free_time / self.max_pt[:, None]
        # Empty retained prefixes still need the event/repair availability
        # floors in the machine observation; their queue loop is skipped below.
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

            # Idle time: sum of gaps between consecutive ops per machine
            # (first op on each machine uses that machine's unavailable-until floor)
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

        # --- 5. Candidate times + PT/mask/indices (vectorized) ---
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
        self.unnorm_candidate_free_time[:, :J] = np.maximum(pred_end_unnorm, unnorm_current_time)

        valid_cand = (candidates >= 0) & (candidates < n)
        vc_env, vc_job = np.where(valid_cand)
        vc_ops = candidates[vc_env, vc_job]
        self.candidate_pt[vc_env, vc_job] = self.normalized_op_pt[vc_env, vc_ops]
        self.candidate_mask[vc_env, vc_job] = self.reverse_process_relation[vc_env, vc_ops]
        self.candidate_pt_indices[vc_env, vc_job] = self.pt_node_idx[vc_env, vc_ops]

        # Earliest-start for (unfinished) candidate ops = candidate_free_time
        active_cand = valid_cand & ~self.job_finished_mask[:, :J]
        if active_cand.any():
            a_env, a_job = np.where(active_cand)
            a_cand = candidates[a_env, a_job]
            self.op_earliest_start_time[a_env, a_cand] = self.candidate_free_time[a_env, a_job]

        # Finished jobs: mask all machines
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
            self.candidate_pt_indices,
            self._build_batch_hetero_graphs(),
            self.device)
