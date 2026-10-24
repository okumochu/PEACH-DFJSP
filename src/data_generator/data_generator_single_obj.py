"""
Single-objective (makespan-only) data generator for FJSP.

Reproduces DANIEL's SD1 and SD2 instance distributions (Wang et al. 2024)
so single-objective HGT models can be trained head-to-head with the
published baselines. Selected via ``config.generator_type``.

SD1 (Song et al. 2023 ``CaseGenerator``):
- operations per job ~ U[int(0.8*n_m), int(1.2*n_m)]
- compatible machines per op ~ U[1, n_m]
- per-op mean PT ~ U[1, 20]; per-(op, machine) PT ~
  U[max(1, round(mean*0.8)), min(20, round(mean*1.2))]

SD2 (DANIEL ``SD2_instance_generator`` in "mix" mode):
- operations per job = n_m (fixed)
- compatible machines per op ~ U[1, n_m]
- per-(op, machine) PT ~ U[1, 99]; incompatible machines zeroed out
"""

import numpy as np
from typing import Any, Dict, List, Tuple


SD_PT_MAX = {'SD1': 20, 'SD2': 99}


class DataGeneratorSingleObj:
    """SD1/SD2 FJSP instance sampler.

    ``sample_env(num_envs)`` shares a ``job_length`` template across the
    batch (matching DANIEL's ``prepare_JobLength`` pattern) so the batched
    env can assume identical episode length per update. For experiment-time
    dynamic benchmarks it also emits the same ``dynamic_events`` /
    ``dynamic_jobs`` envelope as the multi-objective generator. Synthetic
    unit weights and finite due dates are included for OR-Tools/dispatch
    compatibility only; the single-objective experiments pass TWT weight 0.

    SD1 and SD2 are kept as distinct sampler methods rather than unified
    behind a flag; the PT-distribution differences are spec-driven
    (mean-and-deviation vs uniform) and splitting reads more clearly than
    a branched inner loop.
    """

    def __init__(self, config):
        self.config = config
        # Contract: caller (factory) sets a valid generator_type — we let
        # KeyError surface for invalid values rather than masking with a
        # default.
        self.generator_type = config.generator_type
        self.process_time_max = SD_PT_MAX[self.generator_type]

        self.n_j = int(config.n_j)
        self.n_m = int(config.n_m)
        self.params = self._load_params()

        # DANIEL spec constants — fixed by SD1/SD2 definition, not knobs.
        self.machine_per_operation_min = 1
        self.machine_per_operation_max = self.n_m
        self.process_time_min = 1
        # Only SD1 reads these (SD2 uses a fixed n_m ops per job).
        self.operation_per_job_min = int(0.8 * self.n_m)
        self.operation_per_job_max = int(1.2 * self.n_m)

    def sample_env(self, num_envs: int) -> Dict[str, Any]:
        """Sample ``num_envs`` SD1/SD2 instances plus optional dynamic events."""
        static_data = self._generate_static_batch(num_envs)
        dynamic_events = self._generate_dynamic_events()
        dynamic_jobs = self._generate_dynamic_jobs(
            dynamic_events.get('new_job_arrival', []))

        total_initial_ops = int(np.sum(static_data['job_length'][0]))
        total_dynamic_ops = int(np.sum(dynamic_jobs['job_length']))
        total_ops_with_arrivals = total_initial_ops + total_dynamic_ops

        return {
            'static': static_data,
            'dynamic_events': dynamic_events,
            'dynamic_jobs': dynamic_jobs,
            'max_total_jobs': self.n_j + int(self.params['num_new_job_arrivals']),
            'max_total_ops': total_ops_with_arrivals,
            'max_total_pt_nodes': total_ops_with_arrivals * self.n_m,
        }

    def _load_params(self) -> Dict[str, Any]:
        params = getattr(self.config, 'problem_params', None) or {}
        get = lambda key, default: params.get(
            key, getattr(self.config, key, default))
        get_any = lambda keys, default: next(
            (params[k] for k in keys if k in params),
            next((getattr(self.config, k) for k in keys
                  if hasattr(self.config, k)), default))
        def int_range_value(value) -> int:
            if hasattr(value, 'tolist'):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                if not value:
                    return 0
                if len(value) == 1:
                    return int(value[0])
                lo, hi = sorted(int(round(float(v))) for v in value[:2])
                if lo == hi:
                    return lo
                return int(np.random.randint(lo, hi + 1))
            return int(value)

        return {
            'num_new_job_arrivals': int_range_value(get_any((
                'num_new_job_arrival_waves',
                'num_new_job_arrivals_wave',
                'num_new_job_arrivals',
            ), 0)),
            'new_job_lambda_inv_min': float(get('new_job_lambda_inv_min', 25)),
            'new_job_lambda_inv_max': float(get('new_job_lambda_inv_max', 100)),
            'num_machine_breakdowns': int_range_value(get_any((
                'num_machine_breakdown_waves',
                'num_machine_breakdowns_wave',
                'num_machine_breakdowns',
            ), 0)),
            'mtbf_mean': float(get('mtbf_mean', 475)),
            'mttr_mean': float(get('mttr_mean', 25)),
            'num_priority_changes': int_range_value(get_any((
                'num_priority_change_waves',
                'num_priority_changes_wave',
                'num_priority_changes',
            ), 0)),
            'priority_change_mean': float(get('priority_change_mean', 25)),
            'due_date_tightness_min': float(
                get('due_date_tightness_min', 0.5)),
            'due_date_tightness_max': float(
                get('due_date_tightness_max', 1.5)),
        }

    def _generate_static_batch(self, num_envs: int) -> Dict[str, List[np.ndarray]]:
        template_job_length = self._sample_template_job_length(self.n_j)
        dataset_job_length: List[np.ndarray] = []
        dataset_op_pt: List[np.ndarray] = []
        dataset_job_weight: List[np.ndarray] = []
        dataset_job_due_date: List[np.ndarray] = []
        for _ in range(num_envs):
            jl, pt = self._sample_one_instance(template_job_length)
            dataset_job_length.append(jl)
            dataset_op_pt.append(pt.astype(np.float32))
            dataset_job_weight.append(
                np.ones(jl.shape[0], dtype=np.float32))
            dataset_job_due_date.append(
                self._calculate_synthetic_due_dates(jl, pt).astype(np.float32))

        return {
            'job_length': dataset_job_length,
            'op_pt': dataset_op_pt,
            'job_weight': dataset_job_weight,
            'job_due_date': dataset_job_due_date,
        }

    def _sample_template_job_length(self, n_jobs: int) -> np.ndarray:
        if self.generator_type == 'SD1':
            return np.random.randint(
                self.operation_per_job_min,
                self.operation_per_job_max + 1,
                size=n_jobs,
            ).astype(np.int64)
        # SD2: fixed n_m ops per job
        return np.full(n_jobs, self.n_m, dtype=np.int64)

    def _sample_one_instance(self, template_job_length: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        if self.generator_type == 'SD1':
            return self._sample_one_instance_sd1(template_job_length)
        return self._sample_one_instance_sd2(template_job_length)

    def _sample_one_instance_sd1(self, template_job_length: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """SD1: per-op mean U[1,20], then per-machine PT clamped around mean."""
        total_ops = int(np.sum(template_job_length))
        op_pt = np.zeros((total_ops, self.n_m), dtype=np.int64)
        pt_min = self.process_time_min
        pt_max = self.process_time_max
        mch_min = self.machine_per_operation_min
        mch_max = self.machine_per_operation_max
        for op_idx in range(total_ops):
            mch_num = int(np.random.randint(mch_min, mch_max + 1))
            compatible_machines = np.random.choice(
                self.n_m, mch_num, replace=False)
            mean_pt = int(np.random.randint(pt_min, pt_max + 1))
            low_bound = max(pt_min, int(round(mean_pt * 0.8)))
            high_bound = min(pt_max, int(round(mean_pt * 1.2)))
            pts = np.random.randint(low_bound, high_bound + 1, size=mch_num)
            op_pt[op_idx, compatible_machines] = pts
        return template_job_length.copy(), op_pt

    def _sample_one_instance_sd2(self, template_job_length: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """SD2 mix mode: PT U[1,99] then zero out incompatible machines."""
        total_ops = int(np.sum(template_job_length))
        pt_min = self.process_time_min
        pt_max = self.process_time_max
        mch_min = self.machine_per_operation_min
        mch_max = self.machine_per_operation_max
        op_pt = np.random.randint(
            pt_min, pt_max + 1, size=(total_ops, self.n_m)).astype(np.int64)
        for op_idx in range(total_ops):
            mch_num = int(np.random.randint(mch_min, mch_max + 1))
            if mch_num < self.n_m:
                inf_cols = np.random.choice(
                    self.n_m, self.n_m - mch_num, replace=False)
                op_pt[op_idx, inf_cols] = 0
        return template_job_length.copy(), op_pt

    def _calculate_synthetic_due_dates(
        self,
        job_length: np.ndarray,
        op_pt: np.ndarray,
        release_times=None,
    ) -> np.ndarray:
        """Finite compatibility due dates; ignored by makespan-only objective."""
        n_jobs = len(job_length)
        if release_times is None:
            release_times = np.zeros(n_jobs, dtype=np.float32)
        release_times = np.asarray(release_times, dtype=np.float32)

        work_content = np.zeros(n_jobs, dtype=np.float32)
        op_idx = 0
        for j, num_ops in enumerate(job_length):
            total = 0.0
            for _ in range(int(num_ops)):
                compatible = op_pt[op_idx][op_pt[op_idx] > 0]
                if compatible.size > 0:
                    total += float(np.mean(compatible))
                op_idx += 1
            work_content[j] = max(total, 1.0)

        k = np.random.uniform(
            self.params['due_date_tightness_min'],
            self.params['due_date_tightness_max'],
            size=n_jobs,
        ).astype(np.float32)
        return release_times + k * work_content

    def _generate_dynamic_events(self) -> Dict[str, List]:
        """Generate arrival/breakdown/priority event times using shared semantics."""
        num_arrivals = self.params['num_new_job_arrivals']
        lambda_inv_min = self.params['new_job_lambda_inv_min']
        lambda_inv_max = self.params['new_job_lambda_inv_max']

        arrival_times = []
        if num_arrivals > 0:
            lambda_inv = np.random.uniform(lambda_inv_min, lambda_inv_max)
            current_time = 0.0
            for _ in range(num_arrivals):
                current_time += np.random.exponential(lambda_inv)
                arrival_times.append(float(current_time))

        breakdown_events = []
        for _ in range(self.params['num_machine_breakdowns']):
            breakdown_events.append({
                'time': float(np.random.exponential(self.params['mtbf_mean'])),
                'machine_idx': int(np.random.randint(0, self.n_m)),
                'duration': float(np.random.exponential(self.params['mttr_mean'])),
            })
        breakdown_events.sort(key=lambda x: x['time'])

        priority_times = []
        current_time = 0.0
        for _ in range(self.params['num_priority_changes']):
            current_time += np.random.exponential(
                self.params['priority_change_mean'])
            priority_times.append(float(current_time))

        return {
            'new_job_arrival': arrival_times,
            'machine_breakdown': breakdown_events,
            'priority_change': priority_times,
        }

    def _generate_dynamic_jobs(self, arrival_times: List[float]) -> Dict[str, Any]:
        """Generate SD1/SD2 payload for jobs that arrive during inference."""
        num_arrivals = self.params['num_new_job_arrivals']
        if num_arrivals == 0:
            return {
                'job_length': np.array([], dtype=np.int64),
                'op_pt': np.array([], dtype=np.float32).reshape(0, self.n_m),
                'job_weight': np.array([], dtype=np.float32),
                'job_due_date': np.array([], dtype=np.float32),
                'ddt_min': self.params['due_date_tightness_min'],
                'ddt_max': self.params['due_date_tightness_max'],
            }

        template_job_length = self._sample_template_job_length(num_arrivals)
        job_length, op_pt = self._sample_one_instance(template_job_length)
        release_times = np.asarray(arrival_times[:num_arrivals], dtype=np.float32)
        job_due_date = self._calculate_synthetic_due_dates(
            job_length, op_pt, release_times=release_times)

        return {
            'job_length': job_length.astype(np.int64),
            'op_pt': op_pt.astype(np.float32),
            'job_weight': np.ones(num_arrivals, dtype=np.float32),
            'job_due_date': job_due_date.astype(np.float32),
            'ddt_min': self.params['due_date_tightness_min'],
            'ddt_max': self.params['due_date_tightness_max'],
        }


def create_data_generator_single_obj(config) -> DataGeneratorSingleObj:
    return DataGeneratorSingleObj(config)
