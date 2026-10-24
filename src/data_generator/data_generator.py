"""
Data Generator for Multi-Criteria FJSP.

Produces a unified instance shape: jobs with weights and due dates, plus
dynamic events (arrivals / breakdowns / priority changes). When all event
wave counts are zero the events are empty lists — there is no separate
"static" vs "dynamic" mode.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.data_generator.curriculum_leaner import CurriculumPacer
from src.utils.config_util import (
    normalize_generation_ranges,
    put_range_aliases,
    resolve_range,
)


class DataGenerator:
    """Data Generator for Multi-Criteria FJSP instances.

    All settings live in ``config.problem_params``. Dynamic event wave counts
    (``num_new_job_arrival_waves``, ``num_machine_breakdown_waves``,
    ``num_priority_change_waves``) default to 0, so a config without them
    yields a pure static problem. Legacy ``num_new_job_arrivals``,
    ``num_machine_breakdowns`` and ``num_priority_changes`` are accepted as
    wave-count aliases. Event generation is generic; any scenario labels must
    already have been translated into generic ranges by the caller.
    """

    _EVENT_PARAM_KEYS = (
        'num_new_job_arrivals',
        'num_machine_breakdowns',
        'num_priority_changes',
        'num_new_job_arrival_waves',
        'num_machine_breakdown_waves',
        'num_priority_change_waves',
        'new_job_lambda_inv',
        'mtbf',
        'priority_change_interval',
        'new_jobs_per_arrival',
        'machines_per_breakdown',
        'priority_jobs_per_change',
        'new_job_weight',
        'new_job_due_date_tightness',
        'priority_new_weight',
        'repair_duration',
    )

    _GENERATION_PARAM_KEYS = (
        'due_date_tightness',
        'urgency_weight',
        'operation_per_job',
        'machine_per_operation',
        'process_time',
        *_EVENT_PARAM_KEYS,
    )

    def __init__(self, config):
        self.config = config
        self.curriculum_pacer = None
        self.params = self._load_params()

    @staticmethod
    def _set_internal_range(params: Dict[str, Any], key: str,
                            lo: Any, hi: Any) -> None:
        params[key] = [lo, hi]
        aliases = {
            'due_date_tightness': ('due_date_tightness_min',
                                   'due_date_tightness_max'),
            'urgency_weight': ('urgency_weight_min', 'urgency_weight_max'),
            'operation_per_job': ('operation_per_job_min',
                                  'operation_per_job_max'),
            'machine_per_operation': ('machine_per_operation_min',
                                      'machine_per_operation_max'),
            'process_time': ('process_time_min', 'process_time_max'),
            'new_job_lambda_inv': ('new_job_lambda_inv_min',
                                   'new_job_lambda_inv_max'),
            'new_job_weight': ('new_job_weight_min', 'new_job_weight_max'),
            'new_job_due_date_tightness': (
                'new_job_due_date_tightness_min',
                'new_job_due_date_tightness_max'),
            'priority_new_weight': ('priority_new_weight_min',
                                    'priority_new_weight_max'),
            'repair_duration': ('repair_duration_min',
                                'repair_duration_max'),
            'mtbf': ('mtbf_min', 'mtbf_max'),
            'priority_change_interval': (
                'priority_change_interval_min',
                'priority_change_interval_max'),
            'new_jobs_per_arrival': ('new_jobs_per_arrival_min',
                                     'new_jobs_per_arrival_max'),
            'machines_per_breakdown': ('machines_per_breakdown_min',
                                       'machines_per_breakdown_max'),
            'priority_jobs_per_change': (
                'priority_jobs_per_change_min',
                'priority_jobs_per_change_max'),
            'num_new_job_arrivals': ('num_new_job_arrivals_min',
                                     'num_new_job_arrivals_max'),
            'num_machine_breakdowns': ('num_machine_breakdowns_min',
                                       'num_machine_breakdowns_max'),
            'num_priority_changes': ('num_priority_changes_min',
                                     'num_priority_changes_max'),
            'num_new_job_arrival_waves': (
                'num_new_job_arrival_waves_min',
                'num_new_job_arrival_waves_max'),
            'num_machine_breakdown_waves': (
                'num_machine_breakdown_waves_min',
                'num_machine_breakdown_waves_max'),
            'num_priority_change_waves': (
                'num_priority_change_waves_min',
                'num_priority_change_waves_max'),
        }
        if key in aliases:
            lo_key, hi_key = aliases[key]
            params[lo_key] = lo
            params[hi_key] = hi
        wave_to_legacy = {
            'num_new_job_arrival_waves': 'num_new_job_arrivals',
            'num_machine_breakdown_waves': 'num_machine_breakdowns',
            'num_priority_change_waves': 'num_priority_changes',
        }
        legacy_to_wave = {v: k for k, v in wave_to_legacy.items()}
        mirror_key = wave_to_legacy.get(key) or legacy_to_wave.get(key)
        if mirror_key is not None:
            params[mirror_key] = [lo, hi]
            if mirror_key in aliases:
                lo_key, hi_key = aliases[mirror_key]
                params[lo_key] = lo
                params[hi_key] = hi

    def _load_params(self) -> Dict[str, Any]:
        """Load parameters from config.problem_params (falls back to
        static_params / dynamic_params for legacy configs)."""
        params = getattr(self.config, 'problem_params', None)
        if params is None:
            # Legacy fallback chain
            params = (getattr(self.config, 'dynamic_params', None)
                      or getattr(self.config, 'static_params', {}))

        n_j = params.get('job_num', getattr(self.config, 'job_num', 20))
        n_m = params.get('machine_num', getattr(self.config, 'machine_num', 10))

        has_new_job_weight = any(
            k in params for k in (
                'new_job_weight', 'new_job_weight_min', 'new_job_weight_max',
                'rush_job_weight'))
        has_new_job_ddt = any(
            k in params for k in (
                'new_job_due_date_tightness',
                'new_job_due_date_tightness_min',
                'new_job_due_date_tightness_max',
                'rush_ddt_min', 'rush_ddt_max'))

        params = normalize_generation_ranges(params)

        # Dynamic-job defaults inherit the static job distributions unless
        # explicitly overridden.
        if not has_new_job_weight:
            lo, hi = resolve_range(params, 'urgency_weight', (1, 5), integer=True)
            put_range_aliases(params, 'new_job_weight', lo, hi)
        if not has_new_job_ddt:
            lo, hi = resolve_range(params, 'due_date_tightness', (0.5, 1.5))
            put_range_aliases(params, 'new_job_due_date_tightness', lo, hi)

        result = {
            'job_num': n_j,
            'machine_num': n_m,
        }
        for key in self._GENERATION_PARAM_KEYS:
            self._set_internal_range(result, key, params[key][0], params[key][1])

        return result

    def get_params(self) -> Dict[str, Any]:
        """Get the current generation parameters."""
        return self.params

    # ------------------------------------------------------------------ #
    # Curriculum Learning
    # ------------------------------------------------------------------ #
    def init_curriculum(self, curriculum_params: Dict[str, Any],
                        max_updates: int) -> None:
        """Build the CurriculumPacer.

        Pacer's targets are read from ``self.params`` — DDT bounds and the
        event complexity knobs — so phase 3 ramps to whatever was loaded
        into the data generator (training-side defaults come from
        ``TrainingConfig.curriculum_learning['dynamic_event_params']``).
        """
        def event_targets(source: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
            return {
                key: tuple(source[key])
                for key in self._EVENT_PARAM_KEYS
                if key in source
            }

        self.curriculum_pacer = CurriculumPacer(
            max_updates=max_updates,
            target_ddt_range=tuple(self.params['due_date_tightness']),
            initial_ddt_range=resolve_range(
                curriculum_params, 'initial_due_date_tightness',
                (1.0, 2.5)),
            ddt_stage_ratio=curriculum_params.get('ddt_stage_ratio', 0.60),
            ddt_min_substage_frac=curriculum_params.get(
                'ddt_min_substage_ratio', 0.5),
            target_event_ranges=event_targets(self.params),
            events_enabled=curriculum_params.get('events_enabled', True),
        )

    def update_curriculum(self, current_update: int) -> Tuple[float, float]:
        """Advance the curriculum and update generation DDT parameters.

        Returns:
            (ddt_min, ddt_max) being used for the current update.
        """
        if self.curriculum_pacer is None:
            return (self.params['due_date_tightness_min'],
                    self.params['due_date_tightness_max'])

        ddt_min, ddt_max = self.curriculum_pacer.get_ddt_range(current_update)
        self._set_internal_range(
            self.params, 'due_date_tightness', ddt_min, ddt_max)

        for key, (lo, hi) in self.curriculum_pacer.get_event_params(
                current_update).items():
            self._set_internal_range(self.params, key, lo, hi)

        return ddt_min, ddt_max

    def sample_env(self, num_envs: int) -> Dict[str, Any]:
        """Generate a batch of FJSP instances (unified shape).

        Returns a dict with the static problem, dynamic events, arriving-job
        specs, and max allocation sizes. When all event counts are zero the
        events are empty lists and dynamic_jobs is empty — this is what the
        training path sees.
        """
        static_data = self._generate_static_batch(num_envs, self.params)
        dynamic_events = self._generate_dynamic_events()
        dynamic_jobs = self._generate_dynamic_jobs(dynamic_events)

        num_arrivals = len(dynamic_events.get('new_job_arrival', []) or [])
        op_per_job_max = self.params['operation_per_job_max']
        job_num = self.params['job_num']
        machine_num = self.params['machine_num']

        max_total_jobs = job_num + num_arrivals
        max_total_ops = max_total_jobs * op_per_job_max
        max_total_pt_nodes = max_total_ops * machine_num

        return {
            'static': static_data,
            'dynamic_events': dynamic_events,
            'dynamic_jobs': dynamic_jobs,
            'max_total_jobs': max_total_jobs,
            'max_total_ops': max_total_ops,
            'max_total_pt_nodes': max_total_pt_nodes,
        }
    
    def _generate_static_batch(self, num_envs: int, params: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
        """
        Generate static FJSP data: jobs with weights and due dates.
        
        All parallel environments share the same job_length template to ensure
        uniform total operation count.
        
        Args:
            num_envs: Number of parallel environments
            params: Generation parameters
            
        Returns:
            Dict with 'job_length', 'op_pt', 'job_weight', 'job_due_date' lists
        """
        job_num = params['job_num']
        machine_num = params['machine_num']
        op_per_job_min = params['operation_per_job_min']
        op_per_job_max = params['operation_per_job_max']
        mch_per_op_min = params['machine_per_operation_min']
        mch_per_op_max = params['machine_per_operation_max']
        pt_min = params['process_time_min']
        pt_max = params['process_time_max']
        ddt_min = params.get('due_date_tightness_min', 0.5)
        ddt_max = params.get('due_date_tightness_max', 1.5)
        weight_min = params.get('urgency_weight_min', 1)
        weight_max = params.get('urgency_weight_max', 5)
        
        # Sample ONE job-length template for all envs (uniform operation count)
        template_job_length = self._sample_job_length(job_num, op_per_job_min, op_per_job_max)
        
        dataset_job_length = []
        dataset_op_pt = []
        dataset_job_weight = []
        dataset_job_due_date = []
        
        for _ in range(num_envs):
            job_length = template_job_length.copy()
            total_ops = np.sum(job_length)
            
            # Sample machine compatibility and process times
            compatibility = self._sample_compatibility(total_ops, mch_per_op_min, mch_per_op_max)
            op_pt = self._sample_processing_times(job_length, compatibility, machine_num, pt_min, pt_max)
            
            # Sample job weights: U_int[weight_min, weight_max]
            job_weight = np.random.randint(weight_min, weight_max + 1, size=job_num)
            
            # Calculate due dates: d_j = r_j + k * P_j
            # r_j = 0 for static jobs
            job_due_date = self._calculate_due_dates(job_length, op_pt, release_times=None,
                                                      ddt_min=ddt_min, ddt_max=ddt_max)
            
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
            dataset_job_weight.append(job_weight)
            dataset_job_due_date.append(job_due_date)
        
        return {
            'job_length': dataset_job_length,
            'op_pt': dataset_op_pt,
            'job_weight': dataset_job_weight,
            'job_due_date': dataset_job_due_date,
        }
    
    def _sample_job_length(self, job_num: int, op_min: int, op_max: int) -> np.ndarray:
        """Sample the number of operations for each job."""
        if op_min == op_max:
            return np.full(job_num, op_min, dtype=int)
        else:
            return np.random.randint(op_min, op_max + 1, size=job_num)
    
    def _sample_compatibility(self, total_ops: int, mch_min: int, mch_max: int) -> np.ndarray:
        """Sample the number of compatible machines for each operation."""
        if mch_min == mch_max:
            return np.full(total_ops, mch_min, dtype=int)
        else:
            return np.random.randint(mch_min, mch_max + 1, size=total_ops)
    
    def _sample_processing_times(self, job_length: np.ndarray, compatibility: np.ndarray,
                                 machine_num: int, pt_min: int, pt_max: int) -> np.ndarray:
        """Sample process times for operation-machine pairs."""
        total_ops = np.sum(job_length)
        op_pt = np.zeros((total_ops, machine_num), dtype=np.float32)
        
        for op_idx in range(total_ops):
            mch_num = int(compatibility[op_idx])
            compatible_machines = np.random.choice(machine_num, mch_num, replace=False)
            
            if pt_min == pt_max:
                op_pt[op_idx, compatible_machines] = pt_min
            else:
                op_pt[op_idx, compatible_machines] = np.random.randint(pt_min, pt_max + 1, size=mch_num)
        
        return op_pt
    
    def _calculate_due_dates(self, job_length: np.ndarray, op_pt: np.ndarray, 
                              release_times: Optional[np.ndarray] = None,
                              ddt_min: float = 0.5, ddt_max: float = 1.5) -> np.ndarray:
        """
        Calculate due dates using Total Work Content tightness formula.
        
        Formula: d_j = r_j + k * P_j
        """
        job_num = len(job_length)
        
        if release_times is None:
            release_times = np.zeros(job_num)
        
        # Calculate P_j for each job (average total process time)
        P_j = np.zeros(job_num)
        op_idx = 0
        for j in range(job_num):
            num_ops = job_length[j]
            job_total_pt = 0.0
            for _ in range(num_ops):
                op_times = op_pt[op_idx]
                compatible_times = op_times[op_times > 0]
                if len(compatible_times) > 0:
                    job_total_pt += np.mean(compatible_times)
                op_idx += 1
            P_j[j] = job_total_pt
        
        # Sample tightness factor k for each job
        k = np.random.uniform(ddt_min, ddt_max, size=job_num)
        
        # Calculate due dates
        due_dates = release_times + k * P_j
        
        return due_dates
    
    def _range(self, key: str,
               params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
        source = params or self.params
        lo, hi = source[key]
        return lo, hi

    def _sample_int_range(self, key: str, cap: Optional[int] = None,
                          params: Optional[Dict[str, Any]] = None) -> int:
        lo, hi = self._range(key, params=params)
        lo = max(0, int(round(float(lo))))
        hi = max(0, int(round(float(hi))))
        if cap is not None:
            cap = max(0, int(cap))
            lo = min(lo, cap)
            hi = min(hi, cap)
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            value = lo
        else:
            value = int(np.random.randint(lo, hi + 1))
        return value

    def _sample_wave_amount(self, key: str, cap: Optional[int] = None,
                            params: Optional[Dict[str, Any]] = None) -> int:
        """Sample the number of jobs/machines affected in one event wave."""
        if cap is not None and int(cap) <= 0:
            return 0
        value = self._sample_int_range(key, cap=cap, params=params)
        return max(1, int(value))

    def _sample_float_range(self, key: str,
                            params: Optional[Dict[str, Any]] = None) -> float:
        lo, hi = self._range(key, params=params)
        lo = float(lo)
        hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            return lo
        return float(np.random.uniform(lo, hi))

    def _sample_event_mean(self, key: str,
                           params: Optional[Dict[str, Any]] = None) -> float:
        return max(0.0, self._sample_float_range(key, params=params))

    def _next_event_time(self, current_time: float, mean_gap: float) -> float:
        if mean_gap <= 0:
            return float(current_time)
        return float(current_time + np.random.exponential(mean_gap))

    def _generate_dynamic_events(
        self,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List]:
        """Generate generic dynamic event schedules shared across envs."""
        params = params or self.params
        machine_num = int(params['machine_num'])
        job_num = int(params['job_num'])

        arrival_times = []
        arrival_waves = self._sample_int_range(
            'num_new_job_arrival_waves', params=params)
        if arrival_waves > 0:
            mean_gap = self._sample_event_mean(
                'new_job_lambda_inv', params=params)
            current_time = 0.0
            for _ in range(arrival_waves):
                current_time = self._next_event_time(current_time, mean_gap)
                batch_size = self._sample_wave_amount(
                    'new_jobs_per_arrival', params=params)
                arrival_times.extend([float(current_time)] * batch_size)

        breakdown_waves = self._sample_int_range(
            'num_machine_breakdown_waves', params=params)
        breakdown_events = []
        if breakdown_waves > 0 and machine_num > 0:
            mean_gap = self._sample_event_mean('mtbf', params=params)
            current_time = 0.0
            for _ in range(breakdown_waves):
                current_time = self._next_event_time(current_time, mean_gap)
                machine_count = self._sample_wave_amount(
                    'machines_per_breakdown', cap=machine_num, params=params)
                machines = np.random.choice(
                    machine_num, size=machine_count, replace=False)
                for machine_idx in machines:
                    breakdown_events.append({
                        'time': float(current_time),
                        'machine_idx': int(machine_idx),
                        'duration': self._sample_float_range(
                            'repair_duration', params=params),
                    })
        breakdown_events.sort(key=lambda x: x['time'])

        priority_events = []
        priority_waves = self._sample_int_range(
            'num_priority_change_waves', params=params)
        if priority_waves > 0:
            mean_gap = self._sample_event_mean(
                'priority_change_interval', params=params)
            current_time = 0.0
            for _ in range(priority_waves):
                current_time = self._next_event_time(current_time, mean_gap)
                job_count = self._sample_int_range(
                    'priority_jobs_per_change', cap=job_num, params=params)
                if job_count > 0 and job_num > 0:
                    job_indices = np.random.choice(
                        job_num, size=job_count, replace=False)
                    job_indices = sorted(int(j) for j in job_indices.tolist())
                else:
                    job_indices = []
                priority_events.append({
                    'time': float(current_time),
                    'job_indices': job_indices,
                    'new_weight': self._sample_float_range(
                        'priority_new_weight', params=params),
                    'mode': 'set',
                })

        return {
            'new_job_arrival': arrival_times,
            'machine_breakdown': breakdown_events,
            'priority_change': priority_events,
        }

    def _generate_dynamic_jobs(
        self,
        dynamic_events: Optional[Dict[str, List]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate data for jobs that will arrive dynamically."""
        params = params or self.params
        if dynamic_events is None:
            num_arrivals = sum(
                self._sample_wave_amount('new_jobs_per_arrival', params=params)
                for _ in range(
                    self._sample_int_range(
                        'num_new_job_arrival_waves', params=params)))
            arrival_times = None
        else:
            arrival_times = dynamic_events.get('new_job_arrival', []) or []
            num_arrivals = len(arrival_times)

        if num_arrivals == 0:
            return {
                'job_length': np.array([], dtype=int),
                'op_pt': np.array([]).reshape(0, params['machine_num']),
                'job_weight': np.array([], dtype=int),
                'job_due_date': np.array([], dtype=np.float32),
            }
        
        machine_num = params['machine_num']
        op_per_job_min = params['operation_per_job_min']
        op_per_job_max = params['operation_per_job_max']
        mch_per_op_min = params['machine_per_operation_min']
        mch_per_op_max = params['machine_per_operation_max']
        pt_min = params['process_time_min']
        pt_max = params['process_time_max']
        weight_min = int(params.get('new_job_weight_min', 1))
        weight_max = int(params.get('new_job_weight_max', 5))
        ddt_min = float(params.get('new_job_due_date_tightness_min', 0.5))
        ddt_max = float(params.get('new_job_due_date_tightness_max', 1.5))
        
        template_job_length = self._sample_job_length(num_arrivals, op_per_job_min, op_per_job_max)
        total_ops = np.sum(template_job_length)
        
        compatibility = self._sample_compatibility(total_ops, mch_per_op_min, mch_per_op_max)
        op_pt = self._sample_processing_times(template_job_length, compatibility, machine_num, pt_min, pt_max)
        
        job_weight = np.random.randint(weight_min, weight_max + 1, size=num_arrivals)
        release_times = np.asarray(
            arrival_times if arrival_times is not None else np.zeros(num_arrivals),
            dtype=np.float32,
        )
        job_due_date = self._calculate_due_dates(
            template_job_length, op_pt, release_times=release_times,
            ddt_min=ddt_min, ddt_max=ddt_max)

        result = {
            'job_length': template_job_length,
            'op_pt': op_pt,
            'job_weight': job_weight,
            'job_due_date': job_due_date.astype(np.float32),
            'ddt_min': ddt_min,
            'ddt_max': ddt_max,
        }
        return result


def create_data_generator(config) -> DataGenerator:
    """
    Factory function to create a data generator.
    
    Args:
        config: Configuration object
    
    Returns:
        DataGenerator instance
    """
    return DataGenerator(config)
