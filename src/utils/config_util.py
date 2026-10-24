"""
Config helper / builder functions.

Creates argparse.Namespace objects consumed by HGTTrainer and inference
scripts, plus small utilities for wandb setup and result directory creation.
"""

import math
import os
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

from config import training_config, training_config_single_obj, experiment_config
from src.utils.reproducibility import reproducibility_metadata
from src.utils.training_settings import require_training_controls


_RANGE_ALIASES = {
    'due_date_tightness': {
        'min': 'due_date_tightness_min',
        'max': 'due_date_tightness_max',
    },
    'initial_due_date_tightness': {
        'min': 'initial_ddt_min',
        'max': 'initial_ddt_max',
    },
    'urgency_weight': {
        'min': 'urgency_weight_min',
        'max': 'urgency_weight_max',
    },
    'operation_per_job': {
        'min': 'operation_per_job_min',
        'max': 'operation_per_job_max',
    },
    'machine_per_operation': {
        'min': 'machine_per_operation_min',
        'max': 'machine_per_operation_max',
    },
    'process_time': {
        'min': 'process_time_min',
        'max': 'process_time_max',
    },
    'new_job_lambda_inv': {
        'min': 'new_job_lambda_inv_min',
        'max': 'new_job_lambda_inv_max',
    },
    'num_new_job_arrivals': {
        'range': (
            'num_new_job_arrival_waves',
            'num_new_job_arrival_wave',
            'num_new_job_arrivals_waves',
            'num_new_job_arrivals_wave',
            'rush_waves',
        ),
        'min': 'num_new_job_arrivals_min',
        'max': 'num_new_job_arrivals_max',
    },
    'num_machine_breakdowns': {
        'range': (
            'num_machine_breakdown_waves',
            'num_machine_breakdown_wave',
            'num_machine_breakdowns_waves',
            'num_machine_breakdowns_wave',
            'earthquake_waves',
        ),
        'min': 'num_machine_breakdowns_min',
        'max': 'num_machine_breakdowns_max',
    },
    'num_priority_changes': {
        'range': (
            'num_priority_change_waves',
            'num_priority_change_wave',
            'num_priority_changes_waves',
            'num_priority_changes_wave',
            'priority_shocks',
        ),
        'min': 'num_priority_changes_min',
        'max': 'num_priority_changes_max',
    },
    'num_new_job_arrival_waves': {
        'range': (
            'num_new_job_arrivals',
            'num_new_job_arrival_wave',
            'num_new_job_arrivals_waves',
            'num_new_job_arrivals_wave',
            'rush_waves',
        ),
        'min': ('num_new_job_arrival_waves_min',
                'num_new_job_arrival_wave_min',
                'num_new_job_arrivals_waves_min',
                'num_new_job_arrivals_wave_min',
                'num_new_job_arrivals_min'),
        'max': ('num_new_job_arrival_waves_max',
                'num_new_job_arrival_wave_max',
                'num_new_job_arrivals_waves_max',
                'num_new_job_arrivals_wave_max',
                'num_new_job_arrivals_max'),
    },
    'num_machine_breakdown_waves': {
        'range': (
            'num_machine_breakdowns',
            'num_machine_breakdown_wave',
            'num_machine_breakdowns_waves',
            'num_machine_breakdowns_wave',
            'earthquake_waves',
        ),
        'min': ('num_machine_breakdown_waves_min',
                'num_machine_breakdown_wave_min',
                'num_machine_breakdowns_waves_min',
                'num_machine_breakdowns_wave_min',
                'num_machine_breakdowns_min'),
        'max': ('num_machine_breakdown_waves_max',
                'num_machine_breakdown_wave_max',
                'num_machine_breakdowns_waves_max',
                'num_machine_breakdowns_wave_max',
                'num_machine_breakdowns_max'),
    },
    'num_priority_change_waves': {
        'range': (
            'num_priority_changes',
            'num_priority_change_wave',
            'num_priority_changes_waves',
            'num_priority_changes_wave',
            'priority_shocks',
        ),
        'min': ('num_priority_change_waves_min',
                'num_priority_change_wave_min',
                'num_priority_changes_waves_min',
                'num_priority_changes_wave_min',
                'num_priority_changes_min'),
        'max': ('num_priority_change_waves_max',
                'num_priority_change_wave_max',
                'num_priority_changes_waves_max',
                'num_priority_changes_wave_max',
                'num_priority_changes_max'),
    },
    'mtbf': {
        'min': 'mtbf_min',
        'max': 'mtbf_max',
        'fixed': ('mtbf_mean',),
    },
    'priority_change_interval': {
        'min': 'priority_change_interval_min',
        'max': 'priority_change_interval_max',
        'fixed': ('priority_change_mean',),
    },
    'repair_duration': {
        'min': 'repair_duration_min',
        'max': 'repair_duration_max',
        'fixed': ('mttr_mean', 'quake_duration_mean'),
    },
    'new_jobs_per_arrival': {
        'min': 'new_jobs_per_arrival_min',
        'max': 'new_jobs_per_arrival_max',
        'range': ('rush_batch_sizes',),
    },
    'machines_per_breakdown': {
        'range': ('machines_per_quake',),
        'min': ('machines_per_breakdown_min', 'machine_breakdown_min'),
        'max': ('machines_per_breakdown_max', 'machine_breakdown_max'),
    },
    'priority_jobs_per_change': {
        'range': ('priority_jobs_per_shock',),
        'min': ('priority_jobs_per_change_min', 'priority_change_min'),
        'max': ('priority_jobs_per_change_max', 'priority_change_max'),
    },
    'new_job_weight': {
        'min': 'new_job_weight_min',
        'max': 'new_job_weight_max',
        'fixed': ('rush_job_weight',),
    },
    'new_job_due_date_tightness': {
        'min': ('new_job_due_date_tightness_min', 'rush_ddt_min'),
        'max': ('new_job_due_date_tightness_max', 'rush_ddt_max'),
    },
    'priority_new_weight': {
        'min': 'priority_new_weight_min',
        'max': 'priority_new_weight_max',
        'fixed': ('priority_shock_new_weight',),
    },
}

_INT_RANGE_KEYS = {
    'urgency_weight',
    'operation_per_job',
    'machine_per_operation',
    'process_time',
    'num_new_job_arrivals',
    'num_machine_breakdowns',
    'num_priority_changes',
    'num_new_job_arrival_waves',
    'num_machine_breakdown_waves',
    'num_priority_change_waves',
    'new_jobs_per_arrival',
    'machines_per_breakdown',
    'priority_jobs_per_change',
    'new_job_weight',
    'priority_new_weight',
}

_PROBLEM_RANGE_DEFAULTS = {
    'due_date_tightness': (0.5, 1.5),
    'urgency_weight': (1, 5),
    'operation_per_job': (1, 10),
    'machine_per_operation': (1, 10),
    'process_time': (1, 50),
}

_EVENT_RANGE_DEFAULTS = {
    'num_new_job_arrivals': (0, 0),
    'num_machine_breakdowns': (0, 0),
    'num_priority_changes': (0, 0),
    'num_new_job_arrival_waves': (0, 0),
    'num_machine_breakdown_waves': (0, 0),
    'num_priority_change_waves': (0, 0),
    'new_job_lambda_inv': (25, 100),
    'mtbf': (475, 475),
    'priority_change_interval': (25, 25),
    'new_jobs_per_arrival': (1, 1),
    'machines_per_breakdown': (1, 1),
    'priority_jobs_per_change': (1, 1),
    'new_job_weight': (1, 5),
    'new_job_due_date_tightness': (0.5, 1.5),
    'priority_new_weight': (5, 5),
    'repair_duration': (25, 25),
}

_MIN_MAX_COMPAT_KEYS = {
    'due_date_tightness',
    'urgency_weight',
    'operation_per_job',
    'machine_per_operation',
    'process_time',
    'new_job_lambda_inv',
    'num_new_job_arrivals',
    'num_machine_breakdowns',
    'num_priority_changes',
    'num_new_job_arrival_waves',
    'num_machine_breakdown_waves',
    'num_priority_change_waves',
    'mtbf',
    'priority_change_interval',
    'new_jobs_per_arrival',
    'machines_per_breakdown',
    'priority_jobs_per_change',
    'new_job_weight',
    'new_job_due_date_tightness',
    'priority_new_weight',
    'repair_duration',
}

_EVENT_WAVE_TO_LEGACY_KEYS = {
    'num_new_job_arrival_waves': 'num_new_job_arrivals',
    'num_machine_breakdown_waves': 'num_machine_breakdowns',
    'num_priority_change_waves': 'num_priority_changes',
}


def _as_list(value: Any) -> List[Any]:
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(';', ',')
        if not cleaned:
            return []
        if '-' in cleaned and ',' not in cleaned:
            return [p.strip() for p in cleaned.split('-') if p.strip()]
        return [p.strip() for p in cleaned.split(',') if p.strip()]
    return [value]


def _alias_keys(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(v for v in value if v)
    return (value,)


def _parse_range_value(value: Any) -> Tuple[Any, Any]:
    values = _as_list(value)
    if len(values) == 0:
        raise ValueError("Cannot normalize an empty range value")
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    numeric = [float(v) for v in values]
    return min(numeric), max(numeric)


def _coerce_pair(lo: Any, hi: Any, integer: bool) -> Tuple[Any, Any]:
    if integer:
        lo = int(round(float(lo)))
        hi = int(round(float(hi)))
    else:
        lo = float(lo)
        hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def resolve_range(params: Dict[str, Any], key: str, default: Any,
                  integer: Optional[bool] = None) -> Tuple[Any, Any]:
    """Resolve a canonical ``[min, max]`` range with legacy fallbacks.

    Accepted forms:
    - ``key: [lo, hi]`` or ``key: scalar`` (fixed range)
    - legacy ``key_min`` / ``key_max`` aliases
    - legacy fixed means such as ``mtbf_mean`` / ``mttr_mean``
    - legacy list profiles such as ``rush_batch_sizes``; these normalize to
      their min/max envelope.
    """
    if params is None:
        params = {}
    integer = (key in _INT_RANGE_KEYS) if integer is None else bool(integer)

    if key in params:
        lo, hi = _parse_range_value(params[key])
        return _coerce_pair(lo, hi, integer)

    aliases = _RANGE_ALIASES.get(key, {})
    for alias in aliases.get('range', ()):
        if alias in params:
            lo, hi = _parse_range_value(params[alias])
            return _coerce_pair(lo, hi, integer)

    min_keys = _alias_keys(aliases.get('min'))
    max_keys = _alias_keys(aliases.get('max'))
    min_key = next((k for k in min_keys if k in params), None)
    max_key = next((k for k in max_keys if k in params), None)
    if min_key is not None or max_key is not None:
        if min_key is not None:
            lo = params[min_key]
        elif max_key is not None:
            lo = params[max_key]
        else:
            lo = _parse_range_value(default)[0]
        if max_key is not None:
            hi = params[max_key]
        elif min_key is not None:
            hi = params[min_key]
        else:
            hi = _parse_range_value(default)[1]
        return _coerce_pair(lo, hi, integer)

    for alias in aliases.get('fixed', ()):
        if alias in params:
            return _coerce_pair(params[alias], params[alias], integer)

    lo, hi = _parse_range_value(default)
    return _coerce_pair(lo, hi, integer)


def _range_list(lo: Any, hi: Any) -> List[Any]:
    return [lo, hi]


def put_range_aliases(params: Dict[str, Any], key: str, lo: Any,
                      hi: Any) -> None:
    """Write a canonical range and legacy min/max aliases when applicable."""
    params[key] = _range_list(lo, hi)
    aliases = _RANGE_ALIASES.get(key, {})
    min_keys = _alias_keys(aliases.get('min'))
    max_keys = _alias_keys(aliases.get('max'))
    if key in _MIN_MAX_COMPAT_KEYS and min_keys and max_keys:
        min_key = min_keys[0]
        max_key = max_keys[0]
        params[min_key] = lo
        params[max_key] = hi


def normalize_generation_ranges(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``params`` with canonical generation ranges and compat aliases."""
    normalized = dict(params or {})
    all_defaults = {**_PROBLEM_RANGE_DEFAULTS, **_EVENT_RANGE_DEFAULTS}
    for key, default in all_defaults.items():
        lo, hi = resolve_range(
            normalized, key, default, integer=(key in _INT_RANGE_KEYS))
        put_range_aliases(normalized, key, lo, hi)
    for wave_key, legacy_key in _EVENT_WAVE_TO_LEGACY_KEYS.items():
        lo, hi = resolve_range(
            normalized, wave_key, _EVENT_RANGE_DEFAULTS[wave_key],
            integer=True)
        put_range_aliases(normalized, wave_key, lo, hi)
        put_range_aliases(normalized, legacy_key, lo, hi)
    return normalized


def _has_range_source(params: Dict[str, Any], key: str) -> bool:
    if key in params:
        return True
    aliases = _RANGE_ALIASES.get(key, {})
    for alias_key in (
        *_alias_keys(aliases.get('min')),
        *_alias_keys(aliases.get('max')),
        *(aliases.get('fixed', ()) or ()),
        *(aliases.get('range', ()) or ()),
    ):
        if alias_key and alias_key in params:
            return True
    return False


def translate_event_profile_to_generic(params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate legacy scenario knobs into generic dynamic-event ranges."""
    source = dict(params or {})
    if ('priority_shocks' in source
            and not _has_range_source(source, 'num_priority_change_waves')):
        source['num_priority_change_waves'] = source['priority_shocks']
    if ('rush_batch_sizes' in source
            and not _has_range_source(source, 'num_new_job_arrival_waves')):
        source['num_new_job_arrival_waves'] = len(
            _as_list(source['rush_batch_sizes']))
    if ('earthquake_waves' in source
            and not _has_range_source(source, 'num_machine_breakdown_waves')):
        source['num_machine_breakdown_waves'] = source['earthquake_waves']

    out: Dict[str, Any] = {}
    if 'scenario' in source:
        out['scenario'] = source['scenario']
    for key, default in _EVENT_RANGE_DEFAULTS.items():
        lo, hi = resolve_range(source, key, default,
                               integer=(key in _INT_RANGE_KEYS))
        out[key] = _range_list(lo, hi)
    return out


_DYNAMIC_EVENT_KEYS = {
    'scenario',
    'num_new_job_arrivals',
    'num_machine_breakdowns',
    'num_priority_changes',
    'num_new_job_arrival_waves',
    'num_new_job_arrival_wave',
    'num_new_job_arrivals_waves',
    'num_new_job_arrivals_wave',
    'num_machine_breakdown_waves',
    'num_machine_breakdown_wave',
    'num_machine_breakdowns_waves',
    'num_machine_breakdowns_wave',
    'num_priority_change_waves',
    'num_priority_change_wave',
    'num_priority_changes_waves',
    'num_priority_changes_wave',
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
    # Legacy keys accepted as input.
    'new_job_lambda_inv_min',
    'new_job_lambda_inv_max',
    'mtbf_mean',
    'mttr_mean',
    'priority_change_mean',
    'occurred_times_min',
    'occurred_times_max',
    'machine_breakdown_min',
    'machine_breakdown_max',
    'priority_change_min',
    'priority_change_max',
    'priority_shock_new_weight',
    'rush_waves',
    'rush_batch_sizes',
    'rush_job_weight',
    'rush_ddt_min',
    'rush_ddt_max',
    'priority_shocks',
    'priority_jobs_per_shock',
    'earthquake_waves',
    'machines_per_quake',
    'quake_duration_mean',
}


def _event_keys_only(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k in _DYNAMIC_EVENT_KEYS}


def _resolve_dynamic_event_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve legacy scenario profiles into generic generator keys."""
    if not isinstance(params, dict):
        return {}

    event_params = dict(params)
    scenario = event_params.get('scenario')
    scenarios = event_params.get('scenarios')
    if scenario and isinstance(scenarios, dict):
        base = {k: v for k, v in event_params.items() if k != 'scenarios'}
        scenario_defaults = scenarios.get(scenario)
        if isinstance(scenario_defaults, dict):
            profile = {**base, **scenario_defaults}
            profile['scenario'] = scenario
            return _event_keys_only(translate_event_profile_to_generic(profile))

    event_params.pop('scenarios', None)
    return _event_keys_only(translate_event_profile_to_generic(event_params))


#: The problem size the event magnitudes in ``config.py`` were tuned at. Wave
#: amounts there (3-5 machines, 3-4 new jobs) are a 30-50% capacity loss and a
#: 15-20% demand shock AT THIS SIZE, and only at this size.
EVENT_MAGNITUDE_REFERENCE_SIZE = (20, 10)

#: Wave amounts that describe "how much of the shop is hit", and the dimension
#: each is a fraction of.
_EVENT_MAGNITUDE_KEYS = {
    'machines_per_breakdown': 'machine',
    'new_jobs_per_arrival': 'job',
    'priority_jobs_per_change': 'job',
}


def scale_event_magnitudes(
    event_params: Dict[str, Any],
    n_j: int,
    n_m: int,
    reference: Tuple[int, int] = EVENT_MAGNITUDE_REFERENCE_SIZE,
) -> Dict[str, Any]:
    """Rescale dynamic-event wave AMOUNTS so a shock is the same FRACTION of
    the shop at every problem size.

    The generator draws wave amounts as absolute counts, capped at the shop
    size, so a single parameter set means very different things at different
    scales. ``machines_per_breakdown=[3, 5]`` is 30-50% of a 10-machine shop —
    and 60-100% of a 5-machine one, i.e. a total shutdown. ``new_jobs_per_
    arrival=[3, 4]`` is a 15-20% demand shock on 20 jobs and 3-4% on 100.

    For the scale generalization experiment that is a confound on the measured
    axis: a model trained at 10x5 would learn to recover from near-total
    shutdown while one trained at 100x20 learns to absorb a scratch, and the
    "scale mismatch" curve would be measuring both at once.

    Scaled here: amounts only. NOT scaled: ``mtbf``, ``new_job_lambda_inv``,
    ``repair_duration``, wave counts. Inter-arrival times and repair durations
    are properties of the machines, not of the order book — a bigger shop does
    not break down more often per machine, it just has more machines to break.
    That is exactly the part the proportional amounts model.

    Returns a new dict; the input is not modified.
    """
    ref_j, ref_m = int(reference[0]), int(reference[1])
    scaled = dict(event_params or {})
    factors = {'job': n_j / float(ref_j), 'machine': n_m / float(ref_m)}
    caps = {'job': int(n_j), 'machine': int(n_m)}

    for key, dimension in _EVENT_MAGNITUDE_KEYS.items():
        lo, hi = resolve_range(
            scaled, key, _EVENT_RANGE_DEFAULTS[key], integer=True)
        factor, cap = factors[dimension], caps[dimension]
        # floor(x + 0.5): round-half-up. Python's round() is banker's rounding,
        # which would send 2.5 -> 2 and silently shrink the small end.
        new_lo = min(cap, max(1, int(math.floor(lo * factor + 0.5))))
        new_hi = min(cap, max(new_lo, int(math.floor(hi * factor + 0.5))))
        put_range_aliases(scaled, key, new_lo, new_hi)

    return scaled


def _curriculum_dynamic_event_params(config_obj) -> Dict[str, Any]:
    curriculum = getattr(config_obj, 'curriculum_learning', {}) or {}
    if isinstance(curriculum, dict):
        nested = curriculum.get('dynamic_event_params')
        if isinstance(nested, dict):
            return _resolve_dynamic_event_params(nested)

    # Legacy compatibility for older config objects / checkpoints.
    legacy = getattr(config_obj, 'dynamic_event_params', {}) or {}
    return _resolve_dynamic_event_params(legacy)


def _merge_curriculum_config(base: Dict[str, Any],
                             override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    if not isinstance(override, dict):
        return merged

    event_override = override.get('dynamic_event_params')
    for key, value in override.items():
        if key != 'dynamic_event_params':
            merged[key] = value

    if isinstance(event_override, dict):
        base_events = merged.get('dynamic_event_params', {})
        if isinstance(base_events, dict):
            merged['dynamic_event_params'] = {**base_events, **event_override}
        else:
            merged['dynamic_event_params'] = dict(event_override)

    return merged


def _sync_training_event_params(hgt_config) -> None:
    dynamic_event_params = _curriculum_dynamic_event_params(hgt_config)
    # Opt-in, and applied to the RESOLVED base every time rather than to the
    # previous result, so re-syncing after overrides cannot compound the
    # factor. The unscaled base stays in curriculum_learning['dynamic_event_
    # params']; the scaled values go to the keys the generator actually reads.
    if getattr(hgt_config, 'scale_event_magnitudes_to_size', False):
        dynamic_event_params = scale_event_magnitudes(
            dynamic_event_params,
            int(getattr(hgt_config, 'n_j')),
            int(getattr(hgt_config, 'n_m')))
    hgt_config.dynamic_event_params = dynamic_event_params.copy()
    hgt_config.problem_params = {
        **hgt_config.static_params,
        **dynamic_event_params,
    }
    curriculum = dict(getattr(hgt_config, 'curriculum_learning', {}) or {})
    curriculum.setdefault('dynamic_event_params', dynamic_event_params.copy())
    curriculum['resolved_dynamic_event_params'] = dynamic_event_params.copy()
    hgt_config.curriculum_learning = curriculum


def retarget_config_size(hgt_config, n_j: int, n_m: int) -> None:
    """Repoint a built training config at a new problem size, in place.

    This is the config half of a scale-curriculum stage switch, kept here and
    not in the trainer for one reason: it must go through
    ``_sync_training_event_params``, which is the only place
    ``scale_event_magnitudes_to_size`` is ever applied. The trainer used to
    rebuild ``problem_params`` by hand instead, which silently carried the
    FINAL stage's absolute event amounts down the entire ladder - at 10x5 that
    meant every breakdown wave removing 6-10 machines from a 5-machine shop.

    Safe to call repeatedly: the event amounts are recomputed from the
    unscaled base in ``curriculum_learning['dynamic_event_params']``, never
    from the previous result, so the factor cannot compound.
    """
    hgt_config.n_j, hgt_config.n_m = int(n_j), int(n_m)
    sp = dict(getattr(hgt_config, 'static_params', {}) or {})
    sp['job_num'], sp['machine_num'] = int(n_j), int(n_m)
    # Mirror create_hgt_config's per-size clamps: ops/job and compatible
    # machines/op are capped at n_m, because sampling k machines without
    # replacement from n_m < k crashes the generator.
    op_min, _ = resolve_range(sp, 'operation_per_job', (1, n_m), integer=True)
    mch_min, _ = resolve_range(
        sp, 'machine_per_operation', (1, n_m), integer=True)
    put_range_aliases(sp, 'operation_per_job', min(op_min, n_m), n_m)
    put_range_aliases(sp, 'machine_per_operation', min(mch_min, n_m), n_m)
    hgt_config.static_params = sp

    _sync_training_event_params(hgt_config)

    # Keep the flat generator aliases in step with the rebuilt dict.
    pp = hgt_config.problem_params
    hgt_config.operation_per_job_min = pp['operation_per_job_min']
    hgt_config.operation_per_job_max = pp['operation_per_job_max']
    hgt_config.machine_per_operation_min = pp['machine_per_operation_min']
    hgt_config.machine_per_operation_max = pp['machine_per_operation_max']


def _reward_diff_config(config_obj):
    """Return the canonical DP lower-bound reward config."""
    params = getattr(config_obj, 'dp_lb_diff', {'enabled': True})
    return params.copy() if isinstance(params, dict) else params


def resolve_dp_lb_diff_enabled(config_obj, default=True) -> bool:
    """Resolve DP lower-bound reward enabled state from config or checkpoint metadata."""
    if isinstance(config_obj, dict):
        params = config_obj.get('dp_lb_diff')
    else:
        params = getattr(config_obj, 'dp_lb_diff', None)
    if isinstance(params, dict) and 'enabled' in params:
        return bool(params['enabled'])
    return bool(default)


def resolve_dp_lb_measure(config_obj, default='dp') -> str:
    """Resolve WHICH admissible bound the reward differences.

    ``'dp'`` is the bound that consults machine free times; ``'chain'`` is the
    job-chain lower bound prior lower-bound rewards use. Only meaningful when
    ``dp_lb_diff.enabled`` is True. Absent from every checkpoint written before
    2026-08-08, where the default reproduces the old behaviour exactly.
    """
    if isinstance(config_obj, dict):
        params = config_obj.get('dp_lb_diff')
    else:
        params = getattr(config_obj, 'dp_lb_diff', None)
    if isinstance(params, dict) and params.get('measure') is not None:
        measure = str(params['measure'])
        if measure not in ('dp', 'chain'):
            raise ValueError(
                f"dp_lb_diff.measure must be 'dp' or 'chain', got {measure!r}")
        return measure
    return default


def parse_size(s: str) -> Tuple[int, int]:
    """Parse a size string like '10x5' (or '10X5') into (n_j, n_m)."""
    a, b = s.lower().split('x')
    return int(a), int(b)


def setup_wandb_env(result_dir: str, project_name: str) -> None:
    os.environ["WANDB_DIR"] = result_dir
    os.environ["WANDB_PROJECT"] = project_name
    if os.environ.get("WANDB_MODE") is None:
        os.environ["WANDB_MODE"] = "online"


def create_experiment_result_dir(experiment_name: str) -> str:
    """Resolve the run path; create it only after validating its controls."""
    base_result_dir = "result"
    experiment_dir = os.path.join(base_result_dir, experiment_name)
    timestamp = time.strftime("%Y%m%d_%H")
    timestamp_dir = os.path.join(experiment_dir, timestamp)
    return timestamp_dir


def create_hgt_config(experiment_name: str = "exp_hgt", **overrides) -> argparse.Namespace:
    """Create training config as argparse.Namespace for HGTTrainer."""
    tc = training_config
    overrides = dict(overrides)
    create_directory = overrides.pop('create_result_dir', True)
    removed_return_settings = sorted(
        {'normalize_returns', 'return_norm_ref_area', 'return_norm_mode'} & overrides.keys())
    if removed_return_settings:
        raise ValueError(
            f"Removed return normalization settings: {', '.join(removed_return_settings)}. "
            "Use use_popart=True for adaptive value normalization or False for raw value targets.")
    if 'lowerbound_differencing' in overrides:
        raise ValueError(
            "lowerbound_differencing was removed; use dp_lb_diff={'enabled': ...}.")

    size_override = overrides.pop('size', None)
    n_j = tc.problem_params['job_num']
    n_m = tc.problem_params['machine_num']
    size_was_overridden = size_override is not None
    if size_override is not None:
        n_j, n_m = parse_size(str(size_override))

    for attr_name, param_name in (
        ('job_num', 'n_j'),
        ('machine_num', 'n_m'),
    ):
        value = overrides.pop(attr_name, None)
        if value is not None:
            if param_name == 'n_j':
                n_j = int(value)
            else:
                n_m = int(value)
            size_was_overridden = True

    value = overrides.pop('n_j', None)
    if value is not None:
        n_j = int(value)
        size_was_overridden = True
    value = overrides.pop('n_m', None)
    if value is not None:
        n_m = int(value)
        size_was_overridden = True

    if n_j <= 0 or n_m <= 0:
        raise ValueError(f"Training size must be positive, got {n_j}x{n_m}")

    static_params = normalize_generation_ranges(tc.problem_params.copy())
    static_params['job_num'] = n_j
    static_params['machine_num'] = n_m
    if size_was_overridden:
        # Match inference-size semantics and keep 10x5 valid.
        op_min, _ = resolve_range(
            static_params, 'operation_per_job', (1, n_m), integer=True)
        mch_min, _ = resolve_range(
            static_params, 'machine_per_operation', (1, n_m), integer=True)
        put_range_aliases(
            static_params, 'operation_per_job', min(op_min, n_m), n_m)
        put_range_aliases(
            static_params, 'machine_per_operation', min(mch_min, n_m), n_m)

    # On resume the caller passes result_dir=<original run dir>; reuse it instead
    # of minting a fresh timestamped dir, so the rejoined W&B run's local files
    # and all checkpoints stay co-located and no empty dir is left behind.
    result_dir = overrides.get('result_dir') or (
        create_experiment_result_dir(experiment_name) if create_directory
        else os.path.join('result', experiment_name))

    # Checkpoint-name suffix reflects the model design, NOT the problem variant:
    #   _dynamic - the new design (dynamic-event node features on; trained by
    #              train_graph_rl.py and ablation_training_stability.py).
    #   _static  - the old design (use_dynamic_event_features=False, passed as an
    #              override by train_graph_rl_static.py).
    # Resolve the flag the same way it is resolved below (override wins over the
    # TrainingConfig default) so the name is decided before the namespace exists.
    use_dynamic_event_features = bool(overrides.get(
        'use_dynamic_event_features',
        getattr(tc, 'use_dynamic_event_features', True)))
    design_suffix = 'dynamic' if use_dynamic_event_features else 'static'
    model_name = f'hgt_{n_j}X{n_m}_{design_suffix}'

    # Build flat namespace from training config
    hgt_config = argparse.Namespace()

    # Training hyperparams (flat)
    for key, value in tc.training.items():
        setattr(hgt_config, key, value)

    # Validation params (flat)
    for key, value in tc.validation.items():
        # 'seed' inside the validation block would land on hgt_config.seed and
        # collide with the TRAINING seed. It did: validation_seed then resolved
        # to config.seed, seed_context() restored the stream to the same
        # position the first training rollout starts from, and validation
        # instance i was training-batch instance i.
        setattr(hgt_config, 'validation_seed' if key == 'seed' else key, value)

    # Common settings
    hgt_config.device = tc.device
    hgt_config.seed = tc.seed
    hgt_config.use_pt_nodes = getattr(tc, 'use_pt_nodes', True)
    # Dynamic-event node features (machine is_broken + job released_time).
    # Default True (new design); the old/static design passes False as an
    # override. Persisted in the checkpoint so inference rebuilds a matching
    # env.
    hgt_config.use_dynamic_event_features = getattr(
        tc, 'use_dynamic_event_features', True)
    # Off by default: turning it on changes the generated instance
    # distribution at every size except the reference, so it must be an
    # explicit choice of the run, never a silent default. See
    # scale_event_magnitudes().
    hgt_config.scale_event_magnitudes_to_size = bool(getattr(
        tc, 'scale_event_magnitudes_to_size', False))
    hgt_config.wandb_project = tc.wandb_project
    hgt_config.reset_env_episode = tc.reset_env_episode

    # Model / wandb naming
    hgt_config.model_name = model_name
    hgt_config.wandb_run_name = model_name

    # Problem size
    hgt_config.n_j = n_j
    hgt_config.n_m = n_m

    # Data generation parameters
    pp = static_params
    hgt_config.low = pp['process_time_min']
    hgt_config.high = pp['process_time_max']
    hgt_config.operation_per_job_min = pp['operation_per_job_min']
    hgt_config.operation_per_job_max = pp['operation_per_job_max']
    hgt_config.machine_per_operation_min = pp['machine_per_operation_min']
    hgt_config.machine_per_operation_max = pp['machine_per_operation_max']

    # Sub-dicts needed by trainer internals
    hgt_config.common_params = tc.common_params.copy()
    hgt_config.common_params['model_name'] = model_name
    hgt_config.static_params = static_params
    hgt_config.curriculum_learning = dict(tc.curriculum_learning)
    hgt_config.dp_lb_diff = _reward_diff_config(tc)
    hgt_config.objective_weights = tc.objective_weights.copy()

    # DataGenerator reads ``problem_params`` first. Resolve the curriculum's
    # selected event scenario into flat generator keys so phase 3 has its
    # ramp target, and CL-off trains on target events from step 0.
    _sync_training_event_params(hgt_config)

    # Result dir
    hgt_config.result_dir = result_dir

    # Metadata
    hgt_config.model_suffix = ''
    hgt_config.train_size = f"{n_j}x{n_m}"
    hgt_config.test_data = [f"{n_j}x{n_m}"]
    hgt_config.test_model = [f"{n_j}x{n_m}"]
    hgt_config.test_method = []

    # disable_curriculum is a user-facing shortcut; translate into the nested
    # curriculum_learning override before the generic setattr loop runs.
    if overrides.pop('disable_curriculum', False):
        cl = _merge_curriculum_config(
            hgt_config.curriculum_learning,
            overrides.get('curriculum_learning') or {},
        )
        cl['enabled'] = False
        overrides['curriculum_learning'] = cl
    elif isinstance(overrides.get('curriculum_learning'), dict):
        overrides['curriculum_learning'] = _merge_curriculum_config(
            hgt_config.curriculum_learning,
            overrides['curriculum_learning'],
        )
    if isinstance(overrides.get('dynamic_event_params'), dict):
        overrides['curriculum_learning'] = _merge_curriculum_config(
            overrides.get('curriculum_learning') or hgt_config.curriculum_learning,
            {'dynamic_event_params': overrides['dynamic_event_params']},
        )

    # Apply overrides last
    for key, value in overrides.items():
        setattr(hgt_config, key, value)

    if ('curriculum_learning' in overrides
            or 'dynamic_event_params' in overrides
            or 'scale_event_magnitudes_to_size' in overrides):
        _sync_training_event_params(hgt_config)

    # Keep common_params['device'] in sync when device is overridden, mirroring
    # create_inference_config's behavior.
    if 'device' in overrides:
        hgt_config.common_params = dict(hgt_config.common_params)
        hgt_config.common_params['device'] = overrides['device']

    hgt_config.reproducibility = reproducibility_metadata(hgt_config.seed)
    _resolve_mlp_channels(hgt_config, overrides)

    if create_directory:
        require_training_controls(hgt_config)
        os.makedirs(result_dir, exist_ok=True)
        setup_wandb_env(result_dir, hgt_config.wandb_project)

    return hgt_config


def create_hgt_single_obj_config(experiment_name: str = None,
                                  n_j: int = None, n_m: int = None,
                                  generator_type: str = 'SD1',
                                  **overrides) -> argparse.Namespace:
    """Build a single-objective HGT training config (DANIEL replication).

    Pass ``n_j``/``n_m`` to set the training size and ``generator_type``
    (``'SD1'`` or ``'SD2'``) to pick DANIEL's distribution. The SD1/SD2
    sampling constants (PT range, ops/job, compatible-machine bounds)
    live in ``DataGeneratorSingleObj`` and are not exposed here.

    The returned namespace is consumed by ``HGTSingleObjTrainer`` and the
    single-obj data generator. Single-obj is makespan-only and static; the
    config carries no TWT/curriculum/dynamic-event knobs (the SO trainer
    in ``src/HGT_single_objective/`` never reads them).
    """
    from src.data_generator.data_generator_single_obj import SD_PT_MAX

    gen = generator_type.upper() if isinstance(generator_type, str) else 'SD1'
    if gen not in SD_PT_MAX:
        raise ValueError(
            f"generator_type must be one of {sorted(SD_PT_MAX)}, "
            f"got {generator_type!r}")

    if n_j is None or n_m is None:
        raise ValueError("create_hgt_single_obj_config requires n_j and n_m")

    tc = training_config_single_obj
    removed_ablation_keys = {
        'dp_lb_diff',
        'lowerbound_differencing',
        'use_lowerbound_features',
        'disable_dp_lb_diff',
        'use_pt_nodes',
        'disable_pt_nodes',
    }
    requested_removed = sorted(removed_ablation_keys.intersection(overrides))
    if requested_removed:
        raise ValueError(
            "Single-objective PT/DP ablation switches were removed. "
            f"Unsupported override(s): {', '.join(requested_removed)}. "
            "Use ablation_training_stability.py or train_graph_rl.py for "
            "main HGT ablations.")

    model_name = f'hgt_so_{gen}_{n_j}X{n_m}_static'
    if experiment_name is None:
        experiment_name = f'exp_so_{gen}_{n_j}x{n_m}'

    create_directory = overrides.pop('create_result_dir', True)
    result_dir = overrides.get('result_dir') or (
        create_experiment_result_dir(experiment_name) if create_directory
        else os.path.join('result', experiment_name))

    cfg = argparse.Namespace()

    for key, value in tc.training.items():
        setattr(cfg, key, value)
    for key, value in tc.validation.items():
        # Same collision as in the HGT factory above: keep the validation
        # fixture's seed out of cfg.seed.
        setattr(cfg, 'validation_seed' if key == 'seed' else key, value)

    cfg.device = tc.device
    cfg.seed = tc.seed
    cfg.use_pt_nodes = True
    cfg.wandb_project = tc.wandb_project
    cfg.reset_env_episode = tc.reset_env_episode

    cfg.model_name = model_name
    cfg.wandb_run_name = model_name

    cfg.n_j = n_j
    cfg.n_m = n_m

    cfg.generator_type = gen
    cfg.dp_lb_diff = {'enabled': True}

    cfg.result_dir = result_dir

    cfg.train_size = f"{n_j}x{n_m}"

    for key, value in overrides.items():
        setattr(cfg, key, value)

    if create_directory:
        require_training_controls(cfg, scale_required=True)
        os.makedirs(result_dir, exist_ok=True)
        setup_wandb_env(result_dir, cfg.wandb_project)
    cfg.reproducibility = reproducibility_metadata(cfg.seed)
    _resolve_mlp_channels(cfg, overrides)

    return cfg


def _resolve_mlp_channels(config, overrides):
    """Resolve actual channel widths once, including representation-specific inputs."""
    hidden = list(config.mlp_hidden_dims)
    if not hidden or any(type(x) is not int or x <= 0 for x in hidden):
        raise ValueError('mlp_hidden_dims must contain positive integers')
    d = int(config.hidden_dim) * int(config.num_heads)
    config.actor_channels = overrides.get('actor_channels') or [6 * d, *hidden, 1]
    config.critic_channels = overrides.get('critic_channels') or [
        (6 if config.use_pt_nodes else 3) * d, *hidden, 1]


def create_inference_config(
    n_j: int = None,
    n_m: int = None,
    device: str = None,
    seed: int = None,
    num_new_job_arrivals: int = None,
    num_machine_breakdowns: int = None,
    num_priority_changes: int = None,
    num_new_job_arrival_waves: int = None,
    num_machine_breakdown_waves: int = None,
    num_priority_change_waves: int = None,
) -> argparse.Namespace:
    """Unified config creation for inference scripts.

    Event wave counts default to whatever ``experiment_config.problem_params``
    specifies. Pass 0 for any of ``num_*``/``num_*_waves`` to force a pure
    static problem. The legacy
    ``num_new_job_arrivals``,
    ``num_machine_breakdowns`` and ``num_priority_changes`` arguments are
    treated as wave-count aliases.

    Note: objective weights are NOT exposed here. Load them from the model
    checkpoint via ``src.utils.model_utils.load_model`` and pass them
    explicitly to inference functions.
    """
    ec = experiment_config
    cfg = argparse.Namespace()

    cfg.n_j = n_j if n_j is not None else ec.problem_params['job_num']
    cfg.n_m = n_m if n_m is not None else ec.problem_params['machine_num']

    cfg.common_params = ec.common_params.copy()
    if device is not None:
        cfg.common_params['device'] = device
    cfg.device = cfg.common_params['device']
    cfg.seed = seed if seed is not None else ec.seed

    cfg.problem_params = normalize_generation_ranges(ec.problem_params.copy())
    cfg.problem_params['job_num'] = cfg.n_j
    cfg.problem_params['machine_num'] = cfg.n_m
    op_min, _ = resolve_range(
        cfg.problem_params, 'operation_per_job', (1, cfg.n_m), integer=True)
    mch_min, _ = resolve_range(
        cfg.problem_params, 'machine_per_operation', (1, cfg.n_m), integer=True)
    put_range_aliases(
        cfg.problem_params, 'operation_per_job', min(op_min, cfg.n_m), cfg.n_m)
    put_range_aliases(
        cfg.problem_params, 'machine_per_operation',
        min(mch_min, cfg.n_m), cfg.n_m)
    wave_overrides = {
        'num_new_job_arrival_waves': (
            num_new_job_arrival_waves
            if num_new_job_arrival_waves is not None
            else num_new_job_arrivals),
        'num_machine_breakdown_waves': (
            num_machine_breakdown_waves
            if num_machine_breakdown_waves is not None
            else num_machine_breakdowns),
        'num_priority_change_waves': (
            num_priority_change_waves
            if num_priority_change_waves is not None
            else num_priority_changes),
    }
    for wave_key, value in wave_overrides.items():
        if value is None:
            continue
        legacy_key = _EVENT_WAVE_TO_LEGACY_KEYS[wave_key]
        cfg.problem_params[wave_key] = value
        cfg.problem_params[legacy_key] = value

    cfg.max_updates = 1
    cfg.reproducibility = reproducibility_metadata(cfg.seed)

    return cfg
