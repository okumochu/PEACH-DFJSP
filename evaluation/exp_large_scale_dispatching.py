#!/usr/bin/env python3
"""
Large-Scale Dynamic FJSP: HGT vs Dispatching Rules

Compares the trained HGT policy against standard dispatching rules
(FIFO/SPT/EDD/WSPT/ATC_<k>/MS/COVERT_<k>/CR/MOD/AVPRO/NPT) on large-scale
dynamic FJSP instances.

At this scale with dynamic events, dispatching rules are the relevant
baseline: they are what real factories use because they are fast and
automatically adapt to events. The experiment tests whether the learned
graph policy's non-myopic decision making beats these rules in quality
while remaining fast enough for real-time use.

Tunable rules (ATC, COVERT) are pre-expanded in the menu — each k value
appears as its own method name (e.g. ``ATC_1.0`` ... ``ATC_4.0``), so no
calibration step is needed before the benchmark and every variant is
evaluated independently. The script emits raw CSV/JSON results only.

Fairness note: objective weights are loaded from the model checkpoint,
not config.py.

Usage:
    # Full run with defaults from config.py (100x20, wave counts from config)
    python exp_large_scale_dispatching.py

    # Custom size / events / runs
    python exp_large_scale_dispatching.py --n_j 50 --n_m 15 --num_runs 10
    python exp_large_scale_dispatching.py --num_new_job_arrival_waves 3 --num_machine_breakdowns_wave 3

    # Subset of rules (k-variants written inline)
    python exp_large_scale_dispatching.py --rules SPT EDD ATC_2.0 COVERT_1.5
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.HGT.hgt_env import FJSPEnvHGT
from src.HGT.inference_env import InferenceEnv
from src.data_generator import DataGenerator
from src.utils.gantt_util import extract_schedule_from_env
from src.utils.model_utils import load_model
from src.utils.config_util import (
    create_inference_config,
    translate_event_profile_to_generic,
)
from src.utils.reproducibility import derive_seed, seed_context, set_global_seed
from src.utils.resumable_log import ResumableLog, resolve_run_dir
from src.benchmark.dispatching_rules import run_dispatching_rule
from src.benchmark.solver_adapter import make_greedy_solver_fn
from src.HGT_dispatch.dispatching_rules_vec import parse_rule_name
from config import experiment_config as exp_config


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_METHOD = 'rl'
ORACLE_METHOD = 'best_rule_oracle'
SCENARIO_CHOICES = ('priority_shock', 'capacity_shock')
LEGACY_METHOD_NAMES = {
    REFERENCE_METHOD: ('rl_greedy',),
}


def _legacy_method_names(method: str) -> tuple:
    return LEGACY_METHOD_NAMES.get(method, ())


def _parse_int_profile_arg(value) -> Optional[List[int]]:
    """Parse comma/range count profiles from CLI strings.

    ``"3,4,3"`` -> ``[3, 4, 3]``; ``"3-5"`` -> ``[3, 5]``.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    cleaned = str(value).strip().replace(';', ',')
    if not cleaned:
        return []
    if '-' in cleaned and ',' not in cleaned:
        parts = cleaned.split('-')
    else:
        parts = cleaned.split(',')
    return [int(p.strip()) for p in parts if p.strip()]


def _range_bounds(value) -> tuple:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0, 0
        if len(value) == 1:
            return value[0], value[0]
        return min(value), max(value)
    return value, value


def _range_label(value) -> str:
    lo, hi = _range_bounds(value)
    if lo == hi:
        return str(lo)
    return f"{lo}-{hi}"


def _resolve_event_profile(args, large_scale_params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve scenario defaults plus CLI overrides into generator params."""
    base = {
        k: v for k, v in large_scale_params.items()
        if k not in ('scenarios', 'rules')
    }
    scenario_defaults = large_scale_params.get(
        'scenarios', {}).get(args.scenario, {})
    profile = {**base, **dict(scenario_defaults)}
    profile['scenario'] = args.scenario

    wave_keys = [
        'num_new_job_arrival_waves',
        'num_machine_breakdown_waves',
        'num_priority_change_waves',
    ]
    for key in wave_keys:
        value = getattr(args, key, None)
        if value is not None:
            profile[key] = int(value)

    # Scenario-tuning overrides. Each CLI flag maps to a generic generator
    # range so the profile never carries scenario-themed legacy keys.
    #
    # priority_shock (demand-side) knobs.
    if getattr(args, 'arrival_waves', None) is not None:
        profile['num_new_job_arrival_waves'] = int(args.arrival_waves)
    if getattr(args, 'arrival_batch_sizes', None) is not None:
        batch = _parse_int_profile_arg(args.arrival_batch_sizes) or []
        if batch:
            profile['new_jobs_per_arrival'] = [min(batch), max(batch)]
            if (args.arrival_waves is None
                    and args.num_new_job_arrival_waves is None):
                profile['num_new_job_arrival_waves'] = len(batch)
    if getattr(args, 'arrival_job_weight', None) is not None:
        profile['new_job_weight'] = int(args.arrival_job_weight)
    if (getattr(args, 'arrival_ddt_min', None) is not None
            or getattr(args, 'arrival_ddt_max', None) is not None):
        lo, hi = _range_bounds(
            profile.get('new_job_due_date_tightness', [0.5, 1.5]))
        if args.arrival_ddt_min is not None:
            lo = float(args.arrival_ddt_min)
        if args.arrival_ddt_max is not None:
            hi = float(args.arrival_ddt_max)
        profile['new_job_due_date_tightness'] = [lo, hi]
    if getattr(args, 'priority_waves', None) is not None:
        profile['num_priority_change_waves'] = int(args.priority_waves)
    if getattr(args, 'priority_jobs_per_change', None) is not None:
        jobs = _parse_int_profile_arg(args.priority_jobs_per_change) or []
        if jobs:
            profile['priority_jobs_per_change'] = [min(jobs), max(jobs)]
    if getattr(args, 'priority_new_weight', None) is not None:
        profile['priority_new_weight'] = int(args.priority_new_weight)

    # capacity_shock (supply-side) knobs.
    if getattr(args, 'breakdown_waves', None) is not None:
        profile['num_machine_breakdown_waves'] = int(args.breakdown_waves)
    if getattr(args, 'machines_per_breakdown', None) is not None:
        machines = _parse_int_profile_arg(args.machines_per_breakdown) or []
        if machines:
            profile['machines_per_breakdown'] = [min(machines), max(machines)]
    if getattr(args, 'repair_duration_mean', None) is not None:
        profile['repair_duration'] = [float(args.repair_duration_mean),
                                      float(args.repair_duration_mean)]

    return translate_event_profile_to_generic(profile)


def _apply_event_profile_to_config(config, event_profile: Dict[str, Any]) -> None:
    for key, value in event_profile.items():
        config.problem_params[key] = value


def couple_priority_to_arrivals(dynamic_data: Dict[str, Any]) -> None:
    """Retime each priority change to coincide with a new-job arrival wave.

    In this experiment a priority shock is part of the same demand event as a
    rush arrival: when a batch of new (high-priority) jobs arrives, existing
    jobs are simultaneously reprioritized. This rewrites the priority-change
    times in place to the arrival-wave timestamps (paired in order; the last
    wave repeats if there are more priority changes than arrival waves). No-op
    when the instance has no arrivals or no priority changes (e.g. the
    capacity_shock scenario), so it is safe to call on every instance.
    """
    events = dynamic_data.get('dynamic_events') or {}
    arrivals = events.get('new_job_arrival') or []
    priority = events.get('priority_change') or []
    if not arrivals or not priority:
        return
    wave_times = sorted({float(t) for t in arrivals})
    for i, event in enumerate(priority):
        if isinstance(event, dict):
            event['time'] = wave_times[min(i, len(wave_times) - 1)]
    priority.sort(key=lambda p: float(p.get('time', 0.0)))


def _kwargs_for_rule(canonical: str, k: Optional[float]) -> Dict[str, float]:
    """Return the run_dispatching_rule kwarg matching the rule's tunable, if any.

    ``('ATC', 2.0)`` -> ``{'atc_k': 2.0}``; ``('FIFO', None)`` -> ``{}``.
    """
    if canonical == 'ATC' and k is not None:
        return {'atc_k': float(k)}
    if canonical == 'COVERT' and k is not None:
        return {'covert_k': float(k)}
    return {}


def _duration_summary(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {'count': 0, 'min': None, 'mean': None, 'max': None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'count': int(arr.size),
        'min': float(arr.min()),
        'mean': float(arr.mean()),
        'max': float(arr.max()),
    }


def _jsonable(obj: Any) -> Any:
    """Convert numpy/torch values into plain JSON-compatible objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(v) for v in obj)
    return obj


def _schedule_record(
    method: str,
    solver_result: Dict[str, Any],
    total_time: float,
) -> Dict[str, Any]:
    """Serializable final schedule payload for later Gantt reconstruction."""
    best_env = solver_result.get('best_env')
    best_idx = int(solver_result.get('best_env_idx', 0) or 0)
    if best_env is not None:
        schedule_data = extract_schedule_from_env(best_env, env_idx=best_idx)
    else:
        schedule_data = {
            'schedule': [],
            'job_data': [],
            'num_jobs': None,
            'num_machines': None,
            'makespan': solver_result.get('makespan'),
            'twt': solver_result.get('twt'),
            'breakdown_periods': solver_result.get('breakdown_periods', []),
            'interrupted_ops': solver_result.get('interrupted_ops', []),
        }

    return _jsonable({
        'method': method,
        'status': solver_result.get('status', 'OK'),
        'objective': solver_result.get('objective'),
        'makespan': solver_result.get('makespan'),
        'twt': solver_result.get('twt'),
        'total_time': total_time,
        'num_segments': solver_result.get('num_segments'),
        'num_jobs': schedule_data.get('num_jobs'),
        'num_machines': schedule_data.get('num_machines'),
        'schedule': schedule_data.get('schedule', []),
        'job_data': schedule_data.get('job_data', []),
        'breakdown_periods': schedule_data.get('breakdown_periods', []),
        'interrupted_ops': schedule_data.get('interrupted_ops', []),
    })


def summarize_event_profile(dynamic_data: Dict[str, Any],
                            run_idx: int) -> Dict[str, Any]:
    events = dynamic_data.get('dynamic_events') or {}
    arrivals = [float(t) for t in (events.get('new_job_arrival') or [])]
    arrival_groups: Dict[float, int] = {}
    for t in arrivals:
        arrival_groups[t] = arrival_groups.get(t, 0) + 1
    arrival_batches = [
        {'time': float(t), 'batch_size': int(size)}
        for t, size in sorted(arrival_groups.items())
    ]

    priority_items = events.get('priority_change') or []
    priority_shocks = [p for p in priority_items if isinstance(p, dict)]
    priority_counts = [
        len(p.get('job_indices') or []) for p in priority_shocks
    ]

    breakdowns = events.get('machine_breakdown') or []
    breakdown_groups: Dict[float, List[Dict[str, Any]]] = {}
    durations = []
    for bd in breakdowns:
        t = float(bd['time'])
        breakdown_groups.setdefault(t, []).append(bd)
        durations.append(float(bd.get('duration', 0.0)))
    breakdown_waves = [
        {'time': float(t), 'machines': sorted(
            int(bd['machine_idx']) for bd in group)}
        for t, group in sorted(breakdown_groups.items())
    ]
    quake_waves = [w for w in breakdown_waves if len(w['machines']) > 1]

    return {
        'run_idx': int(run_idx),
        'arrival_event_count': int(len(arrivals)),
        'arrival_wave_count': int(len(arrival_batches)),
        'arrival_batch_sizes': [b['batch_size'] for b in arrival_batches],
        'arrival_batch_times': [b['time'] for b in arrival_batches],
        'arrival_batches': arrival_batches,
        'priority_shock_event_count': int(len(priority_shocks)),
        'priority_shock_times': [
            float(p['time']) for p in priority_shocks
        ],
        'priority_affected_job_counts': [int(c) for c in priority_counts],
        'legacy_priority_change_count': int(
            len(priority_items) - len(priority_shocks)),
        'breakdown_wave_count': int(len(breakdown_waves)),
        'machines_affected_per_breakdown_wave': [
            len(w['machines']) for w in breakdown_waves
        ],
        'breakdown_waves': breakdown_waves,
        'earthquake_wave_count': int(len(quake_waves)),
        'machines_affected_per_earthquake_wave': [
            len(w['machines']) for w in quake_waves
        ],
        'earthquake_waves': quake_waves,
        'total_breakdown_events': int(len(breakdowns)),
        'breakdown_durations_summary': _duration_summary(durations),
    }


def run_benchmark(
    policy,
    config,
    data_generator: DataGenerator,
    rules: List[str],
    num_runs: int,
    makespan_weight: float,
    twt_weight: float,
    seed: int,
    log: Optional[ResumableLog] = None,
    schedule_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run RL + each dispatching rule on the same pre-generated instances.

    ``rules`` is a list of menu names; tunable rules carry their k inline
    (e.g. ``'ATC_2.0'``, ``'COVERT_1.5'``). Non-tunable names are passed
    through unchanged. Each k-variant is evaluated as its own fixed rule.
    ``best_rule_oracle`` is computed post-hoc as the per-instance minimum
    objective across all dispatching rules.
    """
    instances = []
    for run_idx in range(num_runs):
        with seed_context(derive_seed(seed, 'instance', run_idx)):
            instance = data_generator.sample_env(num_envs=1)
        couple_priority_to_arrivals(instance)
        instances.append(instance)
    event_profile_summary = [
        summarize_event_profile(instance, run_idx)
        for run_idx, instance in enumerate(instances)
    ]
    if log is not None:
        for run_idx, instance in enumerate(instances):
            instance_key = {'record_type': 'instance', 'run_idx': run_idx}
            if not log.is_done(instance_key):
                log.append(instance_key, {
                    'run_seed': int(derive_seed(seed, 'instance', run_idx)),
                    'data_params': _jsonable(dict(config.problem_params)),
                    'event_profile_summary': event_profile_summary[run_idx],
                })
    schedule_by_run: Dict[int, Dict[str, Any]] = {}
    if schedule_records is not None:
        for run_idx, instance in enumerate(instances):
            rec = {
                'run_idx': int(run_idx),
                'run_seed': int(derive_seed(seed, 'instance', run_idx)),
                'data_params': _jsonable(dict(config.problem_params)),
                'instance_data': _jsonable(instance),
                'event_profile_summary': event_profile_summary[run_idx],
                'schedules': {},
            }
            schedule_records.append(rec)
            schedule_by_run[run_idx] = rec

    has_dynamic_events = any(
        (instance.get('dynamic_events') or {}).get(k)
        for instance in instances
        for k in ('new_job_arrival', 'machine_breakdown', 'priority_change'))

    # Each tuple is (canonical_label, canonical_rule_name, k_or_None).
    # ``canonical_label`` is the lowercased menu name (e.g. 'atc_2.0') so each
    # k-variant has a unique result-key; ``canonical_rule_name`` is the
    # underlying rule the scorer dispatches on.
    rules_with_k: List[tuple] = []
    for r in rules:
        canonical_rule, k_val = parse_rule_name(r)
        rules_with_k.append((r.lower(), canonical_rule, k_val))

    # Rules first, RL last so the proposed method stays visually separated
    # in printed and CSV summaries.
    base_methods = [canonical for canonical, _o, _k in rules_with_k]
    methods = (
        base_methods
        + ([ORACLE_METHOD] if base_methods else [])
        + [REFERENCE_METHOD]
    )
    results: Dict[str, Dict[str, List[float]]] = {
        m: {'makespan': [], 'twt': [], 'objective': [], 'time': []} for m in methods
    }

    _greedy_solver_fn = make_greedy_solver_fn(policy, config.device)

    def _record(method: str, run_idx: int, dyn_data, solve_fn):
        """Run or replay one (run_idx, method) record. ``solve_fn`` is invoked
        only on cache miss; on hit, cached metrics are replayed."""
        key = {'record_type': 'method', 'run_idx': run_idx, 'method': method}
        cached = None
        if log is not None:
            cached = log.get(key)
            if cached is None:
                for legacy_method in _legacy_method_names(method):
                    legacy_key = {
                        'record_type': 'method',
                        'run_idx': run_idx,
                        'method': legacy_method,
                    }
                    cached = log.get(legacy_key)
                    if cached is not None:
                        break
        if cached is not None:
            results[method]['makespan'].append(cached['makespan'])
            results[method]['twt'].append(cached['twt'])
            results[method]['objective'].append(cached['objective'])
            results[method]['time'].append(cached['time'])
            return

        t0 = time.time()
        res = solve_fn(dyn_data)
        elapsed = time.time() - t0
        results[method]['makespan'].append(res['makespan'])
        results[method]['twt'].append(res['twt'])
        results[method]['objective'].append(res['objective'])
        results[method]['time'].append(elapsed)
        if log is not None:
            log.append(key, {
                'makespan': res['makespan'], 'twt': res['twt'],
                'objective': res['objective'], 'time': elapsed,
                'status': res.get('status', 'OK'),
            })
        if run_idx in schedule_by_run:
            schedule_by_run[run_idx]['schedules'][method] = _schedule_record(
                method, res, elapsed)

    def _solve_rl(dyn_data):
        inf_env = InferenceEnv(
            config.n_j, config.n_m, device=config.device,
            use_pt_nodes=getattr(config, 'use_pt_nodes', True),
            use_dynamic_event_features=getattr(
                config, 'use_dynamic_event_features', True))
        return inf_env.run_with_solver(
            _greedy_solver_fn, dyn_data,
            makespan_weight=makespan_weight, twt_weight=twt_weight,
            verbose=False, collect_snapshots=False,
        )


    def _solve_rule(canonical_rule_name, k_val):
        rule_kwargs = _kwargs_for_rule(canonical_rule_name, k_val)
        def _solve(dyn_data):
            env_r = FJSPEnvHGT(
                config.n_j, config.n_m, device=config.device,
                use_pt_nodes=getattr(config, 'use_pt_nodes', True),
                use_dynamic_event_features=getattr(
                    config, 'use_dynamic_event_features', True))
            return run_dispatching_rule(
                env=env_r, dynamic_data=dyn_data, rule_name=canonical_rule_name,
                makespan_weight=makespan_weight, twt_weight=twt_weight,
                device=config.device, verbose=False, **rule_kwargs,
            )
        return _solve

    def _record_oracle(run_idx: int):
        """Compute best_rule_oracle metrics from the per-rule results
        already appended for this run. Time is the *sum* across rules
        (worst-case "you'd need to run all rules to know which is best")."""
        rule_canonicals = [c for c, _o, _k in rules_with_k]
        objs = [results[c]['objective'][-1] for c in rule_canonicals]
        # Pick the best rule by objective and use its makespan/twt.
        best_idx = int(np.argmin(objs))
        winner = rule_canonicals[best_idx]
        results[ORACLE_METHOD]['makespan'].append(
            results[winner]['makespan'][-1])
        results[ORACLE_METHOD]['twt'].append(
            results[winner]['twt'][-1])
        results[ORACLE_METHOD]['objective'].append(
            results[winner]['objective'][-1])
        results[ORACLE_METHOD]['time'].append(
            sum(results[c]['time'][-1] for c in rule_canonicals))

    for run_idx in tqdm(range(num_runs), desc="Instances", unit="instance"):
        dyn_data = instances[run_idx]

        _record(REFERENCE_METHOD, run_idx, dyn_data, _solve_rl)


        for canonical, rule_name, k_val in rules_with_k:
            _record(canonical, run_idx, dyn_data,
                    _solve_rule(rule_name, k_val))

        if base_methods:
            _record_oracle(run_idx)

    return {
        'results': results,
        'methods': methods,
        'rules': rules,
        'num_runs': num_runs,
        'has_dynamic_events': has_dynamic_events,
        'event_profile_summary': event_profile_summary,
    }


# =========================================================================
# Reporting
# =========================================================================

def save_comparison_csv(data: Dict[str, Any], output_path: str):
    """Save comparison results as CSV: one row per method, metrics as columns.

    rl is the reference; rules are compared via objective_improv_pct
    (vs reference) and time_slowdown_x (vs reference). Reference row has '-'
    in those columns. The CSV always contains every menu rule (no paper-plot
    filtering) so downstream analysis stays complete.

    Run-level metadata is stored in runs.json so the CSV invariant
    'one row per method' holds.
    """
    results = data['results']
    methods = data['methods']
    ref = REFERENCE_METHOD

    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        'method',
        'makespan_mean', 'makespan_std',
        'twt_mean', 'twt_std',
        'objective_mean', 'objective_std',
        'time_mean', 'time_std',
        'objective_improv_pct',
        'time_slowdown_x',
    ]

    ref_obj_mean = float(np.mean(results[ref]['objective']))
    ref_time_mean = float(np.mean(results[ref]['time']))

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in methods:
            r = results[m]
            obj_mean = float(np.mean(r['objective']))
            t_mean = float(np.mean(r['time']))
            row = {
                'method': m,
                'makespan_mean': f"{np.mean(r['makespan']):.4f}",
                'makespan_std':  f"{np.std(r['makespan']):.4f}",
                'twt_mean':      f"{np.mean(r['twt']):.4f}",
                'twt_std':       f"{np.std(r['twt']):.4f}",
                'objective_mean': f"{obj_mean:.4f}",
                'objective_std':  f"{np.std(r['objective']):.4f}",
                'time_mean':      f"{t_mean:.4f}",
                'time_std':       f"{np.std(r['time']):.4f}",
            }
            if m == ref:
                row['objective_improv_pct'] = '-'
                row['time_slowdown_x'] = '-'
            else:
                row['objective_improv_pct'] = (
                    f"{(obj_mean - ref_obj_mean) / obj_mean * 100:.2f}"
                    if obj_mean > 0 else "0.00"
                )
                row['time_slowdown_x'] = (
                    f"{t_mean / ref_time_mean:.2f}x"
                    if ref_time_mean > 0 else "N/A"
                )
            writer.writerow(row)


def save_event_profile_summary(data: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    payload = {
        'scenario': data.get('scenario', 'priority_shock'),
        'base_problem_size': data.get('base_problem_size'),
        'event_profile_params': data.get('event_profile_params'),
        'runs': data.get('event_profile_summary', []),
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)


def _large_run_metadata(
    args,
    config,
    event_profile: Dict[str, Any],
    makespan_weight: float,
    twt_weight: float,
    model_hparams: Dict[str, Any],
    repro: Dict[str, Any],
) -> Dict[str, Any]:
    return _jsonable({
        'experiment': 'exp_large_scale_dispatching',
        'script': os.path.basename(__file__),
        'argv': sys.argv,
        'output_contract': {
            'plots': [],
            'tables': ['comparison_results.csv'],
            'raw_json': ['runs.json', 'schedules.json',
                         'event_profile_summary.json'],
            'resume_log': 'runs.json',
            'schedule_results': 'schedules.json',
        },
        'model_path': args.model_path,
        'model_hparams': model_hparams,
        'objective_weights': {
            'makespan_weight': makespan_weight,
            'twt_weight': twt_weight,
        },
        'base_problem_size': {
            'jobs': int(args.n_j),
            'machines': int(args.n_m),
        },
        'scenario': args.scenario,
        'event_profile_params': event_profile,
        'resolved_problem_params': dict(config.problem_params),
        'large_scale_param_defaults': dict(exp_config.large_scale_params),
        'rules': list(args.rules),
        'num_runs': int(args.num_runs),
        'seed': int(args.seed),
        'device': config.device,
        'use_pt_nodes': bool(config.use_pt_nodes),
        'use_dynamic_event_features': bool(
            getattr(config, 'use_dynamic_event_features', True)),
        'reproducibility': repro,
    })


def save_schedule_results(records: List[Dict[str, Any]], output_path: str) -> None:
    """Save generated instances plus method schedules for Gantt reconstruction."""
    merged: Dict[int, Dict[str, Any]] = {}
    if os.path.isfile(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for rec in existing.get('records', []):
                if rec.get('run_idx') is not None:
                    merged[int(rec['run_idx'])] = rec
        except (OSError, json.JSONDecodeError):
            merged = {}

    for rec in records:
        if rec.get('run_idx') is None or not rec.get('schedules'):
            continue
        merged[int(rec['run_idx'])] = rec

    schedule_records = [
        merged[k] for k in sorted(merged)
    ]
    payload = {
        'experiment': 'exp_large_scale_dispatching',
        'records': schedule_records,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(payload), f, indent=2)
        f.write('\n')


def print_summary(data: Dict[str, Any]):
    """Print a summary table comparing every method to REFERENCE_METHOD.

    Includes every method so the table stays a complete audit trail.
    """
    results = data['results']
    methods = data['methods']
    ref = REFERENCE_METHOD
    ref_obj = np.mean(results[ref]['objective'])

    print(f"\n{'='*92}")
    print("LARGE-SCALE DYNAMIC FJSP: HGT vs DISPATCHING RULES")
    print(f"{'='*92}")
    print(f"  {'Method':<12} {'Makespan':>14} {'TWT':>14} {'Objective':>14} "
          f"{'Time(s)':>10} {'Gain%':>8}")
    print(f"  {'-'*90}")

    for m in methods:
        r = results[m]
        ms = np.mean(r['makespan']); ms_s = np.std(r['makespan'])
        twt = np.mean(r['twt']); twt_s = np.std(r['twt'])
        obj = np.mean(r['objective']); obj_s = np.std(r['objective'])
        t = np.mean(r['time'])

        ms_str = f"{ms:8.1f}±{ms_s:5.1f}"
        twt_str = f"{twt:8.1f}±{twt_s:5.1f}"
        obj_str = f"{obj:8.1f}±{obj_s:5.1f}"

        if m == ref:
            gain_str = ' ref'
            tag = f'  <-- {ref}'
        else:
            gain = ((obj - ref_obj) / obj * 100) if obj > 0 else 0.0
            gain_str = f'{gain:6.2f}%'
            tag = ''

        print(f"  {m:<12} {ms_str:>14} {twt_str:>14} {obj_str:>14} "
              f"{t:>10.3f} {gain_str:>8}{tag}")

    print(f"{'='*92}")

    best = min(methods, key=lambda m: np.mean(results[m]['objective']))
    print(f"  Best: {best}  (Objective = {np.mean(results[best]['objective']):.2f})")
    print(f"{'='*92}")

# =========================================================================
# Scenario Validation
# =========================================================================

def _work_content_by_job(job_length: np.ndarray,
                         op_pt: np.ndarray) -> np.ndarray:
    work = np.zeros(len(job_length), dtype=np.float64)
    op_idx = 0
    for j, num_ops in enumerate(job_length):
        total = 0.0
        for _ in range(int(num_ops)):
            op_times = op_pt[op_idx]
            compatible = op_times[op_times > 0]
            if compatible.size > 0:
                total += float(np.mean(compatible))
            op_idx += 1
        work[j] = total
    return work


def _validation_args_for_scenario(
    scenario: str,
    device: Optional[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        scenario=scenario,
        device=device,
        n_j=8,
        n_m=4,
        num_new_job_arrival_waves=None,
        num_machine_breakdown_waves=None,
        num_priority_change_waves=None,
        arrival_waves=None,
        arrival_batch_sizes=None,
        arrival_job_weight=None,
        arrival_ddt_min=None,
        arrival_ddt_max=None,
        priority_waves=None,
        priority_jobs_per_change=None,
        priority_new_weight=None,
        breakdown_waves=None,
        machines_per_breakdown=None,
        repair_duration_mean=None,
    )


def validate_scenarios(device: Optional[str] = None,
                       seed: int = 123) -> None:
    """Lightweight schema and dispatching smoke checks for all scenarios."""
    set_global_seed(seed)
    effective_device = device or exp_config.common_params['device']
    if 'cuda' in str(effective_device) and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        effective_device = 'cpu'

    print("\nScenario validation")
    print(f"  device: {effective_device}")
    for scenario in SCENARIO_CHOICES:
        args = _validation_args_for_scenario(scenario, effective_device)
        profile = _resolve_event_profile(args, exp_config.large_scale_params)
        cfg = create_inference_config(
            n_j=args.n_j,
            n_m=args.n_m,
            device=effective_device,
            seed=seed,
            num_new_job_arrival_waves=profile['num_new_job_arrival_waves'],
            num_machine_breakdown_waves=profile['num_machine_breakdown_waves'],
            num_priority_change_waves=profile['num_priority_change_waves'],
        )
        _apply_event_profile_to_config(cfg, profile)
        generator = DataGenerator(cfg)
        with seed_context(derive_seed(seed, scenario, 'validation_instance')):
            instance = generator.sample_env(num_envs=1)
        events = instance['dynamic_events']
        summary = summarize_event_profile(instance, run_idx=0)

        arr_lo, arr_hi = _range_bounds(profile['num_new_job_arrival_waves'])
        bd_lo, bd_hi = _range_bounds(profile['num_machine_breakdown_waves'])
        pc_lo, pc_hi = _range_bounds(profile['num_priority_change_waves'])
        assert arr_lo <= summary['arrival_wave_count'] <= arr_hi
        assert bd_lo <= summary['breakdown_wave_count'] <= bd_hi
        assert pc_lo <= summary['priority_shock_event_count'] <= pc_hi

        if scenario == 'priority_shock':
            arrivals = events.get('new_job_arrival', [])
            if summary['arrival_event_count'] > 0:
                jobs_lo, jobs_hi = _range_bounds(profile['new_jobs_per_arrival'])
                assert all(
                    jobs_lo <= b <= jobs_hi
                    for b in summary['arrival_batch_sizes'])
                repeated = any(arrivals.count(t) > 1 for t in set(arrivals))
                assert repeated, "priority_shock should batch arrivals by timestamp"

            dyn_jobs = instance['dynamic_jobs']
            assert dyn_jobs['job_weight'].size > 0
            w_lo, w_hi = _range_bounds(profile['new_job_weight'])
            assert np.all(dyn_jobs['job_weight'] >= w_lo)
            assert np.all(dyn_jobs['job_weight'] <= w_hi)
            work = _work_content_by_job(
                dyn_jobs['job_length'], dyn_jobs['op_pt'])
            release = np.asarray(arrivals, dtype=np.float64)
            due = np.asarray(dyn_jobs['job_due_date'], dtype=np.float64)
            tightness = (due - release) / np.maximum(work, 1e-9)
            ddt_lo, ddt_hi = _range_bounds(profile['new_job_due_date_tightness'])
            assert np.all(tightness >= ddt_lo - 1e-6)
            assert np.all(tightness <= ddt_hi + 1e-6)

            priority = events.get('priority_change', [])
            assert priority and all(isinstance(p, dict) for p in priority)
            pj_lo, pj_hi = _range_bounds(profile['priority_jobs_per_change'])
            assert all(
                pj_lo <= len(p.get('job_indices', [])) <= pj_hi
                for p in priority)
            pw_lo, pw_hi = _range_bounds(profile['priority_new_weight'])
            assert all(
                pw_lo <= float(p.get('new_weight')) <= pw_hi
                for p in priority)

        if scenario == 'capacity_shock':
            breakdowns = events.get('machine_breakdown', [])
            times = [float(bd['time']) for bd in breakdowns]
            shared = any(times.count(t) > 1 for t in set(times))
            assert shared, (
                "capacity_shock should group multiple breakdowns "
                "at shared timestamps")
            mch_lo, mch_hi = _range_bounds(profile['machines_per_breakdown'])
            assert all(
                mch_lo <= len(w['machines']) <= mch_hi
                for w in summary['breakdown_waves'])
            dur_lo, dur_hi = _range_bounds(profile['repair_duration'])
            assert (
                summary['breakdown_durations_summary']['min'] >= dur_lo
                and summary['breakdown_durations_summary']['max'] <= dur_hi)

        env = FJSPEnvHGT(cfg.n_j, cfg.n_m, device=effective_device)
        result = run_dispatching_rule(
            env=env,
            dynamic_data=instance,
            rule_name='SPT',
            makespan_weight=0.5,
            twt_weight=0.5,
            device=effective_device,
            verbose=False,
        )
        assert result['makespan'] is not None
        assert result['objective'] is not None
        print(f"  {scenario:<20} OK  events={summary}")


# =========================================================================
# Main
# =========================================================================

def _run_one_scenario(args, scenario, lsp, base_output_dir, *, policy,
                      model_hparams, makespan_weight, twt_weight,
                      device, repro):
    """Run the full HGT-vs-rules benchmark for one scenario into its subdir."""
    args.scenario = scenario
    event_profile = _resolve_event_profile(args, lsp)

    config = create_inference_config(
        n_j=args.n_j, n_m=args.n_m,
        device=device,
        seed=args.seed,
        num_new_job_arrival_waves=event_profile['num_new_job_arrival_waves'],
        num_machine_breakdown_waves=event_profile['num_machine_breakdown_waves'],
        num_priority_change_waves=event_profile['num_priority_change_waves'],
    )
    _apply_event_profile_to_config(config, event_profile)
    config.use_pt_nodes = bool(model_hparams['use_pt_nodes'])
    # Match the checkpoint's design so the inference env's feature dims line up
    # with the trained network. The old/static design omits the dynamic-event
    # features (machine is_broken + job released_time); legacy checkpoints
    # without the key default to the new design (True).
    config.use_dynamic_event_features = bool(
        model_hparams.get('use_dynamic_event_features', True))

    data_generator = DataGenerator(config)
    output_dir = os.path.join(base_output_dir, scenario)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Large-Scale Dynamic FJSP: HGT vs Dispatching Rules  [{scenario}]")
    print(f"{'='*70}")
    print(f"  Model path       : {args.model_path}")
    print(f"  Problem size     : {args.n_j} jobs x {args.n_m} machines")
    print(f"  Scenario         : {scenario}")
    print(f"  Dynamic waves    : "
          f"{_range_label(event_profile['num_new_job_arrival_waves'])} arrivals, "
          f"{_range_label(event_profile['num_machine_breakdown_waves'])} breakdowns, "
          f"{_range_label(event_profile['num_priority_change_waves'])} priority changes")
    print(f"  Per-wave amounts : "
          f"arrival batch={_range_label(event_profile['new_jobs_per_arrival'])}, "
          f"machines/breakdown={_range_label(event_profile['machines_per_breakdown'])}, "
          f"priority jobs={_range_label(event_profile['priority_jobs_per_change'])}")
    print(f"  Event profile    : {event_profile}")
    print(f"  Runs per method  : {args.num_runs}")
    print(f"  rl               : argmax rollout")
    print(f"  Rules            : {args.rules}")
    print(f"  Weights (model)  : mw={makespan_weight}, tw={twt_weight}")
    print(f"  Device           : {device}")
    print(f"  Seed             : {args.seed}")
    print(f"  Output dir       : {output_dir}")
    print(f"{'='*70}\n")

    log = ResumableLog(os.path.join(output_dir, 'runs.json'))
    log.set_metadata(_large_run_metadata(
        args=args,
        config=config,
        event_profile=event_profile,
        makespan_weight=makespan_weight,
        twt_weight=twt_weight,
        model_hparams=model_hparams,
        repro=repro,
    ))
    if log.num_completed() > 0:
        print(f"[resume] Skipping {log.num_completed()} completed records "
              f"from {log.path}\n")

    schedule_records: List[Dict[str, Any]] = []
    try:
        data = run_benchmark(
            policy=policy, config=config, data_generator=data_generator,
            rules=args.rules, num_runs=args.num_runs,
            makespan_weight=makespan_weight, twt_weight=twt_weight,
            seed=args.seed,
            log=log,
                schedule_records=schedule_records,
        )
    finally:
        log.close()

    data.update({
        'scenario': scenario,
        'base_problem_size': {'jobs': int(args.n_j), 'machines': int(args.n_m)},
        'event_profile_params': event_profile,
        'seed': int(args.seed),
        'reproducibility': repro,
        'use_pt_nodes': bool(config.use_pt_nodes),
        'use_dynamic_event_features': bool(
            getattr(config, 'use_dynamic_event_features', True)),
    })
    save_comparison_csv(data, os.path.join(output_dir, 'comparison_results.csv'))
    save_event_profile_summary(
        data, os.path.join(output_dir, 'event_profile_summary.json'))
    save_schedule_results(schedule_records,
                          os.path.join(output_dir, 'schedules.json'))
    print_summary(data)
    print(f"\nResults saved to: {output_dir}")


def main():
    lsp = exp_config.large_scale_params
    parser = argparse.ArgumentParser(
        description='Large-Scale Dynamic FJSP: HGT vs Dispatching Rules')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'models', 'checkpoints',
                                             'generalization-scale-cl-100x20-s42.pth'))
    parser.add_argument('--n_j', type=int, default=lsp['job_num'])
    parser.add_argument('--n_m', type=int, default=lsp['machine_num'])
    parser.add_argument('--scenario', type=str, choices=SCENARIO_CHOICES,
                        default=None,
                        help='Dynamic event profile to run. Omit to sweep '
                             'all scenarios in large_scale_params.scenarios.')
    parser.add_argument(
        '--num_new_job_arrival_waves',
        '--num_new_job_arrivals_wave',
        '--num_new_job_arrivals',
        dest='num_new_job_arrival_waves',
        type=int,
        default=None,
        help='Number of arrival waves. Legacy --num_new_job_arrivals is an alias.')
    parser.add_argument(
        '--num_machine_breakdown_waves',
        '--num_machine_breakdowns_wave',
        '--num_machine_breakdowns',
        dest='num_machine_breakdown_waves',
        type=int,
        default=None,
        help='Number of machine-breakdown waves. Legacy --num_machine_breakdowns is an alias.')
    parser.add_argument(
        '--num_priority_change_waves',
        '--num_priority_changes_wave',
        '--num_priority_changes',
        dest='num_priority_change_waves',
        type=int,
        default=None,
        help='Number of priority-change waves. Legacy --num_priority_changes is an alias.')
    # priority_shock (demand-side) tuning. Legacy --rush_*/--priority_shock*
    # flags are kept as aliases.
    parser.add_argument('--arrival_waves', '--rush_waves',
                        dest='arrival_waves', type=int, default=None)
    parser.add_argument('--arrival_batch_sizes', '--rush_batch_sizes',
                        dest='arrival_batch_sizes', type=str, default=None,
                        help='Comma list such as 3,4,3.')
    parser.add_argument('--arrival_job_weight', '--rush_job_weight',
                        dest='arrival_job_weight', type=int, default=None)
    parser.add_argument('--arrival_ddt_min', '--rush_ddt_min',
                        dest='arrival_ddt_min', type=float, default=None)
    parser.add_argument('--arrival_ddt_max', '--rush_ddt_max',
                        dest='arrival_ddt_max', type=float, default=None)
    parser.add_argument('--priority_waves', '--priority_shocks',
                        dest='priority_waves', type=int, default=None)
    parser.add_argument('--priority_jobs_per_change', '--priority_jobs_per_shock',
                        dest='priority_jobs_per_change', type=str, default=None,
                        help='Range/list such as 8-12 or 8,12.')
    parser.add_argument('--priority_new_weight', '--priority_shock_new_weight',
                        dest='priority_new_weight', type=int, default=None)
    # capacity_shock (supply-side) tuning. Legacy --earthquake_*/--*quake* flags
    # are kept as aliases.
    parser.add_argument('--breakdown_waves', '--earthquake_waves',
                        dest='breakdown_waves', type=int, default=None)
    parser.add_argument('--machines_per_breakdown', '--machines_per_quake',
                        dest='machines_per_breakdown', type=str, default=None,
                        help='Range/list such as 3-5 or 3,5.')
    parser.add_argument('--repair_duration_mean', '--quake_duration_mean',
                        dest='repair_duration_mean', type=float, default=None)
    parser.add_argument('--num_runs', type=int, default=exp_config.num_runs)
    parser.add_argument('--rules', type=str, nargs='+',
                        default=lsp['rules'],
                        help='Menu names. Tunable rules carry k inline '
                             "(e.g. 'ATC_2.0', 'COVERT_1.5'); non-tunable "
                             "names are unchanged. Validated via "
                             'src.HGT_dispatch.dispatching_rules_vec.parse_rule_name.')
    parser.add_argument('--rl_only', action='store_true',
                        help='Evaluate only rl, '
                             'skipping dispatching rules '
                             'and best_rule_oracle.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='result/exp_large_scale_dispatching')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default=None,
                        help='If set, output dir is <output_dir>/<run_name>/ '
                             '(no timestamp). Re-running the same command '
                             'resumes — already-completed records in runs.json '
                             'are skipped. If unset, a '
                             'fresh timestamp dir is created.')
    parser.add_argument('--validate_scenarios', action='store_true',
                        help='Run lightweight scenario schema/dispatching '
                             'smoke checks and exit.')
    args = parser.parse_args()
    repro = set_global_seed(args.seed)
    if args.rl_only:
        args.rules = []

    if args.validate_scenarios:
        validate_scenarios(device=args.device, seed=args.seed)
        return

    base_output_dir, _ = resolve_run_dir(args.output_dir, args.run_name)

    # Device resolution (scenario-independent) so the model loads once and is
    # reused across every scenario.
    device = args.device or exp_config.common_params['device']
    if 'cuda' in str(device) and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        device = 'cpu'

    # Source-of-truth: weights come from the model, not config.py.
    policy, model_hparams, model_weights = load_model(args.model_path, device)
    makespan_weight = model_weights['makespan_weight']
    twt_weight = model_weights['twt_weight']


    # Sweep every scenario unless one was pinned on the CLI. Each writes to its
    # own <base_output_dir>/<scenario>/ subdir (independent resume logs).
    scenarios = ([args.scenario] if args.scenario
                 else list(lsp['scenarios'].keys()))
    print(f"\nScenarios to run: {scenarios}")
    for scenario in scenarios:
        _run_one_scenario(
            args, scenario, lsp, base_output_dir,
            policy=policy, model_hparams=model_hparams,
            makespan_weight=makespan_weight, twt_weight=twt_weight,
                device=device, repro=repro,
        )

    print(f"\nAll scenarios complete. Base output dir: {base_output_dir}")


if __name__ == '__main__':
    main()
