#!/usr/bin/env python3
"""Static FJSP scale comparison: RL and CP-SAT on shared instances.

Each size uses ten cases by default. All methods consume the same static
instance through ``InferenceEnv.run_with_solver``. Objective weights come
from the model checkpoint. CP-SAT uses ``--cp_time_limit`` and
``--cp_num_workers``; ``--skip_cp`` evaluates only the policies. Missing
solutions retain empty metrics and an explicit status.

Usage:
    python evaluation/exp_scale_stress_test.py --size_configs 10x5 20x10
"""

import hashlib
import os
import sys
import json
import time
import csv
import argparse
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm

_MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '__pycache__', 'matplotlib')
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', _MPLCONFIGDIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.HGT.inference_env import InferenceEnv
from src.data_generator import DataGenerator
from src.benchmark.solver_adapter import (
    make_cp_solver_fn,
    make_greedy_solver_fn,
)
from src.utils.gantt_util import extract_schedule_from_env
from src.utils.model_utils import load_model
from src.utils.method_labels import parse_canonical_label
from src.utils.reproducibility import (
    derive_seed, seed_context, set_global_seed,
)
from src.utils.resumable_log import ResumableLog, resolve_run_dir
from src.utils.config_util import create_inference_config
from config import experiment_config as exp_config


# ============================================================
# Stress test helpers
# ============================================================

def aggregate_by_key(rows: List[Dict], group_key: str) -> Dict:
    """Group rows by key and compute mean/std for numeric fields."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    result = {}
    for key_val, group_rows in groups.items():
        agg = {group_key: key_val, 'count': len(group_rows)}
        # Union over ALL rows, not just row 0: a solver that timed out (or was
        # absent from an older resume record) on the first run would otherwise
        # never be aggregated, and would silently vanish from the plots while
        # still appearing in the CSVs.
        numeric_keys = []
        for row in group_rows:
            for k, v in row.items():
                if k in (group_key, 'run_idx') or k in numeric_keys:
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_keys.append(k)
        for nk in numeric_keys:
            vals = [r[nk] for r in group_rows if r.get(nk) is not None]
            if vals:
                agg[f'{nk}_mean'] = np.mean(vals)
                agg[f'{nk}_std'] = np.std(vals)
            else:
                agg[f'{nk}_mean'] = None
                agg[f'{nk}_std'] = None
        result[key_val] = agg
    return result


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
    makespan_weight: float,
    twt_weight: float,
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

    makespan = solver_result.get('makespan')
    twt = solver_result.get('twt')
    objective = solver_result.get('objective')
    if objective is None and makespan is not None and twt is not None:
        objective = makespan_weight * makespan + twt_weight * twt

    return _jsonable({
        'method': method,
        'status': solver_result.get('status', 'OK'),
        'objective': objective,
        'makespan': makespan,
        'twt': twt,
        'total_time': total_time,
        'num_segments': solver_result.get('num_segments'),
        'num_jobs': schedule_data.get('num_jobs'),
        'num_machines': schedule_data.get('num_machines'),
        'schedule': schedule_data.get('schedule', []),
        'job_data': schedule_data.get('job_data', []),
        'breakdown_periods': schedule_data.get('breakdown_periods', []),
        'interrupted_ops': schedule_data.get('interrupted_ops', []),
    })


# ============================================================
# Core evaluation through InferenceEnv
# ============================================================

def evaluate_single_instance(
    policy,
    config,
    dynamic_data: Dict[str, Any],
    makespan_weight: float,
    twt_weight: float,
    cp_time_limit: int,
    cp_num_workers: int,
    cp_time_scale: int,
    verbose: int = 0,
    collect_snapshots: bool = False,
    run_seed: Optional[int] = None,
    skip_cp: bool = False,
) -> Dict[str, Any]:
    """Run RL and CP-SAT on the same static problem and return metrics and timing."""
    inf = InferenceEnv(
        config.n_j, config.n_m, device=config.device,
        use_pt_nodes=getattr(config, 'use_pt_nodes', True),
        use_dynamic_event_features=getattr(
            config, 'use_dynamic_event_features', True))

    # --- RL variant: deterministic argmax reference ---
    rl_solver_factories = {
        'rl': lambda per_seg: make_greedy_solver_fn(
            policy=policy, device=config.device, per_segment_times=per_seg,
        ),
    }
    rl_results: Dict[str, Dict[str, Any]] = {}
    schedule_records: Dict[str, Dict[str, Any]] = {}
    for name, factory in rl_solver_factories.items():
        per_segment_times: List[float] = []
        t0 = time.time()
        res = inf.run_with_solver(
            factory(per_segment_times),
            dynamic_data=dynamic_data,
            makespan_weight=makespan_weight, twt_weight=twt_weight,
            collect_snapshots=collect_snapshots,
        )
        rl_results[name] = {
            'objective': res['objective'],
            'makespan': res['makespan'],
            'twt': res['twt'],
            'total_time': time.time() - t0,
        }
        schedule_records[name] = _schedule_record(
            name, res, rl_results[name]['total_time'],
            makespan_weight, twt_weight,
        )


    # --- CP-SAT ---
    cp: Dict[str, Any] = {}
    cp_total_time = None
    if not skip_cp:
        cp_per_segment_times: List[float] = []
        cp_t0 = time.time()
        cp = inf.run_with_solver(
            make_cp_solver_fn(
                time_limit=cp_time_limit,
                num_workers=cp_num_workers,
                time_scale=cp_time_scale,
                verbose=verbose,
                per_segment_times=cp_per_segment_times,
                seed=derive_seed(run_seed, 'cp') if run_seed is not None else None,
            ),
            dynamic_data=dynamic_data,
            makespan_weight=makespan_weight, twt_weight=twt_weight,
            collect_snapshots=collect_snapshots,
        )
        cp_total_time = time.time() - cp_t0
        schedule_records['cp'] = _schedule_record(
            'cp', cp, cp_total_time, makespan_weight, twt_weight)

    out: Dict[str, Any] = {
        'cp_objective': cp.get('objective'),
        'cp_makespan': cp.get('makespan'),
        'cp_twt': cp.get('twt'),
        'cp_total_time': cp_total_time,
        'cp_status': cp.get('status', 'OK') if not skip_cp else None,
        '_schedules': schedule_records,
    }
    for name, r in rl_results.items():
        out[f'{name}_objective'] = r['objective']
        out[f'{name}_makespan'] = r['makespan']
        out[f'{name}_twt'] = r['twt']
        out[f'{name}_total_time'] = r['total_time']
    return out


# ============================================================
# Sweep
# ============================================================

def _make_config(args, n_j: int, n_m: int):
    """Build a per-config inference Namespace (static — zero dynamic events)."""
    config = create_inference_config(
        n_j=n_j, n_m=n_m,
        num_new_job_arrivals=0,
        num_machine_breakdowns=0,
        num_priority_changes=0,
        device=args.device,
        seed=args.seed,
    )
    config.use_pt_nodes = getattr(args, 'use_pt_nodes', True)
    # Match the checkpoint's design so the inference env's feature dims line up
    # with the trained network (old/static design omits is_broken + released_time).
    config.use_dynamic_event_features = getattr(
        args, 'use_dynamic_event_features', True)
    if 'cuda' in config.device and not torch.cuda.is_available():
        config.device = 'cpu'
        config.common_params['device'] = 'cpu'
    return config


_RUN_RAW_KEYS = {
    'run_seed',
    'instance_seed',
    'data_params',
}


_LEGACY_METHOD_PREFIXES = (
    ('rl_greedy_', 'rl_'),
)


def _normalize_legacy_method_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map pre-rename result keys from resumable logs into current columns."""
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        new_key = key
        for old_prefix, new_prefix in _LEGACY_METHOD_PREFIXES:
            if key.startswith(old_prefix):
                new_key = f'{new_prefix}{key[len(old_prefix):]}'
                break
        normalized[new_key] = value
    return normalized


def _metric_row_from_payload(
    payload: Dict[str, Any],
    run_idx: int,
    n_j: int,
    n_m: int,
    size_str: Optional[str],
) -> Dict[str, Any]:
    payload = _normalize_legacy_method_keys(payload)
    row = {k: v for k, v in payload.items() if k not in _RUN_RAW_KEYS}
    row['run_idx'] = run_idx
    if size_str is not None:
        row['n_j'] = n_j
        row['n_m'] = n_m
        row['size'] = size_str
    return row


def _methods_missing_from_payload(payload: Dict[str, Any], args) -> List[str]:
    """Requested solvers that a cached resume record carries no result for.

    ``ResumableLog`` keys on (size, run_idx) only, so a record is replayed
    regardless of which methods produced it. Without this check, running once
    with ``--skip_cp`` and then again without it silently returns the cached
    CP-less record and never solves a single CP instance.
    """
    optional = [('cp', getattr(args, 'skip_cp', False))]
    missing = []
    for prefix, skipped in optional:
        if skipped:
            continue
        if payload.get(f'{prefix}_objective') is None and \
                payload.get(f'{prefix}_status') is None:
            missing.append(prefix)
    return missing


_CKPT_FINGERPRINTS: Dict[str, str] = {}


def _checkpoint_fingerprint(path: Optional[str]) -> Optional[str]:
    """Content hash of a checkpoint, memoised per path.

    Hashing the BYTES rather than trusting the path: a checkpoint that was
    overwritten in place (which every training run does to ``best_*.pth``)
    keeps its path, so a path-only check would call a different model the same
    model and replay its cached numbers.
    """
    if not path:
        return None
    if path not in _CKPT_FINGERPRINTS:
        try:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
            _CKPT_FINGERPRINTS[path] = h.hexdigest()[:16]
        except OSError:
            _CKPT_FINGERPRINTS[path] = None
    return _CKPT_FINGERPRINTS[path]


def _cache_provenance_mismatch(
    payload: Dict[str, Any],
    args,
    instance_seed: int,
) -> Optional[str]:
    """Name the first thing a cached record disagrees with, or None.

    ``ResumableLog`` keys on (size, run_idx) only, so nothing stops a record
    produced under a different ``--seed`` or a different checkpoint from being
    replayed while ``set_metadata`` rewrites the metadata to claim the current
    ones. Records written before this check exist and carry no fingerprint;
    those are accepted (``None`` means "not recorded", not "different").
    """
    cached_seed = payload.get('instance_seed')
    if cached_seed is not None and int(cached_seed) != int(instance_seed):
        return 'instance seed'
    cached_fp = payload.get('model_fingerprint')
    if cached_fp is not None:
        if cached_fp != _checkpoint_fingerprint(getattr(args, 'model_path', None)):
            return 'model checkpoint'
    return None


def _merge_preserving_cached(
    cached: Optional[Dict[str, Any]],
    fresh: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay a freshly solved payload on a cached one, keeping skipped work.

    ``ResumableLog.replace`` rewrites the entire record, so a run re-solved
    because one solver was missing would blank out the columns of every solver
    the current invocation skips. Fresh values always win; a cached value is
    kept only where the fresh payload has nothing (``None``).

    Guarded on ``instance_seed`` so two records can never be stitched together
    across a change that would alter the instance.
    """
    if not cached:
        return fresh
    c_seed, f_seed = cached.get('instance_seed'), fresh.get('instance_seed')
    if c_seed is not None and f_seed is not None and c_seed != f_seed:
        return fresh
    merged = dict(fresh)
    for k, v in cached.items():
        if v is not None and merged.get(k) is None:
            merged[k] = v
    return merged


def _build_run_payload(
    result: Dict[str, Any],
    config,
    run_seed: int,
    instance_seed: int,
    model_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {k: v for k, v in result.items()
               if not k.startswith('_') and k != 'run_idx'}
    payload.update({
        'run_seed': int(run_seed),
        'instance_seed': int(instance_seed),
        'data_params': _jsonable(dict(config.problem_params)),
        # Provenance, so a resume can tell whether a cached row belongs to the
        # model and seed the current invocation is actually running.
        'model_fingerprint': model_fingerprint,
    })
    return _jsonable(payload)


def _build_schedule_payload(
    result: Dict[str, Any],
    dynamic_data: Dict[str, Any],
    config,
    run_seed: int,
    instance_seed: int,
) -> Dict[str, Any]:
    return _jsonable({
        'size': result.get('size'),
        'n_j': result.get('n_j'),
        'n_m': result.get('n_m'),
        'run_idx': result.get('run_idx'),
        'run_seed': int(run_seed),
        'instance_seed': int(instance_seed),
        'data_params': dict(config.problem_params),
        'instance_data': dynamic_data,
        'schedules': result.get('_schedules', {}),
    })


def _evaluate_runs(
    policy, args, config, makespan_weight, twt_weight,
    num_runs: int,
    log: Optional[ResumableLog] = None,
    size_str: Optional[str] = None,
    schedule_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run num_runs instances and persist completed records for resume."""
    data_gen = DataGenerator(config)
    n_j, n_m = config.n_j, config.n_m
    out = []
    for run_idx in range(num_runs):
        run_seed = derive_seed(args.seed, size_str or f'{n_j}x{n_m}', run_idx)
        instance_seed = derive_seed(run_seed, 'instance')
        key = {"size": size_str, "run_idx": run_idx} if (log is not None and size_str) else None
        cached_payload = None
        if key is not None and log.is_done(key):
            payload = dict(log.get(key))
            # The resume key is only (size, run_idx), so a record written under
            # a DIFFERENT --seed or --model_path would otherwise be replayed
            # verbatim while set_metadata() rewrites the metadata to claim the
            # current ones. Treat provenance drift as a cache MISS and re-solve
            # from scratch rather than silently mixing runs; do NOT keep it as
            # cached_payload either, or the merge would carry the stale run's
            # columns into the new record.
            stale = _cache_provenance_mismatch(payload, args, instance_seed)
            if stale:
                print(f"  [resume] {size_str} run {run_idx}: cached record is "
                      f"from a different {stale}; re-solving from scratch")
            else:
                cached_payload = payload
                missing = _methods_missing_from_payload(payload, args)
                if missing:
                    # A record written by an earlier run that skipped a solver
                    # would otherwise be replayed forever and that solver would
                    # never run — while the metadata, rewritten on every
                    # invocation, claimed it had.
                    print(f"  [resume] {size_str} run {run_idx}: re-solving, "
                          f"cached record has no {', '.join(missing)} result")
                else:
                    out.append(_metric_row_from_payload(
                        payload, run_idx, n_j, n_m, size_str))
                    continue

        with seed_context(instance_seed):
            dyn_data = data_gen.sample_env(num_envs=1)
        result = evaluate_single_instance(
            policy, config, dyn_data,
            makespan_weight=makespan_weight, twt_weight=twt_weight,
            cp_time_limit=args.cp_time_limit,
            cp_num_workers=args.cp_num_workers,
            cp_time_scale=args.cp_time_scale,
            skip_cp=args.skip_cp,
            verbose=args.verbose,
            collect_snapshots=False,
            run_seed=run_seed,
        )
        result['run_idx'] = run_idx
        if size_str is not None:
            result['n_j'] = n_j
            result['n_m'] = n_m
            result['size'] = size_str

        if key is not None:
            payload = _build_run_payload(
                result, config, run_seed, instance_seed,
                model_fingerprint=_checkpoint_fingerprint(
                    getattr(args, 'model_path', None)))
            # Replacing an incomplete record preserves existing results
            # for methods skipped during this invocation.
            log.replace(key, _merge_preserving_cached(cached_payload, payload))
        if schedule_records is not None:
            schedule_records.append(_build_schedule_payload(
                result, dyn_data, config, run_seed, instance_seed))

        out.append(_strip_viz([result])[0])
    return out


def run_size_sweep(policy, args, makespan_weight, twt_weight, output_dir,
                   log: Optional[ResumableLog] = None,
                                  schedule_records: Optional[List[Dict[str, Any]]] = None):
    """Sweep problem sizes; always static."""
    rows = []

    for size_str in tqdm(args.size_configs, desc="Size sweep"):
        n_j, n_m = map(int, size_str.split('x'))

        slow_solvers = [] if args.skip_cp else ['CP-SAT']
        if n_j >= 50 and slow_solvers:
            print(f"  [WARNING] {'/'.join(slow_solvers)} may be very slow for "
                  f"{size_str} and will likely hit the time limit.")

        config = _make_config(args, n_j, n_m)

        results = _evaluate_runs(
            policy, args, config, makespan_weight, twt_weight,
            args.num_runs, log=log, size_str=size_str,
            schedule_records=schedule_records,
        )
        for r in results:
            r['n_j'] = n_j
            r['n_m'] = n_m
            r['size'] = size_str
            rows.append(r)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = os.path.join(output_dir, 'size_sweep.csv')
    _save_rows_long_csv(_strip_viz(rows), csv_path)
    return rows


# ============================================================
# Visualization helpers
# ============================================================

def _strip_viz(rows: List[Dict]) -> List[Dict]:
    """Drop env/snapshot fields before serialization."""
    return [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]


_LONG_CSV_FIELDS = [
    'size', 'n_j', 'n_m', 'run_idx', 'method',
    'objective', 'makespan', 'twt', 'total_time', 'status',
]

# (in-memory wide prefix, canonical CSV label) for each method.
def _method_specs() -> List[tuple]:
    return [
        ('rl',  'rl'),
        ('cp',         'cp'),
    ]


def _save_rows_long_csv(wide_rows: List[Dict], csv_path: str):
    """Flatten wide rows to long format: one CSV row per (size, run_idx, method).

    Always writes all method rows per instance. If a method was skipped
    or failed, its metric cells are empty (None -> ''); ``status`` carries the
    solver status for CP-SAT and is blank elsewhere."""
    if not wide_rows:
        return

    def _fmt(v):
        if v is None:
            return ''
        if isinstance(v, float):
            return f'{v:.4f}'
        return v

    specs = _method_specs()
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_LONG_CSV_FIELDS)
        writer.writeheader()
        for w in wide_rows:
            base = {k: w.get(k, '') for k in ('size', 'n_j', 'n_m', 'run_idx')}
            for prefix, method in specs:
                row = dict(base)
                row['method'] = method
                row['objective']  = _fmt(w.get(f'{prefix}_objective'))
                row['makespan']   = _fmt(w.get(f'{prefix}_makespan'))
                row['twt']        = _fmt(w.get(f'{prefix}_twt'))
                row['total_time'] = _fmt(w.get(f'{prefix}_total_time'))
                row['status']     = w.get(f'{prefix}_status', '') or ''
                writer.writerow(row)


# ============================================================
# Plot functions
# ============================================================

METHOD_COLORS = {
    'rl':                '#9C27B0',
    'cp':                '#2196F3',
}
METHOD_MARKERS = {
    'rl':                'D',
    'cp':                'o',
}
METHOD_LINESTYLES = {
    'rl':                '-',
    'cp':                (0, (5, 1)),
}


def _palette_key(method: str) -> str:
    """Family name without parameter suffix."""
    kind, _ = parse_canonical_label(method)
    return kind


def _row_prefix(method: str) -> str:
    """Map a canonical method label to the wide row-dict column prefix.

    Legacy CSV labels such as ``rl_greedy`` are accepted and normalized to the
    current ``rl`` prefix.
    """
    kind, _ = parse_canonical_label(method)
    return kind


def _get_methods() -> List[str]:
    """Canonical method labels in summary-table order."""
    return [
        'rl', 'cp',
    ]


def _get_plot_methods() -> List[str]:
    """Canonical method labels in plot/legend order."""
    return ['rl', 'cp']


def _has_valid_solution(agg_row: Dict[str, Any], method: str) -> bool:
    """True when the aggregate has at least one solved objective for method."""
    prefix = _row_prefix(method)
    obj_m = agg_row.get(f'{prefix}_objective_mean')
    return obj_m is not None and obj_m > 0


def _sorted_sizes(agg: Dict) -> list:
    return sorted(agg.keys(), key=lambda s: int(s.split('x')[0]) * int(s.split('x')[1]))


def plot_size_sweep_time(agg, size_order, output_dir):
    methods = _get_plot_methods()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        key = f'{_row_prefix(method)}_total_time'
        means = [agg[s].get(f'{key}_mean') for s in size_order]
        valid = [
            (i, m) for i, (s, m) in enumerate(zip(size_order, means))
            if m is not None and m > 0 and _has_valid_solution(agg[s], method)
        ]
        if not valid:
            continue
        idxs, ms = zip(*valid)
        pk = _palette_key(method)
        ax.plot(idxs, ms, color=METHOD_COLORS[pk], marker=METHOD_MARKERS[pk],
                linestyle=METHOD_LINESTYLES[pk], label=method, linewidth=2,
                markersize=6)

    ax.set_xlabel('Problem Size (Jobs x Machines)', fontsize=12)
    ax.set_ylabel('Solving Time (seconds, log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Solving Time vs Problem Size', fontsize=13)
    ax.set_xticks(range(len(size_order)))
    ax.set_xticklabels(size_order)
    ax.legend(fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'size_sweep_time.png'), dpi=150)
    plt.close(fig)


def plot_size_sweep_objective(agg, size_order, output_dir):
    """Average weighted objective by problem size.

    Objective values are plotted directly because the paper table reports the
    underlying makespan and TWT components separately.
    """
    methods = _get_plot_methods()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        key = f'{_row_prefix(method)}_objective'
        means = [agg[s].get(f'{key}_mean') for s in size_order]
        valid = [
            (i, m) for i, m in enumerate(means)
            if m is not None and m > 0
        ]
        if not valid:
            continue
        idxs, ms = zip(*valid)
        pk = _palette_key(method)
        ax.plot(idxs, ms, color=METHOD_COLORS[pk], marker=METHOD_MARKERS[pk],
                linestyle=METHOD_LINESTYLES[pk], label=method, linewidth=2,
                markersize=6)

    ax.set_xlabel('Problem Size (Jobs x Machines)', fontsize=12)
    ax.set_ylabel('Average Objective (lower is better)', fontsize=12)
    ax.set_title('Average Objective vs Problem Size', fontsize=13)
    ax.set_xticks(range(len(size_order)))
    ax.set_xticklabels(size_order)
    ax.legend(fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'size_sweep_objective.png'), dpi=150)
    plt.close(fig)


# ============================================================
# CSV loader for --replot_from_csv mode
# ============================================================

def _load_rows_from_csv(csv_path: str):
    """Read the long-format size_sweep.csv and pivot back to the wide row-dict
    shape that aggregate_by_key/print_and_save_results_table/the plotters
    consume."""
    by_key: Dict[tuple, Dict[str, Any]] = {}

    def _parse_num(v: str):
        if v == '' or v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return v

    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            size = raw['size']
            n_j = int(raw['n_j'])
            n_m = int(raw['n_m'])
            run_idx = int(raw['run_idx'])
            method = raw['method']

            kind, _ = parse_canonical_label(method)
            prefix = kind

            wide = by_key.setdefault((size, run_idx), {
                'size': size, 'n_j': n_j, 'n_m': n_m, 'run_idx': run_idx,
            })
            wide[f'{prefix}_objective']  = _parse_num(raw.get('objective', ''))
            wide[f'{prefix}_makespan']   = _parse_num(raw.get('makespan', ''))
            wide[f'{prefix}_twt']        = _parse_num(raw.get('twt', ''))
            wide[f'{prefix}_total_time'] = _parse_num(raw.get('total_time', ''))
            if prefix == 'cp':
                status = raw.get('status', '') or ''
                wide[f'{prefix}_status'] = status if status else None

    rows = list(by_key.values())
    return rows


def _emit_all_plots(agg, size_order, output_dir):
    """Emit the requested scale-stress plots only."""
    plot_size_sweep_objective(agg, size_order, output_dir)
    plot_size_sweep_time(agg, size_order, output_dir)


def _available_methods(rows: List[Dict], methods: List[str]) -> List[str]:
    """Keep methods that actually appear in a loaded historical run."""
    available = []
    for method in methods:
        prefix = _row_prefix(method)
        metric_cols = [
            f'{prefix}_objective',
            f'{prefix}_makespan',
            f'{prefix}_twt',
            f'{prefix}_total_time',
        ]
        if any(any(row.get(col) is not None for col in metric_cols)
               for row in rows):
            available.append(method)
    return available


def _run_replot(run_dir: str):
    """Regenerate requested plots from an existing size_sweep.csv."""
    csv_path = os.path.join(run_dir, 'size_sweep.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    rows = _load_rows_from_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    agg = aggregate_by_key(rows, 'size')
    size_order = _sorted_sizes(agg)

    print_sweep_summary(agg, size_order)
    methods = _available_methods(rows, _get_methods())
    try:
        print_and_save_results_table(
            rows, size_order, methods, run_dir,
        )
    except PermissionError as exc:
        print(f"[WARN] Could not refresh summary table files: {exc}")
    _emit_all_plots(agg, size_order, run_dir)

    print(f"\n{'='*70}")
    print(f"Replotted into: {run_dir}")
    print(f"{'='*70}")


# ============================================================
# Per-(scale, method) summary table
# ============================================================

_TABLE_METRICS = [
    ('objective', 'Objective'),
    ('makespan', 'Makespan'),
    ('twt', 'TWT'),
    ('total_time', 'Time(s)'),
]


def _format_cell(values: List[float], n_total: int) -> str:
    """Render one (metric, scale, method) cell.

    - All runs failed (n_valid == 0)              -> '-'
    - Partial failure (0 < n_valid < n_total)     -> 'mean ± std (n_valid/n_total)'
    - All runs valid                              -> 'mean ± std'
    """
    n_valid = len(values)
    if n_valid == 0:
        return '-'
    mean = float(np.mean(values))
    std = float(np.std(values))
    base = f"{mean:.2f} ± {std:.2f}"
    if n_valid < n_total:
        return f"{base} ({n_valid}/{n_total})"
    return base


def print_and_save_results_table(rows: List[Dict], size_order: List[str],
                                  methods: List[str], output_dir: str) -> None:
    """Build and emit a Scale | Method | Objective | Makespan | TWT | Time(s) table.

    Long format: one row per (scale, method) pair. Each metric cell is
    aggregated across runs with explicit handling of partial-failure cells
    (e.g., CP-SAT runs that returned no feasible solution).

    Saves a CSV (machine-readable) and also prints to console.
    """
    # Group raw rows by (size, method-prefix) so we can count total runs and
    # collect non-None values per metric.
    by_size: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_size[r['size']].append(r)

    table_rows: List[List[str]] = []
    header = ['Scale', 'Method'] + [label for _, label in _TABLE_METRICS]

    for size in size_order:
        size_rows = by_size.get(size, [])
        n_total = len(size_rows)
        for method in methods:
            prefix = _row_prefix(method)
            cells = [size, method]
            for metric_key, _label in _TABLE_METRICS:
                col = f"{prefix}_{metric_key}"
                values = [r[col] for r in size_rows if r.get(col) is not None]
                cells.append(_format_cell(values, n_total))
            table_rows.append(cells)

    # Console
    col_widths = [max(len(str(row[i])) for row in [header] + table_rows)
                  for i in range(len(header))]
    print(f"\n{'='*70}")
    print("  PER-(SCALE, METHOD) SUMMARY TABLE")
    print(f"{'='*70}")
    fmt_row = lambda r: '  ' + '  '.join(str(r[i]).ljust(col_widths[i]) for i in range(len(r)))
    print(fmt_row(header))
    print('  ' + '  '.join('-' * w for w in col_widths))
    for row in table_rows:
        print(fmt_row(row))
    print()

    # CSV
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table_rows)


# ============================================================
# Console summary
# ============================================================

def print_sweep_summary(agg, size_order):
    methods = _get_methods()

    print(f"\n{'='*70}")
    print("  SIZE SWEEP RESULTS")
    print(f"{'='*70}")

    for key_val in size_order:
        a = agg[key_val]
        print(f"\n  size={key_val} ({a['count']} runs):")
        for method in methods:
            prefix = _row_prefix(method)
            obj_m = a.get(f'{prefix}_objective_mean')
            obj_s = a.get(f'{prefix}_objective_std', 0)
            time_m = a.get(f'{prefix}_total_time_mean')
            if obj_m is not None:
                print(f"    {method:<14}: Obj={obj_m:.2f}+/-{obj_s:.2f}  Time={time_m:.2f}s")
            else:
                print(f"    {method:<14}: N/A")


def _scale_run_metadata(
    args,
    device: str,
    makespan_weight: float,
    twt_weight: float,
    model_hparams: Dict[str, Any],
    repro: Dict[str, Any],
) -> Dict[str, Any]:
    return _jsonable({
        'experiment': 'exp_scale_stress_test',
        'script': os.path.basename(__file__),
        'argv': sys.argv,
        'output_contract': {
            'plots': ['size_sweep_time.png', 'size_sweep_objective.png'],
            'tables': ['size_sweep.csv', 'summary_table.csv'],
            'resume_log': 'runs.json',
            'schedule_results': 'schedules.json',
        },
        'model_path': args.model_path,
        'model_hparams': model_hparams,
        'objective_weights': {
            'makespan_weight': makespan_weight,
            'twt_weight': twt_weight,
        },
        'size_configs': list(args.size_configs),
        'num_runs': int(args.num_runs),
        'seed': int(args.seed),
        'device': device,
        'use_pt_nodes': bool(args.use_pt_nodes),
        'use_dynamic_event_features': bool(
            getattr(args, 'use_dynamic_event_features', True)),
        'reproducibility': repro,
        'problem_param_defaults': dict(exp_config.problem_params),
        'static_event_overrides': {
            'num_new_job_arrivals': 0,
            'num_machine_breakdowns': 0,
            'num_priority_changes': 0,
        },
        'solver': {
            'cp_time_limit': args.cp_time_limit,
            'cp_num_workers': args.cp_num_workers,
            'cp_time_scale': args.cp_time_scale,
            'skip_cp': bool(args.skip_cp),
        },
    })


def save_schedule_results(records: List[Dict[str, Any]], output_path: str) -> None:
    """Save raw schedules and generated instances for downstream Gantt tools."""
    merged: Dict[tuple, Dict[str, Any]] = {}
    if os.path.isfile(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for rec in existing.get('records', []):
                key = (rec.get('size'), rec.get('run_idx'))
                merged[key] = rec
        except (OSError, json.JSONDecodeError):
            merged = {}

    for rec in records:
        schedules = rec.get('schedules')
        if not schedules:
            continue
        out_rec = {
            'size': rec.get('size'),
            'n_j': rec.get('n_j'),
            'n_m': rec.get('n_m'),
            'run_idx': rec.get('run_idx'),
            'run_seed': rec.get('run_seed'),
            'instance_seed': rec.get('instance_seed'),
            'data_params': rec.get('data_params'),
            'instance_data': rec.get('instance_data'),
            'schedules': schedules,
        }
        merged[(out_rec.get('size'), out_rec.get('run_idx'))] = out_rec

    schedule_records = sorted(
        merged.values(),
        key=lambda r: (str(r.get('size')), int(r.get('run_idx') or 0)),
    )
    payload = {
        'experiment': 'exp_scale_stress_test',
        'records': schedule_records,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(payload), f, indent=2)
        f.write('\n')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Static FJSP Scale Stress Test: RL vs CP-SAT')

    parser.add_argument('--model_path', type=str, default=exp_config.model_path,
                        help='Path to trained HGT model .pth file')

    parser.add_argument('--size_configs', type=str, nargs='+',
                        default=['10x5', '20x10', '50x10', '100x20'],
                        help='Problem sizes as NjxNm. Default grid: '
                             '10x5, 20x10, 50x10, 100x20.')

    parser.add_argument('--cp_time_limit', type=int,
                        default=exp_config.solver['cp_time_limit'],
                        help=f'CP-SAT wall-clock budget in seconds '
                             f'(default: {exp_config.solver["cp_time_limit"]})')
    parser.add_argument('--cp_num_workers', type=int,
                        default=exp_config.solver['cp_num_workers'],
                        help=f'CP-SAT search workers — the CP analogue of a '
                             f'thread count; 1 makes a run bit-reproducible '
                             f'(default: {exp_config.solver["cp_num_workers"]})')
    parser.add_argument('--cp_time_scale', type=int,
                        default=exp_config.solver['cp_time_scale'],
                        help=f'Integer ticks per time unit on CP-SAT\'s search '
                             f'grid (default: {exp_config.solver["cp_time_scale"]})')
    parser.add_argument('--skip_cp', action='store_true',
                        help='Skip the CP-SAT baseline.')

    parser.add_argument('--num_runs', type=int, default=10,
                        help='Cases per test size (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default=None, help='Device override')
    parser.add_argument('--output_dir', type=str, default='result/exp_scale_stress_test',
                        help='Output directory')
    parser.add_argument('--run_name', type=str, default=None,
                        help='If set, output dir is <output_dir>/<run_name>/ (no '
                             'timestamp). Re-running the same command resumes — '
                             'completed (size, run_idx) records in runs.json are '
                             'skipped. If unset, a fresh timestamp dir is created.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='CP-SAT search-log verbosity')
    parser.add_argument('--replot_from_csv', type=str, default=None,
                        help='Skip the solve loop: read size_sweep.csv from the '
                             'given run directory, regenerate size_sweep_time.png '
                             'and size_sweep_objective.png into the same directory, '
                             'and exit. Use this to '
                             'refresh plots on a previous run without re-solving.')

    args = parser.parse_args()
    repro = set_global_seed(args.seed)

    if args.replot_from_csv is not None:
        _run_replot(args.replot_from_csv)
        return

    output_dir, _ = resolve_run_dir(args.output_dir, args.run_name)

    first_size = args.size_configs[0]
    n_j0, n_m0 = map(int, first_size.split('x'))
    base_config = create_inference_config(
        n_j=n_j0, n_m=n_m0, device=args.device, seed=args.seed)
    device = base_config.device
    if 'cuda' in device and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        device = 'cpu'
        args.device = 'cpu'

    # Source-of-truth: load weights from the model checkpoint, not from config.
    policy, model_hparams, model_weights = load_model(args.model_path, device)
    makespan_weight = model_weights['makespan_weight']
    twt_weight = model_weights['twt_weight']
    args.use_pt_nodes = bool(model_hparams['use_pt_nodes'])
    # Legacy checkpoints without the key default to the new design (True).
    args.use_dynamic_event_features = bool(
        model_hparams.get('use_dynamic_event_features', True))


    rl_label = "RL"
    exact_solvers = [] if args.skip_cp else ['CP-SAT']
    print(f"Scale Stress Test: "
          f"{' vs '.join([rl_label] + exact_solvers)}")
    print(f"[loaded] Model trained with mw={makespan_weight}, tw={twt_weight}")
    print(f"Output: {output_dir}\n")

    log_path = os.path.join(output_dir, 'runs.json')
    log = ResumableLog(log_path)
    log.set_metadata(_scale_run_metadata(
        args, device, makespan_weight, twt_weight, model_hparams, repro))
    if log.num_completed() > 0:
        print(f"[resume] Skipping {log.num_completed()} completed records "
              f"from {log_path}\n")

    print("--- Problem Size Sweep (static) ---")
    schedule_records: List[Dict[str, Any]] = []
    try:
        rows = run_size_sweep(policy, args,
                              makespan_weight, twt_weight, output_dir,
                              log=log,
                              schedule_records=schedule_records)
    finally:
        log.close()

    agg = aggregate_by_key(_strip_viz(rows), 'size')
    size_order = _sorted_sizes(agg)

    print_sweep_summary(agg, size_order)
    # Same method filter as _run_replot, so a live run and a later
    # --replot_from_csv of that same directory produce identical
    # summary_table.csv instead of the replot silently dropping rows.
    print_and_save_results_table(
        _strip_viz(rows), size_order,
        _available_methods(_strip_viz(rows), _get_methods()), output_dir,
    )
    _emit_all_plots(agg, size_order, output_dir)
    save_schedule_results(schedule_records,
                          os.path.join(output_dir, 'schedules.json'))

    print(f"\n{'='*70}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
