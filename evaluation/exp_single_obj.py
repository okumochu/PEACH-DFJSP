#!/usr/bin/env python3
"""Greedy-only single-objective HGT evaluation against DANIEL Tables I and II.

Synthetic SD1/SD2 benchmark groups only:

  - Table I: SD1_<size> / SD2_<size>+mix instances use the model trained at
    that same size (10x5, 20x5, 15x10, 20x10).
  - Table II: SD1/SD2 30x10 and 40x10 instances are scored by BOTH the 10x5
    and the 20x10 model of the same generator (scale generalization).

Public benchmark groups (Brandimarte / Hurink, DANIEL Table III) and
stochastic sampling inference are still out of scope here.

Usage:
    # Default: use the eight paper checkpoints under ./models/checkpoints/
    python exp_single_obj.py --device cuda:1

    # Explicit checkpoints (overrides auto-discovery):
    python exp_single_obj.py \\
        --model_paths SD1/10x5=models/checkpoints/appendix-sd1-10x5-s42.pth \\
                      SD2/10x5=models/checkpoints/appendix-sd2-10x5-s42.pth

    # Resumable run: re-running the same --run_name picks up where it left off.
    python exp_single_obj.py --run_name exp_so_eval --device cuda:1

    # Quick smoke test on one in-distribution group:
    python exp_single_obj.py --model_paths SD1/10x5=models/checkpoints/appendix-sd1-10x5-s42.pth \\
        --benchmarks SD1_10x5

Output:
    - per_instance.csv / per_benchmark.csv
    - runs.jsonl with original inputs, complete policy schedules and verification
    - A DANIEL Table I-style greedy comparison printed to stdout.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.HGT_single_objective.greedy_inference import greedy_inference
from src.utils.data_util import text_to_matrix
from src.utils.gantt_util import extract_schedule_from_env
from src.utils.model_utils import load_single_obj_model, resolve_device
from src.utils.benchmark_gaps import REFERENCE_ROOT, reference_objectives, mean_instance_gap
from src.utils.reproducibility import set_global_seed
from src.utils.resumable_log import ResumableLog, checkpoint_fingerprint, resolve_run_dir


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DATA = os.path.join(PROJECT_ROOT, 'data', 'benchmarks')

# A model is identified by (generator, size), e.g. ('SD1', '10x5').
ModelKey = Tuple[str, str]
ParsedInstance = Tuple[int, int, np.ndarray, np.ndarray]


class IncompleteEvaluationEvidence(ValueError):
    """A historical cache cannot supply the evidence required for a new run."""


def verify_single_obj_record(record: Dict) -> Dict:
    """Recheck a saved input and full schedule without a policy or solver run."""
    if record.get('evidence_schema') != 1:
        raise IncompleteEvaluationEvidence(
            'Scalar-only or unsupported evaluation cache: use a new --run_name '
            'or --output_dir; historical scores will not be overwritten.')
    problem = record['problem']
    source = problem['fjs_text'].encode('utf-8')
    if hashlib.sha256(source).hexdigest() != problem['fjs_sha256']:
        raise ValueError('Saved FJS text does not match its SHA256')
    job_length, op_pt = text_to_matrix(problem['fjs_text'].splitlines())
    if (not np.array_equal(job_length, problem['job_length']) or
            not np.array_equal(op_pt, problem['op_pt']) or
            int(sum(job_length)) != op_pt.shape[0] or
            record['n_j'] != len(job_length) or record['n_m'] != op_pt.shape[1]):
        raise ValueError('Saved parsed instance differs from the original FJS input')
    if not record.get('model_fingerprint'):
        raise ValueError('A checkpoint fingerprint is required for evaluation evidence')
    by_machine = {}
    placed = {}
    job_of = np.repeat(np.arange(len(job_length)), job_length)
    for operation in record['schedule']:
        op_id = operation['op_id']
        if (not isinstance(op_id, int) or not 0 <= op_id < len(job_of) or
                operation['job_id'] != int(job_of[op_id])):
            raise ValueError('Invalid operation or job identity in saved schedule')
        start, end = operation['start_time'], operation['end_time']
        duration = operation['processing_time']
        machine = operation['machine_id']
        if (not isinstance(machine, int) or not 0 <= machine < op_pt.shape[1] or
                not np.all(np.isfinite([start, end, duration])) or start < -1e-6 or
                op_pt[op_id, machine] <= 0 or
                abs(duration - op_pt[op_id, machine]) > 1e-4 or
                abs(end - start - duration) > 1e-4):
            raise ValueError('Saved processing duration differs from the schedule')
        if op_id in placed:
            raise ValueError('Operation scheduled more than once')
        placed[op_id] = (start, end)
        by_machine.setdefault(machine, []).append((start, end))
    if set(placed) != set(range(len(job_of))):
        raise ValueError('Incomplete operation schedule')
    for op_id in range(1, len(job_of)):
        if job_of[op_id] == job_of[op_id - 1] and placed[op_id][0] < placed[op_id - 1][1] - 1e-6:
            raise ValueError('Job precedence violation')
    for operations in by_machine.values():
        ordered = sorted(operations)
        if any(b[0] < a[1] - 1e-6 for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Machine overlap')
    makespan = max(end for _, end in placed.values())
    if (not np.isfinite(record['makespan']) or
            abs(makespan - record['makespan']) > 1e-4 or
            not np.isfinite(record['time_s']) or record['time_s'] < 0):
        raise ValueError('Invalid saved objective or time')
    return {'feasible': True, 'operations': len(record['schedule']),
            'recomputed_makespan': makespan,
            'checker': 'evaluation.exp_single_obj.verify_single_obj_record'}


# DANIEL paper numbers (Wang et al. 2024, IEEE TNNLS, doi 10.1109/TNNLS.2023.3306421)
# Greedy-strategy DANIEL columns only; transcribed from reference/exp_table/.
#
# Table I (page 3098): model trained at the SAME size as the test set.
# Key: (benchmark, model_size, generator) -> (greedy_objective, greedy_gap_pct)
DANIEL_TABLE_I = {
    ('SD1_10x5',  '10x5',  'SD1'): (106.71, 10.87),
    ('SD1_20x5',  '20x5',  'SD1'): (197.56,  5.03),
    ('SD1_15x10', '15x10', 'SD1'): (161.28, 12.42),
    ('SD1_20x10', '20x10', 'SD1'): (198.50,  1.31),
    ('SD2_10x5',  '10x5',  'SD2'): (408.40, 25.68),
    ('SD2_20x5',  '20x5',  'SD2'): (671.03, 11.52),
    ('SD2_15x10', '15x10', 'SD2'): (591.21, 57.16),
    ('SD2_20x10', '20x10', 'SD2'): (610.16, 31.58),
}

# Table II (page 3099): model trained at 10x5 OR 20x10, tested on larger sizes.
# Same key shape as Table I.
DANIEL_TABLE_II = {
    ('SD1_30x10', '10x5',  'SD1'): (288.61,  5.10),
    ('SD1_30x10', '20x10', 'SD1'): (281.49,  2.50),
    ('SD1_40x10', '10x5',  'SD1'): (379.28,  3.65),
    ('SD1_40x10', '20x10', 'SD1'): (371.45,  1.52),
    ('SD2_30x10', '10x5',  'SD2'): (794.62, 14.85),
    ('SD2_30x10', '20x10', 'SD2'): (774.56, 11.95),
    ('SD2_40x10', '10x5',  'SD2'): (983.37,  0.52),
    ('SD2_40x10', '20x10', 'SD2'): (962.58, -1.67),
}

# OR-Tools reference (best obj, % solved to optimality at 1800s) from DANIEL
# Tables I and II. These aggregates are display-only; gaps use the paired
# per-instance objectives in data/benchmarks/daniel/or_solution.
ORTOOLS_REF = {
    'SD1_10x5':  (96.32,  5),
    'SD1_20x5':  (188.15, 0),
    'SD1_15x10': (143.53, 7),
    'SD1_20x10': (195.98, 0),
    'SD2_10x5':  (326.24, 96),
    'SD2_20x5':  (602.04, 0),
    'SD2_15x10': (377.17, 28),
    'SD2_20x10': (464.16, 1),
    'SD1_30x10': (274.67, 6),
    'SD1_40x10': (365.96, 3),
    'SD2_30x10': (692.26, 0),
    'SD2_40x10': (998.39, 0),
}


def _instance_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    files = sorted(f for f in os.listdir(directory) if f.endswith('.fjs'))
    return [os.path.join(directory, f) for f in files]


def discover_benchmarks() -> Dict[str, List[str]]:
    """Return synthetic benchmark groups: Table I sizes plus Table II sizes."""
    return {
        # Table I: same-size test sets.
        'SD1_10x5':  _instance_files(os.path.join(SOURCE_DATA, 'SD1', '10x5')),
        'SD1_20x5':  _instance_files(os.path.join(SOURCE_DATA, 'SD1', '20x5')),
        'SD1_15x10': _instance_files(os.path.join(SOURCE_DATA, 'SD1', '15x10')),
        'SD1_20x10': _instance_files(os.path.join(SOURCE_DATA, 'SD1', '20x10')),
        'SD2_10x5':  _instance_files(os.path.join(SOURCE_DATA, 'SD2', '10x5+mix')),
        'SD2_20x5':  _instance_files(os.path.join(SOURCE_DATA, 'SD2', '20x5+mix')),
        'SD2_15x10': _instance_files(os.path.join(SOURCE_DATA, 'SD2', '15x10+mix')),
        'SD2_20x10': _instance_files(os.path.join(SOURCE_DATA, 'SD2', '20x10+mix')),
        # Table II: larger-size test sets scored by the 10x5 and 20x10 models.
        'SD1_30x10': _instance_files(os.path.join(SOURCE_DATA, 'SD1', '30x10')),
        'SD1_40x10': _instance_files(os.path.join(SOURCE_DATA, 'SD1', '40x10')),
        'SD2_30x10': _instance_files(os.path.join(SOURCE_DATA, 'SD2', '30x10+mix')),
        'SD2_40x10': _instance_files(os.path.join(SOURCE_DATA, 'SD2', '40x10+mix')),
    }


BENCHMARK_MODELS: Dict[str, List[ModelKey]] = {
    'SD1_10x5':  [('SD1', '10x5')],
    'SD1_20x5':  [('SD1', '20x5')],
    'SD1_15x10': [('SD1', '15x10')],
    'SD1_20x10': [('SD1', '20x10')],
    'SD2_10x5':  [('SD2', '10x5')],
    'SD2_20x5':  [('SD2', '20x5')],
    'SD2_15x10': [('SD2', '15x10')],
    'SD2_20x10': [('SD2', '20x10')],
    'SD1_30x10': [('SD1', '10x5'), ('SD1', '20x10')],
    'SD1_40x10': [('SD1', '10x5'), ('SD1', '20x10')],
    'SD2_30x10': [('SD2', '10x5'), ('SD2', '20x10')],
    'SD2_40x10': [('SD2', '10x5'), ('SD2', '20x10')],
}


def _daniel_lookup(group: str, train_size: str, generator: str
                   ) -> Tuple[Optional[float], Optional[float]]:
    key = (group, train_size, generator)
    if key in DANIEL_TABLE_I:
        return DANIEL_TABLE_I[key]
    return DANIEL_TABLE_II.get(key, (None, None))


def discover_model_paths(model_dir: str) -> Dict[ModelKey, str]:
    """Find the eight packaged SD1/SD2 policies by generator and training size."""
    out: Dict[ModelKey, str] = {}
    for generator in ('SD1', 'SD2'):
        for size in ('10x5', '15x10', '20x5', '20x10'):
            path = os.path.join(
                model_dir, f'appendix-{generator.lower()}-{size}-s42.pth')
            if os.path.isfile(path):
                out[(generator, size)] = path
    return out


def parse_model_paths(specs: List[str]) -> Dict[ModelKey, str]:
    """Parse ['SD1/10x5=path1', 'SD2/10x5=path2'] into {(gen, size): path}."""
    out: Dict[ModelKey, str] = {}
    for spec in specs:
        if '=' not in spec or '/' not in spec.split('=', 1)[0]:
            raise ValueError(
                f'--model_paths expects <GEN>/<size>=<path> entries, got: {spec!r}')
        key, path = spec.split('=', 1)
        gen, size = key.split('/', 1)
        identity = (gen.strip().upper(), size.strip())
        if identity in out:
            raise ValueError(f'Duplicate model identity {identity}')
        out[identity] = path.strip()
    return out


def validate_model_identity(key, hp):
    """The public comparison label must identify the actual training policy."""
    gen, size = key
    expected = (hp.get('generator_type'), f"{hp.get('n_j')}x{hp.get('n_m')}")
    if expected != (gen, size):
        raise ValueError(f'Checkpoint training identity {expected} differs from {key}')


def validate_complete_comparisons(rows, expected):
    """Fail closed even when an entire requested group disappeared."""
    actual = [(r['generator'], r['model_size'], r['benchmark']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(f'Incomplete comparison matrix: expected {sorted(expected)}, got {sorted(actual)}')
    if any(r['n_instances'] != 100 for r in rows):
        raise ValueError('Every public comparison requires exactly 100 instances')


def evaluate_instance(policy, path: str, device: str, *,
                      log: ResumableLog, generator: str, model_size: str,
                      benchmark: str, instance: str,
                      parsed_cache: Dict[str, ParsedInstance],
                      model_fp: Optional[str] = None) -> Dict[str, float]:
    """Run one greedy rollout and retain independently checkable evidence."""
    with open(path, 'rb') as source:
        fjs_bytes = source.read()
    fjs_text = fjs_bytes.decode('utf-8')
    fjs_sha256 = hashlib.sha256(fjs_bytes).hexdigest()

    def _ensure_parsed() -> ParsedInstance:
        cache_key = f'{path}:{fjs_sha256}'
        cached = parsed_cache.get(cache_key)
        if cached is not None:
            return cached
        jl, pt = text_to_matrix(fjs_text.splitlines())
        parsed = (int(jl.shape[0]), int(pt.shape[1]), jl, pt)
        parsed_cache[cache_key] = parsed
        return parsed

    greedy_key = {
        'generator': generator,
        'model_size': model_size,
        'benchmark': benchmark,
        'instance': instance,
        'method': 'greedy',
        # In the KEY, not just the metadata: without it, re-running the sweep
        # with a different checkpoint replays the previous model's numbers.
        'model_fp': model_fp,
    }
    if log.is_done(greedy_key):
        cached = log.get(greedy_key)
        try:
            verify_single_obj_record(cached)
            if (cached['problem']['fjs_sha256'] != fjs_sha256 or
                    cached['model_fingerprint'] != model_fp):
                raise ValueError('Current input or model differs from saved evidence')
        except (ValueError, KeyError, TypeError) as exc:
            raise IncompleteEvaluationEvidence(
                f'{exc}; use a new --run_name or --output_dir. '
                'Existing results will not be overwritten.') from exc
        return {
            'n_j': cached['n_j'],
            'n_m': cached['n_m'],
            'greedy_makespan': cached['makespan'],
            'greedy_time_s': cached['time_s'],
            'greedy_cached': True,
        }

    if not model_fp:
        raise ValueError('A checkpoint fingerprint is required before evaluation')
    n_j, n_m, jl, pt = _ensure_parsed()
    result = greedy_inference(
        policy,
        n_j,
        n_m,
        jl,
        pt,
        device=device,
    )
    record = {
        'makespan': result['makespan'],
        'time_s': result['time_s'],
        'n_j': n_j,
        'n_m': n_m,
        'evidence_schema': 1,
        'model_fingerprint': model_fp,
        'problem': {'source_path': os.path.abspath(path), 'fjs_text': fjs_text,
                    'fjs_sha256': fjs_sha256, 'job_length': jl.tolist(),
                    'op_pt': pt.tolist()},
        'schedule': extract_schedule_from_env(
            result['best_env'], result['best_env_idx'])['schedule'],
    }
    record['verification'] = verify_single_obj_record(record)
    log.append(greedy_key, record)
    return {
        'n_j': n_j,
        'n_m': n_m,
        'greedy_makespan': result['makespan'],
        'greedy_time_s': result['time_s'],
        'greedy_cached': False,
    }


def _safe_gap_pct(makespan: Optional[float], ref: Optional[float]) -> Optional[float]:
    if makespan is None or ref is None or ref <= 0:
        return None
    return (makespan - ref) / ref * 100.0


def aggregate_group(rows: List[Dict[str, float]]
                    ) -> Tuple[Optional[float], Optional[float]]:
    """Return (avg_makespan, avg_time_s) for greedy rows."""
    ms_vals = [r['greedy_makespan'] for r in rows if r.get('greedy_makespan') is not None]
    t_vals = [r['greedy_time_s'] for r in rows if r.get('greedy_time_s') is not None]
    avg_ms = float(np.mean(ms_vals)) if ms_vals else None
    avg_t = float(np.mean(t_vals)) if t_vals else None
    return avg_ms, avg_t


def populate_synthetic_gaps(per_bench_rows: List[Dict], per_instance_rows: List[Dict],
                            *, reference_root=REFERENCE_ROOT, source_root=SOURCE_DATA) -> None:
    """Fill mean per-instance gaps only for complete, correctly paired groups."""
    grouped = {}
    for item in per_instance_rows:
        key = item['generator'], item['model_size'], item['benchmark']
        grouped.setdefault(key, []).append((item['instance'], item['greedy_makespan']))
    seen = set()
    for row in per_bench_rows:
        key = row['generator'], row['model_size'], row['benchmark']
        if key in seen or row['n_instances'] != 100:
            raise ValueError(f"Expected one complete 100-instance benchmark group {key}")
        seen.add(key)
        references = reference_objectives(row['generator'], row['benchmark'],
                                          reference_root=reference_root, source_root=source_root)
        row['avg_gap_pct'] = mean_instance_gap(grouped.get(key, []), references)
    if seen != set(grouped):
        raise ValueError("Measured and aggregate benchmark groups differ")


def populate_win_gaps(per_bench_rows: List[Dict]) -> None:
    """Fill win_gap_pct = daniel_gap_pct - avg_gap_pct in place."""
    for row in per_bench_rows:
        d_gap = row.get('daniel_gap_pct')
        ours_gap = row.get('avg_gap_pct')
        row['win_gap_pct'] = (
            d_gap - ours_gap if d_gap is not None and ours_gap is not None else None
        )


def write_per_instance_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = [
        'generator', 'model_size', 'benchmark', 'instance', 'n_j', 'n_m',
        'greedy_makespan', 'greedy_time_s',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def write_per_benchmark_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = [
        'generator', 'model_size', 'benchmark', 'method',
        'avg_makespan', 'avg_gap_pct', 'avg_time_s',
        'daniel_obj', 'daniel_gap_pct', 'win_gap_pct', 'n_instances',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def _cell_obj(x):
    return f'{x:>8.2f}' if x is not None else '   --   '


def _cell_gap(x):
    return f'{x:>+7.2f}%' if x is not None else '   --   '


def _cell_time(x):
    return f'{x:>7.2f}' if x is not None else '   --   '


def _ortools_cell(group):
    if group not in ORTOOLS_REF:
        return '       --      '
    obj, pct = ORTOOLS_REF[group]
    return f'{obj:>8.2f} ({pct:>2}%)'


TABLE_I_ROWS = [
    ('SD1_10x5',  'SD1 10x5',  'SD1', '10x5'),
    ('SD1_20x5',  'SD1 20x5',  'SD1', '20x5'),
    ('SD1_15x10', 'SD1 15x10', 'SD1', '15x10'),
    ('SD1_20x10', 'SD1 20x10', 'SD1', '20x10'),
    ('SD2_10x5',  'SD2 10x5',  'SD2', '10x5'),
    ('SD2_20x5',  'SD2 20x5',  'SD2', '20x5'),
    ('SD2_15x10', 'SD2 15x10', 'SD2', '15x10'),
    ('SD2_20x10', 'SD2 20x10', 'SD2', '20x10'),
]

TABLE_II_ROWS = [
    ('SD1_30x10', 'SD1 30x10', 'SD1', '10x5'),
    ('SD1_30x10', 'SD1 30x10', 'SD1', '20x10'),
    ('SD1_40x10', 'SD1 40x10', 'SD1', '10x5'),
    ('SD1_40x10', 'SD1 40x10', 'SD1', '20x10'),
    ('SD2_30x10', 'SD2 30x10', 'SD2', '10x5'),
    ('SD2_30x10', 'SD2 30x10', 'SD2', '20x10'),
    ('SD2_40x10', 'SD2 40x10', 'SD2', '10x5'),
    ('SD2_40x10', 'SD2 40x10', 'SD2', '20x10'),
]


def _print_one_table(title: str, rows, lookup) -> None:
    print('\n' + '=' * 92)
    print(title)
    print('=' * 92)
    header = (
        f'  {"Size":<12} {"Model":<10} {"DANIEL_Gr":>10} {"OURS_Gr":>10} '
        f'{"OURS_Gap":>10} {"WinGap":>10} {"OURS_Time":>10} {"OR-Tools":>16}'
    )
    print(header)
    print('  ' + '-' * (len(header) - 2))

    wins = []
    for group, test_label, gen, model_size in rows:
        ours = lookup.get((group, gen, model_size))
        d_obj, _d_gap = _daniel_lookup(group, model_size, gen)
        if ours and ours.get('win_gap_pct') is not None:
            wins.append(ours['win_gap_pct'])
        print(
            f'  {test_label:<12} {gen + "/" + model_size:<10} '
            f'{_cell_obj(d_obj)} '
            f'{_cell_obj(ours["avg_makespan"] if ours else None)} '
            f'{_cell_gap(ours["avg_gap_pct"] if ours else None)} '
            f'{_cell_gap(ours["win_gap_pct"] if ours else None)} '
            f'{_cell_time(ours["avg_time_s"] if ours else None)} '
            f'{_ortools_cell(group):>16}'
        )
    if wins:
        n_win = sum(1 for w in wins if w > 0)
        print('  ' + '-' * (len(header) - 2))
        print(f'  rows won: {n_win}/{len(wins)}   mean WinGap: {np.mean(wins):+.3f} pp')
    return wins


def print_paper_table(per_bench_rows: List[Dict]) -> None:
    lookup = {
        (r['benchmark'], r['generator'], r['model_size']): r
        for r in per_bench_rows
    }
    w1 = _print_one_table(
        'TABLE I: Greedy Performance on Same-Distribution Synthetic Data',
        TABLE_I_ROWS, lookup)
    w2 = _print_one_table(
        'TABLE II: Greedy Performance on Larger-Size Synthetic Data (scale generalization)',
        TABLE_II_ROWS, lookup)
    allw = list(w1) + list(w2)
    if allw:
        n_win = sum(1 for w in allw if w > 0)
        print('\n' + '=' * 92)
        print(f'COMBINED (Table I + II): rows won {n_win}/{len(allw)}   '
              f'mean WinGap {np.mean(allw):+.3f} pp')
        print('=' * 92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Greedy-only single-objective HGT benchmark evaluator')
    parser.add_argument('--model_paths', type=str, nargs='+', default=None,
                        help='<GEN>/<size>=<path> entries, e.g. '
                             'SD1/10x5=models/checkpoints/appendix-sd1-10x5-s42.pth. If omitted, '
                             'checkpoints are auto-discovered under --model_dir.')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'models', 'checkpoints'),
                        help='Directory containing appendix-<gen>-<size>-s42.pth '
                             'when --model_paths is omitted (default: ./models/checkpoints).')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'result', 'exp_single_obj'),
                        help='Base output directory; a subdir per run is created.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='If set, output dir is <output_dir>/<run_name>/ and '
                             'runs.jsonl is reused for resumability.')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=None,
                        help='Subset of same-distribution benchmark groups to evaluate.')
    parser.add_argument('--instance_filter', type=str, default=None,
                        help='Substring filter on instance filenames.')
    parser.add_argument('--reference_root', type=str, default=str(REFERENCE_ROOT),
                        help='DANIEL reference root containing or_solution; complete paired '
                             '100-instance groups are required for gap reporting.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_global_seed(args.seed)

    device = resolve_device(args.device)
    if args.model_paths:
        model_paths = parse_model_paths(args.model_paths)
    else:
        model_paths = discover_model_paths(args.model_dir)
        if not model_paths:
            raise SystemExit(
                f"No checkpoints found under {args.model_dir!r} matching "
                f"appendix-<gen>-<size>-s42.pth. "
                "Pass --model_paths explicitly.")
        print(f"[exp_single_obj] auto-discovered models under {args.model_dir}")

    print(f"[exp_single_obj] device={device}")
    for key, path in model_paths.items():
        print(f"[exp_single_obj] model {key[0]}/{key[1]} -> {path}")

    out_dir, is_resume = resolve_run_dir(args.output_dir, args.run_name)
    print(f"[exp_single_obj] output -> {out_dir}"
          + (" (resume candidate)" if is_resume else ""))

    policies: Dict[ModelKey, torch.nn.Module] = {}
    for key, path in model_paths.items():
        gen, size = key
        print(f"[exp_single_obj] loading {gen}/{size} from {path}")
        policy, hp, _ow = load_single_obj_model(path, device)
        validate_model_identity(key, hp)
        policies[key] = policy

    groups = discover_benchmarks()
    if args.benchmarks:
        selected = set(args.benchmarks)
        if not selected <= set(BENCHMARK_MODELS):
            raise ValueError('Unknown benchmark group requested')
        groups = {k: v for k, v in groups.items() if k in selected}

    requested_groups = set(args.benchmarks or BENCHMARK_MODELS)
    expected = {(gen, size, group) for group in requested_groups
                for gen, size in BENCHMARK_MODELS[group]}
    required_policies = {(gen, size) for gen, size, _ in expected}
    if not required_policies <= set(policies):
        raise ValueError(f'Missing required policies: {sorted(required_policies - set(policies))}')
    if set(groups) != requested_groups:
        raise ValueError('Missing requested benchmark fixture directory')

    if args.instance_filter:
        needle = args.instance_filter.lower()
        groups = {
            k: [p for p in v if needle in os.path.basename(p).lower()]
            for k, v in groups.items()
        }

    log = ResumableLog(os.path.join(out_dir, 'runs.jsonl'))
    per_instance_rows: List[Dict] = []
    per_bench_rows: List[Dict] = []
    parsed_cache: Dict[str, ParsedInstance] = {}

    try:
        if any(record.get('evidence_schema') != 1 for record in log.all_records()):
            raise IncompleteEvaluationEvidence(
                'Scalar-only or unsupported evaluation cache: use a new --run_name '
                'or --output_dir; historical scores will not be overwritten.')
        if log.num_completed() > 0:
            print(f"[resume] Checking {log.num_completed()} saved records from {log.path}\n")
        for group_name, paths in groups.items():
            if not paths:
                continue
            paired_keys = BENCHMARK_MODELS[group_name]
            keys_to_run = [key for key in paired_keys if key in policies]

            for key in keys_to_run:
                gen, model_size = key
                policy = policies[key]
                print(f"\n[{group_name}] {gen}/{model_size} | {len(paths)} instances")

                rows_this: List[Dict] = []
                for inst_path in paths:
                    inst_name = os.path.splitext(os.path.basename(inst_path))[0]
                    t_inst = time.time()
                    try:
                        result = evaluate_instance(
                            policy,
                            inst_path,
                            device=device,
                            log=log,
                            generator=gen,
                            model_size=model_size,
                            benchmark=group_name,
                            instance=inst_name,
                            parsed_cache=parsed_cache,
                            model_fp=checkpoint_fingerprint(model_paths.get(key)),
                        )
                    except IncompleteEvaluationEvidence:
                        raise
                    except Exception as exc:
                        print(f"  [skip] {inst_name}: {exc}")
                        continue

                    row = {
                        'generator': gen,
                        'model_size': model_size,
                        'benchmark': group_name,
                        'instance': inst_name,
                        'n_j': result['n_j'],
                        'n_m': result['n_m'],
                        'greedy_makespan': result['greedy_makespan'],
                        'greedy_time_s': result['greedy_time_s'],
                    }
                    rows_this.append(row)
                    per_instance_rows.append(row)
                    elapsed = time.time() - t_inst
                    cache_suffix = ' (cached)' if result.get('greedy_cached') else ''
                    print(
                        f"  {inst_name}: greedy_ms={row['greedy_makespan']:.1f} "
                        f"[{elapsed:.1f}s]{cache_suffix}"
                    )

                if not rows_this:
                    continue
                avg_ms, avg_t = aggregate_group(rows_this)
                d_obj, d_gap = _daniel_lookup(group_name, model_size, gen)
                per_bench_rows.append({
                    'generator': gen,
                    'model_size': model_size,
                    'benchmark': group_name,
                    'method': 'greedy',
                    'avg_makespan': avg_ms,
                    'avg_gap_pct': None,
                    'avg_time_s': avg_t,
                    'daniel_obj': d_obj,
                    'daniel_gap_pct': d_gap,
                    'win_gap_pct': None,
                    'n_instances': len(rows_this),
                })
    finally:
        log.close()

    per_inst_path = os.path.join(out_dir, 'per_instance.csv')
    per_bench_path = os.path.join(out_dir, 'per_benchmark.csv')
    write_per_instance_csv(per_inst_path, per_instance_rows)
    validate_complete_comparisons(per_bench_rows, expected)
    populate_synthetic_gaps(per_bench_rows, per_instance_rows, reference_root=args.reference_root)
    populate_win_gaps(per_bench_rows)
    write_per_benchmark_csv(per_bench_path, per_bench_rows)
    print(f"\n[exp_single_obj] wrote {per_inst_path}")
    print(f"[exp_single_obj] wrote {per_bench_path}")

    print_paper_table(per_bench_rows)


if __name__ == '__main__':
    main()
