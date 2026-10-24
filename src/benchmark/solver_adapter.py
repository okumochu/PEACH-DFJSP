"""Greedy and CP-SAT adapters for ``InferenceEnv.run_with_solver``.

Both consume the same seeded static segment, including frozen operations,
machine availability floors and job releases. Missing solver results retain
empty metrics and an explicit status.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.HGT.hgt_env import FJSPEnvHGT
from src.utils.reproducibility import derive_seed


# ---------------------------------------------------------------------------
# Snapshot extraction
# ---------------------------------------------------------------------------

def _extract_static_snapshot(env: FJSPEnvHGT) -> Dict[str, Any]:
    """Read the per-segment static problem from a seeded ``FJSPEnvHGT``.

    The env passed to a ``static_solver_fn`` already has prior fixed_ops,
    machine-unavailable-until floors, and job release times applied via
    ``set_initial_data_with_fixed_schedule``. We re-extract those into a flat
    dict that CP-SAT can consume directly, plus a ``static_data`` payload
    that can be re-fed to ``set_initial_data_with_fixed_schedule`` when
    re-seeding a fresh env with the solver's plan.
    """
    n_j = int(env.number_of_jobs)
    n_m = int(env.number_of_machines)
    n_ops = int(env.number_of_ops)

    job_length = env.job_length[0, :n_j].astype(np.int64).copy()
    op_pt = env.op_pt[0, :n_ops, :].astype(np.float32).copy()
    job_weight = env.job_weight[0, :n_j].astype(np.float32).copy()
    job_due_date = env.job_due_date[0, :n_j].astype(np.float32).copy()
    job_release_time = env.job_release_time[0, :n_j].astype(np.float32).copy()
    mch_unavailable_until = env.unnorm_mch_free_time[0, :n_m].astype(np.float32).copy()

    # Already-scheduled ops (prior fixed_ops). Use op_assigned_machine to
    # avoid round-tripping through mch_queue.
    fixed_ops: List[Dict[str, Any]] = []
    if hasattr(env, 'op_scheduled_flag'):
        scheduled_mask = env.op_scheduled_flag[0, :n_ops] == 1
        scheduled_idxs = np.where(scheduled_mask)[0]
        for op_id in scheduled_idxs:
            mch_id = int(env.op_assigned_machine[0, op_id])
            if mch_id < 0:
                continue  # safety
            start = float(env.unnorm_op_start_time[0, op_id])
            end = float(env.unnorm_op_ct[0, op_id])
            fixed_ops.append({
                'op_id': int(op_id),
                'machine_id': mch_id,
                'start': start,
                'end': end,
                'processing_time': end - start,
                'job_id': int(env.op_to_job[0, op_id]),
            })

    breakdown_periods = []
    for bp in getattr(env, 'breakdown_periods', []) or []:
        if bp.get('env_idx', 0) != 0:
            continue
        breakdown_periods.append({
            'machine_idx': int(bp['machine_idx']),
            'start_time': float(bp['start_time']),
            'end_time': float(bp.get('end_time', np.inf)),
        })

    static_data = {
        'static': {
            'job_length': [job_length],
            'op_pt': [op_pt],
            'job_weight': [job_weight],
            'job_due_date': [job_due_date],
        },
        'dynamic_events': {
            'new_job_arrival': [],
            'machine_breakdown': [],
            'priority_change': [],
        },
        'dynamic_jobs': None,
    }

    return {
        'n_j': n_j,
        'n_m': n_m,
        'n_ops': n_ops,
        'job_length': job_length,
        'op_pt': op_pt,
        'job_weight': job_weight,
        'job_due_date': job_due_date,
        'job_release_time': job_release_time,
        'mch_unavailable_until': mch_unavailable_until,
        'fixed_operations': fixed_ops,
        'breakdown_periods': breakdown_periods,
        'static_data': static_data,
        'device': env.device,
        'use_pt_nodes': getattr(env, 'use_pt_nodes', True),
    }


def _seed_env_with_full_plan(
    snap: Dict[str, Any],
    full_fixed_ops: List[Dict[str, Any]],
    mw: float,
    tw: float,
) -> FJSPEnvHGT:
    """Spin a fresh ``FJSPEnvHGT`` with the entire segment plan applied.

    ``full_fixed_ops`` should contain prior fixed ops *plus* the ops the
    external solver assigned in this segment. The fresh env's
    ``extract_schedule_from_env`` will then return a complete schedule that
    the orchestrator can freeze for the next segment.
    """
    fresh = FJSPEnvHGT(
        snap['n_j'], snap['n_m'], device=snap['device'],
        use_pt_nodes=snap.get('use_pt_nodes', True))
    fresh.makespan_weight = mw
    fresh.twt_weight = tw
    fresh.set_initial_data_with_fixed_schedule(
        snap['static_data'],
        fixed_ops=[full_fixed_ops] if full_fixed_ops else None,
        mch_unavailable_until=snap['mch_unavailable_until'].copy(),
        job_release_times=snap['job_release_time'].copy(),
    )
    fresh._recompute_objectives()
    return fresh


def _make_missing_result(reason: str) -> Dict[str, Any]:
    """Result for a solver that could not run at all (dependency absent).

    Explicit ``None`` metrics prevent an unavailable solver's empty seeded
    schedule from being reported as a zero-objective solution.
    """
    return {
        'makespan': None, 'twt': None, 'objective': None,
        'step_count': 0, 'best_env': None, 'best_env_idx': 0,
        'status': reason,
    }


def _availability_windows(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-machine availability floors, expressed as blocking windows.

    ``_extract_static_snapshot`` carries each machine's floor in
    ``mch_unavailable_until`` (``env.unnorm_mch_free_time``), which folds in
    both the segment clock and the end of any in-progress repair. Passing
    only ``breakdown_periods`` would lose those floors because the inference
    orchestrator attaches that history only to its final environment.

    Emit ``[0, floor)`` blocking windows. CP-SAT carves fixed operations out
    of each window so the availability floor does not conflict with their pins.
    """
    floors = snap.get('mch_unavailable_until')
    if floors is None:
        return []
    windows: List[Dict[str, Any]] = []
    for m in range(int(snap['n_m'])):
        until = float(floors[m])
        if until > 0.0:
            windows.append({'machine_idx': m,
                            'start_time': 0.0,
                            'end_time': until})
    return windows


def _record_time(per_segment_times: Optional[List[float]], elapsed: float) -> None:
    """Append elapsed to per_segment_times if the list is provided."""
    if per_segment_times is not None:
        per_segment_times.append(elapsed)


# ---------------------------------------------------------------------------
# Greedy (argmax) wrapper — shared by all inference scripts
# ---------------------------------------------------------------------------

def make_greedy_solver_fn(
    policy,
    device: str,
    per_segment_times: Optional[List[float]] = None,
) -> Callable[[FJSPEnvHGT, float, float], Dict[str, Any]]:
    """Wrap a single-env greedy (argmax) rollout as a static_solver_fn.

    Used by all three experiment scripts via ``InferenceEnv.run_with_solver``
    so that the orchestrator's per-segment freezing path is identical for
    greedy, CP-SAT, and dispatching rules.
    """
    from src.utils.rl_utils import _run_single_greedy, greedy_result_as_dict

    def _solver(env: FJSPEnvHGT, mw: float, tw: float) -> Dict[str, Any]:
        t0 = time.time()
        result = greedy_result_as_dict(*_run_single_greedy(
            policy, env.number_of_jobs, env.number_of_machines,
            None, device, pre_initialized_env=env,
        ), makespan_weight=mw, twt_weight=tw)
        _record_time(per_segment_times, time.time() - t0)
        return result
    return _solver


# ---------------------------------------------------------------------------
# CP-SAT wrapper
# ---------------------------------------------------------------------------

def make_cp_solver_fn(
    time_limit: int,
    num_workers: int = 8,
    verbose: int = 0,
    time_scale: int = 1,
    per_segment_times: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> Callable[[FJSPEnvHGT, float, float], Dict[str, Any]]:
    """Build a ``static_solver_fn`` that solves each segment with CP-SAT.

    Breakdown windows retain non-finite ends, and fixed operations are
    carved out of blocking windows while preserving the remaining downtime.
    """
    try:
        from src.benchmark.CP import CP
    except ImportError:
        CP = None  # type: ignore

    def _solver(env: FJSPEnvHGT, mw: float, tw: float) -> Dict[str, Any]:
        if CP is None:
            _record_time(per_segment_times, 0.0)
            return _make_missing_result('skipped (no ortools)')

        snap = _extract_static_snapshot(env)

        # Translate the environment's 'start'/'end' keys to CP-SAT's
        # 'start_time'/'end_time' keys at the adapter boundary.
        cp_fixed_ops = [
            {'op_id': op['op_id'], 'machine_id': op['machine_id'],
             'start_time': op['start'], 'end_time': op['end']}
            for op in snap['fixed_operations']
        ]

        t0 = time.time()
        try:
            segment_idx = len(per_segment_times) if per_segment_times is not None else 0
            solver_seed = (
                derive_seed(seed, 'cp', segment_idx)
                if seed is not None else None
            )
            solver = CP.from_rescheduling_state(
                job_length=snap['job_length'],
                op_pt=snap['op_pt'],
                job_weight=snap['job_weight'],
                job_due_date=snap['job_due_date'],
                job_release_time=snap['job_release_time'],
                fixed_operations=cp_fixed_ops,
                machine_breakdowns=(list(snap['breakdown_periods'])
                                    + _availability_windows(snap)),
                twt_weight=tw,
                time_scale=time_scale,
            )
            solver.build_model(
                time_limit=time_limit, num_workers=num_workers,
                verbose=verbose, seed=solver_seed,
            )
            result = solver.solve()
        except Exception as e:
            _record_time(per_segment_times, time.time() - t0)
            return {
                'makespan': None, 'twt': None, 'objective': None,
                'step_count': 0, 'best_env': None, 'best_env_idx': 0,
                'status': f'error ({e})',
            }

        _record_time(per_segment_times, time.time() - t0)

        perf = result['performance']
        if perf.get('makespan') is None:
            # No incumbent. Explicit None metrics so downstream CSV/plots treat
            # this as missing rather than as a zero-objective win.
            return {
                'makespan': None, 'twt': None, 'objective': None,
                'step_count': 0, 'best_env': None, 'best_env_idx': 0,
                'status': perf.get('status', 'Infeasible'),
            }

        full_fixed_ops: List[Dict[str, Any]] = []
        for m_id, ops in result['schedule_result'].items():
            for op_id, s, e in ops:
                full_fixed_ops.append({
                    'op_id': int(op_id),
                    'machine_id': int(m_id),
                    'start': float(s),
                    'end': float(e),
                })

        fresh = _seed_env_with_full_plan(snap, full_fixed_ops, mw, tw)
        ms = float(fresh.current_makespan[0])
        twv = float(fresh.current_twt[0])
        return {
            'makespan': ms, 'twt': twv,
            'objective': mw * ms + tw * twv,
            'step_count': 0,
            'best_env': fresh, 'best_env_idx': 0,
            'status': perf.get('status', 'OK'),
            'objective_bound': perf.get('objective_bound'),
            'relative_gap': perf.get('relative_gap'),
        }

    return _solver
