"""Shared dynamic-event helpers used by both training and inference envs.

Training-side note: ``apply_event_to_env`` uses the "simple push"
approximation for machine breakdowns — it bumps ``mch_free_time`` to
``end_t`` but does NOT unschedule in-progress operations on the broken
machine. This is an accepted bias for training; the inference orchestrator
in ``inference_env.py`` handles the full unschedule via segment boundaries.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np


def merge_and_sort_events(dynamic_events: Dict[str, Any]) -> Deque[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for i, t in enumerate(dynamic_events.get('new_job_arrival', []) or []):
        events.append({'type': 'job_arrival', 'time': float(t), 'job_idx': i})
    for bd in dynamic_events.get('machine_breakdown', []) or []:
        events.append({
            'type': 'machine_breakdown',
            'time': float(bd['time']),
            'machine_idx': int(bd['machine_idx']),
            'duration': float(bd.get('duration', np.inf)),
        })
    for item in dynamic_events.get('priority_change', []) or []:
        if isinstance(item, dict):
            evt = {
                'type': 'priority_change',
                'time': float(item['time']),
            }
            if 'job_indices' in item and item['job_indices'] is not None:
                evt['job_indices'] = [int(j) for j in item['job_indices']]
            if 'new_weight' in item:
                evt['new_weight'] = float(item['new_weight'])
            if 'mode' in item:
                evt['mode'] = str(item['mode'])
            events.append(evt)
        else:
            events.append({'type': 'priority_change', 'time': float(item)})
    events.sort(key=lambda e: e['time'])
    return deque(events)


def init_event_queues(shared_events, num_envs: int) -> List[Deque[Dict[str, Any]]]:
    """One independent deque per env so events drain at each env's own sim_time."""
    return [deque(shared_events) for _ in range(num_envs)]


def reset_event_counters(env) -> None:
    """Reset per-env counters for event records actually drained."""
    env.fired_event_counts = {
        'job_arrival': np.zeros(env.number_of_envs, dtype=np.int64),
        'machine_breakdown': np.zeros(env.number_of_envs, dtype=np.int64),
        'priority_change': np.zeros(env.number_of_envs, dtype=np.int64),
    }


def _ensure_event_counters(env) -> Dict[str, np.ndarray]:
    counters = getattr(env, 'fired_event_counts', None)
    if not isinstance(counters, dict):
        reset_event_counters(env)
        return env.fired_event_counts
    for key in ('job_arrival', 'machine_breakdown', 'priority_change'):
        arr = counters.get(key)
        if arr is None or len(arr) != env.number_of_envs:
            reset_event_counters(env)
            return env.fired_event_counts
    return counters


def flip_priority_weights(w: np.ndarray) -> np.ndarray:
    """Flip weights {1, 2} -> 5 and {4, 5} -> 1. Used by both training-time
    in-place event firing and inference-time static-segment rebuilds; keep in
    sync since the two paths must produce identical post-flip distributions.
    """
    w = w.astype(np.float32)
    rounded = np.round(w)
    low = (rounded >= 1) & (rounded <= 2)
    high = (rounded >= 4) & (rounded <= 5)
    return np.where(low, 5.0, np.where(high, 1.0, w))


def apply_priority_change_to_weights(
    weights: np.ndarray,
    evt: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Apply a priority-change event to a weight vector.

    Legacy float priority-change timestamps normalize to an event without
    ``job_indices`` and retain the historical all-job low/high flip. Dict
    events can target a subset of jobs, e.g.
    ``{'job_indices': [1, 4], 'new_weight': 5, 'mode': 'set'}``.
    """
    if evt is None or evt.get('job_indices') is None:
        return flip_priority_weights(weights)

    updated = weights.astype(np.float32, copy=True)
    job_indices = np.asarray(evt.get('job_indices', []), dtype=np.int64)
    if job_indices.size == 0:
        return updated.astype(weights.dtype, copy=False)

    valid = job_indices[(job_indices >= 0) & (job_indices < len(updated))]
    if valid.size == 0:
        return updated.astype(weights.dtype, copy=False)

    mode = str(evt.get('mode', 'set')).lower()
    new_weight = float(evt.get('new_weight', 5.0))
    if mode == 'set':
        updated[valid] = new_weight
    elif mode == 'add':
        updated[valid] += new_weight
    elif mode == 'multiply':
        updated[valid] *= new_weight
    else:
        raise ValueError(
            f"Unknown priority_change mode {mode!r}; "
            "expected 'set', 'add', or 'multiply'.")
    return updated.astype(weights.dtype, copy=False)


def apply_event_to_env(env, env_idx: int, evt: Dict[str, Any]) -> None:
    etype = evt['type']
    if etype == 'job_arrival':
        # Realize the arriving job in the (pre-allocated) training env. No-op if
        # the env has no dynamic_jobs template (inference / static runs).
        if (getattr(env, 'dynamic_jobs', None) is not None
                and hasattr(env, '_activate_arriving_job')):
            env._activate_arriving_job(env_idx, int(evt['job_idx']), float(evt['time']))
        return
    if etype == 'machine_breakdown':
        m = int(evt['machine_idx'])
        end_t = float(evt['time']) + float(evt['duration'])
        norm_end_t = end_t / float(env.max_pt[env_idx])
        env.mch_free_time[env_idx, m] = max(
            float(env.mch_free_time[env_idx, m]), norm_end_t)
        env.unnorm_mch_free_time[env_idx, m] = max(
            float(env.unnorm_mch_free_time[env_idx, m]), end_t)
        env.mch_earliest_start_time[env_idx, m] = max(
            float(env.mch_earliest_start_time[env_idx, m]),
            norm_end_t)
        # Record unnormalized repair-end for the is_broken machine feature
        # (compared against the unnormalized current_makespan frontier).
        if hasattr(env, 'mch_broken_until'):
            env.mch_broken_until[env_idx, m] = max(
                float(env.mch_broken_until[env_idx, m]), end_t)
    elif etype == 'priority_change':
        # Inline arrivals grow per-environment active counts while the initial
        # number_of_jobs stays frozen. Do not include unarrived padding.
        active_jobs = getattr(env, 'n_active_jobs', None)
        J = int(active_jobs[env_idx]) if active_jobs is not None else env.number_of_jobs
        env.job_weight[env_idx, :J] = apply_priority_change_to_weights(
            env.job_weight[env_idx, :J], evt).astype(env.job_weight.dtype)


def fire_due_events(env):
    """Drain each env's queue up to its current sim_time.

    Uses ``env.current_makespan`` as sim_time (already maintained by
    ``step()``), avoiding a redundant reduction over ``unnorm_op_ct``.

    Returns a ``[number_of_envs]`` bool mask of envs that fired at least one
    event (caller resets ``max_endTime`` only for those — the unfired envs
    must keep accumulating reward against their previous LB). Returns
    ``None`` when no queue is attached.
    """
    queues = getattr(env, 'event_queues', None)
    if queues is None:
        return None
    counters = _ensure_event_counters(env)
    fired = np.zeros(env.number_of_envs, dtype=bool)
    done_flag = getattr(env, 'done_flag', None)
    for env_idx in range(env.number_of_envs):
        if done_flag is not None and bool(done_flag[env_idx]):
            continue
        q = queues[env_idx]
        if not q:
            continue
        sim_time = float(env.current_makespan[env_idx])
        if sim_time < 0:  # init_value is -inf before any op completes
            sim_time = 0.0
        while q and q[0]['time'] <= sim_time:
            evt = q.popleft()
            apply_event_to_env(env, env_idx, evt)
            etype = evt.get('type')
            if etype in counters:
                counters[etype][env_idx] += 1
            fired[env_idx] = True
    return fired
