"""Shared segment-boundary logic for dynamic FJSP.

A dynamic FJSP rollout is handled as a chain of *static* segments split at
dynamic-event boundaries. At each boundary we (1) FREEZE the operations that
already started before the event time ``t`` into a carry-forward ``fixed_ops``
prefix and (2) ABSORB the event's effect (machine breakdown unavailable-until
window) into the next segment's machine floors.

This module is the single source of truth for that freeze/absorb rule so the
two consumers apply events identically (train == inference):

- ``src/HGT/inference_env.py`` — the inference orchestrator (batch=1, loops
  once per event boundary).
- ``src/HGT/hgt_trainer.py`` — the anchored-segment training driver (B parallel
  envs; freeze is per-env, breakdown absorb is per-env).

Breakdown semantics (chosen design): the breakdown fires at its generated time
``t``. An op on the broken machine that overlaps ``t`` (``start < t < end``) is
INTERRUPTED — its partial work is discarded and it returns to the pool to be
rescheduled in the next segment (it is NOT counted finished, because the
breakdown directly affects that operation). Arrivals/priority changes have no
overlap problem and are absorbed by the caller (they only grow the static
problem / reweight jobs).
"""

from __future__ import annotations

from typing import Any, Deque, Dict, List, Tuple

import numpy as np


def pop_next_event_set(event_queue: Deque[Dict[str, Any]]
                       ) -> Tuple[List[Dict[str, Any]], float]:
    """Pop all events whose time equals the earliest remaining time.

    Returns ``(event_set, t)``. The generator emits one event at a time today;
    coincident-time events are handled as a set.
    """
    if not event_queue:
        return [], float('inf')
    t = event_queue[0]['time']
    out: List[Dict[str, Any]] = []
    while event_queue and event_queue[0]['time'] == t:
        out.append(event_queue.popleft())
    return out, t


def broken_machines_of(event_set: List[Dict[str, Any]]) -> set:
    """Machine indices broken by this event-set."""
    return {e['machine_idx'] for e in event_set
            if e['type'] == 'machine_breakdown'}


def last_fixed_op_end_on_machine(fixed_ops: List[Dict[str, Any]],
                                 machine_idx: int, t: float) -> float:
    """Largest ``end`` among fixed ops on ``machine_idx`` that started before
    ``t``. Returns 0.0 if none."""
    best = 0.0
    for op in fixed_ops:
        if op['machine_id'] == machine_idx and op['start'] < t:
            if op['end'] > best:
                best = op['end']
    return best


def freeze_segment_ops(segment_schedule: List[Dict[str, Any]],
                       fixed_ops: List[Dict[str, Any]],
                       event_set: List[Dict[str, Any]],
                       t: float,
                       env_idx: int = 0
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Freeze ops that started before ``t`` into ``fixed_ops``.

    Args:
        segment_schedule: this segment's full plan (list of op dicts with
            ``op_id, job_id, machine_id, start_time, end_time,
            processing_time``), e.g. from ``extract_schedule_from_env``.
        fixed_ops: the carry-forward prefix from prior segments (list of dicts
            with ``op_id, job_id, machine_id, start, end, processing_time``).
        event_set: events firing at this boundary.
        t: event time.
        env_idx: stamped into interrupted-op records (for logging / Gantt).

    Returns:
        ``(fixed_ops, interrupted)``. ``fixed_ops`` is the updated prefix (ops
        with ``start < t`` minus breakdown-interrupted ones). ``interrupted`` is
        the list of ops cut by a breakdown this boundary (their remainder is
        rescheduled in the next segment).
    """
    broken = broken_machines_of(event_set)
    # Drop previously-frozen ops in-progress on a newly broken machine — they
    # are replayed via segment_schedule below and recorded as interrupted.
    if broken:
        fixed_ops = [op for op in fixed_ops if not (
            op['machine_id'] in broken and op['end'] > t)]
    fixed_ids = {op['op_id'] for op in fixed_ops}
    interrupted: List[Dict[str, Any]] = []
    for s in segment_schedule:
        op_id = s['op_id']
        if op_id in fixed_ids:
            continue  # already frozen in a prior iteration
        start = s['start_time']
        end = s['end_time']
        if start >= t:
            continue  # future op — returns to the pool
        if s['machine_id'] in broken and end > t:
            # Breakdown interrupt: in-progress op on the broken machine. Its
            # partial work is discarded; it reschedules next segment.
            interrupted.append({
                'env_idx': env_idx,
                'op_idx': op_id,
                'machine_idx': s['machine_id'],
                'start_time': float(start),
                'interrupt_time': float(t),
            })
            continue
        fixed_ops.append({
            'op_id': op_id,
            'job_id': s['job_id'],
            'machine_id': s['machine_id'],
            'start': float(start),
            'end': float(end),
            'processing_time': float(s['processing_time']),
        })
    return fixed_ops, interrupted


def absorb_breakdowns(fixed_ops: List[Dict[str, Any]],
                      mch_unavailable_until: np.ndarray,
                      event_set: List[Dict[str, Any]]
                      ) -> List[Dict[str, Any]]:
    """Raise per-machine unavailable-until floors for breakdowns in
    ``event_set`` (call AFTER ``freeze_segment_ops`` so ``fixed_ops`` reflects
    this segment).

    Mutates ``mch_unavailable_until`` (1D ``[n_m]``) in place; returns the
    breakdown-period log records.
    """
    logs: List[Dict[str, Any]] = []
    for evt in event_set:
        if evt['type'] != 'machine_breakdown':
            continue
        m = int(evt['machine_idx'])
        end_t = float(evt['time']) + float(evt['duration'])
        last_fix_end = last_fixed_op_end_on_machine(fixed_ops, m, evt['time'])
        # Take max with prior floor so an earlier longer breakdown isn't lost
        # when a shorter one overlaps on the same machine.
        mch_unavailable_until[m] = float(max(
            float(mch_unavailable_until[m]), end_t, last_fix_end))
        logs.append({
            'machine_idx': m,
            'start_time': float(evt['time']),
            'end_time': float(end_t),
        })
    return logs
