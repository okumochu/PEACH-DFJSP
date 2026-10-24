"""FJSP inference env — iterates over dynamic events using the training env.

Thin iterator that wraps the training env (``FJSPEnvHGT`` in ``hgt_env.py``)
to handle FJSP as a sequence of static solves:

- Empty dynamic_events list → run the static solver **once**.
- Non-empty dynamic_events list → run the static solver **once per event
  boundary**, freezing operations that started before each event (with a
  breakdown-interrupt special case) and absorbing the event's effect
  (arriving job, priority flip, unavailable-machine window) into the next
  static problem.

There is no "static mode" vs "dynamic mode" split — the same call handles
both cases. The default static solver is deterministic greedy (argmax at
each decision). Callers that need a different static solver can use
``run_with_solver`` directly.

Public API: ``InferenceEnv.run(policy, dynamic_data, ...)``.
"""

from __future__ import annotations

from collections import deque
import numpy as np
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.utils.event_utils import (
    apply_priority_change_to_weights,
    merge_and_sort_events as _merge_and_sort_events,
)
from src.utils.segment_util import (
    pop_next_event_set as _pop_next_event_set,
    freeze_segment_ops,
    absorb_breakdowns,
)
from src.HGT.hgt_env import FJSPEnvHGT
from src.utils.rl_utils import (
    _resolve_objective_weights,
    _run_single_greedy,
    greedy_result_as_dict,
)
from src.utils.gantt_util import extract_schedule_from_env


# ---------------------------------------------------------------------------
# Static-data helpers — grow initial 'static' dict across segments
# ---------------------------------------------------------------------------

def _copy_static(static: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """Deep-copy the 'static' dict so mutations (job arrival, weight flip)
    don't leak across segments."""
    return {k: [arr.copy() for arr in v] for k, v in static.items()}


def _wrap_as_static_data(static: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
    """Wrap a per-env static dict into the shape consumed by FJSPEnvHGT."""
    return {
        'static': static,
        'dynamic_events': {'new_job_arrival': [], 'machine_breakdown': [], 'priority_change': []},
    }


def _append_arriving_job(current_static: Dict[str, List[np.ndarray]],
                          dynamic_data: Dict[str, Any],
                          arrival_evt: Dict[str, Any],
                          ddt_min: float, ddt_max: float) -> None:
    """Append an arriving job (from dynamic_data['dynamic_jobs']) into current_static.

    Mutates current_static in place. Due date uses the same formula as the
    old dynamic env: arrival + uniform(ddt_min, ddt_max) * mean_P_j.
    """
    dyn = dynamic_data.get('dynamic_jobs') or {}
    idx = arrival_evt['job_idx']
    t = arrival_evt['time']

    dyn_job_length = np.asarray(dyn['job_length'])
    dyn_op_pt = np.asarray(dyn['op_pt'])
    dyn_job_weight = np.asarray(dyn['job_weight'])

    num_ops_new = int(dyn_job_length[idx])
    first_dyn_op = int(np.sum(dyn_job_length[:idx]))
    new_op_pt = dyn_op_pt[first_dyn_op:first_dyn_op + num_ops_new].astype(np.float32)
    weight = float(dyn_job_weight[idx])

    # Due date
    if 'job_due_date' in dyn and dyn['job_due_date'] is not None:
        due_date = float(np.asarray(dyn['job_due_date'])[idx])
    else:
        # Older dynamic-data fixtures may not include arrival due dates. Keep
        # the fallback deterministic so repeated solvers see the same instance.
        compat = new_op_pt > 0
        mean_pts = np.where(compat, new_op_pt, 0.0).sum(axis=1)
        counts = compat.sum(axis=1)
        valid = counts > 0
        P_j = float(np.sum(np.where(valid, mean_pts / np.maximum(counts, 1), 0.0)))
        k = 0.5 * (ddt_min + ddt_max)
        due_date = t + k * P_j

    for env_idx in range(len(current_static['job_length'])):
        jl = current_static['job_length'][env_idx]
        op_pt = current_static['op_pt'][env_idx]
        jw = current_static['job_weight'][env_idx]
        jd = current_static['job_due_date'][env_idx]

        current_static['job_length'][env_idx] = np.append(jl, num_ops_new).astype(jl.dtype)
        current_static['op_pt'][env_idx] = np.concatenate([op_pt, new_op_pt], axis=0)
        current_static['job_weight'][env_idx] = np.append(jw, weight).astype(jw.dtype)
        current_static['job_due_date'][env_idx] = np.append(jd, due_date).astype(jd.dtype)


def _apply_priority_change(current_static: Dict[str, List[np.ndarray]],
                           evt: Dict[str, Any]) -> None:
    """Apply a normalized priority-change event to current_static."""
    for env_idx in range(len(current_static['job_weight'])):
        orig = current_static['job_weight'][env_idx]
        current_static['job_weight'][env_idx] = (
            apply_priority_change_to_weights(orig, evt).astype(orig.dtype)
        )


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------

class InferenceEnv:
    """FJSP inference iterator — runs the training env once per event boundary.

    Algorithm (same whether dynamic_events is empty or not):
      0. Inputs: dynamic_events list, current schedule info, fixed_ops=[].
      1. Run static inference on current known problem.
      2. If dynamic_events is empty, terminate.
      3. Pop next event-set at time t.
      4. Freeze ops with start<t (for breakdown on m: also freeze ops on m
         with end<=t; ops on m overlapping the breakdown return to the pool).
      5. Absorb event effects into current schedule info. Repeat from step 1.

    With an empty event list the loop runs once and matches a plain static
    inference call.
    """

    def __init__(self, n_j: int, n_m: int, device, env_factory=None,
                 use_pt_nodes: bool = True,
                 use_dynamic_event_features: bool = True):
        """
        Args:
            n_j: number of jobs (initial; can grow with arrivals).
            n_m: number of machines.
            device: torch device for the env.
            env_factory: callable(n_j, n_m, device) -> env. Defaults to
                ``FJSPEnvHGT``. Pass an alternate factory to use a subclass
                with a different action space (e.g. rule-action variants).
            use_dynamic_event_features: must match the loaded checkpoint's
                design (machine is_broken + job released_time features). The
                old/static design uses False; the default new design uses True.
                Wrong value => env feature dims mismatch the network.
        """
        self.n_j = n_j
        self.n_m = n_m
        self.device = device
        if env_factory is not None:
            self._env_factory = env_factory
        else:
            self._env_factory = (
                lambda n_j, n_m, device:
                FJSPEnvHGT(
                    n_j, n_m, device, use_pt_nodes=use_pt_nodes,
                    use_dynamic_event_features=use_dynamic_event_features)
            )

    def run(
        self,
        policy,
        dynamic_data: Dict[str, Any],
        *,
        makespan_weight: Optional[float] = None,
        twt_weight: Optional[float] = None,
        verbose: bool = False,
        collect_snapshots: bool = False,
    ) -> Dict[str, Any]:
        """Run dynamic RL inference. Thin wrapper over run_with_solver that
        builds a deterministic greedy static-solve callable.
        """
        def _greedy_solver(env, mw, tw):
            env.makespan_weight = mw
            env.twt_weight = tw
            return greedy_result_as_dict(
                *_run_single_greedy(
                    policy, env.number_of_jobs, env.number_of_machines,
                    None, self.device, pre_initialized_env=env,
                ),
                makespan_weight=mw,
                twt_weight=tw,
            )

        return self.run_with_solver(
            _greedy_solver, dynamic_data,
            makespan_weight=makespan_weight, twt_weight=twt_weight,
            verbose=verbose,
            collect_snapshots=collect_snapshots,
        )

    def run_with_solver(
        self,
        static_solver_fn,
        dynamic_data: Dict[str, Any],
        makespan_weight: Optional[float] = None,
        twt_weight: Optional[float] = None,
        verbose: bool = False,
        collect_snapshots: bool = False,
    ) -> Dict[str, Any]:
        """Run dynamic inference with a caller-provided static solver.

        Args:
            static_solver_fn: callable(env, mw, tw) -> result_dict where
                result_dict has keys {best_env, best_env_idx, step_count, ...}.
                The callable must treat env as already-seeded (fixed ops +
                unavailable-until floors already applied).
            dynamic_data: same shape as DataGenerator.sample_env output.
            collect_snapshots: when True, the returned dict includes a
                ``'snapshots'`` list — one entry per segment, each shaped as::

                    {
                        'segment_idx': int,            # 0-based segment index
                        'schedule': List[op_dict],     # this segment's full plan
                        'job_data': List[job_dict],    # jobs known at this point
                        'fixed_op_ids': Set[int],      # ops frozen BEFORE this solve
                        'breakdown_periods': List[dict],  # accumulated so far
                        'interrupted_ops': List[dict],    # accumulated so far
                        'triggering_event_set': List[dict] | None,  # events that
                            # fired between seg-1 and this segment; None for seg 0
                        'time_now': float,             # event time, 0.0 for seg 0
                        'segment_makespan': float,
                    }

                Consumed by ``src.utils.gantt_util.create_gantt_gif`` to
                animate schedule evolution across dynamic-event boundaries.
        """
        mw, tw = _resolve_objective_weights(makespan_weight, twt_weight)

        event_queue = _merge_and_sort_events(dynamic_data.get('dynamic_events', {}))
        current_static = _copy_static(dynamic_data['static'])
        num_machines = current_static['op_pt'][0].shape[1]

        # Default ddt from dynamic_jobs (if present), else fallback.
        dyn_jobs = dynamic_data.get('dynamic_jobs') or {}
        ddt_min = float(dyn_jobs.get('ddt_min', 0.5))
        ddt_max = float(dyn_jobs.get('ddt_max', 1.5))

        fixed_ops: List[Dict[str, Any]] = []  # single-env: list of {op_id, job_id, machine_id, start, end, ...}
        mch_unavailable_until = np.zeros(num_machines, dtype=np.float32)
        job_release_times: List[float] = [0.0] * len(current_static['job_weight'][0])

        breakdown_periods: List[Dict[str, Any]] = []
        interrupted_ops: List[Dict[str, Any]] = []
        total_steps = 0
        num_segments = 0
        last_result: Optional[Dict[str, Any]] = None
        snapshots: List[Dict[str, Any]] = []
        last_triggering_event_set: Optional[List[Dict[str, Any]]] = None
        last_triggering_time: float = 0.0

        while True:
            num_segments += 1
            n_j_current = len(current_static['job_length'][0])

            # --- Step 1: solve current static problem ---
            static_env = self._env_factory(n_j_current, self.n_m, device=self.device)
            static_env.makespan_weight = mw
            static_env.twt_weight = tw
            static_env.set_initial_data_with_fixed_schedule(
                _wrap_as_static_data(current_static),
                fixed_ops=[fixed_ops] if fixed_ops else None,
                mch_unavailable_until=mch_unavailable_until.copy(),
                job_release_times=np.asarray(job_release_times, dtype=np.float32),
                current_time=last_triggering_time,
            )

            result = static_solver_fn(static_env, mw, tw)
            last_result = result
            total_steps += result['step_count']

            # Solver reported no feasible solution (e.g. OR-Tools infeasible or
            # timed out with zero incumbents). Bail with None-metrics so the
            # CSV row records a distinct failure rather than a partial
            # fixed-ops schedule that looks like a valid answer.
            if result.get('best_env') is None:
                return {
                    'makespan': None, 'twt': None, 'objective': None,
                    'step_count': total_steps,
                    'best_env': None, 'best_env_idx': 0,
                    'schedule': [],
                    'num_segments': num_segments,
                    'breakdown_periods': breakdown_periods,
                    'interrupted_ops': interrupted_ops,
                    'snapshots': snapshots,
                    'status': result.get('status', 'Infeasible'),
                }

            # --- Extract segment schedule (shared by snapshot + freeze logic) ---
            best_env = result['best_env']
            extracted = extract_schedule_from_env(
                best_env, env_idx=result['best_env_idx'])
            segment_schedule = extracted['schedule']

            if verbose and num_segments > 1:
                prev_fixed_ids = {op['op_id'] for op in fixed_ops}
                eps = 1e-4
                for sched_op in segment_schedule:
                    if sched_op['op_id'] in prev_fixed_ids:
                        continue
                    if sched_op['start_time'] + eps < last_triggering_time:
                        raise AssertionError(
                            f"[InferenceEnv] segment {num_segments} op "
                            f"{sched_op['op_id']} on M{sched_op['machine_id']} "
                            f"starts at {sched_op['start_time']:.4f} < "
                            f"event_time {last_triggering_time:.4f}")

            if collect_snapshots:
                snapshots.append({
                    'segment_idx': num_segments - 1,
                    'schedule': extracted['schedule'],
                    'job_data': extracted['job_data'],
                    'fixed_op_ids': {op['op_id'] for op in fixed_ops},
                    'breakdown_periods': [dict(bp) for bp in breakdown_periods],
                    'interrupted_ops': [dict(io) for io in interrupted_ops],
                    'triggering_event_set': (
                        [dict(e) for e in last_triggering_event_set]
                        if last_triggering_event_set else None),
                    'time_now': last_triggering_time,
                    'segment_makespan': float(
                        best_env.current_makespan[result['best_env_idx']]),
                })

            # --- Step 5: termination ---
            if not event_queue:
                break

            # --- Step 2: pop the next event-set ---
            event_set, t = _pop_next_event_set(event_queue)

            # --- Step 3: freeze ops with start < t (breakdown interrupt-at-t) ---
            # Shared with the training driver (src/utils/segment_util.py) so
            # train and inference apply events identically.
            fixed_ops, seg_interrupted = freeze_segment_ops(
                segment_schedule, fixed_ops, event_set, t,
                env_idx=result['best_env_idx'])
            interrupted_ops.extend(seg_interrupted)

            # --- Step 4: absorb event effects into machines / static / weights ---
            #   breakdowns -> per-machine unavailable-until floors (shared rule)
            for bd in absorb_breakdowns(fixed_ops, mch_unavailable_until, event_set):
                breakdown_periods.append(
                    {'env_idx': result['best_env_idx'], **bd})
            #   arrivals -> grow the static problem + release times;
            #   priority -> reweight jobs
            for evt in event_set:
                if evt['type'] == 'job_arrival':
                    _append_arriving_job(current_static, dynamic_data, evt,
                                          ddt_min, ddt_max)
                    job_release_times.append(float(evt['time']))
                elif evt['type'] == 'priority_change':
                    _apply_priority_change(current_static, evt)

            last_triggering_event_set = event_set
            last_triggering_time = t

        # --- Finalize: merge the last-segment ops with fixed_ops into the global schedule ---
        final_env = _finalize_env(
            last_result['best_env'], breakdown_periods, interrupted_ops)

        global_schedule = extract_schedule_from_env(final_env, 0)['schedule']
        global_makespan = float(final_env.current_makespan[0])
        global_twt = float(final_env.current_twt[0])

        return {
            'makespan': global_makespan,
            'twt': global_twt,
            'objective': mw * global_makespan + tw * global_twt,
            'step_count': total_steps,
            'best_env': final_env,
            'best_env_idx': 0,
            'schedule': global_schedule,
            'num_segments': num_segments,
            'breakdown_periods': breakdown_periods,
            'interrupted_ops': interrupted_ops,
            'snapshots': snapshots,
            'status': last_result.get('status', 'OK'),
        }


def _finalize_env(best_env,
                  breakdown_periods: List[Dict[str, Any]],
                  interrupted_ops: List[Dict[str, Any]]):
    """Attach dynamic-event logs and ensure objectives are consistent.

    Mch_queue and per-op times are already correct because each segment
    ran static inference with prior fixed_ops seeded into the env.
    """
    best_env._recompute_objectives()
    best_env.breakdown_periods = list(breakdown_periods)
    best_env.interrupted_ops = list(interrupted_ops)
    return best_env
