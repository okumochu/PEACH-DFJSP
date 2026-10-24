# Interval construction adapted from Google OR-Tools v9.15:
# https://github.com/google/or-tools/blob/v9.15/examples/python/flexible_job_shop_sat.py
# Copyright 2010-2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adaptations retain this study's objective, releases, fixed operations,
# breakdowns, integer grid, schedule repair, and existing solver interface.

"""CP-SAT (OR-Tools) model for the Flexible Job Shop Problem.

The reference solver uses optional intervals and no-overlap constraints for
machine routing and capacity. ``solver_adapter`` connects this interface to
the shared static and dynamic inference paths:

    CP.from_data_generator(...) / CP.from_rescheduling_state(...)
    solver.build_model(time_limit=..., num_workers=..., verbose=..., seed=...)
    result = solver.solve()   # {'performance': {...}, 'schedule_result': {...}}

Model
-----
Variables, for operation ``i`` with compatible machine set ``M(i)``:
    start[i], duration[i], end[i]     mandatory master interval
    presence[i, k]  for k in M(i)     Boolean routing literal
    optional local intervals per alternative machine, with their own start
    and end variables. Presence conditionally links local start, duration,
    and end to the master interval, following the official v9.15 example.
    A single compatible machine uses the master interval and constant presence.

Constraints:
    ExactlyOne(presence[i, :])                      routing
    NoOverlap(intervals on machine k)               capacity
    end[prev] <= start[next]                        intra-job precedence
    start[first op of j] >= release[j]              job release / arrival
    fixed presence + fixed start                    already-executed ops
    fixed blocking intervals on machine k           breakdown windows
    Cmax = max_j end[last op of j]
    T[j] = max(0, end[last op of j] - due[j])

Objective (integerised, see ``obj_scale``):
    minimize  (1 - twt_weight) * Cmax + twt_weight * sum_j w[j] * T[j]

Integrality
-----------
CP-SAT is an integer solver, but the instances carry float release times, due
dates and breakdown boundaries (process times and job weights are already
integral). Times are mapped onto an integer grid of ``time_scale`` ticks per
unit, rounding *conservatively* so the resulting schedule is always feasible
for the true float instance:

    process time   exact (integral input)
    release time      ceil   — never start a job before it truly arrives
    breakdown start   floor  } window widened outward, never under-reserved
    breakdown end     ceil   }
    due date          nearest — objective only, never feasibility
    fixed op span     floor the start, ceil the end } widened outward so the
                      grid never advertises a machine as free while it is
                      really busy. Spans are then clipped to stay disjoint per
                      machine, which is what keeps adjacent fixed ops from
                      widening into a false one-tick overlap.

``solve()`` then runs :meth:`_repair_schedule_exact`, which replays the CP
sequence on the original *float* data — same machine assignment, same order per
machine — and left-shifts each operation to its earliest exact-arithmetic start.
The reported schedule therefore carries exact float times, is feasible by
construction, and is never worse than the gridded one — outward rounding is
exactly what buys that last property.

What the grid does NOT buy is optimality. Rounding conservatively shrinks the
feasible region, so the gridded model is a strict *restriction* of the float
instance: any true slack narrower than one tick disappears, which can flip a
sequencing decision. ``status='Optimal'`` therefore means "optimal for the
gridded model", and the accompanying ``relative_gap=0`` says nothing about the
float instance. The time error is O(1/scale); the OBJECTIVE error is not
bounded by it. ``performance['grid_exact']`` reports whether the inputs landed
on the grid at all — when it is False, do not cite the result as an exact
optimum without raising ``solver['cp_time_scale']`` (default 100, which clears
the whole differential-test suite; see verification/difftest_cp_model.py).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional solver dependency
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover
    cp_model = None  # type: ignore


# Objective coefficients must be integral. The makespan/TWT weights are
# fractions (0.5/0.5 by default), so both are multiplied by this constant.
DEFAULT_OBJ_SCALE = 1000


class CP:
    """CP-SAT model for static and dynamic (rescheduling) FJSP instances."""

    def __init__(self,
                 job_length: np.ndarray,
                 op_pt: np.ndarray,
                 job_weight: np.ndarray,
                 job_due_date: np.ndarray,
                 twt_weight: float,
                 job_release_time: Optional[np.ndarray] = None,
                 fixed_operations: Optional[List[Dict]] = None,
                 machine_breakdowns: Optional[List[Dict]] = None,
                 time_scale: int = 1,
                 obj_scale: int = DEFAULT_OBJ_SCALE):
        """
        Args:
            job_length: [J] operations per job.
            op_pt: [N, M] process times; 0 means the machine is incompatible.
            job_weight: [J] tardiness weights.
            job_due_date: [J] due dates.
            twt_weight: TWT share of the objective; makespan gets 1 - twt_weight.
                Required — read it from the model checkpoint via load_model().
            job_release_time: [J] arrival times (default zeros).
            fixed_operations: already-executed ops, dicts with keys
                ``op_id``, ``machine_id``, ``start_time`` and optionally
                ``end_time``.
            machine_breakdowns: unavailability windows, dicts with keys
                ``machine_idx``, ``start_time``, ``end_time``. A non-finite
                ``end_time`` means "down for the rest of the horizon" and is
                modelled exactly.
            time_scale: integer ticks per time unit for the search grid.
            obj_scale: multiplier that makes the fractional objective weights
                integral.
        """
        if cp_model is None:
            raise ImportError(
                "ortools is required for the CP baseline: pip install ortools")
        if twt_weight is None:
            raise ValueError(
                "CP requires twt_weight. Read it from the model checkpoint "
                "via src.utils.model_utils.load_model().")

        self.twt_weight = float(twt_weight)
        self.makespan_weight = 1.0 - self.twt_weight

        self.job_length = np.asarray(job_length).astype(int)
        self.processing_times = np.asarray(op_pt).astype(np.float64)
        self.due_dates = np.asarray(job_due_date).astype(np.float64)
        self.weights = np.asarray(job_weight).astype(np.float64)

        if job_release_time is not None:
            self.release_times = np.asarray(job_release_time).astype(np.float64)
        else:
            self.release_times = np.zeros(len(self.job_length), dtype=np.float64)

        self.fixed_operations = list(fixed_operations or [])
        self.fixed_op_ids = {int(op['op_id']) for op in self.fixed_operations}
        self.fixed_by_id = {int(op['op_id']): op for op in self.fixed_operations}
        # Filled on first use by _fixed_spans_by_op(); needs self.scale, which
        # is not set yet at this point.
        self._fixed_spans_cache: Optional[Dict[int, Tuple[int, int]]] = None

        self.machine_breakdowns = list(machine_breakdowns or [])

        self.num_jobs = len(self.job_length)
        self.num_machines = int(self.processing_times.shape[1])
        self.num_operations = int(self.processing_times.shape[0])

        self.scale = max(1, int(time_scale))
        self.obj_scale = int(obj_scale)

        self._compute_job_operation_mapping()
        self._compute_compatible_machines()
        self._compute_machine_operations()

        self.model = cp_model.CpModel()
        self.solver: Optional["cp_model.CpSolver"] = None
        self._status = None
        self._solve_start_time = None
        # Populated by build_model(); consumed by solve().
        self.start_vars: List[Any] = []
        self.end_vars: List[Any] = []
        self.presence: Dict[Tuple[int, int], Any] = {}
        self.C_max = None
        self.T = []
        self.horizon = 0
        # Diagnostics worth surfacing rather than hiding.
        self.build_warnings: List[str] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_data_generator(cls,
                            dynamic_data: Dict[str, Any],
                            twt_weight: float,
                            env_idx: int = 0,
                            fixed_operations: Optional[List[Dict]] = None,
                            machine_breakdowns: Optional[List[Dict]] = None,
                            **kwargs):
        """Build from ``DataGenerator.sample_env()`` output."""
        static = dynamic_data['static']
        job_release_time = static.get('job_release_time')
        if job_release_time is not None:
            job_release_time = job_release_time[env_idx]

        return cls(
            job_length=static['job_length'][env_idx],
            op_pt=static['op_pt'][env_idx],
            job_weight=static['job_weight'][env_idx],
            job_due_date=static['job_due_date'][env_idx],
            twt_weight=twt_weight,
            job_release_time=job_release_time,
            fixed_operations=fixed_operations,
            machine_breakdowns=machine_breakdowns,
            **kwargs,
        )

    @classmethod
    def from_rescheduling_state(cls,
                                job_length: np.ndarray,
                                op_pt: np.ndarray,
                                job_weight: np.ndarray,
                                job_due_date: np.ndarray,
                                job_release_time: np.ndarray,
                                fixed_operations: List[Dict],
                                twt_weight: float,
                                machine_breakdowns: Optional[List[Dict]] = None,
                                **kwargs):
        """Build a rescheduling model where some ops are already executed."""
        return cls(
            job_length=job_length,
            op_pt=op_pt,
            job_weight=job_weight,
            job_due_date=job_due_date,
            twt_weight=twt_weight,
            job_release_time=job_release_time,
            fixed_operations=fixed_operations,
            machine_breakdowns=machine_breakdowns,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Derived structure
    # ------------------------------------------------------------------

    def _compute_job_operation_mapping(self):
        self.job_first_op_id = np.zeros(self.num_jobs, dtype=int)
        cumsum = np.cumsum(self.job_length)
        self.job_first_op_id[1:] = cumsum[:-1]
        self.job_last_op_id = cumsum - 1

        self.job_operations = [
            list(range(self.job_first_op_id[j], self.job_last_op_id[j] + 1))
            for j in range(self.num_jobs)
        ]
        self.op_to_job = np.zeros(self.num_operations, dtype=int)
        self.op_position = np.zeros(self.num_operations, dtype=int)
        for j in range(self.num_jobs):
            for pos, op_id in enumerate(self.job_operations[j]):
                self.op_to_job[op_id] = j
                self.op_position[op_id] = pos

    def _compute_compatible_machines(self):
        self.compatible_machines = [
            np.where(self.processing_times[i] > 0)[0].tolist()
            for i in range(self.num_operations)
        ]
        empty = [i for i, ms in enumerate(self.compatible_machines) if not ms]
        if empty:
            raise ValueError(
                f"operations {empty[:5]} have no compatible machine; "
                "the instance is infeasible by construction")

    def _compute_machine_operations(self):
        self.machine_operations = [
            np.where(self.processing_times[:, m] > 0)[0].tolist()
            for m in range(self.num_machines)
        ]

    def is_operation_fixed(self, op_id: int) -> bool:
        return int(op_id) in self.fixed_op_ids

    def get_processing_time(self, op_id: int, machine_id: int) -> float:
        return float(self.processing_times[op_id, machine_id])

    # ------------------------------------------------------------------
    # Grid conversion
    # ------------------------------------------------------------------

    def _pt_int(self, op_id: int, machine_id: int) -> int:
        """Process times are integral in this generator; scale exactly."""
        return int(round(self.processing_times[op_id, machine_id] * self.scale))

    def _release_int(self, job_id: int) -> int:
        return int(math.ceil(self.release_times[job_id] * self.scale - 1e-9))

    def _due_int(self, job_id: int) -> int:
        return int(round(self.due_dates[job_id] * self.scale))

    def _fixed_spans_by_op(self) -> Dict[int, Tuple[int, int]]:
        """Gridded ``[start, end)`` span of every fixed op, keyed by op id.

        Rounded OUTWARD (floor the start, ceil the end) so a fixed operation's
        grid span always *contains* its true float window. Round-to-nearest is
        unsafe here: it can shrink the span at either end and hand the solver a
        phantom gap on the machine, which the exact replay then has to undo.
        (Real failure this fixes: a fixed op truly ending at 8.98089 rounds to
        9.0, the next fixed op truly starting at 9.96177 rounds to 10.0, and
        CP happily slots a pt=1 operation into a gap that is really 0.98 wide —
        the repair must push it past both, costing 27% of the objective.)

        Consequence to keep in mind when reading results: the model is now a
        conservative relaxation-free *restriction* of the true problem, so its
        optimum is an upper bound on the true optimum (tight to O(1/scale)) and
        its ``objective_bound`` bounds the gridded problem, not the float one.

        Spans are clipped to stay pairwise disjoint per machine. Two fixed ops
        that truly abut (a.end == b.start, off-grid) would otherwise widen into
        a one-tick overlap and make the model trivially infeasible.
        """
        if self._fixed_spans_cache is not None:
            return self._fixed_spans_cache

        by_machine: Dict[int, List[Tuple[float, float, int]]] = {}
        for op in self.fixed_operations:
            op_id = int(op['op_id'])
            mch = int(op['machine_id'])
            t0 = float(op['start_time'])
            t1 = float(op.get('end_time',
                              t0 + self.get_processing_time(op_id, mch)))
            by_machine.setdefault(mch, []).append((t0, max(t1, t0), op_id))

        spans: Dict[int, Tuple[int, int]] = {}
        for mch, items in by_machine.items():
            items.sort()
            prev_hi = 0
            for t0, t1, op_id in items:
                lo = max(0, int(math.floor(t0 * self.scale)), prev_hi)
                hi = max(int(math.ceil(t1 * self.scale)), lo)
                if hi == lo and t1 > t0:
                    self.build_warnings.append(
                        f'fixed ops overlap on machine {mch}; op {op_id} was '
                        f'clipped to a zero-length span')
                spans[op_id] = (lo, hi)
                prev_hi = hi

        self._fixed_spans_cache = spans
        return spans

    def _fixed_start_int(self, op: Dict) -> int:
        return self._fixed_spans_by_op()[int(op['op_id'])][0]

    def _fixed_end_int(self, op: Dict) -> int:
        return self._fixed_spans_by_op()[int(op['op_id'])][1]

    def _compute_horizon(self) -> int:
        """A safe upper bound on any completion time, on the integer grid.

        Anchored on the LAST obstacle in time, not just on how much work there
        is. Summing breakdown *durations* while ignoring when they start breaks
        in two ways, both of which showed up in ``verification/difftest_cp_model.py``:

        - A machine going down at t=500 in an instance holding 50 units of work
          pushes operations past 500, the interval domains cap out below that,
          and CP-SAT reports Infeasible on a feasible instance.
        - Worse, ``_breakdown_blocks`` clips each window to the horizon and
          drops it when nothing is left, so a breakdown starting beyond the
          horizon is silently discarded and CP plans straight through real
          downtime.

        Past ``anchor`` every usable machine is free, so running the entire shop
        sequentially from there is always possible; ``downtime`` is slack on top.
        """
        worst_pt = sum(
            max(self._pt_int(i, k) for k in self.compatible_machines[i])
            for i in range(self.num_operations)
        )
        latest_release = max(
            [self._release_int(j) for j in range(self.num_jobs)] + [0])
        latest_fixed_end = max(
            [self._fixed_end_int(op) for op in self.fixed_operations] + [0])

        downtime = 0
        latest_breakdown_end = 0
        for bd in self.machine_breakdowns:
            start = float(bd['start_time'])
            end = float(bd.get('end_time', np.inf))
            if np.isfinite(end):
                downtime += max(0, int(math.ceil((end - start) * self.scale)))
                latest_breakdown_end = max(
                    latest_breakdown_end, int(math.ceil(end * self.scale)))
            else:
                # Never-ending: the block runs start -> horizon, so the horizon
                # has to clear the start or the block collapses to nothing and
                # the dead machine looks available.
                latest_breakdown_end = max(
                    latest_breakdown_end, int(math.ceil(start * self.scale)))

        anchor = max(latest_release, latest_fixed_end, latest_breakdown_end)
        return int(anchor + worst_pt + downtime + 1)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self,
                    time_limit: Optional[float] = None,
                    num_workers: Optional[int] = None,
                    verbose: Optional[int] = None,
                    seed: Optional[int] = None,
                    relative_gap: Optional[float] = None):
        """Create variables, constraints and objective.

        Omitted solver parameters retain the installed OR-Tools defaults.
        ``build_model(time_limit=1800)`` sets only ``max_time_in_seconds``.

        Args:
            time_limit: wall-clock budget in seconds (None = unlimited).
            num_workers: optional CP-SAT search-worker override. None leaves
                automatic worker selection to OR-Tools. Report any override
                alongside results because it affects the search.
            verbose: optional search-log override; 1 enables the log.
            seed: CP-SAT random seed. With ``num_workers > 1`` and a wall-clock
                limit the run is still only *approximately* reproducible: which
                worker wins the race depends on machine load. Use
                ``num_workers=1`` for bit-exact repeats.
            relative_gap: optional early-stop relative MIP gap.
        """
        self.horizon = self._compute_horizon()
        model = self.model

        # --- Master/local intervals from the official FJSP example ---
        machine_intervals: List[List[Any]] = [[] for _ in range(self.num_machines)]
        self.start_vars = []
        self.end_vars = []

        for i in range(self.num_operations):
            s = model.NewIntVar(0, self.horizon, f's{i}')
            e = model.NewIntVar(0, self.horizon, f'e{i}')
            self.start_vars.append(s)
            self.end_vars.append(e)

            fx = self.fixed_by_id.get(i)
            sizes = {}
            for k in self.compatible_machines[i]:
                if fx is not None and int(fx['machine_id']) == k:
                    # Already executed: occupy the OUTWARD-rounded span, so the
                    # machine is never advertised as free during a moment it is
                    # really busy. Only reachable when present, and presence is
                    # pinned to 1 just below.
                    lo, hi = self._fixed_spans_by_op()[i]
                    sizes[k] = hi - lo
                else:
                    sizes[k] = self._pt_int(i, k)

            duration = model.NewIntVar(min(sizes.values()), max(sizes.values()),
                                       f'd{i}')
            master_interval = model.NewIntervalVar(s, duration, e, f'iv{i}')
            if len(sizes) > 1:
                literals = []
                for k, size in sizes.items():
                    pres = model.NewBoolVar(f'x{i}_{k}')
                    local_start = model.NewIntVar(0, self.horizon, f's{i}_{k}')
                    local_end = model.NewIntVar(0, self.horizon, f'e{i}_{k}')
                    interval = model.NewOptionalIntervalVar(
                        local_start, size, local_end, pres, f'iv{i}_{k}')
                    model.Add(s == local_start).OnlyEnforceIf(pres)
                    model.Add(duration == size).OnlyEnforceIf(pres)
                    model.Add(e == local_end).OnlyEnforceIf(pres)
                    machine_intervals[k].append(interval)
                    self.presence[(i, k)] = pres
                    literals.append(pres)
                model.AddExactlyOne(literals)
            else:
                k = next(iter(sizes))
                machine_intervals[k].append(master_interval)
                self.presence[(i, k)] = model.NewConstant(1)

        # --- Fixed (already executed) operations ---
        for op in self.fixed_operations:
            op_id = int(op['op_id'])
            mch = int(op['machine_id'])
            if (op_id, mch) not in self.presence:
                raise ValueError(
                    f"fixed op {op_id} is pinned to machine {mch}, which is not "
                    f"in its compatible set {self.compatible_machines[op_id]}")
            model.Add(self.presence[(op_id, mch)] == 1)
            model.Add(self.start_vars[op_id] == self._fixed_start_int(op))

        # --- Machine capacity, including breakdown blocking intervals ---
        for k in range(self.num_machines):
            intervals = list(machine_intervals[k])
            for start_i, end_i in self._breakdown_blocks(k):
                intervals.append(model.NewIntervalVar(
                    start_i, end_i - start_i, end_i, f'bd{k}_{start_i}'))
            if intervals:
                model.AddNoOverlap(intervals)

        # --- Intra-job precedence ---
        for j in range(self.num_jobs):
            ops = self.job_operations[j]
            for pos in range(1, len(ops)):
                prev_op, this_op = ops[pos - 1], ops[pos]
                if (self.is_operation_fixed(prev_op)
                        and self.is_operation_fixed(this_op)):
                    # Both already executed, so precedence is a historical fact
                    # rather than a decision. Posting it anyway is not merely
                    # redundant: two ops that truly abut round outward into a
                    # one-tick overlap, which would contradict the pins and make
                    # the whole model infeasible.
                    continue
                model.Add(self.start_vars[this_op] >= self.end_vars[prev_op])

        # --- Release times ---
        for j in range(self.num_jobs):
            first_op = int(self.job_first_op_id[j])
            release = self._release_int(j)
            if release > 0 and not self.is_operation_fixed(first_op):
                model.Add(self.start_vars[first_op] >= release)

        # --- Makespan and tardiness ---
        self.C_max = model.NewIntVar(0, self.horizon, 'C_max')
        last_ends = [self.end_vars[int(self.job_last_op_id[j])]
                     for j in range(self.num_jobs)]
        model.AddMaxEquality(self.C_max, last_ends)

        self.T = []
        for j in range(self.num_jobs):
            t = model.NewIntVar(0, self.horizon, f'T{j}')
            model.AddMaxEquality(
                t, [last_ends[j] - self._due_int(j), 0])
            self.T.append(t)

        # --- Objective (integerised weights) ---
        mw = int(round(self.makespan_weight * self.obj_scale))
        tw = int(round(self.twt_weight * self.obj_scale))
        twt_terms = [int(round(self.weights[j])) * self.T[j]
                     for j in range(self.num_jobs)]
        model.Minimize(mw * self.C_max + tw * sum(twt_terms))

        # --- Solver parameters ---
        self.solver = cp_model.CpSolver()
        params = self.solver.parameters
        if time_limit is not None:
            params.max_time_in_seconds = float(time_limit)
        if num_workers is not None:
            params.num_search_workers = int(num_workers)
        if verbose is not None:
            params.log_search_progress = bool(verbose)
        if seed is not None:
            # CP-SAT's random_seed proto field is int32, while derive_seed()
            # yields a uint32 — anything above 2^31-1 raises "Value out of
            # range". Fold into the signed range instead of truncating, so
            # distinct derived seeds stay distinct.
            params.random_seed = int(seed) % (2 ** 31 - 1)
        if relative_gap is not None:
            params.relative_gap_limit = float(relative_gap)

    @staticmethod
    def _merge_intervals(spans: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        """Sort and union a list of ``[start, end)`` spans; touching spans merge."""
        ordered = sorted((s, e) for s, e in spans if e > s)
        merged: List[Tuple[Any, Any]] = []
        for s, e in ordered:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def _fixed_busy_int(self, machine_id: int) -> List[Tuple[int, int]]:
        """Gridded spans a machine's fixed operations occupy (see
        :meth:`_fixed_spans_by_op`). Already disjoint; merged so that spans
        which abut collapse and the carve-out below cannot leave a zero-width
        sliver between them."""
        spans = self._fixed_spans_by_op()
        return self._merge_intervals([
            spans[int(op['op_id'])] for op in self.fixed_operations
            if int(op['machine_id']) == machine_id
        ])

    def _breakdown_blocks(self, machine_id: int) -> List[Tuple[int, int]]:
        """Blocking windows for a machine, with fixed-op time carved out.

        Already-executed operations are exempt from breakdown constraints
        because they physically happened. ``AddNoOverlap`` cannot exempt selectively,
        so instead of dropping the whole window we subtract the segments the
        fixed ops occupy and post what remains. Non-fixed operations are still
        excluded from the real downtime, and no spurious infeasibility is
        introduced.

        The blocks are posted as *mandatory* intervals into the same
        ``AddNoOverlap`` as the operation intervals, so they must come out
        pairwise disjoint — two overlapping blocks make the model trivially
        infeasible regardless of the instance. Both the downtime windows and
        the fixed-op busy spans are therefore merged before the carve-out. Two
        overlapping windows on one machine are not a corner case: that is
        exactly what ``segment_util.absorb_breakdowns`` emits when a machine is
        hit by two breakdown waves.
        """
        busy = self._fixed_busy_int(machine_id)

        windows: List[Tuple[int, int]] = []
        for bd in self.machine_breakdowns:
            if int(bd['machine_idx']) != machine_id:
                continue
            raw_start = float(bd['start_time'])
            raw_end = float(bd.get('end_time', np.inf))
            start = max(0, int(math.floor(raw_start * self.scale)))
            end = (self.horizon if not np.isfinite(raw_end)
                   else int(math.ceil(raw_end * self.scale)))
            end = min(end, self.horizon)
            if end > start:
                windows.append((start, end))

        blocks: List[Tuple[int, int]] = []
        for start, end in self._merge_intervals(windows):
            # Subtract the fixed-op segments overlapping this window.
            cursor = start
            for bs, be in busy:
                if be <= cursor or bs >= end:
                    continue
                if bs > cursor:
                    blocks.append((cursor, min(bs, end)))
                cursor = max(cursor, be)
                if cursor >= end:
                    break
            if cursor < end:
                blocks.append((cursor, end))
        return [(s, e) for s, e in blocks if e > s]

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def add_hint(self, schedule: Dict[int, List[Tuple[int, float, float]]]):
        """Warm-start from an existing schedule (e.g. the RL policy's plan).

        ``schedule`` uses the same shape ``solve()`` returns:
        ``machine_id -> [(op_id, start, end), ...]``. Ops missing from the hint
        are simply left unhinted — CP-SAT accepts partial hints.
        """
        if self.C_max is None:
            raise ValueError("call build_model() before add_hint()")
        for machine_id, ops in schedule.items():
            for op_id, start, _end in ops:
                key = (int(op_id), int(machine_id))
                if key not in self.presence:
                    continue
                # Single-machine presences share the constant 1. It is already
                # fixed; hinting it repeatedly would duplicate a hint variable.
                if len(self.compatible_machines[int(op_id)]) > 1:
                    self.model.AddHint(self.presence[key], 1)
                self.model.AddHint(self.start_vars[int(op_id)],
                                   int(round(float(start) * self.scale)))

    # ------------------------------------------------------------------
    # Solve and extraction
    # ------------------------------------------------------------------

    def solve(self) -> Dict[str, Any]:
        """Solve and return performance metrics and the operation schedule.

        Returns:
            {
              "performance": {status, objective, solve_time, makespan,
                              total_weighted_tardiness, objective_bound,
                              relative_gap, num_workers, ...},
              "schedule_result": {machine_id: [(op_id, start, end), ...]}
            }
        """
        if self.solver is None:
            raise ValueError("Model not built. Call build_model() first.")

        self._solve_start_time = time.time()
        self._status = self.solver.Solve(self.model)
        total_solve_time = time.time() - self._solve_start_time

        has_solution = self._status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        makespan = None
        twt = None
        objective = None
        model_objective = None
        objective_bound = None
        relative_gap = None
        schedule_result: Dict[int, List[Tuple[int, float, float]]] = {}

        if has_solution:
            schedule_result = self._extract_machine_schedule()
            schedule_result = self._repair_schedule_exact(schedule_result)
            makespan, twt = self._measure(schedule_result)
            # Reported objective is measured on the exact float schedule.
            objective = self.makespan_weight * makespan + self.twt_weight * twt
            # Optimality is a property of the model CP-SAT actually solved (the
            # integer grid with rounded due dates), so the bound and gap are
            # reported in those terms — converted back to real units. Mixing
            # the float objective with the integer bound would show a spurious
            # non-zero gap on a proven-optimal solve.
            denom = float(self.obj_scale * self.scale)
            model_objective = self.solver.ObjectiveValue() / denom
            objective_bound = self.solver.BestObjectiveBound() / denom
            if abs(model_objective) > 1e-12:
                relative_gap = max(
                    0.0,
                    (model_objective - objective_bound) / abs(model_objective))

        performance = {
            "status": self._get_status_string(self._status, has_solution),
            "objective": objective,
            "solve_time": total_solve_time,
            "makespan": makespan,
            "total_weighted_tardiness": twt,
            # Objective of the gridded model CP-SAT minimised. Differs from
            # ``objective`` only by due-date rounding and by the exact-arithmetic
            # left-shift in _repair_schedule_exact (which can only improve it).
            "model_objective": model_objective,
            "objective_bound": objective_bound,
            "relative_gap": relative_gap,
            "num_workers": int(self.solver.parameters.num_search_workers),
            "num_branches": self.solver.NumBranches(),
            "num_conflicts": self.solver.NumConflicts(),
            "time_scale": self.scale,
            # True only when every time-valued input already lands exactly on
            # the tick grid. When it is False, status='Optimal' means "optimal
            # for the gridded model", NOT "proven optimal for the float
            # instance" — conservative rounding can erase sub-tick slack and
            # cost an unbounded amount of objective while still reporting
            # relative_gap=0. Check this before citing a CP result as an exact
            # optimum, and raise solver['cp_time_scale'] if it is False.
            "grid_exact": self._inputs_on_grid(),
            "warnings": list(self.build_warnings),
        }
        return {"performance": performance, "schedule_result": schedule_result}

    def _inputs_on_grid(self, tol: float = 1e-9) -> bool:
        """Does every time-valued input land exactly on the integer tick grid?

        Process times are integral by construction, so only the quantities
        the rescheduling path introduces can fall between ticks: release times,
        due dates, fixed-operation starts/ends and breakdown boundaries.
        """
        def on_grid(value: float) -> bool:
            scaled = float(value) * self.scale
            return abs(scaled - round(scaled)) <= tol

        for arr in (self.release_times, self.due_dates):
            if not all(on_grid(v) for v in np.asarray(arr).ravel()):
                return False
        for op in self.fixed_operations:
            start = float(op['start_time'])
            end = float(op.get('end_time', start + self.get_processing_time(
                int(op['op_id']), int(op['machine_id']))))
            if not (on_grid(start) and on_grid(end)):
                return False
        for bd in self.machine_breakdowns:
            if not on_grid(float(bd['start_time'])):
                return False
            end = float(bd.get('end_time', np.inf))
            if np.isfinite(end) and not on_grid(end):
                return False
        return True

    def _extract_machine_schedule(self) -> Dict[int, List[Tuple[int, float, float]]]:
        """Read the gridded solution back as machine -> ordered op list."""
        schedule: Dict[int, List[Tuple[int, float, float]]] = {}
        for i in range(self.num_operations):
            assigned = None
            for k in self.compatible_machines[i]:
                if self.solver.Value(self.presence[(i, k)]) == 1:
                    assigned = k
                    break
            if assigned is None:
                continue
            start = self.solver.Value(self.start_vars[i]) / self.scale
            end = self.solver.Value(self.end_vars[i]) / self.scale
            schedule.setdefault(assigned, []).append((i, float(start), float(end)))
        for machine_id in schedule:
            schedule[machine_id].sort(key=lambda t: t[1])
        return schedule

    def _repair_schedule_exact(
        self, schedule: Dict[int, List[Tuple[int, float, float]]]
    ) -> Dict[int, List[Tuple[int, float, float]]]:
        """Replay the CP sequence on exact float data, left-shifting each op.

        Keeps CP's decisions (machine assignment and per-machine order) and
        recomputes start times in float arithmetic against the *true* release
        times, breakdown windows and fixed-op positions. This removes any grid
        rounding from the reported schedule and can only tighten it, since each
        operation is placed at the earliest time its own constraints allow.
        """
        assignment: Dict[int, int] = {}
        order: Dict[int, List[int]] = {}
        for machine_id, ops in schedule.items():
            order[machine_id] = [op_id for op_id, _s, _e in ops]
            for op_id, _s, _e in ops:
                assignment[op_id] = machine_id

        op_end: Dict[int, float] = {}
        op_start: Dict[int, float] = {}
        machine_free: Dict[int, float] = {m: 0.0 for m in order}
        cursor = {m: 0 for m in order}
        remaining = sum(len(v) for v in order.values())
        guard = remaining + 1

        while remaining > 0 and guard > 0:
            guard -= 1
            progressed = False
            for machine_id, seq in order.items():
                idx = cursor[machine_id]
                if idx >= len(seq):
                    continue
                op_id = seq[idx]
                job_id = int(self.op_to_job[op_id])
                pos = int(self.op_position[op_id])

                # Job precedence: previous operation of the same job.
                if pos > 0:
                    prev_op = self.job_operations[job_id][pos - 1]
                    if prev_op not in op_end:
                        continue  # not placed yet; another machine must go first
                    earliest = op_end[prev_op]
                else:
                    earliest = float(self.release_times[job_id])

                earliest = max(earliest, machine_free[machine_id])

                fixed = self.fixed_by_id.get(op_id)
                if fixed is not None:
                    # Already executed: keep its exact original position.
                    start = float(fixed['start_time'])
                    pt = self.get_processing_time(op_id, machine_id)
                    end = float(fixed.get('end_time', start + pt))
                else:
                    pt = self.get_processing_time(op_id, machine_id)
                    start = self._push_past_obstacles(machine_id, earliest, pt)
                    end = start + pt

                op_start[op_id] = start
                op_end[op_id] = end
                # max(), never plain assignment: a fixed op is pinned to its
                # original time, which can lie *before* the current cursor.
                # Assigning would rewind the cursor and let the next operation
                # overlap the one already placed.
                machine_free[machine_id] = max(machine_free[machine_id], end)
                cursor[machine_id] = idx + 1
                remaining -= 1
                progressed = True

            if not progressed:
                # Should be unreachable: CP's solution is acyclic, so some
                # machine head is always placeable. Fall back to the gridded
                # schedule rather than looping or emitting a partial plan.
                self.build_warnings.append(
                    'exact repair stalled; returning gridded schedule')
                return schedule

        repaired: Dict[int, List[Tuple[int, float, float]]] = {}
        for machine_id, seq in order.items():
            repaired[machine_id] = [
                (op_id, op_start[op_id], op_end[op_id]) for op_id in seq
            ]
            repaired[machine_id].sort(key=lambda t: t[1])
        return repaired

    def _push_past_obstacles(self, machine_id: int, earliest: float,
                             pt: float) -> float:
        """Earliest start >= ``earliest`` on ``machine_id`` clear of obstacles.

        Obstacles are the machine's downtime windows *and* the exact float
        windows of its already-executed operations. Fixed operations have to be
        obstacles here even though the per-machine sequence nominally orders
        them: they are replayed at their original float start, which the CP grid
        only approximates, so a left-shifted neighbour can otherwise be placed
        on top of one. Skipping this is what let the repaired schedule report an
        objective below the true optimum.
        """
        windows = [
            (float(bd['start_time']), float(bd.get('end_time', np.inf)))
            for bd in self.machine_breakdowns
            if int(bd['machine_idx']) == machine_id
        ]
        for op in self.fixed_operations:
            if int(op['machine_id']) != machine_id:
                continue
            f_start = float(op['start_time'])
            f_end = float(op.get(
                'end_time',
                f_start + self.get_processing_time(int(op['op_id']), machine_id)))
            if f_end > f_start:
                windows.append((f_start, f_end))
        if not windows:
            return earliest
        windows.sort()
        start = earliest
        moved = True
        while moved:
            moved = False
            for w_start, w_end in windows:
                if not np.isfinite(w_end):
                    w_end = float('inf')
                # Overlap test: [start, start+pt) intersects [w_start, w_end)
                if start < w_end and start + pt > w_start:
                    if not np.isfinite(w_end):
                        raise ValueError(
                            f"machine {machine_id} is permanently down from "
                            f"{w_start}; CP assigned an operation to it")
                    start = w_end
                    moved = True
        return start

    def _measure(self, schedule: Dict[int, List[Tuple[int, float, float]]]
                 ) -> Tuple[float, float]:
        """Makespan and total weighted tardiness of an exact-time schedule."""
        job_completion = np.zeros(self.num_jobs, dtype=np.float64)
        makespan = 0.0
        for ops in schedule.values():
            for op_id, _start, end in ops:
                job_id = int(self.op_to_job[op_id])
                job_completion[job_id] = max(job_completion[job_id], end)
                makespan = max(makespan, end)
        tardiness = np.maximum(0.0, job_completion - self.due_dates)
        twt = float(np.sum(self.weights * tardiness))
        return float(makespan), twt

    def _get_status_string(self, status, has_solution: bool) -> str:
        """Map CP-SAT status to the result table's labels.

        Downstream CSV/plot code keys off these exact strings — in particular
        'Timeout No Feasible' is how a run with no incumbent is dropped from
        plots instead of being read as a zero-objective win.
        """
        if status == cp_model.OPTIMAL:
            return "Optimal"
        if status == cp_model.FEASIBLE:
            return "Feasible (Time Limit)"
        if status == cp_model.INFEASIBLE:
            return "Infeasible"
        if status == cp_model.MODEL_INVALID:
            return "Model Invalid"
        return "Timeout No Feasible" if not has_solution else "Unknown"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_schedule_for_gantt(self) -> List[Dict]:
        """Schedule rows in the Gantt utility's input format."""
        if self.solver is None or self._status not in (
                cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []
        schedule = self._repair_schedule_exact(self._extract_machine_schedule())
        rows: List[Dict] = []
        for machine_id, ops in schedule.items():
            for op_id, start, end in ops:
                rows.append({
                    'op_id': int(op_id),
                    'job_id': int(self.op_to_job[op_id]),
                    'machine_id': int(machine_id),
                    'start_time': float(start),
                    'end_time': float(end),
                    'processing_time': self.get_processing_time(op_id, machine_id),
                    'is_fixed': self.is_operation_fixed(op_id),
                })
        rows.sort(key=lambda r: (r['machine_id'], r['start_time']))
        return rows


# Public names for dynamic and CP-SAT callers.
DynamicCP = CP
CPSAT = CP
