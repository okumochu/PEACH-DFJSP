"""
Curriculum Learning Pacer for Multi-Criteria FJSP Training.

Single-stage schedule ("CL"): ramp due-date tightness from a loose initial
range to the target range. The two bounds are tightened SEQUENTIALLY, not at
the same time: the ramp is split into two sub-phases by ``ddt_min_substage_frac``.

  a. Sub-phase A ramps ``ddt_min`` from initial -> target while ``ddt_max`` is
     held at its initial (loose) value. Lowering the floor first widens the
     range, introducing a hard tail of tight jobs.
  b. Sub-phase B ramps ``ddt_max`` from initial -> target while ``ddt_min`` is
     held at its target value. Lowering the ceiling then narrows the range,
     removing the easy (loose) jobs.

This "widen then narrow" shape mirrors the original multi-phase pacer (which
ramped one bound per phase) instead of moving both bounds together. The ramp
spans the first ``ddt_stage_ratio`` fraction of training; the remainder holds
the target range.

Dynamic events are NOT curriculum-ramped. With the anchored-segment training
mechanism (each rollout is a chain of linked static episodes split at event
boundaries), events are present at their target distribution from update 0;
there is no separate event-introduction stage. ``get_event_params`` therefore
returns the target event ranges every update (or all-zero wave counts when
``events_enabled`` is False, e.g. the old/static design ablation).
"""

import math
from typing import Any, Dict, Tuple


class CurriculumPacer:
    """Progressive DDT pacer (single stage) for curriculum learning."""

    _EVENT_COUNT_KEYS = (
        'num_new_job_arrival_waves',
        'num_machine_breakdown_waves',
        'num_priority_change_waves',
    )
    _EVENT_PERIOD_KEYS = (
        'new_job_lambda_inv',
        'mtbf',
        'priority_change_interval',
    )
    _EVENT_AMOUNT_KEYS = (
        'new_jobs_per_arrival',
        'machines_per_breakdown',
        'priority_jobs_per_change',
        'new_job_weight',
        'new_job_due_date_tightness',
        'priority_new_weight',
        'repair_duration',
    )

    def __init__(
        self,
        max_updates: int,
        target_ddt_range: Tuple[float, float],
        initial_ddt_range: Tuple[float, float] = (1.0, 2.5),
        ddt_stage_ratio: float = 0.60,
        ddt_min_substage_frac: float = 0.5,
        target_event_ranges: Dict[str, Tuple[Any, Any]] = None,
        events_enabled: bool = True,
    ):
        self.max_updates = max(max_updates, 1)
        self.target_ddt_min, self.target_ddt_max = self._clean_float_range(
            *target_ddt_range)
        self.initial_ddt_min, self.initial_ddt_max = self._clean_float_range(
            *initial_ddt_range)
        # Fraction of training over which DDT ramps (then holds target).
        # Clamped to (0, 1].
        self.ddt_stage_ratio = max(1e-6, min(1.0, float(ddt_stage_ratio)))
        # Fraction of the DDT ramp spent on ddt_min (sub-phase A) before ddt_max
        # (sub-phase B). Clamped to [0, 1].
        self.ddt_min_substage_frac = max(
            0.0, min(1.0, float(ddt_min_substage_frac)))
        # When False, dynamic-event wave counts are held at zero (old/static
        # design); periods/amounts are still surfaced (harmless when counts=0).
        self.events_enabled = bool(events_enabled)

        target_event_ranges = target_event_ranges or {}
        self.target_event_ranges: Dict[str, Tuple[Any, Any]] = {}
        for key, value in target_event_ranges.items():
            self.target_event_ranges[key] = self._clean_range(*value)

        self._ddt_end = self.ddt_stage_ratio

    @staticmethod
    def _s_curve(t: float) -> float:
        """Cosine S-curve interpolation in [0, 1]."""
        t = max(0.0, min(1.0, t))
        return 0.5 * (1.0 - math.cos(math.pi * t))

    @staticmethod
    def _clean_float_range(lo: Any, hi: Any) -> Tuple[float, float]:
        lo = float(lo)
        hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    @staticmethod
    def _clean_range(lo: Any, hi: Any) -> Tuple[Any, Any]:
        if isinstance(lo, int) and isinstance(hi, int):
            lo = int(lo)
            hi = int(hi)
        else:
            lo = float(lo)
            hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _progress(self, current_update: int) -> float:
        return max(0.0, min(1.0, float(current_update) / self.max_updates))

    def get_ddt_range(self, current_update: int) -> Tuple[float, float]:
        progress = self._progress(current_update)
        if progress >= self._ddt_end:
            return self.target_ddt_min, self.target_ddt_max

        # Local progress over the DDT ramp in [0, 1).
        local_t = (progress / self.ddt_stage_ratio
                   if self.ddt_stage_ratio > 0 else 1.0)

        # Sequential ramp: split at ``ddt_min_substage_frac`` into sub-phase A
        # (ramp ddt_min, hold ddt_max at initial) then sub-phase B (hold ddt_min
        # at target, ramp ddt_max). The two sub-phases meet continuously at
        # (target_ddt_min, initial_ddt_max).
        split = self.ddt_min_substage_frac
        if local_t < split:
            sub_t = local_t / split if split > 0.0 else 1.0
            s = self._s_curve(sub_t)
            ddt_min = self.initial_ddt_min + s * (
                self.target_ddt_min - self.initial_ddt_min)
            return ddt_min, self.initial_ddt_max

        denom = 1.0 - split
        sub_t = (local_t - split) / denom if denom > 0.0 else 1.0
        s = self._s_curve(sub_t)
        ddt_max = self.initial_ddt_max + s * (
            self.target_ddt_max - self.initial_ddt_max)
        return self.target_ddt_min, ddt_max

    def get_event_params(self, current_update: int) -> Dict[str, Tuple[Any, Any]]:
        """Return generic event ranges for this update.

        Counts are the TARGET ranges throughout (events are present from update
        0 via the anchored-segment chain), or all-zero when events are disabled.
        Periods/amounts are always the configured target ranges.
        """
        event_params: Dict[str, Tuple[Any, Any]] = {}
        for key in self._EVENT_COUNT_KEYS:
            if key not in self.target_event_ranges:
                continue
            event_params[key] = (
                self.target_event_ranges[key] if self.events_enabled
                else (0, 0))
        for key in self._EVENT_PERIOD_KEYS:
            if key in self.target_event_ranges:
                event_params[key] = self.target_event_ranges[key]
        for key in self._EVENT_AMOUNT_KEYS:
            if key in self.target_event_ranges:
                event_params[key] = self.target_event_ranges[key]
        return event_params

    def describe(self, max_updates: int = None) -> str:
        mu = max_updates or self.max_updates
        ddt_end = int(self._ddt_end * mu)
        split_update = int(self._ddt_end * self.ddt_min_substage_frac * mu)
        lines = [
            "CurriculumPacer schedule (single-stage DDT):",
            f"  Ramp A [0, {split_update}): "
            f"ddt_min {self.initial_ddt_min:.2f} -> {self.target_ddt_min:.2f}, "
            f"ddt_max held at {self.initial_ddt_max:.2f}",
            f"  Ramp B [{split_update}, {ddt_end}): "
            f"ddt_min held at {self.target_ddt_min:.2f}, "
            f"ddt_max {self.initial_ddt_max:.2f} -> {self.target_ddt_max:.2f}",
            f"  Hold [{ddt_end}, {mu}]: target ddt "
            f"({self.target_ddt_min:.2f}, {self.target_ddt_max:.2f})",
            f"  Dynamic events: "
            f"{'target distribution every update' if self.events_enabled else 'disabled (wave counts = 0)'}",
        ]
        return "\n".join(lines)
