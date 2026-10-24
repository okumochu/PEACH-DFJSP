"""Scale curriculum: grow the training problem size when the critic is ready.

Motivation (measured, 2026-07-27/28): fine-tuning the 20x10 PEACH checkpoint
directly at 50x10 diverges — value targets are ~4x larger, the inherited
critic starts at value_loss ~41 (vs ~1.5 at 20x10) and never recovers; a
20-update critic-only warm-up merely delayed the same divergence by exactly
its own length. The bottleneck is the value function's adjustment to a
RETURN-SCALE jump, so this module removes the jump: train at a ladder of
sizes, and only step up when the value loss at the current size has
converged and is stable.

The controller is pure bookkeeping over the per-update value_loss series —
no torch, no env — so its gate logic is unit-testable on CPU. The trainer
owns the actual stage switch (rebuild data generator + env; see
HGTTrainer._apply_scale_stage).

Gate per stage (all knobs configurable):
  ADVANCE  when the stage has run >= min_stage_updates AND the last
           ``window`` value losses are a plateau (median within plateau_tol
           of the previous window's median) AND stable (MAD/median <=
           cv_tol). Median/MAD, not mean/CV: healthy PPO value loss here has
           a flat baseline with intermittent spikes (the successful direct
           30x10 fine-tune shows 0.6..14.4 in ten updates), and mean/CV
           flags that profile as unstable while median/MAD sees through
           single spikes yet still blocks a systematic climb.
  DESCEND  when the critic cannot cope at this size and there is a smaller
           stage to fall back to. Two triggers, both dimensionless and both
           requiring the EV gate: a sustained collapse (EV below -1 for
           div_patience updates, the fast case) and a slow decline (median EV
           below 0 over the window, the patient case). Descending replaces
           aborting everywhere except the first stage, where there is nowhere
           to go and the decision is DIVERGED.
  DIVERGED when value_loss exceeds div_factor x the stage's reference level
           (median of its first ref_probe updates) for div_patience
           consecutive updates. The 50x10 failure fingerprint (41 -> 4726)
           trips this within a few updates. Under the EV gate the test is
           EV < -1 instead, and only the first stage still aborts on it.
  STUCK    when max_stage_updates pass without a plateau. The run stops and
           reports rather than stepping up on an unconverged critic — "how
           far the ladder gets" is itself the experiment's answer.
  HOLD     otherwise. The final stage never advances and never goes STUCK:
           it trains until the global update budget, guarded only by
           DIVERGED — unless final_hold_updates is set, in which case the
           final stage returns DONE (graceful, successful stop) after that
           many stage updates. The knob exists because update cost grows
           ~quadratically with problem area (100x20 = ~150 min/update vs
           ~2 min at 20x10): with an update-count budget alone, whatever
           the gates do not consume lands on the most expensive stage, so
           wall clock would swing by days depending on how fast the early
           gates cleared. final_hold_updates pins the expensive tail; give
           --max_updates slack so it never binds first. It counts updates on
           the final stage CUMULATIVELY across visits, because a descend and
           return would otherwise restart the hold from zero and the bound
           would not bound anything. It is read from config at every
           (re)launch, NOT from the resume snapshot, so a resume with a
           larger value extends the hold of a finished run.
"""

import re
import math
from statistics import median


# Decisions returned by ScaleCurriculumController.observe().
HOLD = 'hold'
ADVANCE = 'advance'
DIVERGED = 'diverged'
STUCK = 'stuck'
DONE = 'done'
DESCEND = 'descend'


def parse_stage_spec(spec):
    """Parse '20x10' or '20x10:128' into (n_j, n_m, micro_batch_or_None)."""
    m = re.fullmatch(r'(\d+)x(\d+)(?::(\d+))?', spec.strip())
    if not m:
        raise ValueError(
            f"bad stage spec {spec!r}; want NjxNm or NjxNm:micro, e.g. 30x10:128")
    return int(m.group(1)), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)


def build_auto_ladder(start, target, rho=2.0, micro=None):
    """Generate a scale ladder from its endpoints and one growth factor.

    ``start`` and ``target`` are (n_j, n_m). Stage count comes from the
    total growth in problem AREA:

        K = ceil( log(A_target / A_start) / log(rho) )

    and both dimensions are interpolated GEOMETRICALLY between the
    endpoints, so the shop keeps a sensible jobs-per-machine ratio the whole
    way up instead of growing one dimension first.

    Returns a list of (n_j, n_m, micro) with the endpoints exact.
    """
    (j0, m0), (jT, mT) = start, target
    if j0 < 1 or m0 < 1 or jT < 1 or mT < 1:
        raise ValueError('sizes must be positive')
    a0, aT = j0 * m0, jT * mT
    if aT <= a0:
        raise ValueError(
            f'target area ({aT}) must exceed start area ({a0}); the ladder '
            'only climbs')
    if jT < j0 or mT < m0:
        # Area alone is not enough: 10x20 -> 100x5 grows the area while
        # SHRINKING the machine count, and geometric interpolation would then
        # walk one dimension downward. Every stage must be a superset of the
        # last in both dimensions.
        raise ValueError(
            f'both dimensions must grow: {j0}x{m0} -> {jT}x{mT} shrinks '
            f'{"jobs" if jT < j0 else "machines"}')
    if not (rho > 1.0) or not math.isfinite(rho):
        raise ValueError(f'rho must be a finite number > 1 (got {rho})')

    k = max(1, math.ceil(math.log(aT / a0) / math.log(rho)))
    stages = []
    for i in range(k + 1):
        t = i / k
        n_j = j0 if i == 0 else (jT if i == k else int(round(j0 * (jT / j0) ** t)))
        n_m = m0 if i == 0 else (mT if i == k else int(round(m0 * (mT / m0) ** t)))
        n_j, n_m = max(1, n_j), max(1, n_m)
        if stages and (n_j, n_m) == (stages[-1][0], stages[-1][1]):
            continue          # rounding collapsed this step onto the last
        stages.append((n_j, n_m, micro))
    if len(stages) < 2:
        raise ValueError('ladder collapsed to fewer than 2 stages; lower rho')
    return stages


def parse_auto_ladder_spec(spec, micro=None):
    """Parse 'START:TARGET[:rho]', e.g. '10x5:100x20:2'."""
    parts = [x for x in str(spec).split(':') if x.strip()]
    if len(parts) not in (2, 3):
        raise ValueError(
            f"--auto_ladder wants 'START:TARGET[:rho]', e.g. "
            f"'10x5:100x20:2' (got {spec!r})")
    start = parse_size_pair(parts[0])
    target = parse_size_pair(parts[1])
    rho = float(parts[2]) if len(parts) == 3 else 2.0
    return build_auto_ladder(start, target, rho, micro=micro)


def parse_size_pair(text):
    t = str(text).lower().replace('*', 'x').strip()
    if 'x' not in t:
        raise ValueError(f"size must look like '10x5' (got {text!r})")
    a, b = t.split('x', 1)
    return int(a), int(b)


def parse_stage_list(csv_spec):
    """Parse a comma-separated ladder, smallest first, e.g.
    '20x10:128,30x10:128,50x10:64'. Requires at least 2 stages and a
    non-decreasing job count (a shrinking ladder is almost certainly a typo).
    """
    stages = [parse_stage_spec(s) for s in csv_spec.split(',') if s.strip()]
    if len(stages) < 2:
        raise ValueError('scale curriculum needs at least 2 stages')
    for a, b in zip(stages, stages[1:]):
        if b[0] * b[1] < a[0] * a[1]:
            raise ValueError(
                f'stage {b[0]}x{b[1]} is smaller than its predecessor '
                f'{a[0]}x{a[1]}; ladder must be non-decreasing')
    return stages


class ScaleCurriculumController:
    """Decides, once per PPO update, whether to hold / advance / abort."""

    # Sustained explained variance below this means the critic is much worse
    # than a constant predictor. Dimensionless, so it needs no calibration
    # against the return scale.
    EV_DIVERGED_LEVEL = -1.0

    # Descend when the critic is no better than predicting the mean. Zero is
    # a principled point, not a tuned one, which is why autoCL needs no
    # separate descend threshold.
    EV_DESCEND_LEVEL = 0.0

    # Bumped when a persisted field changes meaning. See load_state_dict.
    SCHEMA_VERSION = 2

    def __init__(self, n_stages, min_stage_updates=20, max_stage_updates=80,
                 window=10, plateau_tol=0.5, cv_tol=0.8,
                 div_factor=5.0, div_patience=10, ref_probe=5,
                 final_hold_updates=None, ev_threshold=None,
                 descend_patience=None, settle_updates=None,
                 ladder_id=None, max_climb_attempts=3):
        if n_stages < 1:
            raise ValueError('need at least one stage')
        if min_stage_updates < 2 * window:
            # The plateau test compares two disjoint windows; advancing
            # earlier than 2*window would compare overlapping data.
            min_stage_updates = 2 * window
        if max_stage_updates <= min_stage_updates:
            # STUCK fires at max_stage_updates and nothing may leave a stage
            # before min_stage_updates, so max <= min means every non-final
            # stage aborts the run on the same update it becomes eligible to
            # advance. Refuse rather than discover it after hours of GPU time.
            raise ValueError(
                f'max_stage_updates ({max_stage_updates}) must exceed '
                f'min_stage_updates ({min_stage_updates}); otherwise a stage '
                'goes STUCK before it can ever advance')
        self.n_stages = int(n_stages)
        self.min_stage_updates = int(min_stage_updates)
        self.max_stage_updates = int(max_stage_updates)
        self.window = int(window)
        self.plateau_tol = float(plateau_tol)
        self.cv_tol = float(cv_tol)
        self.div_factor = float(div_factor)
        self.div_patience = int(div_patience)
        self.ref_probe = int(ref_probe)
        self.final_hold_updates = (
            None if final_hold_updates is None else int(final_hold_updates))
        # When set, advance on critic explained variance instead of the
        # value-loss plateau. EV is dimensionless, so this gate does not
        # care what the return normalizer does - the two mechanisms become
        # independent rather than 'one method with two inseparable parts'.
        if descend_patience is None:
            descend_patience = max(2 * int(window), int(min_stage_updates))
        if settle_updates is None:
            # Grace period after every stage switch. PopArt's statistics need
            # ~1/beta updates (10 at the default beta=0.1) to re-adapt to the
            # new size, during which the critic's explained variance is
            # transiently poor for a reason that has nothing to do with the
            # size being too hard. Without this the transient alone can spend
            # the divergence patience and demote a healthy stage.
            settle_updates = int(window)
        self.ev_threshold = (
            None if ev_threshold is None else float(ev_threshold))
        self.descend_patience = int(descend_patience)
        self.settle_updates = int(settle_updates)
        # Identity of the ladder this controller was built for, e.g.
        # '10x5,15x6,22x8'. Persisted and compared on resume: the stage index
        # alone cannot tell one ladder from another, and two DIFFERENT ladders
        # in this programme happen to have the same length (the queue's 100x20
        # ladder and run_scale_cl.sh's default are both 6 stages), so a resume
        # that lost its LADDER env var would otherwise pass every check and
        # train at the wrong size.
        self.ladder_id = None if ladder_id is None else str(ladder_id)
        # How many times one stage may be retried after falling back to it
        # before the ladder gives up. A rail, not a tuning knob: its job is to
        # stop an unclimbable ladder from spending the whole budget proving it,
        # and the answer the experiment wants (the high-water stage) is the
        # same on the third attempt as on the thirtieth.
        self.max_climb_attempts = int(max_climb_attempts)
        self._ev_hist = []
        self._ev_valid_run = 0
        # Counts, per stage, how many times THAT stage failed and sent the
        # policy back down. Raises the minimum dwell both at the failing stage
        # (a fairer retry) and at the stage below it (do not send the policy
        # straight back up), so an oscillating ladder settles.
        self._demotions = {}

        self.stage_idx = 0
        self.stage_updates = 0
        # How far the ladder has ever climbed. The CURRENT stage is not the
        # answer once descending exists: a run that reaches 68x16 and then
        # oscillates back down would otherwise report the bottom stage, and
        # how far the ladder got IS this experiment's result.
        self.max_stage_idx = 0
        # Updates spent on the final stage across ALL visits. final_hold_
        # updates is a wall-clock bound on the most expensive stage, so it has
        # to survive a descend-and-return cycle; stage_updates alone resets.
        self._final_stage_updates = 0
        self._hist = []          # value_loss series of the CURRENT stage
        self._div_ref = None     # median of the stage's first ref_probe losses
        self._div_run = 0        # consecutive updates above the divergence bar

    # -- helpers -----------------------------------------------------------
    @property
    def on_final_stage(self):
        return self.stage_idx >= self.n_stages - 1

    def stage_ready(self):
        """The advancement gate, also measurable on the target scale stage."""
        if self.stage_updates < self._min_dwell():
            return False
        if self.ev_threshold is not None:
            return (self._ev_valid_run >= self.window
                    and median(self._ev_hist[-self.window:]) >= self.ev_threshold)
        if len(self._hist) < 2 * self.window:
            return False
        if not all(math.isfinite(x) for x in self._hist[-2 * self.window:]):
            return False
        _cur, _prev, rel_change, cv = self._window_stats()
        return rel_change <= self.plateau_tol and cv <= self.cv_tol

    def _window_stats(self):
        """Robust window statistics: median level, MAD/median spread.

        Median/MAD instead of mean/CV because the healthy loss profile here
        is a flat baseline with intermittent spikes; a single spike moves the
        mean by an order of magnitude but barely moves the median, while a
        genuine systematic climb moves both.
        """
        cur = self._hist[-self.window:]
        prev = self._hist[-2 * self.window:-self.window]
        med_cur = median(cur)
        med_prev = median(prev)
        mad = median(abs(x - med_cur) for x in cur)
        spread = mad / max(med_cur, 1e-12)
        rel_change = abs(med_cur - med_prev) / max(med_prev, 1e-12)
        return med_cur, med_prev, rel_change, spread

    # -- API ---------------------------------------------------------------
    def observe(self, value_loss, ev=None):
        """Record one update's value_loss; return HOLD/ADVANCE/DIVERGED/STUCK."""
        vl = float(value_loss)
        self.stage_updates += 1
        if self.on_final_stage:
            self._final_stage_updates += 1
        self._hist.append(vl)
        current_ev = float(ev) if ev is not None else float('nan')
        valid_ev = math.isfinite(current_ev)
        if valid_ev:
            self._ev_hist.append(current_ev)
            self._ev_valid_run += 1
        else:
            # A window means consecutive update measurements, not the last
            # valid measurements at arbitrary ages. Keep the valid history
            # for diagnostics, but require a fresh full window after a gap.
            self._ev_valid_run = 0

        if self.ev_threshold is not None:
            # Dimensionless divergence test. The value-loss test cannot be
            # used here: _div_ref is the median of a stage's first few
            # losses, and under an adaptive normalizer those are taken while
            # sigma still belongs to the PREVIOUS stage, which inflated the
            # trip point ~18000x in review and disarmed the detector for
            # every stage after the first. EV below -1 means the critic is
            # far worse than simply predicting the mean, at any scale.
            if (self.stage_updates > self.settle_updates
                    and valid_ev
                    and current_ev < self.EV_DIVERGED_LEVEL):
                self._div_run += 1
                if self._div_run >= self.div_patience:
                    # A sudden collapse is the failure this ladder was built
                    # for (the measured 50x10 fingerprint: value_loss 41 ->
                    # 4726 within a few updates). Above the first stage the
                    # useful response is to step back down, not to abort:
                    # aborting here is what made DESCEND unreachable in
                    # exactly the case it exists to handle. On the first
                    # stage there is nowhere to descend to.
                    return DESCEND if self.stage_idx > 0 else DIVERGED
            else:
                self._div_run = 0
        else:
            if self._div_ref is None and len(self._hist) >= self.ref_probe:
                self._div_ref = median(self._hist[:self.ref_probe])

            if self._div_ref is not None and vl > self.div_factor * max(self._div_ref, 1e-12):
                self._div_run += 1
                if self._div_run >= self.div_patience:
                    return DIVERGED
            else:
                self._div_run = 0

        # The final stage's wall-clock budget is spent AND the critic is
        # coping. Both halves matter. Checked before the slow-decline descend
        # so a healthy run that has paid for its expensive tail stops promptly
        # instead of walking back down the ladder; gated on health because
        # making the counter cumulative removed the guarantee the per-visit
        # counter carried for free. A per-visit hold of 600 could not contain
        # a demotion, so reaching it meant 600 consecutive healthy updates at
        # the target size. A cumulative one can be reached on the tenth update
        # of a visit that follows thirty demotions, and without this condition
        # the same collapsing EV trace that the previous version reported as
        # DIVERGED would be reported as DONE: a critic that failed at the
        # target size, recorded as a successful curriculum completion.
        if (self.on_final_stage and self.final_hold_updates is not None
                and self._final_stage_updates >= self.final_hold_updates):
            if self._final_stage_is_healthy():
                return DONE
            # Below the bar: keep training rather than stopping. Tried making
            # this terminal and it was wrong. Updates spent HERE are spent at
            # the target size, which is what the run exists to produce, so they
            # are not waste the way oscillating between smaller stages is; and
            # a 30-update hold is far too short a window to conclude a critic
            # will never get there. Measured on a trace that dips at the final
            # stage and then recovers, the terminal version killed the run at
            # update 70. The global budget still bounds this, and the run ends
            # as budget_exhausted rather than claiming a DONE it did not earn.

        # Descend: the critic cannot cope at this size. Only meaningful
        # under the EV gate, since the test is "explains less than nothing",
        # and only possible above the first stage.
        if (self.ev_threshold is not None and self.stage_idx > 0
                and self.stage_updates >= self._min_dwell()
                and self._ev_valid_run >= self.window
                and median(self._ev_hist[-self.window:]) < self.EV_DESCEND_LEVEL
                and self.stage_updates >= self.descend_patience):
            return DESCEND

        if not self.on_final_stage:
            if self.stage_ready():
                return ADVANCE
            if self.stage_updates >= self.max_stage_updates:
                # Out of patience at this size. Above the first stage, falling
                # back is strictly better than stopping: the same policy gets
                # another attempt from a size it can handle, and the demotion
                # damping lengthens the dwell so the retry is a real one.
                #
                # This closes a dead band that the descend test alone leaves
                # open. ADVANCE needs the EV window median at or above
                # ev_threshold and DESCEND needs it below zero, so a stage
                # parked in between - a mediocre critic, not a broken one -
                # triggers neither and can only ever end as STUCK. That band
                # is exactly where a too-high threshold puts a run, which is
                # the failure this ladder is supposed to absorb rather than
                # die of.
                # ... but only for a bounded number of attempts. Descending
                # without a cap converts "this stage cannot be climbed" from an
                # early stop into an oscillation that consumes the ENTIRE
                # update budget: measured on a trace where every stage above
                # the first sits in the dead band, 20000 updates and 36
                # demotions with no terminal decision. On a multi-week run
                # that is worse than the abort it replaced, because the answer
                # (how far the ladder got) is the same either way and only the
                # wall clock differs.
                if (self.ev_threshold is not None and self.stage_idx > 0
                        and self._demotions.get(self.stage_idx, 0)
                        < self.max_climb_attempts):
                    return DESCEND
                return STUCK
        return HOLD

    def gate_metrics(self):
        """Current gate diagnostics for logging (NaN-free, always emittable)."""
        out = {
            'stage_idx': self.stage_idx,
            'stage_updates': self.stage_updates,
            'div_run': self._div_run,
            # Without these the damping is invisible in wandb: a stage whose
            # dwell has been stretched looks identical to a healthy one right
            # up to the moment the run stops.
            'max_stage_idx': self.max_stage_idx,
            'min_dwell': self._min_dwell(),
            'demotions_here': self._demotions.get(self.stage_idx, 0),
            'demotions_total': sum(self._demotions.values()),
            'final_stage_updates': self._final_stage_updates,
        }
        if self._ev_valid_run >= self.window:
            out['ev_window_median'] = median(self._ev_hist[-self.window:])
        if len(self._hist) >= 2 * self.window:
            med_cur, _med_prev, rel_change, spread = self._window_stats()
            out.update({'vl_window_median': med_cur,
                        'vl_rel_change': rel_change,
                        'vl_spread': spread})
        return out

    def _final_stage_is_healthy(self):
        """Is the critic coping well enough for a stop to mean SUCCESS?

        Only asked of the final stage, and only to decide DONE. Without the EV
        gate there is no dimensionless health signal, so the answer is yes and
        behaviour matches the pre-EV design. With it, the bar is the same
        principled level DESCEND uses: explaining at least as much of the
        return variance as predicting the mean would. The settle window is
        required for the same reason the collapse test requires it, and for
        the same reason it would be wrong to certify success on measurements
        taken while PopArt's statistics are still re-adapting to a size change.
        """
        if self.ev_threshold is None:
            return True
        if self.stage_updates < self.settle_updates:
            return False
        if self._ev_valid_run < self.window:
            return False
        # The bar is the ADVANCE threshold, not the descend level. Certifying
        # the target size at "better than predicting the mean" while every
        # stage below it had to clear ev_threshold would make the final stage
        # the LOOSEST gate on the ladder, and the final stage is the one the
        # paper reports. A run whose critic sits below the bar it used to earn
        # its way up does not get to call itself finished; it keeps training,
        # descends, or ends on the global budget as budget_exhausted.
        return median(self._ev_hist[-self.window:]) >= self.ev_threshold

    def _min_dwell(self):
        """Minimum updates at the current stage before it may be left.

        Grows with how often the ladder has fallen back through here, so an
        oscillating ladder settles instead of ping-ponging forever. Two
        counters feed it: demotions OUT of this stage (give the retry a
        fairer trial) and demotions out of the stage ABOVE (do not send the
        policy straight back up to a size it just failed).

        The result is capped below max_stage_updates. Without the cap the
        damping is fatal rather than stabilising: STUCK fires unconditionally
        at max_stage_updates, so once the dwell passes it the stage can
        neither advance nor descend and the run aborts at whatever stage the
        damping happened to reach.
        """
        d = max(self._demotions.get(self.stage_idx, 0),
                self._demotions.get(self.stage_idx + 1, 0))
        dwell = int(self.min_stage_updates * (1.5 ** d))
        if not self.on_final_stage:
            dwell = min(dwell, max(self.min_stage_updates,
                                   self.max_stage_updates - self.window))
        return dwell

    def descend(self):
        """Step back down one stage and reset per-stage state."""
        if self.stage_idx <= 0:
            raise RuntimeError('descend() called on the first stage')
        # Recorded against the stage that FAILED, not the one being dropped
        # into. Charging the lower stage instead would mean the stage that
        # actually cannot cope never gets a longer trial, and the whole cost
        # of the damping lands on the bottom of the ladder.
        self._demotions[self.stage_idx] = self._demotions.get(
            self.stage_idx, 0) + 1
        self.stage_idx -= 1
        self._reset_stage_state()

    def advance(self):
        """Move to the next stage and reset per-stage state."""
        if self.on_final_stage:
            raise RuntimeError('advance() called on the final stage')
        self.stage_idx += 1
        self.max_stage_idx = max(self.max_stage_idx, self.stage_idx)
        self._reset_stage_state()

    def _reset_stage_state(self):
        """Clear everything measured at the stage just left."""
        self.stage_updates = 0
        self._hist = []
        self._ev_hist = []
        self._ev_valid_run = 0
        self._div_ref = None
        self._div_run = 0

    # -- resume support ----------------------------------------------------
    def state_dict(self):
        return {
            # Bumped whenever the meaning of a field changes. Schema 1 (up to
            # and including c0f1165) had no cumulative final-stage counter and
            # no high-water mark, so silently defaulting them on load produces
            # a wrong answer rather than a missing one; see load_state_dict.
            'schema': self.SCHEMA_VERSION,
            'ladder_id': self.ladder_id,
            'stage_idx': self.stage_idx,
            'stage_updates': self.stage_updates,
            'max_stage_idx': self.max_stage_idx,
            'final_stage_updates': self._final_stage_updates,
            'hist': list(self._hist),
            # Without this a resume drops the EV window and the gate
            # cannot advance for another `window` updates.
            'ev_hist': list(self._ev_hist),
            'ev_valid_run': self._ev_valid_run,
            'demotions': dict(self._demotions),
            'div_ref': self._div_ref,
            'div_run': self._div_run,
        }

    def load_state_dict(self, state):
        schema = int(state.get('schema', 1))
        if schema > self.SCHEMA_VERSION:
            # The direction a ROLLBACK produces, and the one an "is it old
            # enough" check misses. A newer snapshot may carry fields this
            # build does not know about, or the same field with a changed
            # meaning, and silently ignoring them is how a resumed run drifts
            # from the run it claims to continue.
            raise ValueError(
                f'resume snapshot uses scale-curriculum schema {schema}, newer '
                f'than this build writes ({self.SCHEMA_VERSION}). It was '
                'written by a later version of this code; run that version, or '
                'start a new run.')
        if schema < self.SCHEMA_VERSION:
            # A schema-1 snapshot has no cumulative final-stage counter and no
            # high-water mark, so loading it does not merely lose information,
            # it produces WRONG numbers: a run 590 updates into a 600-update
            # final hold would restart that hold from zero (measured: 600 more
            # updates instead of 10, on the most expensive stage on the
            # ladder), and a run that had climbed to stage 3 and fallen back
            # would report its high-water mark as wherever it currently sits.
            # Refusing is also the honest answer scientifically: every
            # schema-1 ladder snapshot was written by code carrying the frozen
            # -event-magnitude defect, so its training distribution was wrong
            # at every stage below the last and it cannot be continued into a
            # result anyone would publish.
            raise ValueError(
                f'resume snapshot uses scale-curriculum schema {schema}, this '
                f'build writes schema {self.SCHEMA_VERSION}. Snapshots from '
                'before 2026-08-11 were written by code that froze the '
                'dynamic-event magnitudes at the final stage size, so the run '
                'cannot be continued into a usable result. Start a new run.')

        ladder_id = state.get('ladder_id')
        if (ladder_id is not None and self.ladder_id is not None
                and str(ladder_id) != self.ladder_id):
            # The stage INDEX cannot distinguish two ladders, and two ladders
            # in this programme have the same length (the queue's 100x20
            # ladder and run_scale_cl.sh's default are both 6 stages), so an
            # index-range check passes and the run silently continues at the
            # wrong size while stamping a false reached-size into every
            # checkpoint it writes.
            raise ValueError(
                'resume snapshot was written for a different ladder.\n'
                f'  snapshot: {ladder_id}\n'
                f'  this run: {self.ladder_id}\n'
                'Resume with the ladder the run started with (re-export '
                'LADDER / AUTO_LADDER), or start a new run.')

        stage_idx = int(state['stage_idx'])
        max_stage_idx = max(stage_idx, int(state.get('max_stage_idx', stage_idx)))
        # BOTH indices are bounded. An unbounded high-water mark indexes
        # self._scale_stages inside save_model(), which runs at every best
        # model and again after a DIVERGED/STUCK stop: an IndexError there
        # destroys a multi-day run at the exact moment it is trying to record
        # its result.
        for name, idx in (('stage', stage_idx), ('high-water stage', max_stage_idx)):
            if not 0 <= idx < self.n_stages:
                raise ValueError(
                    f'resume snapshot {name} is {idx + 1} but this run defines '
                    f'only {self.n_stages} stages. The ladder changed between '
                    'launches; resume with the ladder the run started with, or '
                    'start a new run.')
        self.stage_idx = stage_idx
        self.max_stage_idx = max_stage_idx
        self.stage_updates = int(state['stage_updates'])
        self._final_stage_updates = int(state.get('final_stage_updates', 0))
        self._hist = list(state['hist'])
        self._ev_hist = list(state.get('ev_hist', []))
        # Legacy snapshots omit gaps from ev_hist, so their historical
        # continuity cannot be reconstructed. Retain the previous finite-
        # history assumption; new snapshots explicitly preserve any gap.
        self._ev_valid_run = int(state.get('ev_valid_run', len(self._ev_hist)))
        if not 0 <= self._ev_valid_run <= len(self._ev_hist):
            raise ValueError('Invalid consecutive EV measurement count')
        self._demotions = {int(k): int(v) for k, v in
                           (state.get('demotions') or {}).items()}
        self._div_ref = state['div_ref']
        self._div_run = int(state['div_run'])
