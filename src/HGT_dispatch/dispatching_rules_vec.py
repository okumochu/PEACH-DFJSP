"""
Vectorised scorers for the fixed dispatching-rule baselines.

Lift each scorer in ``src/benchmark/dispatching_rules.py`` from ``[J, M]``
to ``[E, J, M]`` for batched score and action comparisons. Each operation
runs over the leading environment axis.

Public API:
  - ``RULE_NAMES``: tuple of canonical rule names (matches ``VALID_RULES``
    in ``src/benchmark/dispatching_rules.py``).
  - ``parse_rule_name(name)``: split ``'ATC_2.0'`` into ``('ATC', 2.0)`` so
    the menu can carry per-slot k-variants directly. Non-tunable rules
    return ``(name, None)``.
  - ``score_per_rule(env, rule_name, atc_k, covert_k) -> [E, J, M]``: score
    every (job, machine) pair in every env under one canonical rule.
  - ``select_actions_batched(env, rule_ids, parsed_menu) -> [E]``: given a
    per-env rule choice (index into ``parsed_menu``), return per-env flat
    action ``j*M + m`` matching the env's ``step()`` interface.

Selection semantics replicate ``_select_action`` from the single-env runner:
restrict to (job, machine) pairs at the earliest eligible start time
(non-delay), score them, take argmax. Tie-break: machine index, then job
(``np.argmax`` C-order).
"""

from __future__ import annotations

from typing import Dict, Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.benchmark.dispatching_rules import (
    VALID_RULES,
    TUNABLE_RULES,
    DEFAULT_ATC_K,
    DEFAULT_COVERT_K,
)


RULE_NAMES = VALID_RULES   # 11 canonical rules
RuleScorer = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# Menu name parser
# ---------------------------------------------------------------------------

ParsedSlot = Tuple[str, Optional[float]]


def parse_rule_name(name: str) -> ParsedSlot:
    """Split a parameterised menu name into (canonical_rule, k_or_None).

    ``'ATC_2.0'`` -> ``('ATC', 2.0)``; ``'COVERT_1.5'`` -> ``('COVERT', 1.5)``;
    ``'FIFO'`` -> ``('FIFO', None)``. Non-tunable rules carrying a ``_<k>``
    suffix are rejected because k would be silently ignored.
    """
    if name in RULE_NAMES:
        return name, None
    if '_' in name:
        head, _, tail = name.partition('_')
        if head in RULE_NAMES:
            if head not in TUNABLE_RULES:
                raise ValueError(
                    f"Rule {head!r} is not tunable but received a k-suffix "
                    f"in menu entry {name!r} (tunable: {TUNABLE_RULES})."
                )
            try:
                return head, float(tail)
            except ValueError as e:
                raise ValueError(
                    f"Could not parse k-value {tail!r} in menu entry {name!r}."
                ) from e
    raise ValueError(
        f"Unknown menu entry {name!r}. Valid forms: {RULE_NAMES} or "
        f"<TUNABLE>_<float> (tunable: {TUNABLE_RULES})."
    )


def parse_menu(menu: Sequence[str]) -> List[ParsedSlot]:
    """Apply :func:`parse_rule_name` to every entry; preserves order."""
    return [parse_rule_name(m) for m in menu]


# ---------------------------------------------------------------------------
# Helpers — operate on [E, ...] arrays.
#
# A "score cache" bundles values that several rules consume (candidate PTs,
# slack, p_avg, etc.). ``_build_score_cache`` materialises it once per
# ``select_actions_batched`` invocation; individual scorers read from it
# instead of recomputing — saves ~9 redundant gather/slack passes per step.
# ---------------------------------------------------------------------------

def _gather_candidate_pts(env) -> np.ndarray:
    """Return op_pt at each env's candidate ops, shape ``[E, J, M]`` float64."""
    E = env.number_of_envs
    n_j = env.number_of_jobs
    candidates = env.candidate[:E, :n_j].astype(np.int64)
    e_idx = np.arange(E)[:, None]
    return env.op_pt[e_idx, candidates, :].astype(np.float64)


def _build_score_cache(env) -> dict:
    """Pre-compute shared per-step state used by multiple rule scorers."""
    E = env.number_of_envs
    n_j = env.number_of_jobs
    max_pt = np.asarray(env.max_pt[:E], dtype=np.float64)

    pts = _gather_candidate_pts(env)                                     # [E, J, M]
    current_time = np.min(
        env.unnorm_mch_free_time[:E].astype(np.float64), axis=1
    )                                                                    # [E]
    due = env.job_due_date[:E, :n_j].astype(np.float64)                   # [E, J]
    remaining_work = (
        env.job_remain_sum_pt[:E, :n_j].astype(np.float64) * max_pt[:, None]
    )                                                                    # [E, J]
    slack = due - current_time[:, None] - remaining_work                 # [E, J]

    om = env.op_mean_pt[:E, :env.number_of_ops].astype(np.float64)
    valid = om > 0
    sums = np.where(valid, om, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    p_avg = np.where(counts > 0, sums / np.maximum(counts, 1), 1.0) * max_pt  # [E]

    return {
        'E': E, 'n_j': n_j, 'n_m': env.number_of_machines,
        'max_pt': max_pt,
        'pts': pts,                       # [E, J, M] float64 — candidate op PTs
        'current_time': current_time,     # [E]
        'due': due,                        # [E, J]
        'remaining_work': remaining_work,  # [E, J]
        'slack': slack,                    # [E, J]
        'p_avg': p_avg,                    # [E]
    }


def _broadcast_per_job_vec(per_job: np.ndarray, n_m: int) -> np.ndarray:
    """[E, J] -> [E, J, M] by broadcasting on the machine axis."""
    return np.broadcast_to(per_job[:, :, None], per_job.shape + (n_m,))


# ---------------------------------------------------------------------------
# Per-rule scorers — return [E, J, M].
# Signature: (env, cache, atc_k, covert_k). ``cache`` is the dict from
# ``_build_score_cache``; ``atc_k``/``covert_k`` are ignored by non-tunable
# rules but kept in the signature so the dispatch table is uniform.
# ---------------------------------------------------------------------------

def _score_fifo(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    E = cache['E']
    n_j = cache['n_j']
    per_job = -env.unnorm_candidate_free_time[:E, :n_j].astype(np.float64)
    return _broadcast_per_job_vec(per_job, cache['n_m'])


def _score_spt(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    return -cache['pts']


def _score_edd(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    return _broadcast_per_job_vec(-cache['due'], cache['n_m'])


def _score_wspt(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    pts = cache['pts']
    w = env.job_weight[:cache['E'], :cache['n_j']].astype(np.float64)[:, :, None]
    return np.where(pts > 0, w / np.maximum(pts, 1e-12), -np.inf)


def _score_atc(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    pts = cache['pts']
    denom = np.maximum(atc_k * cache['p_avg'], 1e-6)[:, None]            # [E, 1]
    w = env.job_weight[:cache['E'], :cache['n_j']].astype(np.float64)[:, :, None]
    urgency = np.exp(-np.maximum(0.0, cache['slack']) / denom)[:, :, None]
    return np.where(pts > 0,
                    (w / np.maximum(pts, 1e-12)) * urgency,
                    -np.inf)


def _score_covert(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    """Cost-Over-Time: like ATC but the slack discount is linear (clamped at 0)
    instead of exponential. Reduces to SPT-WSPT-with-weight at slack=0."""
    pts = cache['pts']
    denom = np.maximum(covert_k * cache['p_avg'], 1e-6)[:, None]
    w = env.job_weight[:cache['E'], :cache['n_j']].astype(np.float64)[:, :, None]
    bracket = np.maximum(0.0, 1.0 - np.maximum(0.0, cache['slack']) / denom)[:, :, None]
    return np.where(pts > 0,
                    (w / np.maximum(pts, 1e-12)) * bracket,
                    -np.inf)


def _score_cr(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    """Critical Ratio: argmin (TTD / (op_pt + remaining_work)) — smaller CR
    means the job is more urgent relative to its remaining workload."""
    pts = cache['pts']
    ttd = (cache['due'] - cache['current_time'][:, None])[:, :, None]    # [E, J, 1]
    denom = pts + cache['remaining_work'][:, :, None]                    # [E, J, M]
    cr = ttd / np.maximum(denom, 1e-6)
    return np.where(pts > 0, -cr, -np.inf)


def _score_mod(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    """Modified Operation Date: argmin max(due, current_time + op_pt). Smaller
    MOD means smaller of (due-date pressure, finish-time-if-started-now)."""
    pts = cache['pts']
    current_time = cache['current_time'][:, None, None]                   # [E, 1, 1]
    due = cache['due'][:, :, None]                                        # [E, J, 1]
    finish_if_now = current_time + pts
    mod_factor = np.maximum(due, finish_if_now)
    return np.where(pts > 0, -mod_factor, -np.inf)


def _score_ms(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    """Minimum-Slack-first: pure due-date pressure. Negative slack -> highest score."""
    return _broadcast_per_job_vec(-cache['slack'], cache['n_m'])


def _score_avpro(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    """Average process time per remaining op: argmin (op_pt + remaining_work)
    / (remaining_ops + 1). Smaller = lower per-op workload, finish faster."""
    pts = cache['pts']
    remaining_ops = env.job_remain_op_nums[:cache['E'], :cache['n_j']].astype(np.float64)
    denom = (remaining_ops + 1.0)[:, :, None]
    avpro = (pts + cache['remaining_work'][:, :, None]) / denom
    return np.where(pts > 0, -avpro, -np.inf)


def _score_npt(env, cache, atc_k=DEFAULT_ATC_K, covert_k=DEFAULT_COVERT_K) -> np.ndarray:
    """Next-PT: prefer jobs whose *next* operation (after the candidate one)
    has small mean PT — frees the next downstream machine quickly. Jobs at
    their last op contribute NPT=0 (highest priority)."""
    E = cache['E']
    n_j = cache['n_j']
    candidates = env.candidate[:E, :n_j].astype(np.int64)
    job_last_op_id = env.job_last_op_id[:E, :n_j].astype(np.int64)
    next_op_id = candidates + 1
    has_next = next_op_id <= job_last_op_id
    safe_next = np.where(has_next, next_op_id, candidates)
    e_idx = np.arange(E)[:, None]
    next_pts = env.op_pt[e_idx, safe_next, :].astype(np.float64)
    compat = next_pts > 0
    sums = np.where(compat, next_pts, 0.0).sum(axis=2)
    counts = compat.sum(axis=2)
    mean_next_pt = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    npt_per_job = np.where(has_next, mean_next_pt, 0.0)
    return _broadcast_per_job_vec(-npt_per_job, cache['n_m'])


_RULES_VEC: Dict[str, RuleScorer] = {
    'FIFO':   _score_fifo,
    'SPT':    _score_spt,
    'EDD':    _score_edd,
    'WSPT':   _score_wspt,
    'ATC':    _score_atc,
    'MS':     _score_ms,
    'COVERT': _score_covert,
    'CR':     _score_cr,
    'MOD':    _score_mod,
    'AVPRO':  _score_avpro,
    'NPT':    _score_npt,
}


def score_per_rule(env, rule_name: str,
                   atc_k: float = DEFAULT_ATC_K,
                   covert_k: float = DEFAULT_COVERT_K,
                   cache=None) -> np.ndarray:
    """Score every (job, machine) pair under ``rule_name`` for every env.

    Returns ``[E, J, M]`` float64 array. Invalid pairs (e.g. ``op_pt == 0``)
    are -inf for per-pair rules; job-level rules return per-job scores
    broadcasted across machines (incompatibility filtered downstream by
    the env's ``candidate_mask``).

    ``cache`` is the dict produced by :func:`_build_score_cache`; pass an
    existing cache to amortise cost across multiple rule calls (used by
    :func:`select_actions_batched`).
    """
    if rule_name not in _RULES_VEC:
        raise ValueError(f"Unknown rule {rule_name!r}. Valid: {RULE_NAMES}")
    if cache is None:
        cache = _build_score_cache(env)
    return _RULES_VEC[rule_name](env, cache, atc_k=atc_k, covert_k=covert_k)


# ---------------------------------------------------------------------------
# Action selection — per-env rule -> flat action.
# ---------------------------------------------------------------------------

def _earliest_start_mask(env) -> tuple:
    """Compute (compat [E, J, M], at_earliest [E, J, M]) — see _select_action.

    A pair is 'at the earliest eligible start' iff it ties the per-env
    minimum start time among compatible pairs, where start time =
    max(candidate_free_time[e, j], mch_free_time[e, m]).
    """
    E = env.number_of_envs
    n_j = env.number_of_jobs
    n_m = env.number_of_machines

    mask = env.candidate_mask[:E, :n_j].astype(bool)                     # True=invalid
    job_active = ~env.job_finished_mask[:E, :n_j].astype(bool)           # [E, J]
    compat = (~mask) & job_active[:, :, None]                            # [E, J, M]

    mch_free = env.unnorm_mch_free_time[:E, :n_m].astype(np.float64)      # [E, M]
    cand_free = env.unnorm_candidate_free_time[:E, :n_j].astype(np.float64)  # [E, J]
    start_time = np.maximum(cand_free[:, :, None], mch_free[:, None, :])  # [E, J, M]
    start_time_compat = np.where(compat, start_time, np.inf)

    earliest = start_time_compat.reshape(E, -1).min(axis=1)               # [E]
    at_earliest = (start_time_compat <= earliest[:, None, None] + 1e-6) & compat
    return compat, at_earliest


def select_actions_batched(env,
                            rule_ids: np.ndarray,
                            parsed_menu: Sequence[ParsedSlot]) -> np.ndarray:
    """Translate per-env rule choices into per-env flat actions.

    Args:
        env: ``FJSPEnvHGT``-compatible env (uses ``candidate``,
            ``candidate_mask``, ``unnorm_*_free_time``, etc.).
        rule_ids: ``[E]`` int array. Each entry is an index into
            ``parsed_menu`` selected by the policy for that env.
        parsed_menu: ordered list of ``(canonical_rule, k_or_None)`` slots —
            one per action. Tunable slots carry their own k; non-tunables
            carry ``None`` (the scorer ignores k anyway). Built once in the
            env via :func:`parse_menu`.

    Returns:
        ``[E]`` int64 array of flat actions ``j * n_m + m`` consumable by
        ``env.step(action)``.

    Done envs (no compatible candidate) get action 0 — same as the
    single-env ``_select_action`` fallback. The env will treat this as a
    no-op once it's flagged done.
    """
    E = env.number_of_envs
    n_j = env.number_of_jobs
    n_m = env.number_of_machines

    # Build the shared score cache once so each rule's gather/slack/p_avg
    # work is paid for once across the menu, not S× (S = menu size).
    cache = _build_score_cache(env)

    # Score every menu slot once over all envs ([S, E, J, M]). The slot's
    # k drives ATC/COVERT; non-tunable scorers ignore both kwargs.
    all_scores = np.stack([
        score_per_rule(
            env, canonical, cache=cache,
            atc_k=k if (canonical == 'ATC' and k is not None) else DEFAULT_ATC_K,
            covert_k=k if (canonical == 'COVERT' and k is not None) else DEFAULT_COVERT_K,
        )
        for canonical, k in parsed_menu
    ], axis=0)                                                           # [S, E, J, M]
    # Gather per-env slot scores:
    e_idx = np.arange(E)
    scores = all_scores[rule_ids, e_idx]                                  # [E, J, M]

    _, at_earliest = _earliest_start_mask(env)
    scores = np.where(at_earliest, scores, -np.inf)

    flat_scores = scores.reshape(E, n_j * n_m)
    actions = np.argmax(flat_scores, axis=1).astype(np.int64)             # [E]

    # Done envs: argmax over all-(-inf) is undefined. Fall back to 0 like
    # the single-env path. The env's step() ignores actions on done envs.
    all_inf = np.isneginf(flat_scores).all(axis=1)
    if all_inf.any():
        actions[all_inf] = 0
    return actions
