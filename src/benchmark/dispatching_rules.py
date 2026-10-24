"""
Dispatching rule baselines for multi-criteria (makespan + TWT) FJSP.

Each rule picks one (job, machine) pair per decision step, matching the
``InferenceEnv.run_with_solver`` static-solver contract used by learned HGT
greedy inference.

Selection is event-driven / non-delay:
  1. Build the set of eligible (job, machine) pairs — job unfinished,
     candidate op present, machine compatible.
  2. Compute each pair's earliest start time as
     max(mch_free_time[m], candidate_free_time[j]).
  3. Restrict to pairs tied at the minimum start time. This reproduces a
     textbook dispatcher: "when a machine becomes idle, pick the best
     waiting op for IT." Multiple simultaneously-idle machines fall out
     as pairs tied at the same minimum start time, and the rule breaks
     ties among them directly — no separate EAM step is needed.
  4. Score the surviving pairs; return argmax as action = j*n_m + m.

Eleven rules organised by what they react to (higher score = preferred,
non-tunable defaults match the corresponding paper formulas):

  Queue-order:
  - FIFO  : -candidate_free_time[j]                       (job-level)

  Process-time:
  - SPT   : -op_pt[cand[j], m]                            (per-pair)
  - NPT   : -mean(op_pt[cand[j]+1, :])                    (job-level, ignores last-op)

  Due-date:
  - EDD   : -due_date[j]                                  (job-level)
  - MOD   : -max(due_date[j], current_time + op_pt[j,m])  (per-pair)

  Slack-driven:
  - MS    : -slack[j]                                     (job-level)
  - CR    : -(due[j] - current_time) / (op_pt[j,m] + remaining_work[j])  (per-pair)

  Weighted/composite:
  - WSPT   : w[j] / op_pt[cand[j], m]                                              (per-pair; Smith)
  - ATC    : (w[j] / p_ij) * exp(-max(0,slack[j]) / (k_atc * p_avg))               (per-pair)
  - COVERT : max(0, 1 - max(0,slack[j]) / (k_covert * p_avg)) / op_pt[j,m]         (per-pair)
  - AVPRO  : -(op_pt[j,m] + remaining_work[j]) / (remaining_ops[j] + 1)            (per-pair)

These eleven cover the canonical paper rules from Luo (2020) Table 5 that
don't require a per-machine queue-PT accumulator. WINQ-family rules
(WINQ, PTWINQS, SPT-WINQ, DPT-WINQ-NPT) are deferred — they would need a
new env state for the sum of remaining PT queued on each machine.

The runner creates a fresh env internally, loops step() until terminal,
and calls _recompute_objectives() at the end — exactly mirroring
src/utils/rl_utils.py:_run_single_greedy.
"""

from typing import Dict, Any, Callable, Optional

import numpy as np

from src.HGT.hgt_env import FJSPEnvHGT
from src.utils.rl_utils import _resolve_objective_weights, _create_static_data


VALID_RULES = ('FIFO', 'SPT', 'EDD', 'WSPT', 'ATC', 'MS',
               'COVERT', 'CR', 'MOD', 'AVPRO', 'NPT')
TUNABLE_RULES = ('ATC', 'COVERT')
DEFAULT_ATC_K = 2.0
DEFAULT_COVERT_K = 2.0


# ---------------------------------------------------------------------------
# Per-pair scorers — higher score = prefer this (job, machine) pair. Each
# returns an [n_j, n_m] array of scores. Job-level rules broadcast over m;
# per-pair rules depend on PT on the specific machine. The caller masks
# invalid pairs (incompatible or not at the earliest eligible start time)
# with -inf before argmax.
#
# Uniform signature: every scorer accepts (env, atc_k, covert_k) so the
# dispatch table _RULES is callable without knowing the rule's tunables.
# Non-tunable scorers ignore both kwargs.
# ---------------------------------------------------------------------------

def _broadcast_per_job(per_job: np.ndarray, n_m: int) -> np.ndarray:
    """Broadcast a [n_j] score vector to a [n_j, n_m] pair-score array."""
    return np.broadcast_to(per_job[:, None], (per_job.shape[0], n_m))


def _score_fifo(env: FJSPEnvHGT,
                atc_k: float = DEFAULT_ATC_K,
                covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    n_j = env.number_of_jobs
    per_job = -env.unnorm_candidate_free_time[0, :n_j].astype(np.float64)
    return _broadcast_per_job(per_job, env.number_of_machines)


def _score_spt(env: FJSPEnvHGT,
               atc_k: float = DEFAULT_ATC_K,
               covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    n_j = env.number_of_jobs
    candidates = env.candidate[0, :n_j]
    return -env.op_pt[0, candidates, :].astype(np.float64)


def _score_edd(env: FJSPEnvHGT,
               atc_k: float = DEFAULT_ATC_K,
               covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    n_j = env.number_of_jobs
    per_job = -env.job_due_date[0, :n_j].astype(np.float64)
    return _broadcast_per_job(per_job, env.number_of_machines)


def _score_wspt(env: FJSPEnvHGT,
                atc_k: float = DEFAULT_ATC_K,
                covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    # float64 promotion avoids tie-break drift from float32 mantissa truncation.
    n_j = env.number_of_jobs
    candidates = env.candidate[0, :n_j]
    pts = env.op_pt[0, candidates, :].astype(np.float64)                # [n_j, n_m]
    w = env.job_weight[0, :n_j].astype(np.float64)[:, None]             # [n_j, 1]
    return np.where(pts > 0, w / np.maximum(pts, 1e-12), -np.inf)


def _slack_block(env: FJSPEnvHGT) -> tuple:
    """Shared slack computation: returns (n_j, max_pt, slack [n_j], current_time)."""
    n_j = env.number_of_jobs
    max_pt = float(env.max_pt[0])
    current_time = float(np.min(env.unnorm_mch_free_time[0]))
    due = env.job_due_date[0, :n_j].astype(np.float64)
    remaining_work = env.job_remain_sum_pt[0, :n_j].astype(np.float64) * max_pt
    return n_j, max_pt, due - current_time - remaining_work, current_time


def _avg_op_pt_unnormalised(env: FJSPEnvHGT, max_pt: float) -> float:
    """Mean PT across all (op, compatible-machine) pairs in real time-units."""
    op_means = env.op_mean_pt[0, :env.number_of_ops]
    valid_om = op_means[op_means > 0]
    return float(valid_om.mean()) * max_pt if valid_om.size > 0 else 1.0


def _score_atc(env: FJSPEnvHGT,
               atc_k: float = DEFAULT_ATC_K,
               covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    n_j, max_pt, slack, _ = _slack_block(env)
    p_avg = _avg_op_pt_unnormalised(env, max_pt)
    denom = max(atc_k * p_avg, 1e-6)

    candidates = env.candidate[0, :n_j]
    pts = env.op_pt[0, candidates, :].astype(np.float64)                # [n_j, n_m]
    w = env.job_weight[0, :n_j].astype(np.float64)[:, None]             # [n_j, 1]
    urgency = np.exp(-np.maximum(0.0, slack) / denom)[:, None]          # [n_j, 1]
    return np.where(pts > 0,
                    (w / np.maximum(pts, 1e-12)) * urgency,
                    -np.inf)


def _score_covert(env: FJSPEnvHGT,
                  atc_k: float = DEFAULT_ATC_K,
                  covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    """Cost-Over-Time: like ATC but the slack discount is linear (clamped at 0)
    instead of exponential. Reduces to SPT-WSPT-with-weight at slack=0."""
    n_j, max_pt, slack, _ = _slack_block(env)
    p_avg = _avg_op_pt_unnormalised(env, max_pt)
    denom = max(covert_k * p_avg, 1e-6)

    candidates = env.candidate[0, :n_j]
    pts = env.op_pt[0, candidates, :].astype(np.float64)                # [n_j, n_m]
    w = env.job_weight[0, :n_j].astype(np.float64)[:, None]             # [n_j, 1]
    bracket = np.maximum(0.0, 1.0 - np.maximum(0.0, slack) / denom)[:, None]  # [n_j, 1]
    return np.where(pts > 0,
                    (w / np.maximum(pts, 1e-12)) * bracket,
                    -np.inf)


def _score_cr(env: FJSPEnvHGT,
              atc_k: float = DEFAULT_ATC_K,
              covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    """Critical Ratio: argmin (TTD / (op_pt + remaining_work)) — smaller CR
    means the job is more urgent relative to its remaining workload."""
    n_j, max_pt, _, current_time = _slack_block(env)
    due = env.job_due_date[0, :n_j].astype(np.float64)
    remaining_work = env.job_remain_sum_pt[0, :n_j].astype(np.float64) * max_pt

    candidates = env.candidate[0, :n_j]
    pts = env.op_pt[0, candidates, :].astype(np.float64)                # [n_j, n_m]
    ttd = (due - current_time)[:, None]                                  # [n_j, 1]
    denom = pts + remaining_work[:, None]                                # [n_j, n_m]
    cr = ttd / np.maximum(denom, 1e-6)
    return np.where(pts > 0, -cr, -np.inf)


def _score_mod(env: FJSPEnvHGT,
               atc_k: float = DEFAULT_ATC_K,
               covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    """Modified Operation Date: argmin max(due, current_time + op_pt). Smaller
    MOD means smaller of (due-date pressure, finish-time-if-started-now)."""
    n_j = env.number_of_jobs
    candidates = env.candidate[0, :n_j]
    pts = env.op_pt[0, candidates, :].astype(np.float64)                # [n_j, n_m]
    current_time = float(np.min(env.unnorm_mch_free_time[0]))
    due = env.job_due_date[0, :n_j].astype(np.float64)[:, None]          # [n_j, 1]
    finish_if_now = current_time + pts                                   # [n_j, n_m]
    mod_factor = np.maximum(due, finish_if_now)
    return np.where(pts > 0, -mod_factor, -np.inf)


def _score_ms(env: FJSPEnvHGT,
              atc_k: float = DEFAULT_ATC_K,
              covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    """Minimum-Slack-first: pure due-date pressure. Negative slack -> highest score."""
    _, _, slack, _ = _slack_block(env)
    return _broadcast_per_job(-slack, env.number_of_machines)


def _score_avpro(env: FJSPEnvHGT,
                 atc_k: float = DEFAULT_ATC_K,
                 covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    """Average process time per remaining op: argmin (op_pt + remaining_work)
    / (remaining_ops + 1). Smaller = lower per-op workload, finish faster."""
    n_j = env.number_of_jobs
    max_pt = float(env.max_pt[0])
    candidates = env.candidate[0, :n_j]
    pts = env.op_pt[0, candidates, :].astype(np.float64)                # [n_j, n_m]
    remaining_work = env.job_remain_sum_pt[0, :n_j].astype(np.float64) * max_pt
    remaining_ops = env.job_remain_op_nums[0, :n_j].astype(np.float64)
    denom = remaining_ops + 1.0                                           # [n_j]
    avpro = (pts + remaining_work[:, None]) / denom[:, None]              # [n_j, n_m]
    return np.where(pts > 0, -avpro, -np.inf)


def _score_npt(env: FJSPEnvHGT,
               atc_k: float = DEFAULT_ATC_K,
               covert_k: float = DEFAULT_COVERT_K) -> np.ndarray:
    """Next-PT: prefer jobs whose *next* operation (after the candidate one)
    has small mean PT — frees the next downstream machine quickly. Jobs at
    their last op contribute NPT=0 (highest priority)."""
    n_j = env.number_of_jobs
    candidates = env.candidate[0, :n_j].astype(np.int64)
    job_last_op_id = env.job_last_op_id[0, :n_j].astype(np.int64)
    next_op_id = candidates + 1
    has_next = next_op_id <= job_last_op_id                               # [n_j]
    safe_next = np.where(has_next, next_op_id, candidates)                # gather-safe
    next_pts = env.op_pt[0, safe_next, :].astype(np.float64)              # [n_j, n_m]
    compat = next_pts > 0
    sums = np.where(compat, next_pts, 0.0).sum(axis=1)
    counts = compat.sum(axis=1)
    mean_next_pt = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    npt_per_job = np.where(has_next, mean_next_pt, 0.0)
    return _broadcast_per_job(-npt_per_job, env.number_of_machines)


_RULES: Dict[str, Callable[..., np.ndarray]] = {
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


# ---------------------------------------------------------------------------
# Action selection: non-delay, per-pair argmax over (job, machine)
# ---------------------------------------------------------------------------

def _select_action(env: FJSPEnvHGT, rule_name: str,
                   atc_k: float = DEFAULT_ATC_K,
                   covert_k: float = DEFAULT_COVERT_K) -> int:
    """Pick the best (job, machine) pair using the rule, restricted to pairs
    that could start at the earliest eligible time.

    The start-time filter advances "simulated time" to the next decision
    event (a machine going idle with a ready compatible op) and is what
    makes the schedule compact — without it a rule can commit a job to a
    busy machine while another idle machine sits empty.

    Returns action = chosen_job * env.number_of_machines + chosen_mch.
    Ties break on machine index, then job index (np.argmax C-order).
    """
    n_j = env.number_of_jobs
    mask = env.candidate_mask[0, :n_j]                    # [n_j, n_m] True=invalid
    job_active = ~env.job_finished_mask[0, :n_j]          # [n_j]
    compat = (~mask) & job_active[:, None]                # [n_j, n_m]

    if not compat.any():
        return 0

    mch_free = env.unnorm_mch_free_time[0].astype(np.float64)              # [n_m]
    cand_free = env.unnorm_candidate_free_time[0, :n_j].astype(np.float64) # [n_j]
    start_time = np.maximum(cand_free[:, None], mch_free[None, :])         # [n_j, n_m]
    start_time_compat = np.where(compat, start_time, np.inf)

    earliest = start_time_compat.min()
    at_earliest = (start_time_compat <= earliest + 1e-6) & compat

    scores = _RULES[rule_name](env, atc_k=atc_k, covert_k=covert_k).astype(np.float64)
    scores = np.where(at_earliest, scores, -np.inf)

    return int(np.argmax(scores))   # flat index == chosen_job * n_m + chosen_mch


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_dispatching_rule(
    env: FJSPEnvHGT,
    dynamic_data: Dict[str, Any],
    rule_name: str,
    makespan_weight: Optional[float] = None,
    twt_weight: Optional[float] = None,
    device: Optional[str] = None,
    atc_k: float = DEFAULT_ATC_K,
    covert_k: float = DEFAULT_COVERT_K,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a dispatching-rule rollout through ``InferenceEnv``.

    Dynamic events (if any) are handled via InferenceEnv — it iterates the
    training env once per event boundary. An empty events list degenerates
    to a single static solve.

    Args:
        env: template env used for n_j/n_m/device introspection only.
        dynamic_data: output of DataGenerator.sample_env(num_envs=1).
        rule_name: one of VALID_RULES.
        makespan_weight, twt_weight: defaulted from experiment_config when None.
        atc_k: ATC look-ahead parameter (ignored by non-ATC rules).
        covert_k: COVERT look-ahead parameter (ignored by non-COVERT rules).

    Returns:
        {'makespan','twt','objective','step_count','best_env','best_env_idx'}
    """
    if rule_name not in _RULES:
        raise ValueError(f"Unknown rule {rule_name!r}. Valid: {VALID_RULES}")

    mw, tw = _resolve_objective_weights(makespan_weight, twt_weight)

    n_j = env.number_of_jobs
    n_m = env.number_of_machines
    dev = device if device is not None else env.device

    def _solve_on_seeded_env(seeded_env, _mw, _tw):
        done = seeded_env.done()
        steps = 0
        while not done.all():
            action = _select_action(seeded_env, rule_name,
                                    atc_k=atc_k, covert_k=covert_k)
            _, _, done = seeded_env.step(np.array([action]))
            steps += 1
        seeded_env._recompute_objectives()
        ms = float(seeded_env.current_makespan[0])
        twv = float(seeded_env.current_twt[0])
        obj = _mw * ms + _tw * twv
        return {
            'makespan': ms, 'twt': twv, 'objective': obj,
            'step_count': steps, 'best_env': seeded_env, 'best_env_idx': 0,
        }

    from src.HGT.inference_env import InferenceEnv
    inf_env = InferenceEnv(n_j, n_m, device=dev)
    result = inf_env.run_with_solver(
        _solve_on_seeded_env, dynamic_data,
        makespan_weight=mw, twt_weight=tw, verbose=verbose,
    )
    ms = result['makespan']
    twt = result['twt']
    obj = result['objective']
    step_count = result['step_count']
    rule_env = result['best_env']

    if verbose:
        print(f"  {rule_name}: MS={ms:.2f}, TWT={twt:.2f}, "
              f"Obj={obj:.2f}, steps={step_count}")

    return {
        'makespan': ms,
        'twt': twt,
        'objective': obj,
        'step_count': step_count,
        'best_env': rule_env,
        'best_env_idx': 0,
    }
