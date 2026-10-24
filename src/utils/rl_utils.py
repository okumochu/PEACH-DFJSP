from torch.distributions.categorical import Categorical
import numpy as np
import torch
from typing import List, Dict, Any


"""
    agent utils
"""


def sample_action(p):
    """
        sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    # Preserve the last action axis even when an environment has one action.
    # Squeezing [batch, 1] would turn environments into categorical outcomes.
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    _, index = torch.max(p, dim=1)
    return index


def _resolve_objective_weights(makespan_weight, twt_weight):
    """Validate that objective weights were explicitly provided.

    Inference must never silently default — a model trained with mw=0.3/tw=0.7
    must not be evaluated at 0.5/0.5. Callers are expected to read both weights
    from the model checkpoint via ``src.utils.model_utils.load_model``.
    """
    if makespan_weight is None or twt_weight is None:
        raise ValueError(
            "makespan_weight and twt_weight must both be provided. "
            "Load them from the model checkpoint via load_model(): "
            "policy, _, weights = load_model(path, device); "
            "then pass weights['makespan_weight'] and weights['twt_weight'].")
    return float(makespan_weight), float(twt_weight)


def _empty_events_dict() -> Dict[str, list]:
    """The empty 'dynamic_events' shape consumed by the env."""
    return {'new_job_arrival': [], 'machine_breakdown': [], 'priority_change': []}


def _create_static_data(dynamic_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create static data structure from dynamic_data (no dynamic events).

    Handles both dynamic-mode data (with max_total_* keys) and static-mode
    data (without them).  When the keys are absent the sizes are derived
    from the static arrays so the env falls back to exact-fit allocation.

    Args:
        dynamic_data: Data from DataGenerator (static or dynamic)

    Returns:
        Static data dict for environment initialization
    """
    return {
        'static': dynamic_data['static'],
        'dynamic_events': _empty_events_dict(),
        'dynamic_jobs': dynamic_data.get('dynamic_jobs'),
        'max_total_jobs': dynamic_data.get('max_total_jobs'),
        'max_total_ops': dynamic_data.get('max_total_ops'),
        'max_total_pt_nodes': dynamic_data.get('max_total_pt_nodes'),
    }


def greedy_result_as_dict(
    greedy_env, makespan: float, twt: float, steps: int,
    makespan_weight: float, twt_weight: float,
) -> Dict[str, Any]:
    """Pack a `_run_single_greedy` 4-tuple into the standard inference-result dict
    shape used by ``InferenceEnv.run_with_solver``."""
    return {
        'makespan': makespan, 'twt': twt,
        'objective': makespan_weight * makespan + twt_weight * twt,
        'step_count': steps,
        'best_env': greedy_env, 'best_env_idx': 0,
    }


# ============================================================================
# Inference Strategy helpers
# ============================================================================

def _run_single_greedy(policy, n_j, n_m, dynamic_data, device,
                       pre_initialized_env=None):
    """Run a single-env greedy rollout. Returns (env, makespan, twt, steps).

    Always runs a static inference. Dynamic events are handled externally by
    InferenceEnv.

    If *pre_initialized_env* is provided, it is used as-is (already seeded
    with set_initial_data_static / set_initial_data_with_fixed_schedule).
    Otherwise a fresh env is created from *dynamic_data*.
    """
    from src.HGT.hgt_env import FJSPEnvHGT
    if pre_initialized_env is not None:
        greedy_env = pre_initialized_env
        state = greedy_env.state
    else:
        init_data = _create_static_data(dynamic_data)
        greedy_env = FJSPEnvHGT(
            n_j, n_m, device=device,
            use_pt_nodes=getattr(policy, 'use_pt_nodes', True))
        state = greedy_env.set_initial_data_static(init_data)
    done = greedy_env.done()
    step_count = 0
    while not done.all():
        with torch.no_grad():
            pi, _ = policy(
                candidate=state.candidate_tensor,
                candidate_mask=state.candidate_mask_tensor,
                candidate_pt_indices=state.candidate_pt_indices,
                graph=state.graph_tensor
            )
        action = greedy_select_action(pi)
        state, reward, done = greedy_env.step(action.cpu().numpy())
        step_count += 1
    greedy_env._recompute_objectives()
    return greedy_env, float(greedy_env.current_makespan[0]), float(greedy_env.current_twt[0]), step_count
