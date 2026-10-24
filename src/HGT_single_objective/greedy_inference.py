"""Greedy inference helpers for single-objective HGT evaluation."""

import time
from typing import Any, Dict

import numpy as np
import torch

from .hgt_env_so import FJSPEnvHGTSingleObj


def _build_static_payload(job_length: np.ndarray, op_pt: np.ndarray) -> Dict[str, Any]:
    """Wrap one parsed FJSP instance in the static env input shape."""
    n_j = int(job_length.shape[0])
    n_ops = int(np.sum(job_length))
    n_m = int(op_pt.shape[1])
    return {
        'static': {
            'job_length': [job_length.copy()],
            'op_pt': [op_pt.astype(np.float32).copy()],
        },
        'max_total_jobs': n_j,
        'max_total_ops': n_ops,
        'max_total_pt_nodes': n_ops * n_m,
    }


def _rollout(env, policy):
    """Run one deterministic argmax rollout to completion."""
    state = env.state
    was_training = policy.training
    policy.eval()
    try:
        while True:
            with torch.no_grad():
                pi, _ = policy(
                    candidate=state.candidate_tensor,
                    candidate_mask=state.candidate_mask_tensor,
                    candidate_pt_indices=state.candidate_pt_indices,
                    graph=state.graph_tensor,
                )
            actions = torch.argmax(pi, dim=1)
            state, _r, done = env.step(actions.cpu().numpy())
            if done.all():
                break
    finally:
        policy.train(was_training)


def greedy_inference(
    policy,
    n_j: int,
    n_m: int,
    job_length: np.ndarray,
    op_pt: np.ndarray,
    device: str,
) -> Dict[str, Any]:
    """Run a single greedy (argmax) rollout on one instance."""
    t0 = time.time()
    env = FJSPEnvHGTSingleObj(
        n_j,
        n_m,
        device=device,
    )
    env.set_initial_data_static(_build_static_payload(job_length, op_pt))
    _rollout(env, policy)
    return {
        'makespan': float(env.current_makespan[0]),
        'time_s': time.time() - t0,
        'best_env': env,
        'best_env_idx': 0,
    }
