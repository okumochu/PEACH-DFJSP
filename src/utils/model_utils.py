import os
import argparse
import torch
from typing import Dict, Any, Optional, Tuple

from src.HGT.hgt_network import HGT
from src.HGT_single_objective.hgt_network_so import HGTSingleObj


def resolve_device(arg_device: Optional[str] = None) -> str:
    """Resolve a torch device string with cuda:1-preferred fallback.

    If ``arg_device`` is given, it takes precedence (only collapsed to 'cpu' when
    a 'cuda' string is requested but CUDA is unavailable). Otherwise prefers
    'cuda:1' on multi-GPU hosts, then 'cuda:0', then 'cpu' — matching the
    convention used by the inference scripts and smoke tests.
    """
    if arg_device is not None:
        if 'cuda' in arg_device and not torch.cuda.is_available():
            print("[WARNING] CUDA not available; falling back to CPU")
            return 'cpu'
        return arg_device
    if not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU (slow)")
        return 'cpu'
    if torch.cuda.device_count() > 1:
        return 'cuda:1'
    print("[warn] cuda:1 unavailable; falling back to cuda:0")
    return 'cuda:0'


def _load_policy_checkpoint(model_path: str, device: str, policy_class):
    """Shared checkpoint-load body for HGT and single-objective HGT.

    Returns (policy, hgt_hyperparams, objective_weights). The caller chooses
    ``policy_class`` (``HGT`` or ``HGTSingleObj``) and applies any additional
    sanity checks before/after.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    loaded = torch.load(model_path, map_location=device)
    if not isinstance(loaded, dict) or 'state_dict' not in loaded:
        raise ValueError(f"Invalid model format at {model_path}")

    hgt_hyperparams = loaded.get('hgt_hyperparams')
    if not isinstance(hgt_hyperparams, dict):
        raise ValueError(
            f"Checkpoint {model_path} has no current 'hgt_hyperparams' "
            "metadata. Re-train with the current training scripts.")
    if 'use_pt_nodes' not in hgt_hyperparams:
        raise ValueError(
            f"Checkpoint {model_path} is missing 'use_pt_nodes' metadata. "
            "Legacy checkpoint schemas are no longer supported.")

    objective_weights = hgt_hyperparams.get('objective_weights')
    if objective_weights is None:
        raise ValueError(
            f"Checkpoint {model_path} has no 'objective_weights'. "
            "Re-train with the current training scripts.")
    objective_weights = {
        'makespan_weight': float(objective_weights['makespan_weight']),
        'twt_weight': float(objective_weights['twt_weight']),
    }

    model_config = argparse.Namespace()
    model_config.device = device
    for key, value in hgt_hyperparams.items():
        setattr(model_config, key, value)

    policy = policy_class(model_config)
    policy.load_state_dict(loaded['state_dict'])
    policy.eval()
    return policy, hgt_hyperparams, objective_weights


def load_model(model_path: str, device: str = 'cuda:0'
               ) -> Tuple[HGT, Dict[str, Any], Dict[str, float]]:
    """Load a trained HGT (J*M-action) checkpoint."""
    return _load_policy_checkpoint(model_path, device, HGT)


def _validate_single_obj_checkpoint(
    model_path: str,
    hgt_hyperparams: Dict[str, Any],
    objective_weights: Dict[str, float],
) -> None:
    """Validate that a checkpoint matches the SD1/SD2 PT node SO policy."""
    if not hgt_hyperparams.get('single_objective', False):
        raise ValueError(
            f"Checkpoint {model_path} is not marked single_objective=True. "
            "Use a checkpoint from train_single_obj.py for makespan-only "
            "SD1/SD2 experiments.")
    if abs(objective_weights['makespan_weight'] - 1.0) > 1e-6 or \
            abs(objective_weights['twt_weight']) > 1e-6:
        raise ValueError(
            f"Checkpoint {model_path} has objective_weights="
            f"{objective_weights}; expected makespan_weight=1.0 and "
            "twt_weight=0.0 for single-objective evaluation.")
    if hgt_hyperparams.get('use_pt_nodes') is not True:
        raise ValueError(
            f"Checkpoint {model_path} is missing use_pt_nodes=True. "
            "Legacy single-objective checkpoints and those without PT nodes are not supported.")
    dp_lb_diff = hgt_hyperparams.get('dp_lb_diff')
    if not isinstance(dp_lb_diff, dict) or dp_lb_diff.get('enabled') is not True:
        raise ValueError(
            f"Checkpoint {model_path} is missing dp_lb_diff.enabled=True. "
            "Legacy single-objective DP ablation checkpoints are not supported.")
    lb_features = hgt_hyperparams.get('lowerbound_features')
    if not isinstance(lb_features, dict) or lb_features.get('enabled') is not True:
        raise ValueError(
            f"Checkpoint {model_path} is missing lowerbound_features.enabled=True. "
            "Legacy single-objective observation schemas are not supported.")


def load_single_obj_model(model_path: str, device: str = 'cuda:0'
                          ) -> Tuple[HGTSingleObj, Dict[str, Any],
                                     Dict[str, float]]:
    """Load and validate a makespan-only HGTSingleObj checkpoint.

    Returns:
        (policy, hgt_hyperparams, objective_weights)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    loaded = torch.load(model_path, map_location=device)
    if not isinstance(loaded, dict) or 'state_dict' not in loaded:
        raise ValueError(f"Invalid model format at {model_path}")
    hgt_hyperparams = loaded.get('hgt_hyperparams')
    if not isinstance(hgt_hyperparams, dict):
        raise ValueError(
            f"Checkpoint {model_path} has no current 'hgt_hyperparams' "
            "metadata.")
    objective_weights = hgt_hyperparams.get('objective_weights')
    if objective_weights is None:
        raise ValueError(
            f"Checkpoint {model_path} has no 'objective_weights'.")
    objective_weights = {
        'makespan_weight': float(objective_weights['makespan_weight']),
        'twt_weight': float(objective_weights['twt_weight']),
    }
    _validate_single_obj_checkpoint(
        model_path, hgt_hyperparams, objective_weights)

    policy, hgt_hyperparams, objective_weights = _load_policy_checkpoint(
        model_path, device, HGTSingleObj)
    return policy, hgt_hyperparams, objective_weights
