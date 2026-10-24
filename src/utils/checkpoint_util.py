"""Checkpoint metadata helpers.

Keep tensor payloads in ``state_dict`` untouched, but make metadata plain
Python data so PyTorch's default ``weights_only=True`` checkpoint loader can
read it.
"""

from argparse import Namespace
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a project dependency.
    np = None  # type: ignore


def checkpoint_safe_metadata(value: Any) -> Any:
    """Return a weights-only-loader-safe copy of checkpoint metadata."""
    if isinstance(value, Namespace):
        return checkpoint_safe_metadata(vars(value))
    if isinstance(value, dict):
        return {
            str(key): checkpoint_safe_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [checkpoint_safe_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        # ``torch.__version__`` is a TorchVersion subclass of str in newer
        # PyTorch builds; normalize it to a real built-in str.
        return str(value)

    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return checkpoint_safe_metadata(value.tolist())

    return str(value)


# These affect where/how a run is observed, not its stochastic learning process.
_OPERATIONAL_KEYS = {
    'device', 'result_dir', 'resume_path', 'init_from', 'create_result_dir',
    'wandb_project', 'wandb_run_name', 'wandb_enabled', 'wandb_run_id',
    'model_name', 'reproducibility', 'checkpoint_every', 'journal_flush_every',
}


def training_contract(config):
    """Freeze resolved scientific settings before a scale stage mutates config.

    Unknown configuration fields are included by default: a new scientific knob
    cannot silently evade resume checks. Runtime labels and device placement are
    omitted recursively, allowing recovery on another authorized host/device.
    """
    import json

    def clean(value):
        if isinstance(value, Namespace):
            value = vars(value)
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in sorted(value.items())
                    if k not in _OPERATIONAL_KEYS and not str(k).startswith('_')}
        if isinstance(value, (list, tuple)):
            return [clean(x) for x in value]
        return checkpoint_safe_metadata(value)

    contract = {'schema': 1, 'settings': clean(vars(config))}
    # Fail closed on non-finite scientific metadata instead of writing NaN JSON.
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


def validate_resume_contract(saved_contract, current_contract, optimizer_state, expected_lr):
    """Reject changed training conditions before loading any mutable state."""
    import math
    if not isinstance(saved_contract, dict) or saved_contract.get('schema') != 1:
        raise ValueError('Resume snapshot lacks the scientific contract; use its original frozen training source')
    if saved_contract != current_contract:
        old, new = saved_contract.get('settings', {}), current_contract.get('settings', {})
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        raise ValueError('Resume changes scientific training conditions: ' + ', '.join(changed))
    groups = optimizer_state.get('param_groups', []) if isinstance(optimizer_state, dict) else []
    if not groups:
        raise ValueError('Resume snapshot lacks optimizer parameter groups')
    for group in groups:
        actual = float(group.get('lr', float('nan')))
        if not math.isfinite(actual) or not math.isclose(actual, float(expected_lr), rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f'Resume optimizer learning rate {actual} differs from configured {expected_lr}')
