"""Central RNG-seeding helpers.

Seeding only — no strict-deterministic GPU kernels (those slow PyG scatter ~2x).
The one determinism that matters, a stable validation fixture, is handled by
``seed_context`` at the fixture's generation site.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
import random
from typing import Any, Dict, Iterator

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a project dependency.
    torch = None  # type: ignore


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable uint32 seed from a base seed and namespace parts."""
    payload = "::".join([str(int(base_seed)), *(str(p) for p in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1)


def reproducibility_metadata(seed: int) -> Dict[str, Any]:
    """Return reader-facing metadata for a seeded run."""
    metadata: Dict[str, Any] = {
        "seed": int(seed),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "numpy_seed": int(seed),
        "python_random_seed": int(seed),
    }
    if torch is not None:
        metadata.update({
            "torch_seed": int(seed),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        })
    return metadata


def set_global_seed(seed: int) -> Dict[str, Any]:
    """Seed Python, NumPy, and Torch RNGs.

    Only RNG seeding is applied. That is enough for the one piece of determinism
    that matters: a stable validation fixture (built under ``seed_context`` and
    reused across the run). We do NOT enable strict-deterministic GPU kernels
    (``torch.use_deterministic_algorithms`` / cuDNN-deterministic): they route
    PyG HGTConv's scatter onto ~2x slower kernels and the PPO update dominates
    training.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return reproducibility_metadata(seed)


@contextmanager
def seed_context(seed: int) -> Iterator[Dict[str, Any]]:
    """Temporarily seed global RNGs and restore their prior states on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state() if torch is not None else None
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_available()
        else None
    )

    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    metadata = reproducibility_metadata(seed)

    try:
        yield metadata
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
