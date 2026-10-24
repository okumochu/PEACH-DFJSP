"""
Resumable logs for crash-tolerant multi-run experiments.

Each completed experiment record (keyed by a JSON-serializable dict like
``{"size": "10x5", "run_idx": 3}``) is flushed to disk immediately after the
run finishes. On restart, the log is reloaded; any record whose key is already
present is skipped.

Paths ending in ``.json`` are stored as one JSON object with ``metadata`` and
``records``. Other paths keep the original append-only JSONL format. A
truncated/partial JSONL last line from an unclean shutdown is detected on load
and the file is rewound to the last good record.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


def _json_default(o: Any) -> Any:
    """Default encoder for numpy types so payloads serialize cleanly."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")


def _canon_key(key: Dict[str, Any]) -> str:
    """Canonical, sort-key-stable string used as the in-memory dedup hash."""
    return json.dumps(key, sort_keys=True, separators=(",", ":"), default=_json_default)


class ResumableLog:
    """JSONL append-log keyed by a JSON-serializable record key.

    Each line on disk is ``{"_v": 1, "key": <key_dict>, "payload": <payload_dict>}``.

    Crash tolerance: on ``__init__``, we walk the file line by line; the first
    parse error truncates the file at the last good byte offset so subsequent
    appends start from a clean position.

    Durability: every ``append`` calls ``flush()`` and ``os.fsync()`` so a
    completed record survives a kernel panic, not just a process kill.
    """

    SCHEMA_V = 1

    def __init__(self, path: str):
        self.path = path
        self._completed: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}
        self._json_mode = path.endswith(".json") and not path.endswith(".jsonl")
        self._fh = None
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._load_existing()
        if not self._json_mode:
            # Open append handle AFTER load so we keep the (possibly truncated) tail.
            self._fh = open(path, "a", buffering=1)

    def _load_existing(self) -> None:
        if self._json_mode:
            self._load_existing_json()
            return
        if not os.path.isfile(self.path):
            return
        last_good_offset = 0
        needs_newline = False
        with open(self.path, "rb") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                    if not isinstance(rec, dict) or rec.get("_v") != self.SCHEMA_V:
                        # Unknown schema or non-dict — treat as bad tail.
                        break
                    key = rec.get("key")
                    payload = rec.get("payload")
                    if not isinstance(key, dict) or not isinstance(payload, dict):
                        break
                    self._completed[_canon_key(key)] = payload
                    last_good_offset += len(raw)
                    needs_newline = not raw.endswith(b"\n")
                except (json.JSONDecodeError, ValueError):
                    break
        size = os.path.getsize(self.path)
        if last_good_offset != size or needs_newline:
            # Preserve a complete final JSON object even if its newline was
            # lost. The next append must not concatenate two JSON objects.
            with open(self.path, "rb+") as fh:
                fh.truncate(last_good_offset)
                if needs_newline:
                    fh.seek(last_good_offset)
                    fh.write(b"\n")
                fh.flush()
                os.fsync(fh.fileno())

    def _load_existing_json(self) -> None:
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        if not isinstance(rec, dict) or rec.get("_v") != self.SCHEMA_V:
            raise ValueError(f"Unknown resumable JSON log schema in {self.path}")
        metadata = rec.get("metadata", {})
        records = rec.get("records", [])
        if not isinstance(metadata, dict) or not isinstance(records, list):
            raise ValueError(f"Malformed resumable JSON log in {self.path}")
        self._metadata = metadata
        for item in records:
            if not isinstance(item, dict):
                raise ValueError(f"Malformed record in {self.path}")
            key = item.get("key")
            payload = item.get("payload")
            if not isinstance(key, dict) or not isinstance(payload, dict):
                raise ValueError(f"Malformed record in {self.path}")
            self._completed[_canon_key(key)] = payload

    def _write_json(self) -> None:
        if not self._json_mode:
            return
        records = []
        for ck, payload in self._completed.items():
            records.append({"key": json.loads(ck), "payload": payload})
        rec = {
            "_v": self.SCHEMA_V,
            "metadata": self._metadata,
            "records": records,
        }
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, default=_json_default)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, self.path)

    def is_done(self, key: Dict[str, Any]) -> bool:
        return _canon_key(key) in self._completed

    def get(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._completed.get(_canon_key(key))

    def append(self, key: Dict[str, Any], payload: Dict[str, Any]) -> None:
        ck = _canon_key(key)
        if ck in self._completed:
            return
        self._completed[ck] = payload
        if self._json_mode:
            self._write_json()
            return
        rec = {"_v": self.SCHEMA_V, "key": key, "payload": payload}
        line = json.dumps(rec, default=_json_default) + "\n"
        self._fh.write(line)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def replace(self, key: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """Overwrite an existing record (``append`` deliberately never does).

        Needed when a run is re-solved because the cached record is incomplete
        — e.g. it was written by an invocation that skipped a solver which the
        current invocation wants. Both loaders are last-wins, so appending a
        fresh line in jsonl mode is equivalent to overwriting.
        """
        ck = _canon_key(key)
        if ck not in self._completed:
            self.append(key, payload)
            return
        self._completed[ck] = payload
        if self._json_mode:
            self._write_json()
            return
        rec = {"_v": self.SCHEMA_V, "key": key, "payload": payload}
        self._fh.write(json.dumps(rec, default=_json_default) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store run-level metadata used to reproduce the experiment."""
        self._metadata = dict(metadata)
        if self._json_mode:
            self._write_json()

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def all_records(self) -> List[Dict[str, Any]]:
        """Return flat dicts merging key+payload, for final aggregation.

        Iteration order matches insertion order (Python 3.7+ dict guarantee).
        """
        out: List[Dict[str, Any]] = []
        for ck, payload in self._completed.items():
            key = json.loads(ck)
            merged = dict(key)
            merged.update(payload)
            out.append(merged)
        return out

    def num_completed(self) -> int:
        return len(self._completed)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> "ResumableLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def resolve_run_dir(
    base_dir: str,
    run_name: Optional[str],
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> Tuple[str, bool]:
    """Choose the directory for an experiment invocation.

    - ``run_name`` set      -> ``<base_dir>/<run_name>/`` (deterministic; same
                                command resumes). ``is_resume_candidate`` is
                                True if the dir already exists.
    - ``run_name`` is None  -> ``<base_dir>/<timestamp>/`` (status-quo fresh
                                dir). ``is_resume_candidate`` always False.
    """
    if run_name:
        run_dir = os.path.join(base_dir, run_name)
        is_resume_candidate = os.path.isdir(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, is_resume_candidate
    timestamp = time.strftime(timestamp_fmt)
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, False


_CKPT_FINGERPRINTS: Dict[str, Tuple[tuple, str]] = {}


def _file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def checkpoint_fingerprint(path: Optional[str]) -> Optional[str]:
    """Short content hash, cached only while the file identity is unchanged.

    Belongs in a resume key, not just in the metadata. ``ResumableLog`` keys
    only on what the caller puts in the key, so a sweep re-run with a different
    ``--model_path`` replays the previous model's rows verbatim while the run
    metadata — rewritten unconditionally on every invocation — claims the new
    one. Hashing the BYTES rather than the path also catches the common case of
    a checkpoint overwritten in place, which every training run does to
    ``best_*.pth``.

    Returns None when there is no path or the file cannot be read, and callers
    treat None as "not recorded" so records written before this existed stay
    replayable.
    """
    if not path:
        return None
    try:
        identity = _file_identity(os.stat(path))
        cached = _CKPT_FINGERPRINTS.get(path)
        if cached is not None and cached[0] == identity:
            return cached[1]
        import hashlib
        # An atomic checkpoint replacement may occur between stat and open.
        # Hash the open file and publish only an identity that stayed stable.
        for _ in range(2):
            with open(path, 'rb') as f:
                identity = _file_identity(os.fstat(f.fileno()))
                h = hashlib.sha256()
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
                unchanged = identity == _file_identity(os.fstat(f.fileno()))
            if unchanged and identity == _file_identity(os.stat(path)):
                fingerprint = h.hexdigest()[:16]
                _CKPT_FINGERPRINTS[path] = (identity, fingerprint)
                return fingerprint
    except OSError:
        _CKPT_FINGERPRINTS.pop(path, None)
        return None
    _CKPT_FINGERPRINTS.pop(path, None)
    raise RuntimeError(f'Checkpoint changed while computing its fingerprint: {path}')
