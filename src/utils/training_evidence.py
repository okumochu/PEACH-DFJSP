"""Durable local training evidence at committed PPO update boundaries."""
import hashlib
import json
import math
import os
from pathlib import Path
from src.utils.checkpoint_util import checkpoint_safe_metadata


def json_value(value):
    value = checkpoint_safe_metadata(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_value(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def atomic_bytes(path, content):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def source_identity():
    repo = Path(__file__).resolve().parents[2]
    paths = [repo / 'config.py', *sorted((repo / 'src').rglob('*.py')),
             *sorted((repo / 'training').glob('*.py')), repo / 'training/settings.json']
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class TrainingJournal:
    """The snapshot commits a prefix; an interrupted later tail is discarded."""
    def __init__(self, root, seed, fixture_sha, next_update=0, *, repair=True):
        self.path = Path(root) / 'train_metrics.jsonl'
        self.seed = int(seed)
        self.fixture_sha = fixture_sha
        lines = self.path.read_bytes().splitlines() if self.path.exists() else []
        if len(lines) < next_update:
            raise ValueError('Training journal is shorter than the committed snapshot')
        for i, line in enumerate(lines[:next_update]):
            row = json.loads(line)
            if (row.get('update_step') != i or row.get('total_updates') != i + 1
                    or row.get('seed') != self.seed
                    or row.get('validation_fixture_sha256') != fixture_sha):
                raise ValueError('Training journal has a noncontiguous or mismatched committed prefix')
        if repair:
            atomic_bytes(self.path, b''.join(line + b'\n' for line in lines[:next_update]))
        self.next_update = int(next_update)

    def append(self, row):
        if row['update_step'] != self.next_update or row['total_updates'] != self.next_update + 1:
            raise ValueError('Training metrics must be contiguous')
        row = dict(row, seed=self.seed, validation_fixture_sha256=self.fixture_sha)
        line = json.dumps(json_value(row), sort_keys=True, allow_nan=False) + '\n'
        with self.path.open('a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.next_update += 1
