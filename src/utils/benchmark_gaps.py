"""Pair DANIEL's synthetic instance names with reference objectives.

DANIEL sorts filenames by their numeric fields and averages per-instance
percentage gaps. A partial or mismatched group must not silently become a
ratio of group means or a comparison with the full published test set.
"""
from pathlib import Path
import math
import re
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT / "data/benchmarks/daniel"
SOURCE_ROOT = ROOT / "data/benchmarks"


def reference_objectives(generator, group, *, reference_root=REFERENCE_ROOT,
                         source_root=SOURCE_ROOT):
    if generator not in ("SD1", "SD2") or not group.startswith(generator + "_"):
        raise ValueError(f"Invalid synthetic benchmark identity {generator}/{group}")
    size = group.split("_", 1)[1]
    if not re.fullmatch(r"\d+x\d+", size):
        raise ValueError(f"Invalid synthetic benchmark size {group}")
    stem = size + ("+mix" if generator == "SD2" else "")
    files = list((Path(source_root) / generator / stem).glob("*.fjs"))
    # These are the two stable sorts in DANIEL data_utils.load_data_from_files.
    files.sort(key=lambda path: int(re.findall(r"\d+", path.name)[0]))
    files.sort(key=lambda path: int(re.findall(r"\d+", path.name)[-1]))
    # SD1 uses published instance IDs such as 3128_10j_5m; SD2 uses
    # size_NNN names. The actual names, not an invented sequence, identify rows.
    names = [path.stem for path in files]
    if len(files) != 100 or len(set(names)) != 100:
        raise ValueError(f"Expected all 100 distinct instance files for {group}")
    folder = Path(reference_root) / "or_solution" / generator
    source = folder / f"solution_{stem}.npy"
    if not source.is_file():
        raise ValueError(f"Missing per-instance reference objectives {source}")
    values = np.load(source, allow_pickle=False)
    if values.shape != (100, 2) or not np.isfinite(values).all() or (values[:, 0] <= 0).any():
        raise ValueError(f"Invalid reference objective array {source}")
    # Some published per-instance directories are incomplete. Every available
    # reference must agree with the aggregate row indexed by its own filename.
    for individual in (folder / stem).glob("*.npy"):
        match = re.fullmatch(re.escape(f"solution_{stem}_") + r"(\d{3})\.npy", individual.name)
        index = int(match.group(1)) - 1 if match else -1
        if not 0 <= index < 100 or not np.array_equal(
                np.load(individual, allow_pickle=False), values[index]):
            raise ValueError(f"Reference row alignment mismatch {individual}")
    return dict(zip(names, values[:, 0].astype(float)))


def mean_instance_gap(items, references):
    """Return the mean paired percentage gap after checking complete coverage."""
    measured = {}
    for name, value in items:
        if name in measured:
            raise ValueError(f"Duplicate benchmark instance {name}")
        if not isinstance(value, (float, int, np.number)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid makespan for benchmark instance {name}")
        measured[name] = float(value)
    if len(references) != 100 or set(measured) != set(references):
        raise ValueError("Mean benchmark gap requires the same 100 measured and reference instances")
    if any(not math.isfinite(value) or value <= 0 for value in references.values()):
        raise ValueError("Reference makespans must be finite and positive")
    return statistics.mean(100 * (value / references[name] - 1)
                           for name, value in measured.items())
