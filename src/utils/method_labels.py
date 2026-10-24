"""Canonical method-label conventions for CSV reports."""

from typing import Optional, Tuple


_ROW_PREFIX_ALIASES = {
    'greedy': 'rl',
    'hgt': 'rl',
    'rl_greedy': 'rl',
    'rl': 'rl',
}


def parse_canonical_label(canonical: str) -> Tuple[str, Optional[float]]:
    """Return ``(wide_row_prefix, None)`` for current non-parametric labels."""
    label = canonical.strip().lower()
    return _ROW_PREFIX_ALIASES.get(label, label), None
