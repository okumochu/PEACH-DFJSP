"""Descriptive statistics used in the manuscript's retained evaluations."""
import math
import statistics as st
from verification.schedules import need


def paired(values, references):
    need(len(values) == len(references) and all(v > 0 for v in references), 'Invalid pairing')
    gaps = [100 * (value / reference - 1) for value, reference in zip(values, references)]
    # The frozen reporting protocol uses a 2e-6 relative and .002 absolute objective tolerances.
    outcomes = [('tie' if math.isclose(v, r, rel_tol=2e-6, abs_tol=.002)
                 else 'win' if v < r else 'loss') for v, r in zip(values, references)]
    return {'n': len(values), 'mean_paired_gap_percent': st.mean(gaps),
            'sample_sd_gap_pp': st.stdev(gaps),
            'wins': outcomes.count('win'), 'ties': outcomes.count('tie'), 'losses': outcomes.count('loss')}
