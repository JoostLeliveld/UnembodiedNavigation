"""Coverage checks for timestamped zero-order-held motion inputs.

A long camera interval can still have dense measured odometry. Its duration is
not a missing-motion interval. These checks use only the recorded input clock.
"""
from __future__ import annotations

import math


def covers_interval(entries, start_s: float, end_s: float, max_gap_s: float) -> bool:
    """Require a finite, ordered input at/before start and bounded gaps to end.

    Future samples cannot fill a past gap. A stale sample before the interval
    also fails, even if the interval itself is short. This checks temporal
    support, not whether the odometry noise model is statistically calibrated.
    """
    if not all(math.isfinite(v) for v in (start_s, end_s, max_gap_s)):
        return False
    if end_s <= start_s or max_gap_s <= 0:
        return False
    relevant = []
    previous = None
    last_stamp = -math.inf
    for entry in entries:
        t, v, w = entry
        if not all(math.isfinite(x) for x in (t, v, w)) or t < last_stamp:
            return False
        last_stamp = t
        if t <= start_s:
            previous = t
        elif t <= end_s:
            relevant.append(t)
    if previous is None or start_s-previous > max_gap_s + 1e-9:
        return False
    for t in [*relevant, end_s]:
        if t-previous > max_gap_s + 1e-9:
            return False
        previous = t
    return True
