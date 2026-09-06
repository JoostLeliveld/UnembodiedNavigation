#!/usr/bin/env python3
"""Shared per-panel scaling for the warehouse residual-arrow maps.

One arrow gain and one colour cap across a five-rung ladder cannot work: raw box error is
about 30 cm and the neural residual about 3 cm, so a scale that suits the first collapses the
second to dots and pins its colour at the bottom of the bar. Each panel therefore gets its own
gain and cap, both derived from that panel's own data, both printed in its title, and each
panel carries a scale bar so a reader can never infer magnitude from arrow length alone.
"""
from __future__ import annotations

import numpy as np

# The gain is set so that a LARGE error (the 90th percentile) draws to a readable length and
# almost nothing has to be clipped. Setting it from the median instead -- which an earlier
# version did -- produces a gain of 40 or 50 on a heavy-tailed panel, clips a third of the
# arrows to the same maximum length, and destroys exactly the information the arrow carries.
TARGET_LONG_ARROW_M = 2.1
LONG_ARROW_QUANTILE = 0.90
GAIN_LIMITS = (1.0, 26.0)
SCALE_BAR_STEPS_CM = (1, 2, 5, 10, 20, 30, 50, 100)


def panel_scale(errors_m: np.ndarray) -> dict:
    """Arrow gain and colour cap for one panel, from that panel's own residuals."""
    finite = np.asarray([value for value in np.asarray(errors_m, dtype=float)
                         if np.isfinite(value)], dtype=float)
    if not finite.size:
        return {"gain": 3.0, "cap": 0.1, "median_m": float("nan"), "n": 0}
    median = float(np.median(finite))
    cap = max(1e-3, float(np.quantile(finite, 0.95)))
    long_error = max(float(np.quantile(finite, LONG_ARROW_QUANTILE)), 1e-4)
    gain = float(np.clip(TARGET_LONG_ARROW_M / long_error, *GAIN_LIMITS))
    return {"gain": gain, "cap": cap, "median_m": median,
            "long_error_m": long_error, "n": int(finite.size)}


def drawn_shrink(dx, dy, gain: float, max_drawn_m: float = 2.6):
    """Shrink factors so no *drawn* arrow runs off the warehouse.

    The gain is applied after the residual, so clipping the residual in world units is not
    enough: a 1 m residual at gain 19 would draw a 19 m arrow across the whole floor. Clip
    the drawn length instead, and report how many arrows that touched.
    """
    length = np.hypot(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)) * gain
    shrink = np.minimum(1.0, max_drawn_m / np.maximum(length, 1e-12))
    return shrink, int(np.sum(shrink < 1.0))


def scale_bar(ax, gain: float, *, layout, colour: str, fontsize: float = 9.0) -> None:
    """Draw 'this is what N cm of error looks like here' inside one panel."""
    span_x = layout.width if hasattr(layout, "width") else None
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = abs(x1 - x0)
    # pick the largest round error whose drawn arrow still fits in a fifth of the panel
    budget = 0.20 * width
    chosen = SCALE_BAR_STEPS_CM[0]
    for step in SCALE_BAR_STEPS_CM:
        if (step / 100.0) * gain <= budget:
            chosen = step
    length = (chosen / 100.0) * gain
    bx = x0 + 0.045 * width
    by = y0 + 0.055 * abs(y1 - y0)
    ax.annotate(
        "", xy=(bx + length, by), xytext=(bx, by),
        arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0, shrinkA=0, shrinkB=0),
        zorder=9, annotation_clip=False,
    )
    ax.text(bx, by + 0.030 * abs(y1 - y0), f"{chosen} cm", fontsize=fontsize,
            color=colour, fontweight="bold", va="bottom", zorder=9)
    return span_x
