"""Is this sighting usable?  The one question that decides whether a correction arrives.

A sighting that fails is an **absent** measurement, not a bad one -- it belongs to the
availability side of the problem.  Dropping these silently would turn a detection-rate
result into a localization result, so the reasons come back with the verdict.

Every test compares a detection against a prediction, so every test is available at runtime:
none needs ground truth or the segmentation mask.
"""
from __future__ import annotations

from camera import IMG_H, IMG_W

# The admission check.  Every test compares a detection against a prediction, so every test
# is available at runtime: none of them needs ground truth or the segmentation mask.
#
# It fails safe.  Corrupt the pose it predicts from and the rate of sightings *wrongly
# kept* stays near 1% while *wrongly dropped* climbs to 29% -- it discards good sightings
# rather than admitting bad ones, which is the right direction, because a bad sighting
# corrupts the estimate while a lost one only costs availability.
GATE = {
    "min_height_fraction": 0.85,   # at least this tall relative to the predicted box
    "max_width_error": 0.10,       # within this fraction of the predicted width
    "max_bottom_above_px": 3.0,    # bottom edge no higher than this above the prediction
    "reject_border_contact": True, # a box touching the frame edge is truncated, not measured
}


def gate(pred_box, det_box, img_w=IMG_W, img_h=IMG_H):
    """Is this sighting usable?  Returns ``(passed, reasons_it_failed)``.

    A sighting that fails is an **absent** measurement, not a bad one: it belongs to the
    availability side of the problem.  Silently dropping these would turn a detection-rate
    result into a localization result, so the reasons are returned rather than discarded.
    """
    u0, v0, u1, v1 = pred_box
    x0, y0, x1, y1 = det_box
    pw, ph = u1 - u0, v1 - v0
    dw, dh = x1 - x0, y1 - y0
    reasons = []
    if ph <= 0 or pw <= 0:
        return False, ["degenerate_prediction"]
    if dh / ph < GATE["min_height_fraction"]:
        reasons.append("too_short")
    if abs(dw - pw) / pw > GATE["max_width_error"]:
        reasons.append("wrong_width")
    if (v1 - y1) > GATE["max_bottom_above_px"]:
        reasons.append("bottom_hidden")
    if GATE["reject_border_contact"] and (
            x0 <= 1 or y0 <= 1 or x1 >= img_w - 1 or y1 >= img_h - 1):
        reasons.append("touches_frame_edge")
    return (not reasons), reasons
