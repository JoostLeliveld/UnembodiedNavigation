"""The frozen detector: its identity, and what it actually does.

An instrument, not a contribution.  Declared here and never changed by anything measured
downstream -- it was once retrained to make a residual smaller, and a label-convention change
then looked like a bias result.

Reported, never tuned.  **Do not quote the training mAP**: its held-out places sit a median
0.70 m from training places against a 0.67 m grid, so it measures near-neighbour
interpolation.  The number that goes in the paper is the detection rate on the separate
commissioning capture, which degrades sensibly with range and occlusion.
"""
from __future__ import annotations

import collections
import json

import numpy as np

from capture import MIN_BIN, TRAINSET

DETECTOR = {
    "weights": "logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt",
    "sha256": "efff1949c1b8cdeeb11438b36de80f6cf8daeef5f3a4682cfce8ae7dfe314f34",
    "confidence_threshold": 0.25,
    "image_size": 960,
    "box_convention": "semantic_mask_half_open_extent: [xmin, ymin, xmax_occupied+1, ymax_occupied+1]",
}


def characterize(rows):
    """What the detector does, reported and not tuned."""
    by_range, by_vis, by_border = (collections.defaultdict(lambda: [0, 0]),
                                   collections.defaultdict(lambda: [0, 0]),
                                   collections.defaultdict(lambda: [0, 0]))
    nbox = collections.Counter()
    for r, d in rows:
        hit = d["detected"] == "1"
        nbox[int(d["n_detections"])] += 1
        try:
            rb = min(int(float(r["camera_range_m"]) // 4), 5)
            by_range[rb][1] += 1; by_range[rb][0] += hit
        except (ValueError, KeyError):
            pass
        try:
            af = float(r["area_fraction"])
            vb = next(i for i, t in enumerate([0.3, 0.6, 0.8, 0.95, 9e9]) if af < t)
            by_vis[vb][1] += 1; by_vis[vb][0] += hit
        except (ValueError, KeyError, StopIteration):
            pass
        try:
            b = int(float(r["border_contact"]))
            by_border[b][1] += 1; by_border[b][0] += hit
        except (ValueError, KeyError):
            pass
    man = json.loads(TRAINSET.read_text())
    tr = np.array(sorted({(round(s["robot_x"], 3), round(s["robot_y"], 3))
                          for s in man["sample_records"] if s["split"] == "train"}))
    va = np.array(sorted({(round(s["robot_x"], 3), round(s["robot_y"], 3))
                          for s in man["sample_records"] if s["split"] == "val"}))
    dmin = np.min(np.linalg.norm(va[:, None, :] - tr[None, :, :], axis=2), axis=1)
    tot = sum(nbox.values())
    hits = sum(1 for _r, d in rows if d["detected"] == "1")
    return {
        "trials": tot, "detections": hits, "detection_rate": hits / tot,
        "multi_box_frames": sum(v for k, v in nbox.items() if k > 1),
        "multi_box_fraction": sum(v for k, v in nbox.items() if k > 1) / tot,
        "by_range": {f"{k*4}-{k*4+4}m": {"n": v[1], "rate": v[0] / v[1]}
                     for k, v in sorted(by_range.items()) if v[1] >= MIN_BIN},
        "by_visible_fraction": {lab: {"n": by_vis[i][1], "rate": by_vis[i][0] / by_vis[i][1]}
                                for i, lab in enumerate(["<0.3", "0.3-0.6", "0.6-0.8",
                                                         "0.8-0.95", ">0.95"])
                                if by_vis[i][1] >= MIN_BIN},
        "at_frame_edge": {("touching" if k else "clear"): {"n": v[1], "rate": v[0] / v[1]}
                          for k, v in sorted(by_border.items()) if v[1] >= MIN_BIN},
        "training_split_hardness": {
            "train_cells": int(tr.shape[0]), "val_cells": int(va.shape[0]),
            "grid_spacing_m": 0.67,
            "val_to_nearest_train_m": {
                "min": float(dmin.min()), "median": float(np.median(dmin)),
                "p90": float(np.percentile(dmin, 90)), "max": float(dmin.max())},
            "note": ("val places sit about one grid step from training places, so the "
                     "training mAP measures near-neighbour interpolation and is NOT "
                     "evidence of generalization to unseen places"),
        },
        "false_positive_rate": None,
        "false_positive_note": ("UNMEASURED. The capture kept one negative frame per "
                                "camera, and the 6173 poses where the robot was not "
                                "visible were logged without saving an image, so "
                                "P(detector fires with no robot in view) is unknown."),
    }
