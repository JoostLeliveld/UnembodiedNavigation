"""The commissioning capture: what was collected, and which job each part may serve.

Two products come out of commissioning and they have opposite needs:

* the **offset** is reading-minus-truth, so it cannot be measured without ground truth --
  but it converges by about ten floor positions;
* the **availability map** needs no truth at all, and is still improving at 250 positions.

So the expensive dataset is the one that needs no truth, and the one that needs truth is
small.  They get **disjoint floor positions** here and nothing measured on one may inform the
other.  That costs nothing: ``admission`` never touches the offset, so they are already
independent.

**Never match a row in one file to a row in another by coordinate.**  The capture stores full
floats, every written file stores formatted ones, and the sample grid lands exactly on
rounding ties -- rounding, lattice-snapping and decimal formatting each fail somewhere.  That
silently corrupted results three times, once inflating "no camera can help here" from 15% to
24%.  Every position carries an integer id, written into every file.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from observation import predicted_box

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
TRAINSET = REPO / "logs/perception_datasets/warehouse_v2_yolo_20260821/merged_v2/dataset_manifest.json"
READINGS = "detector_readings_halfopen_detect_20260825.csv"
OUT = REPO / "logs/studies/measurement_commissioning"

OFFSET_POSITIONS = 20         # measured: the offset converges by ~8-12 positions
OFFSET_PER_BAND = 5           # spots per distance band; bands = OFFSET_POSITIONS / this
EDGE_OFFSET_SAMPLES = 1200
MIN_BIN = 25


def sha256(path: Path) -> str:
    hh = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hh.update(chunk)
    return hh.hexdigest()


def load_capture():
    """Join the capture index with the frozen detector's readings, keeping every trial."""
    readings = {}
    for p in sorted(DATASET.glob(f"camera_*/{READINGS}")):
        for r in csv.DictReader(open(p)):
            readings[(p.parent.name, r["image"])] = r
    rows = []
    for r in csv.DictReader(open(DATASET / "localization_calibration_index_hull.csv")):
        d = readings.get((r["camera_id"], r["image"]))
        if d is not None:
            rows.append((r, d))
    return rows


def position_ids(rows):
    """A stable integer id per floor position, assigned once and written into every file."""
    seen = {}
    for r, _d in rows:
        key = (round(float(r["robot_x"]), 4), round(float(r["robot_y"]), 4))
        if key not in seen:
            seen[key] = len(seen)
    return seen


def split_positions(rows):
    """Give each floor position to exactly one job: bias, or availability.

    **How the bias spots are chosen matters more than how many there are.**  Twenty spots
    spread evenly across distance-to-camera yielded only 133 sightings and left 1.16 cm at
    long range, because even spacing keeps picking places a single camera can see.  Twenty
    spots chosen as the *most overlooked* within each of four distance bands yield 192
    sightings and 0.89 cm.  Both cost the same twenty parkings; one simply puts the robot
    where several cameras are already looking.

    The rule uses only the camera mounts and the building's geometry -- which positions have
    a clear line of sight to how many cameras -- so it can be applied to a floor plan before
    any detector exists.
    """
    bands = max(int(OFFSET_POSITIONS / OFFSET_PER_BAND), 1)
    sightlines = collections.defaultdict(set)
    for p in sorted(DATASET.glob("camera_*/label_diagnostics.csv")):
        cam = p.parent.name
        for r in csv.DictReader(open(p)):
            try:
                if int(float(r["semantic_robot_pixels"] or 0)) > 0:
                    sightlines[(round(float(r["robot_x"]), 4),
                                round(float(r["robot_y"]), 4))].add(cam)
            except (ValueError, KeyError):
                continue
    ranges = collections.defaultdict(list)
    for r, _d in rows:
        pos = (round(float(r["robot_x"]), 4), round(float(r["robot_y"]), 4))
        try:
            ranges[pos].append(float(r["camera_range_m"]))
        except (ValueError, KeyError):
            continue
    candidates = [p for p in ranges if p in sightlines]
    if not candidates:
        return set(), set(ranges)
    width = 24.0 / bands
    grouped = collections.defaultdict(list)
    for p in candidates:
        grouped[min(int(float(np.median(ranges[p])) // width), bands - 1)].append(p)
    bias = set()
    for b in sorted(grouped):
        bias |= set(sorted(grouped[b], key=lambda q: (-len(sightlines[q]), q))[:OFFSET_PER_BAND])
    return bias, set(ranges) - bias


def verify_reconstruction(cams, rows):
    """Rebuilt cameras must reproduce the hull predictions sealed into the capture."""
    err = []
    for r, _d in rows[:4000]:
        try:
            stored = float(r["pred_hull_bottom_v"])
        except (ValueError, KeyError):
            continue
        box = predicted_box(cams[r["camera_id"]], float(r["robot_x"]),
                            float(r["robot_y"]), float(r["robot_yaw"]))
        if box is not None:
            err.append(abs(box[3] - stored))
    err = np.array(err)
    return {"n": int(err.size), "median_px": float(np.median(err)),
            "p99_px": float(np.percentile(err, 99)),
            "passed": bool(err.size and float(np.median(err)) < 0.5)}


def measure_edge_offset(cams, rows):
    """The hull-to-reported-box offset, measured from the masks with the detector absent.

    A property of the robot model, the renderer and the label convention -- so it belongs in
    ``h(x)`` rather than in a fitted correction that would silently absorb it.  Only
    near-fully-visible, non-truncated sightings count: a mask whose bottom is behind a rack
    does not show the robot's true contact edge.
    """
    picked, du, dv = 0, [], []
    for r, _d in rows:
        if picked >= EDGE_OFFSET_SAMPLES:
            break
        try:
            if float(r["height_fraction"]) < 0.98 or int(float(r["border_contact"])) != 0:
                continue
        except (ValueError, KeyError):
            continue
        mask_path = DATASET / r["camera_id"] / r["image"].replace("images/", "masks/")
        if not mask_path.exists():
            continue
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        occupied = (m > 0) if m.ndim == 2 else m.any(axis=2)
        ys, xs = np.nonzero(occupied)
        if ys.size < 30:
            continue
        # the same half-open extent the training labels were built with
        mask_u = 0.5 * (float(xs.min()) + float(xs.max()) + 1.0)
        mask_v = float(ys.max()) + 1.0
        box = predicted_box(cams[r["camera_id"]], float(r["robot_x"]),
                            float(r["robot_y"]), float(r["robot_yaw"]))
        if box is None:
            continue
        du.append(mask_u - 0.5 * (box[0] + box[2]))
        dv.append(mask_v - box[3])
        picked += 1
    du, dv = np.array(du), np.array(dv)
    return {"n": int(du.size),
            "edge_offset_u_px": float(du.mean()), "edge_offset_v_px": float(dv.mean()),
            "sd_u_px": float(du.std()), "sd_v_px": float(dv.std()),
            "standard_error_u_px": float(du.std() / math.sqrt(max(du.size, 1))),
            "standard_error_v_px": float(dv.std() / math.sqrt(max(dv.size, 1)))}


def write_availability(rows, sightings, offset_positions, pos_id):
    """One row per (camera, pose) trial, including the trials where nothing was seen.

    Built by identity while both sides are in memory, never reconstructed later by joining
    files on coordinates.
    """
    usable = {(s["camera"], s["image"]) for s in sightings}
    seen_index = {(r["camera_id"], r["image"]) for r, _d in rows}
    out = []
    for p in sorted(DATASET.glob("camera_*/label_diagnostics.csv")):
        cam = p.parent.name
        for r in csv.DictReader(open(p)):
            try:
                px = int(float(r["semantic_robot_pixels"] or 0))
            except ValueError:
                px = 0
            img = r["image"] or ""
            pos = (round(float(r["robot_x"]), 4), round(float(r["robot_y"]), 4))
            out.append({
                "position_id": pos_id.get(pos, -1),
                "camera": cam, "x": r["robot_x"], "y": r["robot_y"], "yaw": r["robot_yaw"],
                "job": "offset" if pos in offset_positions else "availability",
                "image": img, "line_of_sight": int(px > 0),
                "in_index": int((cam, img) in seen_index) if img else 0,
                "usable": int((cam, img) in usable) if img else 0,
            })
    with open(OUT / "availability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["position_id", "camera", "x", "y", "yaw", "job",
                                           "image", "line_of_sight", "in_index", "usable"])
        w.writeheader()
        for row in out:
            w.writerow(row)
    return {"trials": len(out),
            "line_of_sight": sum(r["line_of_sight"] for r in out),
            "usable": sum(r["usable"] for r in out),
            "availability_trials": sum(1 for r in out if r["job"] == "availability"),
            "offset_trials": sum(1 for r in out if r["job"] == "offset")}


def write_sightings(sightings):
    with open(OUT / "sightings.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["position_id", "camera", "x", "y", "yaw", "range_m", "rel_heading_rad",
                    "du_px", "dv_px", "confidence"])
        for s in sightings:
            w.writerow([s["position_id"], s["camera"], f"{s['x']:.6f}", f"{s['y']:.6f}",
                        f"{s['yaw']:.6f}", f"{s['range_m']:.3f}", f"{s['rel_heading_rad']:.4f}",
                        f"{s['du_px']:.4f}", f"{s['dv_px']:.4f}", f"{s['confidence']:.4f}"])


def write_offset_positions(offset_positions, pos_id):
    with open(OUT / "offset_positions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["position_id", "x", "y"])
        for x, y in sorted(offset_positions):
            w.writerow([pos_id.get((round(x, 4), round(y, 4)), -1), f"{x:.6f}", f"{y:.6f}"])
