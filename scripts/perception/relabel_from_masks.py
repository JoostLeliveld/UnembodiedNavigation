"""Rewrite every YOLO-seg label from its saved mask, keeping split silhouettes whole.

Why this exists
---------------
`capture_yolo_dataset._polygon_from_mask` keeps only the LARGEST contour. On the
TurtleBot3 Burger, at 7.6 x 11.4 px, that was harmless. On the 0.80 x 0.55 m AMR,
at 60 x 39 px, a rack upright bisects the silhouette and the dropped component is
a real piece of the robot.

Measured over the 3037 saved masks of warehouse_v2_yolo_20260821:

* 235 samples lose more than 10% of their labelled area;
* the bottom-centre pixel -- *the measurement this paper reads* -- moves by more
  than 3 px in 139 samples, with a tail to 30.9 px in u and 23.0 px in v.

The median sample is untouched (most extra components are single-pixel boundary
specks), so this is a narrow defect with a severe tail: 4.6% of samples, against a
pixel-noise scale of 0.5-2.5 px.

The policy, and the option that was measured and rejected
--------------------------------------------------------
A YOLO-seg label line is ONE closed polygon, so a genuinely disjoint silhouette
cannot be written exactly. The obvious repair -- the convex hull of every component
-- was implemented and measured first, and it is worse than the disease: it paints
the occluder strip between components as robot, and on this data the hull reaches
**11.3x the mask area** at the tail. Teaching the detector that a rack upright is
part of the robot corrupts the same bottom edge it was meant to protect.

So the policy is graded, and it never over-claims:

* one significant component  -> its own contour, exactly as before;
* several, largest >= 90% of the area -> the largest contour. The dropped remainder
  is under a tenth of the silhouette and the measured bottom point barely moves;
* several, largest < 90%  -> the label still gets the largest contour (never an
  empty label, which would teach a false negative), but the sample is written to
  `split_silhouette_exclusions.csv` and excluded from the training file list.

A silhouette split into pieces of comparable size cannot be represented honestly by
one polygon, so it is not represented: it is recorded, kept out of training and kept
out of any projection target, and it stays in the capture as a fact about that place.

Masks are the ground truth here and are not modified. Labels are derived, so
rewriting them keeps the dataset self-consistent: image <-> mask <-> label.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

MIN_COMPONENT_SHARE = 0.01     # a component below 1% of the mask is a boundary speck
EPSILON_RATIO = 0.010          # matches capture_yolo_dataset._polygon_from_mask
BAND_PX = 3.0


def significant_components(mask_u8: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    total = float(areas.sum())
    keep = [i + 1 for i, a in enumerate(areas) if a / total >= MIN_COMPONENT_SHARE]
    return [(labels == i).astype(np.uint8) for i in keep]


def polygon_union_hull(mask_u8: np.ndarray) -> tuple[np.ndarray | None, dict]:
    comps = significant_components(mask_u8)
    info = {"n_components": len(comps), "largest_share": float("nan"),
            "hull_over_claim": float("nan"), "excluded_from_training": False}
    if not comps:
        return None, info
    areas = [float(c.sum()) for c in comps]
    info["largest_share"] = max(areas) / sum(areas)
    points = []
    for comp in comps:
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points.append(contour.reshape(-1, 2))
    if not points:
        return None, info
    # always the largest component's own contour: exact, and never over-claiming
    biggest = comps[int(np.argmax(areas))]
    contours, _ = cv2.findContours(biggest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return None, info
    info["hull_over_claim"] = area / float((mask_u8 > 0).sum())
    info["excluded_from_training"] = bool(
        len(comps) > 1 and info["largest_share"] < 0.90
    )
    epsilon = max(1.0, EPSILON_RATIO * cv2.arcLength(contour, True))
    pts = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
    if pts.shape[0] < 3:
        return None, info
    return pts, info


def bottom_centre(polygon: np.ndarray) -> tuple[float, float]:
    v = float(polygon[:, 1].max())
    band = polygon[polygon[:, 1] >= v - BAND_PX]
    return (float(band[:, 0].mean()) if len(band) else float(polygon[:, 0].mean())), v


def write_label(path: Path, polygon: np.ndarray | None, w: int, h: int) -> None:
    if polygon is None or len(polygon) < 3:
        path.write_text("", encoding="utf-8")
        return
    values = ["0"]
    for x, y in polygon:
        values.append(f"{np.clip(x / float(w), 0.0, 1.0):.8f}")
        values.append(f"{np.clip(y / float(h), 0.0, 1.0):.8f}")
    path.write_text(" ".join(values) + "\n", encoding="utf-8")


def read_label(path: Path, w: int, h: int) -> np.ndarray | None:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split()
    coords = np.asarray([float(v) for v in parts[1:]], dtype=float).reshape(-1, 2)
    coords[:, 0] *= w
    coords[:, 1] *= h
    return coords


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="per-camera or multi-camera capture root")
    ap.add_argument("--apply", action="store_true", help="write labels; otherwise report only")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    masks = sorted(root.glob("camera_*/masks/*/*")) or sorted(root.glob("masks/*/*"))
    if not masks:
        print(f"no saved masks under {root}")
        return 1

    rows, changed = [], 0
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape[:2]
        label_path = Path(
            str(mask_path).replace("/masks/", "/labels/")
        ).with_suffix(".txt")
        if not label_path.exists():
            continue
        polygon, info = polygon_union_hull(mask)
        before = read_label(label_path, w, h)
        du = dv = float("nan")
        if before is not None and polygon is not None:
            bu, bv = bottom_centre(before)
            nu, nv = bottom_centre(polygon)
            du, dv = nu - bu, nv - bv
        if args.apply:
            write_label(label_path, polygon, w, h)
        if info["n_components"] > 1:
            changed += 1
        rows.append({"mask": str(mask_path.relative_to(root)), **info,
                     "delta_bottom_u_px": du, "delta_bottom_v_px": dv})

    out = root / "relabel_from_masks_report.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def col(key):
        return sorted(r[key] for r in rows if r[key] == r[key])

    excluded = [r for r in rows if r["excluded_from_training"]]
    minor = [r for r in rows if r["n_components"] > 1 and not r["excluded_from_training"]]
    over = col("hull_over_claim")
    print(f"{len(rows)} masks, {changed} with more than one significant component")
    print(f"  minor split, largest >= 90% -> labelled from the largest piece: {len(minor)}")
    print(f"  real split, largest < 90%  -> EXCLUDED from training: {len(excluded)}")
    if over:
        print(f"  polygon area / mask area: median {over[len(over)//2]:.4f}  "
              f"max {max(over):.4f}   (>1 would mean over-claiming; the graded "
              f"policy cannot)")
    if excluded:
        ex = out.parent / "split_silhouette_exclusions.csv"
        with ex.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(excluded[0]))
            writer.writeheader()
            writer.writerows(excluded)
        print(f"  exclusion list -> {ex}")
    print(f"  {'WROTE labels' if args.apply else 'dry run, wrote no labels'} -> report {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
