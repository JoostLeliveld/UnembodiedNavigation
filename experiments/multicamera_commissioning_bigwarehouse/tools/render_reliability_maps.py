#!/usr/bin/env python3
"""M1 (first pass) — per-camera reliability maps + overlap map from the audit grid.

Uses the teleport-grid detection audit (v2_diag_640/detections.csv: x,y,camera,
detected,score over the drivable floor) as a COARSE commissioning reliability
survey with the retrained detector. Renders, per camera, where the robot is
detected (the camera's operational footprint), plus the multi-camera overlap map
(how many cameras see each spot) that fusion/handover depends on.

Coarse (55 poses); a dense commissioning coverage drive refines it. Real data,
retrained detector. No GT-as-input (positions are commanded teleport poses).
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
CSV = REPO / "logs/studies/fourcam_detector_audit/v2_diag_640/detections.csv"
OUT = REPO / "logs/studies/fourcam_detector_audit/reliability_maps_v2diag.png"
CAMS = ["camera_A", "camera_B", "camera_C", "camera_D"]
CAM_XY = {"camera_A": (-6, -10), "camera_B": (-6, 10), "camera_C": (6, -10), "camera_D": (6, 10)}

rows = list(csv.DictReader(open(CSV)))
by_cam = defaultdict(list)
for r in rows:
    by_cam[r["camera"]].append((float(r["x"]), float(r["y"]), int(r["detected"]), float(r["score"])))
poses = sorted({(float(r["x"]), float(r["y"])) for r in rows})
ncam_at = {p: 0 for p in poses}
for r in rows:
    if int(r["detected"]):
        ncam_at[(float(r["x"]), float(r["y"]))] += 1

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, cam in zip(axes.ravel()[:4], CAMS):
    pts = by_cam[cam]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    det = [p[2] for p in pts]
    ax.scatter([x for x, d in zip(xs, det) if not d], [y for y, d in zip(ys, det) if not d],
               c="#d0d0d0", s=45, marker="x", label="miss")
    sc = ax.scatter([x for x, d in zip(xs, det) if d], [y for y, d in zip(ys, det) if d],
                    c=[p[3] for p in pts if p[2]], cmap="viridis", vmin=0, vmax=1, s=60, label="detect")
    cx, cy = CAM_XY[cam]
    ax.plot(cx, cy, "r*", ms=18); ax.text(cx, cy, f" {cam[-1]}", color="r", fontweight="bold")
    rate = np.mean(det) if det else 0
    ax.set_title(f"{cam}  detect-rate {rate:.2f} (n={len(pts)})", fontsize=11)
    ax.set_xlim(-12, 12); ax.set_ylim(-9, 9); ax.set_aspect("equal"); ax.grid(alpha=0.2)
fig.colorbar(sc, ax=list(axes.ravel()[:4]), shrink=0.55, label="detector score")

# overlap map
ax_ov = axes[1, 1]
xs = [p[0] for p in poses]; ys = [p[1] for p in poses]; nc = [ncam_at[p] for p in poses]
s = ax_ov.scatter(xs, ys, c=nc, cmap="RdYlGn", vmin=0, vmax=3, s=70)
for c, (cx, cy) in CAM_XY.items():
    ax_ov.plot(cx, cy, "k*", ms=12)
ax_ov.set_title("overlap: # cameras detecting robot", fontsize=11)
ax_ov.set_xlim(-12, 12); ax_ov.set_ylim(-9, 9); ax_ov.set_aspect("equal"); ax_ov.grid(alpha=0.2)
fig.colorbar(s, ax=ax_ov, shrink=0.7, label="# cameras")

# summary panel
axs = axes[1, 2]; axs.axis("off")
cov1 = sum(1 for p in poses if ncam_at[p] >= 1); cov2 = sum(1 for p in poses if ncam_at[p] >= 2)
lines = [f"poses: {len(poses)}", f"seen by >=1 cam: {cov1} ({cov1/len(poses):.0%})",
         f"seen by >=2 cam: {cov2} ({cov2/len(poses):.0%})", "",
         "per-camera detect-rate (footprint):"]
for cam in CAMS:
    d = [p[2] for p in by_cam[cam]]
    lines.append(f"  {cam}: {np.mean(d):.2f}")
axs.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)
fig.suptitle("M1 commissioning (coarse, retrained detector v2_640_diag): per-camera reliability + overlap",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
print(f"coverage >=1cam {cov1}/{len(poses)}  >=2cam {cov2}/{len(poses)}")
for cam in CAMS:
    d = [p[2] for p in by_cam[cam]]
    print(f"  {cam}: footprint detect-rate {np.mean(d):.3f} (n={len(d)})")
