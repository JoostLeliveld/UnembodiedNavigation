#!/usr/bin/env python3
"""Build a FUSED-4-camera planner artifact from the day-zero prior.

The single-camera paper's planner consumes `P_conservative_plan_map` as its
reliability field -> R_plan. The day-zero artifact
(`camera_a_planner_with_four_camera_maps.npz`) sets that field to camera_A ONLY
(the masquerade: the planner ignores B/C/D). This script produces a sibling
artifact whose `P_conservative_plan_map` IS the fused 4-camera field, so the
planner plans on multi-camera coverage.

Fusion choice:
  --fusion best  -> P_best_4cam_map  (max over cameras = best single camera at
                    each point; matches a selection/handover runtime). DEFAULT.
  --fusion union -> P_union_4cam_map (noisy-OR = P(>=1 camera detects); matches
                    a full-fusion runtime that ingests all cameras).

This preserves every other key (per-camera maps, coverage, xs/ys). It does NOT
invent learned reliability: the underlying fields are the day-zero GEOMETRY
prior. Once per-camera GPs are refit on the M1 detector, re-run against those.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "paper_artifacts" / "gp" / "warehouse_full_4cam_dayzero_v1" / "camera_a_planner_with_four_camera_maps.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--fusion", choices=("best", "union"), default="best")
    ap.add_argument("--out", default=str(REPO / "paper_artifacts" / "gp" / "warehouse_full_4cam_fused_v1" / "fused_planner_four_camera.npz"))
    args = ap.parse_args()

    d = dict(np.load(args.src, allow_pickle=True))
    fused_key = "P_best_4cam_map" if args.fusion == "best" else "P_union_4cam_map"
    fused = np.asarray(d[fused_key], dtype=float)
    cam_a = np.asarray(d["P_camera_A_map"], dtype=float)

    # Repoint the planner-facing fields to the fused 4-camera field.
    d["P_conservative_plan_map"] = fused
    d["P_mean_map"] = fused

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)

    manifest = {
        "artifact": str(out.relative_to(REPO)),
        "source": str(Path(args.src).relative_to(REPO)),
        "planner_map": "P_conservative_plan_map",
        "planner_map_semantics": f"FUSED 4-camera {args.fusion} field ({fused_key}); multi-camera coverage, NOT camera_A-only",
        "fusion": args.fusion,
        "data_source": "day-zero geometry prior (replace with learned per-camera GP fusion after M1 recapture)",
        "planner_mean_camera_A_prev": round(float(cam_a.mean()), 4),
        "planner_mean_fused_now": round(float(fused.mean()), 4),
        "frac_reliable_camera_A_prev": round(float((cam_a > 0.5).mean()), 4),
        "frac_reliable_fused_now": round(float((fused > 0.5).mean()), 4),
    }
    (out.parent / "fused_planner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out.relative_to(REPO)}  (fusion={args.fusion})")
    print(f"planner reliability mean: camera_A {cam_a.mean():.3f} -> fused {fused.mean():.3f}")
    print(f"floor fraction reliable(>0.5): camera_A {(cam_a>0.5).mean():.2f} -> fused {(fused>0.5).mean():.2f}")
    print(f"planner map changed from camera_A: {not np.allclose(cam_a, fused)}")


if __name__ == "__main__":
    main()
