#!/usr/bin/env python3
"""Build planner artifacts from the MONOCULAR-DEPTH availability field.

WHY. The E4 campaign's availability-aware arms consume
`spawn_grid_20260727/gp/{camera_id}/det_hit_expected_kernel_gp.npz`, which is the
GP fitted on a *geometric day-zero prior* — an arm E1 flags `needs_surveyed_model`.
So the closed loop, as first configured, tests availability-awareness but NOT the
paper's headline deployable field. This script produces the same artifacts from the
monocular-depth field so a fourth arm can close that gap end to end.

WHAT IS AND IS NOT DEPLOYABLE HERE. The depth field itself needs only the camera's
own RGB, its calibration, and the 2-D drivable map. Turning a visibility fraction
into a detection probability needs a calibration link, and that link is fitted on
commissioning detections — exactly as every arm in E1 was. No surveyed 3-D obstacle
model is used at any point.

The link is fitted on ALL spawn-grid events per camera, which is the deployment
case (commission once on everything you have). E1's per-fold links exist to measure
held-out prediction; they are not what a deployed system would ship.

Outputs mirror `fused_planner_four_camera.npz`, whose `P_conservative_plan_map` was
verified to equal the noisy-OR of its per-camera maps to 0.0000.

Run:
    python3 experiments/availability_paper/build_mono_depth_planner_artifact.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "src/reliability"))
from reliability.observation_planner_artifact import FieldGrid, write_planner_artifact  # noqa: E402

OUT_ROOT = C.OUT_ROOT / "mono_depth_planner_v1"
MIN_PROB = 1e-4


def noisy_or(fields: list[np.ndarray]) -> np.ndarray:
    miss = np.ones_like(fields[0], dtype=float)
    for f in fields:
        miss = miss * (1.0 - np.clip(f, 0.0, 1.0))
    return 1.0 - miss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--model", default=None, help="monocular model (default: study choice)")
    args = ap.parse_args()
    out_root = Path(args.out)

    if args.model:
        C.set_mono_depth_model(args.model)
    apparatus = C.build_apparatus()
    xs, ys = apparatus.xs, apparatus.ys
    grid = FieldGrid(xs=xs, ys=ys)

    calibrated: dict[str, np.ndarray] = {}
    links: dict[str, dict] = {}
    for camera in C.CAMERAS:
        raw = apparatus.field("mono_depth", camera)
        ev = apparatus.events[camera]
        scores = C.sample_field_at(raw, xs, ys, ev["xy"])
        a, b = C.fit_link(scores, ev["hit"])
        field = np.clip(C.apply_link(raw, a, b), MIN_PROB, 1.0 - MIN_PROB)
        calibrated[camera] = field
        links[camera] = {"a": a, "b": b, "n_events": int(ev["hit"].size),
                         "raw_mean": float(raw.mean()), "calibrated_mean": float(field.mean())}

        pose = C.base.CAMERA_POSES[camera]
        write_planner_artifact(
            str(out_root / "gp" / camera / "det_hit_expected_kernel_gp.npz"),
            grid, field,
            camera_pos=(float(pose[0]), float(pose[1]), float(pose[2])),
            source="mono_depth_linked",
            provenance={
                "monocular_model": C.MONO_DEPTH_MODEL,
                "stamp": C.MONO_DEPTH_STAMP,
                "link": "P_D = sigmoid(a*logit(v)+b) fitted on all spawn-grid events",
                "link_a": a, "link_b": b,
                "needs_surveyed_model": False,
            },
        )

    fused = noisy_or([calibrated[c] for c in C.CAMERAS])
    fused_path = out_root / "fused_planner_four_camera.npz"
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fused_path,
        xs=xs, ys=ys,
        P_mean_map=fused,
        P_conservative_plan_map=fused,
        F_mean_map=np.zeros_like(fused),
        F_std_map=np.full_like(fused, 0.05),
        P_union_4cam_map=fused,
        **{f"P_camera_{c[-1]}_map": calibrated[c] for c in C.CAMERAS},
        coverage_count=np.sum([calibrated[c] > 0.5 for c in C.CAMERAS], axis=0).astype(float),
        camera_ids=np.asarray(list(C.CAMERAS)),
        target_height=np.asarray([0.0], dtype=float),
    )

    # Compare against the GP field the campaign currently uses, so the fourth arm's
    # expected behaviour is known BEFORE any machine time is spent on it.
    gp_fused = np.load(C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz")
    gp_field = np.asarray(gp_fused["P_conservative_plan_map"], dtype=float)
    drive = apparatus.driveable
    corr = float(np.corrcoef(fused[drive], gp_field[drive])[0, 1])

    C.write_json(out_root / "manifest.json", {
        "monocular_model": C.MONO_DEPTH_MODEL,
        "links": links,
        "fused": {"mean": float(fused.mean()), "min": float(fused.min()), "max": float(fused.max())},
        "gp_fused_mean": float(gp_field.mean()),
        "correlation_with_gp_field_on_driveable": corr,
        "needs_surveyed_model": False,
        "artifact_sha256": hashlib.sha256(fused_path.read_bytes()).hexdigest()[:16],
    })

    print(f"wrote {out_root}")
    for c in C.CAMERAS:
        L = links[c]
        print(f"  {c}: link a={L['a']:+.3f} b={L['b']:+.3f}  raw mean {L['raw_mean']:.3f} -> {L['calibrated_mean']:.3f}")
    print(f"\n  fused mono-depth mean {fused.mean():.4f} vs GP field {gp_field.mean():.4f}")
    print(f"  correlation on driveable cells: {corr:.3f}")


if __name__ == "__main__":
    main()
