#!/usr/bin/env python3
"""Build the C4 field: GP residual fitted on a MONOCULAR-DEPTH prior mean.

The operational GP (C2) uses a geometric day-zero prior derived from the surveyed
model, so it inherits a survey dependency. This arm keeps the GP's ability to learn
from detector outcomes but replaces its prior mean with the monocular-depth field,
which needs only the camera's own RGB, calibration and the 2-D drivable map. If it
works it is the strongest deployable field available: geometry from the image,
correction from experience, no survey anywhere.

Separate process for the same reason as gp_refit.py: `fit_belief_aware_gp` does
`from common import ...` and this package also has a `common`.

Fitted on ALL spawn-grid events per camera (deployment case, commission once on
everything available), matching build_mono_depth_planner_artifact.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts/visibility_comparison"))
sys.path.insert(0, str(REPO / "scripts/shared"))
sys.path.insert(0, str(REPO / "src/reliability"))

import fit_belief_aware_gp as F  # noqa: E402
from reliability.observation_planner_artifact import FieldGrid, write_planner_artifact  # noqa: E402

EVENT_ROOT = REPO / "logs/studies/multicamera_commissioning_bigwarehouse/spawn_grid_20260727/events"
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
CAMERA_POSES = {
    "camera_A": (-6.0, -10.0, 6.10), "camera_B": (-6.0, 10.0, 6.10),
    "camera_C": (6.0, -10.0, 6.10),  "camera_D": (6.0, 10.0, 6.10),
}
HP = dict(aggregate_resolution_m=0.3, max_bin_weight=20.0, gp_length_scale=1.2,
          gp_noise_var=0.05, pose_length_scale=0.35, min_certainty=0.05,
          spread_scale=1.0, min_prob=1e-4, beta=0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior-root", required=True, help="mono-depth planner artifacts (per-camera)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fit_args = argparse.Namespace(**HP)
    out_root = Path(args.out)

    per = {}
    for camera in CAMERAS:
        data = F._load_events(EVENT_ROOT / f"{camera}_events.csv", target="hit", min_prob=HP["min_prob"])
        prior_path = Path(args.prior_root) / "gp" / camera / "det_hit_expected_kernel_gp.npz"
        prior = F._load_prior_map(prior_path, map_key="P_mean_map", min_prob=HP["min_prob"])
        if prior is None:
            raise SystemExit(f"could not load mono-depth prior {prior_path}")

        agg = F._aggregate_events(data, resolution_m=HP["aggregate_resolution_m"],
                                  max_bin_weight=HP["max_bin_weight"])
        grid_npz = np.load(prior_path)
        xs = np.asarray(grid_npz["xs"], float); ys = np.asarray(grid_npz["ys"], float)
        XX, YY = np.meshgrid(xs, ys)
        query = np.column_stack([XX.ravel(), YY.ravel()])

        class _Q:  # _predict_mode_at_events wants an EventData-like with X and cov
            pass
        q = _Q(); q.X = query; q.cov = np.tile(np.eye(2) * 1e-6, (len(query), 1, 1))
        p = F._predict_mode_at_events(agg, q, mode="expected_kernel", prior=prior, args=fit_args)
        field = np.clip(np.asarray(p, float).reshape(len(ys), len(xs)), 1e-4, 1 - 1e-4)
        per[camera] = field

        write_planner_artifact(
            str(out_root / "gp" / camera / "det_hit_expected_kernel_gp.npz"),
            FieldGrid(xs=xs, ys=ys), field,
            camera_pos=CAMERA_POSES[camera], source="mono_depth_prior_plus_gp",
            provenance={"prior": "monocular depth", "mode": "expected_kernel",
                        "needs_surveyed_model": False, "n_events": int(data.X.shape[0])},
        )
        print(f"  {camera}: field mean {field.mean():.4f} min {field.min():.4f}")

    fused = 1 - np.prod([1 - per[c] for c in CAMERAS], axis=0)
    fused_path = out_root / "fused_planner_four_camera.npz"
    np.savez(fused_path, xs=xs, ys=ys, P_mean_map=fused, P_conservative_plan_map=fused,
             F_mean_map=np.zeros_like(fused), F_std_map=np.full_like(fused, 0.05),
             P_union_4cam_map=fused,
             **{f"P_camera_{c[-1]}_map": per[c] for c in CAMERAS},
             coverage_count=np.sum([per[c] > 0.5 for c in CAMERAS], axis=0).astype(float),
             camera_ids=np.asarray(list(CAMERAS)), target_height=np.asarray([0.0]))
    print(f"\nwrote {out_root}\n  fused mean {fused.mean():.4f}  min {fused.min():.4f}")


if __name__ == "__main__":
    main()
