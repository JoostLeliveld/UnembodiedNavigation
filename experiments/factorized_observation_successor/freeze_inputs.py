#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

import common as C


def main() -> None:
    required = [C.P_SOURCE, C.R_ROWS, C.R_INDEX, C.R_REGISTRY, C.WORLD, C.ROUTES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"refusing to freeze with missing inputs: {missing}")
    inputs = {str(path.relative_to(C.REPO)): C.sha256(path) for path in required}
    for camera in C.CAMERAS:
        path = C.REPO / f"logs/studies/availability_paper/depth_gp_planner_v1/gp/{camera}/det_hit_expected_kernel_gp.npz"
        inputs[str(path.relative_to(C.REPO))] = C.sha256(path)
    frozen_p = C.OUT / "frozen/p_use_depth_gp_four_camera.npz"
    frozen_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(C.P_SOURCE, frozen_p)
    manifest = {
        "study": "factorized_observation_successor",
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "p_use": {
            "source": str(C.P_SOURCE.relative_to(C.REPO)),
            "frozen_copy": str(frozen_p.relative_to(C.REPO)),
            "sha256": C.sha256(frozen_p),
            "operational_inputs": ["camera_rgb", "camera_calibration", "drivable_map", "detector_outcomes"],
            "surveyed_3d_model_required": False,
            "ground_truth_used": False,
        },
        "r_cond_commissioning_source": "PG-IPM-CURRENT",
        "development_cameras": list(C.DEV_CAMERAS),
        "heldout_configuration_cameras": list(C.HOLDOUT_CAMERAS),
        "heldout_scope_caveat": "configuration-level holdout, not new-data holdout",
        "tasks": list(C.TASKS),
        "path_length_budget_ratio": C.LENGTH_BUDGET_RATIO,
        "inputs_sha256": inputs,
    }
    C.write_json(C.OUT / "frozen/manifest.json", manifest)
    print(C.OUT / "frozen/manifest.json")


if __name__ == "__main__":
    main()

