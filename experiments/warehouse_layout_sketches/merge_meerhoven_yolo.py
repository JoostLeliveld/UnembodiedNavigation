#!/usr/bin/env python3
"""Adapt the audited multi-camera merger to Meerhoven's A--L camera contract.

The merger's validation/copy implementation remains single-sourced in the established
four-camera commissioning tool. This adapter changes only its process-local frozen
camera/world contract before invoking it; source captures and output stay immutable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
MERGER_PATH = (
    REPO
    / "experiments/multicamera_commissioning_bigwarehouse/tools/merge_fourcam_yolo_dataset.py"
)


def _load_merger():
    spec = importlib.util.spec_from_file_location("audited_multicam_yolo_merger", MERGER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load merger: {MERGER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def camera_contracts() -> dict[str, dict[str, str]]:
    contracts = {}
    for letter in "ABCDEFGHIJKL":
        suffix = "" if letter == "A" else f"_{letter.lower()}"
        model = f"external_camera{suffix}"
        contracts[f"camera_{letter}"] = {
            "camera_model": model,
            "image_topic": f"/{model}/image_raw",
            "labels_topic": f"/{model}/segmentation/labels_map",
        }
    return contracts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=REPO / "logs/perception_datasets/warehouse_meerhoven_yolo_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "logs/perception_datasets/warehouse_meerhoven_yolo_12cam_v1",
    )
    args = parser.parse_args()

    merger = _load_merger()
    contracts = camera_contracts()
    merger.CAMERA_CONTRACTS = contracts
    merger.CAMERAS = tuple(contracts)
    config = merger.MergeConfig(
        expected_world="warehouse_meerhoven.world.sdf",
        range_edges_m=(5.0, 8.0, 12.0, 16.0),
        min_train_per_camera=40,
        min_val_per_camera=8,
        # The short/steep E/F cameras deliberately do not span every global
        # range bin. Per-range detector gates are evaluated where each camera
        # actually has opportunities, not fabricated by this merge gate.
        min_train_per_core_range_bin=0,
        min_val_per_core_range_bin=0,
        max_cross_input_duplicate_hashes=0,
        # One immutable empty-scene frame per fixed view. Multiple off-screen
        # robot poses render the same static background and are rejected by the
        # source duplicate guard; pooling A--L yields 12 distinct backgrounds.
        min_negative_train_per_camera=1,
    )
    camera_dirs = {
        camera_id: args.capture_root / camera_id for camera_id in contracts
    }
    result = merger.merge_fourcam_datasets(camera_dirs, args.output_dir, config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
