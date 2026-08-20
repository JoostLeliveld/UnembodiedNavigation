#!/usr/bin/env python3
"""Ablate online ground anchoring on the frozen four-camera depth frames.

This is an evaluator, not part of the method: it runs the RGB-derived depth
pipeline first and opens Gazebo depth/visibility only afterwards for scoring.
The dataset has two timestamps per camera, so it is a mechanism check for the
temporal filter, not evidence of long-run temporal accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import four_camera_study as study

ga = study.ga


MODES = ("legacy", "enhanced", "enhanced_temporal")
DEFAULT_MODELS = ("dav2_relative_small", "unidepth_v2_vits14")


def _configuration(mode: str) -> ga.MethodConfig:
    return ga.MethodConfig(
        anchors=ga.AnchorConfig(quality_filter=mode != "legacy"),
        target=ga.TargetVolume(
            radius_m=0.0,
            z_min_m=0.35,
            z_max_m=0.35,
            n_heights=1,
            n_ring=0,
        ),
    )


def _score_frame(result, record: dict, calib) -> dict:
    """Evaluation-only reads happen in this function and nowhere above it."""
    oracle_grid = np.load(study.DATASET / record["oracle_visibility_grid"]["path"])
    visibility = study._visibility_metrics(result.visibility.p_visible, oracle_grid)

    truth = np.load(study.DATASET / record["oracle_depth_path"])
    floor_depth = study._full_floor_depth(calib)
    structure = (
        np.isfinite(truth)
        & (truth > 0)
        & (~np.isfinite(floor_depth) | (truth < floor_depth - 0.10))
    )
    depth = study._depth_errors(
        result.metric_depth.depth_m,
        truth,
        structure & result.metric_depth.valid,
    )
    return {"visibility": visibility, "structure_depth": depth}


def evaluate(models: tuple[str, ...]) -> dict:
    records = sorted(
        study._records(), key=lambda row: (float(row["timestamp"]), row["camera_id"])
    )
    manifest = json.loads((study.DATASET / "manifest.json").read_text(encoding="utf-8"))
    xs, ys, _ = study._grid(manifest)
    drivable = study._drivable()
    output = {
        "status": "small_mechanism_check_not_statistical_equivalence",
        "dataset": str(study.DATASET.relative_to(study.REPO)),
        "n_frames": len(records),
        "n_cameras": len(study.CAM_LABELS),
        "timestamps_s": list(study.TIMES),
        "method_inputs": [
            "saved monocular prediction",
            "fixed camera calibration",
            "planner traversable regions",
            "floor plane z=0",
        ],
        "evaluation_only_inputs": [
            "Gazebo optical-axis depth",
            "geometry-raycast visibility grid",
        ],
        "external_floor_segmentation_available": False,
        "runtime_environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy_version": np.__version__,
            "timing_scope": (
                "estimate_visibility only; excludes saved-prediction I/O and monocular "
                "network inference; sequential frames, no warm-up discarded"
            ),
        },
        "mode_definitions": {
            "legacy": "geometric floor candidates plus robust single-frame affine",
            "enhanced": "depth-edge rejection plus rank-based confidence/uncertainty weighting",
            "enhanced_temporal": "enhanced anchors plus per-camera Bayesian affine filter",
        },
        "models": {},
    }

    for model in models:
        model_result = {}
        for mode in MODES:
            config = _configuration(mode)
            temporal = ga.TemporalGroundAnchorFilter() if mode.endswith("temporal") else None
            rows = []
            for record in records:
                pred_path = (
                    study.PRED_DIR
                    / model
                    / f"{study._prediction_id(record)}__{model}.json"
                )
                prediction = ga.load_prediction(pred_path)
                calib = ga.camera_from_record(record)
                started = time.perf_counter()
                result = ga.estimate_visibility(
                    prediction,
                    calib,
                    xs,
                    ys,
                    drivable=drivable,
                    config=config,
                    temporal_filter=temporal,
                    timestamp=float(record["timestamp"]),
                )
                elapsed = time.perf_counter() - started
                row = {
                    "camera_id": record["camera_id"],
                    "timestamp_s": float(record["timestamp"]),
                    "status": result.status.value,
                    "method_seconds": elapsed,
                    "n_anchor": result.ground_fit.n_anchor,
                    "fit": result.ground_fit.to_dict(),
                    "anchor_stage_counts": result.provenance["anchor_stage_counts"],
                    "temporal": result.provenance.get("temporal_anchor"),
                    "score": None,
                }
                if result.status.is_ok:
                    row["score"] = _score_frame(result, record, calib)
                rows.append(row)

            valid = [row for row in rows if row["score"] is not None]
            cycles = len(records) / max(1, len(study.CAM_LABELS))
            model_result[mode] = {
                "n_valid": len(valid),
                "n_refused": len(rows) - len(valid),
                "median_structure_depth_mae_m": (
                    float(np.median([
                        row["score"]["structure_depth"]["mae_m"] for row in valid
                        if row["score"]["structure_depth"].get("n", 0)
                    ])) if valid else None
                ),
                "median_visibility_balanced_accuracy": (
                    float(np.median([
                        row["score"]["visibility"]["balanced_accuracy"] for row in valid
                    ])) if valid else None
                ),
                "median_visible_iou": (
                    float(np.median([
                        row["score"]["visibility"]["visible_iou"] for row in valid
                    ])) if valid else None
                ),
                "median_frame_method_seconds": float(
                    np.median([row["method_seconds"] for row in rows])
                ),
                "mean_four_camera_method_seconds": float(
                    sum(row["method_seconds"] for row in rows) / cycles
                ),
                "median_anchor_count": float(np.median([row["n_anchor"] for row in rows])),
                "frames": rows,
            }
        output["models"][model] = model_result
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--out",
        type=Path,
        default=study.OUT / "online_anchor_ablation.json",
    )
    args = parser.parse_args()
    result = evaluate(tuple(args.models))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        model: {
            mode: {key: value for key, value in metrics.items() if key != "frames"}
            for mode, metrics in modes.items()
        }
        for model, modes in result["models"].items()
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
