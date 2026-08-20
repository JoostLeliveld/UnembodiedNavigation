#!/usr/bin/env python3
"""Infer and evaluate the 21-update four-camera temporal anchoring sequence.

Inference is RGB-only. Oracle arrays are opened only in the scoring functions after
``estimate_visibility`` has returned. See ``temporal_anchor_sequence/PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ADAPTER_DIR = REPO / "experiments" / "monocular_depth_adapter"
VISIBILITY_DIR = REPO / "experiments" / "mono_depth_visibility"
for path in (ADAPTER_DIR, VISIBILITY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ground_anchoring as ga  # noqa: E402
from monodepth import (  # noqa: E402
    CameraIntrinsics,
    DepthRequest,
    MonocularDepthAdapter,
    storage,
)

DATASET = REPO / "logs/studies/dynamic_world_oracle/s01_box_in_aisle/run01"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
OUT = HERE / "temporal_anchor_sequence"
PREDICTIONS = OUT / "predictions/dav2_relative_small"
MODEL = "dav2_relative_small"
CAMERAS = (
    "external_camera",
    "external_camera_b",
    "external_camera_c",
    "external_camera_d",
)
SOURCE_TIMES = (
    0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4,
    4.8, 5.0, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0,
)
DROPOUT_TIMES = (2.4, 4.4, 6.0, 8.0)
UPDATE_PERIOD_S = 10.0
EXPECTED_HASHES = {
    "records.jsonl": "eb1ec37bee880e3d41928f2077c55978652123fe845288bcc70e2929ae242205",
    "manifest.json": "5726de71428f4352210d3ff6f439033e55037f52ad536135c0b716695780e5cd",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _records() -> list[dict]:
    for name, expected in EXPECTED_HASHES.items():
        actual = _sha256(DATASET / name)
        if actual != expected:
            raise RuntimeError(f"{name} hash {actual} != frozen {expected}")
    rows = [
        json.loads(line)
        for line in (DATASET / "records.jsonl").read_text().splitlines()
        if line
    ]
    keys = [(row["camera_id"], float(row["timestamp"])) for row in rows]
    expected_keys = [(camera, timestamp) for timestamp in SOURCE_TIMES for camera in CAMERAS]
    if sorted(keys) != sorted(expected_keys) or len(keys) != len(set(keys)):
        raise RuntimeError("records do not exactly match the frozen 21 x 4 sequence")
    return sorted(rows, key=lambda row: (float(row["timestamp"]), row["camera_id"]))


def _prediction_id(record: dict) -> str:
    stamp_ms = int(round(1000.0 * float(record["timestamp"])))
    return f"s01_t{stamp_ms:07d}ms_{record['camera_id']}"


def _intrinsics(record: dict) -> CameraIntrinsics:
    block = record["camera_intrinsics"]
    return CameraIntrinsics(
        fx=float(block["fx"]),
        fy=float(block["fy"]),
        cx=float(block["cx"]),
        cy=float(block["cy"]),
        width=int(block["img_width"]),
        height=int(block["img_height"]),
    )


def _prediction_path(record: dict) -> Path:
    return PREDICTIONS / f"{_prediction_id(record)}__{MODEL}.json"


def run_inference(device: str) -> dict:
    """RGB-only adapter pass. This function has no path to an oracle array."""
    records = _records()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    missing = [record for record in records if not _prediction_path(record).is_file()]
    started = time.time()

    if missing:
        print(f"depth inference: {len(missing)} missing of {len(records)} frames on {device}")
        with MonocularDepthAdapter(
            MODEL, device=device, batch_size=1, uncertainty="none"
        ) as adapter:
            for index, record in enumerate(missing, 1):
                visible = ga.method_visible_record(record)
                rgb_path = DATASET / visible["rgb_path"]
                rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
                request = DepthRequest(
                    _prediction_id(record),
                    rgb,
                    _intrinsics(record),
                    source_path=str(rgb_path.relative_to(REPO)),
                )
                prediction = adapter.predict([request])[0]
                storage.save_prediction(prediction, PREDICTIONS)
                if index == 1 or index % 8 == 0 or index == len(missing):
                    print(f"  inferred {index:3d}/{len(missing)}", flush=True)
    else:
        print("depth inference: all 84 predictions already present")

    predictions = [storage.load_prediction(_prediction_path(record)) for record in records]
    if len(predictions) != 84 or any(pred.model.model_name != MODEL for pred in predictions):
        raise RuntimeError("prediction set is incomplete or contains the wrong model")
    forwards = [float(pred.timing.forward_s) for pred in predictions]
    first = predictions[0]
    try:
        import torch

        environment = {
            "torch": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        }
    except Exception as exc:  # pragma: no cover - torch is an adapter dependency
        environment = {"torch_environment_error": f"{type(exc).__name__}: {exc}"}
    manifest = {
        "status": "complete",
        "model": first.model.as_dict(),
        "n_predictions": len(predictions),
        "frame_ids": [pred.image_id for pred in predictions],
        "source_dataset": str(DATASET.relative_to(REPO)),
        "source_hashes": EXPECTED_HASHES,
        "config": {"device_requested": device, "batch_size": 1, "uncertainty": "none"},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **environment,
        },
        "forward_seconds": {
            "median_per_frame": float(np.median(forwards)),
            "mean_per_frame": float(np.mean(forwards)),
            "sum_recorded": float(np.sum(forwards)),
        },
        "invocation_wall_seconds": time.time() - started,
        "prediction_files_sha256": {
            str(path.relative_to(OUT)): _sha256(path)
            for path in sorted(PREDICTIONS.glob("*"))
            if path.suffix in {".json", ".npz"}
        },
    }
    (OUT / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _drivable() -> list[ga.Footprint]:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    regions = profile["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    return [
        ga.Footprint(
            xmin=float(region["xmin"]),
            xmax=float(region["xmax"]),
            ymin=float(region["ymin"]),
            ymax=float(region["ymax"]),
            name=str(region["name"]),
        )
        for region in regions
        if region.get("type") == "traversable"
    ]


def _grid() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    resolution = float(grid["resolution_m"])
    xs = float(grid["xmin"]) + (np.arange(int(grid["nx"])) + 0.5) * resolution
    ys = float(grid["ymin"]) + (np.arange(int(grid["ny"])) + 0.5) * resolution
    return xs, ys


def _visibility_metrics(probability: np.ndarray, oracle: np.ndarray) -> dict:
    mask = (oracle == 0) | (oracle == 1)
    truth = oracle[mask] == 1
    guess = probability[mask] >= 0.5
    tp = int(np.count_nonzero(guess & truth))
    tn = int(np.count_nonzero(~guess & ~truth))
    fp = int(np.count_nonzero(guess & ~truth))
    fn = int(np.count_nonzero(~guess & truth))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n_cells": int(mask.sum()),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "visible_iou": float(tp / max(1, tp + fp + fn)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _full_floor_depth(calib: ga.CameraCalibration) -> np.ndarray:
    u, v = calib.pixel_grid(step=1)
    depth, _ = ga.analytic_plane_depth(calib, ga.FloorPlane(), u, v)
    return depth.reshape(calib.height, calib.width)


def _structure_depth_metrics(
    prediction: np.ndarray, valid: np.ndarray, truth: np.ndarray, floor: np.ndarray
) -> dict:
    structure = (
        valid
        & np.isfinite(prediction)
        & np.isfinite(truth)
        & (truth > 0.0)
        & (~np.isfinite(floor) | (truth < floor - 0.10))
    )
    error = prediction[structure] - truth[structure]
    return {
        "n_pixels": int(error.size),
        "mae_m": float(np.mean(np.abs(error))) if error.size else float("nan"),
        "bias_m": float(np.mean(error)) if error.size else float("nan"),
        "rmse_m": (
            float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan")
        ),
    }


def _phase(source_time: float) -> str:
    if source_time < 1.0:
        return "clear_before_spawn"
    if source_time < 3.0:
        return "pallet_stationary_south"
    if source_time < 5.0:
        return "pallet_moving"
    if source_time < 7.0:
        return "pallet_stationary_north"
    return "clear_after_removal"


def _score(result, record: dict, calib: ga.CameraCalibration) -> dict:
    """Evaluation-only oracle access, called only after the method result exists."""
    oracle_visibility = np.load(DATASET / record["oracle_visibility_grid"]["path"])
    oracle_depth = np.load(DATASET / record["oracle_depth_path"])
    return {
        "visibility": _visibility_metrics(result.visibility.p_visible, oracle_visibility),
        "structure_depth": _structure_depth_metrics(
            result.metric_depth.depth_m,
            result.metric_depth.valid,
            oracle_depth,
            _full_floor_depth(calib),
        ),
    }


def _run_arm(name: str, *, temporal: bool, inject_dropout: bool) -> list[dict]:
    records = _records()
    xs, ys = _grid()
    drivable = _drivable()
    config = ga.MethodConfig(
        anchors=ga.AnchorConfig(quality_filter=True),
        target=ga.TargetVolume(
            radius_m=0.0, z_min_m=0.35, z_max_m=0.35, n_heights=1, n_ring=0
        ),
    )
    temporal_filter = ga.TemporalGroundAnchorFilter() if temporal else None
    rows = []
    index_by_time = {timestamp: index for index, timestamp in enumerate(SOURCE_TIMES)}
    for record in records:
        source_time = float(record["timestamp"])
        update_index = index_by_time[source_time]
        update_time = UPDATE_PERIOD_S * update_index
        dropout = bool(inject_dropout and source_time in DROPOUT_TIMES)
        prediction = ga.load_prediction(_prediction_path(record))
        calib = ga.camera_from_record(record)
        segmentation = (
            np.zeros(prediction.shape, dtype=bool) if dropout else None
        )
        started = time.perf_counter()
        result = ga.estimate_visibility(
            prediction,
            calib,
            xs,
            ys,
            drivable=drivable,
            config=config,
            floor_segmentation=segmentation,
            temporal_filter=temporal_filter,
            timestamp=update_time,
            scenario_id=str(record["scenario_id"]),
            frame_id=_prediction_id(record),
        )
        elapsed = time.perf_counter() - started
        row = {
            "arm": name,
            "source_timestamp_s": source_time,
            "operational_update_index": update_index + 1,
            "operational_timestamp_s": update_time,
            "phase": _phase(source_time),
            "camera_id": record["camera_id"],
            "dropout_injected": dropout,
            "status": result.status.value,
            "method_seconds": elapsed,
            "scale": result.ground_fit.scale,
            "shift": result.ground_fit.shift,
            "parameter_covariance": result.ground_fit.parameter_covariance.tolist(),
            "n_anchor": result.ground_fit.n_anchor,
            "inlier_fraction": result.ground_fit.inlier_fraction,
            "residual_rms_m": result.ground_fit.residual_rms_m,
            "anchor_stage_counts": result.provenance["anchor_stage_counts"],
            "temporal": result.provenance.get("temporal_anchor"),
            "score": None,
        }
        if result.status.is_ok:
            row["score"] = _score(result, record, calib)
        rows.append(row)
    return rows


def _cycle_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source_timestamp_s"]].append(row)
    cycles = []
    for source_time in SOURCE_TIMES:
        members = grouped[source_time]
        valid = [row for row in members if row["score"] is not None]
        cycles.append({
            "source_timestamp_s": source_time,
            "operational_update_index": members[0]["operational_update_index"],
            "operational_timestamp_s": members[0]["operational_timestamp_s"],
            "phase": members[0]["phase"],
            "dropout_injected": members[0]["dropout_injected"],
            "n_valid_cameras": len(valid),
            "median_structure_depth_mae_m": (
                float(np.median([
                    row["score"]["structure_depth"]["mae_m"] for row in valid
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
            "four_camera_method_seconds": float(sum(row["method_seconds"] for row in members)),
        })
    return cycles


def _successive_parameter_changes(rows: list[dict]) -> dict:
    changes = {"scale": [], "shift": []}
    by_camera: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["score"] is not None:
            by_camera[row["camera_id"]].append(row)
    for camera_rows in by_camera.values():
        ordered = sorted(camera_rows, key=lambda row: row["operational_update_index"])
        for previous, current in zip(ordered, ordered[1:]):
            changes["scale"].append(abs(float(current["scale"]) - float(previous["scale"])))
            changes["shift"].append(abs(float(current["shift"]) - float(previous["shift"])))
    return {
        "median_abs_successive_scale_change": float(np.median(changes["scale"])),
        "median_abs_successive_shift_change": float(np.median(changes["shift"])),
        "n_successive_pairs": len(changes["scale"]),
    }


def _summarize(rows: list[dict]) -> dict:
    cycles = _cycle_rows(rows)
    valid = [row for row in rows if row["score"] is not None]
    valid_cycles = [cycle for cycle in cycles if cycle["n_valid_cameras"] > 0]
    dropout_rows = [row for row in rows if row["dropout_injected"]]
    valid_dropout = [row for row in dropout_rows if row["score"] is not None]
    return {
        "n_camera_updates": len(rows),
        "n_valid_camera_updates": len(valid),
        "n_refused_camera_updates": len(rows) - len(valid),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "temporal_mode_counts": dict(Counter(
            row["temporal"]["mode"] if row["temporal"] else "disabled" for row in rows
        )),
        "n_injected_dropout_updates": len(dropout_rows),
        "n_valid_injected_dropout_updates": len(valid_dropout),
        "median_cycle_structure_depth_mae_m": float(np.median([
            cycle["median_structure_depth_mae_m"] for cycle in valid_cycles
            if cycle["median_structure_depth_mae_m"] is not None
        ])),
        "median_cycle_visibility_balanced_accuracy": float(np.median([
            cycle["median_visibility_balanced_accuracy"] for cycle in valid_cycles
            if cycle["median_visibility_balanced_accuracy"] is not None
        ])),
        "median_cycle_visible_iou": float(np.median([
            cycle["median_visible_iou"] for cycle in valid_cycles
            if cycle["median_visible_iou"] is not None
        ])),
        "median_four_camera_method_seconds": float(np.median([
            cycle["four_camera_method_seconds"] for cycle in cycles
        ])),
        "parameter_stability": _successive_parameter_changes(rows),
        "cycles": cycles,
        "frames": rows,
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-15:
        return None
    return float(numerator / denominator)


def _load_or_run_arm(
    name: str, *, temporal: bool, inject_dropout: bool
) -> list[dict]:
    cache_dir = OUT / "arm_rows"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name.replace('/', '__')}.json"
    if cache_path.is_file():
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(rows) != 84 or {row["arm"] for row in rows} != {name}:
            raise RuntimeError(f"invalid cached arm rows in {cache_path}")
        print(f"using cached {name}", flush=True)
        return rows
    print(f"evaluating {name}", flush=True)
    rows = _run_arm(name, temporal=temporal, inject_dropout=inject_dropout)
    cache_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def evaluate() -> dict:
    records = _records()
    missing = [
        _prediction_path(record)
        for record in records
        if not _prediction_path(record).is_file()
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} predictions are missing; run with --infer first")
    natural_single = _load_or_run_arm(
        "natural/enhanced_single", temporal=False, inject_dropout=False
    )
    natural_temporal = _load_or_run_arm(
        "natural/enhanced_temporal", temporal=True, inject_dropout=False
    )
    dropout_single = _load_or_run_arm(
        "dropout/enhanced_single", temporal=False, inject_dropout=True
    )
    dropout_temporal = _load_or_run_arm(
        "dropout/enhanced_temporal", temporal=True, inject_dropout=True
    )
    arms = {
        "natural/enhanced_single": _summarize(natural_single),
        "natural/enhanced_temporal": _summarize(natural_temporal),
        "dropout/enhanced_single": _summarize(dropout_single),
        "dropout/enhanced_temporal": _summarize(dropout_temporal),
    }
    natural_single_summary = arms["natural/enhanced_single"]
    natural_temporal_summary = arms["natural/enhanced_temporal"]
    dropout_single_summary = arms["dropout/enhanced_single"]
    dropout_temporal_summary = arms["dropout/enhanced_temporal"]
    natural_temporal_dropout_cycles = [
        cycle for cycle in natural_temporal_summary["cycles"]
        if cycle["source_timestamp_s"] in DROPOUT_TIMES
    ]
    dropout_temporal_dropout_cycles = [
        cycle for cycle in dropout_temporal_summary["cycles"]
        if cycle["source_timestamp_s"] in DROPOUT_TIMES
    ]

    def median_cycle(cycles: list[dict], key: str) -> float:
        return float(np.median([cycle[key] for cycle in cycles]))

    result = {
        "status": "longitudinal_mechanism_evidence",
        "statistical_scope": (
            "descriptive 21-update replay; sequential updates are dependent; not a real-time "
            "lighting or hardware-drift experiment"
        ),
        "source_dataset": str(DATASET.relative_to(REPO)),
        "source_hashes": EXPECTED_HASHES,
        "model": MODEL,
        "n_synchronized_updates": len(SOURCE_TIMES),
        "n_cameras": len(CAMERAS),
        "n_camera_frames": len(SOURCE_TIMES) * len(CAMERAS),
        "source_timestamps_s": list(SOURCE_TIMES),
        "operational_update_period_s": UPDATE_PERIOD_S,
        "operational_horizon_s": UPDATE_PERIOD_S * (len(SOURCE_TIMES) - 1),
        "dropout_source_timestamps_s": list(DROPOUT_TIMES),
        "dropout_selection": "predeclared in PROTOCOL.md before inference/evaluation",
        "method_inputs": [
            "RGB-derived DA-V2 relative depth",
            "fixed camera calibration",
            "planner traversable regions",
            "floor plane z=0",
            "optional predeclared floor segmentation mask",
        ],
        "evaluation_only_inputs": [
            "Gazebo optical-axis depth",
            "geometry-raycast visibility grid",
        ],
        "reporting_unit": "synchronized four-camera update; n=21 dependent updates",
        "runtime_environment": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "timing_scope": (
                "estimate_visibility only; sequential; neural inference and I/O excluded"
            ),
        },
        "arms": arms,
        "comparisons": {
            "natural_temporal_minus_single": {
                "valid_camera_updates": (
                    natural_temporal_summary["n_valid_camera_updates"]
                    - natural_single_summary["n_valid_camera_updates"]
                ),
                "median_cycle_structure_mae_delta_m": (
                    natural_temporal_summary["median_cycle_structure_depth_mae_m"]
                    - natural_single_summary["median_cycle_structure_depth_mae_m"]
                ),
                "median_cycle_balanced_accuracy_delta": (
                    natural_temporal_summary["median_cycle_visibility_balanced_accuracy"]
                    - natural_single_summary["median_cycle_visibility_balanced_accuracy"]
                ),
                "successive_scale_change_ratio": _safe_ratio(
                    natural_temporal_summary["parameter_stability"]
                    ["median_abs_successive_scale_change"],
                    natural_single_summary["parameter_stability"]
                    ["median_abs_successive_scale_change"],
                ),
                "successive_shift_change_ratio": _safe_ratio(
                    natural_temporal_summary["parameter_stability"]
                    ["median_abs_successive_shift_change"],
                    natural_single_summary["parameter_stability"]
                    ["median_abs_successive_shift_change"],
                ),
            },
            "dropout_temporal_minus_single": {
                "valid_camera_updates": (
                    dropout_temporal_summary["n_valid_camera_updates"]
                    - dropout_single_summary["n_valid_camera_updates"]
                ),
                "valid_injected_dropout_updates": (
                    dropout_temporal_summary["n_valid_injected_dropout_updates"]
                    - dropout_single_summary["n_valid_injected_dropout_updates"]
                ),
                "temporal_retention_fraction_at_injected_dropouts": (
                    dropout_temporal_summary["n_valid_injected_dropout_updates"]
                    / dropout_temporal_summary["n_injected_dropout_updates"]
                ),
                "dropout_cycle_structure_mae_delta_vs_natural_temporal_m": (
                    median_cycle(
                        dropout_temporal_dropout_cycles,
                        "median_structure_depth_mae_m",
                    )
                    - median_cycle(
                        natural_temporal_dropout_cycles,
                        "median_structure_depth_mae_m",
                    )
                ),
                "dropout_cycle_balanced_accuracy_delta_vs_natural_temporal": (
                    median_cycle(
                        dropout_temporal_dropout_cycles,
                        "median_visibility_balanced_accuracy",
                    )
                    - median_cycle(
                        natural_temporal_dropout_cycles,
                        "median_visibility_balanced_accuracy",
                    )
                ),
                "dropout_cycle_visible_iou_delta_vs_natural_temporal": (
                    median_cycle(dropout_temporal_dropout_cycles, "median_visible_iou")
                    - median_cycle(natural_temporal_dropout_cycles, "median_visible_iou")
                ),
            },
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figures(result)
    write_results_markdown(result)
    return result


def make_figures(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    single = result["arms"]["natural/enhanced_single"]
    temporal = result["arms"]["natural/enhanced_temporal"]
    x = np.arange(1, len(SOURCE_TIMES) + 1)
    teal, orange, navy, grid = "#18A999", "#E07A5F", "#17324D", "#D8E2EA"

    def cycle_values(arm: dict, key: str) -> np.ndarray:
        return np.asarray([cycle[key] for cycle in arm["cycles"]], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True, constrained_layout=True)
    camera_a_single = [
        row for row in single["frames"] if row["camera_id"] == "external_camera"
    ]
    camera_a_temporal = [
        row for row in temporal["frames"] if row["camera_id"] == "external_camera"
    ]
    axes[0].plot(x, [row["scale"] for row in camera_a_single], "o--", color=orange,
                 label="fresh affine each frame", linewidth=1.5, markersize=4)
    axes[0].plot(x, [row["scale"] for row in camera_a_temporal], "o-", color=teal,
                 label="Bayesian affine", linewidth=2.0, markersize=4)
    axes[0].set_ylabel("Camera A scale\n[relative units]")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("21 repeated four-camera updates: temporal anchoring stability and accuracy",
                      color=navy, fontweight="bold")

    axes[1].plot(x, cycle_values(single, "median_structure_depth_mae_m"), "o--",
                 color=orange, label="fresh affine")
    axes[1].plot(x, cycle_values(temporal, "median_structure_depth_mae_m"), "o-",
                 color=teal, label="Bayesian affine")
    axes[1].set_ylabel("Median camera\nstructure MAE [m]")

    axes[2].plot(x, 100 * cycle_values(single, "median_visibility_balanced_accuracy"),
                 "o--", color=orange, label="fresh affine")
    axes[2].plot(x, 100 * cycle_values(temporal, "median_visibility_balanced_accuracy"),
                 "o-", color=teal, label="Bayesian affine")
    axes[2].set_ylabel("Median camera visibility\nbalanced accuracy [%]")
    axes[2].set_xlabel("Operational update index (10 s cadence)")
    for axis in axes:
        axis.grid(color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
    fig.savefig(figure_dir / "01_temporal_sequence.png", dpi=180, facecolor="white")
    fig.savefig(figure_dir / "01_temporal_sequence.pdf", facecolor="white")
    plt.close(fig)

    dropout_single = result["arms"]["dropout/enhanced_single"]
    dropout_temporal = result["arms"]["dropout/enhanced_temporal"]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.7), sharex=True, constrained_layout=True)
    axes[0].step(x, [cycle["n_valid_cameras"] for cycle in dropout_single["cycles"]],
                 where="mid", color=orange, linewidth=2.0, label="fresh affine")
    axes[0].step(x, [cycle["n_valid_cameras"] for cycle in dropout_temporal["cycles"]],
                 where="mid", color=teal, linewidth=2.0, label="Bayesian stale-prior rule")
    axes[0].set_ylim(-0.2, 4.4)
    axes[0].set_yticks(range(5))
    axes[0].set_ylabel("Valid camera maps\nout of four")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("Predeclared floor-anchor dropouts every fifth update",
                      color=navy, fontweight="bold")
    temporal_bacc = [
        cycle["median_visibility_balanced_accuracy"] for cycle in dropout_temporal["cycles"]
    ]
    axes[1].plot(x, 100 * np.asarray(temporal_bacc, dtype=float), "o-", color=teal)
    axes[1].set_ylabel("Temporal visibility\nbalanced accuracy [%]")
    axes[1].set_xlabel("Operational update index (10 s cadence)")
    for axis in axes:
        axis.grid(color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for source_time in DROPOUT_TIMES:
            index = SOURCE_TIMES.index(source_time) + 1
            axis.axvspan(index - 0.35, index + 0.35, color="#F2C14E", alpha=0.28)
    fig.savefig(figure_dir / "02_anchor_dropout.png", dpi=180, facecolor="white")
    fig.savefig(figure_dir / "02_anchor_dropout.pdf", facecolor="white")
    plt.close(fig)

    provenance = {
        "generator": str(Path(__file__).relative_to(REPO)),
        "results": str((OUT / "results.json").relative_to(REPO)),
        "source_hashes": EXPECTED_HASHES,
        "note": "Gazebo oracle data were evaluation-only",
    }
    for path in sorted(figure_dir.glob("*")):
        path.with_suffix(path.suffix + ".provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def write_results_markdown(result: dict) -> None:
    natural_single = result["arms"]["natural/enhanced_single"]
    natural_temporal = result["arms"]["natural/enhanced_temporal"]
    dropout_single = result["arms"]["dropout/enhanced_single"]
    dropout_temporal = result["arms"]["dropout/enhanced_temporal"]
    comparison = result["comparisons"]
    dropout_comparison = comparison["dropout_temporal_minus_single"]
    scale_ratio = comparison["natural_temporal_minus_single"]["successive_scale_change_ratio"]
    shift_ratio = comparison["natural_temporal_minus_single"]["successive_shift_change_ratio"]
    scale_ratio_text = (
        f"{scale_ratio:.3f}"
        if scale_ratio is not None
        else "undefined because the single-frame median change is exactly zero"
    )
    shift_ratio_text = (
        f"{shift_ratio:.3f}"
        if shift_ratio is not None
        else "undefined because the single-frame median change is exactly zero"
    )
    text = f"""# Temporal ground-anchor sequence results

Status: **longitudinal mechanism evidence**, not a real-time drift experiment.

The frozen Gazebo sequence supplies 21 synchronized updates × four cameras = 84 RGB
frames. DA-V2 relative depth was inferred from RGB only. Each source frame is replayed at
a 10 s operational cadence, giving 21 Bayesian updates across 200 s. Gazebo depth and
visibility are opened only afterwards by the evaluator.

## Untouched sequence

| arm | valid camera updates | median cycle structure MAE | median cycle visibility balanced accuracy | median cycle visible IoU | median four-camera method time |
|---|---:|---:|---:|---:|---:|
| enhanced, fresh affine | {natural_single['n_valid_camera_updates']}/84 | {natural_single['median_cycle_structure_depth_mae_m']:.3f} m | {natural_single['median_cycle_visibility_balanced_accuracy']:.2%} | {natural_single['median_cycle_visible_iou']:.2%} | {natural_single['median_four_camera_method_seconds']:.2f} s |
| enhanced + Bayesian affine | {natural_temporal['n_valid_camera_updates']}/84 | {natural_temporal['median_cycle_structure_depth_mae_m']:.3f} m | {natural_temporal['median_cycle_visibility_balanced_accuracy']:.2%} | {natural_temporal['median_cycle_visible_iou']:.2%} | {natural_temporal['median_four_camera_method_seconds']:.2f} s |

Median absolute successive scale change ratio, temporal/single:
{scale_ratio_text}.
The equivalent shift-change ratio is
{shift_ratio_text}.
These are stability diagnostics in DA-V2 relative-depth parameter units, not localization
errors.

## Predeclared floor-anchor dropout

The external floor mask is set empty at four predeclared synchronized updates, producing
16 camera-frame anchor failures. The selection was frozen in `PROTOCOL.md` and did not use
oracle outcomes.

| arm | valid all-sequence updates | valid injected-dropout updates |
|---|---:|---:|
| enhanced, fresh affine | {dropout_single['n_valid_camera_updates']}/84 | {dropout_single['n_valid_injected_dropout_updates']}/16 |
| enhanced + Bayesian affine | {dropout_temporal['n_valid_camera_updates']}/84 | {dropout_temporal['n_valid_injected_dropout_updates']}/16 |

The temporal arm retained
{dropout_comparison['temporal_retention_fraction_at_injected_dropouts']:.0%} of the
injected dropout updates by reusing only a recent scale/shift posterior. It still
computed metric depth and visibility from the current RGB-derived depth frame; obstacle
evidence was not carried over.

At the four dropout cycles, stale-prior reuse changed median structure-depth MAE by
{1000 * dropout_comparison['dropout_cycle_structure_mae_delta_vs_natural_temporal_m']:+.1f}
mm, visibility balanced accuracy by
{100 * dropout_comparison['dropout_cycle_balanced_accuracy_delta_vs_natural_temporal']:+.3f}
percentage points, and visible IoU by
{100 * dropout_comparison['dropout_cycle_visible_iou_delta_vs_natural_temporal']:+.3f}
percentage points relative to the untouched temporal replay of those same frames.

## Boundary and interpretation

This demonstrates repeated Bayesian updates and the intended graceful-degradation
mechanism on real Gazebo renders. It does not demonstrate natural floor-segmentation
failure frequency, 200 seconds of illumination drift, or real-camera ageing. The 21
updates are sequential and dependent, so all summaries are descriptive rather than iid
confidence claims.
"""
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")


def _compact(result: dict) -> dict:
    keep = (
        "n_valid_camera_updates",
        "n_refused_camera_updates",
        "n_valid_injected_dropout_updates",
        "median_cycle_structure_depth_mae_m",
        "median_cycle_visibility_balanced_accuracy",
        "median_cycle_visible_iou",
        "median_four_camera_method_seconds",
        "parameter_stability",
    )
    return {
        arm: {key: summary[key] for key in keep}
        for arm, summary in result["arms"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer", action="store_true", help="run missing RGB-only predictions")
    parser.add_argument("--evaluate", action="store_true", help="score all four frozen arms")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.infer and not args.evaluate:
        args.infer = args.evaluate = True
    if args.infer:
        manifest = run_inference(args.device)
        print(json.dumps(manifest["forward_seconds"], indent=2))
    if args.evaluate:
        result = evaluate()
        print(json.dumps(_compact(result), indent=2))
        print(f"wrote {OUT / 'results.json'} and figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
