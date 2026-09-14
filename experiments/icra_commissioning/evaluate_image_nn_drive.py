#!/usr/bin/env python3
"""Score one frozen image-aware checkpoint on exact RGB crops from one drive."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/icra_commissioning"))
sys.path.insert(0, str(REPO / "experiments/camera_observation_characterization"))
sys.path.insert(0, str(REPO / "experiments/fusion_on_fixed_routes"))
import train_world_position as training  # noqa: E402
from derive_interpretations import camera_models  # noqa: E402
import aligned  # noqa: E402


COLORS = {
    "camera_A": "#2b6cb0", "camera_B": "#dd6b20", "camera_C": "#2f855a",
    "camera_D": "#805ad5", "camera_E": "#c53030",
}


def metric(error: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(error)),
        "median_cm": float(np.median(error) * 100),
        "rms_cm": float(np.sqrt(np.mean(error ** 2)) * 100),
        "p95_cm": float(np.percentile(error, 95) * 100),
        "max_cm": float(np.max(error) * 100),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-reference-gap-s", type=float, default=0.15)
    args = parser.parse_args()

    run = args.run.resolve()
    table = aligned.rows(run)
    truth = aligned.truth_series(
        run, table, max_reference_gap_s=args.max_reference_gap_s)
    manifest = json.loads(args.camera_manifest.read_text())
    models = camera_models(manifest)
    specs = {entry["camera_id"]: entry for entry in manifest["cameras"]}

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "image_world_residual.v1" or not payload.get("use_image"):
        raise RuntimeError("checkpoint is not an image-aware world-residual model")
    if tuple(payload["feature_names"]) != training.FEATURE_NAMES:
        raise RuntimeError("checkpoint feature contract differs from evaluator")
    network = training.WorldResidualNet(len(training.FEATURE_NAMES), True)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)

    records = []
    for path in sorted(args.crops.glob("camera_*_*.npz")):
        camera_id, stamp_text = path.stem.rsplit("_", 1)
        stamp_ns = int(stamp_text)
        stamp = stamp_ns * 1e-9
        gx, gy = truth.at(np.asarray([stamp]))
        if not (math.isfinite(gx[0]) and math.isfinite(gy[0])):
            continue
        with np.load(path) as saved:
            crop = np.asarray(saved["crop"], dtype=np.uint8)
            bbox = np.asarray(saved["bbox_xyxy"], dtype=float)
            selected_uv = np.asarray(saved["selected_uv"], dtype=float)
            height, width = np.asarray(saved["image_shape"], dtype=int)
            confidence = float(saved["confidence"])
        x0, y0, x1, y1 = bbox
        if x0 <= 0.5 or y0 <= 0.5 or x1 >= width - 0.5 or y1 >= height - 0.5:
            continue
        raw = models[camera_id].pixel_to_world(*selected_uv)
        if raw is None:
            continue
        raw = np.asarray(raw, dtype=float)
        camera_xy = np.asarray(models[camera_id].cam_pos[:2], dtype=float)
        direction = raw - camera_xy
        distance = float(np.linalg.norm(direction))
        if not math.isfinite(distance) or distance <= 1e-6:
            continue
        along = direction / distance
        basis = np.column_stack((along, (-along[1], along[0])))
        relative = math.atan2(direction[1], direction[0]) - float(
            specs[camera_id]["pose_xyz_rpy"][5])
        box_width, box_height = x1 - x0, y1 - y0
        features = np.asarray([
            distance, 1.0 / distance, box_width / width, box_height / height,
            box_width / box_height, 0.5 * (x0 + x1) / width, y1 / height,
            math.cos(relative), math.sin(relative), confidence,
            *(float(camera_id == candidate) for candidate in training.CAMERAS),
        ], dtype=np.float32)
        if not np.isfinite(features).all():
            continue
        image = torch.from_numpy(crop.astype(np.float32) / 255.0)[None]
        normalized = torch.from_numpy((features - feature_mean) / feature_std)[None]
        with torch.no_grad():
            delta = network(image, normalized)[0].numpy()
        corrected = raw + basis @ delta
        ground_truth = np.asarray((gx[0], gy[0]))
        residual_ray = basis.T @ (corrected - ground_truth)
        records.append({
            "camera": camera_id, "stamp": stamp, "gt": ground_truth,
            "raw": raw, "corrected": corrected,
            "raw_error": float(np.linalg.norm(raw - ground_truth)),
            "nn_error": float(np.linalg.norm(corrected - ground_truth)),
            "along_error": float(residual_ray[0]),
            "across_error": float(residual_ray[1]),
        })
    if not records:
        raise RuntimeError("no crop had bounded ground-truth support")

    stamps = np.asarray([r["stamp"] for r in records])
    all_truth = np.column_stack((truth.x, truth.y))
    truth_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(all_truth, axis=0), axis=1))]
    distance = np.interp(stamps, truth.t, truth_distance)
    raw_error = np.asarray([r["raw_error"] for r in records])
    nn_error = np.asarray([r["nn_error"] for r in records])
    along = np.asarray([r["along_error"] for r in records])
    across = np.asarray([r["across_error"] for r in records])
    camera = np.asarray([r["camera"] for r in records])

    summary = {
        "schema": "image_world_drive_shadow.v1",
        "status": "single_drive_diagnostic",
        "layer": "individual camera reading at image capture stamp",
        "run": str(run),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_seed": int(payload["seed"]),
        "truth_reference": truth.source,
        "max_reference_gap_s": args.max_reference_gap_s,
        "raw": metric(raw_error),
        "image_nn": metric(nn_error),
        "image_nn_signed_bias_cm": {
            "along_camera_ray": float(np.mean(along) * 100),
            "across_camera_ray": float(np.mean(across) * 100),
        },
        "per_camera": {
            cam: {"raw": metric(raw_error[camera == cam]),
                  "image_nn": metric(nn_error[camera == cam])}
            for cam in sorted(set(camera))
        },
        "drive_distance_m": float(truth_distance[-1]),
        "note": "One drive is diagnostic; camera readings within it are correlated.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output / "readings.npz", stamp=stamps, distance_m=distance,
        camera=camera, raw_error_m=raw_error, nn_error_m=nn_error,
        along_error_m=along, across_error_m=across,
        gt_xy=np.stack([r["gt"] for r in records]),
        raw_xy=np.stack([r["raw"] for r in records]),
        corrected_xy=np.stack([r["corrected"] for r in records]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    axes[0].plot(truth.x, truth.y, color="#1f2937", linewidth=2.0, label="ground truth")
    corrected = np.stack([r["corrected"] for r in records])
    for cam in sorted(set(camera)):
        use = camera == cam
        axes[0].scatter(corrected[use, 0], corrected[use, 1], s=9,
                        alpha=0.5, color=COLORS[cam], label=cam)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("world x [m]")
    axes[0].set_ylabel("world y [m]")
    axes[0].set_title("Image-aware NN camera readings")
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].scatter(distance, raw_error * 100, s=8, color="#9ca3af", alpha=0.28,
                    label="raw box projection")
    for cam in sorted(set(camera)):
        use = camera == cam
        axes[1].scatter(distance[use], nn_error[use] * 100, s=10, alpha=0.55,
                        color=COLORS[cam], label=f"NN {cam}")
    bins = np.linspace(0.0, max(float(distance.max()), 1e-6), 25)
    centres, medians = [], []
    for left, right in zip(bins[:-1], bins[1:]):
        use = (distance >= left) & (distance < right)
        if np.any(use):
            centres.append((left + right) / 2)
            medians.append(np.median(nn_error[use]) * 100)
    axes[1].plot(centres, medians, color="#111827", linewidth=2.2,
                 label="NN local median")
    axes[1].set_xlabel("distance travelled [m]")
    axes[1].set_ylabel("camera-reading position error [cm]")
    axes[1].set_title(
        f"n={len(records)}; median {summary['image_nn']['median_cm']:.1f} cm; "
        f"p95 {summary['image_nn']['p95_cm']:.1f} cm")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(args.output / "error_along_drive.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
