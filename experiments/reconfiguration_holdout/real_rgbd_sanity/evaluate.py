#!/usr/bin/env python3
"""Evaluate the frozen floor-affine protocol on physical TorWIC RGB-D frames.

Read ``PREREGISTRATION.md`` before changing this file.  The sensor depth from
frame 000000 is used once to recover the rig-to-floor plane that TorWIC does not
publish.  Each camera's monocular affine is then frozen and applied unchanged
to seven held-out frames.  Held-out depth is evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for relative in (
    "experiments/monocular_depth_adapter",
    "experiments/mono_depth_visibility",
):
    sys.path.insert(0, str(REPO / relative))

import ground_anchoring as ga  # noqa: E402
from monodepth import CameraIntrinsics, DepthRequest, MonocularDepthAdapter  # noqa: E402


MODEL = "unidepth_v2_vits14"
COMMISSIONING_FRAME = "000000"
TEST_FRAMES = ("000115", "000230", "000345", "000460", "000575", "000690", "000805")
SIDES = ("left", "right")
STRUCTURE_LABELS = np.array([4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15], dtype=np.uint8)
DEPTH_MIN_M = 0.4
DEPTH_MAX_M = 20.0
EROSION_PX = 7
PIXEL_STEP = 4
PLANE_RANSAC_SEED = 20260819
PLANE_RANSAC_TRIALS = 2_000
PLANE_INLIER_TOL_M = 0.04
PLANE_MIN_INLIERS = 1_000
PLANE_MIN_INLIER_FRACTION = 0.70


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_provenance(out_dir: Path, model: str) -> list[dict]:
    """Hash every cached model output consumed by this evaluation."""

    prediction_dir = out_dir / "predictions" / model
    paths = sorted(prediction_dir.glob("*.npz"))
    expected = len(SIDES) * (1 + len(TEST_FRAMES))
    if len(paths) != expected:
        raise RuntimeError(
            f"prediction cache has {len(paths)} files, expected exactly {expected}"
        )
    return [
        {
            "path": str(path.relative_to(REPO)),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]


def _parse_scalar(text: str, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}\s*[:=]\s*([^\s]+)", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing calibration key {key}")
    return float(match.group(1))


def load_calibrations(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, dict] = {}
    for side, number in (("left", 1), ("right", 2)):
        prefix = f"Camera{number}"
        distortion_match = re.search(
            rf"^{prefix}\.D\s*=\s*\[([^\]]+)\]", text, flags=re.MULTILINE
        )
        if not distortion_match:
            raise RuntimeError(f"missing calibration key {prefix}.D")
        distortion = np.fromstring(distortion_match.group(1), sep=",", dtype=float)
        if distortion.size != 5:
            raise RuntimeError(f"{prefix}.D has {distortion.size} values, expected 5")
        width = int(_parse_scalar(text, f"{prefix}.width"))
        height = int(_parse_scalar(text, f"{prefix}.height"))
        K = np.array(
            [
                [_parse_scalar(text, f"{prefix}.fx"), 0.0, _parse_scalar(text, f"{prefix}.cx")],
                [0.0, _parse_scalar(text, f"{prefix}.fy"), _parse_scalar(text, f"{prefix}.cy")],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, distortion, None, K, (width, height), cv2.CV_32FC1
        )
        result[side] = {
            "K": K,
            "D": distortion,
            "width": width,
            "height": height,
            "map_x": map_x,
            "map_y": map_y,
        }
    return result


def _verify_inputs(data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for member in manifest["members"]:
        path = data_root / member["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_sha256(path)
        if actual != member["sha256"]:
            raise RuntimeError(f"input hash mismatch for {path}: {actual} != {member['sha256']}")
    calibration = data_root / manifest["calibration"]["path"]
    if _file_sha256(calibration) != manifest["calibration"]["sha256"]:
        raise RuntimeError("calibration hash does not match fetch manifest")
    if manifest["commissioning_frame"] != COMMISSIONING_FRAME:
        raise RuntimeError("fetched commissioning frame disagrees with frozen evaluator")
    if tuple(manifest["test_frames"]) != TEST_FRAMES:
        raise RuntimeError("fetched test frames disagree with frozen evaluator")
    return manifest


def load_frame(data_root: Path, calibration: dict, side: str, frame: str) -> dict:
    raw = data_root / "raw"
    rgb = np.asarray(Image.open(raw / f"image_{side}/{frame}.png").convert("RGB"), dtype=np.uint8)
    depth_mm = np.asarray(Image.open(raw / f"depth_{side}/{frame}.png"), dtype=np.uint16)
    labels = np.asarray(
        Image.open(raw / f"segmentation_greyscale_{side}/{frame}.png"), dtype=np.uint8
    )
    expected = (calibration["height"], calibration["width"])
    if rgb.shape[:2] != expected or depth_mm.shape != expected or labels.shape != expected:
        raise RuntimeError(
            f"{side}/{frame}: expected {expected}, got RGB {rgb.shape}, depth {depth_mm.shape}, "
            f"labels {labels.shape}"
        )
    map_x, map_y = calibration["map_x"], calibration["map_y"]
    rgb_u = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    depth_u = cv2.remap(depth_mm, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    labels_u = cv2.remap(labels, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return {
        "rgb": np.ascontiguousarray(rgb_u),
        "depth_m": depth_u.astype(np.float32) * 0.001,
        "labels": labels_u,
    }


def eroded_floor_mask(labels: np.ndarray) -> np.ndarray:
    kernel = np.ones((EROSION_PX, EROSION_PX), dtype=np.uint8)
    return cv2.erode((labels == 1).astype(np.uint8), kernel, iterations=1).astype(bool)


def backproject_z_depth(K: np.ndarray, depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v, u = np.nonzero(mask)
    z = depth_m[v, u].astype(np.float64)
    x = (u.astype(float) - K[0, 2]) * z / K[0, 0]
    y = (v.astype(float) - K[1, 2]) * z / K[1, 1]
    return np.column_stack([x, y, z])


def fit_floor_plane(points: np.ndarray) -> dict:
    """Deterministic 3-point RANSAC followed by orthogonal SVD refinement."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N,3) points, got {pts.shape}")
    rng = np.random.default_rng(PLANE_RANSAC_SEED)
    best = np.zeros(len(pts), dtype=bool)
    for _ in range(PLANE_RANSAC_TRIALS):
        sample = pts[rng.choice(len(pts), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-10:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(pts @ normal + offset) <= PLANE_INLIER_TOL_M
        if int(inliers.sum()) > int(best.sum()):
            best = inliers
    if int(best.sum()) < 3:
        raise RuntimeError("floor-plane RANSAC found no non-degenerate model")
    inlier_points = pts[best]
    center = inlier_points.mean(axis=0)
    covariance = (inlier_points - center).T @ (inlier_points - center)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    residual = np.abs(pts @ normal + offset)
    refined = residual <= PLANE_INLIER_TOL_M
    if normal[1] > 0:  # camera y points down; orient the normal upward for readability
        normal, offset = -normal, -offset
    n_inlier = int(refined.sum())
    fraction = float(refined.mean())
    return {
        "normal": normal,
        "offset": offset,
        "n_points": int(len(pts)),
        "n_inlier": n_inlier,
        "inlier_fraction": fraction,
        "residual_rms_m": float(np.sqrt(np.mean(residual[refined] ** 2))),
        "residual_p95_m": float(np.percentile(residual[refined], 95)),
        "passes": n_inlier >= PLANE_MIN_INLIERS and fraction >= PLANE_MIN_INLIER_FRACTION,
    }


def plane_depth_map(K: np.ndarray, shape: tuple[int, int], normal: np.ndarray, offset: float) -> np.ndarray:
    height, width = shape
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    ray_x = (u - K[0, 2]) / K[0, 0]
    ray_y = (v - K[1, 2]) / K[1, 1]
    denominator = normal[0] * ray_x + normal[1] * ray_y + normal[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -float(offset) / denominator
    return np.where(
        np.isfinite(depth) & (depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M), depth, np.nan
    )


def depth_metrics(predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(predicted)
        & np.isfinite(truth)
        & (predicted > 0.0)
        & (truth >= DEPTH_MIN_M)
        & (truth <= DEPTH_MAX_M)
    )
    p = np.asarray(predicted, dtype=np.float64)[valid]
    t = np.asarray(truth, dtype=np.float64)[valid]
    if p.size == 0:
        return {"n": 0}
    error = p - t
    ratio = np.maximum(p / t, t / p)
    return {
        "n": int(p.size),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "abs_rel": float(np.mean(np.abs(error) / t)),
        "delta1": float(np.mean(ratio < 1.25)),
        "bias_m": float(np.mean(error)),
        "median_abs_err_m": float(np.median(np.abs(error))),
        "p95_abs_err_m": float(np.percentile(np.abs(error), 95)),
    }


def oracle_affine(
    predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(predicted)
        & np.isfinite(truth)
        & (predicted > 0.0)
        & (truth >= DEPTH_MIN_M)
        & (truth <= DEPTH_MAX_M)
    )
    p, t = predicted[valid].astype(np.float64), truth[valid].astype(np.float64)
    if p.size < 2:
        return np.full_like(predicted, np.nan, dtype=float), [float("nan"), float("nan")]
    design = np.column_stack([p, np.ones_like(p)])
    (scale, shift), *_ = np.linalg.lstsq(design, t, rcond=None)
    return scale * np.asarray(predicted, dtype=float) + shift, [float(scale), float(shift)]


def _prediction_cache_path(out_dir: Path, model: str, side: str, frame: str) -> Path:
    return out_dir / "predictions" / model / f"{side}_{frame}.npz"


def run_or_load_predictions(
    frames: dict[tuple[str, str], dict], calibrations: dict[str, dict], out_dir: Path,
    model: str, device: str, force: bool,
) -> tuple[dict[tuple[str, str], dict], dict]:
    loaded: dict[tuple[str, str], dict] = {}
    missing: list[tuple[str, str]] = []
    for key, record in frames.items():
        side, frame = key
        path = _prediction_cache_path(out_dir, model, side, frame)
        rgb_hash = sha256(record["rgb"].tobytes()).hexdigest()
        if path.is_file() and not force:
            cache = np.load(path, allow_pickle=False)
            metadata = json.loads(str(cache["metadata_json"].item()))
            if metadata.get("undistorted_rgb_sha256") == rgb_hash and metadata.get("model_name") == model:
                loaded[key] = {
                    "depth": cache["depth"].astype(np.float32),
                    "valid": cache["valid"].astype(bool),
                    "convention": metadata["convention"],
                    "metadata": metadata,
                }
                continue
        missing.append(key)

    run_metadata: dict = {
        "model": model,
        "requested_device": device,
        "cache_hits": len(loaded),
        "frames_inferred_this_run": len(missing),
    }
    if loaded:
        cached_metadata = next(iter(loaded.values()))["metadata"]
        run_metadata["model_info"] = cached_metadata["model"]
        run_metadata["cached_prediction_devices"] = sorted(
            {record["metadata"]["model"]["device"] for record in loaded.values()}
        )
    if missing:
        requests = []
        for side, frame in missing:
            calibration = calibrations[side]
            K = calibration["K"]
            intrinsics = CameraIntrinsics.from_matrix(
                K, calibration["width"], calibration["height"]
            )
            rgb = frames[(side, frame)]["rgb"]
            requests.append(
                DepthRequest(
                    image_id=f"torwic_{side}_{frame}",
                    image=rgb,
                    intrinsics=intrinsics,
                    source_path=f"TorWIC-SLAM/Aisle_CCW/image_{side}/{frame}.png",
                    image_sha256=sha256(rgb.tobytes()).hexdigest(),
                )
            )
        with MonocularDepthAdapter(
            model, device=device, batch_size=1, uncertainty="none", seed=PLANE_RANSAC_SEED
        ) as adapter:
            predictions = adapter.predict(requests)
            run_metadata.update(
                model_info=adapter.info.as_dict(),
                determinism=adapter.determinism_config,
                oom_events=adapter.oom_events,
            )
        for key, prediction in zip(missing, predictions):
            side, frame = key
            metadata = prediction.metadata()
            metadata.update(
                model_name=model,
                undistorted_rgb_sha256=sha256(frames[key]["rgb"].tobytes()).hexdigest(),
            )
            path = _prediction_cache_path(out_dir, model, side, frame)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                depth=prediction.depth.astype(np.float32),
                valid=prediction.valid.astype(np.uint8),
                metadata_json=np.array(json.dumps(metadata)),
            )
            loaded[key] = {
                "depth": prediction.depth.astype(np.float32),
                "valid": prediction.valid.astype(bool),
                "convention": prediction.convention.value,
                "metadata": metadata,
            }
    return loaded, run_metadata


def median_summary(records: Iterable[dict]) -> dict:
    rows = list(records)
    metric_names = ("mae_m", "rmse_m", "abs_rel", "delta1", "bias_m")
    output: dict = {"n_frames": len(rows)}
    for arm in ("raw", "anchored", "oracle_affine", "plane_floor_diagnostic"):
        output[arm] = {}
        for metric in metric_names:
            values = [r[arm][metric] for r in rows if metric in r.get(arm, {})]
            if values:
                output[arm][f"median_{metric}"] = float(np.median(values))
    return output


def _write_csv(path: Path, records: list[dict]) -> None:
    columns = ["camera", "frame", "n_structure"]
    for arm in ("raw", "anchored", "oracle_affine", "plane_floor_diagnostic"):
        for metric in ("n", "mae_m", "rmse_m", "abs_rel", "delta1", "bias_m"):
            columns.append(f"{arm}__{metric}")
    columns += ["mae_improvement_m", "mae_relative_improvement", "oracle_scale", "oracle_shift"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {
                "camera": record["camera"],
                "frame": record["frame"],
                "n_structure": record["anchored"].get("n", 0),
                "mae_improvement_m": record["mae_improvement_m"],
                "mae_relative_improvement": record["mae_relative_improvement"],
                "oracle_scale": record["oracle_affine_parameters"][0],
                "oracle_shift": record["oracle_affine_parameters"][1],
            }
            for arm in ("raw", "anchored", "oracle_affine", "plane_floor_diagnostic"):
                for metric in ("n", "mae_m", "rmse_m", "abs_rel", "delta1", "bias_m"):
                    row[f"{arm}__{metric}"] = record.get(arm, {}).get(metric, "")
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = REPO / "logs/studies/reconfiguration_holdout/real_rgbd_sanity/torwic_subset"
    default_out = REPO / "logs/studies/reconfiguration_holdout/real_rgbd_sanity/results"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-inference", action="store_true")
    args = parser.parse_args()

    manifest = _verify_inputs(args.data)
    calibrations = load_calibrations(args.data / manifest["calibration"]["path"])
    all_frames = (COMMISSIONING_FRAME,) + TEST_FRAMES
    frames = {
        (side, frame): load_frame(args.data, calibrations[side], side, frame)
        for side in SIDES
        for frame in all_frames
    }
    args.out.mkdir(parents=True, exist_ok=True)
    predictions, inference = run_or_load_predictions(
        frames, calibrations, args.out, args.model, args.device, args.force_inference
    )

    commissioning: dict[str, dict] = {}
    fits: dict[str, ga.GroundFit] = {}
    planes: dict[str, tuple[np.ndarray, float]] = {}
    for side in SIDES:
        calibration = calibrations[side]
        K = calibration["K"]
        frame = frames[(side, COMMISSIONING_FRAME)]
        sensor_valid = (
            np.isfinite(frame["depth_m"])
            & (frame["depth_m"] >= DEPTH_MIN_M)
            & (frame["depth_m"] <= DEPTH_MAX_M)
        )
        floor = eroded_floor_mask(frame["labels"])
        stride = np.zeros_like(floor)
        stride[::PIXEL_STEP, ::PIXEL_STEP] = True
        plane_mask = floor & sensor_valid & stride
        plane = fit_floor_plane(backproject_z_depth(K, frame["depth_m"], plane_mask))
        if not plane["passes"]:
            raise RuntimeError(f"{side}: commissioning plane failed frozen gates: {plane}")
        normal = np.asarray(plane["normal"], dtype=float)
        offset = float(plane["offset"])
        planes[side] = (normal, offset)
        analytic = plane_depth_map(K, frame["depth_m"].shape, normal, offset)

        prediction = predictions[(side, COMMISSIONING_FRAME)]
        anchor_mask = (
            floor
            & stride
            & prediction["valid"]
            & np.isfinite(prediction["depth"])
            & np.isfinite(analytic)
        )
        target = analytic[anchor_mask]
        fit = ga.fit_ground_affine(
            prediction["depth"][anchor_mask],
            target,
            prediction["convention"],
            config=ga.FitConfig(
                strict_convention=True,
                metric_scale_band=(0.05, 10.0),
                ransac_seed=0,
            ),
            anchor_depth_span_m=float(np.ptp(np.percentile(target, [5.0, 95.0]))),
            notes=(
                "TorWIC physical sanity check; commissioned on frame 000000 against a "
                "sensor-assisted floor-plane proxy; frozen for all held-out frames"
            ),
        )
        if not fit.status.is_ok:
            raise RuntimeError(f"{side}: commissioning affine failed frozen gates: {fit.to_dict()}")
        fits[side] = fit
        commissioning[side] = {
            "plane": {
                **{k: v for k, v in plane.items() if k not in ("normal",)},
                "normal": normal.tolist(),
                "offset": offset,
            },
            "affine": fit.to_dict(),
            "n_semantic_floor_pixels_before_stride": int((floor & sensor_valid).sum()),
        }

    records: list[dict] = []
    for side in SIDES:
        normal, offset = planes[side]
        fit = fits[side]
        K = calibrations[side]["K"]
        for frame_id in TEST_FRAMES:
            frame = frames[(side, frame_id)]
            prediction = predictions[(side, frame_id)]
            raw = prediction["depth"].astype(float)
            anchored = fit.apply(raw)
            sensor = frame["depth_m"].astype(float)
            structure = np.isin(frame["labels"], STRUCTURE_LABELS)
            common = (
                structure
                & prediction["valid"]
                & np.isfinite(sensor)
                & (sensor >= DEPTH_MIN_M)
                & (sensor <= DEPTH_MAX_M)
                & np.isfinite(raw)
                & (raw > 0.0)
                & np.isfinite(anchored)
                & (anchored > 0.0)
            )
            oracle, oracle_parameters = oracle_affine(raw, sensor, common)
            raw_metrics = depth_metrics(raw, sensor, common)
            anchored_metrics = depth_metrics(anchored, sensor, common)
            oracle_metrics = depth_metrics(oracle, sensor, common)

            analytic = plane_depth_map(K, sensor.shape, normal, offset)
            test_floor = eroded_floor_mask(frame["labels"])
            floor_diagnostic = depth_metrics(analytic, sensor, test_floor)
            raw_mae, anchored_mae = raw_metrics["mae_m"], anchored_metrics["mae_m"]
            records.append(
                {
                    "camera": side,
                    "frame": frame_id,
                    "raw": raw_metrics,
                    "anchored": anchored_metrics,
                    "oracle_affine": oracle_metrics,
                    "oracle_affine_parameters": oracle_parameters,
                    "plane_floor_diagnostic": floor_diagnostic,
                    "mae_improvement_m": raw_mae - anchored_mae,
                    "mae_relative_improvement": (raw_mae - anchored_mae) / raw_mae,
                }
            )

    aggregate = median_summary(records)
    by_camera = {
        side: median_summary(record for record in records if record["camera"] == side)
        for side in SIDES
    }
    raw_median = aggregate["raw"]["median_mae_m"]
    anchored_median = aggregate["anchored"]["median_mae_m"]
    criteria = {
        "commissioning_passed": all(
            commissioning[side]["plane"]["passes"]
            and commissioning[side]["affine"]["status"] == "ok"
            for side in SIDES
        ),
        "at_least_12_frames_with_1000_structure_pixels": sum(
            record["anchored"].get("n", 0) >= 1_000 for record in records
        ) >= 12,
        "each_camera_median_mae_improves": all(
            by_camera[side]["anchored"]["median_mae_m"]
            < by_camera[side]["raw"]["median_mae_m"]
            for side in SIDES
        ),
        "overall_median_mae_improves_by_10pct": (
            (raw_median - anchored_median) / raw_median >= 0.10
        ),
        "at_least_10_of_14_frames_improve": sum(
            record["mae_improvement_m"] > 0.0 for record in records
        ) >= 10,
    }
    summary = {
        "protocol": str((HERE / "PREREGISTRATION.md").relative_to(REPO)),
        "provenance": {
            "protocol_sha256": _file_sha256(HERE / "PREREGISTRATION.md"),
            "frozen_protocol_hash_record_sha256": _file_sha256(HERE / "FROZEN_SHA256"),
            "evaluation_script_sha256": _file_sha256(Path(__file__).resolve()),
            "input_manifest": str((args.data / "manifest.json").relative_to(REPO)),
            "input_manifest_sha256": _file_sha256(args.data / "manifest.json"),
            "prediction_artifacts": _prediction_provenance(args.out, args.model),
        },
        "claim_boundary": (
            "one-route physical component sanity check; mobile TorWIC RGB-D, not an "
            "overhead-camera, visibility, reconfiguration, or navigation replication"
        ),
        "model": args.model,
        "inference": inference,
        "commissioning": commissioning,
        "per_frame": records,
        "aggregate_medians": aggregate,
        "camera_medians": by_camera,
        "frozen_criteria": criteria,
        "supportive": all(criteria.values()),
    }
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.out / "per_frame.csv", records)
    print(json.dumps({
        "supportive": summary["supportive"],
        "frozen_criteria": criteria,
        "raw_median_mae_m": raw_median,
        "anchored_median_mae_m": anchored_median,
        "relative_improvement": (raw_median - anchored_median) / raw_median,
        "frames_improved": sum(record["mae_improvement_m"] > 0 for record in records),
        "n_frames": len(records),
    }, indent=2))
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
