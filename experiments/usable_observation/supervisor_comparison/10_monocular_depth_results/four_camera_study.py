#!/usr/bin/env python3
"""Four-camera depth anchoring and depth-buffer visibility comparison.

The method side reads RGB, camera calibration, and the planner's 2-D traversable
regions.  Gazebo depth and geometry-raycast visibility are opened only after the
predictions have been anchored, for evaluation and plotting.

The clear frame at t=0.4 s is the commissioning frame.  Its affine floor fit is
reused unchanged at t=1.2 s, after a loaded pallet appears in Camera A's aisle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ADAPTER_DIR = REPO / "experiments/monocular_depth_adapter"
VIS_DIR = REPO / "experiments/mono_depth_visibility"
sys.path.insert(0, str(ADAPTER_DIR))
sys.path.insert(0, str(VIS_DIR))

from monodepth import (  # noqa: E402
    CameraIntrinsics,
    DepthRequest,
    MonocularDepthAdapter,
    storage,
)
import ground_anchoring as ga  # noqa: E402


DATASET = REPO / "logs/studies/dynamic_world_oracle/s01_box_in_aisle/run01"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
OUT = HERE / "four_camera"
PRED_DIR = OUT / "predictions"
MAP_DIR = OUT / "maps"
FIG_DIR = HERE / "figures"

TIMES = (0.4, 1.2)
MODELS = (
    "dav2_metric_indoor_small",
    "dav2_relative_small",
    "metric3d_v2_vit_small",
    "unidepth_v2_vits14",
)
MODEL_LABELS = {
    "dav2_metric_indoor_small": "DA-V2 metric S",
    "dav2_relative_small": "DA-V2 relative S",
    "metric3d_v2_vit_small": "Metric3D v2 ViT-S",
    "unidepth_v2_vits14": "UniDepthV2 ViT-S",
}
CAM_LABELS = {
    "external_camera": "Camera A",
    "external_camera_b": "Camera B",
    "external_camera_c": "Camera C",
    "external_camera_d": "Camera D",
}

INK = "#17324D"
SOFT = "#536471"
TEAL = "#18A999"
ORACLE = "#76549A"
ORANGE = "#E07A5F"
GRID = "#D8E2EA"


def _records() -> list[dict]:
    wanted = {round(t, 3) for t in TIMES}
    rows = [json.loads(line) for line in (DATASET / "records.jsonl").read_text().splitlines()]
    return [r for r in rows if round(float(r["timestamp"]), 3) in wanted]


def _prediction_id(record: dict) -> str:
    return f"s01_t{int(round(1000 * float(record['timestamp']))):07d}ms_{record['camera_id']}"


def _intrinsics(record: dict) -> CameraIntrinsics:
    k = record["camera_intrinsics"]
    return CameraIntrinsics(
        fx=float(k["fx"]), fy=float(k["fy"]), cx=float(k["cx"]), cy=float(k["cy"]),
        width=int(k["img_width"]), height=int(k["img_height"]),
    )


def run_inference(records: list[dict], models: tuple[str, ...], device: str) -> None:
    """Run RGB-only inference; no oracle field is read in this function."""
    requests = []
    for record in records:
        visible = ga.method_visible_record(record)
        rgb_path = DATASET / visible["rgb_path"]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        requests.append(DepthRequest(
            _prediction_id(record), rgb, _intrinsics(record),
            source_path=str(rgb_path.relative_to(REPO)),
        ))

    for model in models:
        model_out = PRED_DIR / model
        expected = [model_out / f"{req.image_id}__{model}.json" for req in requests]
        if all(path.is_file() for path in expected):
            print(f"{model}: using {len(expected)} existing predictions")
            continue
        print(f"{model}: predicting {len(requests)} RGB frames", flush=True)
        with MonocularDepthAdapter(
            model, device=device, batch_size=1, uncertainty="none"
        ) as adapter:
            storage.save_all(adapter.predict(requests), model_out)


def _drivable() -> list[ga.Footprint]:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    regions = profile["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    return [
        ga.Footprint(
            xmin=float(r["xmin"]), xmax=float(r["xmax"]),
            ymin=float(r["ymin"]), ymax=float(r["ymax"]), name=str(r["name"]),
        )
        for r in regions if r.get("type") == "traversable"
    ]


def _grid(manifest: dict) -> tuple[np.ndarray, np.ndarray, list[float]]:
    g = manifest["grid"]
    res = float(g["resolution_m"])
    xs = float(g["xmin"]) + (np.arange(int(g["nx"])) + 0.5) * res
    ys = float(g["ymin"]) + (np.arange(int(g["ny"])) + 0.5) * res
    return xs, ys, [float(g["xmin"]), float(g["xmax"]), float(g["ymin"]), float(g["ymax"])]


def _to_ga_prediction(pred, record: dict) -> ga.DepthPrediction:
    return ga.DepthPrediction(
        values=pred.depth,
        convention=pred.convention.value,
        valid_mask=pred.valid,
        uncertainty=None,
        model_name=pred.model.model_name,
        checkpoint=pred.model.checkpoint,
        inference_time_s=pred.timing.forward_s,
        frame_id=pred.image_id,
        camera_id=record["camera_id"],
    )


def _load_prediction(model: str, record: dict):
    stem = f"{_prediction_id(record)}__{model}.json"
    return storage.load_prediction(PRED_DIR / model / stem)


def _fit_clear_frame(pred, record: dict, drivable: list[ga.Footprint]):
    calib = ga.camera_from_record(record)
    gp = _to_ga_prediction(pred, record)
    anchors = ga.select_floor_anchors(
        calib, ga.FloorPlane(), drivable,
        config=ga.AnchorConfig(pixel_step=4, require_drivable=True),
        valid_mask=gp.valid_mask,
    )
    pred_z = ga.to_optical_axis(gp.values, gp.convention, calib)
    anchor_z = ga.to_optical_axis(
        gp.values[anchors.v.astype(int), anchors.u.astype(int)], gp.convention, calib,
        u=anchors.u, v=anchors.v,
    )
    fit = ga.fit_ground_affine(
        anchor_z, anchors.depth_m, gp.convention,
        # Keep the affine diagnostic available even when a model that claims to
        # be metric needs a surprisingly large correction. The normal
        # operational [0.5, 2.0] gate is recorded separately and controls model
        # selection for the visibility figure.
        config=ga.FitConfig(strict_convention=True, metric_scale_band=(0.05, 10.0)),
        anchor_depth_span_m=anchors.depth_span_m,
        notes="commissioning fit at t=0.4 s; reused at t=1.2 s",
    )
    if not fit.status.is_ok:
        raise RuntimeError(f"{record['camera_id']} {pred.model.model_name}: {fit.status.value}: {fit.notes}")
    return fit, calib, anchors


def _metric_from_fit(pred, record: dict, fit, calib):
    gp = _to_ga_prediction(pred, record)
    pred_z = ga.to_optical_axis(gp.values, gp.convention, calib)
    depth = fit.apply(pred_z)
    sigma = ga.predicted_depth_sigma(fit, pred_z, depth, None)
    valid = gp.valid_mask & np.isfinite(depth) & np.isfinite(sigma) & (depth > 0) & (depth < 60)
    return depth, sigma, valid


def _full_floor_depth(calib) -> np.ndarray:
    u, v = calib.pixel_grid(step=1)
    depth, _ = ga.analytic_plane_depth(calib, ga.FloorPlane(), u, v)
    return depth.reshape(calib.height, calib.width)


def _depth_errors(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    ok = mask & np.isfinite(pred) & np.isfinite(truth) & (truth > 0) & (pred > 0)
    p, t = pred[ok].astype(float), truth[ok].astype(float)
    if not p.size:
        return {"n": 0}
    err = p - t
    ratio = np.maximum(p / t, t / p)
    return {
        "n": int(p.size),
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_err_m": float(np.median(np.abs(err))),
        "bias_m": float(np.mean(err)),
        "abs_rel": float(np.mean(np.abs(err) / t)),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def _robust_affine(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    p, t = pred[mask].astype(float), target[mask].astype(float)
    keep = np.isfinite(p) & np.isfinite(t)
    a, b = 1.0, 0.0
    for _ in range(3):
        if int(keep.sum()) < 64:
            break
        a, b = np.linalg.lstsq(
            np.column_stack([p[keep], np.ones(int(keep.sum()))]), t[keep], rcond=None
        )[0]
        residual = np.abs(a * p + b - t)
        scale = float(np.std(residual[keep])) or 1.0
        keep = np.isfinite(p) & np.isfinite(t) & (residual < 2.5 * scale)
    return float(a), float(b)


def _oracle_affine(pred, truth: np.ndarray, mask: np.ndarray, calib) -> np.ndarray:
    gp = _to_ga_prediction(pred, {"camera_id": calib.camera_id})
    pred_z = ga.to_optical_axis(gp.values, gp.convention, calib)
    target = np.where(truth > 0, 1.0 / truth, np.nan) if gp.convention.is_inverse else truth
    a, b = _robust_affine(pred_z, target, mask & gp.valid_mask)
    fitted = a * pred_z + b
    if gp.convention.is_inverse:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(fitted > 1e-9, 1.0 / fitted, np.nan)
    return fitted


def _visibility_metrics(prob: np.ndarray, oracle_grid: np.ndarray) -> dict:
    mask = (oracle_grid == 0) | (oracle_grid == 1)
    truth = oracle_grid[mask] == 1
    guess = prob[mask] >= 0.5
    tp = int(np.sum(guess & truth)); tn = int(np.sum(~guess & ~truth))
    fp = int(np.sum(guess & ~truth)); fn = int(np.sum(~guess & truth))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n_cells": int(mask.sum()),
        "accuracy": float((tp + tn) / max(1, mask.sum())),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "visible_iou": float(tp / max(1, tp + fp + fn)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def evaluate(records: list[dict], models: tuple[str, ...]) -> dict:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    xs, ys, extent = _grid(manifest)
    drivable = _drivable()
    by_key = {(r["camera_id"], round(float(r["timestamp"]), 3)): r for r in records}
    result = {
        "status": "four_camera_mechanism_evidence",
        "dataset": str(DATASET.relative_to(REPO)),
        "frames": len(records),
        "cameras": list(CAM_LABELS),
        "times_s": list(TIMES),
        "commissioning_fit": "per camera and model at clear t=0.4 s, reused unchanged at t=1.2 s",
        "method_inputs": ["RGB", "fixed camera calibration", "planner 2-D traversable regions", "floor plane z=0"],
        "evaluation_only_inputs": ["co-located Gazebo optical-axis depth", "geometry-raycast visibility grid"],
        "visibility_target": "single point at z=0.35 m, exactly matching the oracle",
        "grid": manifest["grid"],
        "models": {},
    }
    all_maps = {}

    for model in models:
        model_rows, maps = [], {}
        fits = {}
        default_gate = {}
        for camera_id in CAM_LABELS:
            clear = by_key[(camera_id, 0.4)]
            clear_pred = _load_prediction(model, clear)
            fit, calib, anchors = _fit_clear_frame(clear_pred, clear, drivable)
            fits[camera_id] = (fit, calib, anchors)
            default_gate[camera_id] = (
                not clear_pred.convention.is_metric or 0.5 <= fit.scale <= 2.0
            )

        for record in records:
            camera_id, timestamp = record["camera_id"], round(float(record["timestamp"]), 3)
            pred = _load_prediction(model, record)
            fit, calib, anchors = fits[camera_id]
            depth, sigma, valid = _metric_from_fit(pred, record, fit, calib)

            # Evaluation begins here. The method above has not read either oracle.
            truth = np.load(DATASET / record["oracle_depth_path"])
            floor_depth = _full_floor_depth(calib)
            structure = (
                np.isfinite(truth) & (truth > 0)
                & (~np.isfinite(floor_depth) | (truth < floor_depth - 0.10))
            )
            floor_metrics = _depth_errors(depth, truth, structure & valid)
            raw_metrics = None
            if pred.convention.is_metric:
                raw_z = ga.to_optical_axis(pred.depth, pred.convention.value, calib)
                raw_metrics = _depth_errors(raw_z, truth, structure & pred.valid)
            oracle_depth = _oracle_affine(pred, truth, structure, calib)
            oracle_metrics = _depth_errors(oracle_depth, truth, structure & pred.valid)

            los = ga.line_of_sight_field(
                calib, depth, sigma, valid, xs, ys,
                plane=ga.FloorPlane(),
                target=ga.TargetVolume(radius_m=0.0, z_min_m=0.35, z_max_m=0.35,
                                       n_heights=1, n_ring=0),
            )
            oracle_grid = np.load(DATASET / record["oracle_visibility_grid"]["path"])
            vis_metrics = _visibility_metrics(los.p_visible, oracle_grid)
            key = f"{camera_id}@{timestamp:.1f}"
            maps[key] = {
                "p_visible": los.p_visible,
                "p_unknown": los.p_unknown,
                "in_fov": los.in_fov,
                "oracle": oracle_grid,
            }
            model_rows.append({
                "camera_id": camera_id,
                "camera_label": CAM_LABELS[camera_id],
                "timestamp_s": timestamp,
                "n_structure_pixels": int(structure.sum()),
                "raw": raw_metrics,
                "floor_anchored_clear_fit_reused": floor_metrics,
                "oracle_affine": oracle_metrics,
                "visibility": vis_metrics,
                "commissioning_fit": fit.to_dict(),
                "passes_default_metric_scale_gate": bool(default_gate[camera_id]),
                "n_floor_anchor_samples": len(anchors),
            })

        # Any-camera fusion is max probability because oracle "any" is a logical OR.
        fused = {}
        for timestamp in TIMES:
            cam_maps = [maps[f"{cam}@{timestamp:.1f}"] for cam in CAM_LABELS]
            p_any = np.max([m["p_visible"] for m in cam_maps], axis=0)
            rec = by_key[("external_camera", round(timestamp, 3))]
            oracle_any = np.load(DATASET / rec["oracle_visibility_grid"]["any_camera_path"])
            occupied = cam_maps[0]["oracle"] == 3
            oracle_for_score = np.where(occupied, 3, oracle_any)
            fused[f"{timestamp:.1f}"] = {
                "p_visible": p_any,
                "oracle": oracle_any,
                "occupied": occupied,
                "metrics": _visibility_metrics(p_any, oracle_for_score),
            }

        def median_metric(arm: str, field: str):
            vals = [r[arm][field] for r in model_rows if r[arm] and r[arm].get("n", 0)]
            return float(np.median(vals)) if vals else None

        result["models"][model] = {
            "label": MODEL_LABELS[model],
            "convention": _load_prediction(model, records[0]).convention.value,
            "n_frames": len(model_rows),
            "default_operational_metric_scale_gate": {
                "accepted_band": [0.5, 2.0],
                "per_camera": default_gate,
                "all_cameras_pass": bool(all(default_gate.values())),
                "reading": "relative/inverse models are exempt; metric models outside the band are refused by the default operational pipeline",
            },
            "aggregate_median_over_frames": {
                "raw_mae_m": median_metric("raw", "mae_m"),
                "floor_mae_m": median_metric("floor_anchored_clear_fit_reused", "mae_m"),
                "oracle_affine_mae_m": median_metric("oracle_affine", "mae_m"),
                "visibility_balanced_accuracy": float(np.median([r["visibility"]["balanced_accuracy"] for r in model_rows])),
                "visibility_visible_iou": float(np.median([r["visibility"]["visible_iou"] for r in model_rows])),
            },
            "per_frame": model_rows,
            "fused_any_camera": {t: v["metrics"] for t, v in fused.items()},
        }
        all_maps[model] = {"per_camera": maps, "fused": fused}

    # Use the lowest depth-MAE model for the supervisor map, and state that this is
    # evaluation-selected rather than a deployment-time choice.
    eligible = [
        m for m in models
        if result["models"][m]["default_operational_metric_scale_gate"]["all_cameras_pass"]
    ]
    chosen = min(eligible, key=lambda m: result["models"][m]["aggregate_median_over_frames"]["floor_mae_m"])
    result["map_model"] = chosen
    result["map_model_selection"] = (
        "lowest median four-camera structure-depth MAE among models passing the default "
        "commissioning scale gate on all cameras"
    )

    # Camera-A obstacle-shadow score: cells visible when clear and hidden after spawn.
    chosen_maps = all_maps[chosen]["per_camera"]
    c0, c1 = chosen_maps["external_camera@0.4"], chosen_maps["external_camera@1.2"]
    oracle_shadow = (c0["oracle"] == 1) & (c1["oracle"] == 0)
    predicted_shadow = (c0["p_visible"] >= 0.5) & (c1["p_visible"] < 0.5)
    tp = int(np.sum(oracle_shadow & predicted_shadow))
    result["camera_a_pallet_shadow"] = {
        "oracle_cells": int(oracle_shadow.sum()),
        "predicted_cells": int(predicted_shadow.sum()),
        "intersection_cells": tp,
        "recall": float(tp / max(1, oracle_shadow.sum())),
        "precision": float(tp / max(1, predicted_shadow.sum())),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    for model, map_data in all_maps.items():
        np.savez_compressed(
            MAP_DIR / f"visibility_maps__{model}.npz",
            xs=xs, ys=ys,
            **{
                f"{key.replace('@', '__').replace('.', 'p')}__{field}": value
                for key, blob in map_data["per_camera"].items()
                for field, value in blob.items()
            },
            **{
                f"fused__{t.replace('.', 'p')}__{field}": value
                for t, blob in map_data["fused"].items()
                for field, value in blob.items() if isinstance(value, np.ndarray)
            },
        )
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    make_plots(result, all_maps[chosen], xs, ys, extent)
    return result


def _style_map(ax, extent, title: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _categorical_method(blob: dict) -> np.ndarray:
    # 0 outside FOV, 1 occluded, 2 visible, 3 unknown
    out = np.zeros(blob["p_visible"].shape, dtype=int)
    out[blob["in_fov"]] = 1
    out[blob["in_fov"] & (blob["p_visible"] >= 0.5)] = 2
    out[blob["p_unknown"] > 0.5] = 3
    return out


def make_plots(result: dict, chosen_maps: dict, xs, ys, extent) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = list(result["models"])
    labels = [
        MODEL_LABELS[m] + ("\n(scale gate fail)" if not result["models"][m]
                           ["default_operational_metric_scale_gate"]["all_cameras_pass"] else "")
        for m in models
    ]
    cameras = list(CAM_LABELS)

    # Depth accuracy by camera, pooled across the clear and pallet frames only
    # through a median so the frame remains the statistical unit.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(models)); width = 0.35
    for ax, cam in zip(axes.flat, cameras):
        floor, oracle = [], []
        for model in models:
            rows = [r for r in result["models"][model]["per_frame"] if r["camera_id"] == cam]
            floor.append(np.median([r["floor_anchored_clear_fit_reused"]["mae_m"] for r in rows]))
            oracle.append(np.median([r["oracle_affine"]["mae_m"] for r in rows]))
        ax.bar(x - width / 2, floor, width, color=TEAL, label="Floor fit from clear frame")
        ax.bar(x + width / 2, oracle, width, color=ORACLE, label="Oracle affine")
        ax.axhline(0.3, color=ORANGE, linestyle="--", linewidth=1.5, label="0.3 m guide")
        ax.set_title(CAM_LABELS[cam], color=INK, fontweight="bold")
        ax.set_xticks(x, labels, rotation=15, ha="right", fontsize=8.5)
        ax.set_ylabel("structure-depth MAE [m]")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Four-camera depth test: clear-frame floor calibration reused after a pallet appears\n"
        "Median of two frames per camera · Gazebo depth used only for scoring · lower is better",
        fontsize=15, color=INK, fontweight="bold",
    )
    fig.savefig(FIG_DIR / "04_four_camera_depth_accuracy.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Per-camera event visibility: inferred depth-buffer cast vs geometry oracle.
    method_cmap = ListedColormap(["#ECEFF2", "#304C68", TEAL, "#F2C14E"])
    oracle_cmap = ListedColormap(["#304C68", TEAL, "#ECEFF2", "#252525"])
    fig, axes = plt.subplots(5, 3, figsize=(11.5, 17), constrained_layout=True)
    event_t = "1.2"
    for row, cam in enumerate(cameras):
        blob = chosen_maps["per_camera"][f"{cam}@1.2"]
        method_cls = _categorical_method(blob)
        oracle = blob["oracle"]
        compared = (oracle == 0) | (oracle == 1)
        disagreement = np.full(oracle.shape, np.nan)
        disagreement[compared] = ((method_cls[compared] == 2) != (oracle[compared] == 1)).astype(float)
        axes[row, 0].imshow(method_cls, origin="lower", extent=extent, cmap=method_cmap, vmin=0, vmax=3)
        axes[row, 1].imshow(oracle, origin="lower", extent=extent, cmap=oracle_cmap, vmin=0, vmax=3)
        axes[row, 2].imshow(disagreement, origin="lower", extent=extent,
                            cmap=ListedColormap(["#F4F7F9", "#D1495B"]), vmin=0, vmax=1)
        score = next(r["visibility"] for r in result["models"][result["map_model"]]["per_frame"]
                     if r["camera_id"] == cam and r["timestamp_s"] == 1.2)
        _style_map(axes[row, 0], extent, f"{CAM_LABELS[cam]} · depth raycast")
        _style_map(axes[row, 1], extent, f"{CAM_LABELS[cam]} · geometry oracle")
        _style_map(axes[row, 2], extent, f"disagreement · balanced acc. {score['balanced_accuracy']:.1%}")

    fused = chosen_maps["fused"][event_t]
    f_pred = fused["p_visible"] >= 0.5
    f_oracle = fused["oracle"] == 1
    standable = ~fused["occupied"]
    f_dis = np.full(f_pred.shape, np.nan)
    f_dis[standable] = (f_pred[standable] != f_oracle[standable]).astype(float)
    axes[4, 0].imshow(f_pred, origin="lower", extent=extent,
                      cmap=ListedColormap(["#304C68", TEAL]), vmin=0, vmax=1)
    axes[4, 1].imshow(f_oracle, origin="lower", extent=extent,
                      cmap=ListedColormap(["#304C68", TEAL]), vmin=0, vmax=1)
    axes[4, 2].imshow(f_dis, origin="lower", extent=extent,
                      cmap=ListedColormap(["#F4F7F9", "#D1495B"]), vmin=0, vmax=1)
    fused_score = result["models"][result["map_model"]]["fused_any_camera"][event_t]
    _style_map(axes[4, 0], extent, "A–D fused · depth raycast")
    _style_map(axes[4, 1], extent, "A–D fused · geometry oracle")
    _style_map(axes[4, 2], extent, f"disagreement · balanced acc. {fused_score['balanced_accuracy']:.1%}")
    fig.suptitle(
        f"Visibility/raycast map from monocular depth vs geometry oracle\n"
        f"Pallet present at t=1.2 s · {MODEL_LABELS[result['map_model']]} · target z=0.35 m",
        fontsize=15, color=INK, fontweight="bold",
    )
    fig.text(0.5, 0.002,
             "Teal = visible · navy = occluded · pale = outside FOV · yellow = unknown · red = disagreement. "
             "The floor affine was fitted at t=0.4 s and reused unchanged.",
             ha="center", fontsize=9, color=SOFT)
    fig.savefig(FIG_DIR / "05_four_camera_visibility_raycast.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Isolate the dynamic effect for Camera A, where the pallet was deliberately placed.
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3), constrained_layout=True)
    clear = chosen_maps["per_camera"]["external_camera@0.4"]
    event = chosen_maps["per_camera"]["external_camera@1.2"]
    for row, (name, a, b) in enumerate((
        ("Depth raycast", clear["p_visible"], event["p_visible"]),
        ("Geometry oracle", (clear["oracle"] == 1).astype(float), (event["oracle"] == 1).astype(float)),
    )):
        axes[row, 0].imshow(a, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].imshow(b, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1)
        loss = np.clip(a - b, 0, 1)
        axes[row, 2].imshow(loss, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1)
        _style_map(axes[row, 0], extent, f"{name} · clear t=0.4 s")
        _style_map(axes[row, 1], extent, f"{name} · pallet t=1.2 s")
        _style_map(axes[row, 2], extent, "lost visibility (clear − pallet)")
    shadow = result["camera_a_pallet_shadow"]
    fig.suptitle(
        "Camera A: does monocular depth reproduce the pallet's visibility shadow?\n"
        f"Oracle shadow {shadow['oracle_cells']} cells · detected {shadow['intersection_cells']} · "
        f"recall {shadow['recall']:.1%} · precision {shadow['precision']:.1%}",
        fontsize=15, color=INK, fontweight="bold",
    )
    fig.savefig(FIG_DIR / "06_camera_a_dynamic_shadow.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    records = _records()
    models = tuple(args.models)
    if not args.plots_only:
        run_inference(records, models, args.device)
    result = evaluate(records, models)
    print(json.dumps({
        model: result["models"][model]["aggregate_median_over_frames"] for model in models
    }, indent=2))
    print(f"map model: {result['map_model']}")
    print(f"wrote {OUT / 'results.json'} and figures 04–06")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
