#!/usr/bin/env python3
"""Audit camera failure frames with consistent timestamp and calibration semantics.

For each selected frame this plots:
  * red bbox and selected detector pixel from the detection diagnostic at
    ``diag_stamp``;
  * green projected truth footprint interpolated from ``experiment.csv`` at the
    same frame-capture stamp;
  * separate numeric labels for detector-at-capture error and the legacy logged
    ``localization_error_calibrated_m`` value.

The logged localization-error column is intentionally not treated as detector
accuracy: it compares the latest pixel pose, which can be stale, to truth at
``log_stamp``.  This script writes a JSON sidecar next to the figure so every
panel can be traced to its run row, frame file, timestamps, and metric values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"
DEFAULT_WORLD = "warehouse_aws.world.sdf"


def finite_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def parse_frame_file(path: Path) -> tuple[float, str]:
    parts = path.stem.split("_")
    try:
        stamp = float(parts[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse frame stamp from {path}") from exc
    prefix = "_".join(parts[:-1]) if len(parts) > 1 else ""
    return stamp, prefix


def collect_frame_files(frame_dir: Path, *, allow_mixed_prefixes: bool = False) -> list[dict]:
    raw_dir = frame_dir / "raw"
    paths = sorted(raw_dir.glob("*.png")) if raw_dir.is_dir() else []
    if not paths:
        paths = sorted(frame_dir.glob("*.png"))
    frames = []
    for path in paths:
        try:
            stamp, prefix = parse_frame_file(path)
        except ValueError:
            continue
        frames.append({"stamp": stamp, "prefix": prefix, "path": str(path)})
    if not frames:
        raise FileNotFoundError(f"No parseable PNG frames found in {frame_dir}")
    prefixes = {f["prefix"] for f in frames}
    if len(prefixes) > 1 and not allow_mixed_prefixes:
        shown = ", ".join(sorted(prefixes)[:6])
        raise ValueError(
            f"Frame directory mixes {len(prefixes)} capture prefixes ({shown}). "
            "Pass --allow-mixed-frame-prefixes only if those frames are known to be from the same run."
        )
    return sorted(frames, key=lambda f: f["stamp"])


def nearest_frame(frames: list[dict], stamp: float, max_dt: float) -> dict | None:
    if not frames or not math.isfinite(stamp):
        return None
    best = min(frames, key=lambda f: abs(float(f["stamp"]) - stamp))
    dt = abs(float(best["stamp"]) - stamp)
    if dt > max_dt:
        return None
    out = dict(best)
    out["dt_s"] = dt
    return out


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def campaign_yaml_for_run(run_dir: Path) -> Path | None:
    parts = run_dir.resolve().parts
    if "visibility_comparison" not in parts:
        return None
    idx = parts.index("visibility_comparison")
    if idx + 1 >= len(parts):
        return None
    campaign = parts[idx + 1]
    path = ROOT / "scripts" / "visibility_comparison" / f"{campaign}.yaml"
    return path if path.is_file() else None


def load_campaign_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def parse_affine(raw: object) -> list[float] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        values = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        return None
    return values if len(values) == 6 else None


def calibration_from_sources(
    manifest: dict,
    campaign_cfg: dict,
    *,
    affine_override: str | None,
    y_offset_override: float | None,
    bbox_contact_z_override: float | None,
) -> dict:
    source = "run_manifest"
    affine_raw = str(manifest.get("bev_affine_calibration", "") or "").strip()
    y_offset = finite_float(manifest, "bev_y_calibration_offset_m", 0.0)
    bbox_contact_z = finite_float(manifest, "bbox_contact_z_m", 0.0)

    if not affine_raw and campaign_cfg:
        affine_raw = str(campaign_cfg.get("bev_affine_calibration", "") or "").strip()
        if affine_raw:
            source = "campaign_yaml"
    if (not math.isfinite(y_offset) or y_offset == 0.0) and campaign_cfg:
        try:
            y_offset = float(campaign_cfg.get("bev_y_calibration_offset_m", y_offset))
            source = "campaign_yaml" if source == "run_manifest" else source
        except (TypeError, ValueError):
            pass
    if (not math.isfinite(bbox_contact_z) or bbox_contact_z == 0.0) and campaign_cfg:
        try:
            bbox_contact_z = float(campaign_cfg.get("bbox_contact_z_m", bbox_contact_z))
        except (TypeError, ValueError):
            pass

    if affine_override is not None:
        affine_raw = affine_override
        source = "cli"
    if y_offset_override is not None:
        y_offset = float(y_offset_override)
        source = "cli"
    if bbox_contact_z_override is not None:
        bbox_contact_z = float(bbox_contact_z_override)
        source = "cli"

    return {
        "source": source,
        "bev_y_calibration_offset_m": float(y_offset if math.isfinite(y_offset) else 0.0),
        "bev_affine_calibration": affine_raw,
        "bev_affine_values": parse_affine(affine_raw),
        "bbox_contact_z_m": float(bbox_contact_z if math.isfinite(bbox_contact_z) else 0.0),
    }


def apply_bev_calibration(x: float, y: float, calibration: dict) -> tuple[float, float]:
    affine = calibration.get("bev_affine_values")
    if affine is not None:
        return (
            affine[0] * x + affine[1] * y + affine[2],
            affine[3] * x + affine[4] * y + affine[5],
        )
    return x, y + float(calibration.get("bev_y_calibration_offset_m", 0.0))


def load_camera(run_dir: Path, manifest: dict):
    from experiments.core.world_profiles import compute_look_at_from_pose, load_profile
    from unav_common.camera_model import ObliqueCameraModel

    profile_path = run_dir / "world_profiles.yaml"
    if not profile_path.is_file():
        profile_path = DEFAULT_PROFILES
    world = str(manifest.get("world") or DEFAULT_WORLD)
    _profile, intr, _world_path, cp = load_profile(str(profile_path), world)
    cam_pos = [float(cp[0]), float(cp[1]), float(cp[2])]
    look_at = compute_look_at_from_pose(cam_pos, float(cp[3]), float(cp[4]), float(cp[5]))
    camera = ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=int(intr["img_width"]),
        img_height=int(intr["img_height"]),
        fov_h_rad=float(intr["fov_h_rad"]),
    )
    return camera, {
        "world": world,
        "world_profiles": str(profile_path),
        "cam_pose_xyz_rpy": [float(v) for v in cp],
        "cam_pos": cam_pos,
        "look_at": [float(v) for v in look_at],
        "img_width": int(intr["img_width"]),
        "img_height": int(intr["img_height"]),
        "fov_h_rad": float(intr["fov_h_rad"]),
    }


def truth_trajectory(exp_rows: list[dict]) -> np.ndarray:
    values = []
    for row in exp_rows:
        stamp = finite_float(row, "stamp")
        x = finite_float(row, "odom_map_x")
        y = finite_float(row, "odom_map_y")
        if math.isfinite(stamp) and math.isfinite(x) and math.isfinite(y):
            values.append((stamp, x, y))
    if not values:
        raise ValueError("experiment.csv has no finite truth trajectory")
    return np.asarray(values, dtype=float)


def truth_at(traj: np.ndarray, stamp: float) -> tuple[float, float, bool]:
    clipped = bool(stamp < traj[0, 0] or stamp > traj[-1, 0])
    x = float(np.interp(stamp, traj[:, 0], traj[:, 1]))
    y = float(np.interp(stamp, traj[:, 0], traj[:, 2]))
    return x, y, clipped


def pixel_to_world_for_detection(camera, u: float, v: float, calibration: dict) -> tuple[float, float] | None:
    z = float(calibration.get("bbox_contact_z_m", 0.0))
    if z > 0.0:
        world = camera.pixel_to_world_at_z(u, v, z)
    else:
        world = camera.pixel_to_world(u, v)
    if world is None:
        return None
    return apply_bev_calibration(float(world[0]), float(world[1]), calibration)


def build_candidates(
    per_rows: list[dict],
    truth: np.ndarray,
    frames: list[dict],
    camera,
    calibration: dict,
    *,
    max_frame_dt: float,
) -> list[dict]:
    candidates = []
    for row_idx, row in enumerate(per_rows):
        detected = finite_float(row, "yolo_detected_after_threshold")
        if not math.isfinite(detected):
            detected = finite_float(row, "detected")
        if detected < 0.5:
            continue

        diag_stamp = finite_float(row, "diag_stamp")
        obs_u = finite_float(row, "obs_u")
        obs_v = finite_float(row, "obs_v")
        if not (math.isfinite(diag_stamp) and math.isfinite(obs_u) and math.isfinite(obs_v)):
            continue

        frame = nearest_frame(frames, diag_stamp, max_frame_dt)
        if frame is None:
            continue

        det_world = pixel_to_world_for_detection(camera, obs_u, obs_v, calibration)
        if det_world is None:
            continue
        odom_map_x, odom_map_y, truth_clipped = truth_at(truth, diag_stamp)
        detector_capture_error_m = math.hypot(det_world[0] - odom_map_x, det_world[1] - odom_map_y)

        log_stamp = finite_float(row, "log_stamp")
        pixel_pose_stamp = finite_float(row, "pixel_pose_stamp")
        pixel_pose_age_s = (
            max(log_stamp - pixel_pose_stamp, 0.0)
            if math.isfinite(log_stamp) and math.isfinite(pixel_pose_stamp)
            else math.nan
        )
        source_stamp_mismatch_s = (
            abs(pixel_pose_stamp - diag_stamp)
            if math.isfinite(pixel_pose_stamp)
            else math.nan
        )

        raw_world = camera.pixel_to_world(obs_u, obs_v)
        if raw_world is not None:
            ru, rv, _ = camera.world_to_pixel(raw_world[0], raw_world[1], 0.0)
            roundtrip_px_error = math.hypot(ru - obs_u, rv - obs_v)
        else:
            roundtrip_px_error = math.nan

        candidates.append({
            "row_index": row_idx,
            "frame": frame,
            "diag_stamp": diag_stamp,
            "log_stamp": log_stamp,
            "pixel_pose_stamp": pixel_pose_stamp,
            "pixel_pose_age_s": pixel_pose_age_s,
            "source_stamp_mismatch_s": source_stamp_mismatch_s,
            "obs_u": obs_u,
            "obs_v": obs_v,
            "bbox_xmin": finite_float(row, "bbox_xmin"),
            "bbox_ymin": finite_float(row, "bbox_ymin"),
            "bbox_xmax": finite_float(row, "bbox_xmax"),
            "bbox_ymax": finite_float(row, "bbox_ymax"),
            "det_world_x": float(det_world[0]),
            "det_world_y": float(det_world[1]),
            "truth_capture_x": odom_map_x,
            "truth_capture_y": odom_map_y,
            "truth_capture_extrapolated": truth_clipped,
            "detector_capture_error_m": detector_capture_error_m,
            "logger_localization_error_calibrated_m": finite_float(
                row, "localization_error_calibrated_m"
            ),
            "roundtrip_px_error": roundtrip_px_error,
            "yolo_inference_ms": finite_float(row, "yolo_inference_ms"),
            "detector_total_latency_s": finite_float(row, "detector_total_latency_s"),
            "frame_age_at_publish_s": finite_float(row, "frame_age_at_publish_s"),
            "selected_pixel_source": str(row.get("yolo_selected_pixel_source") or ""),
        })
    return candidates


def metric_value(candidate: dict, key: str) -> float:
    value = candidate.get(key, math.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return value if math.isfinite(value) else -math.inf


def draw_figure(picks: list[dict], out_path: Path, camera, sort_by: str) -> None:
    ncol = min(4, max(1, len(picks)))
    nrow = (len(picks) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow))
    axs = np.atleast_1d(axs).ravel()

    for ax, candidate in zip(axs, picks):
        ax.imshow(mpimg.imread(candidate["frame"]["path"]))
        x0 = candidate["bbox_xmin"]
        y0 = candidate["bbox_ymin"]
        x1 = candidate["bbox_xmax"]
        y1 = candidate["bbox_ymax"]
        if all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=1.7))
        ax.plot([candidate["obs_u"]], [candidate["obs_v"]], "x", color="red", ms=10, mew=2.2)

        tx = candidate["truth_capture_x"]
        ty = candidate["truth_capture_y"]
        tu, tv, _ = camera.world_to_pixel(tx, ty, 0.0)
        footprint = [
            camera.world_to_pixel(tx + 0.09 * math.cos(a), ty + 0.09 * math.sin(a), 0.0)
            for a in np.linspace(0.0, 2.0 * math.pi, 40)
        ]
        ax.plot([p[0] for p in footprint], [p[1] for p in footprint], "-", color="lime", lw=1.8)
        ax.plot([tu], [tv], "+", color="lime", ms=12, mew=2.2)

        title = (
            f"det@cap {candidate['detector_capture_error_m']:.2f} m | "
            f"logged {candidate['logger_localization_error_calibrated_m']:.2f} m\n"
            f"age {candidate['pixel_pose_age_s']:.2f}s | src dt {candidate['source_stamp_mismatch_s']:.2f}s"
        )
        ax.set_title(title, fontsize=8.5)

        if math.isfinite(tu) and math.isfinite(tv):
            cu = 0.5 * (candidate["obs_u"] + tu)
            cv = 0.5 * (candidate["obs_v"] + tv)
            ax.set_xlim(cu - 100, cu + 100)
            ax.set_ylim(cv + 82, cv - 82)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[len(picks):]:
        ax.axis("off")

    fig.legend(handles=[
        Line2D([0], [0], marker="x", color="red", lw=0, ms=10, mew=2, label="detector selected pixel"),
        Line2D([0], [0], color="red", lw=1.7, label="detected bbox"),
        Line2D([0], [0], marker="+", color="lime", lw=0, ms=12, mew=2, label="truth @ frame capture (+footprint)"),
    ], loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"Failure-frame audit sorted by {sort_by}: red is detector on that frame; "
        "green is capture-time truth. Logged error is a separate stale/log-time metric.",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def write_provenance(
    out_path: Path,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    frame_dir: Path,
    camera_info: dict,
    calibration: dict,
    candidates: list[dict],
    picks: list[dict],
) -> Path:
    def summarize(key: str) -> dict:
        vals = []
        for candidate in candidates:
            value = metric_value(candidate, key)
            if math.isfinite(value):
                vals.append(value)
        if not vals:
            return {"n": 0}
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
        }

    def clean_candidate(candidate: dict) -> dict:
        return {
            key: value
            for key, value in candidate.items()
            if key != "frame"
        } | {
            "frame_path": candidate["frame"]["path"],
            "frame_stamp": candidate["frame"]["stamp"],
            "frame_dt_s": candidate["frame"]["dt_s"],
            "frame_prefix": candidate["frame"]["prefix"],
        }

    sidecar = out_path.with_suffix(".provenance.json")
    payload = {
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "frame_dir": str(frame_dir),
        "output": str(out_path),
        "sort_by": args.sort_by,
        "n_requested": args.n,
        "max_frame_dt_s": args.max_frame_dt,
        "candidate_count": len(candidates),
        "candidate_summary": {
            "detector_capture_error_m": summarize("detector_capture_error_m"),
            "logger_localization_error_calibrated_m": summarize("logger_localization_error_calibrated_m"),
            "pixel_pose_age_s": summarize("pixel_pose_age_s"),
            "source_stamp_mismatch_s": summarize("source_stamp_mismatch_s"),
        },
        "camera": camera_info,
        "calibration": calibration,
        "metric_definitions": {
            "detector_capture_error_m": "distance between calibrated detector selected pixel back-projection and truth interpolated at diag_stamp",
            "logger_localization_error_calibrated_m": "legacy perception.csv value: latest pixel_pose back-projection compared to truth at log_stamp",
            "source_stamp_mismatch_s": "abs(pixel_pose_stamp - diag_stamp); nonzero means logged error and drawn bbox are different measurements",
        },
        "selected": [clean_candidate(c) for c in picks],
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("out", nargs="?", type=Path, default=Path("/tmp/failure_frames.png"))
    parser.add_argument("n", nargs="?", type=int, default=8)
    parser.add_argument(
        "--sort-by",
        choices=[
            "detector_capture_error_m",
            "logger_localization_error_calibrated_m",
            "pixel_pose_age_s",
            "source_stamp_mismatch_s",
        ],
        default="detector_capture_error_m",
    )
    parser.add_argument("--max-frame-dt", type=float, default=0.18)
    parser.add_argument("--allow-mixed-frame-prefixes", action="store_true")
    parser.add_argument("--bev-affine", default=None)
    parser.add_argument("--bev-y-offset", type=float, default=None)
    parser.add_argument("--bbox-contact-z", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_dir = args.run_dir.resolve()
    frame_dir = args.frame_dir.resolve()
    out_path = args.out.resolve()

    manifest = load_manifest(run_dir)
    campaign_yaml = campaign_yaml_for_run(run_dir)
    campaign_cfg = load_campaign_yaml(campaign_yaml)
    calibration = calibration_from_sources(
        manifest,
        campaign_cfg,
        affine_override=args.bev_affine,
        y_offset_override=args.bev_y_offset,
        bbox_contact_z_override=args.bbox_contact_z,
    )
    camera, camera_info = load_camera(run_dir, manifest)
    exp_rows = load_rows(run_dir / "experiment.csv")
    per_rows = load_rows(run_dir / "perception.csv")
    truth = truth_trajectory(exp_rows)
    frames = collect_frame_files(frame_dir, allow_mixed_prefixes=args.allow_mixed_frame_prefixes)
    candidates = build_candidates(
        per_rows,
        truth,
        frames,
        camera,
        calibration,
        max_frame_dt=args.max_frame_dt,
    )
    if not candidates:
        raise RuntimeError("No detected perception rows had matching frames and finite truth")
    picks = sorted(candidates, key=lambda c: metric_value(c, args.sort_by), reverse=True)[:args.n]

    draw_figure(picks, out_path, camera, args.sort_by)
    sidecar = write_provenance(
        out_path,
        args=args,
        run_dir=run_dir,
        frame_dir=frame_dir,
        camera_info=camera_info,
        calibration=calibration,
        candidates=candidates,
        picks=picks,
    )
    print(f"frames available: {len(frames)}; candidates: {len(candidates)}; showing: {len(picks)}")
    print(f"calibration source: {calibration['source']}; affine active: {calibration['bev_affine_values'] is not None}")
    print(f"wrote {out_path}")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
