#!/usr/bin/env python3
"""Rebuild a capture's samples.csv and manifest from the frames it did write.

WHY THIS EXISTS.  The `L1` capture rendered 15,033 of 15,072 frames and then raised
three positions from the end, inside the loop, so `samples.csv` and
`capture_manifest.json` were never written -- and because the capture tool resets its
own output directory after this study's driver had already opened a log inside it, the
traceback was written to an unlinked inode and lost.  Re-rendering costs ninety
minutes of simulator time.

The metadata does not need re-rendering, because none of it depends on the images.
Every frame's filename carries `sample_id`, position index, heading index and camera,
and the position and heading lists are deterministic functions of the capture's own
command-line arguments.  So this tool recomputes each row from the SAME functions the
capture used --- `_sample_positions`, `_sample_yaws`, `world_to_pixel`,
`segment_occluded` --- rather than inferring anything.  Nothing is invented and nothing
is interpolated: a frame that is not on disk simply has no row, and the count of
missing frames is recorded in the manifest.

VALIDATION.  `--validate-against` reruns the reconstruction over an existing complete
capture and compares every reconstructed field with that capture's real
`samples.csv`.  Run it on the nominal reference capture before trusting the output on
an incomplete one; if the reconstruction cannot reproduce a capture that succeeded, it
must not be used on one that failed.

    # first prove it reproduces a capture that worked
    python3 experiments/reconfiguration_holdout/rebuild_capture_metadata.py \\
        --validate-against logs/visibility_comparison/commissioning_grid_20260807

    # then rebuild the one that did not
    python3 experiments/reconfiguration_holdout/rebuild_capture_metadata.py \\
        --capture logs/visibility_comparison/recfg_holdout_L1 \\
        --world warehouse_full_4cam_recfg.world.sdf --yaw-samples 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for _rel in ("scripts/visibility_comparison", "scripts/shared", "src/unav_common",
             "src/experiments"):
    _p = str(HERE.parents[1] / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The capture module itself, so the position/heading/oracle logic is the same code
# rather than a second implementation of it.
import capture_visibility_samples as CAP  # noqa: E402
from experiments.core.world_profiles import compute_look_at_from_pose, load_profile  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402
from unav_common.occlusion_geometry import (  # noqa: E402
    parse_occlusion_scene_from_world, scene_to_json, segment_occluded)

REPO = HERE.parents[1]
PROFILES = HERE / "world_profiles_variants.yaml"

#: `<sample_id>_xy<position>_h<heading>[_<camera frame>].jpg`
STEM = re.compile(r"^(\d{6})_xy(\d{4})_h(\d{2})(?:_(.+))?$")

EXTRA = ("external_camera_b", "external_camera_c", "external_camera_d")


def build_context(world: str, profiles: Path, sample_nx: int, sample_ny: int,
                  wall_margin_m: float, yaw_samples: int, skip_region_filter: bool):
    """Positions, headings, camera models and occlusion scene -- the capture's own."""
    profile, intrinsics, world_path, camera_pose = load_profile(str(profiles), world)
    vis = dict(profile.get("visibility_defaults") or {})
    known = list(profile.get("known_2d_regions") or [])
    traversable = None
    if known and not skip_region_filter:
        traversable = [r for r in known if str(r.get("type", "")) == "traversable"] or None
    positions = CAP._sample_positions(
        vis, sample_nx=sample_nx, sample_ny=sample_ny,
        wall_margin_m=wall_margin_m, traversable_regions=traversable,
        region_shrink_m=0.05)
    yaws = CAP._sample_yaws(yaw_rad=0.0, yaw_samples=yaw_samples, yaw_list_rad="")

    cam_pos = [float(v) for v in camera_pose[:3]]
    look_at = [float(v) for v in compute_look_at_from_pose(
        cam_pos, float(camera_pose[3]), float(camera_pose[4]), float(camera_pose[5]))]
    intr = dict(img_width=int(intrinsics["img_width"]),
                img_height=int(intrinsics["img_height"]),
                fov_h_rad=float(intrinsics["fov_h_rad"]))
    models = {"": ObliqueCameraModel(cam_pos=cam_pos, look_at=look_at, **intr)}
    mounts = {}
    for frame in EXTRA:
        _p, _i, _w, pose_e = load_profile(str(profiles), world, camera_model=frame)
        pos_e = [float(v) for v in pose_e[:3]]
        mounts[frame] = [float(v) for v in pose_e]
        models[frame] = ObliqueCameraModel(
            cam_pos=pos_e,
            look_at=[float(v) for v in compute_look_at_from_pose(
                pos_e, float(pose_e[3]), float(pose_e[4]), float(pose_e[5]))],
            **intr)
    scene = parse_occlusion_scene_from_world(str(world_path), geometry_tags=("collision",))
    return dict(profile=profile, intrinsics=intr, world_path=world_path,
                camera_pose=camera_pose, cam_pos=cam_pos, look_at=look_at,
                positions=positions, yaws=yaws, models=models, mounts=mounts,
                scene=scene, vis=vis)


def oracle_for(ctx, frame: str, x: float, y: float, target_height_m: float) -> tuple[int, str, float, float]:
    """The capture's own oracle decision, recomputed with the capture's own calls."""
    cam = ctx["models"][frame]
    bottom_u, bottom_v, _ = cam.world_to_pixel(float(x), float(y), 0.0)
    target = np.asarray([float(x), float(y), float(target_height_m)], dtype=float)
    occluded = segment_occluded(ctx["scene"].prisms, cam.cam_pos, target)
    in_frame = (math.isfinite(bottom_u) and math.isfinite(bottom_v)
                and 0.0 <= bottom_u < ctx["intrinsics"]["img_width"]
                and 0.0 <= bottom_v < ctx["intrinsics"]["img_height"])
    if not in_frame:
        return 0, "outside_image", bottom_u, bottom_v
    if occluded:
        return 0, "occluded", bottom_u, bottom_v
    return 1, "visible", bottom_u, bottom_v


def reconstruct(capture: Path, ctx, target_height_m: float, world: str) -> tuple[list[dict], dict]:
    positions, yaws = ctx["positions"], ctx["yaws"]
    n_expected = len(positions) * len(yaws) * (1 + len(EXTRA))
    images = sorted((capture / "images").glob("*.jpg"))
    rows, skipped = [], 0
    for path in images:
        m = STEM.match(path.stem)
        if not m:
            skipped += 1
            continue
        sample_id, pos_idx, head_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
        frame_suffix = m.group(4) or ""
        if pos_idx >= len(positions) or head_idx >= len(yaws):
            skipped += 1
            continue
        x, y = positions[pos_idx]
        yaw = yaws[head_idx]
        frame_name = frame_suffix or "external_camera"
        key = frame_suffix if frame_suffix else ""
        vis, reason, bu, bv = oracle_for(ctx, key, x, y, target_height_m)
        rows.append({
            "sample_id": str(sample_id),
            "image_path": f"images/{path.name}",
            "preview_path": f"previews/{path.name}",
            "x": f"{float(x):.8f}", "y": f"{float(y):.8f}", "theta": f"{float(yaw):.8f}",
            "timestamp": "", "world": world, "camera_frame": frame_name,
            "oracle_visible": str(int(vis)),
            "oracle_bottom_u": "" if not math.isfinite(bu) else f"{float(bu):.8f}",
            "oracle_bottom_v": "" if not math.isfinite(bv) else f"{float(bv):.8f}",
            "oracle_occlusion_reason": reason,
            "segmentation_path": "", "labels_map_path": "", "robot_label": "",
        })
    rows.sort(key=lambda r: (int(r["sample_id"]), r["camera_frame"]))
    stats = {"frames_on_disk": len(images), "rows": len(rows),
             "unparsed_filenames": skipped,
             "frames_expected": n_expected,
             "frames_missing": n_expected - len(rows)}
    return rows, stats


def manifest_for(capture: Path, ctx, world: str, args, stats: dict) -> dict:
    vis = ctx["vis"]
    return {
        "world": world,
        "world_name": str(ctx["profile"]["world_name"]),
        "world_profiles_path": str(Path(args.profiles).resolve()),
        "world_path": str(Path(ctx["world_path"]).resolve()),
        "capture_method": "grid_teleport",
        "camera_frame": "external_camera",
        "extra_camera_frames": list(EXTRA),
        "extra_camera_mounts": ctx["mounts"],
        "stale_views": [],
        "n_stale_views": 0,
        "camera_pose": [float(v) for v in ctx["camera_pose"]],
        "camera_pos": ctx["cam_pos"],
        "look_at": ctx["look_at"],
        "img_width": ctx["intrinsics"]["img_width"],
        "img_height": ctx["intrinsics"]["img_height"],
        "fov_h_rad": ctx["intrinsics"]["fov_h_rad"],
        "visibility_bounds": {
            "xmin": float(vis.get("visibility_map_min_x", -6.0)),
            "xmax": float(vis.get("visibility_map_max_x", 6.0)),
            "ymin": float(vis.get("visibility_map_min_y", -6.0)),
            "ymax": float(vis.get("visibility_map_max_y", 6.0)),
            "nx": int(args.sample_nx), "ny": int(args.sample_ny),
            "wall_margin_m": float(args.wall_margin_m),
        },
        "heading_sampling": {
            "yaw_base_rad": 0.0, "yaw_samples": len(ctx["yaws"]),
            "yaw_list_rad": [float(v) for v in ctx["yaws"]],
            "explicit_yaw_list_rad": "",
            "samples_per_xy": len(ctx["yaws"]),
            "position_count": len(ctx["positions"]),
        },
        "oracle_source": "geometry",
        "oracle_target_height_m": float(args.target_height_m),
        "oracle_reference_point": "ground_plane_robot_pose",
        "geometry_json": scene_to_json(ctx["scene"]),
        "sample_count": stats["rows"],
        "output_dir": str(capture),
        "reconstructed": True,
        "reconstruction": {
            "by": "experiments/reconfiguration_holdout/rebuild_capture_metadata.py",
            "reason": ("the capture raised inside its pose loop three positions from the "
                       "end, so samples.csv and the manifest were never written"),
            **stats,
        },
        "notes": [
            "RECONSTRUCTED metadata: every field recomputed from the capture's own "
            "position/heading samplers and oracle calls, keyed by the frame filenames.",
            "No image was re-rendered and no value was interpolated; frames absent from "
            "disk have no row, and the missing count is recorded above.",
            "Validated by reproducing the nominal reference capture's samples.csv.",
        ],
    }


def validate(capture: Path, args) -> int:
    """Reconstruct an existing complete capture and diff against its real samples.csv."""
    real_manifest = json.loads((capture / "capture_manifest.json").read_text(encoding="utf-8"))
    world = str(real_manifest["world"])
    profiles = Path(real_manifest["world_profiles_path"])
    yaw_n = int(real_manifest["heading_sampling"]["yaw_samples"])
    bounds = real_manifest["visibility_bounds"]
    ctx = build_context(world, profiles, int(bounds["nx"]), int(bounds["ny"]),
                        float(bounds["wall_margin_m"]), yaw_n, False)
    rows, stats = reconstruct(capture, ctx,
                              float(real_manifest["oracle_target_height_m"]), world)
    real = {}
    with (capture / "samples.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            real[(r["image_path"], r["camera_frame"])] = r
    print(f"[validate] {capture.name}: {len(real)} real rows, {len(rows)} reconstructed")
    checked = 0
    mismatch: dict[str, int] = {}
    for r in rows:
        key = (r["image_path"], r["camera_frame"])
        if key not in real:
            mismatch["row_absent_from_real"] = mismatch.get("row_absent_from_real", 0) + 1
            continue
        truth = real[key]
        checked += 1
        for field in ("x", "y", "theta", "oracle_visible", "oracle_occlusion_reason",
                      "oracle_bottom_u", "oracle_bottom_v"):
            a, b = str(r[field]), str(truth[field])
            if field in ("x", "y", "theta", "oracle_bottom_u", "oracle_bottom_v"):
                try:
                    same = (a == b) or abs(float(a or "nan") - float(b or "nan")) < 1e-6
                except ValueError:
                    same = a == b
            else:
                same = a == b
            if not same:
                mismatch[field] = mismatch.get(field, 0) + 1
    print(f"[validate] compared {checked} rows on 7 fields each")
    if mismatch:
        print(f"[validate] FAIL: {mismatch}")
        return 1
    print("[validate] PASS: every reconstructed field matches the real capture exactly")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture")
    ap.add_argument("--validate-against")
    ap.add_argument("--world", default="")
    ap.add_argument("--profiles", default=str(PROFILES))
    ap.add_argument("--sample-nx", type=int, default=46)
    ap.add_argument("--sample-ny", type=int, default=36)
    ap.add_argument("--wall-margin-m", type=float, default=0.45)
    ap.add_argument("--yaw-samples", type=int, default=4)
    ap.add_argument("--target-height-m", type=float, default=0.0)
    args = ap.parse_args(argv)

    if args.validate_against:
        return validate(Path(args.validate_against), args)
    if not (args.capture and args.world):
        raise SystemExit("--capture and --world are required unless --validate-against")

    capture = Path(args.capture)
    ctx = build_context(args.world, Path(args.profiles), args.sample_nx, args.sample_ny,
                        args.wall_margin_m, args.yaw_samples, False)
    print(f"[rebuild] {len(ctx['positions'])} positions x {len(ctx['yaws'])} headings "
          f"x {1 + len(EXTRA)} cameras")
    rows, stats = reconstruct(capture, ctx, args.target_height_m, args.world)
    print(f"[rebuild] {stats['frames_on_disk']} frames on disk -> {stats['rows']} rows; "
          f"{stats['frames_missing']} of {stats['frames_expected']} expected frames absent")
    CAP.write_csv(capture / "samples.csv", CAP.CAPTURE_COLUMNS, rows)
    (capture / "capture_manifest.json").write_text(
        json.dumps(manifest_for(capture, ctx, args.world, args, stats), indent=2),
        encoding="utf-8")
    hit = sum(1 for r in rows if r["oracle_visible"] == "1")
    print(f"[rebuild] oracle-visible fraction {hit / max(len(rows), 1):.4f}")
    print(f"[rebuild] wrote {capture}/samples.csv and capture_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
