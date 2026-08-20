#!/usr/bin/env python3
"""Monocular-depth availability field for one environment, from the cameras' own RGB.

This is the adaptive arm.  For each of the four cameras it takes ONE frame the camera
rendered in that environment, predicts monocular depth, anchors that depth on the
floor the calibration already knows, and asks for every candidate robot position
whether anything now stands between it and the camera.  No surveyed 3-D model, no
depth sensor, no ground truth.

**The floor anchor is commissioned once, in `L0`, and reused unchanged everywhere
else.** That is the whole claim being tested: a deployed camera fits its floor scale
when it is installed and does not get to re-commission because somebody restocked a
rack or turned a lamp off.  `--env L0` fits and saves the anchor; every other
environment loads it and only the depth prediction is recomputed from the new frame.

The scene frame per camera is chosen deterministically as the capture sample whose
commanded position is farthest from that camera, so the robot is small and peripheral
rather than standing in the middle of the scene the depth model has to reconstruct.
It is not removed: a deployed camera does not get a robot-free frame on demand, and
using one would quietly make the arm easier than it is.

    python3 experiments/reconfiguration_holdout/mono_depth_field.py --env L1

Writes logs/studies/reconfiguration_holdout/mono_depth/<env>_visibility.npz with
per-camera p_visible / p_unknown / in_fov on the study's working grid, plus
<env>_fit.json recording the anchor fit actually used and where it came from.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
for _rel in ("experiments/mono_depth_visibility", "experiments/monocular_depth_adapter",
             "scripts/shared"):
    _p = str(HERE.parents[1] / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(HERE))

import ground_anchoring as ga  # noqa: E402
from monodepth import CameraIntrinsics, DepthRequest, MonocularDepthAdapter  # noqa: E402
import common as C  # noqa: E402
import yaml  # noqa: E402

#: The availability study's selected monocular model: UniDepthV2 ViT-S won on
#: floor-anchored depth MAE (0.247 m against 0.327 / 0.337 / 0.420 m).  Kept as the
#: default rather than re-chosen, so the two studies share one depth backbone.
DEFAULT_MODEL = "unidepth_v2_vits14"

OUT_DIR = C.OUT_ROOT / "mono_depth"

#: Camera mount poses, as the world SDF declares them: (x, y, z, roll, pitch, yaw).
#: Read from the capture manifest rather than restated, so a camera cannot silently
#: differ between the field and the frames it was computed from.


def scene_frames(env: C.Environment) -> dict[str, dict]:
    """One frame per camera, plus that camera's calibration, from the capture record."""
    manifest = json.loads((env.capture / "capture_manifest.json").read_text(encoding="utf-8"))
    width, height = int(manifest["img_width"]), int(manifest["img_height"])
    fov = float(manifest["fov_h_rad"])
    f = (width / 2.0) / math.tan(fov / 2.0)
    intrinsics = {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0,
                  "img_width": width, "img_height": height}

    from experiments.core.world_profiles import compute_look_at_from_pose
    extr: dict[str, dict] = {
        str(manifest["camera_frame"]): {
            "cam_pos": [float(v) for v in manifest["camera_pos"]],
            "look_at": [float(v) for v in manifest["look_at"]],
        }
    }
    for frame, pose in (manifest.get("extra_camera_mounts") or {}).items():
        pos = [float(v) for v in pose[:3]]
        extr[str(frame)] = {
            "cam_pos": pos,
            "look_at": [float(v) for v in compute_look_at_from_pose(
                pos, float(pose[3]), float(pose[4]), float(pose[5]))],
        }

    best: dict[str, tuple[float, str, float, float]] = {}
    with (env.capture / "samples.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frame = str(row.get("camera_frame") or manifest["camera_frame"])
            if frame not in extr:
                continue
            x, y = float(row["x"]), float(row["y"])
            cx, cy = extr[frame]["cam_pos"][0], extr[frame]["cam_pos"][1]
            d = math.hypot(x - cx, y - cy)
            if frame not in best or d > best[frame][0]:
                best[frame] = (d, str(row["image_path"]), x, y)

    out = {}
    for frame, (d, img, x, y) in sorted(best.items()):
        out[frame] = {
            "camera_id": frame,
            "camera_intrinsics": dict(intrinsics),
            "camera_extrinsics": dict(extr[frame]),
            "rgb_path": img,
            "robot_xy": [x, y],
            "robot_range_m": d,
        }
    missing = [c for c in C.CAMERAS if c not in out]
    if missing:
        raise RuntimeError(f"{env.key}: no scene frame found for {missing}")
    return out


def drivable_footprints() -> list:
    """The planner's own 2-D traversable regions -- a deployment input, not a survey."""
    profile = yaml.safe_load(
        (C.REPO / "src/experiments/config/world_profiles.yaml").read_text(encoding="utf-8"))
    regions = profile["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    return [ga.Footprint(xmin=float(r["xmin"]), xmax=float(r["xmax"]),
                         ymin=float(r["ymin"]), ymax=float(r["ymax"]), name=str(r["name"]))
            for r in regions if r.get("type") == "traversable"]


def predict_depth(frames: dict[str, dict], capture: Path, model: str, device: str) -> dict:
    """RGB-only monocular depth for each camera's scene frame."""
    requests, order = [], []
    for frame, rec in frames.items():
        rgb = np.asarray(Image.open(capture / rec["rgb_path"]).convert("RGB"), dtype=np.uint8)
        k = rec["camera_intrinsics"]
        requests.append(DepthRequest(
            f"{frame}", rgb,
            CameraIntrinsics(fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"],
                             width=k["img_width"], height=k["img_height"]),
            source_path=str((capture / rec["rgb_path"]).relative_to(C.REPO)),
        ))
        order.append(frame)
    with MonocularDepthAdapter(model, device=device, batch_size=1, uncertainty="none") as adapter:
        preds = list(adapter.predict(requests))
    return dict(zip(order, preds))


def to_ga(pred, frame: str) -> ga.DepthPrediction:
    return ga.DepthPrediction(
        values=pred.depth, convention=pred.convention.value, valid_mask=pred.valid,
        uncertainty=None, model_name=pred.model.model_name, checkpoint=pred.model.checkpoint,
        inference_time_s=pred.timing.forward_s, frame_id=pred.image_id, camera_id=frame,
    )


def fit_anchor(pred, record: dict, drivable: list):
    """Robust affine fit of predicted depth onto analytically known floor depth."""
    calib = ga.camera_from_record(record)
    gp = to_ga(pred, record["camera_id"])
    anchors = ga.select_floor_anchors(
        calib, ga.FloorPlane(), drivable,
        config=ga.AnchorConfig(pixel_step=4, require_drivable=True),
        valid_mask=gp.valid_mask,
    )
    anchor_z = ga.to_optical_axis(
        gp.values[anchors.v.astype(int), anchors.u.astype(int)], gp.convention, calib,
        u=anchors.u, v=anchors.v,
    )
    fit = ga.fit_ground_affine(
        anchor_z, anchors.depth_m, gp.convention,
        config=ga.FitConfig(strict_convention=True, metric_scale_band=(0.05, 10.0)),
        anchor_depth_span_m=anchors.depth_span_m,
        notes="reconfiguration-holdout commissioning fit on L0; reused unchanged elsewhere",
    )
    if not fit.status.is_ok:
        raise RuntimeError(f"{record['camera_id']}: anchor fit failed: "
                           f"{fit.status.value}: {fit.notes}")
    return fit, calib, len(anchors)


def apply_saved_fit(saved: dict, pred_z: np.ndarray) -> np.ndarray:
    """Reapply a commissioned affine fit, reproducing GroundFit.apply exactly.

    Written out rather than reconstructing a ``GroundFit`` because that object carries
    status enums and a covariance matrix that do not round-trip through JSON.  The
    ``inverse_depth`` branch matters: a relative-depth model is fitted in inverse
    space, and applying the affine in the wrong space silently produces a depth map
    that is smooth, plausible and wrong.
    """
    y = float(saved["scale"]) * np.asarray(pred_z, dtype=float) + float(saved["shift"])
    if str(saved["fit_space"]) == "inverse_depth":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y > 1e-9, 1.0 / y, np.nan)
    return y


def metric_depth(pred, calib, fit):
    gp = to_ga(pred, "")
    pred_z = ga.to_optical_axis(gp.values, gp.convention, calib)
    depth = fit.apply(pred_z)
    sigma = ga.predicted_depth_sigma(fit, pred_z, depth, None)
    valid = (gp.valid_mask & np.isfinite(depth) & np.isfinite(sigma)
             & (depth > 0) & (depth < 60))
    return depth, sigma, valid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=[e.key for e in C.ENVIRONMENTS])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    env = C.ENV_BY_KEY[args.env]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xs, ys = C.working_grid()
    drivable = drivable_footprints()

    frames = scene_frames(env)
    print(f"[mono {env.key}] scene frames: " + ", ".join(
        f"{C.SHORT[f]}={Path(r['rgb_path']).name} (robot {r['robot_range_m']:.1f} m away)"
        for f, r in frames.items()))

    preds = predict_depth(frames, env.capture, args.model, args.device)

    anchor_path = OUT_DIR / f"anchor_{C.DEVELOPMENT_ENV}_{args.model}.json"
    reuse = env.key != C.DEVELOPMENT_ENV
    if reuse and not anchor_path.is_file():
        raise RuntimeError(
            f"{env.key} must reuse the {C.DEVELOPMENT_ENV} anchor fit, but "
            f"{anchor_path} does not exist. Run --env {C.DEVELOPMENT_ENV} first.")
    saved = json.loads(anchor_path.read_text(encoding="utf-8")) if reuse else {}

    maps: dict[str, np.ndarray] = {"xs": xs, "ys": ys}
    record_out: dict[str, dict] = {}
    for frame in C.CAMERAS:
        rec, pred = frames[frame], preds[frame]
        calib = ga.camera_from_record(rec)
        if reuse:
            f = saved["cameras"][frame]
            gp = to_ga(pred, frame)
            pred_z = ga.to_optical_axis(gp.values, gp.convention, calib)
            depth = apply_saved_fit(f, pred_z)
            sigma = np.full_like(depth, float(f["sigma_fit"]))
            valid = (gp.valid_mask & np.isfinite(depth) & (depth > 0) & (depth < 60))
            scale, shift, n_anchor = float(f["scale"]), float(f["shift"]), int(f["n_anchor"])
        else:
            fit, calib, n_anchor = fit_anchor(pred, rec, drivable)
            depth, sigma, valid = metric_depth(pred, calib, fit)
            scale, shift = float(fit.scale), float(fit.shift)
            record_out[frame] = {
                "scale": scale, "shift": shift, "fit_space": str(fit.fit_space),
                "sigma_fit": float(fit.sigma_fit),
                "residual_rms_m": float(fit.residual_rms_m),
                "n_anchor": int(fit.n_anchor), "n_inlier": int(fit.n_inlier),
                "status": fit.status.value,
            }

        los = ga.line_of_sight_field(
            calib, depth, sigma, valid, xs, ys, plane=ga.FloorPlane(),
            target=ga.TargetVolume(radius_m=0.0, z_min_m=C.TARGET_HEIGHT_M,
                                   z_max_m=C.TARGET_HEIGHT_M, n_heights=1, n_ring=0),
        )
        maps[f"{frame}__p_visible"] = np.asarray(los.p_visible, dtype=float)
        maps[f"{frame}__p_unknown"] = np.asarray(los.p_unknown, dtype=float)
        maps[f"{frame}__in_fov"] = np.asarray(los.in_fov, dtype=float)
        print(f"[mono {env.key}] {C.SHORT[frame]}: affine scale {scale:.4f} shift {shift:+.4f} "
              f"({'reused from ' + C.DEVELOPMENT_ENV if reuse else f'fitted on {n_anchor} floor anchors'}), "
              f"p_visible mean {maps[f'{frame}__p_visible'].mean():.4f}")

    out_npz = C.mono_depth_path(env.key)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **maps)

    if not reuse:
        anchor_path.write_text(json.dumps(
            {"env": env.key, "model": args.model, "cameras": record_out}, indent=2),
            encoding="utf-8")
        print(f"[mono {env.key}] wrote commissioning anchor to {anchor_path.name}")

    (OUT_DIR / f"{env.key}_fit.json").write_text(json.dumps({
        "env": env.key, "world": env.world_name, "model": args.model,
        "anchor_source": C.DEVELOPMENT_ENV if reuse else env.key,
        "anchor_reused_unchanged": bool(reuse),
        "frames": {f: {"rgb_path": r["rgb_path"], "robot_range_m": r["robot_range_m"]}
                   for f, r in frames.items()},
    }, indent=2), encoding="utf-8")
    print(f"[mono {env.key}] wrote {out_npz.relative_to(C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
