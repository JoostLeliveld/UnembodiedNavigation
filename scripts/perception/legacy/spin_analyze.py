#!/usr/bin/env python3
"""Definitive STATIC heading-bias test from the controlled spin capture.

At each FIXED (x,y) the true world position is constant, so feeding each frame
through the real runtime localization pipeline (YOLO box-bottom-centre ->
ground homography -> BEV affine) and comparing to (x,y) gives the detector's
heading-dependent localization error with NO dynamic confounds (no motion blur,
no latency, no truth-interp). Red-body segmentation gives an independent image
reference. Camera params from load_profile(warehouse_aws); affine from the
campaign config. Stale frames (robot not yet settled at the pose, red_px<80)
are dropped.
"""
import csv, glob, json, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage

ROOT = "/home/joostleliveld/Thesis/UnembodiedNavigation"
MODEL = f"{ROOT}/logs/perception_models/archive/aws_yolo_simseg_v4_imgsz960_20260617_paper_yolo_audit_broken/model.pt"
OUTDIR = "/tmp/spin_test_frames"
AFFINE = [0.996142, -0.002705, -0.002021, -0.002609, 0.991001, 0.066112]
MIN_RED_PX = 80


def red_seg(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    m = (R > 120) & (R - G > 60) & (R - B > 40)
    if m.sum() < MIN_RED_PX:
        return None
    lab, n = ndimage.label(m)
    k = int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1
    ys, xs = np.where(lab == k)
    return dict(cx=float(xs.mean()), cy=float(ys.mean()), bottom=(float(xs.mean()), float(ys.max())),
                npx=int((lab == k).sum()), x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max()))


def marker_near(im, cx, cy, rad=45):
    """Small blue mast-marker: blue pixels within rad px of the red centroid."""
    B, R, G = im[..., 2], im[..., 0], im[..., 1]
    mb = (B > 130) & (B - R > 30) & (B - G > 20)
    ys, xs = np.where(mb)
    if len(xs) == 0:
        return None
    d = np.hypot(xs - cx, ys - cy)
    sel = d < rad
    if sel.sum() < 8:
        return None
    return float(xs[sel].mean()), float(ys[sel].mean())


def main():
    from unav_common.camera_model import ObliqueCameraModel
    from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
    from ultralytics import YOLO
    _p, intr, _w, cp = load_profile(f"{ROOT}/src/experiments/config/world_profiles.yaml", "warehouse_aws.world.sdf")
    cam = ObliqueCameraModel(cam_pos=[cp[0], cp[1], cp[2]],
                             look_at=compute_look_at_from_pose([cp[0], cp[1], cp[2]], cp[3], cp[4], cp[5]),
                             img_width=intr["img_width"], img_height=intr["img_height"], fov_h_rad=intr["fov_h_rad"])

    def to_world(u, v):
        x, y = cam.pixel_to_world(u, v)
        return AFFINE[0]*x + AFFINE[1]*y + AFFINE[2], AFFINE[3]*x + AFFINE[4]*y + AFFINE[5]

    model = YOLO(MODEL)
    rows = list(csv.DictReader(open(f"{OUTDIR}/poses.csv")))
    bytag = {}
    for r in rows:
        bytag.setdefault(r["tag"], []).append(r)

    fig, axs = plt.subplots(1, len(bytag), figsize=(6.3 * len(bytag), 5.2))
    axs = np.atleast_1d(axs)
    summary = {}
    for ax, (tag, items) in zip(axs, bytag.items()):
        items.sort(key=lambda r: float(r["yaw_deg"]))
        x, y = float(items[0]["x"]), float(items[0]["y"])
        yaws, yolo_err, red_err, marker_orbit = [], [], [], []
        mk_pts, red_cen = [], []
        skipped = []
        for r in items:
            yd = float(r["yaw_deg"])
            im = mpimg.imread(r["frame"]); im = (im * 255 if im.max() <= 1.0 else im)[..., :3].astype(float)
            rs = red_seg(im)
            if rs is None:
                skipped.append(yd); continue
            # runtime pipeline: live YOLO box-bottom-centre -> world
            res = model.predict(r["frame"], imgsz=960, conf=0.05, verbose=False, device=0)[0]
            ye = math.nan
            if res.boxes is not None and len(res.boxes):
                b = res.boxes.xyxy.cpu().numpy()[int(res.boxes.conf.cpu().numpy().argmax())]
                wx, wy = to_world((b[0] + b[2]) / 2, b[3])
                ye = math.hypot(wx - x, wy - y)
            # independent: red-seg bottom-centre -> world
            rwx, rwy = to_world(*rs["bottom"])
            re = math.hypot(rwx - x, rwy - y)
            mk = marker_near(im, rs["cx"], rs["cy"])
            yaws.append(yd); yolo_err.append(ye); red_err.append(re)
            red_cen.append((rs["cx"], rs["cy"]))
            if mk is not None:
                mk_pts.append(mk)
        yaws = np.array(yaws); yolo_err = np.array(yolo_err); red_err = np.array(red_err)
        red_cen = np.array(red_cen); mk_pts = np.array(mk_pts)
        # marker orbit radius about red centroid (the heading-dependent bias source)
        orbit = float("nan")
        if len(mk_pts) and len(red_cen):
            rc = red_cen.mean(0)
            orbit = float(np.median(np.hypot(mk_pts[:, 0] - rc[0], mk_pts[:, 1] - rc[1])))
        ax.plot(yaws, yolo_err, "o-", color="#d62728", label="YOLO box-bottom -> world (runtime pipeline)")
        ax.plot(yaws, red_err, "s--", color="#1a5fd6", label="red-seg bottom -> world (independent)")
        ax.axhline(0.05, color="gray", ls=":", lw=1)
        ax.set_xlabel("robot heading yaw (deg)"); ax.set_ylabel("static world loc error (m)")
        ax.set_ylim(0, max(0.2, np.nanmax(np.concatenate([yolo_err, red_err])) * 1.15))
        ax.set_xticks(range(0, 360, 60)); ax.grid(alpha=0.3)
        ym = np.nanmedian(yolo_err); yx = np.nanmax(yolo_err)
        sk = f" (dropped stale yaws {skipped})" if skipped else ""
        ax.set_title(f"{tag}  ({x},{y})\nYOLO-pipeline err: median {ym:.3f} m, max {yx:.3f} m\nmarker orbit {orbit:.0f}px{sk}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        summary[tag] = dict(x=x, y=y, n=int(len(yaws)),
                            yolo_world_err_median=float(ym), yolo_world_err_max=float(yx),
                            red_world_err_median=float(np.nanmedian(red_err)),
                            marker_orbit_px=orbit, dropped_stale=skipped)
    fig.suptitle("Static heading-bias test: localization error vs robot yaw at FIXED (x,y). "
                 "Truth is constant, so any error is pure detector/projection heading bias.",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = f"{ROOT}/paper_artifacts/figures/diagnostics/spin_heading_bias.png"
    fig.savefig(out, dpi=100, bbox_inches="tight"); fig.savefig("/tmp/spin_heading_bias.png", dpi=100, bbox_inches="tight")
    json.dump(summary, open(out.replace(".png", ".json"), "w"), indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
