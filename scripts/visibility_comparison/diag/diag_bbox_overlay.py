#!/usr/bin/env python3
"""Camera-feed video + montage: what the camera sees, with the YOLO box, the
selected bottom pixel, and where the robot TRULY is (truth projected to pixel).

This is the "trust no module" visual check of (a) the bbox framing the robot and
(b) the bottom-pixel projection. Uses the LOGGED bbox/selected-pixel from
perception.csv (no YOLO re-run) and the saved raw frames.

Usage:
    python diag_bbox_overlay.py <run_dir> [--frames <dir>] [--out <dir>] [--fps 8]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio

import diag_common as dc


def expected_box(cam, tx, ty):
    """Project the robot 3D model (cylinder r=0.09, h=0.19) to a pixel bbox."""
    pts = []
    for a in np.linspace(0, 2 * math.pi, 16):
        for z in (0.0, 0.19):
            u, v, _ = cam.world_to_pixel(tx + 0.09 * math.cos(a), ty + 0.09 * math.sin(a), z)
            pts.append((u, v))
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    return min(us), min(vs), max(us), max(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--frames", default=None, help="frames dir (raw/*.png). default: auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=8.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    perc, exp, summary = dc.load_run(run_dir)
    cam = dc.build_camera()
    fx, fy, fyaw = dc.truth_interp_fns(exp)

    frames_dir = args.frames
    if frames_dir is None:
        # default diag baseline location, else look for a sibling frames dir
        guess = run_dir
        for _ in range(4):
            cand = list(guess.glob("frames_*"))
            if cand:
                frames_dir = cand[0]
                break
            guess = guess.parent
    if frames_dir is None:
        frames_dir = dc.ROOT / "logs/visibility_comparison/_diag_baseline/frames_c2_a3mid_seed0"
    fidx = dc.index_frames(frames_dir)
    print(f"frames indexed: {len(fidx)} from {frames_dir}")

    out_dir = Path(args.out) if args.out else (run_dir / "diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    det = perc[perc["detected"] == 1].copy() if "detected" in perc else perc.copy()
    det = det.dropna(subset=["diag_stamp"]).sort_values("diag_stamp")

    rows = []
    for _, r in det.iterrows():
        st = float(r["diag_stamp"])
        path, cap = dc.frame_at(fidx, st)
        if path is None:
            continue
        rows.append((r, path, cap))
    print(f"matched {len(rows)} detected frames to images")
    if not rows:
        print("NO matched frames — is the frames dir correct?")
        return

    # ---- video ----
    vid_frames = []
    errs = []
    for (r, path, cap) in rows:
        img = imageio.imread(path)
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=90)
        ax.imshow(img)
        # logged YOLO bbox
        x0, y0, x1, y1 = (float(r.get(k, np.nan)) for k in
                          ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
        if math.isfinite(x0):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=2.0))
        # selected bottom pixel
        su, sv = float(r.get("obs_u", np.nan)), float(r.get("obs_v", np.nan))
        if math.isfinite(su):
            ax.plot([su], [sv], "o", color="yellow", ms=6, mec="black", mew=0.6)
        # truth projected to pixel at capture time
        tx, ty = fx(cap), fy(cap)
        tu, tv, vis = cam.world_to_pixel(tx, ty, 0.0)
        ax.plot([tu], [tv], "+", color="lime", ms=12, mew=2.2)
        ex0, ey0, ex1, ey1 = expected_box(cam, tx, ty)
        ax.add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fill=False,
                               edgecolor="lime", lw=1.4, ls="--"))
        # projected world error (planner offset path + affine)
        loc_aff = float(r.get("localization_error_calibrated_m", np.nan))
        loc_cap = float(r.get("localization_error_captime_m", np.nan))
        errs.append(loc_cap)
        inf_ms = float(r.get("yolo_inference_ms", np.nan))
        age = float(r.get("frame_age_at_publish_s", np.nan))
        ax.set_title(
            f"t={cap:.1f}s  truth=({tx:.2f},{ty:.2f})  "
            f"loc_err affine={loc_aff:.3f}m captime={loc_cap:.3f}m\n"
            f"inference={inf_ms:.0f}ms  frame_age={age:.2f}s   "
            f"(red=YOLO box, yellow=bottom pixel, green=truth)",
            fontsize=8)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        vid_frames.append(buf.copy())
        plt.close(fig)

    gif = out_dir / "camera_view_bbox.gif"
    imageio.mimsave(gif, vid_frames, duration=1.0 / max(args.fps, 1e-3), loop=0)
    print(f"wrote {gif}  ({len(vid_frames)} frames @ {args.fps}fps, {gif.stat().st_size/1e6:.1f}MB)")

    # ---- montage: low / median / high captime error frames ----
    order = np.argsort([e if math.isfinite(e) else -1 for e in errs])
    picks = [("LOW err", order[len(order) // 10]),
             ("MEDIAN err", order[len(order) // 2]),
             ("HIGH err", order[-1])]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (lab, idx) in zip(axs, picks):
        r, path, cap = rows[idx]
        ax.imshow(imageio.imread(path))
        x0, y0, x1, y1 = (float(r.get(k, np.nan)) for k in
                          ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
        if math.isfinite(x0):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=2.2))
        tx, ty = fx(cap), fy(cap)
        tu, tv, _ = cam.world_to_pixel(tx, ty, 0.0)
        ax.plot([tu], [tv], "+", color="lime", ms=14, mew=2.5)
        ex0, ey0, ex1, ey1 = expected_box(cam, tx, ty)
        ax.add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fill=False,
                               edgecolor="lime", lw=1.6, ls="--"))
        cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
        ax.set_xlim(cx - 150, cx + 150)
        ax.set_ylim(cy + 120, cy - 120)
        ax.set_title(f"{lab}: captime_err={errs[idx]:.3f}m  t={cap:.1f}s", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{run_dir.name}: YOLO box (red) vs truth model box (green dashed)", fontsize=12)
    fig.tight_layout()
    png = out_dir / "bbox_montage.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
