#!/usr/bin/env python3
"""Runtime 3-way audit: actual robot (image) vs YOLO box vs projected truth.

The robot body renders as a saturated red (mean ~(199,69,70)), distinct from
yellow racks / blue rails / tan boxes. We segment it directly to get the robot's
ACTUAL image position (ground truth from the picture, independent of detector and
projection). For every captured frame we compare three things at the frame's
capture time:
  * ACTUAL robot  = red-body segmentation centroid/bbox (green),
  * DETECTED      = live YOLO box on that exact frame (red),
  * PROJECTED TRUTH = experiment.csv truth (interp to capture time) projected
    ground->top (cyan), with footprint (lime).
Reports, split by turning, the YOLO-vs-actual gap (detector error) and the
projected-truth-vs-actual gap (calibration/truth error), and renders examples.
Camera from load_profile(warehouse_aws); full-3D world_to_pixel.
"""
import csv, glob, json, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy import ndimage

ROOT = "/home/joostleliveld/Thesis/UnembodiedNavigation"
RUN = sorted(glob.glob(f"{ROOT}/logs/visibility_comparison/aws_f31b1_v4_b5c2_debug/b5_a4_apron_to_a2_low/C2/seed*/experiment_*"))[0]
FRAMES = "/tmp/b5_c2_debug_frames"


def ff(r, k):
    try:
        v = float(r[k]); return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def red_body(im):
    """Return (cx, cy, x0, y0, x1, y1, npx) of the robot's red body, or None."""
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    m = (R > 120) & (R - G > 60) & (R - B > 40)
    if m.sum() < 12:
        return None
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == k)
    return float(xs.mean()), float(ys.mean()), xs.min(), ys.min(), xs.max(), ys.max(), int((lab == k).sum())


def main():
    from unav_common.camera_model import ObliqueCameraModel
    from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
    from ultralytics import YOLO
    _p, intr, _w, cp = load_profile(f"{ROOT}/src/experiments/config/world_profiles.yaml", "warehouse_aws.world.sdf")
    cam = ObliqueCameraModel(cam_pos=[cp[0], cp[1], cp[2]],
                             look_at=compute_look_at_from_pose([cp[0], cp[1], cp[2]], cp[3], cp[4], cp[5]),
                             img_width=intr["img_width"], img_height=intr["img_height"], fov_h_rad=intr["fov_h_rad"])

    def w2p(x, y, z=0.0):
        u, v, _ = cam.world_to_pixel(x, y, z); return u, v

    exp = list(csv.DictReader(open(f"{RUN}/experiment.csv")))
    tr = np.array([[ff(r, "stamp"), ff(r, "truth_x"), ff(r, "truth_y"), ff(r, "cmd_w")] for r in exp
                   if math.isfinite(ff(r, "stamp")) and math.isfinite(ff(r, "truth_x"))])

    def at(t, col):
        return float(np.interp(t, tr[:, 0], tr[:, col]))

    frames = sorted(glob.glob(f"{FRAMES}/raw/*.png"),
                    key=lambda p: float(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
    model = YOLO(f"{ROOT}/logs/perception_models/warehouse_yolo_detector_v1/model.pt")
    recs = []
    for p in frames:
        st = float(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        if st < tr[0, 0] or st > tr[-1, 0]:
            continue
        im = mpimg.imread(p)
        im = (im * 255 if im.max() <= 1.0 else im)[..., :3].astype(float)
        rb = red_body(im)
        if rb is None:
            continue
        rcx, rcy, rx0, ry0, rx1, ry1, npx = rb
        res = model.predict(p, imgsz=960, conf=0.05, verbose=False, device=0)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        b = res.boxes.xyxy.cpu().numpy()[int(res.boxes.conf.cpu().numpy().argmax())]
        ycx, ycy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        tx, ty, w = at(st, 1), at(st, 2), abs(at(st, 3))
        # projected truth at body mid-height (~0.075 m), compared to red-body centroid
        tu, tv = w2p(tx, ty, 0.075)
        recs.append(dict(st=st, path=p, w=w,
                         red=(rcx, rcy, rx0, ry0, rx1, ry1), yolo=tuple(b), ycen=(ycx, ycy),
                         truth=(tx, ty), tproj=(tu, tv),
                         gap_yolo=math.hypot(ycx - rcx, ycy - rcy),
                         gap_truth=math.hypot(tu - rcx, tv - rcy)))
    gy = np.array([r["gap_yolo"] for r in recs]); gt = np.array([r["gap_truth"] for r in recs])
    w = np.array([r["w"] for r in recs])
    print(f"frames audited: {len(recs)}")
    print("ALL  : YOLO-vs-actual px med %.1f p90 %.1f | truth-proj-vs-actual px med %.1f p90 %.1f"
          % (np.median(gy), np.percentile(gy, 90), np.median(gt), np.percentile(gt, 90)))
    for lab, sel in (("straight |w|<=0.3", w <= 0.3), ("turning  |w|>0.3", w > 0.3)):
        if sel.any():
            print("%-18s YOLO-vs-actual med %.1f px | truth-proj-vs-actual med %.1f px (n=%d)"
                  % (lab, np.median(gy[sel]), np.median(gt[sel]), int(sel.sum())))

    recs.sort(key=lambda r: r["gap_truth"])
    picks = [("best", recs[0]), ("median", recs[len(recs)//2]),
             ("p90", recs[int(len(recs)*0.9)]), ("worst", recs[-1]),
             ("worst-2", recs[-2]), ("worst-3", recs[-3])]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10)); axs = axs.ravel()
    for ax, (lab, r) in zip(axs, picks):
        ax.imshow(mpimg.imread(r["path"]))
        rcx, rcy, rx0, ry0, rx1, ry1 = r["red"]
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, edgecolor="lime", lw=2.2))  # actual robot (red-seg)
        b = r["yolo"]; ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, edgecolor="red", lw=1.8, ls="--"))  # YOLO
        tx, ty = r["truth"]
        zs = np.linspace(0, 0.19, 6); pts = [w2p(tx, ty, z) for z in zs]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color="cyan", ms=3, lw=1.5)  # truth ground->top
        cx, cy = rcx, rcy
        ax.set_xlim(cx - 90, cx + 90); ax.set_ylim(cy + 75, cy - 75); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: YOLO-vs-actual {r['gap_yolo']:.0f}px | truth-vs-actual {r['gap_truth']:.0f}px | |w|={r['w']:.2f}", fontsize=9)
    fig.legend(handles=[Line2D([0],[0],color="lime",lw=2.2,label="ACTUAL robot (red-body segmentation)"),
                        Line2D([0],[0],color="red",lw=1.8,ls="--",label="DETECTED (live YOLO box)"),
                        Line2D([0],[0],color="cyan",lw=1.5,label="PROJECTED TRUTH ground->top")],
               loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Runtime 3-way audit (b5 C2): actual robot vs YOLO vs projected truth", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = f"{ROOT}/paper_artifacts/figures/diagnostics/runtime_robot_audit_b5c2.png"
    fig.savefig(out, dpi=95, bbox_inches="tight"); fig.savefig("/tmp/runtime_robot_audit.png", dpi=95, bbox_inches="tight")
    json.dump({"n": len(recs),
               "yolo_vs_actual_px": {"median": float(np.median(gy)), "p90": float(np.percentile(gy, 90))},
               "truth_vs_actual_px": {"median": float(np.median(gt)), "p90": float(np.percentile(gt, 90))},
               "turning": {"yolo_med": float(np.median(gy[w > 0.3])), "truth_med": float(np.median(gt[w > 0.3]))},
               "straight": {"yolo_med": float(np.median(gy[w <= 0.3])), "truth_med": float(np.median(gt[w <= 0.3]))}},
              open(out.replace(".png", ".json"), "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
