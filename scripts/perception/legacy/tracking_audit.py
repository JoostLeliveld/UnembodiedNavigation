#!/usr/bin/env python3
"""Definitive tracking audit on a solid footing.

For EVERY captured frame: run YOLO live on that exact frame (box guaranteed to
match the image), back-project the box-bottom-centre to the ground (z=0 homography)
+ the runtime affine BEV calibration, and compare to truth interpolated to the
frame's CAPTURE time (experiment.csv). Reports the error distribution and shows
best / quartile / worst frames with the live box (red) and capture-time truth
footprint (green). Camera params from load_profile(warehouse_aws); affine from
the campaign config. No hardcoded camera, no stale perception columns.
"""
import csv, glob, json, math, os
import numpy as np
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = "/home/joostleliveld/Thesis/UnembodiedNavigation"
RUN = sorted(glob.glob(f"{ROOT}/logs/visibility_comparison/aws_f31b1_v4_b5c2_debug/b5_a4_apron_to_a2_low/C2/seed*/experiment_*"))[0]
FRAMES = "/tmp/b5_c2_debug_frames"
CFG = f"{ROOT}/scripts/visibility_comparison/aws_f31b1_v4_b5c2_debug.yaml"


def ff(r, k):
    try:
        v = float(r[k]); return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def main():
    from unav_common.camera_model import ObliqueCameraModel
    from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
    from ultralytics import YOLO
    _p, intr, _w, cp = load_profile(f"{ROOT}/src/experiments/config/world_profiles.yaml", "warehouse_aws.world.sdf")
    cam = ObliqueCameraModel(cam_pos=[cp[0], cp[1], cp[2]],
                             look_at=compute_look_at_from_pose([cp[0], cp[1], cp[2]], cp[3], cp[4], cp[5]),
                             img_width=intr["img_width"], img_height=intr["img_height"], fov_h_rad=intr["fov_h_rad"])
    cfg = yaml.safe_load(open(CFG))
    aff = [float(v) for v in str(cfg.get("bev_affine_calibration", "")).split(",")] if cfg.get("bev_affine_calibration") else None

    def to_world(u, v):
        x, y = cam.pixel_to_world(u, v)
        if aff and len(aff) == 6:
            return aff[0]*x + aff[1]*y + aff[2], aff[3]*x + aff[4]*y + aff[5]
        return x, y

    exp = list(csv.DictReader(open(f"{RUN}/experiment.csv")))
    tr = np.array([[ff(r, "stamp"), ff(r, "truth_x"), ff(r, "truth_y"), ff(r, "cmd_v"), ff(r, "cmd_w")] for r in exp
                   if math.isfinite(ff(r, "stamp")) and math.isfinite(ff(r, "truth_x"))])

    def truth_at(t):
        return float(np.interp(t, tr[:, 0], tr[:, 1])), float(np.interp(t, tr[:, 0], tr[:, 2]))

    def speed_at(t):
        return abs(float(np.interp(t, tr[:, 0], tr[:, 3]))), abs(float(np.interp(t, tr[:, 0], tr[:, 4])))

    frames = sorted(glob.glob(f"{FRAMES}/raw/*.png"),
                    key=lambda p: float(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
    model = YOLO(f"{ROOT}/logs/perception_models/warehouse_yolo_detector_v1/model.pt")
    recs = []
    for p in frames:
        st = float(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        if st < tr[0, 0] or st > tr[-1, 0]:
            continue
        res = model.predict(p, imgsz=960, conf=0.05, verbose=False, device=0)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        b = res.boxes.xyxy.cpu().numpy(); c = res.boxes.conf.cpu().numpy(); i = int(c.argmax())
        x0, y0, x1, y1 = [float(z) for z in b[i]]
        wx, wy = to_world((x0 + x1) / 2, y1)
        tx, ty = truth_at(st)
        v, w = speed_at(st)
        recs.append(dict(st=st, box=(x0, y0, x1, y1), err=math.hypot(wx - tx, wy - ty),
                         tx=tx, ty=ty, v=v, w=w, path=p))
    errs = np.array([r["err"] for r in recs])
    print(f"frames scored (live YOLO): {len(recs)}")
    print(f"world err (box-bottom->ground+affine vs capture-truth): "
          f"median {np.median(errs):.3f}  mean {errs.mean():.3f}  p90 {np.percentile(errs,90):.3f}  max {errs.max():.3f} m")
    # correlation with turning
    w_arr = np.array([r["w"] for r in recs])
    turn = errs[w_arr > 0.3]; straight = errs[w_arr <= 0.3]
    print(f"  err median when TURNING(|w|>0.3): {np.median(turn):.3f} (n={len(turn)}) | else {np.median(straight):.3f} (n={len(straight)})")

    recs.sort(key=lambda r: r["err"])
    picks = [("best", recs[0]), ("p25", recs[len(recs)//4]), ("median", recs[len(recs)//2]),
             ("p90", recs[int(len(recs)*0.9)]), ("worst", recs[-1])]
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.4, 1])
    for j, (lab, r) in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(mpimg.imread(r["path"]))
        x0, y0, x1, y1 = r["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=2.2))
        circ = [cam.world_to_pixel(r["tx"] + 0.09*math.cos(a), r["ty"] + 0.09*math.sin(a), 0.0)[:2]
                for a in np.linspace(0, 2*math.pi, 20)]
        ax.plot([p[0] for p in circ], [p[1] for p in circ], "-", color="lime", lw=2)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - 95, cx + 95); ax.set_ylim(cy + 80, cy - 80); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: err {r['err']:.2f} m\n|w|={r['w']:.2f} v={r['v']:.2f}", fontsize=9)
    axh = fig.add_subplot(gs[1, :2]); axh.hist(errs, bins=40, color="#1a5fd6"); axh.axvline(np.median(errs), color="r", ls="--", label=f"median {np.median(errs):.2f}")
    axh.set_xlabel("world localization error (m)"); axh.set_ylabel("frames"); axh.legend(); axh.set_title("error distribution (live YOLO, capture-time truth)")
    axt = fig.add_subplot(gs[1, 2:])
    ts = np.array([r["st"] for r in recs]); order = ts.argsort()
    axt.plot(ts[order], errs[order], ".-", ms=3, lw=0.6, color="#1a5fd6", label="loc err")
    axt.plot(ts[order], w_arr[order] * 0.2, "-", color="orange", lw=0.8, alpha=0.7, label="|cmd_w| (scaled)")
    axt.set_xlabel("sim time (s)"); axt.set_ylabel("m"); axt.legend(fontsize=8); axt.set_title("error vs time (peaks track turning)")
    fig.legend(handles=[Line2D([0],[0],color="red",lw=2.2,label="DETECTED box (live YOLO)"),
                        Line2D([0],[0],color="lime",lw=2,label="capture-time truth footprint (r=0.09 m)")],
               loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Tracking audit (b5 C2): live-YOLO boxes on the robot; tracking median {np.median(errs):.2f} m, "
                 f"tail to {errs.max():.2f} m concentrated in turns", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = f"{ROOT}/paper_artifacts/figures/diagnostics/tracking_audit_b5c2.png"
    fig.savefig(out, dpi=95, bbox_inches="tight"); fig.savefig("/tmp/tracking_audit.png", dpi=95, bbox_inches="tight")
    json.dump({"n": len(recs), "median": float(np.median(errs)), "p90": float(np.percentile(errs, 90)),
               "max": float(errs.max()), "affine": aff,
               "turn_median": float(np.median(turn)), "straight_median": float(np.median(straight))},
              open(out.replace(".png", ".json"), "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
