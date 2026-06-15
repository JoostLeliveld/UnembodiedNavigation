#!/usr/bin/env python3
"""Build a YOLO-seg dataset from captured RGB frames + analytic labels.

Frames are saved by the yolo node as raw/frame_<pid>_<stamp>.png (one pid per
run). For each run we read the yolo-node pid and its perception.csv from the
launch log, then for every frame look up the robot's TRUE (x,y) at that sim
stamp and project the robot's 3D extent (a ~0.14 m cylinder, 0.19 m tall) through
the SAME ObliqueCameraModel the runtime uses -> an exact bounding box. This gives
a current-camera dataset with perfect labels (independent of the stale detector).

Label = class 0 + a rectangular polygon (the bbox corners); the runtime uses the
box (masks off), so a rectangular seg label yields the correct box.
"""
import sys, glob, csv, re, math, random, shutil
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/unav_common"))
from unav_common.camera_model import ObliqueCameraModel

CAPTURE = ROOT / "logs/_capture_dataset"
RUNROOT = ROOT / "logs/visibility_comparison/_capture_run"
OUT = ROOT / "logs/perception_datasets/aws_simseg_v3_currentcam"
IMG_W, IMG_H = 1280, 720
ROBOT_R, ROBOT_Z = 0.075, 0.19
cam = ObliqueCameraModel(cam_pos=(0.0, -5.5, 4.8), look_at=(0.0, -1.845, 0.0),
                         fov_h_rad=1.5708, img_width=IMG_W, img_height=IMG_H)


def project3d(x, y, z):
    """Proper 3D pinhole projection (world_to_pixel uses a z=0 homography and
    ignores z, so it cannot project elevated points)."""
    cam_pt = cam.R @ (np.array([x, y, z], dtype=float) - cam.cam_pos)
    if cam_pt[2] <= 1e-6:
        return None
    img = cam.K @ cam_pt
    return img[0] / img[2], img[1] / img[2]


def robot_bbox(tx, ty):
    """Project the robot cylinder at (tx,ty) -> image bbox (xmin,ymin,xmax,ymax) or None."""
    us, vs = [], []
    for z in (0.0, ROBOT_Z):
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            p = project3d(tx + ROBOT_R * math.cos(a), ty + ROBOT_R * math.sin(a), z)
            if p is None:
                return None
            us.append(p[0]); vs.append(p[1])
    x0, y0, x1, y1 = min(us), min(vs), max(us), max(vs)
    # require the box to be substantially inside the frame
    if x1 < 2 or y1 < 2 or x0 > IMG_W - 2 or y0 > IMG_H - 2:
        return None
    x0 = max(0.0, x0); y0 = max(0.0, y0); x1 = min(IMG_W - 1.0, x1); y1 = min(IMG_H - 1.0, y1)
    if (x1 - x0) < 3 or (y1 - y0) < 3:
        return None
    return x0, y0, x1, y1


def map_pid_to_perception():
    """pid -> perception.csv, matching each run's launch.log (has the yolo pid)
    to the run exp dir by closest start timestamp."""
    from datetime import datetime
    # run dirs with their start datetime
    runs = []
    for d in glob.glob(str(RUNROOT / "*/*/seed0/experiment_*")):
        m = re.search(r"experiment_(\d{8})_(\d{6})", d)
        if m and (Path(d) / "perception.csv").exists():
            runs.append((datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), Path(d) / "perception.csv"))
    out = {}
    for lg in glob.glob(str(RUNROOT / "_ros_logs/*/launch.log")):
        txt = Path(lg).read_text(errors="ignore")
        pm = re.search(r"yolo_robot_detector_node-\d+\]: process started with pid \[(\d+)\]", txt)
        tm = re.search(r"/_ros_logs/(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", lg)
        if not (pm and tm):
            continue
        lt = datetime.strptime("".join(tm.groups()), "%Y%m%d%H%M%S")
        cand = sorted(runs, key=lambda r: abs((r[0] - lt).total_seconds()))
        if cand and abs((cand[0][0] - lt).total_seconds()) < 60:
            out[pm.group(1)] = cand[0][1]
    return out


def load_poses(perc):
    rows = list(csv.reader(open(perc))); h = rows[0]; ci = {n: i for i, n in enumerate(h)}
    st, tx, ty = [], [], []
    for r in rows[1:]:
        try:
            st.append(float(r[ci["log_stamp"]] if r[ci.get("log_stamp", -1)] else r[ci["diag_stamp"]]))
            tx.append(float(r[ci["true_x"]])); ty.append(float(r[ci["true_y"]]))
        except Exception:
            pass
    return np.array(st), np.array(tx), np.array(ty)


def main():
    pidmap = map_pid_to_perception()
    print(f"runs mapped (pid->perception.csv): {len(pidmap)}")
    poses = {pid: load_poses(perc) for pid, perc in pidmap.items()}
    frames = sorted(glob.glob(str(CAPTURE / "raw/frame_*.png")))
    print(f"raw frames: {len(frames)}")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images/train").mkdir(parents=True); (OUT / "images/val").mkdir(parents=True)
    (OUT / "labels/train").mkdir(parents=True); (OUT / "labels/val").mkdir(parents=True)
    random.seed(0); n_lab = 0; n_skip = 0
    for fr in frames:
        m = re.match(r"frame_(\d+)_([\d.]+)\.png", Path(fr).name)
        if not m:
            n_skip += 1; continue
        pid, fst = m.group(1), float(m.group(2))
        if pid not in poses:
            n_skip += 1; continue
        st, tx, ty = poses[pid]
        if len(st) == 0:
            n_skip += 1; continue
        k = int(np.argmin(np.abs(st - fst)))
        if abs(st[k] - fst) > 0.5:
            n_skip += 1; continue
        bb = robot_bbox(tx[k], ty[k])
        if bb is None:
            n_skip += 1; continue
        x0, y0, x1, y1 = bb
        split = "val" if random.random() < 0.2 else "train"
        stem = f"{pid}_{fst:.3f}"
        shutil.copy(fr, OUT / f"images/{split}/{stem}.png")
        # rectangular polygon (normalized), class 0
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        poly = " ".join(f"{u/IMG_W:.6f} {v/IMG_H:.6f}" for u, v in pts)
        (OUT / f"labels/{split}/{stem}.txt").write_text(f"0 {poly}\n")
        n_lab += 1
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: robot\n")
    print(f"labeled {n_lab} frames, skipped {n_skip}; dataset at {OUT}")
    print(f"train: {len(list((OUT/'images/train').glob('*.png')))}  val: {len(list((OUT/'images/val').glob('*.png')))}")


if __name__ == "__main__":
    main()
