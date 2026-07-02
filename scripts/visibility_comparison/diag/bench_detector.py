#!/usr/bin/env python3
"""Idle inference benchmark + TorchScript bit-identity gate (no ROS, GPU free).

Answers the critical question: is the in-run ~290ms inference GIL/launch
contention (idle would be ~30ms) or just a slow GPU? And: does the TorchScript
export produce bit-identical detections vs the .pt model?

Usage:
    python bench_detector.py [<frames_dir_with_raw_pngs>] [--n 40]
"""
import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
PT = ROOT / "logs/perception_models/warehouse_yolo_detector_v1/model.pt"
TS = ROOT / "logs/perception_models/warehouse_yolo_detector_v1/model.torchscript"
IMGSZ = 960
CONF = 0.05
IOU = 0.45


def load_frames(frames_dir, k=12):
    raw = sorted(glob.glob(str(Path(frames_dir) / "raw" / "*.png")))
    if not raw:
        raw = sorted(glob.glob(str(Path(frames_dir) / "*.png")))
    import cv2
    imgs = [cv2.imread(p) for p in raw[:: max(1, len(raw) // k)][:k]]
    return [im for im in imgs if im is not None], raw


def time_model(model, imgs, n, label):
    # warmup
    for _ in range(3):
        model.predict(imgs[0], imgsz=IMGSZ, conf=0.0, iou=IOU, verbose=False, device=0)
    ts = []
    for i in range(n):
        im = imgs[i % len(imgs)]
        t0 = time.perf_counter()
        model.predict(im, imgsz=IMGSZ, conf=0.0, iou=IOU, verbose=False, device=0)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    print(f"  [{label}] inference_ms: mean={ts.mean():.1f} med={np.median(ts):.1f} "
          f"p90={np.percentile(ts,90):.1f} max={ts.max():.1f}  (n={n}, GPU idle)")
    return ts


def selected_pixel(res, use_masks=False):
    """Replicate the detector's bbox-bottom selection: best box, bottom-centre."""
    b = res.boxes
    if b is None or len(b) == 0:
        return None
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    i = int(conf.argmax())
    x0, y0, x1, y1 = xyxy[i]
    return (float((x0 + x1) / 2), float(y1), float(conf[i]), (float(x0), float(y0), float(x1), float(y1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", nargs="?",
                    default=str(ROOT / "logs/visibility_comparison/_diag_baseline/frames_c2_a3mid_seed0"))
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    from ultralytics import YOLO
    imgs, raw = load_frames(args.frames_dir)
    print(f"loaded {len(imgs)} sample frames from {args.frames_dir} ({len(raw)} total)")
    if not imgs:
        print("NO FRAMES — pass a frames dir with raw/*.png")
        return

    print("\n=== IDLE INFERENCE TIMING (GPU free, single process) ===")
    m_pt = YOLO(str(PT))
    time_model(m_pt, imgs, args.n, "model.pt (eager)")

    ts_ok = TS.is_file()
    m_ts = None
    if ts_ok:
        m_ts = YOLO(str(TS))
        time_model(m_ts, imgs, args.n, "model.torchscript")
    else:
        print(f"  [torchscript] MISSING at {TS}")

    print("\n=== BIT-IDENTITY GATE (.pt vs .torchscript selected pixel) ===")
    if not ts_ok:
        print("  skipped (no torchscript)")
        return
    max_du = max_dv = max_dconf = 0.0
    n_cmp = 0
    for im in imgs:
        r_pt = m_pt.predict(im, imgsz=IMGSZ, conf=0.0, iou=IOU, verbose=False, device=0)[0]
        r_ts = m_ts.predict(im, imgsz=IMGSZ, conf=0.0, iou=IOU, verbose=False, device=0)[0]
        s_pt = selected_pixel(r_pt)
        s_ts = selected_pixel(r_ts)
        if s_pt is None or s_ts is None:
            print(f"    frame: detection mismatch pt={s_pt is not None} ts={s_ts is not None}")
            continue
        du, dv = abs(s_pt[0] - s_ts[0]), abs(s_pt[1] - s_ts[1])
        dconf = abs(s_pt[2] - s_ts[2])
        max_du, max_dv, max_dconf = max(max_du, du), max(max_dv, dv), max(max_dconf, dconf)
        n_cmp += 1
    print(f"  compared {n_cmp} frames; max |du|={max_du:.3f}px |dv|={max_dv:.3f}px "
          f"|dconf|={max_dconf:.4f}")
    print("  VERDICT:", "BIT-IDENTICAL (<0.5px)" if max(max_du, max_dv) < 0.5
          else "DIVERGENT — investigate before using torchscript")


if __name__ == "__main__":
    main()
