#!/usr/bin/env python3
"""Where do the milliseconds go in the four-camera batch, and what buys back Hz?

The four-camera pipeline emits observations at ~1.1 Hz per camera against 5 Hz
cameras. Only ~18 frames in 200 s were dropped by the stamp-skew guard, so the loss
is throughput, not synchronisation: one batched inference takes about as long as the
observed period. This measures that directly, on real captured frames, off-line.

Configurations, each timed the way the runtime calls the model
(``model.predict(source=[4 images], imgsz=..., conf=..., iou=...)``):

* **imgsz 960** -- the deployed setting, from the model manifest.
* **imgsz 768, 640** -- the obvious lever. Cost falls with pixel count, but the
  checkpoint was trained at 960, so this reports what accuracy it costs as well as
  what speed it buys: agreement of the box-bottom pixel against the 960 result on
  the same frame is the quantity the projection consumes.
* **forward pass only** -- the network without Ultralytics' postprocess. The
  checkpoint is a *segmentation* model, so predict also builds mask prototypes. If
  the forward pass is much faster than predict, the mask work is the target and
  ``torchscript_detection_only`` is worth compiling; if not, it is not.

Usage:  python3 benchmark_detector_rate.py [--run-tag notebook_gpu] [--batches 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "src").is_dir() and (p / "logs").is_dir())
sys.path.insert(0, str(HERE))

import notebook_data as nd  # noqa: E402

MODEL = REPO / "logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt"
CONF_FLOOR = 0.05
IOU = 0.45


def load_batches(run_tag: str | None, n_batches: int) -> list[list[np.ndarray]]:
    """n_batches lists of four BGR frames, one per camera, nearest-in-time."""
    import cv2

    capture = nd.load_capture(run_tag)
    per_camera = {cam: capture.frames(cam) for cam in nd.CAMERAS}
    missing = [c for c, f in per_camera.items() if not f]
    if missing:
        raise SystemExit(f"no recorded frames for {missing} in {capture.name}")

    reference = per_camera["camera_A"]
    step = max(1, len(reference) // n_batches)
    batches = []
    for stamp, path_a in reference[::step][:n_batches]:
        images = []
        for cam in nd.CAMERAS:
            hit = capture.frame_at(cam, stamp, tol_s=1.0)
            if hit is None:
                images = []
                break
            image = cv2.imread(str(hit[1]))
            if image is None:
                images = []
                break
            images.append(image)
        if len(images) == len(nd.CAMERAS):
            batches.append(images)
    if not batches:
        raise SystemExit("could not assemble a single four-camera batch of frames")
    return batches


def box_bottoms(results) -> list[tuple[float, float] | None]:
    """The bottom-centre pixel of the most confident box per image, or None."""
    out = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            out.append(None)
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        best = int(np.argmax(conf))
        x1, _y1, x2, y2 = xyxy[best]
        out.append((float((x1 + x2) / 2.0), float(y2)))
    return out


def time_predict(model, batches, imgsz: int, device) -> dict:
    kwargs = dict(imgsz=imgsz, conf=CONF_FLOOR, iou=IOU, batch=4,
                  stream=False, verbose=False)
    if device is not None:
        kwargs["device"] = device
    for _ in range(3):                                   # warm up / autotune
        model.predict(source=list(batches[0]), **kwargs)
    times, bottoms, confidences = [], [], []
    for images in batches:
        start = time.perf_counter()
        results = model.predict(source=list(images), **kwargs)
        times.append((time.perf_counter() - start) * 1e3)
        bottoms.append(box_bottoms(results))
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                confidences.append(float(boxes.conf.max()))
    median = statistics.median(times)
    return {
        "label": f"predict imgsz {imgsz}",
        "imgsz": imgsz,
        "ms_median": round(median, 1),
        "ms_p90": round(sorted(times)[int(0.9 * (len(times) - 1))], 1),
        "hz_per_camera": round(1000.0 / median, 2),
        "detections": len(confidences),
        "images": 4 * len(batches),
        "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
        "bottoms": bottoms,
    }


def time_forward_only(model, batches, imgsz: int, device) -> dict:
    """The network alone: letterbox on CPU, one forward pass, no postprocess."""
    import cv2
    import torch

    net = model.model.eval()
    target = torch.device(device if device is not None else "cpu")

    def prepare(images):
        prepared = []
        for image in images:
            height, width = image.shape[:2]
            gain = min(imgsz / width, imgsz / height)
            rw, rh = round(width * gain), round(height * gain)
            canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            px, py = round((imgsz - rw) / 2 - 0.1), round((imgsz - rh) / 2 - 0.1)
            canvas[py:py + rh, px:px + rw] = cv2.resize(image, (rw, rh))
            prepared.append(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
        return torch.from_numpy(np.stack(prepared)).to(target).float().div_(255.0)

    tensors = [prepare(images) for images in batches]
    with torch.no_grad():
        for _ in range(3):
            net(tensors[0])
        if target.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for tensor in tensors:
            start = time.perf_counter()
            net(tensor)
            if target.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e3)
    median = statistics.median(times)
    return {"label": f"forward only imgsz {imgsz}", "imgsz": imgsz,
            "ms_median": round(median, 1), "hz_per_camera": round(1000.0 / median, 2),
            "detections": None, "mean_confidence": None, "bottoms": None}


def pixel_agreement(reference: list, candidate: list) -> dict:
    """How far the box-bottom pixel moves when imgsz changes, where both detect."""
    deltas, ref_only, cand_only = [], 0, 0
    for ref_batch, cand_batch in zip(reference, candidate):
        for ref, cand in zip(ref_batch, cand_batch):
            if ref is None and cand is None:
                continue
            if ref is None:
                cand_only += 1
            elif cand is None:
                ref_only += 1
            else:
                deltas.append(float(np.hypot(cand[0] - ref[0], cand[1] - ref[1])))
    return {
        "shared_detections": len(deltas),
        "median_pixel_shift": round(statistics.median(deltas), 2) if deltas else None,
        "p90_pixel_shift": round(sorted(deltas)[int(0.9 * (len(deltas) - 1))], 2) if deltas else None,
        "lost_vs_960": ref_only,
        "gained_vs_960": cand_only,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--batches", type=int, default=15)
    parser.add_argument("--sizes", type=int, nargs="+", default=[960, 768, 640])
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else None
    print(f"device: {'cuda:0 ' + torch.cuda.get_device_name(0) if device == 0 else 'cpu'}")
    batches = load_batches(args.run_tag, args.batches)
    print(f"batches: {len(batches)} x 4 real captured frames "
          f"({batches[0][0].shape[1]}x{batches[0][0].shape[0]})\n")

    model = YOLO(str(MODEL))
    rows = []
    for size in args.sizes:
        row = time_predict(model, batches, size, device)
        rows.append(row)
        print(f"  {row['label']:22s} {row['ms_median']:7.1f} ms  -> {row['hz_per_camera']:5.2f} Hz"
              f"   detections {row['detections']:3d}/{row['images']}"
              f"   mean conf {row['mean_confidence']}")
    forward = time_forward_only(model, batches, args.sizes[0], device)
    print(f"  {forward['label']:22s} {forward['ms_median']:7.1f} ms  -> "
          f"{forward['hz_per_camera']:5.2f} Hz   (postprocess excluded)")

    baseline = rows[0]
    if forward["ms_median"] >= baseline["ms_median"]:
        # This happened on the Quadro P2000: the hand-rolled forward pass came out
        # SLOWER than the full predict. Ultralytics' predict goes through AutoBackend
        # with a warmed, fixed-shape, inference-mode path; calling `model.model`
        # directly is simply a different code path, not "predict minus postprocess".
        # So this probe cannot isolate the mask work, and must not pretend to.
        print(f"\n  the forward-only probe ({forward['ms_median']:.0f} ms) is not faster than "
              f"predict ({baseline['ms_median']:.0f} ms), so it is measuring a different\n"
              f"  code path, not predict-minus-postprocess. It says nothing about the mask cost.")
    else:
        print(f"\n  postprocess share at imgsz {baseline['imgsz']}: "
              f"{baseline['ms_median'] - forward['ms_median']:.0f} ms of "
              f"{baseline['ms_median']:.0f} ms "
              f"({100 * (1 - forward['ms_median'] / baseline['ms_median']):.0f}%)")

    print("\n  what a smaller imgsz costs, against imgsz 960 on the same frames:")
    comparisons = {}
    for row in rows[1:]:
        agreement = pixel_agreement(baseline["bottoms"], row["bottoms"])
        comparisons[row["label"]] = agreement
        print(f"    imgsz {row['imgsz']}: {row['hz_per_camera'] / baseline['hz_per_camera']:.2f}x "
              f"the rate | box-bottom moves {agreement['median_pixel_shift']} px median, "
              f"{agreement['p90_pixel_shift']} px p90 | "
              f"lost {agreement['lost_vs_960']}, gained {agreement['gained_vs_960']} detections")

    out = REPO / "logs/studies/filter_notebook/detector_rate_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(MODEL.relative_to(REPO)),
        "n_batches": len(batches),
        "timings": [{k: v for k, v in row.items() if k != "bottoms"} for row in rows],
        "forward_only": forward,
        "imgsz_comparisons": comparisons,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
