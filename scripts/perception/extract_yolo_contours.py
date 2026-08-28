#!/usr/bin/env python3
"""Extract resumable RGB-derived YOLO segmentation contours for shape-update rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FIELDS = (
    "camera", "sample_index", "image", "detected", "confidence",
    "box_x1", "box_y1", "box_x2", "box_y2", "polygon_json", "polygon_points",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_images(dataset: Path) -> list[dict[str, str]]:
    with (dataset / "records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unique = {}
    for row in rows:
        key = (row["camera"], row["sample_index"])
        unique[key] = {"camera": row["camera"], "sample_index": row["sample_index"],
                       "image": row["image"]}
    return [unique[key] for key in sorted(unique)]


def extract(dataset: Path, model_path: Path, output: Path, *, imgsz: int, batch: int,
            confidence: float, device: str, chunk_size: int) -> dict:
    from ultralytics import YOLO

    dataset, model_path, output = dataset.resolve(), model_path.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    staged = output.with_name(output.name + ".incomplete")
    staged.mkdir(parents=True, exist_ok=True)
    records_path = staged / "contours.csv"
    done: set[tuple[str, str]] = set()
    if records_path.exists():
        with records_path.open(newline="", encoding="utf-8") as handle:
            done = {(row["camera"], row["sample_index"]) for row in csv.DictReader(handle)}
    sources = _source_images(dataset)
    pending = [row for row in sources if (row["camera"], row["sample_index"]) not in done]
    model = YOLO(str(model_path))
    mode = "a" if records_path.exists() else "w"
    with records_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            results = model.predict(
                source=[row["image"] for row in chunk], imgsz=imgsz, conf=confidence,
                batch=batch, device=device, verbose=False,
            )
            if len(results) != len(chunk):
                raise RuntimeError(f"expected {len(chunk)} predictions, received {len(results)}")
            for source, result in zip(chunk, results):
                row = dict(source, detected=0, confidence="", box_x1="", box_y1="",
                           box_x2="", box_y2="", polygon_json="", polygon_points=0)
                boxes = result.boxes
                polygons = None if result.masks is None else result.masks.xy
                if boxes is not None and len(boxes) and polygons is not None:
                    scores = boxes.conf.detach().cpu().numpy()
                    best = int(np.argmax(scores))
                    if best < len(polygons):
                        polygon = np.asarray(polygons[best], dtype=float)
                        polygon = polygon[np.all(np.isfinite(polygon[:, :2]), axis=1), :2]
                        if len(polygon) >= 3:
                            x1, y1, x2, y2 = boxes.xyxy.detach().cpu().numpy()[best, :4]
                            row.update({
                                "detected": 1, "confidence": f"{scores[best]:.10f}",
                                "box_x1": f"{x1:.10f}", "box_y1": f"{y1:.10f}",
                                "box_x2": f"{x2:.10f}", "box_y2": f"{y2:.10f}",
                                "polygon_json": json.dumps(polygon.tolist(), separators=(",", ":")),
                                "polygon_points": len(polygon),
                            })
                writer.writerow(row)
            handle.flush()
            print(f"contours: {len(done) + min(start + len(chunk), len(pending))}/{len(sources)}", flush=True)
    with records_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(sources):
        raise RuntimeError(f"incomplete contour extraction: {len(rows)} of {len(sources)}")
    payload = {
        "status": "complete_provisional_rgb_contour_predictions",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "model": str(model_path), "model_sha256": _sha256(model_path),
        "inference": {"imgsz": imgsz, "batch": batch, "confidence": confidence, "device": device},
        "counts": {"images": len(rows), "with_predicted_contour": sum(int(r["detected"]) for r in rows)},
        "online_input": "RGB image only", "records_sha256": _sha256(records_path),
    }
    manifest = staged / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staged / ".complete").write_text(json.dumps({"manifest_sha256": _sha256(manifest)}) + "\n")
    os.replace(staged, output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(extract(args.dataset, args.model, args.out, imgsz=args.imgsz,
                             batch=args.batch, confidence=args.confidence, device=args.device,
                             chunk_size=args.chunk_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
