#!/usr/bin/env python3
"""Derive a one-keypoint YOLO-pose dataset from the frozen shared RGB capture.

The learned observation is an RGB image.  Its target is the image projection of a
virtual point directly above ``base_link`` at a known height.  Back-projecting that
point onto the same horizontal plane recovers the robot centre without asking a
network to learn camera-to-world geometry.

The source capture already contains exact commanded poses, surveyed camera models,
semantic masks, and spatially grouped train/validation splits.  Images are symlinked;
only pose labels and provenance are newly written.  Existing output is never replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml


CAMERA_DIRS = ("camera_A", "camera_B", "camera_C", "camera_D", "camera_E")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _look_at(pose: list[float]) -> tuple[float, float, float]:
    """Gazebo camera include pose -> the point where its optical axis meets z=0."""
    x, y, z, _roll, pitch, yaw = (float(value) for value in pose)
    forward = (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    )
    if forward[2] >= -1.0e-9:
        raise ValueError(f"camera does not look down: {pose}")
    scale = -z / forward[2]
    return x + scale * forward[0], y + scale * forward[1], 0.0


def project_point(
    xyz: tuple[float, float, float],
    *,
    camera_pose_xyz_rpy: list[float],
    image_width: int,
    image_height: int,
    fov_h_rad: float,
) -> tuple[float, float, bool]:
    """Project one world point using the repository's oblique-camera convention."""
    import numpy as np

    camera = np.asarray(camera_pose_xyz_rpy[:3], dtype=float)
    target = np.asarray(_look_at(camera_pose_xyz_rpy), dtype=float)
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.asarray((right, down, forward))
    camera_point = rotation @ (np.asarray(xyz, dtype=float) - camera)
    if camera_point[2] <= 1.0e-9:
        return float("nan"), float("nan"), False
    focal = (float(image_width) / 2.0) / math.tan(float(fov_h_rad) / 2.0)
    u = focal * camera_point[0] / camera_point[2] + float(image_width) / 2.0
    v = focal * camera_point[1] / camera_point[2] + float(image_height) / 2.0
    inside = 0.0 <= u < image_width and 0.0 <= v < image_height
    return float(u), float(v), bool(inside)


def pose_label(
    bbox: tuple[float, float, float, float] | None,
    keypoint: tuple[float, float] | None,
    *,
    visibility: int,
    width: int,
    height: int,
) -> tuple[str, bool]:
    """Return a YOLO pose row and whether the visible box needed expansion.

    The detector box remains the semantic-mask extent, expanded only when necessary
    to contain the virtual centre.  This keeps assignment well-defined for strongly
    occluded robots whose visible fragment lies to one side of the true centre.
    """
    if bbox is None or keypoint is None:
        return "", False
    x0, y0, x1, y1 = (float(value) for value in bbox)
    u, v = (float(value) for value in keypoint)
    expanded = not (x0 <= u <= x1 and y0 <= v <= y1)
    x0, y0 = min(x0, u - 1.0), min(y0, v - 1.0)
    x1, y1 = max(x1, u + 1.0), max(y1, v + 1.0)
    x0, y0 = max(x0, 0.0), max(y0, 0.0)
    x1, y1 = min(x1, float(width)), min(y1, float(height))
    if not (x0 < x1 and y0 < y1 and 0 <= visibility <= 2):
        raise ValueError("invalid box, keypoint, or visibility")
    values = (
        0,
        0.5 * (x0 + x1) / width,
        0.5 * (y0 + y1) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
        u / width,
        v / height,
        visibility,
    )
    return (
        f"{values[0]} " + " ".join(f"{value:.8f}" for value in values[1:7])
        + f" {values[7]}\n",
        expanded,
    )


def _mask_hit(mask_path: Path, u: float, v: float, radius: int = 2) -> bool:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cannot decode semantic mask: {mask_path}")
    x, y = int(round(u)), int(round(v))
    window = mask[
        max(y - radius, 0):min(y + radius + 1, mask.shape[0]),
        max(x - radius, 0):min(x + radius + 1, mask.shape[1]),
    ]
    return bool(window.size and (window > 0).any())


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def build_dataset(source_root: Path, output: Path, *, centre_z_m: float = 0.35) -> dict:
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    staged = output.with_name(output.name + ".incomplete")
    if staged.exists():
        raise FileExistsError(f"staging directory already exists: {staged}")
    staged.mkdir(parents=True)

    split_by_pose: dict[tuple[str, str, str], str] = {}
    records: list[dict[str, object]] = []
    source_files: dict[str, dict[str, str]] = {}
    summary = {
        split: {"images": 0, "positives": 0, "negatives": 0, "visible": 0,
                "occluded": 0, "box_expanded": 0, "target_outside_image": 0}
        for split in ("train", "val")
    }
    try:
        for camera_name in CAMERA_DIRS:
            camera_root = source_root / camera_name
            manifest_path = camera_root / "dataset_manifest.json"
            diagnostics_path = camera_root / "label_diagnostics.csv"
            if not manifest_path.is_file() or not diagnostics_path.is_file():
                raise FileNotFoundError(f"incomplete source camera dataset: {camera_root}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pose = [float(value) for value in manifest["camera_pose_xyz_rpy"]]
            intrinsics = manifest["camera_intrinsics"]
            width, height = int(intrinsics["img_width"]), int(intrinsics["img_height"])
            source_files[camera_name] = {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "diagnostics": str(diagnostics_path),
                "diagnostics_sha256": _sha256(diagnostics_path),
            }

            with diagnostics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["accepted"] != "1":
                        continue
                    split = row["split"]
                    if split not in summary:
                        raise RuntimeError(f"unexpected split {split!r}")
                    pose_key = (row["robot_x"], row["robot_y"], row["robot_yaw"])
                    previous = split_by_pose.setdefault(pose_key, split)
                    if previous != split:
                        raise RuntimeError(
                            f"pose leakage across splits for {pose_key}: {previous} vs {split}"
                        )

                    source_image = camera_root / row["image"]
                    destination_name = f"{camera_name}__{source_image.name}"
                    destination_image = staged / "images" / split / destination_name
                    destination_label = staged / "labels" / split / Path(destination_name).with_suffix(".txt")
                    _symlink(source_image, destination_image)
                    destination_label.parent.mkdir(parents=True, exist_ok=True)

                    record: dict[str, object] = {
                        "image": str((output / "images" / split / destination_name)),
                        "split": split,
                        "camera": camera_name,
                        "sample_index": row["sample_index"],
                        "robot_x": row["robot_x"],
                        "robot_y": row["robot_y"],
                        "robot_yaw": row["robot_yaw"],
                        "range_m": row["camera_range_m"],
                        "occlusion_state": row["occlusion_state"],
                        "localization_qualified": row["localization_qualified"],
                        "camera_pose_xyz_rpy": json.dumps(pose, separators=(",", ":")),
                        "image_width": width,
                        "image_height": height,
                        "fov_h_rad": float(intrinsics["fov_h_rad"]),
                        "centre_z_m": float(centre_z_m),
                    }
                    summary[split]["images"] += 1
                    if row["sample_kind"] != "positive":
                        destination_label.write_text("", encoding="utf-8")
                        summary[split]["negatives"] += 1
                        record.update({"positive": 0, "visibility": 0, "centre_u": "", "centre_v": ""})
                        records.append(record)
                        continue

                    u, v, inside = project_point(
                        (float(row["robot_x"]), float(row["robot_y"]), float(centre_z_m)),
                        camera_pose_xyz_rpy=pose,
                        image_width=width,
                        image_height=height,
                        fov_h_rad=float(intrinsics["fov_h_rad"]),
                    )
                    if not inside:
                        # The image contains a clipped robot fragment but not its centre.
                        # It is useful as a negative for this centre-observation task.
                        destination_label.write_text("", encoding="utf-8")
                        summary[split]["negatives"] += 1
                        summary[split]["target_outside_image"] += 1
                        record.update({"positive": 0, "visibility": 0, "centre_u": u, "centre_v": v})
                        records.append(record)
                        continue

                    mask_path = camera_root / row["mask"]
                    visibility = 2 if _mask_hit(mask_path, u, v) else 1
                    bbox = tuple(float(row[key]) for key in (
                        "mask_bbox_x0", "mask_bbox_y0", "mask_bbox_x1", "mask_bbox_y1"
                    ))
                    # Diagnostics maxima are occupied pixel indices; YOLO boxes are half-open.
                    bbox = (bbox[0], bbox[1], bbox[2] + 1.0, bbox[3] + 1.0)
                    label, expanded = pose_label(
                        bbox, (u, v), visibility=visibility, width=width, height=height
                    )
                    destination_label.write_text(label, encoding="utf-8")
                    summary[split]["positives"] += 1
                    summary[split]["visible" if visibility == 2 else "occluded"] += 1
                    summary[split]["box_expanded"] += int(expanded)
                    record.update({
                        "positive": 1, "visibility": visibility,
                        "centre_u": f"{u:.10f}", "centre_v": f"{v:.10f}",
                    })
                    records.append(record)

        fields = list(records[0])
        records_path = staged / "records.csv"
        with records_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

        data = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "robot"},
            "kpt_shape": [1, 3],
            "flip_idx": [0],
            "task": "pose",
        }
        (staged / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        derived_manifest = {
            "status": "complete",
            "created_utc": _timestamp(),
            "task": "pose",
            "observation": "full RGB infrastructure-camera image",
            "target": {
                "name": "amodal_base_link_centre",
                "definition": f"projection of (robot_x, robot_y, z={centre_z_m:.3f} m)",
                "why_pixel_space": "camera geometry remains explicit at runtime",
                "visibility": "2 when a 5x5 neighbourhood overlaps the semantic robot mask; otherwise 1",
            },
            "box": "half-open semantic-mask box, minimally expanded to contain target",
            "split_contract": "source spatial-cell split; exact (x,y,yaw) may occur in only one split across all cameras",
            "source_root": str(source_root),
            "sources": source_files,
            "splits": summary,
            "records_sha256": _sha256(records_path),
        }
        manifest_out = staged / "dataset_manifest.json"
        manifest_out.write_text(
            json.dumps(derived_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staged / ".complete").write_text(
            json.dumps({"status": "complete", "manifest_sha256": _sha256(manifest_out)}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, output)
        return derived_manifest
    except BaseException:
        # Preserve the incomplete derivation for diagnosis; it cannot be mistaken for complete.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--centre-z-m", type=float, default=0.35)
    args = parser.parse_args()
    result = build_dataset(args.source_root, args.out, centre_z_m=args.centre_z_m)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
