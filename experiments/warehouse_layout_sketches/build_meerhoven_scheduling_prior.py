#!/usr/bin/env python3
"""Build the geometry-only A--L map used to bootstrap camera scheduling.

This artifact ranks cameras before learned GP maps exist. It is not detector
evidence and must not be used as the final planner reliability artifact.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
LAYOUT = HERE.with_name("exp4_meerhoven_hub.py")
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_meerhoven.world.sdf"
DEFAULT_OUTPUT = (
    REPO / "logs/studies/warehouse_layout_sketches/meerhoven_scheduling_prior_v1/"
    "meerhoven_geometry_12cam.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_layout():
    spec = importlib.util.spec_from_file_location("meerhoven_layout_contract", LAYOUT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {LAYOUT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    layout = _load_layout()
    xs, ys, gx, gy, drivable = layout.build_grid()
    payload: dict[str, np.ndarray] = {
        "xs": np.asarray(xs, dtype=float),
        "ys": np.asarray(ys, dtype=float),
        "drivable_mask": np.asarray(drivable, dtype=np.uint8),
    }
    per_camera = []
    for letter, camera in zip("ABCDEFGHIJKL", layout.CAMERAS, strict=True):
        name, x, y, z, yaw, _provenance, _note = camera
        model = layout.camera_model(
            x, y, z, yaw, pitch_deg=layout.CAMERA_PITCH.get(name)
        )
        seen = layout.in_frame(model, gx, gy) & layout.line_of_sight(model, gx, gy) & drivable
        probability = np.where(seen, 0.95, 0.005).astype(float)
        payload[f"P_camera_{letter}_map"] = probability
        per_camera.append(probability)
    stacked = np.stack(per_camera, axis=0)
    payload.update({
        "P_union_12cam_map": np.max(stacked, axis=0),
        "best_camera_index_map": np.argmax(stacked, axis=0).astype(np.int16),
        "camera_ids": np.asarray(
            [f"camera_{letter}" for letter in "ABCDEFGHIJKL"], dtype=np.str_
        ),
        "kind": np.asarray(["geometry_only_camera_scheduling_prior"], dtype=np.str_),
        "evidence_status": np.asarray(
            ["bootstrap_only_not_learned_not_planner_evidence"], dtype=np.str_
        ),
        "world_sdf": np.asarray([str(WORLD)], dtype=np.str_),
        "world_sha256": np.asarray([_sha256(WORLD)], dtype=np.str_),
        "layout_source": np.asarray([str(LAYOUT)], dtype=np.str_),
        "layout_source_sha256": np.asarray([_sha256(LAYOUT)], dtype=np.str_),
    })
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    np.savez_compressed(output, **payload)
    return output


if __name__ == "__main__":
    path = build()
    print(path)
