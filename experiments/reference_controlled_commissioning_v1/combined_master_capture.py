#!/usr/bin/env python3
"""Read the v3 master capture and the v4 edge extension as ONE dataset.

The two captures are separate directories on purpose: v3 is frozen evidence and
was never re-run.  Their pose_id / position_id spaces both start at 0, so they
collide; this module is the single place that resolves that, rather than every
consumer inventing its own join.

Use `load_rows()` for capture-index rows and `load_records()` for frozen-detector
inference records.  Both return rows carrying three added fields:

    capture_source     'v3' or 'extension'
    global_position_id v3 position_id, or 400 + extension position_id
    global_pose_id     v3 pose_id, or 3200 + extension pose_id

The originals are left untouched, so anything keyed on the per-capture ids still
works.  Rows whose capture_status is not 'ok' are dropped by default.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

V3_CAPTURE = REPO / "logs/thesis_final_pipeline_v1/master_capture"
EXT_CAPTURE = REPO / "logs/thesis_final_pipeline_v1/master_capture_v4_edge_extension"

# v3 is the frozen base; the extension is numbered above it.
V3_POSITIONS = 400
V3_POSES = 3200

SOURCES = (
    ("v3", V3_CAPTURE, 0, 0),
    ("extension", EXT_CAPTURE, V3_POSITIONS, V3_POSES),
)


def _stamp(row: dict, source: str, pos_offset: int, pose_offset: int) -> dict:
    row = dict(row)
    row["capture_source"] = source
    row["global_position_id"] = int(row["position_id"]) + pos_offset
    row["global_pose_id"] = int(row["pose_id"]) + pose_offset
    return row


def load_rows(*, only_ok: bool = True, sources: tuple[str, ...] = ("v3", "extension")) -> list[dict]:
    """Capture-index rows from both captures, with global ids attached."""
    out: list[dict] = []
    for name, root, pos_offset, pose_offset in SOURCES:
        if name not in sources:
            continue
        index = root / "capture_index.csv"
        if not index.is_file():
            raise FileNotFoundError(index)
        with index.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if only_ok and row.get("capture_status") != "ok":
                    continue
                out.append(_stamp(row, name, pos_offset, pose_offset))
    _check_unique(out)
    return out


def load_records(paths: dict[str, Path]) -> list[dict]:
    """Inference records keyed by source name, e.g. {'v3': ..., 'extension': ...}."""
    offsets = {name: (p, q) for name, _, p, q in SOURCES}
    out: list[dict] = []
    for name, path in paths.items():
        if name not in offsets:
            raise KeyError(f"unknown capture source {name!r}")
        pos_offset, pose_offset = offsets[name]
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append(_stamp(json.loads(line), name, pos_offset, pose_offset))
    _check_unique(out)
    return out


def _check_unique(rows: list[dict]) -> None:
    seen = {(r["global_pose_id"], r["camera_id"]) for r in rows}
    if len(seen) != len(rows):
        raise RuntimeError("global (pose, camera) identity is not unique across captures")


def image_path(row: dict) -> Path:
    """Absolute path to a row's RGB image, in whichever capture it came from."""
    root = V3_CAPTURE if row["capture_source"] == "v3" else EXT_CAPTURE
    return (root / row["image"]).resolve()


def summary(rows: list[dict]) -> dict:
    import collections
    by = collections.Counter(r["capture_source"] for r in rows)
    return {
        "rows": len(rows),
        "by_source": dict(by),
        "positions": len({r["global_position_id"] for r in rows}),
        "poses": len({r["global_pose_id"] for r in rows}),
        "splits": dict(collections.Counter(r["dataset_split"] for r in rows)),
    }


if __name__ == "__main__":
    print(json.dumps(summary(load_rows()), indent=1))
