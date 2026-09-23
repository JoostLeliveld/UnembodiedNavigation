#!/usr/bin/env python3
"""Read the v7 balanced reference dataset: v5 thinned, plus the camera C supplement.

v7 is an INDEX, not a copy. Its rows are

* every v5 row (both halves, via combined_recapture_v5) whose position is not in the
  thinning list of `recapture_v7_balanced/thinned_positions.json`, and
* every ok row of `recapture_v7_balanced/master_capture_supplement`, captured against
  `capture_poses_v7_supplement.json`.

Supplement poses sit after the v5 plan: their `plan_pose_index` is 10276 + their index in
the supplement pose file, and their position keys are `Qnnnn`, so they can never collide
with a v5 pose. The thinning rule and the supplement positions are produced by
`plot_intended_dataset.py` and recorded in the lock amendment of 2026-09-23.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    from . import combined_recapture_v5 as v5
except ImportError:  # imported as a top-level module from sys.path
    import combined_recapture_v5 as v5

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "logs/thesis_final_pipeline_v1/recapture_v7_balanced"
THINNED = ROOT / "thinned_positions.json"
SUPPLEMENT_DIR = ROOT / "master_capture_supplement"
SUPPLEMENT_POSES = ROOT / "capture_poses_v7_supplement.json"
V5_PLAN_POSES = 10276

SOURCES = v5.SOURCES + (("supplement", SUPPLEMENT_DIR, SUPPLEMENT_POSES),)
_PLAN_FIELDS = ("stratum", "kind", "anchor", "block_id", "n_cameras")
# v5 plan fields a supplement pose does not have. Downstream records copy them verbatim,
# so they are set explicitly rather than left missing.
_SUPPLEMENT_DEFAULTS = {"anchor": -1, "block_id": "supplement_camera_c", "n_cameras": -1}


def removed_positions() -> set[str]:
    return set(json.loads(THINNED.read_text(encoding="utf-8"))["remove"])


def _supplement_rows(only_ok: bool) -> list[dict]:
    poses = json.loads(SUPPLEMENT_POSES.read_text(encoding="utf-8"))
    out = []
    with (SUPPLEMENT_DIR / "capture_index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if only_ok and row.get("capture_status") != "ok":
                continue
            i = int(row["pose_id"])
            p = poses[i]
            if (abs(float(row["robot_x"]) - p["x"]) > 1e-6
                    or abs(float(row["robot_y"]) - p["y"]) > 1e-6):
                raise RuntimeError(f"supplement pose_id {i} does not match its pose file")
            row = dict(row)
            row["capture_source"] = "supplement"
            row["position_key"] = p["position_key"]
            row["yaw_idx"] = int(p["yaw_idx"])
            row["plan_pose_index"] = V5_PLAN_POSES + i
            row.update(_SUPPLEMENT_DEFAULTS)
            for f in _PLAN_FIELDS:
                if f in p:
                    row[f] = p[f]
            out.append(row)
    return out


def load_rows(*, only_ok: bool = True) -> list[dict]:
    drop = removed_positions()
    rows = [r for r in v5.load_rows(only_ok=only_ok) if r["position_key"] not in drop]
    rows += _supplement_rows(only_ok)
    v5._check_unique(rows)
    return rows


def load_records(paths: dict[str, Path]) -> list[dict]:
    drop = removed_positions()
    v5_paths = {k: p for k, p in paths.items() if k != "supplement"}
    out = [r for r in v5.load_records(v5_paths) if r["position_key"] not in drop]
    if "supplement" in paths:
        poses = json.loads(SUPPLEMENT_POSES.read_text(encoding="utf-8"))
        with Path(paths["supplement"]).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = dict(json.loads(line))
                p = poses[int(rec["pose_id"])]
                rec["capture_source"] = "supplement"
                rec["position_key"] = p["position_key"]
                rec["yaw_idx"] = int(p["yaw_idx"])
                rec["plan_pose_index"] = V5_PLAN_POSES + int(rec["pose_id"])
                rec.update(_SUPPLEMENT_DEFAULTS)
                for f in _PLAN_FIELDS:
                    if f in p:
                        rec[f] = p[f]
                out.append(rec)
    v5._check_unique(out)
    return out


def image_path(row: dict) -> Path:
    capdir = dict((n, d) for n, d, _ in SOURCES)[row["capture_source"]]
    return capdir / row["image"]


def coverage() -> dict:
    from collections import Counter
    rows = load_rows()
    positions = {}
    for r in rows:
        positions[r["position_key"]] = r["stratum"]
    return {
        "rows": len(rows),
        "rows_by_source": dict(Counter(r["capture_source"] for r in rows)),
        "positions_by_role": dict(Counter(positions.values())),
        "unique_images_working": len({r["image_sha1"] for r in rows
                                      if r["stratum"] in ("D_mu", "D_R", "D_dev")}),
    }


if __name__ == "__main__":
    print(json.dumps(coverage(), indent=2))
