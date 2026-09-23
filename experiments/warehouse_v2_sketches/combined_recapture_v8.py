#!/usr/bin/env python3
"""Read the v8 uniform reference dataset: all of v5, plus two small extensions.

v8 is an INDEX, not a copy. Nothing from v5 is removed. Its rows are

* every ok v5 row (both halves, via combined_recapture_v5);
* the 12 camera-C supplement positions captured on 2026-09-23
  (`recapture_v8_uniform/master_capture_supplement`, keys `Qnnnn`);
* the 50 uniform top-up positions (`recapture_v8_uniform/master_capture_topup`, keys
  `Unnnn`), planned by `plan_uniform_topup.py`: every 1 m cell of the operating domain
  below 8 positions per m^2 (scaled by its footprint-valid fraction) is filled to target;
* the repair capture (`recapture_v8_uniform/master_capture_repair`) of the 4,182 v5 part-1
  poses listed in `v5_robot_absent_poses.json`. In v5 session 71daa8ab the robot vanished
  from the simulator at pose 4214 and every later frame is empty background, although the
  rows were marked ok. Those v5 rows are dropped and the repaired rows take their place,
  under the same plan pose index, position key and role. The repair ran in passes
  (`master_capture_repair`, then `master_capture_repair2` after a laptop sleep broke the
  first pass); a later pass overrides an earlier one for every plan pose it contains.

Every pose is also checked for presence: a pose where at least two cameras should see the
robot but none does is legitimate at one position (racks can hide all its headings), but
a run of such poses spanning three or more positions within one capture session means the
robot was absent, and every pose in the run is rejected. `load_rows` raises if any such
run remains.

Every added position takes the role of its nearest v5 position, so near-duplicate views
never cross roles, and none is next to a final_audit position, so the sealed audit set is
the v5 one. Added poses sit after the v5 plan in `plan_pose_index` (10276 + offset), and
their keys cannot collide with v5's `Pnnnn`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from . import combined_recapture_v5 as v5
except ImportError:  # imported as a top-level module from sys.path
    import combined_recapture_v5 as v5

REPO = Path(__file__).resolve().parents[2]
L = REPO / "logs/thesis_final_pipeline_v1"
V5_POSITIONS = L / "recapture_v5/capture_positions_v5.csv"
V5_PLAN_POSES = 10276
# (name, capture dir, pose file, plan_pose_index offset)
EXTENSIONS = (
    ("supplement", L / "recapture_v8_uniform/master_capture_supplement",
     L / "recapture_v8_uniform/capture_poses_supplement.json", V5_PLAN_POSES),
    ("topup", L / "recapture_v8_uniform/master_capture_topup",
     L / "recapture_v8_uniform/capture_poses_v8_topup.json", V5_PLAN_POSES + 48),
)
# Repair passes in order; a later pass overrides an earlier one for the plan poses it holds.
REPAIRS = (
    ("repair", L / "recapture_v8_uniform/master_capture_repair",
     L / "recapture_v8_uniform/capture_poses_repair.json"),
    ("repair2", L / "recapture_v8_uniform/master_capture_repair2",
     L / "recapture_v8_uniform/capture_poses_repair2.json"),
)
ABSENT_LIST = L / "recapture_v8_uniform/v5_robot_absent_poses.json"
SOURCES = (v5.SOURCES + tuple((name, directory, poses) for name, directory, poses, _ in EXTENSIONS)
           + REPAIRS)
_DEFAULTS = {"anchor": -1, "n_cameras": -1}


def _nearest_v5_role() -> Callable[[float, float], str]:
    rows = list(csv.DictReader(V5_POSITIONS.open(encoding="utf-8")))
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    roles = [r["role"] for r in rows]

    def role(x: float, y: float) -> str:
        return roles[int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))]
    return role


def _extension_rows(only_ok: bool) -> list[dict]:
    role_of = _nearest_v5_role()
    out = []
    for name, directory, pose_file, offset in EXTENSIONS:
        poses = json.loads(pose_file.read_text(encoding="utf-8"))
        with (directory / "capture_index.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if only_ok and row.get("capture_status") != "ok":
                    continue
                i = int(row["pose_id"])
                p = poses[i]
                if (abs(float(row["robot_x"]) - p["x"]) > 1e-6
                        or abs(float(row["robot_y"]) - p["y"]) > 1e-6):
                    raise RuntimeError(f"{name} pose_id {i} does not match its pose file")
                role = role_of(p["x"], p["y"])
                if role == "final_audit":
                    raise RuntimeError(f"{name} position {p['position_key']} borders final_audit")
                row = dict(row)
                row.update(_DEFAULTS)
                row.update({
                    "capture_source": name, "position_key": p["position_key"],
                    "yaw_idx": int(p["yaw_idx"]), "plan_pose_index": offset + i,
                    "stratum": role, "kind": p.get("kind", name), "block_id": name,
                })
                out.append(row)
    return out


def _repair_rows(only_ok: bool) -> list[dict]:
    passes = []
    for name, directory, pose_file in REPAIRS:
        poses = json.loads(pose_file.read_text(encoding="utf-8"))
        rows = []
        index = directory / "capture_index.csv"
        if index.is_file():
            with index.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if only_ok and row.get("capture_status") != "ok":
                        continue
                    p = poses[int(row["pose_id"])]
                    if (abs(float(row["robot_x"]) - p["x"]) > 1e-6
                            or abs(float(row["robot_y"]) - p["y"]) > 1e-6):
                        raise RuntimeError(f"{name} pose_id {row['pose_id']} does not match its pose file")
                    row = dict(row)
                    row.update({
                        "capture_source": name, "position_key": p["position_key"],
                        "yaw_idx": int(p["yaw_idx"]), "plan_pose_index": int(p["v5_plan_pose_index"]),
                    })
                    for f in ("stratum", "kind", "anchor", "block_id", "n_cameras"):
                        if f in p:
                            row[f] = p[f]
                    rows.append(row)
        passes.append(({int(p["v5_plan_pose_index"]) for p in poses}, rows))
    out = []
    for i, (_, rows) in enumerate(passes):
        overridden = set().union(*(planned for planned, _ in passes[i + 1:]))
        out += [r for r in rows if int(r["plan_pose_index"]) not in overridden]
    return out


def _presence_rejects(rows: list[dict], run_positions: int = 3, min_in_frame: int = 2) -> set:
    """(capture_source, plan_pose_index) of every pose inside a run of empty poses that
    spans at least `run_positions` distinct positions."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["capture_source"], r["capture_session_id"],
                           int(r["pose_id"]), int(r["plan_pose_index"])), []).append(r)
    rejected, run, last = set(), [], None
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        rs = groups[key]
        in_frame = sum(r.get("nominal_in_frame") in ("1", "True") for r in rs)
        visible = sum(int(float(r.get("semantic_robot_pixels") or 0)) > 0 for r in rs)
        if (key[0], key[1]) != last:
            run, last = [], (key[0], key[1])
        if in_frame >= min_in_frame and visible == 0:
            run.append((key[0], key[3], rs[0]["position_key"]))
            if len({position for _, _, position in run}) >= run_positions:
                rejected.update((source, pose) for source, pose, _ in run)
        else:
            run = []
    return rejected


def load_rows(*, only_ok: bool = True) -> list[dict]:
    absent = set(json.loads(ABSENT_LIST.read_text(encoding="utf-8"))["plan_pose_indices"])
    rows = [r for r in v5.load_rows(only_ok=only_ok)
            if not (r["capture_source"] == "part1" and int(r["plan_pose_index"]) in absent)]
    rows += _extension_rows(only_ok) + _repair_rows(only_ok)
    rejected = _presence_rejects(rows)
    if rejected:
        raise RuntimeError(f"{len(rejected)} poses lie in robot-absent runs; re-capture them "
                           "before using the dataset")
    v5._check_unique(rows)
    return rows


def load_records(paths: dict[str, Path]) -> list[dict]:
    raise NotImplementedError("v8 stages read inference records through the gate dataset")


def image_path(row: dict) -> Path:
    capdir = dict((n, d) for n, d, _ in SOURCES)[row["capture_source"]]
    return capdir / row["image"]


def coverage() -> dict:
    from collections import Counter
    rows = load_rows()
    positions = {r["position_key"]: r["stratum"] for r in rows}
    return {
        "rows": len(rows),
        "rows_by_source": dict(Counter(r["capture_source"] for r in rows)),
        "positions_by_role": dict(Counter(positions.values())),
        "unique_images_working": len({r["image_sha1"] for r in rows
                                      if r["stratum"] in ("D_mu", "D_R", "D_dev")}),
    }


if __name__ == "__main__":
    print(json.dumps(coverage(), indent=2))
