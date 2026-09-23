#!/usr/bin/env python3
"""The reference-position dataset (v8): one loader for every pipeline stage.

v8 is an INDEX over the capture passes, not a copy. Its rows are

* every ok row of the v5 capture, which ran in two passes over one plan
  (`capture_poses_v5.json`): part 1, and part 2 for the poses part 1 never reached or lost.
  Part 2 was launched with its own pose file, so `pose_id` restarts at 0 there; rows are
  joined back onto the plan as pose_id -> own pose file -> (position_key, yaw_idx);
* the 12 camera-C supplement positions (keys `Qnnnn`);
* the 50 uniform top-up positions (keys `Unnnn`, planned by `capture/plan_topup.py`): every
  1 m cell of the operating domain below 8 positions per m^2, scaled by its footprint-valid
  fraction, is filled to target;
* the repair capture of the 4,182 v5 part-1 poses in `v5_robot_absent_poses.json`: in v5
  session 71daa8ab the robot vanished from the simulator at pose 4214 and every later frame
  is empty background although marked ok. Those rows are dropped and the repaired rows take
  their place under the same plan pose index, position key and role. The repair ran in two
  passes; a later pass overrides an earlier one for every plan pose it contains.

Every added position takes the role of its nearest v5 position (none borders final_audit,
so the sealed audit set is the v5 one) and sits after the v5 plan in `plan_pose_index`.

Presence check: a pose where at least two cameras should see the robot but none does is
legitimate at one position (racks can hide all its headings), but such a run spanning three
or more positions within one capture session means the robot was absent. `load_rows`
raises if any such run remains, and if a (plan pose, camera) is captured twice.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
L = REPO / "logs/thesis_final_pipeline_v1"
V5 = L / "recapture_v5"
V8 = L / "recapture_v8_uniform"

V5_PLAN = V5 / "capture_poses_v5.json"
V5_POSITIONS = V5 / "capture_positions_v5.csv"
V5_PLAN_POSES = 10276
# (name, capture dir, the pose file that capture was launched with)
V5_PASSES = (
    ("part1", V5 / "master_capture", V5 / "capture_poses_v5.json"),
    ("part2", V5 / "master_capture_part2", V5 / "capture_poses_v5_part2.json"),
)
# (name, capture dir, pose file, plan_pose_index offset)
EXTENSIONS = (
    ("supplement", V8 / "master_capture_supplement", V8 / "capture_poses_supplement.json",
     V5_PLAN_POSES),
    ("topup", V8 / "master_capture_topup", V8 / "capture_poses_v8_topup.json",
     V5_PLAN_POSES + 48),
)
# repair passes in order; a later pass overrides an earlier one for the plan poses it holds
REPAIRS = (
    ("repair", V8 / "master_capture_repair", V8 / "capture_poses_repair.json"),
    ("repair2", V8 / "master_capture_repair2", V8 / "capture_poses_repair2.json"),
)
ABSENT_LIST = V8 / "v5_robot_absent_poses.json"
SOURCES = V5_PASSES + tuple((n, d, p) for n, d, p, _ in EXTENSIONS) + REPAIRS

LOCK_PATH = REPO / "pipeline/dataset_lock.json"
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
CAMPAIGN_ROOT = REPO / LOCK["campaign_root"]
EXPECTED_WORKING_OPPORTUNITIES = int(LOCK["opportunity_accounting"]["expected_working_opportunities"])
EXPECTED_WORKING_UNIQUE_IMAGES = int(LOCK["opportunity_accounting"]["expected_working_unique_images"])

_PLAN_FIELDS = ("stratum", "kind", "anchor", "block_id", "n_cameras")
_DEFAULTS = {"anchor": -1, "n_cameras": -1}


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_pose(name: str, row: dict, pose: dict) -> None:
    """The pose file is the authority for where the robot stood."""
    if (abs(float(row["robot_x"]) - pose["x"]) > 1e-6
            or abs(float(row["robot_y"]) - pose["y"]) > 1e-6):
        raise RuntimeError(f"{name}: row pose_id {row['pose_id']} at "
                           f"({row['robot_x']},{row['robot_y']}) does not match its pose file "
                           f"({pose['x']},{pose['y']})")


def _index_rows(directory: Path, only_ok: bool):
    index = directory / "capture_index.csv"
    if not index.is_file():
        raise FileNotFoundError(index)
    with index.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if only_ok and row.get("capture_status") != "ok":
                continue
            yield row


def _v5_rows(only_ok: bool) -> list[dict]:
    plan = {(p["position_key"], int(p["yaw_idx"])): i for i, p in enumerate(_load_json(V5_PLAN))}
    out = []
    for name, directory, pose_file in V5_PASSES:
        poses = _load_json(pose_file)
        for row in _index_rows(directory, only_ok):
            i = int(row["pose_id"])
            if i >= len(poses):
                raise RuntimeError(f"{name}: pose_id {i} outside its pose file")
            p = poses[i]
            _check_pose(name, row, p)
            key = (p["position_key"], int(p["yaw_idx"]))
            if key not in plan:
                raise RuntimeError(f"{name}: pose {key} is not in the v5 plan")
            row = dict(row)
            row.update({"capture_source": name, "position_key": p["position_key"],
                        "yaw_idx": int(p["yaw_idx"]), "plan_pose_index": plan[key]})
            row.update({f: p[f] for f in _PLAN_FIELDS if f in p})
            out.append(row)
    return out


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
        poses = _load_json(pose_file)
        for row in _index_rows(directory, only_ok):
            i = int(row["pose_id"])
            p = poses[i]
            _check_pose(name, row, p)
            role = role_of(p["x"], p["y"])
            if role == "final_audit":
                raise RuntimeError(f"{name} position {p['position_key']} borders final_audit")
            row = dict(row)
            row.update(_DEFAULTS)
            row.update({"capture_source": name, "position_key": p["position_key"],
                        "yaw_idx": int(p["yaw_idx"]), "plan_pose_index": offset + i,
                        "stratum": role, "kind": p.get("kind", name), "block_id": name})
            out.append(row)
    return out


def _repair_rows(only_ok: bool) -> list[dict]:
    passes = []
    for name, directory, pose_file in REPAIRS:
        poses = _load_json(pose_file)
        rows = []
        if (directory / "capture_index.csv").is_file():
            for row in _index_rows(directory, only_ok):
                p = poses[int(row["pose_id"])]
                _check_pose(name, row, p)
                row = dict(row)
                row.update({"capture_source": name, "position_key": p["position_key"],
                            "yaw_idx": int(p["yaw_idx"]),
                            "plan_pose_index": int(p["v5_plan_pose_index"])})
                row.update({f: p[f] for f in _PLAN_FIELDS if f in p})
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


def _check_unique(rows: list[dict]) -> None:
    seen: dict[tuple[int, str], str] = {}
    for r in rows:
        k = (int(r["plan_pose_index"]), r["camera_id"])
        if k in seen:
            raise RuntimeError(f"plan pose {k[0]} camera {k[1]} captured in both "
                               f"{seen[k]} and {r['capture_source']}")
        seen[k] = r["capture_source"]


def load_rows(*, only_ok: bool = True) -> list[dict]:
    """Every dataset row, keyed onto the plan, with capture_source, position_key, yaw_idx,
    plan_pose_index and the role (`stratum`) attached."""
    absent = set(_load_json(ABSENT_LIST)["plan_pose_indices"])
    rows = [r for r in _v5_rows(only_ok)
            if not (r["capture_source"] == "part1" and int(r["plan_pose_index"]) in absent)]
    rows += _extension_rows(only_ok) + _repair_rows(only_ok)
    rejected = _presence_rejects(rows)
    if rejected:
        raise RuntimeError(f"{len(rejected)} poses lie in robot-absent runs; re-capture them "
                           "before using the dataset")
    _check_unique(rows)
    return rows


def image_path(row: dict) -> Path:
    """Absolute path to a row's RGB image, in whichever capture pass it came from."""
    return dict((n, d) for n, d, _ in SOURCES)[row["capture_source"]] / row["image"]


def coverage() -> dict:
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
