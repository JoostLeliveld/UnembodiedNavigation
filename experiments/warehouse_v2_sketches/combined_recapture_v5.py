#!/usr/bin/env python3
"""Read the v5 master recapture -- both halves -- as ONE dataset.

The capture ran in two directories.  Part 2 covers the 1869 poses part 1 never
reached plus the 530 it lost to `set_pose failed` when the simulator died, and
it was launched with its OWN pose file, so `capture_bbox_grid.py` renumbered
`pose_id` and `position_id` from 0.  The two halves therefore collide in every
per-capture id: part 2's pose_id 0..1528 all exist in part 1 as well, naming
completely different poses.

This module is the single place that resolves that.  Unlike the v3/v4 combiner
it does NOT offset the ids, because the halves are not two datasets stacked end
to end -- they are two passes over ONE plan, `capture_poses_v5.json`.  The join
is therefore back onto that plan:

    pose_id -> index into the half's own pose file -> (position_key, yaw_idx)

which is the canonical identity a pose has in the v5 plan regardless of which
half captured it.  That mapping is exact: every row's robot_x/robot_y matches
its pose-file entry (verified over all 49,720 rows captured so far).

Rows carry four added fields:

    capture_source    'part1' or 'part2'
    position_key      canonical 'Pnnnn' from the v5 plan
    yaw_idx           canonical heading index within that position
    plan_pose_index   index into capture_poses_v5.json  (the global pose id)

plus the plan's own `stratum`, `kind`, `anchor` and `block_id`, so the D_mu /
D_R / D_dev role travels with the row and consumers never re-derive it.

The originals are left untouched.  Rows whose capture_status is not 'ok' are
dropped by default -- part 1's 2650 failed rows are exactly what part 2 re-ran.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "logs/thesis_final_pipeline_v1/recapture_v5"

PLAN = ROOT / "capture_poses_v5.json"

# (name, capture dir, the pose file that capture was LAUNCHED with)
SOURCES = (
    ("part1", ROOT / "master_capture",       ROOT / "capture_poses_v5.json"),
    ("part2", ROOT / "master_capture_part2", ROOT / "capture_poses_v5_part2.json"),
)

# plan fields copied onto every row
_PLAN_FIELDS = ("stratum", "kind", "anchor", "block_id", "n_cameras")


def _load_poses(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _plan_index() -> dict[tuple[str, int], int]:
    """(position_key, yaw_idx) -> index into the v5 plan."""
    return {(p["position_key"], int(p["yaw_idx"])): i
            for i, p in enumerate(_load_poses(PLAN))}


def load_rows(*, only_ok: bool = True,
              sources: tuple[str, ...] = ("part1", "part2")) -> list[dict]:
    """Capture-index rows from both halves, keyed back onto the v5 plan."""
    plan = _plan_index()
    out: list[dict] = []
    for name, capdir, posefile in SOURCES:
        if name not in sources:
            continue
        index = capdir / "capture_index.csv"
        if not index.is_file():
            raise FileNotFoundError(index)
        poses = _load_poses(posefile)
        with index.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if only_ok and row.get("capture_status") != "ok":
                    continue
                i = int(row["pose_id"])
                if i >= len(poses):
                    raise RuntimeError(f"{name}: pose_id {i} outside its pose file")
                p = poses[i]
                # The pose file is the authority for where the robot stood; the
                # row must agree with it or the renumbering assumption is wrong.
                if (abs(float(row["robot_x"]) - p["x"]) > 1e-6
                        or abs(float(row["robot_y"]) - p["y"]) > 1e-6):
                    raise RuntimeError(
                        f"{name}: row pose_id {i} at ({row['robot_x']},{row['robot_y']}) "
                        f"does not match pose file ({p['x']},{p['y']})")
                key = (p["position_key"], int(p["yaw_idx"]))
                if key not in plan:
                    raise RuntimeError(f"{name}: pose {key} is not in the v5 plan")
                row = dict(row)
                row["capture_source"] = name
                row["position_key"] = p["position_key"]
                row["yaw_idx"] = int(p["yaw_idx"])
                row["plan_pose_index"] = plan[key]
                for f in _PLAN_FIELDS:
                    if f in p:
                        row[f] = p[f]
                out.append(row)
    _check_unique(out)
    return out


def load_records(paths: dict[str, Path]) -> list[dict]:
    """Frozen-detector inference records keyed by half, e.g. {'part1': ...}."""
    plan = _plan_index()
    posefiles = {name: pf for name, _, pf in SOURCES}
    out: list[dict] = []
    for name, path in paths.items():
        if name not in posefiles:
            raise KeyError(f"unknown capture source {name!r}")
        poses = _load_poses(posefiles[name])
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                p = poses[int(rec["pose_id"])]
                key = (p["position_key"], int(p["yaw_idx"]))
                rec = dict(rec)
                rec["capture_source"] = name
                rec["position_key"] = p["position_key"]
                rec["yaw_idx"] = int(p["yaw_idx"])
                rec["plan_pose_index"] = plan[key]
                for f in _PLAN_FIELDS:
                    if f in p:
                        rec[f] = p[f]
                out.append(rec)
    _check_unique(out)
    return out


def _check_unique(rows: list[dict]) -> None:
    """A (plan pose, camera) may be captured once.  The halves are disjoint by
    construction -- part 2 only ran poses part 1 has no ok row for -- so a
    duplicate here means the two directories overlap and the merge is unsafe."""
    seen: dict[tuple[int, str], str] = {}
    for r in rows:
        k = (r["plan_pose_index"], r["camera_id"])
        if k in seen:
            raise RuntimeError(
                f"plan pose {r['plan_pose_index']} camera {r['camera_id']} captured "
                f"in both {seen[k]} and {r['capture_source']}")
        seen[k] = r["capture_source"]


def image_path(row: dict) -> Path:
    """Absolute path to a row's RGB image, in whichever half it came from."""
    capdir = dict((n, d) for n, d, _ in SOURCES)[row["capture_source"]]
    return capdir / row["image"]


def coverage() -> dict:
    """How much of the v5 plan is captured, by half and by stratum."""
    from collections import Counter
    plan = _load_poses(PLAN)
    rows = load_rows()
    got = {r["plan_pose_index"] for r in rows}
    by_src = Counter(r["capture_source"] for r in rows)
    plan_by_stratum = Counter(p["stratum"] for p in plan)
    got_by_stratum = Counter(plan[i]["stratum"] for i in got)
    return {
        "plan_poses": len(plan),
        "captured_poses": len(got),
        "missing_poses": len(plan) - len(got),
        "rows": len(rows),
        "rows_by_source": dict(by_src),
        "poses_by_stratum": {s: (got_by_stratum.get(s, 0), plan_by_stratum[s])
                             for s in sorted(plan_by_stratum)},
    }


if __name__ == "__main__":
    print(json.dumps(coverage(), indent=2))
