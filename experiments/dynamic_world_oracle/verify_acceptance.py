#!/usr/bin/env python3
"""Check a dynamic-world-oracle run against the acceptance criteria.

    python3 experiments/dynamic_world_oracle/verify_acceptance.py \
        --run  logs/studies/dynamic_world_oracle/s01_box_in_aisle/run01 \
        --repeat logs/studies/dynamic_world_oracle/s01_box_in_aisle/run02

Each check is stated as the claim it is defending, so a failure line reads as a
statement about the dataset rather than an assertion id.

1. output contract        every record carries the agreed fields
2. reproducible           two runs of one scenario produce byte-identical artifacts
3. partial occlusion      the obstacle hides some cells and leaves others visible
4. event timing           visibility changes at the event instants and nowhere else
5. camera synchronisation all four cameras deliver frames at the same simulated stamp
6. removal restores       taking the obstacle away returns the original oracle map
7. oracle is trustworthy  the fast ray cast agrees with the repo's scalar primitive
8. depth agrees           Gazebo's own depth buffer confirms each obstacle occlusion
9. boundary holds         nothing under src/ reads this study's ground truth

Check 8 is the only one that compares the oracle with something other than
itself, and it is the one that found a real defect: bounding each obstacle with a
single box made the ray cast claim occlusions the renderer disagreed with. Keep it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "scripts" / "shared"))

import oracle as ora  # noqa: E402
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)

CONTRACT_FIELDS = (
    "scenario_id", "timestamp", "camera_id", "rgb_path", "oracle_depth_path",
    "camera_intrinsics", "camera_extrinsics", "obstacle_state", "oracle_visibility_grid",
)


@dataclass
class Check:
    name: str
    claim: str
    passed: bool
    detail: str

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.claim}\n        {self.detail}"


def _load(run: Path) -> tuple[list[dict], dict, list[dict]]:
    records = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines() if line]
    manifest = json.loads((run / "manifest.json").read_text())
    rows = (run / "events.csv").read_text().splitlines()
    header = rows[0].split(",")
    events = [dict(zip(header, row.split(","))) for row in rows[1:] if row]
    return records, manifest, events


def _grids(run: Path, records: list[dict]) -> dict[tuple[float, str], np.ndarray]:
    return {
        (r["timestamp"], r["camera_id"]): np.load(run / r["oracle_visibility_grid"]["path"])
        for r in records
    }


def check_contract(records: list[dict]) -> Check:
    missing = sorted({f for r in records for f in CONTRACT_FIELDS if f not in r})
    return Check(
        "output contract", "every record carries the agreed fields",
        not missing,
        f"{len(records)} records, all {len(CONTRACT_FIELDS)} contract fields present"
        if not missing else f"missing fields: {missing}",
    )


def check_reproducible(run_a: Path, run_b: Path) -> Check:
    digests = {}
    for run in (run_a, run_b):
        path = run / "checksums.sha256"
        if not path.exists():
            return Check("reproducible", "two runs produce byte-identical artifacts",
                         False, f"{path} is missing")
        digests[run] = dict(
            (line.split("  ", 1)[1], line.split("  ", 1)[0])
            for line in path.read_text().splitlines() if line
        )
    a, b = digests[run_a], digests[run_b]
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    differing = sorted(f for f in set(a) & set(b) if a[f] != b[f])
    passed = not (only_a or only_b or differing)
    detail = (f"{len(a)} artifacts compared, all identical "
              f"(RGB, oracle depth, oracle visibility, records, events)")
    if not passed:
        detail = (f"{len(differing)} differ (e.g. {differing[:5]}); "
                  f"{len(only_a)} only in run A, {len(only_b)} only in run B")
    return Check("reproducible", "two runs of one scenario produce byte-identical artifacts",
                 passed, detail)


def check_partial_occlusion(records: list[dict], grids: dict) -> Check:
    """The obstacle must darken part of one camera's map and leave the rest alone."""
    times = sorted({r["timestamp"] for r in records})
    cameras = sorted({r["camera_id"] for r in records})
    baseline_t = next((t for t in times if not _obstacles_at(records, t)), None)
    if baseline_t is None:
        return Check("partial occlusion", "the obstacle hides some cells and leaves others visible",
                     False, "no capture instant without an obstacle to compare against")

    best = None
    for t in times:
        if not _obstacles_at(records, t):
            continue
        for camera in cameras:
            base, now = grids[(baseline_t, camera)], grids[(t, camera)]
            newly_hidden = int(((base == ora.VISIBLE) & (now != ora.VISIBLE)).sum())
            still_visible = int((now == ora.VISIBLE).sum())
            if newly_hidden > 0 and still_visible > 0:
                # Compare the VISIBLE mask, not the whole code grid: a cell whose
                # centre falls inside the obstacle turns OCCUPIED in every
                # camera's grid, because occupancy is a fact about the world
                # rather than about a viewpoint. Judging "unaffected" on the raw
                # codes would therefore say no camera is ever unaffected, which
                # is the opposite of what this check is asking.
                unaffected = [
                    c for c in cameras
                    if c != camera
                    and ((grids[(baseline_t, c)] == ora.VISIBLE)
                         == (grids[(t, c)] == ora.VISIBLE)).all()
                ]
                # Rank instants where OTHER cameras are untouched above instants
                # where the obstacle simply hides more: "some cells but not
                # others" is the claim, and a camera that sees no change at all
                # is the sharpest evidence of it.
                candidate = (len(unaffected), newly_hidden, still_visible, t, camera, unaffected)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    if best is None:
        return Check("partial occlusion", "the obstacle hides some cells and leaves others visible",
                     False, "no instant where an obstacle removed any visible cell")
    _n_unaffected, hidden, visible, t, camera, unaffected = best
    return Check(
        "partial occlusion", "the obstacle hides some cells and leaves others visible",
        True,
        f"at t={t:.1f}s {camera} loses {hidden} visible cells and keeps {visible}; "
        f"cameras with an unchanged map: {', '.join(unaffected) if unaffected else 'none'}",
    )


def _obstacles_at(records: list[dict], t: float) -> list[dict]:
    for r in records:
        if abs(r["timestamp"] - t) < 1e-9:
            return r["obstacle_state"]
    return []


def check_event_timing(records: list[dict], grids: dict, manifest: dict) -> Check:
    """Visibility must change across the event instants, and hold steady between them."""
    times = sorted({r["timestamp"] for r in records})
    cameras = sorted({r["camera_id"] for r in records})
    events = manifest["scenario"]["events"]

    def changed(t_prev: float, t_next: float) -> bool:
        return any(not (grids[(t_prev, c)] == grids[(t_next, c)]).all() for c in cameras)

    # windows during which the world is meant to be changing: an event instant, or
    # a move in progress
    def is_moving_window(t_prev: float, t_next: float) -> bool:
        for event in events:
            if event["kind"] == "move" and event["duration_s"] > 0:
                if t_prev < event["t"] + event["duration_s"] and t_next > event["t"]:
                    return True
        return False

    problems, confirmed = [], []
    for event in events:
        if event["kind"] == "stop":
            continue  # a stop holds the pose; by design nothing moves across it
        before = [t for t in times if t <= event["t"]]
        after = [t for t in times if t > event["t"]]
        if not before or not after:
            problems.append(f"{event['kind']}@{event['t']}s has no capture on both sides")
            continue
        t_prev, t_next = before[-1], after[0]
        if changed(t_prev, t_next):
            confirmed.append(f"{event['kind']}@{event['t']:.1f}s (seen between "
                             f"{t_prev:.1f}s and {t_next:.1f}s)")
        else:
            problems.append(f"{event['kind']}@{event['t']}s left the oracle unchanged")

    event_windows = {(t_prev, t_next)
                     for event in events
                     for t_prev in [max((t for t in times if t <= event["t"]), default=None)]
                     for t_next in [min((t for t in times if t > event["t"]), default=None)]
                     if t_prev is not None and t_next is not None}
    spurious = []
    for t_prev, t_next in zip(times, times[1:]):
        if (t_prev, t_next) in event_windows or is_moving_window(t_prev, t_next):
            continue
        if changed(t_prev, t_next):
            spurious.append(f"{t_prev:.1f}->{t_next:.1f}s")
    if spurious:
        problems.append(f"oracle changed with no event in flight at {', '.join(spurious)}")

    return Check(
        "event timing", "visibility changes at the event instants and nowhere else",
        not problems,
        "; ".join(confirmed) + (f"; PROBLEMS: {problems}" if problems else
                                "; no change at any other capture interval"),
    )


def check_camera_sync(records: list[dict]) -> Check:
    times = sorted({r["timestamp"] for r in records})
    cameras = sorted({r["camera_id"] for r in records})
    problems = []
    for t in times:
        at_t = [r for r in records if abs(r["timestamp"] - t) < 1e-9]
        if len(at_t) != len(cameras):
            problems.append(f"t={t}: {len(at_t)} of {len(cameras)} cameras")
        for r in at_t:
            if abs(r["rgb_stamp_s"] - t) > 1e-6 or abs(r["oracle_depth_stamp_s"] - t) > 1e-6:
                problems.append(
                    f"t={t} {r['camera_id']}: rgb@{r['rgb_stamp_s']} depth@{r['oracle_depth_stamp_s']}")
    return Check(
        "camera synchronisation", "all four cameras deliver frames at the same simulated stamp",
        not problems,
        f"{len(times)} instants x {len(cameras)} cameras; every RGB and depth frame carries "
        f"its instant's exact simulated stamp" if not problems else f"{problems[:5]}",
    )


def check_removal_restores(records: list[dict], grids: dict) -> Check:
    times = sorted({r["timestamp"] for r in records})
    cameras = sorted({r["camera_id"] for r in records})
    empty = [t for t in times if not _obstacles_at(records, t)]
    occupied = [t for t in times if _obstacles_at(records, t)]
    if not empty or not occupied:
        return Check("removal restores", "taking the obstacle away returns the original oracle map",
                     False, "run has no obstacle-free and obstacle-present instants to compare")
    baseline_t = empty[0]
    after = [t for t in empty if t > occupied[-1]]
    if not after:
        return Check("removal restores", "taking the obstacle away returns the original oracle map",
                     False, "no capture after the last obstacle was removed")
    mismatch = [(t, c) for t in after for c in cameras
                if not (grids[(t, c)] == grids[(baseline_t, c)]).all()]
    changed_meanwhile = any(
        not (grids[(t, c)] == grids[(baseline_t, c)]).all() for t in occupied for c in cameras
    )
    return Check(
        "removal restores", "taking the obstacle away returns the original oracle map",
        not mismatch and changed_meanwhile,
        f"maps at t={after[0]:.1f}-{after[-1]:.1f}s match the t={baseline_t:.1f}s baseline "
        f"cell-for-cell on all {len(cameras)} cameras, having differed while the obstacle was present"
        if not mismatch and changed_meanwhile else f"mismatches: {mismatch[:5]}",
    )


def check_oracle_agrees_with_primitive(manifest: dict, n_rays: int = 600) -> Check:
    """The fast slab test must match unav_common's scalar segment_occluded."""
    sys.path.insert(0, str(REPO / "src" / "unav_common"))
    from unav_common.occlusion_geometry import segment_occluded

    world = REPO / manifest["world_file"]
    scene = ora.OracleScene.from_world(world, manifest["cameras"])
    grid_meta = manifest["grid"]
    grid = ora.FloorGrid(grid_meta["xmin"], grid_meta["xmax"],
                         grid_meta["ymin"], grid_meta["ymax"], grid_meta["resolution_m"])
    points = grid.points_at_height(manifest["scenario"]["target_height_m"])
    rng = np.random.default_rng(manifest["scenario"]["seed"])
    index = rng.choice(points.shape[0], min(n_rays, points.shape[0]), replace=False)

    disagreements = 0
    for camera in scene.cameras.values():
        fast = ora.segments_hit_any_prism(camera.cam_pos, points[index], scene.static_prisms)
        slow = np.array([segment_occluded(scene.static_prisms, camera.cam_pos, p)
                         for p in points[index]])
        disagreements += int((fast != slow).sum())
    total = len(index) * len(scene.cameras)
    return Check(
        "oracle is trustworthy",
        "the fast ray cast agrees with the repo's scalar occlusion primitive",
        disagreements == 0,
        f"{total} rays cross-checked against unav_common.segment_occluded, "
        f"{disagreements} disagreements",
    )


def check_depth_agrees_with_the_oracle(run: Path, records: list[dict], manifest: dict) -> Check:
    """Cross-check the ray cast against Gazebo's own renderer.

    Every other check compares the oracle with itself. This one asks the
    simulator: for each cell the obstacle newly hides from a camera, the depth
    buffer along that pixel must come back *shorter* than the distance to the
    cell — i.e. the renderer also sees something in the way. If the ray cast and
    the render disagree, the ray cast is wrong.

    Restricted to cells hidden by the *obstacle*, deliberately. Cells hidden by
    racks are not a fair test: the oracle treats a rack as a solid prism while the
    renderer draws open ShelfD/E shelving you can partly see through, so the two
    legitimately differ there. That gap is reported as a number rather than
    asserted away.
    """
    sys.path.insert(0, str(REPO / "src" / "unav_common"))
    from unav_common.camera_model import ObliqueCameraModel

    grid_meta = manifest["grid"]
    grid = ora.FloorGrid(grid_meta["xmin"], grid_meta["xmax"],
                         grid_meta["ymin"], grid_meta["ymax"], grid_meta["resolution_m"])
    points = grid.points_at_height(manifest["scenario"]["target_height_m"])

    world = REPO / manifest["world_file"]
    scene = ora.OracleScene.from_world(world, manifest["cameras"])
    clear = ora.visibility_grids(scene.cameras, grid, scene.static_prisms,
                                 target_height_m=manifest["scenario"]["target_height_m"])

    checked = agreed = 0
    static_checked = static_agreed = 0
    worst = None
    for record in records:
        if not record["obstacle_state"]:
            continue
        camera_id = record["camera_id"]
        intr, extr = record["camera_intrinsics"], record["camera_extrinsics"]
        camera = ObliqueCameraModel(
            cam_pos=extr["cam_pos"], look_at=extr["look_at"],
            img_width=intr["img_width"], img_height=intr["img_height"],
            fov_h_rad=intr["fov_h_rad"])
        codes = np.load(run / record["oracle_visibility_grid"]["path"])
        depth = np.load(run / record["oracle_depth_path"])

        hidden_by_obstacle = ((clear[camera_id] == ora.VISIBLE) & (codes == ora.OCCLUDED)).ravel()
        still_visible = ((clear[camera_id] == ora.VISIBLE) & (codes == ora.VISIBLE)).ravel()

        cam_pts = (points - camera.cam_pos) @ camera.R.T
        planar = cam_pts[:, 2]
        u = camera.K[0, 0] * cam_pts[:, 0] / planar + camera.K[0, 2]
        v = camera.K[1, 1] * cam_pts[:, 1] / planar + camera.K[1, 2]
        inside = (planar > 0) & (u >= 0) & (u < intr["img_width"]) & (v >= 0) & (v < intr["img_height"])

        for mask, tally in ((hidden_by_obstacle, "obstacle"), (still_visible, "static")):
            index = np.flatnonzero(mask & inside)
            if index.size == 0:
                continue
            rendered = depth[v[index].astype(int), u[index].astype(int)]
            expected = planar[index]
            if tally == "obstacle":
                ok = rendered < expected - 1.0e-3        # something in front of the cell
                checked += index.size
                agreed += int(ok.sum())
                if not ok.all() and worst is None:
                    worst = (record["timestamp"], camera_id,
                             float(rendered[~ok][0]), float(expected[~ok][0]))
            else:
                ok = rendered > expected - 1.0e-3        # clear all the way to the cell
                static_checked += index.size
                static_agreed += int(ok.sum())

    if checked == 0:
        return Check("depth agrees with the ray cast",
                     "Gazebo's depth buffer confirms every obstacle-caused occlusion",
                     False, "no obstacle-occluded cell was inside any camera's image")
    passed = agreed == checked
    detail = (f"{checked} obstacle-hidden cells checked against the rendered depth, "
              f"{agreed} confirmed blocked ({100.0 * agreed / checked:.1f}%); "
              f"separately, {static_agreed}/{static_checked} "
              f"({100.0 * static_agreed / max(static_checked, 1):.1f}%) of cells the oracle "
              f"calls visible have an unobstructed rendered depth")
    if worst is not None:
        detail += (f"; first disagreement t={worst[0]}s {worst[1]}: "
                   f"rendered {worst[2]:.2f} m vs {worst[3]:.2f} m to the cell")
    return Check("depth agrees with the ray cast",
                 "Gazebo's depth buffer confirms every obstacle-caused occlusion",
                 passed, detail)


def check_boundary() -> Check:
    """No runtime package may import this study or read its ground truth."""
    patterns = ["dynamic_world_oracle", "oracle_visibility", "oracle_depth"]
    hits = []
    for pattern in patterns:
        result = subprocess.run(
            ["grep", "-rln", "--include=*.py", "--include=*.yaml", "--include=*.launch.py",
             pattern, str(REPO / "src")],
            capture_output=True, text=True,
        )
        hits.extend(line for line in result.stdout.splitlines() if line.strip())
    return Check(
        "boundary holds", "nothing under src/ reads this study's ground truth",
        not hits,
        f"searched src/ for {patterns}: no runtime file references them"
        if not hits else f"runtime files referencing oracle data: {sorted(set(hits))}",
    )


def verify(run: Path, repeat: Path | None) -> list[Check]:
    records, manifest, _events = _load(run)
    grids = _grids(run, records)
    checks = [
        check_contract(records),
        check_partial_occlusion(records, grids),
        check_event_timing(records, grids, manifest),
        check_camera_sync(records),
        check_removal_restores(records, grids),
        check_oracle_agrees_with_primitive(manifest),
        check_depth_agrees_with_the_oracle(run, records, manifest),
        check_boundary(),
    ]
    if repeat is not None:
        checks.insert(1, check_reproducible(run, repeat))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--repeat", type=Path, default=None,
                        help="a second run of the same scenario, for the reproducibility check")
    args = parser.parse_args(argv)

    checks = verify(args.run, args.repeat)
    print(f"acceptance check: {args.run}")
    if args.repeat is None:
        print("  (no --repeat given: the reproducibility check is skipped, not passed)")
    print()
    for check in checks:
        print(check.line())
    failed = [c for c in checks if not c.passed]
    print()
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
