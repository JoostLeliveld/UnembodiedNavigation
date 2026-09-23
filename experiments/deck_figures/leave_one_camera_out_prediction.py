#!/usr/bin/env python3
"""Predict, from frozen planning fields alone, which camera removals can change a route.

For each task this compares how much information every camera contributes along the route
the planner chose against the route it did not choose. A removal can only flip the decision
if it costs more on the chosen route than on its competitor, so the difference ranks the
camera removals by how much each one destabilises the current choice.

This reads only the frozen matched-covariance planning artifact and the executed plans of
the final campaign. It runs no simulation and fits nothing, so it is a scoping tool for
choosing which dropout arms are worth executing, not a result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "logs/thesis_final_pipeline_v1/stage09_final_analysis_v2"
PLANNING = ROOT / "logs/thesis_final_pipeline_v1/planning_precision/m2_planning_precision.npz"


def load_field(artifact: Path):
    with np.load(artifact, allow_pickle=False) as archive:
        xs = np.asarray(archive["xs"])
        ys = np.asarray(archive["ys"])
        ids = archive["camera_ids"].astype(str).tolist()
        precision = np.asarray(archive["matched_precision_m2_inv"])
    return xs, ys, ids, 0.5 * np.trace(precision, axis1=-2, axis2=-1)


def read_plan(run_dir: Path) -> np.ndarray | None:
    path = run_dir / "global_plan.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(r["x"]), float(r["y"])] for r in rows])


def corridor_mask(grid_x, grid_y, plan: np.ndarray, halfwidth: float) -> np.ndarray:
    mask = np.zeros(grid_x.shape, dtype=bool)
    for px, py in plan[::3]:
        mask |= ((grid_x - px) ** 2 + (grid_y - py) ** 2) <= halfwidth**2
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--planning", type=Path, default=PLANNING)
    parser.add_argument("--halfwidth", type=float, default=0.6,
                        help="corridor half-width in metres around the route")
    args = parser.parse_args()

    rows = json.loads((args.analysis / "run_results.json").read_text())
    xs, ys, ids, info = load_field(args.planning)
    grid_x, grid_y = np.meshgrid(xs, ys)

    def lowest_seed(task: str, condition: str):
        cand = [r for r in rows
                if r["task"] == task and r["condition"] == condition and r.get("evidence_valid")]
        return sorted(cand, key=lambda r: int(r["seed"]))[0] if cand else None

    report: dict[str, dict] = {}
    for task in sorted({r["task"] for r in rows}):
        intact = lowest_seed(task, "spatial_intact")
        removal = lowest_seed(task, "spatial_removal")
        if intact is None or removal is None:
            continue
        chosen_plan = read_plan(ROOT / intact["run"])
        other_plan = read_plan(ROOT / removal["run"])
        if chosen_plan is None or other_plan is None:
            continue

        chosen = corridor_mask(grid_x, grid_y, chosen_plan, args.halfwidth)
        competing = corridor_mask(grid_x, grid_y, other_plan, args.halfwidth)

        entry = {}
        for i, cam in enumerate(ids):
            on_chosen = float(info[i][chosen].mean())
            on_competing = float(info[i][competing].mean())
            entry[cam] = {
                "information_on_chosen_route": on_chosen,
                "information_on_competing_route": on_competing,
                "advantage_lost_if_removed": on_chosen - on_competing,
            }
        report[task] = dict(
            sorted(entry.items(), key=lambda kv: kv[1]["advantage_lost_if_removed"], reverse=True))

    print(json.dumps(report, indent=2))
    for task, entry in report.items():
        print(f"\n{task}: removal ranked by destabilising power")
        for rank, (cam, stats) in enumerate(entry.items(), 1):
            print(f"  {rank}. {cam:10} {stats['advantage_lost_if_removed']:9.1f}")


if __name__ == "__main__":
    main()
