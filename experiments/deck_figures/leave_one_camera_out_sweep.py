#!/usr/bin/env python3
"""Leave-one-camera-out planning sweep: which removals can move a route, and which way.

The planner picks a route by total expected free energy, which combines an ambiguity term
with risk and obstacle terms through belief propagation along the trajectory. This script
does NOT reproduce that optimisation. It scores the one ingredient the covariance model
controls: how much camera information each candidate route loses when a camera is removed.

A removal can only push the planner off its current route if it costs more information on
that route than on the competing one. The reported quantity is therefore differential,

    pressure(camera) = information lost on the incumbent route
                     - information lost on the competing route,

and positive pressure means the removal pushes the choice toward the competitor. The
incumbent is read from the executed campaign where one exists, so the prediction is anchored
to what the planner actually chose rather than to a proxy for its objective.

This is a scoping tool for choosing which dropout arms to execute. It ranks candidates; it
does not predict success, cost, or closed-loop behaviour. Closed-loop execution remains the
confirmatory test. Reads only frozen artifacts: no simulator, no fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "experiments/thesis_pipeline_lock/stage09_navigation_tasks.yaml"
PLANNING = ROOT / "logs/thesis_final_pipeline_v1/planning_precision"
ANALYSIS = ROOT / "logs/thesis_final_pipeline_v1/stage09_final_analysis_v2"
WORLD = "warehouse_v2.world.sdf"

MODELS = (
    ("m0_planning_precision.npz", "global"),
    ("m1_planning_precision.npz", "per_camera"),
    ("m2_planning_precision.npz", "spatial"),
)


def load(artifact: Path):
    with np.load(artifact, allow_pickle=False) as archive:
        return (np.asarray(archive["xs"]), np.asarray(archive["ys"]),
                archive["camera_ids"].astype(str).tolist(),
                np.asarray(archive["matched_precision_m2_inv"]))


def route_points(start: dict, waypoints: list, step: float = 0.15) -> np.ndarray:
    nodes = [[float(start["x"]), float(start["y"])]] + [[float(a), float(b)]
                                                        for a, b in waypoints]
    out: list[list[float]] = []
    for (x0, y0), (x1, y1) in zip(nodes[:-1], nodes[1:]):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        for s in np.linspace(0.0, 1.0, n, endpoint=False):
            out.append([x0 + s * (x1 - x0), y0 + s * (y1 - y0)])
    out.append(nodes[-1])
    return np.asarray(out)


def sample(xs, ys, field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ix = np.clip(np.searchsorted(xs, pts[:, 0]) - 1, 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, pts[:, 1]) - 1, 0, len(ys) - 1)
    return field[iy, ix]


def executed_incumbent(task: str) -> str | None:
    """Route the executed intact spatial runs actually selected, if the campaign covers it."""
    results = ANALYSIS / "run_results.json"
    if not results.exists():
        return None
    rows = [r for r in json.loads(results.read_text())
            if r["task"] == task and r["condition"] == "spatial_intact"
            and r.get("evidence_valid")]
    if not rows:
        return None
    source = sorted(rows, key=lambda r: int(r["seed"]))[0].get("selected_route_source", "")
    return source.rsplit(":", 1)[-1] or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--planning-dir", type=Path, default=PLANNING)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = yaml.safe_load(args.tasks.read_text(encoding="utf-8"))["tasks"][WORLD]
    tasks = [t for t in spec if len(t.get("route_seeds") or []) == 2]

    report: list[dict] = []
    for artifact, model in ((args.planning_dir / f, label) for f, label in MODELS):
        xs, ys, ids, precision = load(artifact)
        info = 0.5 * np.trace(precision, axis1=-2, axis2=-1)
        for task in tasks:
            names = [s["name"] for s in task["route_seeds"]]
            pts = {s["name"]: route_points(task["start"], s["waypoints"])
                   for s in task["route_seeds"]}
            incumbent = executed_incumbent(task["name"])
            known = incumbent in names
            held = incumbent if known else names[0]
            rival = next(n for n in names if n != held)
            for index, camera in enumerate(ids):
                lost_held = float(sample(xs, ys, info[index], pts[held]).mean())
                lost_rival = float(sample(xs, ys, info[index], pts[rival]).mean())
                report.append({
                    "model": model, "task": task["name"], "camera": camera,
                    "incumbent_route": held, "incumbent_from_campaign": known,
                    "competing_route": rival,
                    "information_lost_on_incumbent": lost_held,
                    "information_lost_on_competitor": lost_rival,
                    "pressure_toward_competitor": lost_held - lost_rival,
                })

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("wrote", args.out)

    for task in sorted({r["task"] for r in report}):
        head = next(r for r in report if r["task"] == task)
        anchor = "executed campaign" if head["incumbent_from_campaign"] else "first seed (no run)"
        print(f"\n== {task}")
        print(f"   incumbent '{head['incumbent_route']}' vs '{head['competing_route']}'"
              f"  [{anchor}]")
        print(f"   {'camera':10}" + "".join(f"{m:>14}" for _, m in MODELS))
        for camera in sorted({r["camera"] for r in report if r["task"] == task}):
            cells = []
            for _, model in MODELS:
                row = next(r for r in report if r["task"] == task
                           and r["model"] == model and r["camera"] == camera)
                cells.append(f"{row['pressure_toward_competitor']:14.1f}")
            print(f"   {camera:10}" + "".join(cells))
    print("\npositive = removing that camera pushes the choice toward the competing route")
    print("R0 and R1 are spatially uniform, so their pressure reflects route geometry only.")


if __name__ == "__main__":
    main()
