#!/usr/bin/env python3
"""The real-drive figure, with the shape model no longer handed the answer.

Figure 19 (`08_on_a_real_drive`) puts three columns side by side and the hull column wins.
Two things make that column incomparable to the others:

1.  It is a DIFFERENT DRIVE.  Raw and fixed collided at ~81 s; the hull column is a run that
    reached the goal at 148 s.  A column difference there mixes the interpretation with the
    drive it happened to be recorded on.
2.  It is an ORACLE.  `hull_estimates` linearises the silhouette around the TRUE pose and the
    TRUE yaw at each capture stamp, so the answer is one of its inputs.

This script fixes both at once.  Every column replays the SAME deduplicated readings from the
SAME recorded drive -- the layout already used by figure 23 -- and the shape model appears in
two forms so the reader can see what the oracle was worth:

    hull        started AT the true position and true heading   (kept, greyed, as a bound)
    hull_fair   started at the raw back-projection, heading solved from the box, no truth

`hull_fair` starts where the learned corrections start and sees what they see.  On the frozen
characterization tiles that change costs the shape model its lead: 3.1 cm against 3.4 cm for
the neural correction, which then wins the tail.  This figure asks the same question on a
moving robot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for rel in ("experiments/camera_observation_characterization",
            "experiments/fusion_on_fixed_routes",
            "experiments/deck_figures"):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import replay_learned_on_actual_run as L  # noqa: E402
from fair_hull_comparison import solve_pose  # noqa: E402

FOLDER = "14_fair_hull_on_a_real_drive"
FIGURE = "24_fair_hull_over_time.png"

COLUMNS = (
    ("raw", "Raw box → floor", "the box and its camera", True),
    ("linear", "Learned linear correction", "the box and its camera", True),
    ("neural", "Learned neural correction", "the box and its camera", True),
    ("hull_fair", "Robot-shape model, heading solved", "the box and its camera", True),
    ("hull", "Robot-shape model started AT the answer", "the TRUE position and heading", False),
)


def fair_hull_estimates(readings, rows, optics) -> tuple[np.ndarray, np.ndarray]:
    """The shape model with no truth: solve position and heading from the box alone.

    Returns the estimates and the solved heading per reading.  Heading is searched over a half
    circle because the hull is exactly symmetric under a 180 deg flip -- one box cannot say
    which way the robot faces, and for POSITION that ambiguity is free.
    """
    estimates = np.full((len(rows), 2), np.nan, dtype=float)
    headings = np.full(len(rows), np.nan, dtype=float)
    for index, row in enumerate(rows):
        cam = optics[row["camera_id"]]
        start = np.asarray([float(row["raw_x"]), float(row["raw_y"])], dtype=float)
        measured = np.asarray([float(row["u_bbox_bottom"]), float(row["v_bbox_bottom"])])
        measured_box = np.asarray([float(row["x0"]), float(row["y0"]),
                                   float(row["x1"]), float(row["y1"])])
        solved = solve_pose(cam, start, measured, measured_box)
        if solved is None:
            continue
        estimates[index] = solved[0]
        headings[index] = solved[1]
    return estimates, headings


def build_columns(run, run_manifest, run_summary, readings, elapsed, evaluated):
    """One column per interpretation, all on the same drive and the same readings."""
    import aligned as A  # noqa: E402
    import plot_real_run_bias as R  # noqa: E402

    first_cmd = float(run_summary["first_cmd_stamp"])
    stop_stamp = float(run_summary["stop_stamp"])
    table = A.rows(run)
    truth = A.truth_series(run, table)
    in_drive = (truth.t >= first_cmd) & (truth.t <= stop_stamp)
    route = np.asarray(json.loads(run_manifest["preselected_route_json"]), dtype=float)
    run_id = str(run_manifest.get("run_id", run.name))
    collision_s = (
        float(run_summary["first_crash_stamp"]) - first_cmd
        if run_summary.get("collision_any")
        and run_summary.get("first_crash_stamp") is not None
        else None
    )

    columns = []
    for method, label, given, fair in COLUMNS:
        scores = evaluated[method]
        replayed = []
        for index, source in enumerate(readings):
            if not np.isfinite(scores["magnitude"][index]):
                continue
            item = dict(source)
            item["error"] = scores["error"][index]
            item["error_cm"] = float(scores["magnitude"][index] * 100.0)
            item["magnitude_m"] = float(scores["magnitude"][index])
            item["along_m"] = float(scores["along"][index])
            item["across_m"] = float(scores["across"][index])
            replayed.append(item)
        kept = np.asarray([np.isfinite(scores["magnitude"][i])
                           for i in range(len(readings))])
        times = elapsed[kept]
        errors_cm = np.asarray([item["error_cm"] for item in replayed], dtype=float)
        note = "same readings, same drive" if fair else "⚠ GIVEN THE ANSWER — a ceiling, not a method"
        columns.append({
            "run": run,
            "manifest": run_manifest,
            "summary": run_summary,
            "readings": replayed,
            "times": times,
            "errors_cm": errors_cm,
            "duration_s": float(run_summary["elapsed_after_first_cmd_s"]),
            "collision_s": collision_s,
            "spans": R.blind_spans(times),
            "route": route,
            "truth_xy": (truth.x[in_drive], truth.y[in_drive]),
            "observation_model": label,
            "run_id": run_id,
            "panel_id": method,
            "arm": "offline_replay",
            "oracle": not fair,
            "context_line": f"given: {given}\n{note}",
            "completion": str(run_summary.get("completion_reason", "unknown")),
            "per_camera": {
                camera: sum(item["camera"] == camera for item in replayed)
                for camera in "ABCDE"
            },
        })
    return columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=L.DEFAULT_CAPTURE)
    parser.add_argument("--run", type=Path, default=L.DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=L.DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import aligned as A  # noqa: E402
    import plot_real_run_bias as R  # noqa: E402

    capture = args.capture.expanduser().resolve()
    run = args.run.expanduser().resolve()
    output = args.out.expanduser().resolve() / FOLDER
    output.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and (output / FIGURE).exists():
        raise RuntimeError("Output exists; pass --overwrite")

    capture_manifest = json.loads((capture / "capture_manifest.json").read_text())
    run_manifest = json.loads((run / "run_manifest.json").read_text())
    run_summary = json.loads((run / "run_summary.json").read_text())

    # Same frozen models, same training tiles, same refit path as figure 23.
    geometry = L.camera_geometry(capture_manifest)
    optics = L.camera_models(capture_manifest)
    project = L.projectors(capture_manifest)
    import csv as _csv
    rows_all = list(_csv.DictReader(
        (capture / "bias_update_interpretations.csv").open(encoding="utf-8")))
    train = [row for row in rows_all
             if row["split"] == "train" and row["raw_valid"] == "1"]
    models = L.fit_models(train, geometry, args.seed)

    loaded = A.readings(run, admitted_only=False, dedupe=True, require_capture_time=True)
    first_cmd = float(run_summary["first_cmd_stamp"])
    stop_stamp = float(run_summary["stop_stamp"])
    readings = [item for item in loaded if first_cmd <= item["obs_stamp"] <= stop_stamp]
    if not readings:
        raise RuntimeError("No deduplicated readings during actual motion")
    rows = L.replay_rows(readings, project)
    truths = np.stack([reading["truth"] for reading in readings])
    elapsed = np.asarray([reading["obs_stamp"] - first_cmd for reading in readings])

    corrections = L.corrections_for_rows(rows, models, geometry)
    oracle_hull = L.hull_estimates(readings, rows, optics)
    fair_hull, solved_yaw = fair_hull_estimates(readings, rows, optics)

    methods = tuple(method for method, *_ in COLUMNS)
    saved_methods = L.METHODS
    L.METHODS = methods
    try:
        evaluated = L.evaluate(rows, truths, corrections, geometry,
                               direct={"hull": oracle_hull, "hull_fair": fair_hull})
    finally:
        L.METHODS = saved_methods

    scores = {method: L.summary(evaluated[method]["magnitude"]) for method in methods}
    signed = {method: L.signed_summary(evaluated[method]["along"]) for method in methods}

    truth_yaw = np.asarray([float(reading["truth_yaw"]) for reading in readings])
    folded = (solved_yaw - truth_yaw) % math.pi
    heading_error = np.degrees(np.minimum(folded, math.pi - folded))
    heading_error = heading_error[np.isfinite(heading_error)]

    columns = build_columns(run, run_manifest, run_summary, readings, elapsed, evaluated)
    fair_best = min(("raw", "linear", "neural", "hull_fair"),
                    key=lambda key: scores[key]["median_cm"])
    R.draw_sheet(
        columns,
        output,
        filename=FIGURE,
        suptitle=(
            "THE SAME RECORDED GAZEBO DRIVE, EVERY INTERPRETATION ON THE SAME READINGS\n"
            f"All {len(readings)} deduplicated raw-box readings replayed five ways — "
            "only the last column is given the true pose\n"
            "Each row shares one scale; every reading is scored against truth at its own "
            "capture timestamp"
        ),
        footnote=(
            "Camera-reading layer only — not fused, belief or planner error. Every column is "
            "an offline replay of ONE drive: none of them steered the robot or changed the "
            "trajectory, so the columns differ only by interpretation.\n"
            "The last column is started at the TRUE position and TRUE heading and is a "
            "ceiling, not an achievable method. The fourth column is the same shape model "
            "solving its own heading from the box, which is what the robot could actually run."
        ),
    )

    manifest = {
        "status": "complete",
        "schema": "fair_hull_actual_drive_replay.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Figure 19 compared the hull on a DIFFERENT drive and with the true pose as an "
            "input. This replays every interpretation on one drive and one reading set, and "
            "splits the shape model into an oracle rung and an operational rung."
        ),
        "run": str(run),
        "run_id": str(run_manifest.get("run_id", run.name)),
        "n_readings": len(readings),
        "methods": {method: {"label": label, "given": given, "fair": fair}
                    for method, label, given, fair in COLUMNS},
        "scores_cm": scores,
        "signed_along_cm": signed,
        "solved_heading_error_deg": {
            "note": "absolute error modulo 180 deg; a flip is not identifiable from one box",
            "n": int(heading_error.size),
            "median": float(np.median(heading_error)) if heading_error.size else None,
            "p90": float(np.quantile(heading_error, 0.9)) if heading_error.size else None,
        },
        "coverage": {method: int(np.sum(evaluated[method]["valid"])) for method in methods},
        "boundaries": [
            "Offline replay of one recorded drive; not closed loop and not replicated.",
            "The hull column is an evaluation-only oracle and never a deployment input.",
            "The runtime admission gate was enabled in the source drive, so detector misses "
            "and boxes rejected before fusion_observations.csv cannot be recovered here.",
        ],
    }
    (output / "fair_hull_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(output / FIGURE),
                      "n_readings": len(readings),
                      "best_fair": fair_best,
                      "scores_cm": scores,
                      "heading": manifest["solved_heading_error_deg"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
