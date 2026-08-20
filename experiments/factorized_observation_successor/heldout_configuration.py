#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
import sys

import numpy as np

import common as C
import decision_planner as D


def availability_apparatus():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-factorized-successor")
    directory = C.REPO / "experiments/availability_paper"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    path = directory / "common.py"
    name = "availability_common_for_successor_holdout"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_apparatus()


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if take.any():
            total += float(take.mean()) * abs(float(y[take].mean() - p[take].mean()))
    return total


def near_route(xy: np.ndarray, path: np.ndarray, radius: float = 0.5) -> np.ndarray:
    a, b = path[:-1], path[1:]
    segment = b - a
    length2 = np.maximum(np.sum(segment**2, axis=1), 1e-12)
    near = np.zeros(len(xy), dtype=bool)
    for k, point in enumerate(xy):
        t = np.clip(np.sum((point - a) * segment, axis=1) / length2, 0.0, 1.0)
        projection = a + t[:, None] * segment
        near[k] = float(np.min(np.linalg.norm(point - projection, axis=1))) <= radius
    return near


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    threshold = float(np.quantile(p, 0.2))
    tail = p <= threshold
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece_10bin": ece(y, p), "tail_definition": "lowest predicted-probability quintile",
        "tail_n": int(tail.sum()), "tail_p_mean": float(p[tail].mean()),
        "tail_hit_rate": float(y[tail].mean()), "tail_absolute_calibration_gap": float(abs(p[tail].mean() - y[tail].mean())),
    }


def main() -> None:
    dev_path = C.OUT / "offline_development/gate.json"
    if not dev_path.is_file():
        raise RuntimeError("run offline_gate.py first")
    dev = json.loads(dev_path.read_text())
    if dev["gate"] != "PASS":
        raise RuntimeError("development gate failed; B+C remains locked")

    apparatus = availability_apparatus()
    p = C.load_p()
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    xy_b = np.asarray(apparatus.events["camera_B"]["xy"], float)
    xy_c = np.asarray(apparatus.events["camera_C"]["xy"], float)
    if not np.allclose(xy_b, xy_c):
        raise RuntimeError("B/C commissioning events are not pose-aligned")
    y = np.maximum(apparatus.events["camera_B"]["hit"], apparatus.events["camera_C"]["hit"])
    p_events = 1.0 - (
        1.0 - C.sample(np.asarray(p["P_camera_B_map"], float), xs, ys, xy_b)
    ) * (
        1.0 - C.sample(np.asarray(p["P_camera_C_map"], float), xs, ys, xy_b)
    )
    global_calibration = calibration(np.asarray(y, float), p_events)

    p_cad = dict(p)
    for camera in C.HOLDOUT_CAMERAS:
        p_cad[f"P_{camera}_map"] = np.asarray(apparatus.fields["cad_reference"][camera], float)

    tasks = []
    for task in C.TASKS:
        solved = D.solve(task, C.HOLDOUT_CAMERAS, p)
        shortest, selected = solved["shortest"], solved["selected"]
        ref_scores = {item.route_id: D.score_route(item.route_id, item.path, C.HOLDOUT_CAMERAS, p_cad) for item in solved["eligible"]}
        reference_best = min(ref_scores.values(), key=lambda item: (item.expected_longest_miss_steps, item.length_m, item.route_id))
        shortest_ref = ref_scores[shortest.route_id]
        selected_ref = ref_scores[selected.route_id]
        local = near_route(xy_b, selected.path)
        local_cal = calibration(y[local], p_events[local]) if local.any() else {"n": 0}
        tasks.append({
            "task": task, "shortest": D.serialise(shortest), "ds_route": D.serialise(selected),
            "budget_m": solved["budget_m"], "route_separation_m": C.max_route_separation(shortest.path, selected.path),
            "operational_expected_gap_reduction_fraction": float(
                (shortest.expected_longest_miss_s - selected.expected_longest_miss_s)
                / max(shortest.expected_longest_miss_s, 1e-12)
            ),
            "reference_best_route_id": reference_best.route_id,
            "cad_regret_shortest_s": float(shortest_ref.expected_longest_miss_s - reference_best.expected_longest_miss_s),
            "cad_regret_ds_route_s": float(selected_ref.expected_longest_miss_s - reference_best.expected_longest_miss_s),
            "route_local_calibration": local_cal,
            "budget_respected": bool(selected.length_m <= solved["budget_m"] + 1e-9),
        })

    median_reduction = float(np.median([row["operational_expected_gap_reduction_fraction"] for row in tasks]))
    median_regret_shortest = float(np.median([row["cad_regret_shortest_s"] for row in tasks]))
    median_regret_ds = float(np.median([row["cad_regret_ds_route_s"] for row in tasks]))
    checks = {
        "all_routes_within_5pct_length_budget": all(row["budget_respected"] for row in tasks),
        "median_operational_expected_gap_reduction_positive": median_reduction > 0.0,
        "median_cad_regret_no_worse_than_shortest": median_regret_ds <= median_regret_shortest + 1e-12,
        "at_least_20_route_local_events_each_task": all(row["route_local_calibration"].get("n", 0) >= 20 for row in tasks),
        "tail_calibration_reported": global_calibration["tail_n"] > 0,
    }
    gate = "PASS" if all(checks.values()) else "FAIL"
    rcond = json.loads((C.OUT / "rcond/summary.json").read_text())
    combined = gate == "PASS" and dev["gate"] == "PASS" and rcond["geometry_gate"] == "PASS"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": "held-out camera pairing", "cameras": list(C.HOLDOUT_CAMERAS),
        "scope_caveat": "configuration-level holdout, not new-data holdout",
        "global_tail_calibration": global_calibration, "tasks": tasks,
        "median_operational_expected_gap_reduction_fraction": median_reduction,
        "median_cad_regret_shortest_s": median_regret_shortest,
        "median_cad_regret_ds_route_s": median_regret_ds,
        "checks": checks, "heldout_gate": gate,
        "closed_loop_authorized": combined,
        "closed_loop_blockers": [
            name for name, passed in (
                ("R_cond geometry gate", rcond["geometry_gate"] == "PASS"),
                ("development route gate", dev["gate"] == "PASS"),
                ("held-out B+C gate", gate == "PASS"),
            ) if not passed
        ],
        "evaluation_reference": "CAD raycast; evaluation-only, not deployable truth",
    }
    C.write_json(C.OUT / "heldout_BC/result.json", payload)
    decision = {
        "decision": "RUN_MATCHED_CLOSED_LOOP" if combined else "STOP_FAIL_CLOSED",
        "authorized": combined, "blockers": payload["closed_loop_blockers"],
        "note": "No simulator campaign was launched by this script.",
    }
    C.write_json(C.OUT / "closed_loop/decision.json", decision)
    print(f"B+C held-out gate: {gate}")
    print(f"closed-loop decision: {decision['decision']}")
    if decision["blockers"]:
        print("blockers: " + ", ".join(decision["blockers"]))


if __name__ == "__main__":
    main()
