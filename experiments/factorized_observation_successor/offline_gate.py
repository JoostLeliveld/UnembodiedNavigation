#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import os
import sys

import numpy as np

import common as C
import decision_planner as D


def driveable_mask() -> np.ndarray:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-factorized-successor")
    path = C.REPO / "experiments/availability_paper/common.py"
    spec = importlib.util.spec_from_file_location("availability_common_for_successor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return np.asarray(module.build_apparatus().driveable, bool)


def main() -> None:
    r_summary_path = C.OUT / "rcond/summary.json"
    if not r_summary_path.is_file():
        raise RuntimeError("run commission_rcond.py first")
    r_summary = __import__("json").loads(r_summary_path.read_text())
    p = C.load_p()
    fused = C.fused_p(p, C.DEV_CAMERAS)
    mask = driveable_mask()
    dynamic_range = float(np.ptp(fused[mask]))
    rows = []
    for task in C.TASKS:
        solved = D.solve(task, C.DEV_CAMERAS, p)
        shortest, selected = solved["shortest"], solved["selected"]
        reduction = (shortest.expected_longest_miss_s - selected.expected_longest_miss_s) / max(shortest.expected_longest_miss_s, 1e-12)
        rows.append({
            "task": task, "shortest": D.serialise(shortest), "ds_route": D.serialise(selected),
            "budget_m": solved["budget_m"], "route_separation_m": C.max_route_separation(shortest.path, selected.path),
            "expected_longest_miss_reduction_fraction": float(reduction),
            "budget_respected": bool(selected.length_m <= solved["budget_m"] + 1e-9),
            "candidate_count": len(solved["all"]), "eligible_candidate_count": len(solved["eligible"]),
        })
    changed = sum(row["route_separation_m"] >= 0.25 for row in rows)
    median_reduction = float(np.median([row["expected_longest_miss_reduction_fraction"] for row in rows]))
    checks = {
        "rcond_commissioning_complete": r_summary["selected_model"] in ("constant", "geometry"),
        "routes_changed_on_at_least_two_tasks": changed >= 2,
        "median_expected_gap_reduction_at_least_15pct": median_reduction >= 0.15,
        "all_routes_within_5pct_length_budget": all(row["budget_respected"] for row in rows),
        "p_use_dynamic_range_at_least_0p10": dynamic_range >= 0.10,
    }
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": "development", "cameras": list(C.DEV_CAMERAS),
        "planner": "DS-Route", "rcond_selected_model": r_summary["selected_model"],
        "rcond_spatial_claim_allowed": r_summary["geometry_gate"] == "PASS",
        "fused_p_use_driveable_dynamic_range": dynamic_range,
        "changed_task_count": changed, "median_expected_gap_reduction_fraction": median_reduction,
        "checks": checks, "gate": "PASS" if all(checks.values()) else "FAIL", "tasks": rows,
    }
    C.write_json(C.OUT / "offline_development/gate.json", payload)
    print(f"development route-sensitivity gate: {payload['gate']}")
    for row in rows:
        print(f"  {row['task']}: separation={row['route_separation_m']:.2f} m, gap reduction={100*row['expected_longest_miss_reduction_fraction']:.1f}%")


if __name__ == "__main__":
    main()
