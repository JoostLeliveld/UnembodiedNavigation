#!/usr/bin/env python3
"""Check that the v8 refit is no worse than v5 on the SAME D_dev positions.

Only positions present in the v5 D_dev role enter, so a change in the D_dev population
cannot pass for an improvement. Two estimators are reported for every metric, as fixed in
the 2026-09-23 amendment before any v8 result was seen:

* position-balanced: every physical position weighs the same;
* density-weighted: each 1 m cell carries at most 8 positions' worth of weight
  (K / (pi (2 l_R)^2) with K=16, l_R=0.4 m), shared equally by its positions.

    python3 pipeline/decisions/compare_v5_v8_ddev.py
"""
from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
L = REPO / "logs"
RUNS = {"v5": L / "track_a_draft/recapture_v5_fits", "v8": L / "thesis/fits"}
CAP = 16 / (math.pi * 0.8 ** 2)


def per_position(root: Path) -> dict[str, dict]:
    with (root / "ddev_evaluation/ddev_per_position.csv").open(newline="") as handle:
        return {r["position_key"]: r for r in csv.DictReader(handle)}


def main() -> int:
    all_v5 = list(csv.DictReader((L / "thesis/captures/v5/capture_positions_v5.csv").open()))
    v5_positions = {r["position_id"]: (float(r["x"]), float(r["y"])) for r in all_v5
                    if r["role"] == "D_dev"}
    tables = {name: per_position(root) for name, root in RUNS.items()}
    common = sorted(set(v5_positions) & set(tables["v5"]) & set(tables["v8"]))
    # Density counts every v5 position in the cell, whatever its role.
    cells = collections.Counter((math.floor(float(r["x"])), math.floor(float(r["y"])))
                                for r in all_v5)
    density_weight = {k: min(1.0, CAP / cells[(math.floor(v5_positions[k][0]),
                                               math.floor(v5_positions[k][1]))]) for k in common}
    metrics = [c for c in tables["v5"][common[0]] if c.endswith(("_rmse_m", "_mean_nll", "_mean_nis"))
               or c == "correction_mean_error_m"]
    report = {"positions": len(common), "metrics": {}}
    for metric in metrics:
        row = {}
        for name, table in tables.items():
            values = np.array([float(table[k][metric]) for k in common])
            weights = np.array([density_weight[k] for k in common])
            row[name] = {"position_balanced": float(values.mean()),
                         "density_weighted": float(np.average(values, weights=weights))}
        report["metrics"][metric] = row
    for name, root in RUNS.items():
        manifest = json.loads((root / "ddev_evaluation/manifest.json").read_text())
        cam = manifest["D_dev_correction_metrics"]["structured_plus_visibility"]["by_camera"]["camera_C"]
        report.setdefault("camera_C_pooled_all_D_dev", {})[name] = {
            "rmse_m": cam["rmse_m"], "median_m": cam["median_m"], "n": cam["n"]}
    out = L / "thesis/evidence/ddev_comparison_v5_v8.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'metric':38}{'v5 pos':>9}{'v8 pos':>9}{'v5 dens':>9}{'v8 dens':>9}")
    for metric, row in report["metrics"].items():
        print(f"{metric:38}{row['v5']['position_balanced']:9.4f}{row['v8']['position_balanced']:9.4f}"
              f"{row['v5']['density_weighted']:9.4f}{row['v8']['density_weighted']:9.4f}")
    print("camera C pooled (all D_dev):", report["camera_C_pooled_all_D_dev"])
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
