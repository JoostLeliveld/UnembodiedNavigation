#!/usr/bin/env python3
"""Persist the preregistered E1 falsifier tests from fold-clean unit scores.

The primary score in the preregistration is raw Brier, not Brier skill.  This script
therefore computes the two explicit degradation interactions implied by falsifiers
F1 and F3 and adjusts that two-test family with Holm.  Skill remains available in
``e1_degradation.csv`` as a descriptive sensitivity analysis, but it is not used
here to replace a null raw-Brier result.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

BOOTSTRAP = 10_000
SEED = 20260819


def exact_sign_test(values: np.ndarray) -> tuple[int, int, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    nz = v[v != 0.0]
    n = int(len(nz))
    positive = int(np.sum(nz > 0))
    if n == 0:
        return 0, 0, 1.0
    k = min(positive, n - positive)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return positive, n, float(min(1.0, 2.0 * tail))


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(v), size=(BOOTSTRAP, len(v)))
    means = v[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm(p_values: list[float]) -> list[float]:
    """Holm adjusted p-values in original order, with monotonicity enforced."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(units_path: Path, environment: str) -> list[dict[str, object]]:
    with units_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_key = {
        (r["arm"], r["environment"], r["camera"], int(r["fold"])): float(r["brier"])
        for r in rows
    }
    unit_keys = sorted({(r["camera"], int(r["fold"])) for r in rows
                        if r["environment"] == "L0"})

    comparisons = (
        ("F1_gp_minus_mono_degradation", "gp", "mono_depth",
         "positive means the historical GP degraded more than recomputed monocular depth"),
        ("F3_hybrid_minus_mono_degradation", "hybrid", "mono_depth",
         "positive means the frozen-residual hybrid degraded more than monocular depth"),
    )
    out: list[dict[str, object]] = []
    for name, stale_arm, adaptive_arm, meaning in comparisons:
        values = []
        for camera, fold in unit_keys:
            delta_stale = (by_key[(stale_arm, environment, camera, fold)] -
                           by_key[(stale_arm, "L0", camera, fold)])
            delta_adaptive = (by_key[(adaptive_arm, environment, camera, fold)] -
                              by_key[(adaptive_arm, "L0", camera, fold)])
            values.append(delta_stale - delta_adaptive)
        values_arr = np.asarray(values, dtype=float)
        lo, hi = bootstrap_ci(values_arr)
        positive, non_ties, p_value = exact_sign_test(values_arr)
        out.append({
            "environment": environment,
            "comparison": name,
            "estimand": "(Brier_arm_env-Brier_arm_L0) difference",
            "meaning": meaning,
            "n_units": len(values_arr),
            "mean_interaction": float(np.mean(values_arr)),
            "ci95_low": lo,
            "ci95_high": hi,
            "positive_units": positive,
            "non_tied_units": non_ties,
            "sign_test_p_raw": p_value,
        })
    adjusted = holm([float(row["sign_test_p_raw"]) for row in out])
    for row, p_adj in zip(out, adjusted):
        row["sign_test_p_holm_2"] = p_adj
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", required=True)
    ap.add_argument("--environment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)

    units_path = Path(args.units)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    rows = summarize(units_path, args.environment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": args.environment,
        "primary_metric": "raw Brier score",
        "family": "preregistered F1 and F3 degradation interactions",
        "multiplicity": "Holm across 2 interactions",
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "unit": "camera x spatial block"},
        "sign_test": "exact two-sided; exact-zero ties dropped",
        "inputs": {str(units_path): file_sha256(units_path)},
        "script_sha256": file_sha256(Path(__file__)),
        "output_sha256": file_sha256(out_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['comparison']}: {row['mean_interaction']:+.6f} "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}], "
            f"p={row['sign_test_p_raw']:.6g}, Holm={row['sign_test_p_holm_2']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
