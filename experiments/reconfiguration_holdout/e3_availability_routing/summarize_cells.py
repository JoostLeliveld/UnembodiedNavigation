#!/usr/bin/env python3
"""Summarise the 25 routing cells with paired inference and multiplicity control.

The experimental unit is one start--goal task.  For every camera-subset x detour-
budget cell, the primary contrast is the difference in differences (DiD)

    [(changed frozen GP) - (L0 frozen GP)]
      - [(changed recomputed mono-depth) - (L0 recomputed mono-depth)].

A positive value means that the frozen field degraded more across the warehouse
change.  The per-cell p-value is an exact two-sided sign test over paired tasks,
with exact-zero ties dropped.  Holm's step-down procedure controls family-wise
error over all 25 subset x budget cells.  The direct changed-environment
GP-minus-mono-depth gap is also emitted, with its own explicitly separate 25-cell
Holm family, so sparse-camera reversals cannot be hidden by reporting only
degradation.  ``--changed-environment`` selects the comparison and keyed output
filenames prevent a later replication from overwriting the L1 analysis.

This script deliberately reads the persisted route table instead of recomputing
routes.  It can therefore regenerate every inferential number without launching
Gazebo or refitting an availability field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_RESULTS = REPO / "logs/studies/reconfiguration_holdout/e3_availability_routing"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260819
ALPHA = 0.05
TIE_TOLERANCE_M = 1e-12
EXPECTED_CELLS = 25

SUMMARY_COLUMNS = (
    "changed_environment",
    "subset",
    "budget",
    "n_tasks",
    "mean_blind_l0_gp_m",
    "mean_blind_l0_mono_depth_m",
    "mean_blind_changed_gp_m",
    "mean_blind_changed_mono_depth_m",
    "mean_gp_degradation_m",
    "mean_mono_depth_degradation_m",
    "mean_did_m",
    "did_ci95_low_m",
    "did_ci95_high_m",
    "did_positive",
    "did_negative",
    "did_ties",
    "did_sign_p_raw",
    "did_sign_p_holm_25",
    "did_holm_reject_0_05",
    "mean_changed_gp_minus_mono_depth_m",
    "changed_gap_ci95_low_m",
    "changed_gap_ci95_high_m",
    "changed_gap_gp_higher",
    "changed_gap_gp_lower",
    "changed_gap_ties",
    "changed_gap_sign_p_raw",
    "changed_gap_sign_p_holm_25",
    "changed_gap_holm_reject_0_05",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_two_sided_sign_test(
    differences: Iterable[float], *, tie_tolerance: float = TIE_TOLERANCE_M
) -> tuple[int, int, int, float]:
    """Return positive, negative, ties, and the exact two-sided binomial p-value."""

    values = np.asarray(list(differences), dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("sign-test differences must be a finite one-dimensional sequence")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be non-negative")

    positive = int(np.sum(values > tie_tolerance))
    negative = int(np.sum(values < -tie_tolerance))
    ties = int(values.size - positive - negative)
    n = positive + negative
    if n == 0:
        return positive, negative, ties, 1.0
    tail_k = min(positive, negative)
    lower_tail = sum(comb(n, k) for k in range(tail_k + 1)) / (2.0**n)
    return positive, negative, ties, float(min(1.0, 2.0 * lower_tail))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-adjust p-values while preserving the caller's original order."""

    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("Holm p-values must be a non-empty finite one-dimensional sequence")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Holm p-values must lie in [0, 1]")

    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(p.size, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (p.size - rank) * float(p[index])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(1.0, running_max)

    adjusted = np.empty(p.size, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def bootstrap_mean_ci(
    values: Sequence[float], *, rng: np.random.Generator, resamples: int
) -> tuple[float, float]:
    """Deterministic percentile interval for the paired-task mean."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be a non-empty finite one-dimensional sequence")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    draw_index = rng.integers(0, data.size, size=(resamples, data.size))
    means = data[draw_index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _canonical_budget(value: str | float, budgets: Sequence[float]) -> float:
    observed = float(value)
    matches = [budget for budget in budgets if abs(observed - budget) <= 1e-12]
    if len(matches) != 1:
        raise ValueError(f"route row has undeclared budget {observed!r}")
    return float(matches[0])


def _load_design(route_manifest_path: Path) \
        -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...], tuple[str, ...]]:
    with route_manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    subsets = tuple(str(value) for value in manifest["subsets"])
    budgets = tuple(float(value) for value in manifest["budgets"])
    tasks = tuple(str(value) for value in manifest["tasks"])
    if len(set(subsets)) != len(subsets) or len(set(budgets)) != len(budgets):
        raise ValueError("route manifest has duplicate subsets or budgets")
    if len(set(tasks)) != len(tasks) or not tasks:
        raise ValueError("route manifest must declare a non-empty unique task list")
    if len(subsets) * len(budgets) != EXPECTED_CELLS:
        raise ValueError(
            f"the registered routing family must contain {EXPECTED_CELLS} cells; "
            f"manifest declares {len(subsets)} x {len(budgets)}"
        )
    arms = set(str(value) for value in manifest.get("arms", ()))
    if not {"gp", "mono_depth"}.issubset(arms):
        raise ValueError("route manifest must declare both gp and mono_depth arms")
    # Schema-v1 route manifests predated multiple changed environments and implied
    # L0/L1.  Keep them readable while every newly generated manifest is explicit.
    environments = tuple(str(value) for value in manifest.get("environments", ("L0", "L1")))
    if not environments or environments[0] != "L0" or len(set(environments)) != len(environments):
        raise ValueError("route manifest environments must be unique and begin with L0")
    return subsets, budgets, tasks, environments


def _load_paired_values(
    routes_path: Path,
    *,
    subsets: Sequence[str],
    budgets: Sequence[float],
    tasks: Sequence[str],
    declared_environments: Sequence[str],
    changed_environment: str,
) -> dict[tuple[str, float, str, str, str], float]:
    expected_subsets = set(subsets)
    expected_tasks = set(tasks)
    values: dict[tuple[str, float, str, str, str], float] = {}
    required_columns = {"environment", "subset", "task", "budget", "arm", "blind_true_m"}

    with routes_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"route table is missing columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            if row["arm"] not in {"gp", "mono_depth"}:
                continue
            environment = row["environment"]
            subset = row["subset"]
            task = row["task"]
            if environment not in set(declared_environments):
                raise ValueError(f"row {row_number}: undeclared environment {environment!r}")
            if environment not in {"L0", changed_environment}:
                continue
            if subset not in expected_subsets:
                raise ValueError(f"row {row_number}: undeclared subset {subset!r}")
            if task not in expected_tasks:
                raise ValueError(f"row {row_number}: undeclared task {task!r}")
            budget = _canonical_budget(row["budget"], budgets)
            blind = float(row["blind_true_m"])
            if not np.isfinite(blind):
                raise ValueError(f"row {row_number}: blind_true_m is not finite")
            key = (subset, budget, task, environment, row["arm"])
            if key in values:
                raise ValueError(f"row {row_number}: duplicate paired observation {key}")
            values[key] = blind

    missing = []
    for subset in subsets:
        for budget in budgets:
            for task in tasks:
                for environment in ("L0", changed_environment):
                    for arm in ("gp", "mono_depth"):
                        key = (subset, float(budget), task, environment, arm)
                        if key not in values:
                            missing.append(key)
    if missing:
        preview = ", ".join(map(str, missing[:3]))
        raise ValueError(f"route table is missing {len(missing)} paired observations; first: {preview}")
    return values


def summarize_routes(
    routes_path: Path,
    route_manifest_path: Path,
    *,
    changed_environment: str = "L1",
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    """Build one typed summary row per registered subset x budget cell."""

    subsets, budgets, tasks, declared_environments = _load_design(route_manifest_path)
    if changed_environment == "L0" or changed_environment not in declared_environments:
        raise ValueError(
            f"changed environment {changed_environment!r} must be declared and differ from L0"
        )
    values = _load_paired_values(
        routes_path,
        subsets=subsets,
        budgets=budgets,
        tasks=tasks,
        declared_environments=declared_environments,
        changed_environment=changed_environment,
    )
    rng = np.random.default_rng(bootstrap_seed)
    summaries: list[dict[str, object]] = []

    for subset in subsets:
        for budget in budgets:
            g0 = np.asarray(
                [values[(subset, budget, task, "L0", "gp")] for task in tasks], dtype=float
            )
            m0 = np.asarray(
                [values[(subset, budget, task, "L0", "mono_depth")] for task in tasks],
                dtype=float,
            )
            gc = np.asarray(
                [values[(subset, budget, task, changed_environment, "gp")]
                 for task in tasks], dtype=float
            )
            mc = np.asarray(
                [values[(subset, budget, task, changed_environment, "mono_depth")]
                 for task in tasks],
                dtype=float,
            )
            gp_degradation = gc - g0
            mono_degradation = mc - m0
            did = gp_degradation - mono_degradation
            changed_gap = gc - mc
            did_lo, did_hi = bootstrap_mean_ci(
                did, rng=rng, resamples=bootstrap_resamples
            )
            gap_lo, gap_hi = bootstrap_mean_ci(
                changed_gap, rng=rng, resamples=bootstrap_resamples
            )
            did_pos, did_neg, did_ties, did_p = exact_two_sided_sign_test(did)
            gap_pos, gap_neg, gap_ties, gap_p = exact_two_sided_sign_test(changed_gap)
            summaries.append(
                {
                    "changed_environment": changed_environment,
                    "subset": subset,
                    "budget": float(budget),
                    "n_tasks": len(tasks),
                    "mean_blind_l0_gp_m": float(g0.mean()),
                    "mean_blind_l0_mono_depth_m": float(m0.mean()),
                    "mean_blind_changed_gp_m": float(gc.mean()),
                    "mean_blind_changed_mono_depth_m": float(mc.mean()),
                    "mean_gp_degradation_m": float(gp_degradation.mean()),
                    "mean_mono_depth_degradation_m": float(mono_degradation.mean()),
                    "mean_did_m": float(did.mean()),
                    "did_ci95_low_m": did_lo,
                    "did_ci95_high_m": did_hi,
                    "did_positive": did_pos,
                    "did_negative": did_neg,
                    "did_ties": did_ties,
                    "did_sign_p_raw": did_p,
                    "mean_changed_gp_minus_mono_depth_m": float(changed_gap.mean()),
                    "changed_gap_ci95_low_m": gap_lo,
                    "changed_gap_ci95_high_m": gap_hi,
                    "changed_gap_gp_higher": gap_pos,
                    "changed_gap_gp_lower": gap_neg,
                    "changed_gap_ties": gap_ties,
                    "changed_gap_sign_p_raw": gap_p,
                }
            )

    did_adjusted = holm_adjust([float(row["did_sign_p_raw"]) for row in summaries])
    gap_adjusted = holm_adjust([
        float(row["changed_gap_sign_p_raw"]) for row in summaries
    ])
    for row, did_p, gap_p in zip(summaries, did_adjusted, gap_adjusted):
        row["did_sign_p_holm_25"] = did_p
        row["did_holm_reject_0_05"] = did_p <= ALPHA
        row["changed_gap_sign_p_holm_25"] = gap_p
        row["changed_gap_holm_reject_0_05"] = gap_p <= ALPHA
    return summaries


def _csv_value(column: str, value: object) -> object:
    if column == "budget":
        return f"{float(value):.2f}"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_summary(
    routes_path: Path,
    route_manifest_path: Path,
    out_path: Path,
    out_manifest_path: Path,
    *,
    changed_environment: str = "L1",
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    summaries = summarize_routes(
        routes_path,
        route_manifest_path,
        changed_environment=changed_environment,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({column: _csv_value(column, row[column]) for column in SUMMARY_COLUMNS})

    metadata = {
        "schema_version": 1,
        "input_routes": str(routes_path),
        "input_routes_sha256": sha256(routes_path),
        "input_route_manifest": str(route_manifest_path),
        "input_route_manifest_sha256": sha256(route_manifest_path),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "experimental_unit": "start_goal_task",
        "baseline_environment": "L0",
        "changed_environment": changed_environment,
        "inferential_scope": (
            f"conditional on the single fixed L0-to-{changed_environment} layout; "
            "task-level tests do not "
            "establish generalization across independent warehouse reconfigurations"
        ),
        "n_tasks_per_cell": sorted({int(row["n_tasks"]) for row in summaries}),
        "cells": "camera_subset x detour_budget",
        "n_cells": len(summaries),
        "primary_contrast": (
            f"({changed_environment}_gp-L0_gp)-"
            f"({changed_environment}_mono_depth-L0_mono_depth)"
        ),
        "primary_positive_direction": "frozen_gp_degrades_more_than_recomputed_mono_depth",
        "primary_test": "exact_two_sided_sign_test; ties_abs_le_1e-12_m_dropped",
        "primary_multiplicity": "Holm family-wise correction across all 25 cells",
        "secondary_contrast": (
            f"{changed_environment}_gp-{changed_environment}_mono_depth"
        ),
        "secondary_positive_direction": "frozen_gp_has_more_blind_distance",
        "secondary_multiplicity": "separate Holm family-wise correction across all 25 cells",
        "alpha": ALPHA,
        "bootstrap": {
            "method": "paired-task percentile bootstrap of the mean",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
        },
        "output_csv": str(out_path),
        "output_csv_sha256": sha256(out_path),
        "descriptive_counts": {
            "mean_did_positive_cells": sum(float(row["mean_did_m"]) > 0.0 for row in summaries),
            "raw_did_p_below_0_05": sum(float(row["did_sign_p_raw"]) < ALPHA for row in summaries),
            "holm_did_rejections_0_05": sum(bool(row["did_holm_reject_0_05"]) for row in summaries),
            "holm_changed_gap_rejections_0_05": sum(
                bool(row["changed_gap_holm_reject_0_05"]) for row in summaries
            ),
        },
    }
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, default=DEFAULT_RESULTS / "e3_routes.csv")
    parser.add_argument(
        "--route-manifest", type=Path, default=DEFAULT_RESULTS / "manifest.json"
    )
    parser.add_argument("--changed-environment", default="L1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    out = args.out or DEFAULT_RESULTS / f"e3_cell_summary_{args.changed_environment}.csv"
    out_manifest = args.out_manifest or DEFAULT_RESULTS / (
        f"e3_cell_summary_{args.changed_environment}_manifest.json"
    )

    summaries = write_summary(
        args.routes,
        args.route_manifest,
        out,
        out_manifest,
        changed_environment=args.changed_environment,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    raw = sum(float(row["did_sign_p_raw"]) < ALPHA for row in summaries)
    holm = sum(bool(row["did_holm_reject_0_05"]) for row in summaries)
    positive = sum(float(row["mean_did_m"]) > 0.0 for row in summaries)
    print(
        f"[e3-stats] wrote {out} ({len(summaries)} cells; "
        f"mean DiD positive in {positive}; raw p<0.05 in {raw}; Holm rejections {holm})"
    )
    print(f"[e3-stats] wrote {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
