#!/usr/bin/env python3
"""Replay F3's logged camera batches through F3 and the new F4, without closing the loop.

This is a mechanism diagnostic, not paper evidence. It uses the schema-5 one-seed campaign,
whose registry status is diagnostic-only, and applies both estimators to exactly the same
admitted observations at their common capture time. The required aligned loader supplies the
observations and ground-truth interpolation.

Writes two plots and a machine-readable summary under
``logs/studies/fusion_on_fixed_routes/joint_network_diagnostic``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "experiments/deck_figures"))
sys.path.insert(0, str(REPO / "src/reliability"))

import aligned as A  # noqa: E402
import style as D  # noqa: E402
from reliability.contracts import CameraQuality  # noqa: E402
from reliability.fusion import (  # noqa: E402
    MapObservation,
    independent_measurement_fusion_2d,
    joint_network_estimate_2d,
)

CAMPAIGN = REPO / "logs/studies/fusion_on_fixed_routes/diagnostic_schema5_20260831"
OUT = REPO / "logs/studies/fusion_on_fixed_routes/joint_network_diagnostic"
RULES = {
    "independent": (independent_measurement_fusion_2d, D.OLD, "F3 independent"),
    "joint": (joint_network_estimate_2d, D.GOOD, "F4 joint network"),
}


def _valid_covariance(covariance: np.ndarray) -> bool:
    return bool(np.isfinite(covariance).all() and np.linalg.det(covariance) > 0.0)


def load_batches(root: Path = CAMPAIGN) -> list[dict]:
    """One simultaneous admitted batch per source id, from each route's F3 run."""

    runs = sorted(path.parent for path in root.glob("*/F3/seed0/*/run_manifest.json"))
    if not runs:
        raise SystemExit(f"no schema-5 F3 runs below {root}")
    batches = []
    for run_index, run in enumerate(runs):
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
        identity = (manifest.get("logging_schema_version"),
                    manifest.get("manager_fusion_rule"),
                    manifest.get("manager_observation_model"))
        if identity != (5, "independent", "hull"):
            raise SystemExit(f"{run}: expected schema-5 independent/hull, got {identity}")
        if not summary.get("completed") or not summary.get("valid_run", False):
            raise SystemExit(f"{run}: diagnostic replay requires a completed valid run")
        truth = A.truth_series(run)
        grouped: dict[str, dict[str, dict]] = {}
        for row in A.observations(run):
            batch_id = row["source_batch_id"]
            if not batch_id or not row["used"]:
                continue
            grouped.setdefault(batch_id, {}).setdefault(row["camera"], row)
        for batch_id, by_camera in grouped.items():
            rows = list(by_camera.values())
            common_stamps = [row["common_capture_stamp"] for row in rows
                             if math.isfinite(row["common_capture_stamp"])]
            if not common_stamps:
                continue
            stamp = float(np.median(common_stamps))
            gx, gy = truth.at([stamp])
            if not (math.isfinite(gx[0]) and math.isfinite(gy[0])):
                continue
            observations = []
            for row in rows:
                xy = row["aligned_xy"]
                covariance = row["aligned_cov"]
                if not (np.isfinite(xy).all() and _valid_covariance(covariance)):
                    continue
                observations.append(MapObservation(
                    camera_id=row["camera"],
                    timestamp_s=stamp,
                    xy_m=tuple(xy),
                    covariance_m2=tuple(tuple(value for value in matrix_row)
                                        for matrix_row in covariance),
                    quality=CameraQuality(camera_id=row["camera"]),
                    source=batch_id,
                ))
            if observations:
                batches.append({
                    "run": run,
                    "run_index": run_index,
                    "route": manifest["task"],
                    "batch_id": batch_id,
                    "truth": np.array([gx[0], gy[0]], dtype=float),
                    "observations": observations,
                })
    return batches


def evaluate(batches: list[dict]) -> list[dict]:
    rows = []
    for batch in batches:
        for rule, (estimator, _colour, _label) in RULES.items():
            mean, covariance = estimator(batch["observations"])
            residual = np.asarray(mean) - batch["truth"]
            cov = np.asarray(covariance)
            rows.append({
                "rule": rule,
                "route": batch["route"],
                "run_index": batch["run_index"],
                "n": len(batch["observations"]),
                "error_cm": float(np.linalg.norm(residual) * 100.0),
                "sigma_cm": float(np.sqrt(np.trace(cov) / 2.0) * 100.0),
                "nees": float(residual @ np.linalg.solve(cov, residual)),
            })
    return rows


def _route_values(rows: list[dict], rule: str, n: int, key: str,
                  aggregate=np.median) -> list[float]:
    values = []
    for route in sorted({row["route"] for row in rows}):
        selected = [row[key] for row in rows
                    if row["route"] == route and row["rule"] == rule and row["n"] == n
                    and math.isfinite(row[key])]
        if selected:
            values.append(float(aggregate(selected)))
    return values


def _draw_range(ax, xs, medians, lows, highs, colour, label):
    ax.plot(xs, medians, "-o", lw=2.6, ms=8, color=colour, label=label)
    ax.fill_between(xs, lows, highs, color=colour, alpha=0.12)


def plot_error_and_claim(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.1), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "error_cm", "actual correction error"),
        (axes[1], "sigma_cm", "Gaussian claim after fusion (1σ)"),
    ):
        for rule, (_estimator, colour, label) in RULES.items():
            xs, medians, lows, highs = [], [], [], []
            for n in range(1, 6):
                values = _route_values(rows, rule, n, key)
                if not values:
                    continue
                xs.append(n)
                medians.append(float(np.median(values)))
                lows.append(float(min(values)))
                highs.append(float(max(values)))
            _draw_range(ax, xs, medians, lows, highs, colour, label)
        ax.set_title(title, loc="left", fontsize=16, color=D.INK)
        ax.set_xlabel("simultaneous admitted cameras", fontsize=12.5)
        ax.set_ylabel("centimetres", fontsize=12.5)
        ax.set_xticks(range(1, 6))
        ax.grid(True, color="#ecebe6", lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=12)
    fig.suptitle("The joint estimator changes the claim when cameras disagree",
                 x=0.005, ha="left", fontsize=20, color=D.INK)
    fig.text(0.005, -0.035,
             "DIAGNOSTIC REPLAY ONLY · schema 5 · one seed per route · old per-camera "
             "covariance · not a paper result. Points are medians of route medians; shading "
             "is the route range. Both rules receive the exact same admitted batch and are "
             "scored at its common capture time using the required aligned loader.",
             fontsize=11.2, color=D.INK2, va="top", linespacing=1.45)
    fig.savefig(OUT / "01_error_and_claim_vs_camera_count.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def plot_consistency_and_runtime(rows: list[dict], batches: list[dict]) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.8), constrained_layout=True)
    for rule, (_estimator, colour, label) in RULES.items():
        xs, medians, lows, highs = [], [], [], []
        coverage = []
        coverage_x = []
        for n in range(1, 6):
            values = _route_values(rows, rule, n, "nees")
            if values:
                xs.append(n); medians.append(float(np.median(values)))
                lows.append(float(min(values))); highs.append(float(max(values)))
            route_coverage = _route_values(
                [{**row, "covered": float(row["nees"] <= A.CHI2_95_2D)} for row in rows],
                rule, n, "covered", aggregate=np.mean)
            if route_coverage:
                coverage_x.append(n); coverage.append(float(np.median(route_coverage)) * 100.0)
        _draw_range(axes[0], xs, medians, lows, highs, colour, label)
        axes[1].plot(coverage_x, coverage, "-o", lw=2.6, ms=8, color=colour, label=label)

    axes[0].axhline(A.NEES_MEDIAN_TARGET, color=D.INK, ls="--", lw=1.8,
                    label="Gaussian target")
    axes[0].set_title("median NEES", loc="left", fontsize=15.5)
    axes[0].set_ylabel("target = 1.386", fontsize=12)
    axes[1].axhline(95.0, color=D.INK, ls="--", lw=1.8)
    axes[1].set_title("truth inside stated 95% ellipse", loc="left", fontsize=15.5)
    axes[1].set_ylabel("coverage (%) · target = 95%", fontsize=12)

    timings = {}
    for rule, (estimator, colour, label) in RULES.items():
        ns, microseconds = [], []
        for n in range(1, 6):
            samples = [batch["observations"] for batch in batches
                       if len(batch["observations"]) == n][:100]
            if not samples:
                continue
            repeats = max(5, 1000 // len(samples))
            start = time.perf_counter_ns()
            calls = 0
            for _ in range(repeats):
                for sample in samples:
                    estimator(sample)
                    calls += 1
            elapsed_us = (time.perf_counter_ns() - start) / 1000.0 / calls
            ns.append(n); microseconds.append(float(elapsed_us))
        timings[rule] = {"n": ns, "microseconds_per_batch": microseconds}
        axes[2].plot(ns, microseconds, "-o", lw=2.6, ms=8, color=colour, label=label)
    axes[2].set_title("fusion computation only", loc="left", fontsize=15.5)
    axes[2].set_ylabel("microseconds per batch", fontsize=12)
    for ax in axes:
        ax.set_xlabel("simultaneous admitted cameras", fontsize=11.5)
        ax.set_xticks(range(1, 6))
        ax.grid(True, color="#ecebe6", lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=10.5)
    fig.suptitle("Consistency remains the test; the joint solve is not the speed bottleneck",
                 x=0.005, ha="left", fontsize=20, color=D.INK)
    fig.text(0.005, -0.04,
             "DIAGNOSTIC REPLAY ONLY · schema 5 · one seed per route · old per-camera "
             "covariance · not a paper result. NEES and coverage use one physical source "
             "batch once. Runtime is a local single-process microbenchmark of fusion only; "
             "it excludes detection, ROS transport, simulation, and control.",
             fontsize=11.0, color=D.INK2, va="top", linespacing=1.45)
    fig.savefig(OUT / "02_consistency_and_runtime.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return timings


def main() -> int:
    root = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else CAMPAIGN
    batches = load_batches(root)
    rows = evaluate(batches)
    OUT.mkdir(parents=True, exist_ok=True)
    plot_error_and_claim(rows)
    timings = plot_consistency_and_runtime(rows, batches)
    counts = {str(n): sum(len(batch["observations"]) == n for batch in batches)
              for n in range(1, 6)}
    diagnostic_metrics = {}
    for rule in RULES:
        diagnostic_metrics[rule] = {}
        for n in range(1, 6):
            error = _route_values(rows, rule, n, "error_cm")
            sigma = _route_values(rows, rule, n, "sigma_cm")
            nees = _route_values(rows, rule, n, "nees")
            coverage = _route_values(
                [{**row, "covered": float(row["nees"] <= A.CHI2_95_2D)} for row in rows],
                rule, n, "covered", aggregate=np.mean)
            if error:
                diagnostic_metrics[rule][str(n)] = {
                    "median_of_route_median_error_cm": float(np.median(error)),
                    "median_of_route_median_sigma_cm": float(np.median(sigma)),
                    "median_of_route_median_nees": float(np.median(nees)),
                    "median_route_coverage_95": float(np.median(coverage)),
                }
    summary = {
        "status": "diagnostic_only_not_paper_evidence",
        "campaign": str(root.relative_to(REPO)),
        "required_loader": "experiments/fusion_on_fixed_routes/aligned.py",
        "runs": len({str(batch["run"]) for batch in batches}),
        "batches": len(batches),
        "batches_by_camera_count": counts,
        "diagnostic_metrics": diagnostic_metrics,
        "timings": timings,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
