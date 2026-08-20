#!/usr/bin/env python3
"""Audit availability-map differences against routes the planner actually solved.

This diagnostic keeps three objects separate:

1. availability fields A1/A2/A3 over all driveable cells;
2. the E5 runtime-objective offline solver's recorded route-seed selection;
3. persisted ``global_plan.csv`` coordinates from the interrupted E4 testing run.

It does not use Dijkstra/map-implied routes, executed trajectories, or ground truth.
The run list is frozen below; discovery globs are used only to assert that no
unexpected persisted plan has silently entered the analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import render_all as base  # noqa: E402


OUT = C.OUT_ROOT / "e6_offline_map_plan_audit"
FIGURES = C.OUT_ROOT / "figures"
E5 = C.OUT_ROOT / "e5_offline_efe_solve/e5_offline_efe_solve.csv"
RUN_ROOT = C.REPO / "logs/visibility_comparison/e4_availability_closed_loop_v1/mc_blind_L"

FIELDS = {
    "A1_operational_gp": C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz",
    "A2_mono_depth": C.OUT_ROOT / "mono_depth_planner_v1/fused_planner_four_camera.npz",
    "A3_depth_plus_gp": C.OUT_ROOT / "depth_gp_planner_v1/fused_planner_four_camera.npz",
}

# Frozen campaign inventory. C3 seed3 was interrupted before a global plan was
# persisted; it is kept in the manifest but is not a plan-level experimental unit.
RUNS = (
    ("C1", 0, "C1/seed0/experiment_20260818_092254", True),
    ("C1", 1, "C1/seed1/experiment_20260818_092500", True),
    ("C1", 2, "C1/seed2/experiment_20260818_092730", True),
    ("C1", 3, "C1/seed3/experiment_20260818_093039", True),
    ("C1", 4, "C1/seed4/experiment_20260818_093247", True),
    ("C2", 0, "C2/seed0/experiment_20260818_093532", True),
    ("C2", 1, "C2/seed1/experiment_20260818_093841", True),
    ("C2", 3, "C2/seed3/experiment_20260818_095825", True),
    ("C2", 4, "C2/seed4/experiment_20260818_100035", True),
    ("C3", 0, "C3/seed0/experiment_20260818_100342", True),
    ("C3", 1, "C3/seed1/experiment_20260818_100754", True),
    ("C3", 2, "C3/seed2/experiment_20260818_101252", True),
    ("C3", 3, "C3/seed3/experiment_20260818_101725", False),
)

COLOURS = {
    "A1_operational_gp": "#1f4fd8",
    "A2_mono_depth": "#d89000",
    "A3_depth_plus_gp": "#7b53b5",
    "C1": "#d62728",
    "C2": "#1f4fd8",
    "C3": "#00a6a6",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_plan(path: Path) -> np.ndarray:
    rows = _read_csv(path)
    plan = np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=float)
    if plan.ndim != 2 or plan.shape[1] != 2 or len(plan) < 2 or not np.isfinite(plan).all():
        raise RuntimeError(f"invalid persisted global plan: {path}")
    return plan


def _plan_length(plan: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(plan, axis=0), axis=1)))


def _arclength(plan: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(plan, axis=0), axis=1))])


def _sample(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, plan: np.ndarray) -> np.ndarray:
    return np.asarray(C.sample_field_at(field, xs, ys, plan), dtype=float)


def _route_hash(plan: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(plan, dtype="<f8").tobytes()).hexdigest()


def main() -> int:
    apparatus = C.build_apparatus()
    fields = {
        name: np.asarray(np.load(path)["P_conservative_plan_map"], dtype=float)
        for name, path in FIELDS.items()
    }
    expected_shape = apparatus.driveable.shape
    for name, field in fields.items():
        if field.shape != expected_shape:
            raise RuntimeError(f"{name} shape {field.shape} != apparatus {expected_shape}")

    expected_plan_dirs = {rel for _condition, _seed, rel, has_plan in RUNS if has_plan}
    discovered = {
        str(path.parent.relative_to(RUN_ROOT))
        for path in RUN_ROOT.glob("C*/seed*/experiment_*/global_plan.csv")
    }
    if discovered != expected_plan_dirs:
        raise RuntimeError(
            "persisted-plan inventory changed; update the frozen manifest explicitly: "
            f"missing={sorted(expected_plan_dirs - discovered)}, extra={sorted(discovered - expected_plan_dirs)}"
        )

    plan_rows: list[dict[str, object]] = []
    plans: list[tuple[str, int, np.ndarray]] = []
    for condition, seed, relative, expect_plan in RUNS:
        run_dir = RUN_ROOT / relative
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"missing frozen run summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        plan_path = run_dir / "global_plan.csv"
        meta_path = run_dir / "global_plan_meta.json"
        if plan_path.is_file() != expect_plan or meta_path.is_file() != expect_plan:
            raise RuntimeError(f"unexpected persisted-plan state for {run_dir}")
        if not expect_plan:
            plan_rows.append({
                "condition": condition,
                "seed": seed,
                "run_id": run_dir.name,
                "plan_persisted": False,
                "completion_reason": summary.get("completion_reason", ""),
                "selected_source": "",
                "n_points": "",
                "length_m": "",
                "route_hash": "",
                "max_deviation_from_canonical_m": "",
            })
            continue
        plan = _load_plan(plan_path)
        meta = json.loads(meta_path.read_text())
        plans.append((condition, seed, plan))
        plan_rows.append({
            "condition": condition,
            "seed": seed,
            "run_id": run_dir.name,
            "plan_persisted": True,
            "completion_reason": summary.get("completion_reason", ""),
            "selected_source": meta.get("selected_source", ""),
            "n_points": len(plan),
            "length_m": _plan_length(plan),
            "route_hash": _route_hash(plan),
            "max_deviation_from_canonical_m": "",  # filled after canonical selection
        })

    canonical = plans[0][2]
    if any(plan.shape != canonical.shape for _condition, _seed, plan in plans):
        raise RuntimeError("persisted plans have different shapes; pointwise audit is undefined")
    deviations = [float(np.max(np.linalg.norm(plan - canonical, axis=1))) for _, _, plan in plans]
    persisted_index = 0
    for row in plan_rows:
        if row["plan_persisted"]:
            row["max_deviation_from_canonical_m"] = deviations[persisted_index]
            persisted_index += 1

    mask = np.asarray(apparatus.driveable, dtype=bool)
    pair_rows: list[dict[str, object]] = []
    names = list(fields)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            delta = fields[left][mask] - fields[right][mask]
            pair_rows.append({
                "field_a": left,
                "field_b": right,
                "n_driveable_cells": int(mask.sum()),
                "mean_abs_difference": float(np.mean(np.abs(delta))),
                "rmse_difference": float(np.sqrt(np.mean(delta**2))),
                "max_abs_difference": float(np.max(np.abs(delta))),
                "pearson": float(np.corrcoef(fields[left][mask], fields[right][mask])[0, 1]),
            })

    route_s = _arclength(canonical)
    profile_rows: list[dict[str, object]] = []
    route_profiles: dict[str, np.ndarray] = {}
    for name, field in fields.items():
        profile = _sample(field, apparatus.xs, apparatus.ys, canonical)
        route_profiles[name] = profile
        for point_index, (distance_m, value) in enumerate(zip(route_s, profile)):
            profile_rows.append({
                "field": name,
                "point_index": point_index,
                "distance_m": float(distance_m),
                "p_use": float(value),
            })

    e5_rows = _read_csv(E5)
    if len(e5_rows) != 12:
        raise RuntimeError(f"expected 12 frozen E5 solves, found {len(e5_rows)}")
    e5_same = sum(row["selected"] == "solver:route:availability_blind" for row in e5_rows)
    e5_warm_or_cold = sum(row["selected"] == "solver:warm_or_cold" for row in e5_rows)
    e5_tasks = sorted({row["task"] for row in e5_rows})
    e5_task_consensus = {
        task: sorted({row["selected"] for row in e5_rows if row["task"] == task})
        for task in e5_tasks
    }
    n_e5_consensus_tasks = sum(len(values) == 1 for values in e5_task_consensus.values())

    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(
        OUT / "map_pairwise.csv",
        pair_rows,
        ("field_a", "field_b", "n_driveable_cells", "mean_abs_difference", "rmse_difference", "max_abs_difference", "pearson"),
    )
    _write_csv(
        OUT / "plan_inventory.csv",
        plan_rows,
        ("condition", "seed", "run_id", "plan_persisted", "completion_reason", "selected_source", "n_points", "length_m", "route_hash", "max_deviation_from_canonical_m"),
    )
    _write_csv(OUT / "route_profiles.csv", profile_rows, ("field", "point_index", "distance_m", "p_use"))

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.0), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.96, top=0.88, bottom=0.13, wspace=0.23, hspace=0.32)
    fig.suptitle("Offline audit: map differences versus routes the planner actually solved", fontsize=15, weight="bold")

    ax = axes[0, 0]
    labels = [f"{row['field_a'][:2]} vs {row['field_b'][:2]}" for row in pair_rows]
    values = [float(row["mean_abs_difference"]) for row in pair_rows]
    bars = ax.bar(labels, values, color=["#4C78A8", "#7B53B5", "#D89000"], edgecolor="#222222")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel(r"mean absolute difference in $p_{use}$")
    ax.set_title("(a) Maps differ over driveable cells", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    for name, profile in route_profiles.items():
        ax.plot(route_s, profile, lw=2.2, label=name.replace("_", " "), color=COLOURS[name])
    ax.axhline(0.2, color="#B0271F", lw=1.2, ls="--", label="low-availability threshold")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("distance along persisted global plan [m]")
    ax.set_ylabel(r"sampled $p_{use}$")
    ax.set_title("(b) The same solved route crosses different map values", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 0]
    base.draw_field(ax, fields["A1_operational_gp"], apparatus.xs, apparatus.ys,
                    apparatus.driveable, apparatus.prisms, title="")
    for artist in list(ax.texts):
        artist.set_clip_on(True)
    # Draw widest first so coincident lines remain legible without spatial offsets.
    widths = {"C1": 6.0, "C2": 4.0, "C3": 2.0}
    styles = {"C1": "-", "C2": "--", "C3": ":"}
    for condition in ("C1", "C2", "C3"):
        representative = next(plan for cond, _seed, plan in plans if cond == condition)
        ax.plot(representative[:, 0], representative[:, 1], color=COLOURS[condition],
                lw=widths[condition], ls=styles[condition], label=condition, zorder=8)
    ax.set_title("(c) Persisted global plans overlap exactly", weight="bold")
    ax.legend(title="planner condition", fontsize=8, loc="lower left")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax = axes[1, 1]
    ax.axis("off")
    unique_routes = len({str(row["route_hash"]) for row in plan_rows if row["plan_persisted"]})
    max_deviation = max(deviations)
    condition_counts = {
        condition: sum(1 for cond, _seed, _plan in plans if cond == condition)
        for condition in ("C1", "C2", "C3")
    }
    text = (
        "Runtime-objective EFE replay\n"
        f"  {n_e5_consensus_tasks}/{len(e5_tasks)} tasks: all four fields chose the same source\n"
        f"  availability_blind: {e5_same} solves; warm_or_cold: {e5_warm_or_cold}\n"
        "  optimized coordinates were not persisted by E5\n\n"
        "Persisted global plans from testing campaign\n"
        f"  {len(plans)} plans: C1={condition_counts['C1']}, C2={condition_counts['C2']}, C3={condition_counts['C3']}\n"
        f"  unique coordinate arrays: {unique_routes}\n"
        f"  maximum pointwise deviation: {1000.0 * max_deviation:.6f} mm\n"
        "  C3 seed3: interrupted before a plan was persisted\n\n"
        "Conclusion\n"
        "  The maps differ. E5 did not change the selected seed class;\n"
        "  the persisted mc_blind_L plans did not change coordinates."
    )
    ax.text(0.04, 0.95, text, va="top", ha="left", fontsize=11.0, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#F8FAFC", edgecolor="#98A2B3"))
    ax.set_title("(d) Decision audit", weight="bold")

    fig.text(
        0.5, 0.025,
        "Map unit: driveable grid cell. Plan unit: one persisted global_plan.csv from the frozen run list. "
        "No Dijkstra routes, executed trajectories, or ground truth are used.\n"
        "Testing evidence only: the campaign is incomplete and this is not a navigation-performance comparison.",
        ha="center", fontsize=8.8, color="#475467",
    )
    figure_path = FIGURES / "11_offline_map_vs_solved_routes.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)

    summary = {
        "status": "testing_diagnostic_incomplete_campaign",
        "world": "warehouse_full_4cam.world.sdf",
        "map_metric_object": "availability_prediction_field",
        "map_experimental_unit": "driveable_grid_cell",
        "plan_metric_object": "persisted_global_planner_solution",
        "plan_experimental_unit": "global_plan_csv",
        "reference": "none; direct field and route comparison",
        "online_inputs": ["availability artifact", "planner prior", "goal", "registered route seeds", "runtime EFE objective"],
        "evaluation_only_inputs": [],
        "ground_truth_used": False,
        "dijkstra_routes_used": False,
        "executed_trajectories_used": False,
        "map_artifacts": {name: str(path.relative_to(C.REPO)) for name, path in FIELDS.items()},
        "run_inventory": [
            {"condition": condition, "seed": seed, "relative_run_dir": relative, "plan_expected": expect_plan}
            for condition, seed, relative, expect_plan in RUNS
        ],
        "n_persisted_plans": len(plans),
        "persisted_plan_counts": condition_counts,
        "unique_persisted_route_arrays": unique_routes,
        "max_pointwise_plan_deviation_m": max_deviation,
        "e5_n_solves": len(e5_rows),
        "e5_n_selecting_availability_blind": e5_same,
        "e5_n_selecting_warm_or_cold": e5_warm_or_cold,
        "e5_task_selected_source_sets": e5_task_consensus,
        "e5_n_tasks_with_cross_field_selected_source_consensus": n_e5_consensus_tasks,
        "map_pairwise": pair_rows,
        "route_profile_summary": {
            name: {
                "mean_p_use": float(np.mean(profile)),
                "min_p_use": float(np.min(profile)),
                "fraction_points_below_0_2": float(np.mean(profile < 0.2)),
            }
            for name, profile in route_profiles.items()
        },
        "restrictions": [
            "campaign is incomplete",
            "persisted-plan comparison covers mc_blind_L only",
            "E5 records selected seed and costs but does not persist optimized coordinates",
            "route profiles are evaluated on one coordinate-identical persisted route",
            "no navigation outcome or localization accuracy claim",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {figure_path}")
    print(f"wrote {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
