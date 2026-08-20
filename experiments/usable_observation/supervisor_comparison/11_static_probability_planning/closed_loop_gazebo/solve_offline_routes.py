#!/usr/bin/env python3
"""Solve and export complete, clearance-validated routes without starting Gazebo.

This is the executable-route bridge between the static probability experiment and a
future closed-loop run. Every condition chooses among the same complete grid routes.
No continuous EFE optimization is involved in global route discovery here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/static_puse_offline_routes_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SUPERVISOR = STUDY.parent
REPO = HERE.parents[4]
OUTPUT = HERE / "offline_routes"
STATIC_INPUTS = STUDY / "results/static_planning_inputs.npz"

for source in (REPO / "src/reliability", REPO / "src/planning", REPO / "src/unav_common"):
    sys.path.insert(0, str(source))
sys.path.insert(0, str(SUPERVISOR))
sys.path.insert(0, str(STUDY))

import render_all as base  # noqa: E402
import run_experiment as offline  # noqa: E402
from unav_common.occlusion_geometry import signed_distance_to_union_xy  # noqa: E402


TASK_NAME = "route_tall_shadow_west_safe_start"
# Use the stricter global keep-in contract (the local controller requires 0.20 m).
# This leaves at least 0.05 m nominal headroom for local tracking error.
MIN_DRIVEABLE_CLEARANCE_M = 0.25
ENDPOINT_TOLERANCE_M = 0.08  # < one 9.8 cm grid step
VALIDATION_STEP_M = 0.04
METHODS = {
    "C1": "availability_blind_shortest",
    "C2": "r_over_p_shortcut",
    "C3": "explicit_hit_miss",
}
LABELS = {
    "C1": "constant R: shortest complete route",
    "C2": r"static $p_{use}$: $R/p$",
    "C3": r"static $p_{use}$: explicit hit/miss",
}
COLOURS = {"C1": "#d55e00", "C2": "#0072b2", "C3": "#009e73"}


class NoFeasibleRouteError(RuntimeError):
    """Raised instead of returning or executing a partial global route."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_polyline(path: np.ndarray, step_m: float = VALIDATION_STEP_M) -> np.ndarray:
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return path.copy()
    samples = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(b - a))
        count = max(1, int(math.ceil(distance / max(step_m, 1e-6))))
        samples.extend(a + alpha * (b - a) for alpha in np.linspace(0.0, 1.0, count + 1)[1:])
    return np.asarray(samples, dtype=float)


def driveable_clearance(path: np.ndarray, driveable_prisms) -> np.ndarray:
    samples = _sample_polyline(path)
    signed = signed_distance_to_union_xy(tuple(driveable_prisms), samples, keep_in=True)
    # keep-in signed distance is <= 0 inside the union.
    return -np.asarray(signed, dtype=float)


def validate_complete_route(
    path: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    driveable_prisms,
    *,
    required_clearance_m: float = MIN_DRIVEABLE_CLEARANCE_M,
    endpoint_tolerance_m: float = ENDPOINT_TOLERANCE_M,
) -> dict:
    """Fail closed unless the whole polyline is executable under the local contract."""
    route = np.asarray(path, dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
        raise NoFeasibleRouteError("route is empty or malformed")
    if not np.all(np.isfinite(route)):
        raise NoFeasibleRouteError("route contains non-finite coordinates")
    start_error = float(np.linalg.norm(route[0] - np.asarray(start_xy, dtype=float)))
    goal_error = float(np.linalg.norm(route[-1] - np.asarray(goal_xy, dtype=float)))
    if start_error > endpoint_tolerance_m:
        raise NoFeasibleRouteError(
            f"route start mismatch {start_error:.3f} m > {endpoint_tolerance_m:.3f} m"
        )
    if goal_error > endpoint_tolerance_m:
        raise NoFeasibleRouteError(
            f"route goal mismatch {goal_error:.3f} m > {endpoint_tolerance_m:.3f} m"
        )
    clearances = driveable_clearance(route, driveable_prisms)
    if len(clearances) == 0 or not np.all(np.isfinite(clearances)):
        raise NoFeasibleRouteError("route clearance could not be evaluated")
    minimum = float(np.min(clearances))
    if minimum + 1e-9 < required_clearance_m:
        raise NoFeasibleRouteError(
            f"route clearance {minimum:.3f} m < required {required_clearance_m:.3f} m"
        )
    return {
        "complete": True,
        "start_error_m": start_error,
        "goal_error_m": goal_error,
        "minimum_driveable_clearance_m": minimum,
        "required_driveable_clearance_m": float(required_clearance_m),
        "validation_step_m": VALIDATION_STEP_M,
    }


def _safe_grid_mask(ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(ctx["xs"], ctx["ys"])
    points = np.column_stack((gx.ravel(), gy.ravel()))
    clearance = -np.asarray(
        signed_distance_to_union_xy(tuple(ctx["driveable_prisms"]), points, keep_in=True),
        dtype=float,
    ).reshape(gx.shape)
    safe = np.asarray(ctx["driveable"], dtype=bool) & (clearance >= MIN_DRIVEABLE_CLEARANCE_M)
    return safe, clearance


def lean_context() -> dict:
    """Load only frozen route inputs; never rebuild depth/raycast supervisor fields."""
    with np.load(STATIC_INPUTS, allow_pickle=False) as artifact:
        context = {
            "xs": np.asarray(artifact["xs"], dtype=float),
            "ys": np.asarray(artifact["ys"], dtype=float),
            "driveable": np.asarray(artifact["driveable"], dtype=bool),
            "selected_index": np.asarray(artifact["selected_camera_index"], dtype=np.int16),
            "selected_p": np.asarray(artifact["selected_p_use"], dtype=float),
            "planning_fields": {
                camera: np.asarray(artifact[f"planning_{camera}"], dtype=float)
                for camera in offline.CAMERAS
            },
        }
    context["driveable_prisms"] = base.profile_regions()
    context["prisms"] = tuple(
        base.parse_occlusion_scene_from_world(
            str(base.WORLD),
            model_name="warehouse_rack_occluders",
            geometry_tags=("collision",),
        ).prisms
    )
    context["tasks"] = base.tasks()
    return context


def _with_exact_endpoints(path: np.ndarray, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    route = np.asarray(path, dtype=float).reshape(-1, 2)
    points = [np.asarray(start_xy, dtype=float)]
    points.extend(route)
    points.append(np.asarray(goal_xy, dtype=float))
    deduped = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - deduped[-1])) > 1e-8:
            deduped.append(point)
    return np.asarray(deduped, dtype=float)


def _segment_valid(a: np.ndarray, b: np.ndarray, driveable_prisms) -> bool:
    clearance = driveable_clearance(np.vstack((a, b)), driveable_prisms)
    return bool(len(clearance) and np.min(clearance) + 1e-9 >= MIN_DRIVEABLE_CLEARANCE_M)


def simplify_clearance_safe(path: np.ndarray, driveable_prisms) -> np.ndarray:
    """Greedy line-of-sight simplification that preserves the clearance gate."""
    route = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(route) <= 2:
        return route.copy()
    output = [route[0]]
    index = 0
    while index < len(route) - 1:
        chosen = index + 1
        for candidate in range(len(route) - 1, index, -1):
            if _segment_valid(route[index], route[candidate], driveable_prisms):
                chosen = candidate
                break
        output.append(route[chosen])
        index = chosen
    simplified = np.asarray(output, dtype=float)
    validate_complete_route(
        simplified, route[0], route[-1], driveable_prisms,
        endpoint_tolerance_m=1e-6,
    )
    return simplified


def build_validated_candidates(ctx: dict, config: dict, selected_p: np.ndarray) -> list[dict]:
    task = ctx["tasks"][TASK_NAME]
    start = np.asarray([task["start"]["x"], task["start"]["y"]], dtype=float)
    goal = np.asarray([task["goal"]["x"], task["goal"]["y"]], dtype=float)
    safe_mask, clearance_map = _safe_grid_mask(ctx)
    valid: list[dict] = []
    failures: list[str] = []
    seen: set[bytes] = set()
    for reliability_lambda in config["route_search"]["candidate_reliability_lambdas"]:
        try:
            grid_path = offline.weighted_route(
                selected_p,
                safe_mask,
                ctx["xs"],
                ctx["ys"],
                tuple(start),
                tuple(goal),
                float(reliability_lambda),
            )
            route = _with_exact_endpoints(grid_path, start, goal)
            validation = validate_complete_route(route, start, goal, ctx["driveable_prisms"])
        except (RuntimeError, NoFeasibleRouteError) as exc:
            failures.append(f"lambda={reliability_lambda}: {exc}")
            continue
        key = np.round(route, 5).tobytes()
        if key in seen:
            continue
        seen.add(key)
        valid.append({
            "candidate_id": len(valid),
            "reliability_lambda": float(reliability_lambda),
            "path": route,
            "validation": validation,
        })
    if not valid:
        detail = "; ".join(failures) if failures else "no candidate was generated"
        raise NoFeasibleRouteError(f"No complete clearance-safe route for {TASK_NAME}: {detail}")
    # Endpoint safety is checked against exact geometry, not merely the raster mask.
    del clearance_map
    return valid


def solve() -> tuple[dict, list[dict], list[dict]]:
    config = offline.read_config()
    ctx = lean_context()
    if TASK_NAME not in ctx["tasks"]:
        raise KeyError(f"Task {TASK_NAME!r} is absent from {base.TASKS}")
    r_cond, r_provenance = offline.load_r_cond()
    planning_fields = ctx["planning_fields"]
    selected_index, selected_p = ctx["selected_index"], ctx["selected_p"]
    candidates = build_validated_candidates(ctx, config, selected_p)
    step_m = float(config["dynamics"]["dt_s"]) * float(config["dynamics"]["speed_mps"])

    candidate_rows: list[dict] = []
    evaluated: dict[str, list[dict]] = {method: [] for method in METHODS.values()}
    for candidate in candidates:
        controller_path = simplify_clearance_safe(candidate["path"], ctx["driveable_prisms"])
        dense = offline.resample_path(controller_path, step_m)
        # Reassert exact task endpoints after resampling.
        dense[0] = candidate["path"][0]
        dense[-1] = candidate["path"][-1]
        validation = validate_complete_route(
            dense, dense[0], dense[-1], ctx["driveable_prisms"], endpoint_tolerance_m=1e-6
        )
        for method in METHODS.values():
            covariance_model = (
                "explicit_hit_miss" if method == "explicit_hit_miss" else "r_over_p_shortcut"
            )
            score = offline.propagate_route(
                dense,
                covariance_model,
                selected_index,
                planning_fields,
                r_cond,
                ctx["xs"],
                ctx["ys"],
                config,
            )
            record = {
                **candidate,
                "controller_path": controller_path,
                "dense_path": dense,
                "score": score,
                "validation": validation,
            }
            evaluated[method].append(record)
            candidate_rows.append({
                "candidate_id": candidate["candidate_id"],
                "reliability_lambda": candidate["reliability_lambda"],
                "method": method,
                "controller_waypoint_count": int(len(controller_path) - 1),
                **{k: v for k, v in score.items() if k != "camera_counts"},
                **validation,
            })

    selections: dict[str, dict] = {}
    selected_rows: list[dict] = []
    for condition, method in METHODS.items():
        if not evaluated[method]:
            raise NoFeasibleRouteError(f"No feasible candidate remained for {condition}")
        if method == "availability_blind_shortest":
            winner = min(
                evaluated[method],
                key=lambda item: (item["score"]["path_length_m"], item["score"]["objective_m"]),
            )
        else:
            winner = min(
                evaluated[method],
                key=lambda item: (item["score"]["objective_m"], item["score"]["path_length_m"]),
            )
        simplified = np.asarray(winner["controller_path"], dtype=float)
        validation = validate_complete_route(
            simplified,
            winner["path"][0],
            winner["path"][-1],
            ctx["driveable_prisms"],
            endpoint_tolerance_m=1e-6,
        )
        selections[condition] = {
            "condition": condition,
            "method": method,
            "candidate_id": int(winner["candidate_id"]),
            "reliability_lambda": float(winner["reliability_lambda"]),
            "path_xy": winner["dense_path"].tolist(),
            "controller_waypoints_xy": simplified[1:].tolist(),
            "score": {k: v for k, v in winner["score"].items() if k != "camera_counts"},
            "validation": validation,
        }
        selected_rows.append({
            "condition": condition,
            "method": method,
            "candidate_id": int(winner["candidate_id"]),
            "reliability_lambda": float(winner["reliability_lambda"]),
            "controller_waypoint_count": int(len(simplified) - 1),
            **{k: v for k, v in winner["score"].items() if k != "camera_counts"},
            **validation,
        })

    task = ctx["tasks"][TASK_NAME]
    payload = {
        "schema_version": 1,
        "status": "offline_executable_route_solve",
        "world": base.WORLD.name,
        "task": TASK_NAME,
        "start_xy": [float(task["start"]["x"]), float(task["start"]["y"])],
        "goal_xy": [float(task["goal"]["x"]), float(task["goal"]["y"])],
        "route_contract": {
            "global_solver": "8-neighbor Dijkstra over clearance-eroded driveable grid",
            "candidate_generation": "common reliability-lambda sweep",
            "selection": {
                "C1": "minimum complete path length",
                "C2": "minimum path_length + weighted covariance integral using R/p",
                "C3": "minimum path_length + weighted covariance integral using explicit hit/miss",
            },
            "required_driveable_clearance_m": MIN_DRIVEABLE_CLEARANCE_M,
            "validation_sample_step_m": VALIDATION_STEP_M,
            "endpoint_tolerance_m": ENDPOINT_TOLERANCE_M,
            "failure_behavior": "raise NoFeasibleRouteError; never return a partial path",
        },
        "controller_interface": (
            "controller_waypoints_xy excludes the start and includes the exact mission goal; "
            "global continuous EFE optimization must be bypassed when consuming this artifact"
        ),
        "probability_field": "per-cell camera selected by frozen expected-information policy; per-camera P_conservative_plan_map",
        "conditional_covariance_provenance": r_provenance,
        "candidate_count": len(candidates),
        "selections": selections,
    }
    return payload, candidate_rows, selected_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(payload: dict, output: Path) -> None:
    ctx = lean_context()
    selected_p = ctx["selected_p"]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.7), sharex=True, sharey=True)
    image = None
    for axis, condition in zip(axes, METHODS):
        image = axis.imshow(
            np.where(ctx["driveable"], selected_p, np.nan),
            origin="lower",
            extent=(ctx["xs"][0], ctx["xs"][-1], ctx["ys"][0], ctx["ys"][-1]),
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        base.draw_geometry(axis, ctx["prisms"], alpha=0.72)
        selection = payload["selections"][condition]
        path = np.asarray(selection["path_xy"], dtype=float)
        waypoints = np.vstack((np.asarray(payload["start_xy"]), selection["controller_waypoints_xy"]))
        axis.plot(path[:, 0], path[:, 1], color=COLOURS[condition], linewidth=3.0, zorder=8)
        axis.plot(waypoints[:, 0], waypoints[:, 1], color="white", linewidth=1.0,
                  marker="o", markersize=3.8, markeredgecolor=COLOURS[condition], zorder=9)
        axis.scatter(*payload["start_xy"], s=65, color="#f0e442", edgecolor="#222", zorder=10)
        axis.scatter(*payload["goal_xy"], s=160, marker="*", color="#cc79a7",
                     edgecolor="white", zorder=10)
        score, valid = selection["score"], selection["validation"]
        axis.set_title(f"{condition}: {LABELS[condition]}\n"
                       f"candidate {selection['candidate_id']} · $\lambda$={selection['reliability_lambda']:g}",
                       color=COLOURS[condition], weight="bold")
        axis.text(
            0.02, 0.02,
            f"COMPLETE     yes\n"
            f"path         {score['path_length_m']:.2f} m\n"
            f"goal error   {valid['goal_error_m']:.3f} m\n"
            f"min clear.   {valid['minimum_driveable_clearance_m']:.3f} m\n"
            f"mean p_use   {score['mean_planning_p_use']:.3f}\n"
            f"min p_use    {score['min_planning_p_use']:.3f}",
            transform=axis.transAxes, va="bottom", fontsize=8.1, family="monospace",
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "#aab3bb"}, zorder=20,
        )
        axis.set_xlim(-11.5, 11.8)
        axis.set_ylim(-8.8, 8.8)
        axis.set_aspect("equal")
        axis.set_xlabel("map x [m]")
        axis.grid(color="white", alpha=0.20, linewidth=0.5)
    axes[0].set_ylabel("map y [m]")
    fig.colorbar(image, ax=axes, fraction=0.020, pad=0.014,
                 label=r"frozen selected-camera $p_{use}(x)$")
    fig.suptitle("Offline global routes that satisfy the closed-loop execution contract",
                 x=0.045, ha="left", fontsize=16, weight="bold")
    fig.text(0.045, 0.925,
             "All conditions choose among the same complete, clearance-validated candidates; no Gazebo or continuous global EFE solve.",
             fontsize=9.5, color="#4d5965")
    fig.text(0.5, 0.018,
             "OFFLINE SOLVE — these paths are ready for later closed-loop execution but are not navigation outcomes.",
             ha="center", fontsize=8.7, color="#8a3232", weight="bold")
    fig.subplots_adjust(left=0.05, right=0.93, top=0.84, bottom=0.12, wspace=0.07)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload, candidates, selected = solve()
    routes_path = OUTPUT / "routes.json"
    routes_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(OUTPUT / "candidates.csv", candidates)
    _write_csv(OUTPUT / "selected.csv", selected)
    point_rows = []
    for condition, selection in payload["selections"].items():
        for index, xy in enumerate(selection["path_xy"]):
            point_rows.append({"condition": condition, "point_index": index, "x": xy[0], "y": xy[1]})
    _write_csv(OUTPUT / "route_points.csv", point_rows)
    figure = OUTPUT / "01_executable_routes.png"
    _render(payload, figure)
    manifest = {
        "generator": str(Path(__file__).relative_to(REPO)),
        "inputs": {
            str(path.relative_to(REPO)): _sha256(path)
            for path in (offline.CONFIG_PATH, offline.PG_SUMMARY, base.PROFILE, base.TASKS,
                         base.GP_FUSED, STATIC_INPUTS)
        },
        "outputs": {
            path.name: _sha256(path)
            for path in (routes_path, OUTPUT / "candidates.csv", OUTPUT / "selected.csv",
                         OUTPUT / "route_points.csv", figure)
        },
        "gazebo_processes_started": False,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {routes_path}")
    print(f"Wrote {figure}")
    for row in selected:
        print(
            f"{row['condition']}: candidate={row['candidate_id']} "
            f"length={row['path_length_m']:.2f}m clearance={row['minimum_driveable_clearance_m']:.3f}m "
            f"goal_error={row['goal_error_m']:.3f}m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
