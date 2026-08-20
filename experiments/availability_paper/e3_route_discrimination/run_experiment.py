#!/usr/bin/env python3
"""E3 — does the availability field change the route, and does the choice transfer?

Two questions, one candidate set:

1. **Discrimination.** Planning with an availability field instead of ignoring it
   changes the selected route by how much detour, and buys how much predicted
   terminal uncertainty?
2. **Decision regret.** A route chosen using a deployable field is executed in a
   warehouse whose real occlusion is whatever it is. Scoring every source's chosen
   route under one common reference field measures what the estimator error
   actually costs at the only place it matters — the decision.

Belief forward model: position-only, per-camera Bernoulli hit/miss mixture, in the
same sequential form the runtime uses. No ground-truth pose is read. The empirical
hit rate along each route is a model-free cross-check taken from the real detector
outcomes near the path.

Run:
    python3 experiments/availability_paper/e3_route_discrimination/run_experiment.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402

RESULTS = C.OUT_ROOT / "e3_route_discrimination"
PG_SUMMARY = C.REPO / "logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json"
R_COND_ARM = "raw IPM (floor, no correction)"

#: Reliability weights that generate the candidate route set. One route per weight
#: per task; every source then chooses from the SAME candidate set, so a source can
#: only be penalised for choosing badly, never for having a different search.
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

DT_S = 0.4
SPEED_MPS = 0.6
PROCESS_NOISE_XY = 0.012
INITIAL_SIGMA_XY_M = 0.05
#: Weight on integrated positional uncertainty in the route objective, in the same
#: units as experiment 11 so the two studies' objectives are comparable.
UNCERTAINTY_WEIGHT = 50.0
EMPIRICAL_RADIUS_M = 0.5

#: The same four declared warehouse tasks as experiment 11, so the two studies'
#: route numbers are comparable. Frozen before any source was scored.
DEFAULT_TASKS = (
    "mc_blind_L",
    "mc_m2_w2e_traverse",
    "full_traverse_handover",
    "route_tall_shadow_west",
)

#: Camera subsets, in decreasing network density.
#:
#: WHY THIS SWEEP EXISTS. With four cameras and noisy-OR fusion, the probability
#: that at least one camera sees the robot is high almost everywhere, so there is
#: very little for an availability-aware planner to avoid — which is exactly what
#: the four-camera numbers show. Whether availability modelling matters at all is
#: therefore a question about network density, not a single number. A sparse
#: network is also the more realistic deployment.
#:
#: The two-camera pairs are chosen to separate geometry from count: A and B are on
#: opposite walls (south-west and north-west), A and C share the south wall. The
#: repository's two-camera work found that opposing views pay and same-side views
#: do not, so a density sweep that ignored placement would confound the two.
SUBSETS: dict[str, tuple[str, ...]] = {
    "four": ("camera_A", "camera_B", "camera_C", "camera_D"),
    "three": ("camera_A", "camera_B", "camera_C"),
    "two_opposite": ("camera_A", "camera_B"),
    "two_same_side": ("camera_A", "camera_C"),
    "one": ("camera_A",),
}


def load_r_cond() -> dict[str, float]:
    """Per-camera isotropic conditional sigma, from the zero-parameter IPM study."""

    if not PG_SUMMARY.is_file():
        raise RuntimeError(f"R_cond source missing: {PG_SUMMARY}")
    payload = json.loads(PG_SUMMARY.read_text())
    arm = payload["arms"][R_COND_ARM]
    out: dict[str, float] = {}
    for camera in C.CAMERAS:
        stats = arm["per_camera"][camera]
        radial = float(stats["radial_sd_m"])
        lateral = float(stats["lateral_sd_m"])
        out[camera] = math.sqrt(0.5 * (radial**2 + lateral**2))
    return out


#: The global planner's keep-in clearance contract. Route search must respect it or
#: it produces routes the controller will refuse.
#:
#: WHY THIS IS HERE. Dijkstra over a binary driveable mask has no clearance term, so
#: it hugs boundaries: the first version of this experiment selected routes whose
#: minimum keep-in clearance was 0.062 m on mc_blind_L and 0.005 m on
#: mc_m2_w2e_traverse, against a 0.25 m contract. Those are not routes the robot can
#: drive, and E4 seeds its planner from these polylines. Eroding the mask by the
#: required clearance makes the constraint explicit instead of hoping the optimizer
#: fixes it later.
REQUIRED_CLEARANCE_M = 0.25


def clearance_grid(xs: np.ndarray, ys: np.ndarray, driveable_prisms) -> np.ndarray:
    """Per-cell distance inside the driveable union, in metres.

    Sign convention: ``signed_distance_to_union_xy(..., keep_in=True)`` is <= 0
    INSIDE the union, so clearance is its negation. Getting this backwards makes
    every cell in the warehouse look like a violation.
    """

    sys.path.insert(0, str(C.REPO / "src/unav_common"))
    from unav_common.occlusion_geometry import signed_distance_to_union_xy

    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    signed = signed_distance_to_union_xy(tuple(driveable_prisms), pts, keep_in=True)
    return (-np.asarray(signed, dtype=float)).reshape(len(ys), len(xs))


def fused_field(per_camera: dict[str, np.ndarray]) -> np.ndarray:
    """Noisy-OR over cameras: probability that at least one camera delivers."""

    miss = np.ones_like(next(iter(per_camera.values())), dtype=float)
    for field in per_camera.values():
        miss = miss * (1.0 - np.clip(field, 0.0, 1.0))
    return 1.0 - miss


def dense_route(
    field: np.ndarray,
    driveable: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    lam: float,
) -> np.ndarray:
    """Dijkstra with a reliability penalty; returns the DENSE cell path.

    ``render_all.dijkstra_route`` simplifies the path to corner points, which is
    right for drawing and wrong for integrating a belief along it.
    """

    start = base_nearest(driveable, xs, ys, start_xy)
    goal = base_nearest(driveable, xs, ys, goal_xy)
    ny, nx = field.shape
    dist = np.full((ny, nx), np.inf)
    dist[start] = 0.0
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    while queue:
        cost, current = heapq.heappop(queue)
        if current == goal:
            break
        if cost != dist[current]:
            continue
        i, j = current
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx) or not driveable[ni, nj]:
                continue
            if di != 0 and dj != 0 and not (driveable[i, nj] and driveable[ni, j]):
                continue  # no diagonal corner cutting
            step = math.hypot(di * dy, dj * dx)
            reliability = float(np.clip(field[ni, nj], 0.0, 1.0))
            new_cost = cost + step * (1.0 + lam * (1.0 - reliability) ** 2)
            if new_cost < dist[ni, nj]:
                dist[ni, nj] = new_cost
                previous[(ni, nj)] = current
                heapq.heappush(queue, (new_cost, (ni, nj)))
    if goal != start and goal not in previous:
        raise RuntimeError(f"no route from {start_xy} to {goal_xy} at lambda={lam}")
    cells = [goal]
    while cells[-1] != start:
        cells.append(previous[cells[-1]])
    cells.reverse()
    return np.asarray([[xs[j], ys[i]] for i, j in cells], dtype=float)


def base_nearest(mask, xs, ys, xy):
    j0 = int(np.argmin(np.abs(xs - xy[0])))
    i0 = int(np.argmin(np.abs(ys - xy[1])))
    if mask[i0, j0]:
        return i0, j0
    ii, jj = np.where(mask)
    k = int(np.argmin((ii - i0) ** 2 + (jj - j0) ** 2))
    return int(ii[k]), int(jj[k])


def path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def simplify(
    path: np.ndarray,
    tolerance_m: float = 0.12,
    *,
    clearance: np.ndarray | None = None,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    required_m: float = REQUIRED_CLEARANCE_M,
) -> list[list[float]]:
    """Reduce a dense cell path to waypoints, for seeding the online planner.

    Douglas-Peucker rather than the direction-change rule in ``render_all``: grid
    Dijkstra produces staircases on diagonals, and a direction-change rule keeps
    every single stair tread.

    CLEARANCE-AWARE, and it has to be. Simplification cuts corners, so a polyline can
    leave the cells that satisfied the keep-in contract even when the dense route
    respected it: mc_m2_w2e_traverse measured 0.300 m dense and 0.235 m simplified,
    against a 0.25 m requirement. Since these waypoints are what E4 seeds its planner
    with, the tolerance is tightened until the simplified line itself clears the
    contract.
    """

    if clearance is not None and xs is not None and ys is not None:
        tol = float(tolerance_m)
        for _ in range(8):
            candidate = _douglas_peucker(path, tol)
            walked = _resample_polyline(np.asarray(candidate, dtype=float), 0.04)
            if walked.shape[0] == 0:
                break
            worst = float(np.min(C.sample_field_at(clearance, xs, ys, walked)))
            if worst >= required_m:
                return candidate
            tol *= 0.5
        # Fall through to the densest attempt rather than returning a route that
        # violates the contract silently.
        return _douglas_peucker(path, tol)
    return _douglas_peucker(path, tolerance_m)


def _resample_polyline(pts: np.ndarray, step_m: float) -> np.ndarray:
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, total, max(2, int(round(total / step_m))))
    return np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])


def _douglas_peucker(path: np.ndarray, tolerance_m: float) -> list[list[float]]:

    pts = np.asarray(path, dtype=float)
    if len(pts) <= 2:
        return [[float(p[0]), float(p[1])] for p in pts]

    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        a, b = pts[lo], pts[hi]
        ab = b - a
        norm = float(np.linalg.norm(ab))
        seg = pts[lo + 1 : hi]
        if norm < 1e-9:
            dist = np.linalg.norm(seg - a, axis=1)
        else:
            dist = np.abs(np.cross(ab, seg - a)) / norm
        k = int(np.argmax(dist))
        if float(dist[k]) > tolerance_m:
            idx = lo + 1 + k
            keep[idx] = True
            stack.extend([(lo, idx), (idx, hi)])
    return [[float(p[0]), float(p[1])] for p in pts[keep]]


def propagate(
    path: np.ndarray,
    per_camera: dict[str, np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    r_cond: dict[str, float],
) -> tuple[float, float]:
    """Integrate the expected position covariance along a route.

    Per step: predict with isotropic process noise, then apply each camera's
    Bernoulli hit/miss mixture in sequence,

        P <- p_c * (P^-1 + R_c^-1)^-1 + (1 - p_c) * P,

    which is the position-only form of the runtime's ``hit_miss_posterior_ca``:
    a measurement that never arrives buys no information.

    Returns ``(integral of trace(P) over arclength, terminal sigma)``.
    """

    step_len = float(SPEED_MPS * DT_S)
    total = path_length(path)
    n_steps = max(2, int(round(total / step_len)))
    s = np.linspace(0.0, total, n_steps)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    pts = np.column_stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])

    p_along = {c: C.sample_field_at(per_camera[c], xs, ys, pts) for c in per_camera}

    var = float(INITIAL_SIGMA_XY_M**2)
    q = float(PROCESS_NOISE_XY**2)
    integral = 0.0
    ds = total / max(n_steps - 1, 1)
    for k in range(n_steps):
        if k > 0:
            var = var + q
        for camera, field_along in p_along.items():
            p = float(np.clip(field_along[k], 0.0, 1.0))
            r = float(r_cond[camera] ** 2)
            hit = 1.0 / (1.0 / var + 1.0 / r)
            var = p * hit + (1.0 - p) * var
        integral += 2.0 * var * ds  # trace of an isotropic 2-D covariance
    return float(integral), float(math.sqrt(var))


def objective(length_m: float, integral: float) -> float:
    return float(length_m + UNCERTAINTY_WEIGHT * integral)


#: A pose is "unobserved" when the fused probability of at least one usable
#: detection falls below this. Chosen as the value below which the belief is
#: effectively running on odometry alone.
LOW_P_THRESHOLD = 0.2


def exposure_metrics(
    path: np.ndarray, fused_reference: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> dict[str, float]:
    """The route-archetype certification metrics from ``06_world_camera_design`` §5.

    Terminal sigma is a poor endpoint here: the belief re-converges once the robot
    leaves a blackspot, so a route can cross nine metres of unobserved aisle and
    still arrive sharp. What separates routes is time spent unobserved, so this
    returns exposure, the longest continuous unobserved run, and the unobserved
    fraction of the path.
    """

    total = path_length(path)
    step_len = float(SPEED_MPS * DT_S)
    n_steps = max(2, int(round(total / step_len)))
    s = np.linspace(0.0, total, n_steps)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    pts = np.column_stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])
    p = C.sample_field_at(fused_reference, xs, ys, pts)
    ds = total / max(n_steps - 1, 1)

    low = p < LOW_P_THRESHOLD
    longest = 0
    run = 0
    for flag in low:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    return {
        "exposure_m": float(np.sum(1.0 - p) * ds),
        "unobserved_fraction": float(np.mean(low)),
        "longest_unobserved_run_m": float(longest * ds),
        "longest_unobserved_run_s": float(longest * DT_S),
    }


def empirical_hit_rate(
    path: np.ndarray, apparatus: C.Apparatus, radius: float = EMPIRICAL_RADIUS_M
) -> tuple[float, int]:
    """Model-free: fraction of real detector events near the route that were hits.

    'Near' is any sampled spawn pose within ``radius`` of the polyline, pooled over
    cameras and counted as at-least-one-camera-hit at that pose.
    """

    xy = apparatus.events[C.CAMERAS[0]]["xy"]
    any_hit = np.zeros(xy.shape[0], dtype=bool)
    for camera in C.CAMERAS:
        any_hit |= apparatus.events[camera]["hit"] > 0.5

    seg_a = path[:-1]
    seg_b = path[1:]
    seg = seg_b - seg_a
    seg_len2 = np.maximum(np.sum(seg**2, axis=1), 1e-12)
    near = np.zeros(xy.shape[0], dtype=bool)
    for k in range(xy.shape[0]):
        rel = xy[k] - seg_a
        t = np.clip(np.sum(rel * seg, axis=1) / seg_len2, 0.0, 1.0)
        proj = seg_a + t[:, None] * seg
        if float(np.min(np.linalg.norm(xy[k] - proj, axis=1))) <= radius:
            near[k] = True
    if not near.any():
        return float("nan"), 0
    return float(np.mean(any_hit[near])), int(near.sum())


ROUTE_COLUMNS = (
    "task", "cameras", "source", "label", "needs_surveyed_model", "chosen_lambda",
    "length_m", "detour_vs_shortest_m", "trace_integral", "terminal_sigma_m",
    "objective_own_field", "objective_reference_field", "regret_vs_reference",
    "regret_normalised", "candidate_objective_range", "mean_p_use_own",
    "mean_p_use_reference", "min_p_use_reference", "exposure_m",
    "unobserved_fraction", "longest_unobserved_run_m", "longest_unobserved_run_s",
    "empirical_hit_rate", "empirical_events",
)
CANDIDATE_COLUMNS = (
    "task", "cameras", "generating_source", "lambda", "length_m", "terminal_sigma_m",
    "mean_p_use_reference", "objective_reference_field",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument("--reference", default="cad_reference")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    args = parser.parse_args()
    out = Path(args.out)

    apparatus = C.build_apparatus()
    r_cond = load_r_cond()
    xs, ys = apparatus.xs, apparatus.ys

    # Route search runs on the mask ERODED by the keep-in contract, so every route
    # this experiment reports is one the controller would accept. The un-eroded mask
    # is kept for drawing and for reporting how much floor the contract removes.
    clearance = clearance_grid(xs, ys, apparatus.ctx["driveable_prisms"])
    driveable_raw = apparatus.driveable
    driveable = driveable_raw & (clearance >= REQUIRED_CLEARANCE_M)
    print(
        f"keep-in contract {REQUIRED_CLEARANCE_M} m removes "
        f"{int(driveable_raw.sum() - driveable.sum())} of {int(driveable_raw.sum())} "
        f"driveable cells ({100.0 * (1 - driveable.sum() / max(driveable_raw.sum(), 1)):.1f}%)",
        flush=True,
    )
    if not driveable.any():
        raise RuntimeError("the clearance contract removed every driveable cell")

    sources = [s for s in C.SOURCES if s.key != "constant"]
    reference_key = args.reference

    task_specs: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for name in args.tasks:
        if name not in apparatus.tasks:
            raise RuntimeError(
                f"Task '{name}' not in the world task table; available: {sorted(apparatus.tasks)}"
            )
        spec = apparatus.tasks[name]
        start, goal = spec["start"], spec["goal"]
        task_specs[name] = (
            (float(start["x"]), float(start["y"])),
            (float(goal["x"]), float(goal["y"])),
        )

    route_rows: list[dict] = []
    candidate_rows: list[dict] = []
    #: Selected polylines, keyed camera-subset -> task -> source. E4 seeds the
    #: online global EFE multistart with these, so the planner still solves online
    #: but is not asked to discover a 15 m route from a cold start (the pilot's
    #: failure mode).
    selected_routes: dict[str, dict[str, dict[str, list[list[float]]]]] = {}

    for subset_name, subset_cameras in SUBSETS.items():
        fused = {
            s.key: fused_field({c: apparatus.fields[s.key][c] for c in subset_cameras})
            for s in sources
        }
        reference_per_camera = {c: apparatus.fields[reference_key][c] for c in subset_cameras}
        r_cond_subset = {c: r_cond[c] for c in subset_cameras}

        for task, (start_xy, goal_xy) in sorted(task_specs.items()):
            # One shared candidate set: every source's field generates routes, and
            # every source then chooses from the union.
            candidates: dict[tuple, np.ndarray] = {}
            for source in sources:
                for lam in LAMBDAS:
                    path = dense_route(fused[source.key], driveable, xs, ys, start_xy, goal_xy, lam)
                    candidates[(source.key, lam)] = path

            # Deduplicate identical polylines so ties are not counted many times.
            unique: dict[bytes, tuple[tuple, np.ndarray]] = {}
            for key, path in candidates.items():
                unique.setdefault(np.round(path, 4).tobytes(), (key, path))
            candidate_list = list(unique.values())

            scored_reference: dict[bytes, tuple[float, float, float, float]] = {}
            for key, path in candidate_list:
                integral, terminal = propagate(path, reference_per_camera, xs, ys, r_cond_subset)
                length = path_length(path)
                obj = objective(length, integral)
                p_ref = C.sample_field_at(fused[reference_key], xs, ys, path)
                scored_reference[np.round(path, 4).tobytes()] = (obj, terminal, float(np.mean(p_ref)), float(np.min(p_ref)))
                candidate_rows.append(
                    {
                        "task": task,
                        "cameras": subset_name,
                        "generating_source": key[0],
                        "lambda": key[1],
                        "length_m": f"{length:.4f}",
                        "terminal_sigma_m": f"{terminal:.6f}",
                        "mean_p_use_reference": f"{np.mean(p_ref):.6f}",
                        "objective_reference_field": f"{obj:.6f}",
                    }
                )

            best_reference_objective = min(v[0] for v in scored_reference.values())
            worst_reference_objective = max(v[0] for v in scored_reference.values())
            objective_range = worst_reference_objective - best_reference_objective

            # The availability-blind baseline is the shortest driveable path, computed
            # explicitly rather than inferred from whichever arm happened to pick
            # lambda = 0. This is the C1 arm of the headline closed-loop comparison.
            blind_path = dense_route(
                np.ones_like(driveable, dtype=float), driveable, xs, ys, start_xy, goal_xy, 0.0
            )
            blind_length = path_length(blind_path)
            blind_integral, blind_terminal = propagate(blind_path, reference_per_camera, xs, ys, r_cond_subset)
            blind_objective = objective(blind_length, blind_integral)
            selected_routes.setdefault(subset_name, {}).setdefault(task, {})["availability_blind"] = simplify(
            blind_path, clearance=clearance, xs=xs, ys=ys
        )
            blind_p_ref = C.sample_field_at(fused[reference_key], xs, ys, blind_path)
            blind_rate, blind_events = empirical_hit_rate(blind_path, apparatus)
            route_rows.append(
                {
                    "task": task,
                    "cameras": subset_name,
                    "source": "availability_blind",
                    "label": "Availability-blind shortest path",
                    "needs_surveyed_model": "false",
                    "chosen_lambda": 0.0,
                    "length_m": f"{blind_length:.4f}",
                    "detour_vs_shortest_m": "0.0000",
                    "trace_integral": f"{blind_integral:.6f}",
                    "terminal_sigma_m": f"{blind_terminal:.6f}",
                    "objective_own_field": f"{blind_length:.6f}",
                    "objective_reference_field": f"{blind_objective:.6f}",
                    "regret_vs_reference": f"{blind_objective - best_reference_objective:.6f}",
                    "regret_normalised": (
                        f"{(blind_objective - best_reference_objective) / objective_range:.6f}"
                        if objective_range > 0
                        else ""
                    ),
                    "candidate_objective_range": f"{objective_range:.6f}",
                    "mean_p_use_own": "",
                    "mean_p_use_reference": f"{np.mean(blind_p_ref):.6f}",
                    "min_p_use_reference": f"{np.min(blind_p_ref):.6f}",
                    **{
                        k: f"{v:.6f}"
                        for k, v in exposure_metrics(blind_path, fused[reference_key], xs, ys).items()
                    },
                    "empirical_hit_rate": f"{blind_rate:.6f}" if np.isfinite(blind_rate) else "",
                    "empirical_events": blind_events,
                }
            )

            for source in sources:
                best = None
                for key, path in candidate_list:
                    integral = propagate(path, {c: apparatus.fields[source.key][c] for c in subset_cameras}, xs, ys, r_cond_subset)[0]
                    obj_own = objective(path_length(path), integral)
                    if best is None or obj_own < best[0]:
                        best = (obj_own, key, path)
                assert best is not None
                obj_own, key, path = best
                digest = np.round(path, 4).tobytes()
                obj_ref, terminal_ref, mean_p_ref, min_p_ref = scored_reference[digest]
                integral_ref, _ = propagate(path, reference_per_camera, xs, ys, r_cond_subset)
                p_own = C.sample_field_at(fused[source.key], xs, ys, path)
                rate, n_events = empirical_hit_rate(path, apparatus)
                selected_routes.setdefault(subset_name, {}).setdefault(task, {})[source.key] = simplify(
                path, clearance=clearance, xs=xs, ys=ys
            )

                route_rows.append(
                    {
                        "task": task,
                        "cameras": subset_name,
                        "source": source.key,
                        "label": source.label,
                        "needs_surveyed_model": str(source.needs_surveyed_model).lower(),
                        "chosen_lambda": key[1],
                        "length_m": f"{path_length(path):.4f}",
                        "detour_vs_shortest_m": f"{path_length(path) - blind_length:.4f}",
                        "trace_integral": f"{integral_ref:.6f}",
                        "terminal_sigma_m": f"{terminal_ref:.6f}",
                        "objective_own_field": f"{obj_own:.6f}",
                        "objective_reference_field": f"{obj_ref:.6f}",
                        "regret_vs_reference": f"{obj_ref - best_reference_objective:.6f}",
                        "regret_normalised": (
                            f"{(obj_ref - best_reference_objective) / objective_range:.6f}"
                            if objective_range > 0
                            else ""
                        ),
                        "candidate_objective_range": f"{objective_range:.6f}",
                        "mean_p_use_own": f"{np.mean(p_own):.6f}",
                        "mean_p_use_reference": f"{mean_p_ref:.6f}",
                        "min_p_use_reference": f"{min_p_ref:.6f}",
                        **{
                            k: f"{v:.6f}"
                            for k, v in exposure_metrics(path, fused[reference_key], xs, ys).items()
                        },
                        "empirical_hit_rate": f"{rate:.6f}" if np.isfinite(rate) else "",
                        "empirical_events": n_events,
                    }
                )
            print(f"[{subset_name}] {task}: {len(candidate_list)} unique candidates", flush=True)

    untagged = [r for r in route_rows if not r.get("cameras")]
    if untagged:
        raise RuntimeError(
            f"{len(untagged)} route rows have no camera-subset tag; the density sweep "
            "would be unlabelled and its rows unattributable. Refusing to write."
        )
    C.write_csv(out / "e3_routes.csv", ROUTE_COLUMNS, route_rows)
    C.write_csv(out / "e3_candidates.csv", CANDIDATE_COLUMNS, candidate_rows)
    C.write_json(
        out / "e3_selected_routes.json",
        {
            "note": (
                "Waypoints for seeding the E4 online global EFE multistart. The planner "
                "still solves online; these only replace a cold-start route search."
            ),
            "simplify_tolerance_m": 0.12,
            "routes": selected_routes,
        },
    )
    C.write_json(
        out / "manifest.json",
        {
            "experiment_id": "EXP-AVAIL-ROUTE",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reference_field": reference_key,
            "reference_caveat": (
                "The reference is a CAD raycast, which requires a surveyed 3-D model. "
                "It is an evaluation reference, not a deployable method and not truth."
            ),
            "lambdas": list(LAMBDAS),
            "required_clearance_m": REQUIRED_CLEARANCE_M,
            "clearance_note": (
                "Route search runs on the driveable mask eroded by the global keep-in "
                "contract. Without this, grid Dijkstra hugs boundaries and selected "
                "routes with 0.005-0.062 m clearance, which the local controller "
                "rejects at step zero."
            ),
            "belief_model": {
                "form": "position-only sequential Bernoulli hit/miss mixture",
                "dt_s": DT_S,
                "speed_mps": SPEED_MPS,
                "process_noise_xy": PROCESS_NOISE_XY,
                "initial_sigma_xy_m": INITIAL_SIGMA_XY_M,
                "uncertainty_weight": UNCERTAINTY_WEIGHT,
            },
            "r_cond_isotropic_sigma_m": r_cond,
            "r_cond_source": str(PG_SUMMARY.relative_to(C.REPO)),
            "tasks": sorted(task_specs),
            "camera_subsets": {k: list(v) for k, v in SUBSETS.items()},
            "why_subsets": (
                "With four cameras and noisy-OR fusion there is little for an "
                "availability-aware planner to avoid, so whether availability modelling "
                "matters is a question about network density. The two-camera pairs "
                "separate placement from count: A/B oppose, A/C share the south wall."
            ),
            "evaluation_only_inputs": [],
            "inputs_sha256": C.input_manifest(extra=[PG_SUMMARY]),
        },
    )

    print(f"\nwrote {out}/e3_routes.csv ({len(route_rows)} rows)")
    print(f"wrote {out}/e3_candidates.csv ({len(candidate_rows)} rows)")

    def mean_over(rows: list[dict], name: str) -> float:
        vals = np.asarray([float(r[name]) for r in rows if r[name] != ""], dtype=float)
        return float(np.mean(vals)) if vals.size else float("nan")

    print("\nHow much does modelling availability help, and does that depend on how many")
    print("cameras there are? Mean over the four declared tasks, per camera subset.")
    print("'blind gap' and 'aware gap' are the longest continuous stretch in seconds with")
    print("no camera likely to see the robot, planning without and with the field.\n")
    header = (
        f"{'cameras':<15}{'blind gap':>11}{'CAD gap':>10}{'depth gap':>11}"
        f"{'CAD saves':>11}{'depth saves':>13}{'detour m':>10}"
    )
    print(header)
    print("-" * len(header))
    for subset_name in SUBSETS:
        blind = [r for r in route_rows if r["cameras"] == subset_name and r["source"] == "availability_blind"]
        cad = [r for r in route_rows if r["cameras"] == subset_name and r["source"] == "cad_reference"]
        depth = [r for r in route_rows if r["cameras"] == subset_name and r["source"] == "mono_depth"]
        if not (blind and cad and depth):
            continue
        g_blind = mean_over(blind, "longest_unobserved_run_s")
        g_cad = mean_over(cad, "longest_unobserved_run_s")
        g_depth = mean_over(depth, "longest_unobserved_run_s")
        print(
            f"{subset_name:<15}{g_blind:>11.2f}{g_cad:>10.2f}{g_depth:>11.2f}"
            f"{g_blind - g_cad:>11.2f}{g_blind - g_depth:>13.2f}"
            f"{mean_over(cad, 'detour_vs_shortest_m'):>10.3f}"
        )

    print("\nPer-arm detail on the full four-camera network:\n")
    header = (
        f"{'arm':<44}{'regret':>8}{'detour m':>10}{'exposure m':>12}"
        f"{'worst gap s':>13}{'unobs frac':>12}{'emp hit rate':>14}"
    )
    print(header)
    print("-" * len(header))
    for key in ["availability_blind"] + [s.key for s in sources]:
        rows = [r for r in route_rows if r["source"] == key and r["cameras"] == "four"]
        if not rows:
            continue

        def mean(name: str, _rows=rows) -> float:
            return mean_over(_rows, name)

        print(
            f"{rows[0]['label']:<44}{mean('regret_vs_reference'):>8.4f}"
            f"{mean('detour_vs_shortest_m'):>10.3f}{mean('exposure_m'):>12.3f}"
            f"{mean('longest_unobserved_run_s'):>13.2f}{mean('unobserved_fraction'):>12.3f}"
            f"{mean('empirical_hit_rate'):>14.3f}"
        )

    print(
        "\nTask-level detail is in e3_routes.csv. Terminal sigma is deliberately NOT the\n"
        "headline: the belief re-converges after a blackspot, so it hides the gap."
    )


if __name__ == "__main__":
    main()
