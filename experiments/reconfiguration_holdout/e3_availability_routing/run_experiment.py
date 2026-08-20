#!/usr/bin/env python3
"""E3: does a better availability field change the route the robot drives?

E1 asks which estimator predicts detector outcomes best.  That is a prediction
question, and this experiment asks the planning question separately.

RETRACTED 2026-08-20 -- this docstring previously cited the e4 closed-loop campaign
(ambiguity 292.7 -> 337.6 against risk 511,684, every arm coordinate-identical) as
evidence that a field cannot move a route under the deployed objective.  That
campaign is uninformative: all arms were handed the SAME two candidate routes 0.19 m
apart, its planner ran with heading pinned at the non-informative pi^2 prior, and the
quoted ratio omits the obstacle term, where the availability-blind arm pays 0 and the
availability-aware arms pay 1,017,055.  Do not cite it.  The budgeted rule below
stands on its own evidence and needs no such premise.  The visibility term is frozen
method and is deliberately NOT reweighted here.

This experiment asks the planning question with a decision rule that can act on
availability: among the routes whose length is within a declared budget of the
shortest one, drive the one with the least expected blind distance.  The budget is
the parameter; the field is the input.  Nothing about the EFE objective is touched.

    route*(eps) = argmin_route  sum_cells  d(cell) * (1 - p_any(cell))
                  subject to    length(route) <= (1 + eps) * length(shortest)

The planner reads only its own estimator's field.  Routes are then scored against
REAL DETECTOR OUTCOMES from the capture -- how much of the driven distance actually
had no camera on the robot -- which is evaluation-only and never a planner input.

The claim this tests: a frozen learned field routes the robot through ground that has
since gone dark, while a field recomputed from the camera's own image routes around
it, and the gap only opens after the warehouse changes.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_exact(name: str, path: Path):
    expected = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != expected:
            raise ImportError(f"{name} resolves to an unexpected module")
        return existing
    spec = importlib.util.spec_from_file_location(name, expected)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {expected}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


C = _load_exact("_reconfiguration_holdout_common", HERE.parent / "common.py")
CL = _load_exact("_reconfiguration_holdout_choose_layout", HERE.parent / "choose_layout.py")
IHF = _load_exact("_reconfiguration_holdout_e3_fields", HERE / "independent_heading_fields.py")
STATS = _load_exact("_reconfiguration_holdout_e3_stats", HERE / "summarize_cells.py")
ora = C.ora
# choose_layout predates exact-path loading; make its global oracle explicit too.
CL.ora = ora

OUT = C.OUT_ROOT / "e3_availability_routing"

#: Length budgets the decision rule is evaluated at, as a fraction over shortest.
BUDGETS = (0.05, 0.10, 0.20, 0.35, 0.50)

#: Lagrange weights swept to trace the length / blind-distance frontier.  lambda = 0
#: is the pure shortest path, so the frontier always contains the unconstrained route.
LAMBDAS = np.concatenate([[0.0], np.logspace(-1.0, 2.5, 16)])

#: The planning arms.  "shortest" is the no-availability control: it is what the
#: robot drives today.  cad_env is a reference, not deployable: it needs a re-survey.
ROUTING_ARMS = (
    ("shortest", "Shortest route (no availability model)", False),
    ("gp", "Frozen field learned from past detections", False),
    ("mono_depth", "Field recomputed from the camera's own image", False),
    ("hybrid", "Recomputed prior + frozen learned residual", False),
    ("cad_l0", "Surveyed model of the old warehouse", True),
    ("cad_env", "Surveyed model, re-surveyed after the change", True),
)

#: Start/goal pairs taken verbatim from the campaign's declared tasks in the
#: four-camera world, so the routes are ones the robot is actually asked to drive.
WAYPOINTS = (
    ("dock_sw", (-7.77, -7.5)), ("dock_s", (0.0, -7.6)), ("dock_se", (7.77, -7.5)),
    ("dock_nw", (-7.77, 7.5)), ("dock_n", (0.0, 7.6)), ("dock_ne", (7.77, 7.5)),
    ("aisle_w", (-10.0, -1.0)), ("aisle_e", (10.0, 1.0)),
    ("blind_L", (-3.57, 0.0)), ("blind_R", (3.57, 0.0)),
    ("mid_nw", (-5.68, 5.5)), ("mid_se", (5.68, -5.5)),
    ("mid_ne", (5.68, 5.5)), ("mid_sw", (-5.68, -5.5)),
    ("aisle_wn", (-10.0, 6.0)), ("aisle_es", (10.0, -6.0)),
)

#: Camera subsets, reused verbatim from the availability study's pre-registered set so the
#: network-density axis means the same thing here as it does there.  Every subset is
#: reported; none is selected after the fact.
SUBSETS = ("4", "3", "2_opposite", "2_same_wall", "1")

#: Every unordered waypoint pair separated by at least this much, so each task admits a
#: genuine route choice rather than a single corridor.  Declared before any route was run.
MIN_SEPARATION_M = 8.0


def tasks() -> list[tuple[str, tuple, tuple]]:
    out = []
    for i in range(len(WAYPOINTS)):
        for j in range(i + 1, len(WAYPOINTS)):
            (na, a), (nb, b) = WAYPOINTS[i], WAYPOINTS[j]
            if float(np.hypot(a[0] - b[0], a[1] - b[1])) >= MIN_SEPARATION_M:
                out.append((f"{na}__{nb}", a, b))
    return out


# --------------------------------------------------------------------------
# The lane network the robot may drive on
# --------------------------------------------------------------------------

def driveable() -> np.ndarray:
    """Cells the robot may occupy: declared traversable lanes minus rack footprints.

    Identical in both layouts by construction -- the restock adds nothing below
    2.09 m -- which is what lets a route difference be attributed to the observation
    model rather than to obstacle avoidance.
    """
    grid = C.floor_grid()
    mask = CL.driveable_mask(grid, CL.lanes())
    base = ora.OracleScene.from_world(
        C.WORLDS / f"{C.ENV_BY_KEY['L0'].world_name}.world.sdf", list(C.CAMERAS))
    gx, gy = np.meshgrid(grid.x_centres, grid.y_centres)
    for p in base.static_prisms:
        mask &= ~((gx >= p.xmin) & (gx <= p.xmax) & (gy >= p.ymin) & (gy <= p.ymax))
    return mask


# --------------------------------------------------------------------------
# Availability fields, calibrated and fused exactly as E1 does
# --------------------------------------------------------------------------

def fused_fields(threshold: float, environments: tuple[str, ...] = ("L0", "L1")) \
        -> tuple[dict, dict, dict]:
    """p_any on the working grid for every arm and environment.

    The model and each arm's two-parameter link are fitted only on L0's four
    diagonal headings.  L0's cardinal headings are reserved as nominal route truth
    and match the four headings available in L1.  The cameras are then combined by
    the same noisy-OR E1 uses.  A deployed system does not get to recalibrate against
    evaluation outcomes, so (a, b) is fitted once on the diagonal L0 partition and
    frozen.
    """
    per_env, l0_train, learned_link_oof, provenance = IHF.build_fields(
        threshold, environments
    )

    link = {}
    for arm in {a for a, _l, _s in ROUTING_ARMS} - {"shortest"}:
        for cam in C.CAMERAS:
            if arm in learned_link_oof:
                # The deployment field uses all diagonal outcomes, but its link is
                # estimated from pooled predictions for which the corresponding
                # spatial block was absent from the latent-field fit.
                raw = learned_link_oof[arm][cam]["score"]
                hit = learned_link_oof[arm][cam]["hit"]
            else:
                # These fields do not consume detector outcomes, so sampling them
                # on the diagonal calibration partition is already out of model fit.
                raw = C.sample_at(per_env["L0"][arm][cam], l0_train[cam]["xy"])
                hit = l0_train[cam]["hit"]
            link[(arm, cam)] = C.fit_link(raw, hit)

    out = {}
    for env in environments:
        for arm, _label, _sv in ROUTING_ARMS:
            if arm == "shortest":
                continue
            for sub in SUBSETS:
                cams = C.CAMERA_SUBSETS[sub]
                p_none = np.ones_like(per_env[env][arm][C.CAMERAS[0]])
                for cam in cams:
                    a, b = link[(arm, cam)]
                    p_none = p_none * (1.0 - C.apply_link(per_env[env][arm][cam], a, b))
                out[(env, arm, sub)] = np.clip(1.0 - p_none, 1e-6, 1 - 1e-6)
    return out, link, provenance


def measured_availability(threshold: float,
                          environments: tuple[str, ...] = ("L0", "L1")) \
        -> dict[str, np.ndarray]:
    """Ground truth: the fraction of headings at which SOME camera actually detected.

    Real detector outcomes from the cardinal-heading partition of each capture,
    pooled over the four headings at each captured position and nearest-neighbour
    transferred onto the working grid.  These L0 outcomes are disjoint from the
    diagonal headings used to fit and calibrate the fields; L1 uses the matching
    cardinal headings.  This is evaluation-only: no planner reads it.
    """
    xs, ys = C.working_grid()
    gx, gy = np.meshgrid(xs, ys)
    out = {}
    for env_key in environments:
        ev = IHF.load_evaluation_events(env_key, threshold)
        for sub in SUBSETS:
            keys, any_hit = {}, {}
            for cam in C.CAMERA_SUBSETS[sub]:
                e = ev[cam]
                for i in range(len(e["hit"])):
                    k = (round(e["xy"][i, 0], 3), round(e["xy"][i, 1], 3),
                         round(e["theta"][i], 3))
                    any_hit[k] = max(any_hit.get(k, 0.0), e["hit"][i])
            for (x, y, _t), h in any_hit.items():
                keys.setdefault((x, y), []).append(h)
            pts = np.array(list(keys.keys()))
            vals = np.array([float(np.mean(v)) for v in keys.values()])
            field = np.zeros_like(gx)
            for iy in range(gx.shape[0]):
                d2 = (pts[:, 0][None, :] - gx[iy][:, None]) ** 2 + \
                     (pts[:, 1][None, :] - gy[iy][:, None]) ** 2
                field[iy] = vals[np.argmin(d2, axis=1)]
            out[(env_key, sub)] = field
    return out


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def neighbours(iy: int, ix: int, ny: int, nx: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx:
                yield jy, jx, float(np.hypot(dy, dx))


def dijkstra(mask: np.ndarray, cost_mult: np.ndarray, start: tuple, goal: tuple,
             res: float) -> list[tuple[int, int]]:
    """Least-cost 8-connected path; edge cost is step length times the arrival cell."""
    ny, nx = mask.shape
    dist = np.full(mask.shape, np.inf)
    prev = {}
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        if node == goal:
            break
        for jy, jx, step in neighbours(node[0], node[1], ny, nx):
            if not mask[jy, jx]:
                continue
            nd = d + step * res * cost_mult[jy, jx]
            if nd < dist[jy, jx]:
                dist[jy, jx] = nd
                prev[(jy, jx)] = node
                heapq.heappush(pq, (nd, (jy, jx)))
    if not np.isfinite(dist[goal]):
        return []
    path, node = [goal], goal
    while node != start:
        node = prev[node]
        path.append(node)
    return path[::-1]


def route_metrics(path: list, p_true: np.ndarray, p_plan: np.ndarray, res: float) -> dict:
    """Length, and blind distance under the truth and under the planner's own field."""
    length = blind_true = blind_plan = 0.0
    for a, b in zip(path[:-1], path[1:]):
        step = float(np.hypot(b[0] - a[0], b[1] - a[1])) * res
        length += step
        blind_true += step * (1.0 - p_true[b])
        blind_plan += step * (1.0 - p_plan[b])
    return {"length_m": length, "blind_true_m": blind_true, "blind_plan_m": blind_plan}


def frontier(mask, p_plan, p_true, start, goal, res):
    """Every route the Lagrange sweep reaches, once. Budgets then select from this."""
    shortest = dijkstra(mask, np.ones_like(p_plan), start, goal, res)
    if not shortest:
        return None, None
    l_star = route_metrics(shortest, p_true, p_plan, res)["length_m"]
    cand = [(shortest, route_metrics(shortest, p_true, p_plan, res))]
    for lam in LAMBDAS[1:]:
        path = dijkstra(mask, 1.0 + lam * (1.0 - p_plan), start, goal, res)
        if path:
            cand.append((path, route_metrics(path, p_true, p_plan, res)))
    return cand, l_star


def select(cand, l_star, budget):
    """The decision rule: least predicted blind distance inside the length budget."""
    best, best_m = None, None
    for path, m in cand:
        if m["length_m"] <= (1.0 + budget) * l_star + 1e-9:
            if best_m is None or m["blind_plan_m"] < best_m["blind_plan_m"]:
                best, best_m = path, dict(m)
    best_m["length_shortest_m"] = l_star
    best_m["detour_frac"] = best_m["length_m"] / l_star - 1.0
    return best, best_m


def snap(mask: np.ndarray, xs, ys, xy) -> tuple[int, int]:
    gx, gy = np.meshgrid(xs, ys)
    d2 = (gx - xy[0]) ** 2 + (gy - xy[1]) ** 2
    d2[~mask] = np.inf
    return np.unravel_index(int(np.argmin(d2)), mask.shape)


_W = {}


def _init(mask, truth, p_hat, res):
    _W.update(mask=mask, truth=truth, p_hat=p_hat, res=res)


def _one(job):
    """One (env, subset, task) unit: each arm's frontier once, selected at every budget."""
    env, sub, task, s_xy, g_xy = job
    mask, truth, p_hat, res = _W["mask"], _W["truth"], _W["p_hat"], _W["res"]
    xs, ys = C.working_grid()
    s, g = snap(mask, xs, ys, s_xy), snap(mask, xs, ys, g_xy)
    p_true = truth[(env, sub)]
    out, geo = [], {}
    for arm, label, needs_survey in ROUTING_ARMS:
        if arm == "shortest":
            path = dijkstra(mask, np.ones_like(p_true), s, g, res)
            if not path:
                return [], {}
            m = route_metrics(path, p_true, p_true, res)
            m["length_shortest_m"] = m["length_m"]; m["detour_frac"] = 0.0
            cand = None
        else:
            cand, l_star = frontier(mask, p_hat[(env, arm, sub)], p_true, s, g, res)
            if cand is None:
                return [], {}
        for budget in BUDGETS:
            if cand is None:
                sel_path, sel_m = path, dict(m)
            else:
                sel_path, sel_m = select(cand, l_star, budget)
            out.append({"environment": env, "subset": sub, "task": task, "budget": budget,
                        "arm": arm, "label": label, "needs_survey": needs_survey, **sel_m})
            geo[f"{env}|{sub}|{task}|{budget}|{arm}"] = [
                [float(xs[ix]), float(ys[iy])] for iy, ix in sel_path]
    return out, geo


def _validate_environments(values: list[str]) -> tuple[str, ...]:
    environments = tuple(values)
    if len(environments) < 2 or environments[0] != "L0":
        raise ValueError("--environments must begin with L0 and include a changed environment")
    if len(set(environments)) != len(environments):
        raise ValueError("--environments contains duplicates")
    unknown = [key for key in environments if key not in C.ENV_BY_KEY]
    if unknown:
        raise ValueError(f"unknown environments: {unknown}")
    missing_capture = [
        key for key in environments
        if not (C.ENV_BY_KEY[key].capture / "perception_targets.csv").is_file()
    ]
    if missing_capture:
        raise ValueError(f"environments have no scored capture: {missing_capture}")
    missing_prior = [key for key in environments if not C.mono_depth_path(key).is_file()]
    if missing_prior:
        raise ValueError(f"environments have no monocular-depth field: {missing_prior}")
    return environments


def main(argv: list[str] | None = None) -> int:
    import multiprocessing as mp
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--environments",
        nargs="+",
        default=["L0", "L1"],
        help="L0 followed by one or more changed environments (default: L0 L1)",
    )
    args = parser.parse_args(argv)
    try:
        environments = _validate_environments(args.environments)
    except ValueError as exc:
        parser.error(str(exc))

    OUT.mkdir(parents=True, exist_ok=True)
    xs, ys = C.working_grid()
    res = float(xs[1] - xs[0])
    mask = driveable()
    TASKS = tasks()
    print(f"[e3] driveable {int(mask.sum())} cells, {len(TASKS)} tasks, "
          f"{len(SUBSETS)} subsets", flush=True)

    print(f"[e3] environments: {', '.join(environments)}", flush=True)
    p_hat, link, field_provenance = fused_fields(C.PRIMARY_THRESHOLD, environments)
    truth = measured_availability(C.PRIMARY_THRESHOLD, environments)
    for sub in SUBSETS:
        means = " ".join(
            f"{env}={truth[(env, sub)][mask].mean():.4f}" for env in environments
        )
        print(f"[e3] subset {sub}: measured p_any over driveable {means}", flush=True)

    jobs = [(env, sub, t, a, b) for env in environments for sub in SUBSETS
            for t, a, b in TASKS]
    rows, routes = [], {}
    with mp.Pool(min(10, mp.cpu_count() - 2), initializer=_init,
                 initargs=(mask, truth, p_hat, res)) as pool:
        for i, (out, geo) in enumerate(pool.imap_unordered(_one, jobs), 1):
            rows.extend(out); routes.update(geo)
            if i % 50 == 0:
                print(f"[e3] {i}/{len(jobs)} units", flush=True)

    import csv as _csv
    fields = ["environment", "subset", "task", "budget", "arm", "label", "needs_survey",
              "length_m", "length_shortest_m", "detour_frac", "blind_true_m", "blind_plan_m"]
    routes_path = OUT / "e3_routes.csv"
    with routes_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["environment"], r["subset"], r["task"],
                                             r["budget"], r["arm"])):
            w.writerow({k: r[k] for k in fields})
    (OUT / "e3_route_geometry.json").write_text(json.dumps(routes), encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps({
        "threshold": C.PRIMARY_THRESHOLD, "budgets": list(BUDGETS),
        "subsets": list(SUBSETS), "lambdas": [float(v) for v in LAMBDAS],
        "environments": list(environments),
        "waypoints": {n: list(xy) for n, xy in WAYPOINTS},
        "min_separation_m": MIN_SEPARATION_M,
        "tasks": [t[0] for t in TASKS], "arms": [a for a, _l, _s in ROUTING_ARMS],
        "driveable_cells": int(mask.sum()), "resolution_m": res,
        "link": {f"{a}|{c}": [float(v) for v in ab] for (a, c), ab in link.items()},
        "independent_heading_protocol": field_provenance,
    }, indent=1), encoding="utf-8")
    summary_paths = []
    for changed_environment in environments[1:]:
        out_summary = OUT / f"e3_cell_summary_{changed_environment}.csv"
        out_summary_manifest = OUT / f"e3_cell_summary_{changed_environment}_manifest.json"
        STATS.write_summary(
            routes_path,
            manifest_path,
            out_summary,
            out_summary_manifest,
            changed_environment=changed_environment,
        )
        summary_paths.append(out_summary)
        # Preserve the original L1 filenames as an explicit compatibility alias.
        # A later L2-only summary cannot overwrite them.
        if changed_environment == "L1":
            STATS.write_summary(
                routes_path,
                manifest_path,
                OUT / "e3_cell_summary.csv",
                OUT / "e3_cell_summary_manifest.json",
                changed_environment=changed_environment,
            )
    print(f"[e3] wrote {routes_path} ({len(rows)} rows)", flush=True)
    print(
        f"[e3] wrote {', '.join(path.name for path in summary_paths)} "
        f"({len(SUBSETS) * len(BUDGETS)} cells each)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
