#!/usr/bin/env python3
"""Offline-lab view of the EXACT Gazebo-campaign multistart, per task.

Purpose: show that the route-choice multistart is not "overpowered" or
hand-scripted. For each campaign task we replicate the locked config EXACTLY
(optimizer_multistart=true, include_direct=false, lateral_offsets=0.0,
maxiter=120, maxfun=720, warm_start=true, global_horizon=120) and render:

  * the multistart SEED routes actually handed to the optimizer
    (the two condition-neutral lane-graph homotopy routes via_connector /
     via_lower, identical for C1 and C2, + the neutral midpoint lateral seed),
  * the per-seed converged EFE cost (the basin landscape), and
  * the route the multistart actually CHOSE for C1 vs C2 (selected_source).

The story: a coarse lane-graph router emits the two topologically distinct
routes; the local EFE optimizer refines each; the lowest-cost VALID candidate
wins. Same seeds for both conditions -> any C1/C2 route divergence is the EFE's,
not the seed's. Backend keep-in fix is active (plans stay in the lane union).

Writes logs/paper_figures/multistart_seeds_offline.png + a JSON cache.
"""
import sys, json, copy
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D


def _dissolve_driveable(prisms):
    """Plotting-only: dissolve the keep-in prism union into the polygon the planner
    actually solves against (the union boundary, no internal cell seams)."""
    from shapely.ops import unary_union
    from shapely.geometry import box
    u = unary_union([box(p["xmin"], p["ymin"], p["xmax"], p["ymax"]) for p in prisms])
    return list(u.geoms) if u.geom_type == "MultiPolygon" else [u]


def _polygon_patch(poly, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        cs = list(ring.coords); verts += cs
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(cs) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kw)

ROOT = Path(__file__).resolve().parents[2]
for p in ["scripts/visibility_comparison", "scripts/paper_figures", "src/planning", "src/unav_common"]:
    sys.path.insert(0, str(ROOT / p))
import yaml
from efe_offline_lab import build_planner, load_setup, _find_task  # noqa: E402
from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy  # noqa: E402

CONFIG = ROOT / "scripts/visibility_comparison/aws_f31b1_final_config.yaml"
GPZ = ROOT / "paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
OUT = ROOT / "logs/paper_figures/multistart_seeds_offline.png"
CACHE = ROOT / "logs/paper_figures/multistart_seeds_offline.json"

CFG = yaml.safe_load(open(CONFIG))
TASKS = list(CFG["tasks"].keys())
PRISMS = tuple(scene_from_json(CFG["driveable_geometry_json"]).prisms)


def pct_outside(st):
    sd = signed_distance_to_union_xy(PRISMS, st[:, :2], keep_in=True)
    return 100.0 * float((sd > 1e-4).mean())


def make_planner_for(task, cond):
    """Replicate the campaign launch EXACTLY: merge the per-task route seeds into
    the top-level cfg, then build the planner with the locked optimizer settings."""
    s = load_setup(CONFIG, condition=cond, seed=0, task_override=task)
    cfg = dict(s.config)
    task_cfg = CFG["tasks"][task]
    # campaign applies the per-task optimizer_initial_routes_json (run_visibility_campaign.py:387)
    cfg["optimizer_initial_routes_json"] = task_cfg["optimizer_initial_routes_json"]
    # hierarchical RUNTIME global solve uses global_horizon (offline-lab default is legacy 20)
    cfg["horizon"] = int(cfg.get("global_horizon", 120))
    pl = build_planner(
        cfg, planner_kind=s.planner_kind, camera_params=s.camera_params,
        visibility_artifact_path=str(s.gp_path), visibility_geometry_json=s.geometry_json,
        collision_geometry_json=s.geometry_json, driveable_geometry_json=s.driveable_geometry_json,
        seed=0, visibility_target_height_m=s.gp["target_height"],
    )
    return s, pl, cfg


def solve_full(task, cond):
    """Authoritative: fresh full-multistart plan() exactly as Gazebo runs it."""
    s, pl, cfg = make_planner_for(task, cond)
    pl.prev_controls_flat = None
    pl._prev_goal_xy = None
    plan = pl.plan(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float), np.asarray(s.goal_xy, float))
    st = np.asarray(plan.states, float)
    return dict(
        states=st.tolist(),
        selected_source=str(plan.selected_source),
        total=float(plan.total_cost),
        risk=float(plan.risk_cost), ambiguity=float(plan.ambiguity_cost), obstacle=float(plan.obstacle_cost),
        valid=bool(plan.rollout_valid), pct_out=pct_outside(st),
        d_goal=float(np.hypot(st[-1, 0] - s.goal_xy[0], st[-1, 1] - s.goal_xy[1])),
        start=list(map(float, s.start_xy_yaw)), goal=list(map(float, s.goal_xy)),
    )


def solve_per_seed(task, cond):
    """Basin landscape: refine each named lane-graph seed ALONE (single-candidate
    multistart, no warm/direct/lateral) to expose its converged EFE cost + path."""
    out = {}
    routes = json.loads(CFG["tasks"][task]["optimizer_initial_routes_json"])
    for rt in routes:
        s, pl, cfg = make_planner_for(task, cond)
        pl.optimizer_initial_routes = pl._parse_initial_routes(json.dumps([rt]))
        pl.optimizer_multistart_include_direct = False
        pl.optimizer_multistart_lateral_offsets = []
        pl.optimizer_warm_start = False
        pl.prev_controls_flat = None
        pl._prev_goal_xy = None
        plan = pl.plan(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float), np.asarray(s.goal_xy, float))
        st = np.asarray(plan.states, float)
        out[rt["name"]] = dict(states=st.tolist(), total=float(plan.total_cost),
                               valid=bool(plan.rollout_valid), pct_out=pct_outside(st),
                               waypoints=rt["waypoints"])
    return out


def main():
    if "--from-cache" in sys.argv and CACHE.exists():
        data = json.loads(CACHE.read_text())
        print(f"loaded cache {CACHE}")
        _render(data); return
    data = {}
    for task in TASKS:
        data[task] = {"seeds": {}, "chosen": {}}
        for cond in ("C1", "C2"):
            data[task]["chosen"][cond] = solve_full(task, cond)
        # per-seed landscape from the C2 planner (geometry/keep-in identical across conditions;
        # cost differs only via the GP-modulated covariance, shown in the chosen-route costs).
        data[task]["seeds"]["C1"] = solve_per_seed(task, "C1")
        data[task]["seeds"]["C2"] = solve_per_seed(task, "C2")
        for cond in ("C1", "C2"):
            ch = data[task]["chosen"][cond]
            print(f"{task:26} {cond}: chose {ch['selected_source']:24} total={ch['total']:8.1f} "
                  f"valid={ch['valid']} out={ch['pct_out']:.1f}% d={ch['d_goal']:.2f}")
    CACHE.write_text(json.dumps(data))
    _render(data)


def _render(data):
    # ---- figure ----
    gp = np.load(GPZ, allow_pickle=True)
    xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
             for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]
    drive = json.loads(CFG["driveable_geometry_json"])["prisms"]
    SEEDC = {"via_connector": "#ff7f0e", "via_lower": "#9467bd"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, task in zip(axes.ravel(), TASKS):
        d = data[task]
        ax.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.9, alpha=0.85, zorder=0)
        for poly in _dissolve_driveable(drive):
            ax.add_patch(_polygon_patch(poly, facecolor="none", edgecolor="#39ff14", lw=1.4, alpha=0.75, zorder=1))
        for (x0, x1, y0, y1) in racks:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="#f2cf23", edgecolor="#0b6f8a", lw=0.6, zorder=3))
        st0 = d["chosen"]["C1"]
        start, goal = st0["start"], st0["goal"]
        # multistart SEED routes (raw lane-graph guesses, condition-neutral, dashed)
        for rt in json.loads(CFG["tasks"][task]["optimizer_initial_routes_json"]):
            w = np.vstack([[start[0], start[1]], np.array(rt["waypoints"], float)])
            ax.plot(w[:, 0], w[:, 1], "o--", color=SEEDC[rt["name"]], lw=1.6, ms=4, alpha=0.9, zorder=5,
                    label=f"seed: {rt['name']}")
        # CHOSEN refined routes
        for cond, colr in (("C1", "#e8000b"), ("C2", "#111111")):
            ch = d["chosen"][cond]
            st = np.asarray(ch["states"], float)
            won = ch["selected_source"].replace("solver:route:", "via ").replace("solver:", "")
            ax.plot(st[:, 0], st[:, 1], "-", color=colr, lw=2.6, zorder=8,
                    label=f"{cond} chose [{won}]  J={ch['total']:.0f}")
        ax.scatter([start[0]], [start[1]], s=90, c="#16a34a", edgecolor="k", zorder=10)
        ax.scatter([goal[0]], [goal[1]], s=140, c="#e41a1c", marker="*", edgecolor="k", zorder=10)
        sc = d["seeds"]["C2"]
        sub = "  ".join(f"{k}:J={v['total']:.0f}{'✓' if v['valid'] else '✗'}" for k, v in sc.items())
        ax.set_title(f"{task}\nper-seed converged (C2): {sub}", fontsize=9, fontweight="bold")
        ax.set_xlim(-5.7, 5.5); ax.set_ylim(-3.7, 5.0)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(loc="upper left", fontsize=7.0, framealpha=0.85)
    fig.suptitle("Campaign multistart, replicated offline (locked config, backend keep-in fix active)\n"
                 "Two condition-neutral lane-graph homotopy seeds + neutral midpoint; lowest-cost VALID candidate wins. "
                 "Identical seeds for C1 & C2.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
