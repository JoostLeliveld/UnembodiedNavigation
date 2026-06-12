#!/usr/bin/env python3
"""Solve diagnostics for the locked campaign config. For each task x condition,
instrument the one-shot global EFE solve to capture, per multistart attempt:
the objective-vs-iteration convergence curve, the L-BFGS-B iteration count,
the convergence status, and the final projected-gradient inf-norm. Repeats the
solve to report wall-time stability and whether the chosen route / total cost
are identical run to run. Renders, per task: (left) the convergence curves
showing HOW each solve descended and whether it truly converged or hit maxiter,
(right) the resulting C1/C2 paths on the GP map. This is the pre-launch check:
is the optimization well-behaved (numerically), not just fast/slow (contention)."""
import sys, json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import yaml, warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
for p in ["scripts/paper_figures", "scripts/visibility_comparison", "src/planning", "src/unav_common"]:
    sys.path.insert(0, str(ROOT / p))
import planning.planners.base_planner as bp  # noqa: E402
from diag_route_suite import make_planner  # noqa: E402
from unav_common.lane_graph_routes import generate_route_seeds  # noqa: E402
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402

CONFIG = ROOT / "scripts/visibility_comparison/aws_f31b1_final_config.yaml"
GPZ = ROOT / "paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
OUT = ROOT / "logs/paper_figures/solve_diagnostics.png"
CACHE = ROOT / "logs/paper_figures/solve_diagnostics.json"
CFG = yaml.safe_load(open(CONFIG)); DJ = CFG["driveable_geometry_json"]
MAXITER = int(CFG.get("optimizer_maxiter", 60))
DRIVE = json.loads(DJ)["prisms"]
TASKS = list(CFG["tasks"].keys())
T = {t["name"]: t for t in yaml.safe_load(open(ROOT / "src/experiments/config/tasks.yaml"))["tasks"]["warehouse_aws.world.sdf"]}
gp = np.load(GPZ, allow_pickle=True); xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
N_TIMING = 2  # extra clean (untraced) repeats for wall-time + route/cost stability

# ---- spy on scipy minimize to capture per-attempt convergence + result ----
_real = bp.minimize; CAP = []; TRACE_ON = [False]
def _spy(fun, x0, *a, **kw):
    trace = []
    if TRACE_ON[0]:
        def cb(xk):
            fv = fun(xk)
            trace.append(float(fv[0] if isinstance(fv, (tuple, list)) else fv))
        kw = dict(kw); kw["callback"] = cb
    t0 = time.perf_counter(); r = _real(fun, x0, *a, **kw); dt = time.perf_counter() - t0
    g = np.asarray(getattr(r, "jac", []), float)
    CAP.append(dict(trace=trace, nit=int(getattr(r, "nit", -1)),
                    success=bool(getattr(r, "success", False)), status=int(getattr(r, "status", -1)),
                    ginf=(float(np.max(np.abs(g))) if g.size else None),
                    fun=float(getattr(r, "fun", np.nan)), dt=dt))
    return r
bp.minimize = _spy


def _make_pl(cond, name):
    t = T[name]; s = (t["start"]["x"], t["start"]["y"]); g = (t["goal"]["x"], t["goal"]["y"])
    gen = generate_route_seeds(DJ, s, g)
    sp, pl = make_planner(cond, name, gen, None, maxiter=MAXITER, multistart=True)
    pl.optimizer_initial_routes = pl._parse_initial_routes(json.dumps(gen))
    pl.optimizer_multistart = True; pl.optimizer_multistart_include_direct = False
    pl.optimizer_multistart_lateral_offsets = []; pl.optimizer_warm_start = True
    pl.prev_controls_flat = None; pl._prev_goal_xy = None
    return sp, pl, gen


def _solve(cond, name, trace):
    sp, pl, gen = _make_pl(cond, name)
    CAP.clear(); TRACE_ON[0] = trace
    plan = pl.plan(np.asarray(sp.start_xy_yaw, float), np.asarray(sp.S0, float), np.asarray(sp.goal_xy, float))
    return dict(route=str(plan.selected_source).replace("solver:route:", "").replace("solver:", ""),
                total=float(plan.total_cost), states=np.asarray(plan.states, float).tolist(),
                attempts=[dict(c) for c in CAP],
                start=list(map(float, sp.start_xy_yaw)), goal=list(map(float, sp.goal_xy)), seeds=gen)


def collect():
    data = {}
    for name in TASKS:
        data[name] = {}
        for cond in ("C1", "C2"):
            traced = _solve(cond, name, True)
            ts, routes, costs = [], [], []
            for _ in range(N_TIMING):
                r = _solve(cond, name, False)
                ts.append(sum(a["dt"] for a in r["attempts"])); routes.append(r["route"]); costs.append(r["total"])
            traced["solve_s_mean"] = float(np.mean(ts)); traced["solve_s_std"] = float(np.std(ts))
            traced["route_stable"] = (len(set(routes + [traced["route"]])) == 1)
            traced["cost_spread"] = float(max(costs + [traced["total"]]) - min(costs + [traced["total"]]))
            data[name][cond] = traced
            a = traced["attempts"]
            print(f"{name:28} {cond}: route={traced['route']:16} solve={traced['solve_s_mean']:.1f}+-{traced['solve_s_std']:.1f}s "
                  f"nit={[x['nit'] for x in a]}/{MAXITER} conv={[x['success'] for x in a]} "
                  f"ginf={[round(x['ginf'],1) if x['ginf'] is not None else None for x in a]} "
                  f"route_stable={traced['route_stable']} cost_spread={traced['cost_spread']:.2g}", flush=True)
    CACHE.write_text(json.dumps(data)); return data


# ---------- rendering ----------
def _dissolve(prisms):
    from shapely.ops import unary_union
    from shapely.geometry import box
    u = unary_union([box(p["xmin"], p["ymin"], p["xmax"], p["ymax"]) for p in prisms])
    return list(u.geoms) if u.geom_type == "MultiPolygon" else [u]


def _patch(poly, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        cs = list(ring.coords); verts += cs
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(cs) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kw)


def render(data):
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
             for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(15, 3.6 * len(TASKS)))
    CC = {"C1": "#e8000b", "C2": "#111111"}
    for i, name in enumerate(TASKS):
        axc, axp = axes[i, 0], axes[i, 1]
        # ---- convergence panel ----
        for cond in ("C1", "C2"):
            d = data[name][cond]
            for k, att in enumerate(d["attempts"]):
                tr = att["trace"]
                if not tr:
                    continue
                hit = "MAXIT" if att["nit"] >= MAXITER else ("conv" if att["success"] else f"st{att['status']}")
                ls = "-" if k == 0 else "--"
                axc.semilogy(range(1, len(tr) + 1), tr, ls, color=CC[cond], lw=1.8, alpha=0.85,
                             label=f"{cond} a{k}: {att['nit']}it {hit} g={att['ginf']:.1f}" if att["ginf"] is not None else f"{cond} a{k}")
        axc.axvline(MAXITER, color="grey", ls=":", lw=1)
        axc.set_xlabel("L-BFGS-B iteration"); axc.set_ylabel("EFE objective (log)")
        c1, c2 = data[name]["C1"], data[name]["C2"]
        axc.set_title(f"{name}\nsolve C1={c1['solve_s_mean']:.0f}+-{c1['solve_s_std']:.0f}s "
                      f"C2={c2['solve_s_mean']:.0f}+-{c2['solve_s_std']:.0f}s  "
                      f"route_stable C1={c1['route_stable']} C2={c2['route_stable']}", fontsize=9)
        axc.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        axc.grid(True, which="both", alpha=0.2)
        # ---- path panel ----
        axp.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.9, alpha=0.85, zorder=0)
        for poly in _dissolve(DRIVE):
            axp.add_patch(_patch(poly, facecolor="none", edgecolor="#39ff14", lw=1.2, alpha=0.7, zorder=1))
        for (x0, x1, y0, y1) in racks:
            axp.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f2cf23", edgecolor="#0b6f8a", lw=0.6, zorder=3))
        st0 = data[name]["C1"]["start"]; gl = data[name]["C1"]["goal"]
        for rt in data[name]["C1"]["seeds"]:
            w = np.vstack([[st0[0], st0[1]], np.array(rt["waypoints"], float)])
            axp.plot(w[:, 0], w[:, 1], "--", color="#888", lw=1.0, alpha=0.6, zorder=4)
        for cond in ("C1", "C2"):
            d = data[name][cond]; s = np.asarray(d["states"], float)
            axp.plot(s[:, 0], s[:, 1], "-", color=CC[cond], lw=2.6, zorder=7,
                     label=f"{cond}: {d['route']} cost={d['total']:.0f}")
        axp.scatter([st0[0]], [st0[1]], s=80, c="#16a34a", edgecolor="k", zorder=10)
        axp.scatter([gl[0]], [gl[1]], s=140, c="#e41a1c", marker="*", edgecolor="k", zorder=10)
        axp.set_xlim(-5.7, 5.5); axp.set_ylim(-3.7, 5.0); axp.set_xlabel("x [m]"); axp.set_ylabel("y [m]")
        axp.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle(f"Global EFE solve diagnostics (offline, no contention) - maxiter={MAXITER}, "
                 f"multistart=lane-graph seeds. Convergence curves + wall-time stability + chosen paths.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=135, bbox_inches="tight"); print(f"wrote {OUT}")


if __name__ == "__main__":
    if "--from-cache" in sys.argv and CACHE.exists():
        render(json.loads(CACHE.read_text()))
    else:
        render(collect())
