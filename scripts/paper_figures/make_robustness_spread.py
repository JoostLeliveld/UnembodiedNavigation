#!/usr/bin/env python3
"""Robustness spread map (2x2): per task, overlay all per-seed truth trajectories for C1
and C2 on the GP reliability map, marking collided runs. Shows route consistency + spread
+ where the constant-R baseline crashes vs the visibility-aware route staying observable.

For frozen paper figures, per-run outcomes come from the archived metrics CSV
used by the paper table. For newer campaign snapshots, the same counts can be
derived from campaign_log.json + run_summary.json so the figure is not tied to
the old metrics artifact.
"""
import sys, json, glob, math, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

def _dissolve_driveable(prisms):
    """Plotting-only: merge driveable prisms into clean polygons (dissolve internal cell seams)."""
    from shapely.ops import unary_union
    from shapely.geometry import box
    u = unary_union([box(p["xmin"], p["ymin"], p["xmax"], p["ymax"]) for p in prisms])
    return list(u.geoms) if u.geom_type == "MultiPolygon" else [u]

def _polygon_patch(poly, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        cs = list(ring.coords); verts += cs
        codes += [MplPath.MOVETO] + [MplPath.LINETO]*(len(cs)-2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kw)

ROOT = Path(__file__).resolve().parents[2]
# Frozen data source for the current robustness spread figure. This archive is
# separated from later rerun attempts so the figure stays consistent with the
# table inputs until the campaign is deliberately replaced.
CAMP = ROOT/"logs/visibility_comparison/_paper_runs/robustness_campaign_keepout_lanegraph_v1"
METRICS = ROOT/"paper_artifacts/metrics/archive/robustness_metrics.csv"
GPZ  = ROOT/"paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CONFIG = ROOT/"scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
OUT  = ROOT/"paper_artifacts/figures/robustness_spread.png"
PROV = OUT.with_suffix(".provenance.json")
TASKS = ["route_apron_to_a3_mid","route_apron_to_a2_mid","route_west_to_a1_upper","control_west_to_a1_low"]
SHORT = {
    "route_apron_to_a3_mid": "apron to A3",
    "route_apron_to_a2_mid": "apron to A2",
    "route_west_to_a1_upper": "west to A1 upper",
    "control_west_to_a1_low": "visible control",
}
LEGACY_TASK_DIRS = {
    "route_apron_to_a3_mid": "F31_b1_apron_a3_mid",
    "route_apron_to_a2_mid": "b5_a4_apron_to_a2_mid",
    "route_west_to_a1_upper": "b2_a0_west_to_a1_upper",
    "control_west_to_a1_low": "b6_a0_west_to_a1_low_control",
}
CANONICAL_TASK = {v: k for k, v in LEGACY_TASK_DIRS.items()}

# Optional CLI overrides so a NEW campaign can be plotted without editing source:
#   python make_robustness_spread.py --campaign-root logs/visibility_comparison/robustness_campaign_v2
import argparse as _argparse  # noqa: E402
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--campaign-root", default=str(CAMP))
_ap.add_argument("--metrics", default=str(METRICS))
_ap.add_argument("--gp", default=str(GPZ))
_ap.add_argument("--config", default=str(CONFIG))
_ap.add_argument("--out", default=str(OUT))
_args, _ = _ap.parse_known_args()
CAMP = Path(_args.campaign_root) if Path(_args.campaign_root).is_absolute() else ROOT / _args.campaign_root
METRICS = Path(_args.metrics) if Path(_args.metrics).is_absolute() else ROOT / _args.metrics
GPZ = Path(_args.gp) if Path(_args.gp).is_absolute() else ROOT / _args.gp
CONFIG = Path(_args.config) if Path(_args.config).is_absolute() else ROOT / _args.config
OUT = Path(_args.out) if Path(_args.out).is_absolute() else ROOT / _args.out
PROV = OUT.with_suffix(".provenance.json")

def _ival(x):
    try: return int(float(x))
    except (TypeError, ValueError): return 0

def _bval(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in ("1", "true", "yes", "y")

def _read_json(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}

def _rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

OUTCOME = {}                       # (task, cond, seed) -> {"collided": bool, "clean": bool}
COUNTS = {}                        # (task, cond) -> [n_clean, n_coll, n_total]
COUNT_SOURCE = None
if METRICS.is_file():
    # Outcomes from the metrics CSV (same source as the paper table).
    COUNT_SOURCE = METRICS
    for r in csv.DictReader(open(METRICS)):
        task_name = CANONICAL_TASK.get(r["task"], r["task"])
        cond = r["condition"]
        seed = str(r["seed"])
        collided = bool(_ival(r.get("is_collision")))
        clean = bool(_ival(r.get("is_clean_success")))
        key = (task_name, cond, seed)
        OUTCOME[key] = {"collided": collided, "clean": clean}
        ck = (task_name, cond); COUNTS.setdefault(ck, [0, 0, 0])
        COUNTS[ck][0] += int(clean); COUNTS[ck][1] += int(collided); COUNTS[ck][2] += 1
else:
    # Outcomes from the campaign's selected 40 runs. This path is used for the
    # current honest campaign, where no separate paper-table CSV is committed.
    campaign_log = CAMP / "campaign_log.json"
    COUNT_SOURCE = campaign_log
    for _key, entry in _read_json(campaign_log).items():
        task_name = CANONICAL_TASK.get(str(entry.get("task", "")), str(entry.get("task", "")))
        cond = str(entry.get("condition", ""))
        seed = str(entry.get("seed", ""))
        run_dir = Path(str(entry.get("run_dir", "")))
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        summary = _read_json(run_dir / "run_summary.json")
        completion = str(summary.get("completion_reason") or entry.get("completion_reason") or entry.get("outcome") or "")
        collided = (
            _bval(summary.get("collision_any"))
            or _bval(summary.get("collision_geom"))
            or completion == "collision"
        )
        clean = (
            (completion in ("goal_reached", "goal_reached_stable") or _bval(summary.get("goal_reached")) or _bval(entry.get("goal_reached")))
            and not collided
            and float(summary.get("max_obstacle_penetration_m", 0.0) or 0.0) <= 0.0
            and float(summary.get("max_wall_penetration_m", 0.0) or 0.0) <= 0.0
        )
        key = (task_name, cond, seed)
        OUTCOME[key] = {"collided": collided, "clean": clean, "run_dir": str(run_dir)}
        ck = (task_name, cond); COUNTS.setdefault(ck, [0, 0, 0])
        COUNTS[ck][0] += int(clean); COUNTS[ck][1] += int(collided); COUNTS[ck][2] += 1

def load_runs(task, cond):
    """Return [(states Nx2, collided)] per seed; trajectory from archive, collided from CSV."""
    out = []
    selected = [
        (seed, Path(meta["run_dir"]))
        for (task_name, cond_name, seed), meta in sorted(OUTCOME.items(), key=lambda x: int(x[0][2]) if str(x[0][2]).isdigit() else 999)
        if task_name == task and cond_name == cond and meta.get("run_dir")
    ]
    if selected:
        for seed, run_dir in selected:
            csv_path = run_dir / "experiment.csv"
            if not csv_path.is_file():
                continue
            tx, ty = [], []
            for row in csv.DictReader(open(csv_path)):
                try: x = float(row["gt_x"]); y = float(row["gt_y"])
                except (KeyError, ValueError): continue
                if math.isfinite(x) and math.isfinite(y): tx.append(x); ty.append(y)
            if not tx:
                continue
            collided = OUTCOME.get((task, cond, str(seed)), {}).get("collided", False)
            out.append((np.column_stack([tx, ty]), collided))
        return out

    task_dirs = [task]
    legacy = LEGACY_TASK_DIRS.get(task)
    if legacy:
        task_dirs.append(legacy)
    seeddirs = []
    for task_dir in task_dirs:
        seeddirs = sorted(glob.glob(str(CAMP/task_dir/cond/"seed*")))
        if seeddirs:
            break
    for seeddir in seeddirs:
        seed = Path(seeddir).name.replace("seed", "")
        exps = sorted(glob.glob(seeddir+"/experiment_*/experiment.csv"))
        if not exps: continue
        tx, ty = [], []
        for row in csv.DictReader(open(exps[-1])):
            # GROUND-TRUTH path only (gt_x/gt_y). odom_map_x/y = /odom drifts from
            # the true pose and would misplace the plotted trajectory.
            try: x = float(row["gt_x"]); y = float(row["gt_y"])
            except (KeyError, ValueError): continue
            if math.isfinite(x) and math.isfinite(y): tx.append(x); ty.append(y)
        if not tx: continue
        collided = OUTCOME.get((task, cond, seed), {}).get("collided", False)
        out.append((np.column_stack([tx, ty]), collided))
    return out

gp = np.load(GPZ, allow_pickle=True); xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"]) for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]
import yaml; drive = json.loads(yaml.safe_load(open(CONFIG))["driveable_geometry_json"])["prisms"]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.0))
for ax, task in zip(axes.ravel(), TASKS):
    ax.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.9, alpha=0.85, aspect="equal", zorder=0)
    for poly in _dissolve_driveable(drive):
        ax.add_patch(_polygon_patch(poly, facecolor="none", edgecolor="#39ff14", lw=1.6, alpha=0.75, zorder=1))
    for (x0, x1, y0, y1) in racks:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="#f2cf23", edgecolor="#0b6f8a", lw=0.5, zorder=4))
    for cond, colr in (("C1", "#ff2d2d"), ("C2", "#1a1aff")):
        for st, collided in load_runs(task, cond):
            ax.plot(st[:, 0], st[:, 1], color=colr, lw=2.3, alpha=0.8, solid_capstyle="round", zorder=6)
            if collided:
                ax.scatter([st[-1, 0]], [st[-1, 1]], marker="x", s=120, color=colr, lw=3.0, zorder=9)
    r0 = load_runs(task, "C1") or load_runs(task, "C2")
    if r0:
        st0 = r0[0][0]; ax.scatter([st0[0, 0]], [st0[0, 1]], s=80, c="#16a34a", edgecolor="k", zorder=10)
    c1 = COUNTS.get((task, "C1"), [0, 0, 0]); c2 = COUNTS.get((task, "C2"), [0, 0, 0])
    ax.set_title(f"{SHORT[task]}\n"
                 f"C1 {c1[0]}/{c1[2]} goal, {c1[1]} safe-stall  |  C2 {c2[0]}/{c2[2]} goal, {c2[1]} safe-stall",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-5.6, 5.4); ax.set_ylim(-3.7, 5.0); ax.set_xticks([]); ax.set_yticks([])
handles = [Line2D([0], [0], color="#ff2d2d", lw=2.2, label="C1 constant-$R$ (per seed)"),
           Line2D([0], [0], color="#1a1aff", lw=2.2, label="C2 visibility-aware (per seed)"),
           Line2D([0], [0], marker="x", color="#333", lw=0, markersize=10, label="safe stall (0 region-exit)"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a", markeredgecolor="k", markersize=10, label="start")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.035, 1, 1.0]); OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
PROV.write_text(json.dumps({
    "figure": "robustness_spread",
    "layout": "2x2",
    "trajectory_source": _rel(CAMP),
    "counts_source": _rel(COUNT_SOURCE),
    "gp": _rel(GPZ),
    "config": _rel(CONFIG),
    "tasks": TASKS,
    "conditions": ["C1", "C2"],
    "seeds_per_condition": 5,
    "output": _rel(OUT),
}, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
for t in TASKS:
    print(f"  {SHORT[t]:14} C1 {COUNTS.get((t,'C1'))}  C2 {COUNTS.get((t,'C2'))}")
