#!/usr/bin/env python3
"""6-task offline EFE decomposition figure under the LOCKED config.

Top row : per task, the GP rho background + driveable lanes + racks + the CHOSEN C1 and C2
          planned paths (chosen = lower-total of the two seeds) + start/goal.
Bottom  : per task, stacked cost decomposition (risk / ambiguity / no-go) for each starting
          seed x condition {C1-short, C1-long, C2-short, C2-long}; the chosen seed is starred.

Uses the committed config values verbatim (no goal-prior override): belief-tube no-go on,
ambiguity_weight=1, etc. Run after the locked config is in place.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/paper_figures"))
sys.path.insert(0, str(ROOT/"scripts/visibility_comparison"))
sys.path.insert(0, str(ROOT/"src/planning"))
from diag_route_suite import TASKS, make_planner, realized   # reuse task defs + planner builder
from efe_offline_lab import load_setup

GPZ = ROOT/"paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
OUT = ROOT/"logs/paper_figures/suite_decomposition.png"
CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"
TASK_ORDER = ["F31_b1_apron_a3_mid","b5_a4_apron_to_a2_mid","b2_a0_west_to_a1_upper",
              "b3_lowereast_to_a2_mid","b6_a0_west_to_a1_low_control","b7_a0_west_to_a2_mid"]
SHORT_LABEL = {"F31_b1_apron_a3_mid":"F31_b1","b5_a4_apron_to_a2_mid":"b5","b2_a0_west_to_a1_upper":"b2",
               "b3_lowereast_to_a2_mid":"b3","b6_a0_west_to_a1_low_control":"b6(ctrl)","b7_a0_west_to_a2_mid":"b7(ctrl)"}
MAXITER = 350
TERMS = ["risk","ambiguity","obstacle"]
TERMC = {"risk":"#d94b41","ambiguity":"#7b3294","obstacle":"#2c7fb8"}


def single_seed(cond, task, seed_name):
    tc = TASKS[task]
    routes = [{"name": seed_name, "waypoints": tc[seed_name]}]
    s, pl = make_planner(cond, task, routes, None, maxiter=MAXITER, multistart=False)  # gfs=None -> config value
    plan = pl.plan(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float), np.asarray(s.goal_xy, float))
    st = np.asarray(plan.states, float)
    c = {k: float(getattr(plan, k+"_cost", 0.0)) for k in ["risk","ambiguity","obstacle","total"]}
    return c, st, s.start_xy_yaw, s.goal_xy, tc


def main():
    gp = np.load(GPZ, allow_pickle=True)
    xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    geom = json.loads(str(gp["geometry_json"]).strip("[]'")) if "geometry_json" in gp else {"prisms":[]}
    racks = [(p["xmin"],p["xmax"],p["ymin"],p["ymax"]) for p in geom.get("prisms",[])]
    import yaml
    drive = json.loads(yaml.safe_load(open(CONFIG))["driveable_geometry_json"])["prisms"]

    # ---- solve everything ----
    data = {}
    for task in TASK_ORDER:
        data[task] = {}
        for cond in ("C1","C2"):
            for seed in ("short","long"):
                c, st, start, goal, tc = single_seed(cond, task, seed)
                data[task][(cond,seed)] = dict(costs=c, states=st, start=start, goal=goal)
        print(f"solved {task}")

    fig, axes = plt.subplots(2, 6, figsize=(26, 9), gridspec_kw={"height_ratios":[1.25,1.0]})
    for col, task in enumerate(TASK_ORDER):
        d = data[task]
        start = d[("C1","short")]["start"]; goal = d[("C1","short")]["goal"]
        chosen = {cond: min(("short","long"), key=lambda sd: d[(cond,sd)]["costs"]["total"]) for cond in ("C1","C2")}

        # ---- path panel ----
        ax = axes[0, col]
        ax.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=0.9, alpha=0.85, aspect="equal", zorder=0)
        for pr in drive:
            ax.add_patch(Rectangle((pr["xmin"],pr["ymin"]),pr["xmax"]-pr["xmin"],pr["ymax"]-pr["ymin"],
                                   facecolor="none",edgecolor="#39ff14",lw=0.5,alpha=0.5,zorder=1))
        for (x0,x1,y0,y1) in racks:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor="#f2cf23",edgecolor="#0b6f8a",lw=0.5,zorder=4))
        for cond,colr in (("C1","#ff2d2d"),("C2","#111111")):
            st = d[(cond,chosen[cond])]["states"]
            ax.plot(st[:,0],st[:,1],color=colr,lw=2.0,zorder=6,
                    label=f"{cond}: {'blind/short' if chosen[cond]=='short' else 'observable/long'}")
        ax.scatter([start[0]],[start[1]],s=55,c="#16a34a",edgecolor="k",zorder=8)
        ax.scatter([goal[0]],[goal[1]],s=60,c="#e41a1c",marker="*",edgecolor="k",zorder=8)
        ax.set_title(f"{SHORT_LABEL[task]}  ({TASKS[task]['kind']})", fontsize=11, fontweight="bold")
        ax.set_xlim(-5.6,5.4); ax.set_ylim(-3.7,5.0); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower center", fontsize=7, ncol=1, framealpha=0.7)

        # ---- decomposition panel ----
        ax2 = axes[1, col]
        bars = [("C1","short"),("C1","long"),("C2","short"),("C2","long")]
        xlabels = ["C1\nshort","C1\nlong","C2\nshort","C2\nlong"]
        x = np.arange(len(bars))
        bottoms = np.zeros(len(bars))
        for term in TERMS:
            vals = np.array([d[b]["costs"][term] for b in bars])
            ax2.bar(x, vals, bottom=bottoms, color=TERMC[term], label=term if col==0 else None, width=0.7)
            bottoms += vals
        # star the chosen seed per condition
        for i,b in enumerate(bars):
            if chosen[b[0]] == b[1]:
                ax2.text(x[i], bottoms[i]*1.02, "★", ha="center", va="bottom", fontsize=12, color="#000")
        ax2.set_xticks(x); ax2.set_xticklabels(xlabels, fontsize=8)
        ax2.set_title("cost decomposition (★=chosen)", fontsize=9)
        if col==0: ax2.set_ylabel("EFE cost"); ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("6-task offline EFE under locked config (belief-tube no-go ON, ambiguity_weight=1, gp aws_gp_v7b) — chosen path (top) + per-seed cost decomposition (bottom)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")

    # text summary
    print("\n== route choice + totals ==")
    for task in TASK_ORDER:
        d = data[task]
        ch = {cond: min(("short","long"), key=lambda sd: d[(cond,sd)]["costs"]["total"]) for cond in ("C1","C2")}
        print(f"  {SHORT_LABEL[task]:9} C1->{ch['C1']:5} (tot {d[('C1',ch['C1'])]['costs']['total']:8.1f})   "
              f"C2->{ch['C2']:5} (tot {d[('C2',ch['C2'])]['costs']['total']:8.1f})   "
              f"{'SPLIT' if ch['C1']=='short' and ch['C2']=='long' else ('same' if ch['C1']==ch['C2'] else 'other')}")


if __name__ == "__main__":
    main()
