#!/usr/bin/env python3
"""F51: multi-seed diagnosis for the F50 AWS B1 route-choice campaign."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml


ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
LOG = ROOT / "logs/visibility_comparison/f50_b1_tight_local_goal_3seed_v1"
OUT = ROOT / "timing_presentation/figures/F51"
GP = ROOT / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
WORLD_PROFILES = ROOT / "src/experiments/config/world_profiles.yaml"
WORLD = "warehouse_aws.world.sdf"
TASK = "F31_b1_apron_a3_mid"

C = {"C1": "#2563eb", "C2": "#dc2626"}
RACKS = [
    ("R4L", 1.725, 2.275, -0.8, 1.25),
    ("R4U", 1.725, 2.275, 2.2, 4.25),
    ("R5L", 3.875, 4.425, -0.8, 1.25),
    ("R5U", 3.875, 4.425, 2.2, 4.25),
]


def load_run(cond: str, seed: int):
    root = LOG / TASK / cond / f"seed{seed}"
    exps = sorted(root.glob("experiment_*"))
    if not exps:
        return None
    d = exps[-1]
    exp_path = d / "experiment.csv"
    if not exp_path.exists():
        return None
    exp = pd.read_csv(exp_path)
    summary_path = d / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    plan_path = d / "plan_samples.csv"
    plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
    perc_path = d / "perception.csv"
    perception = pd.read_csv(perc_path) if perc_path.exists() else pd.DataFrame()
    return {"dir": d, "exp": exp, "summary": summary, "plan": plan, "perception": perception}


def rel_time(exp: pd.DataFrame, summary: dict) -> pd.Series:
    t0 = summary.get("first_cmd_stamp")
    if t0 is None:
        t0 = exp["stamp"].iloc[0] if "stamp" in exp else 0.0
    return exp["stamp"] - float(t0)


def load_gp():
    if not GP.exists():
        return None
    d = np.load(GP, allow_pickle=True)
    if {"xs", "ys", "P_conservative_plan_map"}.issubset(d.files):
        return d["xs"], d["ys"], d["P_conservative_plan_map"]
    return None


def load_regions():
    d = yaml.safe_load(WORLD_PROFILES.read_text())
    return d["worlds"][WORLD].get("known_2d_regions", [])


def draw_scene(ax):
    gp = load_gp()
    if gp is not None:
        xs, ys, p = gp
        ax.contourf(xs, ys, p, levels=np.linspace(0, 1, 16), cmap="RdYlGn", alpha=0.35)
    for r in load_regions():
        if "non_driveable" not in str(r.get("type", "")):
            continue
        x, y = float(r["xmin"]), float(r["ymin"])
        w, h = float(r["xmax"]) - x, float(r["ymax"]) - y
        ax.add_patch(Rectangle((x, y), w, h, fc="#fca5a5", ec="#dc2626", alpha=0.25, lw=0.8))
    for nm, xmin, xmax, ymin, ymax in RACKS:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec="#111827", lw=1.7))
        ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, nm, ha="center", va="center", fontsize=7)
    ax.scatter([3.3], [-1.0], s=80, c="#16a34a", edgecolors="white", zorder=5, label="start")
    ax.scatter([1.0], [1.75], s=130, c="#111827", marker="*", zorder=5, label="goal")
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-3.2, 4.8)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.15)


def first_global(plan: pd.DataFrame):
    if plan.empty or "plan_stamp" not in plan:
        return None
    s = plan["plan_stamp"].min()
    gp = plan[plan["plan_stamp"] == s].sort_values("point_idx")
    if len(gp) < 2:
        return None
    return gp


def status_counts(exp: pd.DataFrame) -> str:
    if "optimizer_status" not in exp:
        return "-"
    counts = exp["optimizer_status"].dropna().value_counts().to_dict()
    return ", ".join(f"{int(k)}:{int(v)}" for k, v in sorted(counts.items()))


def summarize(run):
    if run is None:
        return {}
    exp = run["exp"]
    summ = run["summary"]
    solve = exp["solve_time_ms"].dropna() if "solve_time_ms" in exp else pd.Series(dtype=float)
    solve = solve[solve > 10].drop_duplicates()
    stale = 0.0
    age_col = "planner_pixel_correction_age_s"
    if age_col in exp and len(exp[age_col].dropna()):
        stale = float(exp[age_col].max())
    detect_rate = np.nan
    if "detected" in run["perception"] and len(run["perception"]):
        detect_rate = float(run["perception"]["detected"].astype(float).mean())
    return {
        "reason": summ.get("completion_reason", "unknown"),
        "goal": bool(summ.get("goal_reached", False)) or summ.get("completion_reason") == "goal_reached",
        "min_goal": summ.get("minimum_goal_distance"),
        "path": summ.get("path_length_m"),
        "min_obs": summ.get("min_obstacle_distance_m"),
        "mean_err": summ.get("mean_truth_state_error_m"),
        "solve_mean": float(solve.mean()) if len(solve) else np.nan,
        "solve_p90": float(solve.quantile(0.9)) if len(solve) else np.nan,
        "max_pixel_age": stale,
        "detect_rate": detect_rate,
        "status": status_counts(exp),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {(cond, seed): load_run(cond, seed) for cond in ("C1", "C2") for seed in (0, 1, 2)}

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.15, 1.15, 1.1], height_ratios=[1, 1, 0.82])
    fig.suptitle("F51 - F50 multi-seed diagnosis: route choice works once, execution is not robust", fontsize=16, fontweight="bold")

    for col, cond in enumerate(("C1", "C2")):
        ax = fig.add_subplot(gs[0:2, col])
        draw_scene(ax)
        for seed in (0, 1, 2):
            run = runs[(cond, seed)]
            if run is None:
                continue
            exp = run["exp"].dropna(subset=["truth_x", "truth_y"])
            if len(exp):
                ls = "-" if seed == 1 else "--" if seed == 0 else ":"
                ax.plot(exp["truth_x"], exp["truth_y"], color=C[cond], lw=2.0, ls=ls, label=f"seed {seed}")
                end = exp.iloc[-1]
                marker = "*" if run["summary"].get("completion_reason") == "goal_reached" else "X"
                ax.scatter(end["truth_x"], end["truth_y"], c=C[cond], marker=marker, s=100, edgecolors="white", zorder=6)
            gp = first_global(run["plan"])
            if gp is not None:
                ax.plot(gp["x"], gp["y"], color="#111827", alpha=0.25, lw=1.0)
        ax.set_title(f"{cond} executed trajectories and first global plans")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=8, loc="lower left")

    ax = fig.add_subplot(gs[0, 2])
    for cond in ("C1", "C2"):
        for seed in (0, 1, 2):
            run = runs[(cond, seed)]
            if run is None:
                continue
            exp = run["exp"]
            if "goal_dist" not in exp:
                continue
            ax.plot(rel_time(exp, run["summary"]), exp["goal_dist"], color=C[cond], alpha=0.45 + 0.15 * seed, lw=1.6, label=f"{cond}s{seed}")
    ax.axhline(0.25, color="#6b7280", ls="--", lw=1)
    ax.set_title("Goal distance")
    ax.set_xlabel("t after first cmd [s]")
    ax.set_ylabel("m")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1, 2])
    for cond in ("C1", "C2"):
        for seed in (0, 1, 2):
            run = runs[(cond, seed)]
            if run is None:
                continue
            exp = run["exp"]
            if "planner_pixel_correction_age_s" in exp:
                ax.plot(
                    rel_time(exp, run["summary"]),
                    exp["planner_pixel_correction_age_s"],
                    color=C[cond],
                    alpha=0.45 + 0.15 * seed,
                    lw=1.5,
                    label=f"{cond}s{seed}",
                )
    ax.axhline(1.25, color="#6b7280", ls="--", lw=1, label="pixel timeout")
    ax.set_title("Pixel correction age")
    ax.set_xlabel("t after first cmd [s]")
    ax.set_ylabel("s")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    rows = []
    for cond in ("C1", "C2"):
        for seed in (0, 1, 2):
            s = summarize(runs[(cond, seed)])
            rows.append([
                cond,
                str(seed),
                s.get("reason", "-"),
                "yes" if s.get("goal") else "no",
                f"{s.get('min_goal'):.2f}" if isinstance(s.get("min_goal"), (float, int)) else "-",
                f"{s.get('path'):.2f}" if isinstance(s.get("path"), (float, int)) else "-",
                f"{s.get('mean_err'):.2f}" if isinstance(s.get("mean_err"), (float, int)) else "-",
                f"{s.get('max_pixel_age'):.1f}",
                f"{s.get('detect_rate'):.2f}" if np.isfinite(s.get("detect_rate", np.nan)) else "-",
                f"{s.get('solve_mean'):.0f}" if np.isfinite(s.get("solve_mean", np.nan)) else "-",
                s.get("status", "-")[:32],
            ])
    cols = [
        "cond",
        "seed",
        "reason",
        "goal",
        "min goal",
        "path",
        "mean err",
        "max pix age",
        "det rate",
        "solve mean ms",
        "opt status counts",
    ]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = OUT / "F51_f50_seed_diagnosis.png"
    pdf = OUT / "F51_f50_seed_diagnosis.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)

    note = OUT / "F51_f50_seed_diagnosis.md"
    note.write_text(
        "# F51 - F50 Multi-Seed Diagnosis\n\n"
        "Files:\n"
        f"- `{png}`\n"
        f"- `{pdf}`\n\n"
        "Conclusion: F49 was a useful smoke success but F50 is not robust paper evidence. "
        "C1 fails in all completed seeds, which is consistent with the risky baseline, but "
        "C2 only reaches in seed 1. C2 seed 0 collides early and seed 2 times out after long "
        "stale visual-correction periods. The route-choice layer can produce the intended "
        "visibility-aware route, but the execution layer is still dominated by first global-solve "
        "latency, local max-iteration events, and stale visual updates during tracking.\n\n"
        "Next fix should target the runtime architecture, not the ambiguity weight: make the "
        "global plan a preflight or cached solve, tighten failure classification for repeated "
        "safe-stops, and improve local tracking robustness before treating AWS B1 as evidence.\n"
    )
    print(png)
    print(pdf)
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
