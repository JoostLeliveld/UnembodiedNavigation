#!/usr/bin/env python3
"""Stage 1 sanity check — can the EFE planner drive to a goal with PERFECT state?

Runs `closed_loop_offline` (receding-horizon planning against truth dynamics, no
Gazebo, no perception) from a campaign config's task start to its goal, for one or
both conditions. Prints whether the goal is reached and plots trajectory +
per-step goal distance / solve time / EFE.

This isolates the planner objective from perception entirely: if the robot cannot
reach the goal here, the problem is the EFE/optimizer, not YOLO or the EKF.

Usage:
    python3 scripts/diagnostics/sanity_offline_track.py \
        --config scripts/visibility_comparison/aws_sanity_open_config.yaml \
        --task visible_aisle_sanity_aws --conditions C1 C2 --n-steps 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = REPO_ROOT / "scripts" / "visibility_comparison"
sys.path.insert(0, str(LAB_DIR))

from efe_offline_lab import load_setup, closed_loop_offline  # noqa: E402


def run_one(config_path: Path, task: str, condition: str, n_steps: int,
            single_start: bool = False):
    setup = load_setup(config_path, condition=condition, task_override=task)
    if single_start:
        # Straight-line sanity needs no route choice; single warm-started solve
        # per step is far cheaper and isolates the objective from multistart.
        setup.planner.optimizer_multistart = False
        setup.planner.optimizer_initial_routes = []
        setup.planner.optimizer_multistart_lateral_offsets = []
    sx, sy, syaw = setup.start_xy_yaw
    m0 = np.array([sx, sy, syaw], dtype=float)
    result = closed_loop_offline(
        setup.planner, m0, setup.S0, setup.goal_xy, n_steps=n_steps, warm_start=True,
    )
    infos = result["infos"]
    traj = result["traj_m"]
    d_final = infos[-1]["d_goal"] if infos else float("nan")
    d_min = min((i["d_goal"] for i in infos), default=float("nan"))
    reached = d_min < 0.20
    solve_ms = [i["solver_time_ms"] for i in infos]
    # monotonic check: fraction of steps where d_goal decreased
    dgoals = [i["d_goal"] for i in infos]
    mono = (
        sum(1 for a, b in zip(dgoals, dgoals[1:]) if b <= a + 1e-3) / max(len(dgoals) - 1, 1)
    )
    return {
        "setup": setup, "result": result, "traj": traj,
        "d_final": d_final, "d_min": d_min, "reached": reached,
        "solve_mean": float(np.mean(solve_ms)) if solve_ms else float("nan"),
        "solve_max": float(np.max(solve_ms)) if solve_ms else float("nan"),
        "mono_frac": mono, "n_steps": len(infos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None, help="task name override")
    ap.add_argument("--conditions", nargs="+", default=["C1", "C2"])
    ap.add_argument("--n-steps", type=int, default=150)
    ap.add_argument("--single-start", action="store_true",
                    help="Disable multistart/route seeds (cheap straight-line sanity)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "logs" / "diagnostics" / "stage1_offline" / config_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for cond in args.conditions:
        r = run_one(config_path, args.task, cond, args.n_steps,
                    single_start=args.single_start)
        runs[cond] = r
        s = r["setup"]
        print(f"[{cond}] task={s.task_name} planner={s.planner_kind}")
        print(f"   start={s.start_xy_yaw[:2]} goal={s.goal_xy}")
        print(f"   reached={r['reached']}  d_min={r['d_min']:.3f}m  d_final={r['d_final']:.3f}m")
        print(f"   solve: mean={r['solve_mean']:.0f}ms max={r['solve_max']:.0f}ms over {r['n_steps']} steps")
        print(f"   monotonic-decrease fraction={r['mono_frac']:.0%}")
        print()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
    colors = {"C1": "#2563eb", "C2": "#dc2626", "C3": "#16a34a"}
    ax = axes[0]
    for cond, r in runs.items():
        t = r["traj"]
        ax.plot(t[:, 0], t[:, 1], "-o", ms=2, color=colors.get(cond, "#333"),
                label=f"{cond} ({'reached' if r['reached'] else 'FAILED'})")
        s = r["setup"]
        ax.scatter([s.start_xy_yaw[0]], [s.start_xy_yaw[1]], c="green", s=80, zorder=5)
        ax.scatter([s.goal_xy[0]], [s.goal_xy[1]], c="black", marker="*", s=160, zorder=5)
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Stage 1: offline trajectory (perfect state)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for cond, r in runs.items():
        dg = [i["d_goal"] for i in r["result"]["infos"]]
        ax.plot(dg, color=colors.get(cond, "#333"), label=cond)
    ax.axhline(0.20, color="#888", ls="--", label="0.20 m")
    ax.set_xlabel("step"); ax.set_ylabel("goal distance [m]")
    ax.set_title("Goal distance"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for cond, r in runs.items():
        sm = [i["solver_time_ms"] for i in r["result"]["infos"]]
        ax.plot(sm, color=colors.get(cond, "#333"), label=cond)
    ax.set_xlabel("step"); ax.set_ylabel("solve time [ms]")
    ax.set_title("Per-step solve time"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    png = out_dir / f"stage1_{(args.task or 'default')}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {png}")


if __name__ == "__main__":
    main()
