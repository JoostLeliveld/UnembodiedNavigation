"""
Quick trajectory overlay on GP background.
Usage:
  python3 scripts/visibility_comparison/plot_traj.py \
    --log-root logs/visibility_comparison/experiment_b_aws_v30_smoke \
    --gp-artifact logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz \
    --out logs/visibility_comparison/experiment_b_aws_v30_smoke/figures/traj.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RACKS = [
    # R1-R5 rack bodies (non-traversable, xmin xmax ymin ymax)
    (-4.60, -3.64, -2.03, 4.28),
    (-2.58, -1.62, -2.03, 4.28),
    (-0.56,  0.47, -2.03, 4.28),
    ( 1.50,  2.46, -2.03, 4.28),
    ( 3.52,  4.55, -2.03, 4.28),
]
R4_STACK = (2.29, 2.67, -0.42, 0.42)   # high occluder pad

def load_best_run(condition_dir: Path):
    """Return experiment.csv from the most recent run in a condition/seed dir."""
    csvs = sorted(condition_dir.rglob("experiment.csv"))
    return pd.read_csv(csvs[-1]) if csvs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--gp-artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="B1_clean_route_choice")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load GP background
    gp = np.load(args.gp_artifact)
    P = gp['P_mean_map']
    xs, ys = gp['xs'], gp['ys']

    # Campaign log
    camp_log = {}
    clog_path = log_root / "campaign_log.json"
    if clog_path.exists():
        camp_log = json.loads(clog_path.read_text())

    # Discover runs
    task_dir = log_root / args.task
    conditions = sorted(d.name for d in task_dir.iterdir() if d.is_dir()) if task_dir.exists() else []

    all_runs = {}
    for cond in conditions:
        for seed_dir in sorted((task_dir / cond).iterdir()):
            seed = seed_dir.name
            df = load_best_run(seed_dir)
            if df is not None:
                key = f"{cond}/{seed}"
                all_runs[key] = df

    seeds = sorted(set(s.split("/")[1] for s in all_runs))
    conds = sorted(set(s.split("/")[0] for s in all_runs))

    fig, axes = plt.subplots(len(conds), len(seeds),
                             figsize=(5 * len(seeds), 4.5 * len(conds)), squeeze=False)

    colors = {'C1': 'tab:blue', 'C2': 'tab:orange', 'C3': 'tab:green'}

    for ci, cond in enumerate(conds):
        for si, seed in enumerate(seeds):
            ax = axes[ci][si]
            key = f"{cond}/{seed}"

            # GP background
            ax.pcolormesh(xs, ys, P, cmap='viridis', vmin=0, vmax=1, shading='auto', zorder=0)

            # Rack boxes
            for (x0, x1, y0, y1) in RACKS:
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=0.5, edgecolor='white',
                                               facecolor='none', linestyle='--', zorder=2))
            # R4 high stack
            x0, x1, y0, y1 = R4_STACK
            ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                           linewidth=1.5, edgecolor='red',
                                           facecolor='none', zorder=3))

            # Goal marker
            ax.plot(3.25, 1.72, 'g*', ms=10, zorder=5)

            # Start marker
            ax.plot(2.15, -3.15, 'go', ms=6, zorder=5)

            if key in all_runs:
                df = all_runs[key]
                traj = df[['truth_x', 'truth_y']].dropna()
                color = colors.get(cond, 'gray')
                ax.plot(traj.truth_x, traj.truth_y, color=color, lw=1.5, zorder=4)
                ax.plot(traj.truth_x.iloc[0], traj.truth_y.iloc[0], 'o', color=color, ms=4, zorder=5)
                ax.plot(traj.truth_x.iloc[-1], traj.truth_y.iloc[-1], 'x', color=color, ms=6, zorder=5, mew=2)

            # Title
            camp_key = f"B1_clean_route_choice__{cond}__seed{seed.replace('seed','')}"
            info = camp_log.get(camp_key, {})
            outcome = info.get('outcome', 'unknown')
            mg = info.get('minimum_goal_distance')
            mg_str = f"  min_g={mg:.2f}m" if mg is not None else ""
            ax.set_title(f"{cond} {seed}  [{outcome}]{mg_str}", fontsize=8)

            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())
            ax.set_aspect('equal')
            ax.set_xlabel("x (m)", fontsize=7)
            ax.set_ylabel("y (m)", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"{log_root.name}", fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
