#!/usr/bin/env python3
"""Plot mean/std inferred and planned paths for one batch condition."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _goal_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if abs(_f(r, "goal_x")) > 1e-9 or abs(_f(r, "goal_y")) > 1e-9]


def _resample_path_xy(x: np.ndarray, y: np.ndarray, n_pts: int) -> Optional[np.ndarray]:
    if x.size != y.size or x.size < 2:
        return None
    xy = np.column_stack([x, y]).astype(float)
    if not np.all(np.isfinite(xy)):
        mask = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        xy = xy[mask]
    if xy.shape[0] < 2:
        return None
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    keep = np.ones_like(s, dtype=bool)
    keep[1:] = d > 1e-9
    xy = xy[keep]
    s = s[keep]
    if xy.shape[0] < 2 or s[-1] <= 1e-9:
        return None
    su = np.linspace(0.0, float(s[-1]), int(n_pts))
    xr = np.interp(su, s, xy[:, 0])
    yr = np.interp(su, s, xy[:, 1])
    return np.column_stack([xr, yr])


def _load_plan_snapshots(run_dir: Path) -> List[Tuple[float, np.ndarray]]:
    path = run_dir / "plan_samples.csv"
    if not path.exists():
        return []
    grouped: Dict[float, List[Tuple[int, float, float]]] = {}
    for row in _read_csv(path):
        try:
            stamp = float(row["plan_stamp"])
            idx = int(float(row["point_idx"]))
            x = float(row["x"])
            y = float(row["y"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(stamp, []).append((idx, x, y))

    out: List[Tuple[float, np.ndarray]] = []
    for stamp in sorted(grouped.keys()):
        pts = sorted(grouped[stamp], key=lambda t: t[0])
        xy = np.asarray([[p[1], p[2]] for p in pts], dtype=float)
        if xy.shape[0] >= 2:
            out.append((float(stamp), xy))
    return out


def _select_plan_snapshot(
    plan_snaps: Sequence[Tuple[float, np.ndarray]],
    goal_start_stamp: float,
    mode: str,
) -> Optional[np.ndarray]:
    if not plan_snaps:
        return None
    if mode == "first_after_goal":
        for stamp, xy in plan_snaps:
            if stamp >= goal_start_stamp - 1e-9:
                return xy
        return plan_snaps[0][1]
    if mode == "last":
        return plan_snaps[-1][1]
    if mode == "longest":
        best_xy = None
        best_len = -1.0
        for _, xy in plan_snaps:
            d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
            L = float(np.sum(d))
            if L > best_len:
                best_len = L
                best_xy = xy
        return best_xy
    raise ValueError(f"Unknown plan selection mode: {mode}")


def _stack_stats(paths: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.stack(paths, axis=0)  # [N, P, 2]
    mean_xy = np.mean(arr, axis=0)
    std_xy = np.std(arr, axis=0, ddof=0)
    std_r = np.sqrt(np.sum(std_xy ** 2, axis=1))
    return mean_xy, std_xy, std_r


def _sigma_match(row_sigma: str, sigma_arg: str) -> bool:
    rs = (row_sigma or "").strip()
    sa = (sigma_arg or "").strip()
    if sa == "":
        return rs == ""
    try:
        return abs(float(rs) - float(sa)) < 1e-9
    except ValueError:
        return False


def build_plot(
    runs_csv: Path,
    regime: str,
    planner: str,
    task: str,
    sigma_pix: str,
    success_threshold: float,
    n_points: int,
    plan_selection: str,
    output: Optional[Path],
) -> Path:
    rows = _read_csv(runs_csv)
    subset = [
        r for r in rows
        if (r.get("regime", "").strip().upper() == regime.upper())
        and (r.get("planner", "").strip().lower() == planner.lower())
        and (r.get("task", "").strip() == task)
        and _sigma_match(r.get("sigma_pix", ""), sigma_pix)
        and r.get("run_dir", "").strip()
    ]
    if not subset:
        raise RuntimeError("No matching runs found in runs CSV.")

    inferred_paths: List[np.ndarray] = []
    planned_paths: List[np.ndarray] = []
    start_points: List[np.ndarray] = []
    goal_points: List[np.ndarray] = []
    used_runs: List[str] = []
    planned_run_names: List[str] = []

    for rec in subset:
        run_dir = Path(rec["run_dir"]).resolve()
        exp_csv = run_dir / "experiment.csv"
        if not exp_csv.exists():
            continue
        exp_rows = _read_csv(exp_csv)
        goal_rows = _goal_rows(exp_rows)
        if len(goal_rows) < 2:
            continue

        x = np.asarray([_f(r, "x") for r in goal_rows], dtype=float)
        y = np.asarray([_f(r, "y") for r in goal_rows], dtype=float)
        path_rs = _resample_path_xy(x, y, n_points)
        if path_rs is None:
            continue
        inferred_paths.append(path_rs)
        used_runs.append(run_dir.name)

        start_points.append(np.asarray([x[0], y[0]], dtype=float))
        goal_points.append(np.asarray([_f(goal_rows[0], "goal_x"), _f(goal_rows[0], "goal_y")], dtype=float))

        goal_start_stamp = _f(goal_rows[0], "stamp")
        plan_snaps = _load_plan_snapshots(run_dir)
        if plan_snaps:
            selected = _select_plan_snapshot(plan_snaps, goal_start_stamp, plan_selection)
            if selected is not None:
                p_rs = _resample_path_xy(selected[:, 0], selected[:, 1], n_points)
                if p_rs is not None:
                    planned_paths.append(p_rs)
                    planned_run_names.append(run_dir.name)

    if not inferred_paths:
        raise RuntimeError("No valid inferred trajectories found.")

    mean_inf, std_inf_xy, std_inf_r = _stack_stats(inferred_paths)
    mean_plan = std_plan_xy = std_plan_r = None
    if planned_paths:
        mean_plan, std_plan_xy, std_plan_r = _stack_stats(planned_paths)

    start_mean = np.mean(np.stack(start_points, axis=0), axis=0)
    goal_mean = np.mean(np.stack(goal_points, axis=0), axis=0)

    fig = plt.figure(figsize=(12.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_sd = fig.add_subplot(gs[0, 1])

    # XY panel
    for p in inferred_paths:
        ax_xy.plot(p[:, 0], p[:, 1], color="#94a3b8", lw=1.0, alpha=0.35)
    ax_xy.plot(mean_inf[:, 0], mean_inf[:, 1], color="#0b3c5d", lw=2.5, label="mean inferred path")

    if mean_plan is not None:
        for p in planned_paths:
            ax_xy.plot(p[:, 0], p[:, 1], color="#f59e0b", lw=0.9, alpha=0.25, ls="--")
        ax_xy.plot(mean_plan[:, 0], mean_plan[:, 1], color="#d97706", lw=2.2, ls="--", label=f"mean planned path ({plan_selection})")

    # Std circles along mean path for readability
    mark_idxs = np.linspace(0, n_points - 1, min(10, n_points), dtype=int)
    for i in np.unique(mark_idxs):
        ax_xy.add_patch(plt.Circle((mean_inf[i, 0], mean_inf[i, 1]), std_inf_r[i], fill=False, color="#0b3c5d", alpha=0.25, lw=1.0))
        if mean_plan is not None and std_plan_r is not None:
            ax_xy.add_patch(plt.Circle((mean_plan[i, 0], mean_plan[i, 1]), std_plan_r[i], fill=False, color="#d97706", alpha=0.20, lw=1.0, ls="--"))

    ax_xy.scatter([start_mean[0]], [start_mean[1]], c="#16a34a", s=55, marker="o", label="mean start", zorder=4)
    ax_xy.scatter([goal_mean[0]], [goal_mean[1]], c="#ea580c", s=110, marker="*", label="mean goal", zorder=5)
    ax_xy.add_patch(plt.Circle((goal_mean[0], goal_mean[1]), success_threshold, color="#ea580c", fill=False, ls="--", lw=1.4, alpha=0.7))
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Mean path with path spread (arc-length aligned)")
    ax_xy.legend(loc="best", fontsize=9, frameon=True)

    # Std-vs-progress panel
    prog = np.linspace(0.0, 1.0, n_points)
    ax_sd.plot(prog, std_inf_r, color="#0b3c5d", lw=2.0, label="inferred path std (radial)")
    if mean_plan is not None and std_plan_r is not None:
        ax_sd.plot(prog, std_plan_r, color="#d97706", lw=2.0, ls="--", label="planned path std (radial)")
    ax_sd.set_xlabel("normalized path progress")
    ax_sd.set_ylabel("path std [m]")
    ax_sd.grid(True, alpha=0.25)
    ax_sd.set_title("Across-run path variability")
    ax_sd.legend(loc="best", fontsize=9, frameon=True)

    sigma_label = sigma_pix if sigma_pix.strip() else "NA"
    fig.suptitle(
        f"{planner.upper()} | regime={regime.upper()} | task={task} | sigma_pix={sigma_label} | "
        f"n_runs={len(inferred_paths)} | n_plans={len(planned_paths)}",
        fontsize=13,
    )

    if output is None:
        stem = f"path_stats_{regime.upper()}_{planner.lower()}_{task}"
        if sigma_pix.strip():
            stem += f"_pix{sigma_pix.strip()}"
        output = runs_csv.with_name(stem + ".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot mean/std inferred and planned paths for one batch condition.")
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--regime", required=True, help="A, B, or C")
    ap.add_argument("--planner", required=True, help="efe1, efe2, efer, mpc")
    ap.add_argument("--task", required=True)
    ap.add_argument("--sigma-pix", default="", help="Required for regime B, leave empty for A/C")
    ap.add_argument("--success-threshold", type=float, default=0.35)
    ap.add_argument("--n-points", type=int, default=100)
    ap.add_argument("--plan-selection", choices=["first_after_goal", "longest", "last"], default="first_after_goal")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    out = build_plot(
        runs_csv=Path(args.runs_csv).resolve(),
        regime=args.regime,
        planner=args.planner,
        task=args.task,
        sigma_pix=args.sigma_pix,
        success_threshold=float(args.success_threshold),
        n_points=int(args.n_points),
        plan_selection=args.plan_selection,
        output=Path(args.output).resolve() if args.output else None,
    )
    print(f"wrote_png={out}")
    print(f"wrote_pdf={out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
