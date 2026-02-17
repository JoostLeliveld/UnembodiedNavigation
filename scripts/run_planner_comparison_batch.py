#!/usr/bin/env python3
"""Run comparative EFE/MPC benchmark batches and export summary CSVs.

This script executes `ros2 launch experiments boundary_only.launch.py` for a
cross-product of planners, tasks, and seeds, then summarizes metrics from each
run's `experiment.csv`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import pathlib
import signal
import statistics
import subprocess
import time
from itertools import product
from typing import Dict, Iterable, List, Optional


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _kill_gazebo() -> None:
    out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    pids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, cmd = parts
        cmd = cmd.strip()
        if cmd.startswith("ign gazebo server") or cmd.startswith("ign gazebo gui"):
            try:
                pids.append(int(pid_s))
            except ValueError:
                pass

    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(1.5)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _list_runs(log_root: pathlib.Path) -> set[pathlib.Path]:
    return {p.resolve() for p in log_root.glob("experiment_*") if p.is_dir()}


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _goal_valid_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if abs(_f(row, "goal_x")) > 1e-9 or abs(_f(row, "goal_y")) > 1e-9
    ]


def _parse_metrics(csv_path: pathlib.Path, success_threshold: float) -> Optional[Dict[str, float]]:
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None

    rows = _goal_valid_rows(rows)
    if not rows:
        return None

    stamps = [_f(r, "stamp") for r in rows]
    gds = [_f(r, "goal_dist") for r in rows]
    plan_lengths = [_f(r, "plan_length") for r in rows]
    efe_tot = [_f(r, "efe_total") for r in rows]
    efe_risk = [_f(r, "efe_risk") for r in rows]
    efe_amb = [_f(r, "efe_ambiguity") for r in rows]
    cmd_v = [_f(r, "cmd_v") for r in rows]
    cmd_w = [_f(r, "cmd_w") for r in rows]

    min_goal = min(gds)
    final_goal = gds[-1]
    start_goal = gds[0]
    success = int(min_goal <= success_threshold)
    time_to_goal: Optional[float] = None
    if success:
        for s, gd in zip(stamps, gds):
            if gd <= success_threshold:
                time_to_goal = s - stamps[0]
                break

    dist_reduction = start_goal - min_goal
    dist_reduction_frac = dist_reduction / start_goal if start_goal > 1e-9 else 0.0

    return {
        "samples": len(rows),
        "duration_s": max(0.0, stamps[-1] - stamps[0]),
        "start_goal_dist": start_goal,
        "min_goal_dist": min_goal,
        "final_goal_dist": final_goal,
        "success": success,
        "time_to_goal_s": time_to_goal if time_to_goal is not None else "",
        "dist_reduction": dist_reduction,
        "dist_reduction_frac": dist_reduction_frac,
        "final_plan_length": plan_lengths[-1],
        "mean_efe_total": statistics.fmean(efe_tot) if efe_tot else 0.0,
        "mean_efe_risk": statistics.fmean(efe_risk) if efe_risk else 0.0,
        "mean_efe_ambiguity": statistics.fmean(efe_amb) if efe_amb else 0.0,
        "mean_cmd_v": statistics.fmean(cmd_v) if cmd_v else 0.0,
        "mean_abs_cmd_w": statistics.fmean([abs(x) for x in cmd_w]) if cmd_w else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run comparative planner benchmark batch.")
    parser.add_argument("--world", default="arena10.world.sdf")
    parser.add_argument("--tasks", default="T1_edge_hug,T2_diag_far,T3_corners")
    parser.add_argument("--planners", default="efe1,efe2,mpc,efer")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--timeout-s", type=int, default=55)
    parser.add_argument("--success-threshold", type=float, default=0.35)
    parser.add_argument("--state-source", default="pixel")
    parser.add_argument("--perception-backend", default="homography")
    parser.add_argument("--use-pixel-correction", default="true")
    parser.add_argument("--obs-mode", default="uv")
    parser.add_argument("--optimizer-backend", default="scipy")
    parser.add_argument("--optimizer-maxiter", default="50")
    parser.add_argument("--optimizer-gtol", default="1e-4")
    parser.add_argument("--optimizer-warm-start", default="true")
    parser.add_argument("--boundary-weight", default="1.0")
    parser.add_argument("--use-obs-risk", default="true")
    parser.add_argument("--add-ambiguity", default="true")
    parser.add_argument("--use-ambiguity", default="true")
    parser.add_argument("--process-noise-xy", default="")
    parser.add_argument("--process-noise-theta", default="")
    parser.add_argument("--obs-noise-uv", default="")
    parser.add_argument("--obs-noise-yaw", default="")
    parser.add_argument("--use-rviz", default="false")
    parser.add_argument("--ros-log-dir", default="/tmp/ros_log_batch")
    parser.add_argument("--log-dir", default="logs/experiments")
    parser.add_argument("--tmp-log-dir", default="/tmp/compare_batch_logs")
    args = parser.parse_args()

    root = pathlib.Path(".").resolve()
    log_root = (root / args.log_dir).resolve()
    tmp_log_root = pathlib.Path(args.tmp_log_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    tmp_log_root.mkdir(parents=True, exist_ok=True)

    tasks = _parse_list(args.tasks)
    planners = _parse_list(args.planners)
    seeds = [int(x) for x in _parse_list(args.seeds)]
    combos = list(product(seeds, tasks, planners))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_csv = log_root / f"comparison_{stamp}_runs.csv"
    agg_csv = log_root / f"comparison_{stamp}_aggregate.csv"

    print(f"Running {len(combos)} runs...")
    records: List[Dict[str, object]] = []

    for i, (seed, task, planner) in enumerate(combos, start=1):
        _kill_gazebo()
        before = _list_runs(log_root)
        run_name = f"{i:02d}_{planner}_{task}_s{seed}"
        log_path = tmp_log_root / f"{run_name}.log"

        noise_args = ""
        if str(args.process_noise_xy).strip():
            noise_args += f" process_noise_xy:={args.process_noise_xy}"
        if str(args.process_noise_theta).strip():
            noise_args += f" process_noise_theta:={args.process_noise_theta}"
        if str(args.obs_noise_uv).strip():
            noise_args += f" obs_noise_uv:={args.obs_noise_uv}"
        if str(args.obs_noise_yaw).strip():
            noise_args += f" obs_noise_yaw:={args.obs_noise_yaw}"

        cmd = (
            f"mkdir -p {args.ros_log_dir} && "
            f"export ROS_LOG_DIR={args.ros_log_dir} && "
            "source install/setup.bash && "
            f"timeout {args.timeout_s} ros2 launch experiments boundary_only.launch.py "
            f"world:={args.world} task:={task} planner:={planner} "
            f"state_source:={args.state_source} "
            f"perception_backend:={args.perception_backend} "
            f"use_pixel_correction:={args.use_pixel_correction} pixel_timeout_s:=0.5 "
            f"obs_mode:={args.obs_mode} optimizer_backend:={args.optimizer_backend} "
            f"optimizer_maxiter:={args.optimizer_maxiter} optimizer_gtol:={args.optimizer_gtol} "
            f"optimizer_warm_start:={args.optimizer_warm_start} "
            f"boundary_weight:={args.boundary_weight} use_obs_risk:={args.use_obs_risk} "
            f"add_ambiguity:={args.add_ambiguity} use_ambiguity:={args.use_ambiguity} "
            f"seed:={seed} use_rviz:={args.use_rviz}"
            f"{noise_args}"
        )

        print(f"[{i:02d}/{len(combos)}] planner={planner} task={task} seed={seed}")
        with log_path.open("w") as lf:
            proc = subprocess.run(["bash", "-lc", cmd], stdout=lf, stderr=subprocess.STDOUT)

        _kill_gazebo()
        after = _list_runs(log_root)
        new_runs = sorted(after - before, key=lambda p: p.stat().st_mtime)
        run_dir = new_runs[-1] if new_runs else None

        rec: Dict[str, object] = {
            "planner": planner,
            "task": task,
            "seed": seed,
            "return_code": proc.returncode,
            "run_dir": str(run_dir) if run_dir else "",
            "log_path": str(log_path),
        }

        if run_dir:
            metrics = _parse_metrics(run_dir / "experiment.csv", args.success_threshold)
            if metrics:
                rec.update(metrics)

        records.append(rec)
        print(
            f"    -> rc={proc.returncode}, run_dir={run_dir.name if run_dir else 'NONE'}, "
            f"min_goal={rec.get('min_goal_dist', 'NA')}"
        )

    run_fields = [
        "planner",
        "task",
        "seed",
        "return_code",
        "run_dir",
        "log_path",
        "samples",
        "duration_s",
        "start_goal_dist",
        "min_goal_dist",
        "final_goal_dist",
        "success",
        "time_to_goal_s",
        "dist_reduction",
        "dist_reduction_frac",
        "final_plan_length",
        "mean_efe_total",
        "mean_efe_risk",
        "mean_efe_ambiguity",
        "mean_cmd_v",
        "mean_abs_cmd_w",
    ]
    with run_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for r in records:
            w.writerow(r)

    agg_rows: List[Dict[str, object]] = []
    for task in tasks:
        for planner in planners:
            rs = [
                r
                for r in records
                if r.get("task") == task
                and r.get("planner") == planner
                and r.get("min_goal_dist") is not None
            ]
            if not rs:
                continue

            def _vals(key: str) -> List[float]:
                out: List[float] = []
                for r in rs:
                    val = r.get(key)
                    if val in ("", None):
                        continue
                    out.append(float(val))
                return out

            start = _vals("start_goal_dist")
            min_goal = _vals("min_goal_dist")
            final_goal = _vals("final_goal_dist")
            red = _vals("dist_reduction")
            red_frac = _vals("dist_reduction_frac")
            success = _vals("success")
            t_goal = _vals("time_to_goal_s")
            cmd_v = _vals("mean_cmd_v")
            cmd_w = _vals("mean_abs_cmd_w")
            amb = _vals("mean_efe_ambiguity")

            agg_rows.append(
                {
                    "task": task,
                    "planner": planner,
                    "n_runs": len(rs),
                    "success_rate": statistics.fmean(success) if success else 0.0,
                    "start_goal_dist_mean": statistics.fmean(start) if start else 0.0,
                    "min_goal_dist_mean": statistics.fmean(min_goal) if min_goal else 0.0,
                    "min_goal_dist_std": statistics.pstdev(min_goal) if len(min_goal) > 1 else 0.0,
                    "final_goal_dist_mean": statistics.fmean(final_goal) if final_goal else 0.0,
                    "dist_reduction_mean": statistics.fmean(red) if red else 0.0,
                    "dist_reduction_frac_mean": statistics.fmean(red_frac) if red_frac else 0.0,
                    "dist_reduction_frac_std": statistics.pstdev(red_frac) if len(red_frac) > 1 else 0.0,
                    "time_to_goal_mean": statistics.fmean(t_goal) if t_goal else "",
                    "time_to_goal_std": statistics.pstdev(t_goal) if len(t_goal) > 1 else "",
                    "mean_cmd_v": statistics.fmean(cmd_v) if cmd_v else 0.0,
                    "mean_abs_cmd_w": statistics.fmean(cmd_w) if cmd_w else 0.0,
                    "mean_efe_ambiguity": statistics.fmean(amb) if amb else 0.0,
                }
            )

    agg_fields = list(agg_rows[0].keys()) if agg_rows else []
    with agg_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    print("Per-run summary:", run_csv)
    print("Aggregate summary:", agg_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
