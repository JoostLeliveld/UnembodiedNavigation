#!/usr/bin/env python3
"""Run investigative EFE/MPC agent batches across A/B/C observation regimes.

Regimes:
- A: true oracle state mode (/odom -> TF -> /state/bev), no pixel correction
- B: homography observation with injected sensor pixel noise sweep
- C: end-to-end ArUco observation (uvt) with fixed calibrated model noise

This script is scoped to the paper's no-costmap, agent-mode study. It launches
the paper wrapper `investigative_agent.launch.py`, which fixes JAX/no-costmap
assumptions and exposes only study-facing arguments.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for item in _parse_list(raw):
        out.append(float(item))
    return out


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
    except (TypeError, ValueError):
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


def _planner_flags(planner: str) -> Tuple[str, str]:
    planner = planner.strip().lower()
    if planner in ("efe1", "efe2"):
        return ("true", "true")
    if planner in ("efer", "mpc"):
        return ("false", "true")
    raise ValueError(f"Unknown planner '{planner}'. Expected efe1|efe2|efer|mpc.")


def _regime_specific_args(
    regime: str,
    sigma_pix: Optional[float],
    args: argparse.Namespace,
) -> Dict[str, str]:
    regime = regime.upper().strip()
    if regime == "A":
        # True oracle state mode: /odom is transformed to map_bev and forwarded to /state/bev.
        return {
            "state_source": "oracle",
            "perception_backend": "homography",
            "obs_mode": args.obs_mode_oracle,
            "use_pixel_correction": "false",
            "sensor_pixel_noise_sigma": "0.0",
            "obs_noise_uv": f"{args.obs_noise_uv_model}",
            "obs_noise_yaw": f"{args.obs_noise_yaw_model}",
            "min_state_cov": f"{args.homography_min_state_cov}",
        }
    if regime == "B":
        if sigma_pix is None:
            raise ValueError("Regime B requires sigma_pix.")
        return {
            "state_source": "pixel",
            "perception_backend": "homography",
            "obs_mode": args.obs_mode_homography,
            "use_pixel_correction": "true",
            # Paper launch keeps state-node noise off; inject only at measurement source.
            "sensor_pixel_noise_sigma": f"{sigma_pix}",
            # Fixed model-noise assumption across sweep.
            "obs_noise_uv": f"{args.obs_noise_uv_model}",
            "obs_noise_yaw": f"{args.obs_noise_yaw_model}",
            "min_state_cov": f"{args.homography_min_state_cov}",
        }
    if regime == "C":
        return {
            "state_source": "pixel",
            "perception_backend": "aruco",
            "obs_mode": args.obs_mode_aruco,
            "use_pixel_correction": "true",
            "sensor_pixel_noise_sigma": "0.0",
            "obs_noise_uv": f"{args.obs_noise_uv_model}",
            "obs_noise_yaw": f"{args.obs_noise_yaw_model}",
            "min_state_cov": f"{args.aruco_min_state_cov}",
        }
    raise ValueError(f"Unknown regime '{regime}'. Expected A|B|C.")


def _combo_rows(
    regimes: Sequence[str],
    tasks: Sequence[str],
    planners: Sequence[str],
    seeds: Sequence[int],
    sigma_levels: Sequence[float],
) -> List[Dict[str, object]]:
    combos: List[Dict[str, object]] = []
    for regime in regimes:
        r = regime.upper().strip()
        if r == "B":
            for task, planner, seed, sigma in product(tasks, planners, seeds, sigma_levels):
                combos.append(
                    {
                        "regime": r,
                        "task": task,
                        "planner": planner,
                        "seed": seed,
                        "sigma_pix": float(sigma),
                    }
                )
        else:
            for task, planner, seed in product(tasks, planners, seeds):
                combos.append(
                    {
                        "regime": r,
                        "task": task,
                        "planner": planner,
                        "seed": seed,
                        "sigma_pix": "",
                    }
                )
    return combos


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the investigative 3-regime x 4-planner agent study."
    )
    parser.add_argument("--world", default="empty_notebook.world.sdf")
    parser.add_argument(
        "--tasks",
        default="M1_short_direct,M2_long_diagonal,M3_projective_stress",
    )
    parser.add_argument("--planners", default="efe1,efe2,efer,mpc")
    parser.add_argument("--regimes", default="A,B,C")
    parser.add_argument("--pixel-noise-levels", default="1,4,8")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--timeout-s", type=int, default=55)
    parser.add_argument("--success-threshold", type=float, default=0.35)

    parser.add_argument("--dt", type=float, default=0.2)
    # Defaults mirror `investigative_agent.launch.py` unless intentionally overridden.
    parser.add_argument("--plan-rate", type=float, default=2.0)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--control-weight", type=float, default=0.1)
    parser.add_argument("--risk-weight-state", type=float, default=0.0)
    parser.add_argument("--risk-weight-obs", type=float, default=1.0)
    parser.add_argument("--ambiguity-weight", type=float, default=1.0)
    parser.add_argument("--goal-sigma-yaw", type=float, default=100.0)

    parser.add_argument("--process-noise-xy", type=float, default=0.00025629572291239897)
    parser.add_argument("--process-noise-theta", type=float, default=0.0038065154008503684)
    parser.add_argument("--obs-noise-uv-model", type=float, default=4.252440636699618)
    parser.add_argument("--obs-noise-yaw-model", type=float, default=0.8275084392611833)

    parser.add_argument("--obs-mode-oracle", default="uv")
    parser.add_argument("--obs-mode-homography", default="uv")
    parser.add_argument("--obs-mode-aruco", default="uvt")
    parser.add_argument("--homography-min-state-cov", type=float, default=1e-6)
    parser.add_argument("--aruco-min-state-cov", type=float, default=0.0025)
    parser.add_argument("--pixel-timeout-s", type=float, default=0.5)

    parser.add_argument("--optimizer-maxiter", type=int, default=50)
    parser.add_argument("--optimizer-gtol", type=float, default=1e-4)
    parser.add_argument("--optimizer-warm-start", default="true")

    parser.add_argument("--aruco-dict", default="DICT_4X4_50")
    parser.add_argument("--target-marker-id", type=int, default=0)
    parser.add_argument("--publish-yaw-from-marker", default="true")

    parser.add_argument("--use-rviz", default="false")
    parser.add_argument("--ros-log-dir", default="/tmp/ros_log_investigative")
    parser.add_argument("--log-dir", default="logs/experiments")
    parser.add_argument("--tmp-log-dir", default="/tmp/investigative_batch_logs")
    args = parser.parse_args()

    root = pathlib.Path(".").resolve()
    log_root = (root / args.log_dir).resolve()
    tmp_log_root = pathlib.Path(args.tmp_log_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    tmp_log_root.mkdir(parents=True, exist_ok=True)

    tasks = _parse_list(args.tasks)
    planners = _parse_list(args.planners)
    regimes = [r.upper() for r in _parse_list(args.regimes)]
    seeds = [int(x) for x in _parse_list(args.seeds)]
    sigma_levels = _parse_float_list(args.pixel_noise_levels)

    combos = _combo_rows(regimes, tasks, planners, seeds, sigma_levels)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_csv = log_root / f"investigative_{stamp}_runs.csv"
    agg_csv = log_root / f"investigative_{stamp}_aggregate.csv"
    launch_file = "investigative_agent.launch.py"

    print(f"Running {len(combos)} runs...")
    records: List[Dict[str, object]] = []
    for i, combo in enumerate(combos, start=1):
        regime = str(combo["regime"])
        task = str(combo["task"])
        planner = str(combo["planner"])
        seed = int(combo["seed"])
        sigma_pix = combo["sigma_pix"]
        sigma_for_regime: Optional[float] = None if sigma_pix == "" else float(sigma_pix)

        use_ambiguity, use_obs_risk = _planner_flags(planner)
        regime_args = _regime_specific_args(regime, sigma_for_regime, args)

        _kill_gazebo()
        before = _list_runs(log_root)
        run_name = f"{i:03d}_{regime}_{planner}_{task}_s{seed}"
        if sigma_for_regime is not None:
            run_name += f"_pix{sigma_for_regime:g}"
        log_path = tmp_log_root / f"{run_name}.log"

        cmd = (
            f"mkdir -p {args.ros_log_dir} && "
            f"export ROS_LOG_DIR={args.ros_log_dir} && "
            "source install/setup.bash && "
            f"timeout {args.timeout_s} ros2 launch experiments {launch_file} "
            f"world:={args.world} task:={task} planner:={planner} seed:={seed} "
            f"state_source:={regime_args['state_source']} "
            f"perception_backend:={regime_args['perception_backend']} "
            f"obs_mode:={regime_args['obs_mode']} "
            f"use_pixel_correction:={regime_args['use_pixel_correction']} "
            f"pixel_timeout_s:={args.pixel_timeout_s} "
            f"sensor_pixel_noise_sigma:={regime_args['sensor_pixel_noise_sigma']} "
            f"use_ambiguity:={use_ambiguity} use_obs_risk:={use_obs_risk} "
            f"plan_rate:={args.plan_rate} horizon:={args.horizon} dt:={args.dt} "
            f"control_weight:={args.control_weight} "
            f"risk_weight_state:={args.risk_weight_state} risk_weight_obs:={args.risk_weight_obs} "
            f"ambiguity_weight:={args.ambiguity_weight} "
            f"goal_sigma_yaw:={args.goal_sigma_yaw} "
            f"process_noise_xy:={args.process_noise_xy} process_noise_theta:={args.process_noise_theta} "
            f"obs_noise_uv:={regime_args['obs_noise_uv']} obs_noise_yaw:={regime_args['obs_noise_yaw']} "
            f"optimizer_maxiter:={args.optimizer_maxiter} "
            f"optimizer_gtol:={args.optimizer_gtol} optimizer_warm_start:={args.optimizer_warm_start} "
            f"min_state_cov:={regime_args['min_state_cov']} "
            f"aruco_dict:={args.aruco_dict} target_marker_id:={args.target_marker_id} "
            f"publish_yaw_from_marker:={args.publish_yaw_from_marker} "
            f"use_rviz:={args.use_rviz}"
        )

        print(
            f"[{i:03d}/{len(combos)}] regime={regime} planner={planner} task={task} "
            f"seed={seed} sigma_pix={sigma_for_regime if sigma_for_regime is not None else 'NA'}"
        )
        with log_path.open("w") as lf:
            proc = subprocess.run(["bash", "-lc", cmd], stdout=lf, stderr=subprocess.STDOUT)

        _kill_gazebo()
        after = _list_runs(log_root)
        new_runs = sorted(after - before, key=lambda p: p.stat().st_mtime)
        run_dir = new_runs[-1] if new_runs else None

        rec: Dict[str, object] = {
            "regime": regime,
            "sigma_pix": sigma_for_regime if sigma_for_regime is not None else "",
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
        "regime",
        "sigma_pix",
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
    keys = sorted({(r["regime"], r.get("sigma_pix", ""), r["task"], r["planner"]) for r in records})
    for regime, sigma_pix, task, planner in keys:
        rs = [
            r
            for r in records
            if r.get("regime") == regime
            and r.get("sigma_pix", "") == sigma_pix
            and r.get("task") == task
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

        min_goal = _vals("min_goal_dist")
        final_goal = _vals("final_goal_dist")
        red_frac = _vals("dist_reduction_frac")
        success = _vals("success")
        t_goal = _vals("time_to_goal_s")
        cmd_v = _vals("mean_cmd_v")
        cmd_w = _vals("mean_abs_cmd_w")
        amb = _vals("mean_efe_ambiguity")
        risk_v = _vals("mean_efe_risk")

        agg_rows.append(
            {
                "regime": regime,
                "sigma_pix": sigma_pix,
                "task": task,
                "planner": planner,
                "n_runs": len(rs),
                "success_rate": statistics.fmean(success) if success else 0.0,
                "min_goal_dist_mean": statistics.fmean(min_goal) if min_goal else 0.0,
                "min_goal_dist_std": statistics.pstdev(min_goal) if len(min_goal) > 1 else 0.0,
                "final_goal_dist_mean": statistics.fmean(final_goal) if final_goal else 0.0,
                "dist_reduction_frac_mean": statistics.fmean(red_frac) if red_frac else 0.0,
                "dist_reduction_frac_std": statistics.pstdev(red_frac) if len(red_frac) > 1 else 0.0,
                "time_to_goal_mean": statistics.fmean(t_goal) if t_goal else "",
                "time_to_goal_std": statistics.pstdev(t_goal) if len(t_goal) > 1 else "",
                "mean_cmd_v": statistics.fmean(cmd_v) if cmd_v else 0.0,
                "mean_abs_cmd_w": statistics.fmean(cmd_w) if cmd_w else 0.0,
                "mean_efe_risk": statistics.fmean(risk_v) if risk_v else 0.0,
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
