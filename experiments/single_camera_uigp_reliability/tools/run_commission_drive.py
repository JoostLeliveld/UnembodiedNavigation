#!/usr/bin/env python3
"""Orchestrate one single-camera commissioning coverage capture in ``warehouse_aws``.

Launches the goal-less navigation stack (``enable_mission:=false`` so the EFE
planner stays silent but its belief EKF + logger keep running), waits for the
run to start logging, drives the serpentine coverage path with
``commission_coverage_drive.py`` on ``/cmd_vel_raw``, then tears the launch down.

Nothing about the belief EKF is changed: every calibration / detector /
noise-model launch arg is taken EXACTLY as ``_build_launch_cmd`` produced it
(changing them would corrupt the belief the GP is trained on). Only the mission
toggles are overridden (``enable_mission``, ``auto_stop_on_goal``,
``stuck_window_s``, ``run_timeout_after_first_cmd_s``, ``headless``).

Output layout is the 4-level ``route/condition/seed`` tree
(``coverage/commission/seed0``) required by ``build_belief_gp_events.py``'s
``*/*/*/experiment_*`` glob.

The caller sources the ROS workspace; this script does not source. Example:

    source install/setup.bash
    python3 run_commission_drive.py --passes 2
    python3 run_commission_drive.py --dry-run     # prints cmds, no processes
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_CAMPAIGN = REPO_ROOT / "scripts/visibility_comparison/run_visibility_campaign.py"
_COVERAGE_DRIVER = (
    REPO_ROOT
    / "experiments/single_camera_uigp_reliability/tools/commission_coverage_drive.py"
)

_DEFAULT_CONFIG = "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
_DEFAULT_LOG_ROOT = "logs/visibility_comparison/single_cam_commissioning_v1"

# The route=coverage / condition=commission / seed=seed0 names satisfy the
# build_belief_gp_events.py glob '*/*/*/experiment_*' (campaign_dir being the
# --log-root). Frozen here so the offline events builder finds the run.
_ROUTE_DIR = "coverage"
_CONDITION_DIR = "commission"
_SEED_DIR = "seed0"

# The launch task whose spawn (3.3, -1.0, yaw 0) the coverage waypoints assume.
_LAUNCH_TASK = "route_apron_to_a3_mid"
_LAUNCH_CONDITION = "C1"
_LAUNCH_SEED = 0

_DRIVER_SPAWN_X = 3.3
_DRIVER_SPAWN_Y = -1.0
_DRIVER_SPAWN_YAW = 0.0
_DRIVER_SPEED_MPS = 0.18          # gentler than v_max (0.22): less momentum carry at corners
_DRIVER_ARRIVAL_RADIUS_M = 0.12   # tighter than default 0.20: less corner overshoot toward walls
_DRIVER_COMMAND_TOPIC = "/cmd_vel_raw"
_DRIVER_ODOM_TOPIC = "/odom"


def _load_campaign_module():
    """Import run_visibility_campaign.py (a script, not a package) by file path.

    It has no ROS imports, so this succeeds without a sourced workspace.
    """
    if not _RUN_CAMPAIGN.is_file():
        raise RuntimeError(f"Cannot reuse campaign helpers: {_RUN_CAMPAIGN} missing.")
    name = "run_visibility_campaign"
    spec = importlib.util.spec_from_file_location(name, _RUN_CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_campaign = _load_campaign_module()
# Reused verbatim — do NOT duplicate the config loader, arg builder, domain map,
# or the zombie-clearing routine.
_load_config = _campaign._load_config
_build_launch_cmd = _campaign._build_launch_cmd
_force_fresh = _campaign._force_fresh
_ros_domain_for_run = _campaign._ros_domain_for_run


def _set_launch_arg(cmd: list[str], key: str, value: str) -> None:
    """Replace an existing ``key:=...`` token in-place, or append if absent.

    Guarantees the launch arg appears exactly once (ros2 launch takes the last
    occurrence, but a single token keeps the command unambiguous).
    """
    token = f"{key}:={value}"
    prefix = f"{key}:="
    for idx, existing in enumerate(cmd):
        if existing.startswith(prefix):
            cmd[idx] = token
            # Drop any later duplicates of the same key.
            cmd[:] = [c for j, c in enumerate(cmd) if j == idx or not c.startswith(prefix)]
            return
    cmd.append(token)


def _build_final_launch_cmd(cfg: dict, log_dir: Path, drive_timeout_s: float) -> list[str]:
    """Base campaign launch cmd with ONLY the mission toggles overridden."""
    cmd = _build_launch_cmd(
        cfg,
        task_name=_LAUNCH_TASK,
        condition_id=_LAUNCH_CONDITION,
        seed=_LAUNCH_SEED,
        log_dir=log_dir,
    )
    # Goal-less commissioning: planner silent, no auto-stop/stuck termination,
    # generous first-cmd timeout, headless. Everything else (calibration,
    # detector, noise, EKF) stays exactly as _build_launch_cmd produced it.
    _set_launch_arg(cmd, "enable_mission", "false")
    _set_launch_arg(cmd, "auto_stop_on_goal", "false")
    _set_launch_arg(cmd, "stuck_window_s", "0.0")
    _set_launch_arg(cmd, "run_timeout_after_first_cmd_s", str(drive_timeout_s))
    _set_launch_arg(cmd, "headless", "true")
    # A commissioning coverage drive must not abort the whole capture on a soft
    # geometry-breach near-miss (the logger is a required launch node -> its exit
    # shuts everything down). Physical contact still terminates regardless; the
    # conservative coverage path keeps the robot off the walls to avoid that.
    _set_launch_arg(cmd, "terminate_on_geom_collision", "false")
    return cmd


def _build_driver_cmd(log_dir: Path, passes: int, drive_budget_s: float) -> list[str]:
    return [
        sys.executable, str(_COVERAGE_DRIVER),
        "--command-topic", _DRIVER_COMMAND_TOPIC,
        "--odom-topic", _DRIVER_ODOM_TOPIC,
        "--spawn-x", str(_DRIVER_SPAWN_X),
        "--spawn-y", str(_DRIVER_SPAWN_Y),
        "--spawn-yaw", str(_DRIVER_SPAWN_YAW),
        "--passes", str(passes),
        "--speed-mps", str(_DRIVER_SPEED_MPS),
        "--arrival-radius-m", str(_DRIVER_ARRIVAL_RADIUS_M),
        "--max-sim-runtime-s", str(drive_budget_s),
        "--manifest-path", str(log_dir / "coverage_manifest.json"),
    ]


def _experiment_csv_rows(log_dir: Path) -> tuple[Path | None, int]:
    """Return the newest experiment_*/experiment.csv under log_dir and its data-row count."""
    best_dir: Path | None = None
    for exp_dir in sorted(log_dir.glob("experiment_*"), reverse=True):
        csv_path = exp_dir / "experiment.csv"
        if csv_path.is_file():
            best_dir = exp_dir
            try:
                with csv_path.open("r", encoding="utf-8") as handle:
                    lines = sum(1 for _ in handle)
            except OSError:
                lines = 0
            return csv_path, max(0, lines - 1)  # minus header
    return best_dir, 0


def _wait_for_ready(
    log_dir: Path, launch_proc: subprocess.Popen, *, timeout_s: float, min_rows: int
) -> bool:
    """Poll until the run's experiment.csv appears and has grown a couple of rows."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if launch_proc.poll() is not None:
            print(f"  ERROR: launch process exited early (code {launch_proc.returncode}) "
                  f"before the run started logging.")
            return False
        csv_path, rows = _experiment_csv_rows(log_dir)
        if csv_path is not None and rows >= min_rows:
            print(f"  ready: {csv_path} has {rows} logged rows.")
            return True
        time.sleep(2.0)
    print(f"  ERROR: run did not start logging within {timeout_s:.0f}s.")
    return False


def _terminate_group(pgid: int | None, *, grace_s: float = 5.0) -> None:
    if pgid is None:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_DEFAULT_CONFIG,
                        help="Campaign config YAML (defines the belief EKF / detector args).")
    parser.add_argument("--log-root", default=_DEFAULT_LOG_ROOT,
                        help="Commissioning capture root (campaign_dir for the events builder).")
    parser.add_argument("--passes", type=int, default=1,
                        help="Coverage passes; odd passes reverse direction.")
    parser.add_argument("--drive-timeout-s", type=float, default=1200.0,
                        help="Sim-time budget for the drive; also the launch "
                             "run_timeout_after_first_cmd_s.")
    parser.add_argument("--readiness-timeout-s", type=float, default=180.0,
                        help="Max wall seconds to wait for the run to start logging.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the final launch args + driver command, then exit.")
    args = parser.parse_args()

    if args.passes < 1:
        parser.error("--passes must be >= 1")

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1
    cfg = _load_config(config_path)

    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = REPO_ROOT / log_root
    log_root = log_root.resolve()
    log_dir = log_root / _ROUTE_DIR / _CONDITION_DIR / _SEED_DIR

    launch_cmd = _build_final_launch_cmd(cfg, log_dir, args.drive_timeout_s)
    driver_cmd = _build_driver_cmd(log_dir, int(args.passes), args.drive_timeout_s)
    ros_domain_id = _ros_domain_for_run(cfg, 0)
    if ros_domain_id is None and "ros_domain_id_base" in cfg:
        ros_domain_id = str(int(cfg["ros_domain_id_base"]))

    print(f"Config    : {config_path}")
    print(f"Log root  : {log_root}")
    print(f"Run dir   : {log_dir}")
    print(f"ROS domain: {ros_domain_id}")
    print("LAUNCH CMD:", " ".join(str(p) for p in launch_cmd))
    print("DRIVER CMD:", " ".join(str(p) for p in driver_cmd))

    if args.dry_run:
        print("\n=== DRY RUN — no processes started ===")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    run_env = dict(os.environ)
    if ros_domain_id is not None:
        run_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    run_env.setdefault("ROS_LOG_DIR", str(log_root / "_ros_logs"))
    Path(run_env["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    # Clear any zombie sim/bridge/node stragglers before launching (reused helper).
    print("Clearing stragglers before launch...")
    _force_fresh()

    launch_proc = subprocess.Popen(launch_cmd, start_new_session=True, env=run_env)
    try:
        launch_pgid: int | None = os.getpgid(launch_proc.pid)
    except ProcessLookupError:
        launch_pgid = None

    driver_rc = 2
    try:
        print("Waiting for the run to start logging...")
        if not _wait_for_ready(
            log_dir, launch_proc,
            timeout_s=args.readiness_timeout_s, min_rows=2,
        ):
            return 1

        print("Starting coverage driver...")
        driver_proc = subprocess.Popen(driver_cmd, env=run_env)
        # The driver self-terminates via its own sim/wall deadman; this wall
        # backstop only guards against a hung driver.
        wall_backstop_s = args.drive_timeout_s * 6.0 + 300.0
        try:
            driver_rc = driver_proc.wait(timeout=wall_backstop_s)
        except subprocess.TimeoutExpired:
            print(f"  WARN: driver exceeded {wall_backstop_s:.0f}s wall backstop — killing.")
            driver_proc.kill()
            driver_proc.wait()
            driver_rc = 2
        print(f"Coverage driver exited (code {driver_rc}).")
    finally:
        print("Tearing down launch...")
        _terminate_group(launch_pgid)
        _force_fresh()

    print("\n=== commissioning capture complete ===")
    print(f"  run dir        : {log_dir}")
    print(f"  experiment logs: {log_dir}/experiment_*/ (experiment.csv + perception.csv)")
    print(f"  drive manifest : {log_dir / 'coverage_manifest.json'}")
    print(f"  next: python3 scripts/visibility_comparison/build_belief_gp_events.py "
          f"--campaign {log_root} --out {log_root / 'belief_gp_events'}")
    return 0 if driver_rc == 0 else driver_rc


if __name__ == "__main__":
    raise SystemExit(main())
