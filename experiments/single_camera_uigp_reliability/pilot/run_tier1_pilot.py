#!/usr/bin/env python3
"""Tier-1 detection-half pilot (single-camera, real Gazebo).

Validates the closed loop on REAL perception: a real warehouse_aws drive with a
mid-route CALIBRATION drift injected into the camera measurement, checking that
the health monitor goes HEALTHY -> (drift) -> DEGRADED as NIS/innovation climb.

Self-contained + ISOLATED (collision-safe): sets ROS_DOMAIN_ID + IGN/GZ_PARTITION
so it cannot see or touch any other ROS/Gazebo process; the faithful honest_v1
bringup runs via run_visibility_campaign (single run => the runner's broad pkill
is NOT triggered); the fault injector + health monitor are co-spawned in the same
domain; teardown kills ONLY this script's own child process groups.

  fault injector: /perception/pixel_pose --(calib drift)--> /perception/pixel_pose_faulted
  planner:        pixel_topic:=/perception/pixel_pose_faulted   (via the committed harness arg)
  health monitor: /planner/pixel_correction_diagnostics --> /reliability/localization_* + CSV

Run DETACHED so the launch is not hit by sibling-shell SIGINT:
  setsid python3 run_tier1_pilot.py > <out>/pilot.log 2>&1 < /dev/null &
then poll <out>/STATUS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "experiments" / "single_camera_uigp_reliability" / "tools"
INJECTOR = TOOLS / "single_cam_fault_injector.py"
HEALTH = TOOLS / "localization_health_node.py"
HONEST = REPO / "scripts" / "visibility_comparison" / "warehouse_visibility_campaign_honest_v1.yaml"
RUNNER = REPO / "scripts" / "visibility_comparison" / "run_visibility_campaign.py"

BUSY_PATTERNS = ("ign gazebo", "gz sim", "ruby", "parameter_bridge",
                 "unicycle_planner", "batched_four_camera", "yolo_robot_detector",
                 "audit_detection", "opportunity")


def machine_busy():
    hits = []
    for pat in BUSY_PATTERNS:
        try:
            r = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True, timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            continue
        for line in r.stdout.splitlines():
            if "run_tier1_pilot" in line or "pgrep" in line:
                continue
            hits.append(line.strip())
    return hits


def write_status(out: Path, text: str):
    (out / "STATUS.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(os.environ.get("PILOT_OUT", "")) if os.environ.get("PILOT_OUT") else ""))
    ap.add_argument("--domain", type=int, default=191)
    ap.add_argument("--task", default="route_apron_to_a3_mid")
    ap.add_argument("--condition", default="C2")
    ap.add_argument("--fault", default="calib_drift")
    ap.add_argument("--fault-after-s", type=float, default=18.0)
    ap.add_argument("--calib-lookat-drift", default="0.5,0.5,0.0")
    ap.add_argument("--drift-ramp-s", type=float, default=18.0)
    ap.add_argument("--run-timeout-after-first-cmd-s", type=float, default=150.0)
    ap.add_argument("--hard-timeout-s", type=float, default=600.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="generate the pilot config and run the runner's --dry-run (prints the launch CMD); no gazebo, no nodes")
    args = ap.parse_args()

    out = Path(args.out).resolve() if args.out else (REPO / "logs" / "studies" /
              "single_camera_uigp_reliability" / f"tier1_pilot_{int(time.time())}")
    out.mkdir(parents=True, exist_ok=True)
    health_csv = out / "health_trace.csv"
    log_root = out / "campaign"

    # 1) refuse to launch if the machine is not free (collision-safety)
    busy = machine_busy()
    if busy:
        write_status(out, "# Tier-1 pilot ABORTED\n\nMachine not free; refused to launch:\n\n"
                     + "\n".join(f"- `{b}`" for b in busy[:20]) + "\n")
        print("ABORT: machine busy\n" + "\n".join(busy))
        return 2

    # 2) faithful single-run pilot config derived from the locked honest_v1
    cfg = yaml.safe_load(HONEST.read_text())
    orig_task = cfg["tasks"].get(args.task, {})
    cfg["conditions"] = {args.condition: cfg["conditions"][args.condition]}
    cfg["tasks"] = {args.task: {"conditions": [args.condition], "seeds": [0],
                                **({"optimizer_initial_routes_json": orig_task["optimizer_initial_routes_json"]}
                                   if "optimizer_initial_routes_json" in orig_task else {})}}
    cfg["pixel_topic"] = "/perception/pixel_pose_faulted"
    cfg["cleanup_sim_stragglers"] = False
    cfg["ros_domain_id_base"] = args.domain
    cfg["run_timeout_after_first_cmd_s"] = args.run_timeout_after_first_cmd_s
    pilot_cfg = out / "pilot_config.yaml"
    pilot_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # 3) isolated env inherited by the runner + gazebo + all nodes
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(args.domain)
    env["IGN_PARTITION"] = env["GZ_PARTITION"] = "tier1pilot"
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["IGN_IP"] = env["GZ_IP"] = "127.0.0.1"
    env["ROS_LOG_DIR"] = str(out / "_ros_logs")

    if args.dry_run:
        dr = subprocess.run([sys.executable, str(RUNNER), "--config", str(pilot_cfg),
                             "--log-root", str(log_root), "--dry-run"], env=env,
                            capture_output=True, text=True, timeout=120)
        cmdline = next((ln for ln in dr.stdout.splitlines() if "pixel_topic:=" in ln), "")
        ok = "/perception/pixel_pose_faulted" in cmdline
        print(dr.stdout[-3000:])
        print("\nDRY-RUN:", "PASS (pixel_topic:=/perception/pixel_pose_faulted in launch CMD)" if ok
              else "FAIL (faulted pixel_topic NOT in launch CMD)")
        print("stderr tail:", dr.stderr[-800:])
        return 0 if ok else 3

    children = []

    def spawn(cmd):
        p = subprocess.Popen(cmd, env=env, start_new_session=True,
                             stdout=open(out / (Path(cmd[1]).stem + ".log"), "w"),
                             stderr=subprocess.STDOUT)
        children.append(p)
        return p

    started = time.time()
    write_status(out, f"# Tier-1 pilot RUNNING\n\nstarted {time.ctime(started)}\n"
                 f"domain {args.domain} · task {args.task} · condition {args.condition}\n"
                 f"fault {args.fault} after {args.fault_after_s}s, lookat-drift {args.calib_lookat_drift}, "
                 f"ramp {args.drift_ramp_s}s\n")
    try:
        # helper nodes first: they wait for their input topics to appear
        spawn([sys.executable, str(INJECTOR), "--fault", args.fault,
               "--in-topic", "/perception/pixel_pose", "--out-topic", "/perception/pixel_pose_faulted",
               "--fault-after-s", str(args.fault_after_s),
               "--calib-lookat-drift", args.calib_lookat_drift,
               "--drift-ramp-s", str(args.drift_ramp_s)])
        spawn([sys.executable, str(HEALTH), "--log-csv", str(health_csv)])
        time.sleep(2.0)

        # faithful bringup + drive (blocks until the single run completes)
        runner_cmd = [sys.executable, str(RUNNER), "--config", str(pilot_cfg),
                      "--log-root", str(log_root), "--run-timeout", "300", "--first-cmd-timeout", "270"]
        with open(out / "runner.log", "w") as rlog:
            rp = subprocess.Popen(runner_cmd, env=env, start_new_session=True,
                                 stdout=rlog, stderr=subprocess.STDOUT)
            children.append(rp)
            try:
                rp.wait(timeout=args.hard_timeout_s)
                runner_rc = rp.returncode
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(rp.pid), signal.SIGINT)
                time.sleep(5)
                runner_rc = -1
    finally:
        for p in children:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGINT)
                except (ProcessLookupError, OSError):
                    pass
        time.sleep(4)
        for p in children:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

    # 4) analyse the health trace + best-effort runner outcome
    summary = analyse(health_csv, log_root, args)
    summary_head = (f"# Tier-1 pilot DONE\n\nout: `{out}`\n\nrunner_rc={locals().get('runner_rc')}  "
                    f"elapsed={time.time()-started:.0f}s\n\n")
    write_status(out, summary_head + summary)
    print(summary_head + summary)
    return 0


def analyse(health_csv: Path, log_root: Path, args) -> str:
    lines = []
    if not health_csv.exists():
        lines.append("**health trace: MISSING** — the health monitor produced no CSV "
                     "(no /planner/pixel_correction_diagnostics? check health.log / runner.log).")
    else:
        rows = [r.split(",") for r in health_csv.read_text().splitlines()[1:] if r.strip()]
        if not rows:
            lines.append("**health trace: EMPTY** — monitor ran but received 0 diagnostic frames.")
        else:
            def col(i):
                return [float(r[i]) for r in rows if len(r) > i]
            t = col(0); h = col(1); nis_ewma = col(4); degraded = [int(float(r[7])) for r in rows if len(r) > 7]
            n = len(rows)
            healthy = degraded.count(0); deg = degraded.count(1)
            first_deg_t = next((t[i] for i in range(n) if degraded[i] == 1), None)
            last_healthy_t = next((t[i] for i in range(n - 1, -1, -1) if degraded[i] == 0), None)
            reached = deg > 0
            lines.append(f"**health trace: {n} frames** — HEALTHY(0) {healthy}, DEGRADED(1) {deg}")
            lines.append(f"- health h: {h[0]:.2f} (start) -> {min(h):.2f} (min)")
            lines.append(f"- nis_ewma: {min(nis_ewma):.2f} (min) -> {max(nis_ewma):.2f} (max)")
            if reached:
                span = (first_deg_t - t[0])
                lines.append(f"- **DEGRADED reached** at t+{span:.0f}s (drift onset ~t+{args.fault_after_s:.0f}s "
                             f"after first detection); {'AFTER' if span >= args.fault_after_s - 5 else 'BEFORE'} the expected onset window")
                lines.append(f"- **VERDICT: detection loop CLOSED on real perception** "
                             f"(healthy phase then DEGRADED after the calibration drift).")
            else:
                lines.append(f"- **DEGRADED never reached** — drift may be too weak / rejected cleanly / "
                             f"onset after run end. Inspect nis_ewma trend + fault-after vs run length.")
    # best-effort runner outcome
    try:
        jsons = list(log_root.rglob("*.json"))
        cl = next((p for p in jsons if "campaign" in p.name.lower() or "log" in p.name.lower()), None)
        if cl:
            data = json.loads(cl.read_text())
            entry = next(iter(data.values())) if isinstance(data, dict) else None
            if isinstance(entry, dict):
                lines.append("")
                lines.append(f"**runner outcome:** goal_reached={entry.get('goal_reached')} "
                             f"crashed={entry.get('crashed')} outcome={entry.get('outcome')} "
                             f"mean_belief_error_gt_m={entry.get('mean_belief_error_gt_m')} "
                             f"completion={entry.get('completion_reason')}")
    except Exception as exc:  # pragma: no cover
        lines.append(f"\n(runner outcome parse skipped: {exc})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
