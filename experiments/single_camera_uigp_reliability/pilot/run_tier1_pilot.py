#!/usr/bin/env python3
"""Tier-1 detection-half pilot (single-camera, real Gazebo) + reusable run_one().

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

`run_one(...)` runs ONE isolated cell and returns a metrics dict; the envelope
sweep (run_tier1_envelope.py) calls it across a severity x onset grid.

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
GATE = TOOLS / "safe_degradation_gate.py"
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
            if "run_tier1_" in line or "pgrep" in line:
                continue
            hits.append(line.strip())
    return hits


def wait_machine_free(timeout_s=45.0, poll_s=3.0):
    """Poll until no BUSY_PATTERNS process (other than ours) is alive, or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not machine_busy():
            return True
        time.sleep(poll_s)
    return not machine_busy()


def build_pilot_cfg(out: Path, *, domain: int, task: str, condition: str, seed: int = 0,
                    pixel_topic: str, run_timeout_after_first_cmd_s: float,
                    command_noise_output_topic: str = "/cmd_vel") -> Path:
    cfg = yaml.safe_load(HONEST.read_text())
    orig_task = cfg["tasks"].get(task, {})
    cfg["conditions"] = {condition: cfg["conditions"][condition]}
    cfg["tasks"] = {task: {"conditions": [condition], "seeds": [seed],
                           **({"optimizer_initial_routes_json": orig_task["optimizer_initial_routes_json"]}
                              if "optimizer_initial_routes_json" in orig_task else {})}}
    cfg["pixel_topic"] = pixel_topic
    cfg["command_noise_output_topic"] = command_noise_output_topic
    cfg["cleanup_sim_stragglers"] = False
    cfg["ros_domain_id_base"] = domain
    cfg["run_timeout_after_first_cmd_s"] = run_timeout_after_first_cmd_s
    p = out / "pilot_config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def isolated_env(domain: int, partition: str, out: Path) -> dict:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(domain)
    env["IGN_PARTITION"] = env["GZ_PARTITION"] = partition
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["IGN_IP"] = env["GZ_IP"] = "127.0.0.1"
    env["ROS_LOG_DIR"] = str(out / "_ros_logs")
    return env


def compute_metrics(health_csv: Path, log_root: Path, fault_after_s: float) -> dict:
    m = {"reached_degraded": None, "degraded_latency_s": None, "min_health": None,
         "max_nis_ewma": None, "max_bias": None, "n_frames": 0, "healthy_frames": 0,
         "degraded_frames": 0, "baseline_false_alarms": None,
         "goal_reached": None, "crashed": None, "outcome": None, "mean_belief_error_gt_m": None}
    if health_csv.exists():
        rows = [r.split(",") for r in health_csv.read_text().splitlines()[1:] if r.strip()]
        if rows:
            t = [float(r[0]) for r in rows]
            h = [float(r[1]) for r in rows]
            state = [int(float(r[2])) for r in rows]
            nis_ewma = [float(r[4]) for r in rows]
            bias = [float(r[5]) for r in rows]
            degraded = [int(float(r[7])) for r in rows]
            n = len(rows)
            first_deg = next((i for i in range(n) if degraded[i] == 1), None)
            t0 = t[0]
            # baseline = frames before the drift onset (health-trace time ~ aligned to run start)
            base_cut = t0 + fault_after_s
            baseline_bad = sum(1 for i in range(n) if t[i] < base_cut and state[i] != 0)
            m.update(n_frames=n, min_health=round(min(h), 3), max_nis_ewma=round(max(nis_ewma), 2),
                     max_bias=round(max(bias), 3), healthy_frames=degraded.count(0),
                     degraded_frames=degraded.count(1), baseline_false_alarms=baseline_bad,
                     reached_degraded=(first_deg is not None))
            if first_deg is not None:
                # latency from drift onset to first DEGRADED (onset ~ t0 + fault_after_s)
                m["degraded_latency_s"] = round(max(0.0, t[first_deg] - base_cut), 1)
    try:
        cl = next((p for p in log_root.rglob("*.json")
                   if "campaign" in p.name.lower() or "log" in p.name.lower()), None)
        if cl:
            data = json.loads(cl.read_text())
            entry = next(iter(data.values())) if isinstance(data, dict) else None
            if isinstance(entry, dict):
                m.update(goal_reached=entry.get("goal_reached"), crashed=entry.get("crashed"),
                         outcome=entry.get("outcome"), mean_belief_error_gt_m=entry.get("mean_belief_error_gt_m"))
    except Exception:
        pass
    return m


def run_one(out: Path, *, domain: int, task: str, condition: str, fault: str,
            fault_after_s: float, lookat_drift: str, drift_ramp_s: float,
            run_timeout_after_first_cmd_s: float, hard_timeout_s: float,
            seed: int = 0, safe_stop: bool = False, partition: str = "tier1pilot",
            require_free: bool = True) -> dict:
    """Run ONE isolated real-Gazebo cell; return a metrics dict. Collision-safe.

    safe_stop=True enables the N3 actuation: the actuation-noise node publishes to
    /cmd_vel_pregate and a safe-degradation gate republishes to /cmd_vel, latching a
    STOP once /reliability/localization_degraded goes True. safe_stop=False is N0
    (health monitored + logged, but the robot keeps driving on the faulted camera).
    """
    out.mkdir(parents=True, exist_ok=True)
    health_csv = out / "health_trace.csv"
    log_root = out / "campaign"

    if require_free and machine_busy():
        return {"aborted": "machine_busy", **compute_metrics(health_csv, log_root, fault_after_s)}

    cmd_out = "/cmd_vel_pregate" if safe_stop else "/cmd_vel"
    pilot_cfg = build_pilot_cfg(out, domain=domain, task=task, condition=condition, seed=seed,
                                pixel_topic="/perception/pixel_pose_faulted",
                                run_timeout_after_first_cmd_s=run_timeout_after_first_cmd_s,
                                command_noise_output_topic=cmd_out)
    env = isolated_env(domain, partition, out)
    children = []

    def spawn(cmd, logname):
        p = subprocess.Popen(cmd, env=env, start_new_session=True,
                             stdout=open(out / logname, "w"), stderr=subprocess.STDOUT)
        children.append(p)
        return p

    runner_rc = None
    try:
        spawn([sys.executable, str(INJECTOR), "--fault", fault,
               "--in-topic", "/perception/pixel_pose", "--out-topic", "/perception/pixel_pose_faulted",
               "--fault-after-s", str(fault_after_s), "--calib-lookat-drift", lookat_drift,
               "--drift-ramp-s", str(drift_ramp_s)], "injector.log")
        spawn([sys.executable, str(HEALTH), "--log-csv", str(health_csv)], "health.log")
        if safe_stop:
            spawn([sys.executable, str(GATE), "--in-topic", "/cmd_vel_pregate",
                   "--out-topic", "/cmd_vel", "--degraded-topic", "/reliability/localization_degraded",
                   "--slow-factor", "0.0", "--latch-stop"], "gate.log")
        time.sleep(2.0)
        with open(out / "runner.log", "w") as rlog:
            rp = subprocess.Popen([sys.executable, str(RUNNER), "--config", str(pilot_cfg),
                                   "--log-root", str(log_root), "--run-timeout", "300",
                                   "--first-cmd-timeout", "270"],
                                  env=env, start_new_session=True, stdout=rlog, stderr=subprocess.STDOUT)
            children.append(rp)
            try:
                rp.wait(timeout=hard_timeout_s)
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

    m = compute_metrics(health_csv, log_root, fault_after_s)
    m["runner_rc"] = runner_rc
    m["safe_stop"] = safe_stop
    m["seed"] = seed
    return m


def analyse_md(m: dict, fault_after_s: float) -> str:
    lines = []
    if m.get("aborted"):
        return f"**ABORTED: {m['aborted']}**\n"
    if not m.get("n_frames"):
        return "**health trace MISSING/EMPTY** — monitor received 0 diagnostic frames (check health.log/runner.log).\n"
    lines.append(f"**health trace: {m['n_frames']} frames** — deg=0 {m['healthy_frames']}, deg=1 {m['degraded_frames']}")
    lines.append(f"- baseline (pre-drift) non-HEALTHY frames: **{m['baseline_false_alarms']}** (0 = no false alarms)")
    lines.append(f"- health h min {m['min_health']}; nis_ewma max {m['max_nis_ewma']}; bias max {m['max_bias']} m")
    if m["reached_degraded"]:
        lines.append(f"- **DEGRADED reached**, latency from onset ~ **{m['degraded_latency_s']} s**")
        lines.append("- **VERDICT: detection loop CLOSED on real perception.**")
    else:
        lines.append("- **DEGRADED never reached** — drift too weak / rejected cleanly / onset after run end.")
    lines.append("")
    lines.append(f"**runner outcome:** goal_reached={m['goal_reached']} crashed={m['crashed']} "
                 f"outcome={m['outcome']} mean_belief_error_gt_m={m['mean_belief_error_gt_m']}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--domain", type=int, default=191)
    ap.add_argument("--task", default="route_apron_to_a3_mid")
    ap.add_argument("--condition", default="C2")
    ap.add_argument("--fault", default="calib_drift")
    ap.add_argument("--fault-after-s", type=float, default=18.0)
    ap.add_argument("--calib-lookat-drift", default="0.5,0.5,0.0")
    ap.add_argument("--drift-ramp-s", type=float, default=18.0)
    ap.add_argument("--run-timeout-after-first-cmd-s", type=float, default=150.0)
    ap.add_argument("--hard-timeout-s", type=float, default=600.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve() if args.out else (REPO / "logs" / "studies" /
              "single_camera_uigp_reliability" / f"tier1_pilot_{int(time.time())}")
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        pilot_cfg = build_pilot_cfg(out, domain=args.domain, task=args.task, condition=args.condition,
                                    pixel_topic="/perception/pixel_pose_faulted",
                                    run_timeout_after_first_cmd_s=args.run_timeout_after_first_cmd_s)
        env = isolated_env(args.domain, "tier1pilot", out)
        dr = subprocess.run([sys.executable, str(RUNNER), "--config", str(pilot_cfg),
                             "--log-root", str(out / "campaign"), "--dry-run"], env=env,
                            capture_output=True, text=True, timeout=120)
        cmdline = next((ln for ln in dr.stdout.splitlines() if "pixel_topic:=" in ln), "")
        ok = "/perception/pixel_pose_faulted" in cmdline
        print("DRY-RUN:", "PASS" if ok else "FAIL", "(faulted pixel_topic in launch CMD)")
        return 0 if ok else 3

    started = time.time()
    (out / "STATUS.md").write_text(f"# Tier-1 pilot RUNNING\n\nstarted {time.ctime(started)}\n")
    m = run_one(out, domain=args.domain, task=args.task, condition=args.condition, fault=args.fault,
                fault_after_s=args.fault_after_s, lookat_drift=args.calib_lookat_drift,
                drift_ramp_s=args.drift_ramp_s,
                run_timeout_after_first_cmd_s=args.run_timeout_after_first_cmd_s,
                hard_timeout_s=args.hard_timeout_s)
    head = (f"# Tier-1 pilot DONE\n\nout: `{out}`\n\nrunner_rc={m.get('runner_rc')}  "
            f"elapsed={time.time()-started:.0f}s\n\n")
    (out / "STATUS.md").write_text(head + analyse_md(m, args.fault_after_s))
    print(head + analyse_md(m, args.fault_after_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
