#!/usr/bin/env python3
"""Drive headless Gazebo runs for timing + horizon comparison (durable copy).

AWS B1 task, C1/C2 comparison using the locked two-stage hierarchical method.
Writes run logs to /home/joostleliveld/Thesis/timing_presentation/runs/gazebo/<label>/
(outside the git repo so the IDE's git clean cannot wipe them).

Locked method (hier_c1 / hier_c2):
  global H=60, multistart, ambiguity → waypoints → local H=12, 4 Hz, no ambiguity
  Only difference between C1 and C2: planner-facing observation covariance.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
DURABLE = Path("/home/joostleliveld/Thesis/timing_presentation")
WORLD = "warehouse_aws.world.sdf"
TASK = "B1_apron_a4_to_uppermid_a3"
GP = REPO / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
YOLO = REPO / "logs/perception_models/aws_yolo_simseg_v2/model.pt"
OUT_ROOT = DURABLE / "runs/gazebo"
LATERAL = "-2.0,2.0"

# Extra multistart seed for B1: go north through A4, cross mid-cross-aisle to A3, reach goal.
# Avoids the shelf-end staging zone and gives the optimizer a non-degenerate feasible basin.
INITIAL_ROUTES_B1 = json.dumps([
    {"name": "via_A4_mid_cross_A3",
     "waypoints": [[3.1, 1.3], [1.0, 1.3], [1.0, 1.75]]}
])

RUNS = [
    # ── Archived (stale — pre-C1-ambiguity-fix, softplus nogo) ──────────────
    # (moved to runs/gazebo/archive/; entries kept here for reference only)
    # ("H200_ms0", ...), ("H200_ms1", ...), ("H80_ms*", ...), ("H40_ms1", ...)

    # ── Exploratory smoke run (C2 only, before runtime contract lock) ────────
    # (label, H_base, ms, hier, condition)
    ("hier_v1", 60, True, True, "visibility_aware_efe"),

    # ── Locked C1 vs C2 comparison (hier method, runtime contract v1) ────────
    # Both runs use identical params; only planner-facing covariance differs.
    # C1: constant-R EFE (risk + ambiguity, NO GP)
    # C2: GP-conditioned EFE (risk + ambiguity, WITH GP)
    # v1: first attempt — C1 degenerate global plan, C2 crashed at A3/A4 transition
    # ("hier_c1", 60, True, True, "constant_R_efe"),
    # ("hier_c2", 60, True, True, "visibility_aware_efe"),

    # v2: adds via-A4-mid-cross-A3 route seed — still stuck (global belief-nogo over H=60 causes zero-vel)
    # ("hier_c1_v2", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v2", 60, True, True, "visibility_aware_efe"),

    # v3: global belief-nogo disabled — both moved but crashed from localization loss (local solver 1500ms >> 250ms budget due to local_use_belief_nogo_cost)
    # ("hier_c1_v3", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v3", 60, True, True, "visibility_aware_efe"),

    # v4: same as v3 but local_use_belief_nogo reverted — still 1500ms (route seed hits local multistart too)
    # ("hier_c1_v4", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v4", 60, True, True, "visibility_aware_efe"),

    # v5: local_optimizer_multistart:=false — local still zero-vel (warm-start locks in from cold)
    # ("hier_c1_v5", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v5", 60, True, True, "visibility_aware_efe"),

    # v6: local multistart back on; efe_agent_node.py fix passes optimizer_initial_routes_json='' to local
    # ("hier_c1_v6", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v6", 60, True, True, "visibility_aware_efe"),

    # v7: local solver speed fix — maxfun capped (local_maxiter*4=120), direct_goal disabled for local
    #     BROKEN: disabling direct_goal caused zero-vel trap on first call (no seed → cold→zero-vel→warm inherits)
    # ("hier_c1_v7", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v7", 60, True, True, "visibility_aware_efe"),

    # v8: re-enable direct_goal for local but keep maxfun=120/candidate (3 cands × 120 × 0.45ms ≈ 162ms)
    #     STILL 2490ms — per-iteration cost ~28ms; 3 cands × 30 iters × 28ms = 2520ms
    # ("hier_c1_v8", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v8", 60, True, True, "visibility_aware_efe"),

    # v9: warm-start-only (multistart=False) + bootstrap prev_controls from direct-goal at GLOBAL→LOCAL
    #     1 candidate × 10 maxiter × 28ms ≈ 280ms worst-case; ~100ms typical (warm near-optimal)
    #     RESULT: mean=767ms (worse than expected — per-iter cost ~70ms, not 28ms due to eval overhead)
    #     Crash at y≈2.1 (A4→A3 transition) — warm-start cuts diagonally, only 10 iters insufficient to reorient
    # ("hier_c1_v9", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v9", 60, True, True, "visibility_aware_efe"),

    # v10: H_local=8 (from 12) to halve per-iter CasADi cost; 1 cand × 15 maxiter × ~46ms ≈ 690ms
    #     C1 mean=319ms but stuck at (3.24, 2.17) — warm-start carries northward momentum at 90° turn
    #     C2 crashed at 5.2s (913ms mean — crashed before warm start warmed up)
    # ("hier_c1_v10", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v10", 60, True, True, "visibility_aware_efe"),

    # v11: waypoint-transition reseeding — at each wp_idx advance, reset prev_controls_flat from
    #      _controls_for_waypoints(m0[:3], [new_target]) so warm start points toward new waypoint
    #      C1: crashed at 8s (y→3.5, localization drift 1.1m, belief error 0.72m) — reseeding too aggressive
    #      C2: moved to (2.676, 0.911) then stuck — nogo_weight=200 pins robot at A4 western boundary (0.10m margin)
    # ("hier_c1_v11", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v11", 60, True, True, "visibility_aware_efe"),

    # v12: local_nogo_weight 200→80 to release western A4 boundary trap; keep reseeding
    #     C1: collision path=3.79m min_goal=1.53m crashed=(2.34,2.50) solve=381ms belief_err=0.258m
    #     C2: collision path=3.47m min_goal=1.22m crashed=(2.13,2.21) solve=344ms belief_err=0.222m
    #     C2 gets 19% closer to goal; lower solve and belief error — locked comparison pair
    # ("hier_c1_v12", 60, True, True, "constant_R_efe"),
    # ("hier_c2_v12", 60, True, True, "visibility_aware_efe"),
]


def reap(world: str) -> None:
    pats = [f"ign gazebo.*{world}", f"ruby /usr/bin/ign gazebo.*{world}", f"gz sim.*{world}",
            "efe_agent", "experiment_logger", "yolo_robot_detector_node",
            "pixel_to_bev_state_node", "goal_mission_node"]
    for p in pats:
        try:
            subprocess.run(["pkill", "-f", p], timeout=3, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


def build_cmd(H, ms, log_dir, first_cmd_to, hier=False,
              condition="visibility_aware_efe"):
    plan_rate = "4.0" if hier else "1.0"
    use_gp = (condition == "visibility_aware_efe")
    cmd = ["ros2", "launch", "experiments", "warehouse_primary_comparison.launch.py",
           f"world:={WORLD}", f"task:={TASK}", f"planner:={condition}", "seed:=0",
           f"log_dir:={log_dir}", f"horizon:={H}", "dt:=0.25", "v_max:=1.5", f"plan_rate:={plan_rate}",
           f"yolo_model:={YOLO}",
           *([ f"visibility_artifact_path:={GP}" ] if use_gp else []),
           "r_visible_uv:=8.0", "r_miss_uv:=25.0", "ambiguity_weight:=6.0",
           "goal_prior_u_std_start:=80.0", "goal_prior_v_std_start:=80.0",
           "goal_prior_u_std_final:=20.0", "goal_prior_v_std_final:=20.0",
           "goal_tightening_power:=0.45", "optimizer_maxiter:=150", "optimizer_maxfun:=900",
           "optimizer_warm_start:=true",
           f"optimizer_multistart:={'true' if (ms or hier) else 'false'}",
           "optimizer_multistart_include_direct:=true",
           "use_command_noise:=false", "use_encoder_noise:=false", "use_odom_for_predict:=true",
           "odom_topic:=/odom", "auto_stop_on_goal:=true", "goal_success_radius:=0.20",
           f"run_timeout_after_first_cmd_s:={first_cmd_to}",
           "use_nogo_cost:=true", "nogo_mode:=keep_in",
           "headless:=true", "use_rviz:=false", "reset_world:=false"]
    if hier:
        # belief log-barrier keep-in (penalise belief 2-sigma outside lanes) + two-stage
        cmd += ["nogo_penalty_type:=log_barrier", "nogo_weight:=40.0", "nogo_safe_distance:=0.25",
                "nogo_logbarrier_scale:=0.25", "nogo_logbarrier_eps:=0.01",
                "nogo_softplus_scale:=0.05",
                "use_belief_nogo_cost:=false",
                # local nogo: softplus for smoother gradients → faster L-BFGS-B convergence
                "local_nogo_penalty_type:=softplus", "local_nogo_weight:=80.0",
                "use_hierarchical:=true", "global_horizon:=60", "local_horizon:=8",
                "local_plan_rate:=4.0", "local_optimizer_maxiter:=15",
                "global_use_ambiguity:=true", "local_use_ambiguity:=false",
                "waypoint_spacing_m:=1.0", "waypoint_arrival_radius_m:=0.35",
                f"optimizer_multistart_lateral_offsets:={LATERAL}",
                f"optimizer_initial_routes_json:={INITIAL_ROUTES_B1}"]
    else:
        # single-stage keep-in (softplus state) baseline
        cmd += ["nogo_penalty_type:=softplus", "nogo_weight:=600.0",
                "nogo_safe_distance:=0.25", "nogo_softplus_scale:=0.05",
                "use_belief_nogo_cost:=false"]
        if ms:
            cmd.append(f"optimizer_multistart_lateral_offsets:={LATERAL}")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subprocess-timeout", type=float, default=300.0)
    ap.add_argument("--first-cmd-timeout", type=float, default=80.0)
    ap.add_argument("--cleanup-delay", type=float, default=8.0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    runs = [r for r in RUNS if (not only or r[0] in only)]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in runs:
        label, H, ms = entry[0], entry[1], entry[2]
        hier = entry[3] if len(entry) > 3 else False
        condition = entry[4] if len(entry) > 4 else "visibility_aware_efe"
        log_dir = OUT_ROOT / label
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== RUN {label} (H={H} ms={ms} hier={hier} cond={condition}) =====", flush=True)
        reap(WORLD); time.sleep(2.0)
        log_file = log_dir / "launch_stdout.log"
        t0 = time.perf_counter(); status = "ok"
        ros_cmd = build_cmd(H, ms, log_dir, args.first_cmd_timeout,
                            hier=hier, condition=condition)
        shell_cmd = (f"source {REPO}/install/setup.bash && "
                     + " ".join(shlex.quote(str(a)) for a in ros_cmd))
        with log_file.open("w") as lf:
            try:
                subprocess.run(["bash", "-c", shell_cmd],
                               timeout=args.subprocess_timeout, stdout=lf,
                               stderr=subprocess.STDOUT, check=False, cwd=str(REPO))
            except subprocess.TimeoutExpired:
                status = "subprocess_timeout"
            except Exception as exc:  # noqa: BLE001
                status = f"error:{type(exc).__name__}"
        dt = time.perf_counter() - t0
        reap(WORLD)
        print(f"[{label}] status={status} wall={dt:.0f}s", flush=True)
        results.append((label, status, round(dt, 1)))
        time.sleep(args.cleanup_delay)
    print("\n===== SUMMARY =====")
    for label, status, dt in results:
        print(f"  {label:10s} {status:18s} {dt:7.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
