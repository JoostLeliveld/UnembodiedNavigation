#!/usr/bin/env python3
"""Offline no-spin gate for the LOCAL reference-segment TRACKER (engineering gate,
NOT scientific evidence).

Builds the CONDITION-NEUTRAL local executor with the proper reference-segment
tracking objective (ref_weight + terminal_ref_weight + du_weight; NO GP, NO
visibility, NO ambiguity) using the LOCAL params from aws_beliefloop_v1_config.yaml,
and runs a waypoint-tracking rollout that mirrors the runtime efe_agent_node LOCAL
path: at each step it builds the per-step reference segment OUTSIDE the solve by
projecting the current belief onto the waypoint polyline and sampling H points
forward, then passes ref_seq + prev_u into planner.plan(...).

REQUIRE: from start (3.3, -1.0, yaw 0) toward the occluded route
[[3.3,1.5],[1.5,1.5],[1.0,1.75]] and the visible route
[[3.3,-2.2],[1.2,-2.2],[1.0,1.75]], the robot TURNS toward the first waypoint and
ADVANCES (cmd_w not saturated the whole time; net displacement grows; reaches the
final waypoint), with NO spin.

Run:
  python3 scripts/visibility_comparison/test_ref_tracker_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "scripts" / "visibility_comparison"
sys.path.insert(0, str(LAB))

from efe_offline_lab import load_setup, build_planner  # noqa: E402

CONFIG = LAB / "aws_beliefloop_v1_config.yaml"
TASK = "F31_b1_apron_a3_mid"

# Local tracker tuning mirrored from aws_beliefloop_v1_config.yaml.
LOCAL_H = 8
LOCAL_DT = 0.25
V_MAX = 0.60
W_MAX = 1.5  # matches w_max bound used by the planner bounds for the local solve

REF_WEIGHT = 10.0
TERMINAL_REF_WEIGHT = 4.0
DU_WEIGHT = 0.5
NOGO_WEIGHT = 40.0
# Runtime densifies the global plan into waypoints every waypoint_spacing_m metres.
# A reference/lookahead tracker legitimately cuts sharp corner vertices, so we follow
# runtime by densifying the route polyline before tracking (matches aws_beliefloop_v1
# waypoint_spacing_m: 0.40) and advance on arc-length progress, not corner proximity.
WAYPOINT_SPACING_M = 0.40


def build_local_tracker(setup):
    cfg = dict(setup.config)
    cfg["horizon"] = LOCAL_H
    cfg["goal_progress_n_steps"] = LOCAL_H
    cfg["goal_progress_weight"] = 0.0          # reference tracking REPLACES the single goal point
    cfg["ref_weight"] = REF_WEIGHT
    cfg["terminal_ref_weight"] = TERMINAL_REF_WEIGHT
    cfg["du_weight"] = DU_WEIGHT
    cfg["control_weight"] = 0.0
    cfg["v_max"] = V_MAX
    cfg["w_max"] = W_MAX
    cfg["optimizer_maxiter"] = 45
    cfg["optimizer_multistart"] = False
    cfg["optimizer_initial_routes_json"] = ""
    cfg["optimizer_multistart_lateral_offsets"] = ""
    # local belief-tube no-go (condition-neutral), lowered to 40 so the ref pull carries the turn
    cfg["use_nogo_cost"] = True
    cfg["nogo_penalty_type"] = "log_barrier"
    cfg["nogo_weight"] = NOGO_WEIGHT
    cfg["nogo_safe_distance"] = 0.30
    cfg["use_belief_nogo_cost"] = True
    cfg["nogo_belief_kappa"] = 2.0
    cfg["nogo_logbarrier_scale"] = 0.10
    cfg["nogo_logbarrier_eps"] = 0.01
    # condition-neutral local: constant R, no GP, no ambiguity, no obs risk.
    return build_planner(
        cfg,
        planner_kind="constant_R_efe",
        camera_params=setup.camera_params,
        visibility_artifact_path="",                       # NO GP in the local executor
        visibility_geometry_json="",
        collision_geometry_json=setup.geometry_json,
        driveable_geometry_json=setup.driveable_geometry_json,
        seed=1,
        visibility_target_height_m=setup.gp["target_height"],
    )


def build_reference_segment(m_xy, waypoints, H, step):
    """Mirror efe_agent_node._build_local_reference_segment (computed OUTSIDE solve)."""
    pos = np.asarray(m_xy[:2], dtype=float).reshape(2)
    pts = [np.asarray(w, dtype=float).reshape(2) for w in waypoints]
    if len(pts) == 1:
        return np.tile(pts[0], (H, 1))
    best_seg, best_proj, best_d2 = 0, pts[0], float("inf")
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom <= 1e-12 else float(np.clip(((pos - a) @ ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d2 = float(np.sum((pos - proj) ** 2))
        if d2 < best_d2:
            best_d2, best_seg, best_proj = d2, i, proj
    ref = np.zeros((H, 2), dtype=float)
    seg, cur = best_seg, best_proj.copy()
    for k in range(H):
        remaining = step
        while remaining > 1e-9 and seg < len(pts) - 1:
            nxt = pts[seg + 1]
            seg_vec = nxt - cur
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len <= 1e-9:
                seg += 1
                continue
            if seg_len >= remaining:
                cur = cur + (remaining / seg_len) * seg_vec
                remaining = 0.0
            else:
                cur = nxt.copy()
                remaining -= seg_len
                seg += 1
        ref[k] = cur
    return ref


def densify(waypoints, spacing):
    """Resample a corner polyline into waypoints every `spacing` metres, matching
    the runtime global-plan waypoint extraction (waypoint_spacing_m)."""
    pts = [np.asarray(w, dtype=float).reshape(2) for w in waypoints]
    out = [pts[0]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        L = float(np.linalg.norm(seg))
        n = max(int(np.ceil(L / spacing)), 1)
        for k in range(1, n + 1):
            out.append(a + (k / n) * seg)
    return out


def track(planner, m0, S0, waypoints, n_steps=200, arrival_r=0.20, dt=LOCAL_DT):
    m = np.asarray(m0, dtype=float).reshape(3).copy()
    S = np.asarray(S0, dtype=float).reshape(3, 3).copy()
    wps = densify(waypoints, WAYPOINT_SPACING_M)
    wp_idx = 0
    planner.optimizer_warm_start = True
    planner.prev_controls_flat = None
    step = max(V_MAX * dt, 1e-3)
    vs, ws, traj = [], [], [m.copy()]
    last_cmd = np.array([0.0, 0.0], dtype=float)
    for _ in range(n_steps):
        # Advance along the densified route by arc-length progress: jump wp_idx to
        # the closest dense waypoint AHEAD of the robot (monotone), so a lookahead
        # tracker that legitimately cuts corners still makes progress. This mirrors
        # a runtime dense-waypoint follower far better than corner-vertex proximity.
        prev_idx = wp_idx
        dists = [float(np.linalg.norm(m[:2] - wps[j])) for j in range(wp_idx, len(wps))]
        nearest = wp_idx + int(np.argmin(dists))
        wp_idx = max(wp_idx, nearest)
        target = wps[wp_idx]
        advanced = wp_idx != prev_idx
        if advanced or planner.prev_controls_flat is None:
            try:
                planner.prev_controls_flat = np.asarray(
                    planner._controls_for_waypoints(m[:3], [target]), dtype=float).reshape(-1)
            except Exception:
                pass
        ref_seq = build_reference_segment(m, wps, planner.horizon, step)
        plan = planner.plan(m, S, target, ref_seq=ref_seq, prev_u=last_cmd)
        u0 = np.asarray(plan.controls[0], dtype=float)
        vs.append(float(u0[0]))
        ws.append(float(u0[1]))
        last_cmd = u0.copy()
        m, S = planner.predict(m, S, u0)
        traj.append(m.copy())
        if wp_idx == len(wps) - 1 and float(np.linalg.norm(m[:2] - wps[-1])) < arrival_r:
            break
    v = np.array(vs, dtype=float)
    w = np.array(ws, dtype=float)
    traj = np.asarray(traj)
    w_sat = W_MAX - 1e-3
    return {
        "v": v, "w": w, "traj": traj,
        "reached": bool(np.linalg.norm(traj[-1][:2] - wps[-1]) < arrival_r),
        "final_d": float(np.linalg.norm(traj[-1][:2] - wps[-1])),
        "net": float(np.linalg.norm(traj[-1][:2] - traj[0][:2])),
        "frac_moving": float(np.mean(np.abs(v) > 0.05)) if v.size else 0.0,
        "frac_w_sat": float(np.mean(np.abs(w) >= w_sat)) if w.size else 0.0,
        "max_zero_streak": _max_zero_streak(v, dt),
        "wp_reached": wp_idx,
        "n_dense_wp": len(wps),
        "steps": int(v.size),
    }


def _max_zero_streak(v, dt):
    best = cur = 0
    for vi in v:
        cur = cur + 1 if abs(vi) <= 0.05 else 0
        best = max(best, cur)
    return best * dt


ROUTES = {
    "occluded (N then W turn)": [(3.3, 1.5), (1.5, 1.5), (1.0, 1.75)],
    "visible (S then W then N)": [(3.3, -2.2), (1.2, -2.2), (1.0, 1.75)],
}


def main() -> int:
    setup = load_setup(CONFIG, condition="C1", task_override=TASK)
    m0 = np.array(setup.start_xy_yaw, dtype=float)
    S0 = setup.S0.copy()
    reach = LOCAL_H * LOCAL_DT * V_MAX
    print(f"start={tuple(round(v,2) for v in m0)}  local reach (H{LOCAL_H}*dt{LOCAL_DT}*vmax{V_MAX})={reach:.2f} m")
    print(f"ref_weight={REF_WEIGHT} terminal_ref_weight={TERMINAL_REF_WEIGHT} "
          f"du_weight={DU_WEIGHT} local_nogo_weight={NOGO_WEIGHT}")
    print("LOCAL reference-segment TRACKER (condition-neutral; C1 build, but objective is identical for C1/C2/C3).\n")

    all_pass = True
    for route_name, wps in ROUTES.items():
        planner = build_local_tracker(setup)
        assert planner.use_visibility_model is False, "local executor must not use GP"
        assert planner.ref_weight > 0.0, "ref tracking must be active"
        r = track(planner, m0, S0, wps)
        no_spin = r["frac_w_sat"] < 0.95 and r["max_zero_streak"] < 1.5
        advances = r["net"] > 1.0 and r["frac_moving"] > 0.5
        verdict_ok = no_spin and advances and r["reached"]
        all_pass = all_pass and verdict_ok
        print(f"=== route: {route_name} ===")
        print(f"  reached={r['reached']} dense_wp={r['wp_reached']}/{r['n_dense_wp']-1} steps={r['steps']} "
              f"final_d={r['final_d']:.2f}m net={r['net']:.2f}m")
        print(f"  frac|v|>0.05={r['frac_moving']:.2f} frac_w_saturated={r['frac_w_sat']:.2f} "
              f"max_zero_streak={r['max_zero_streak']:.2f}s")
        print(f"  -> {'PASS (turns + advances, no spin)' if verdict_ok else 'FAIL'}\n")

    print("OVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
