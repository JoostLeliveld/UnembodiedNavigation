#!/usr/bin/env python3
"""Offline global-solve check of the planning horizon change, no Gazebo, no
localization (uses the TRUE start pose). Reproduces the RUNTIME global planner
exactly: efe_agent_node builds self.global_planner = UnicyclePlannerBase(
horizon=global_horizon, dt=global_dt), so overriding the flat horizon/dt keys in
the offline lab's build_planner is a faithful stand-in for the one-shot global
solve.

Compares, for every task x {C1,C2}, the PAPER horizon (120 x 0.25 = 30 s
lookahead) against the CURRENT horizon (30 x 0.4 = 12 s). Reports optimizer
convergence, predicted terminal goal distance, selected route seed, the EFE cost
breakdown, iteration count, and wall-clock solve time.

Run:
  python3 scripts/visibility_comparison/diag/offline_horizon_check.py
"""
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "visibility_comparison" / "archive"))
import efe_offline_lab as lab  # noqa: E402
# efe_offline_lab was moved into archive/, so its REPO_ROOT (= parents[2]) now
# points at scripts/ instead of the repo root. Repair the path constants so
# tasks.yaml / world_profiles.yaml / gp_artifact resolve correctly.
lab.REPO_ROOT = REPO
lab.TASKS_PATH = REPO / "src" / "experiments" / "config" / "tasks.yaml"
lab.WORLD_PROFILES_PATH = REPO / "src" / "experiments" / "config" / "world_profiles.yaml"
from efe_offline_lab import load_setup, run_optimizer  # noqa: E402

CONFIG = REPO / "scripts" / "visibility_comparison" / "warehouse_visibility_campaign.yaml"

# (label, horizon, dt) -> lookahead seconds. The runtime global plan is one-shot
# and frozen (efe_agent_node: "chosen once and never replanned"); if it stops
# short of goal the local tracker drives a STRAIGHT LINE to the goal. So the
# horizon must cover the whole route (~30 s @ v=0.6 -> 18 m), not just be fast.
# Candidates below hold ~30 s coverage while cutting the step COUNT via larger dt.
SETTINGS = [
    # paper_120x0.25 (30 s, reach 0.04-0.58 on a3/a2) and 30x0.40 (12 s, 2.8 m short)
    # already measured in the first run; here we test the 30 s-coverage candidates
    # that cut the step COUNT for a faster solve.
    ("h75_dt0.40_30s", 75, 0.40),         # same 30 s coverage, 1.6x fewer steps than paper
    ("h60_dt0.50_30s", 60, 0.50),         # same 30 s coverage, 2x fewer steps than paper
]

CONDS = ["C1", "C2"]


def _patched_config(base_cfg: dict, horizon: int, dt: float) -> Path:
    """Write a temp copy of the campaign config with horizon/dt overridden so the
    offline lab's build_planner (which reads the flat horizon/dt keys) builds the
    planner the runtime global stage would build."""
    cfg = dict(base_cfg)
    cfg["horizon"] = int(horizon)
    cfg["dt"] = float(dt)
    fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/tmp"
    )
    yaml.safe_dump(cfg, fd)
    fd.close()
    return Path(fd.name)


def main() -> None:
    base_cfg = yaml.safe_load(open(CONFIG))
    tasks = list(base_cfg["tasks"].keys())

    print(f"config = {CONFIG.name}")
    print(f"gp     = {base_cfg['gp_artifact']}")
    print(f"nogo   = mode={base_cfg.get('nogo_mode')} weight={base_cfg.get('nogo_weight')} "
          f"safe_d={base_cfg.get('nogo_safe_distance')} belief_kappa={base_cfg.get('nogo_belief_kappa')}")
    print()
    hdr = (f"{'task':<24}{'cond':<5}{'setting':<22}{'ok':<4}"
           f"{'term_gd':<9}{'route':<20}{'risk':<10}{'ambig':<9}{'obst':<9}"
           f"{'nit':<5}{'solve_s':<9}")
    print(hdr)
    print("-" * len(hdr))

    # Pre-build one patched config per horizon setting (shared across tasks/conds).
    patched = {lbl: _patched_config(base_cfg, H, dt) for (lbl, H, dt) in SETTINGS}

    summary = {lbl: {"ok": 0, "reach": 0, "n": 0, "solve": []} for (lbl, _, _) in SETTINGS}

    for t in tasks:
        for c in CONDS:
            for (lbl, H, dt) in SETTINGS:
                try:
                    s = load_setup(patched[lbl], condition=c, seed=0, task_override=t)
                    m0 = np.array(s.start_xy_yaw, dtype=float)
                    t0 = time.perf_counter()
                    plan = run_optimizer(s.planner, m0, s.S0, s.goal_xy)
                    solve_s = time.perf_counter() - t0

                    ok = bool(getattr(plan, "optimizer_success", False))
                    term = float(getattr(plan, "terminal_goal_distance_pred", np.nan))
                    route = str(getattr(plan, "selected_source", "?"))[:18]
                    risk = float(getattr(plan, "risk_cost", np.nan))
                    amb = float(getattr(plan, "ambiguity_cost", np.nan))
                    obst = float(getattr(plan, "obstacle_cost", np.nan))
                    nit = int(getattr(plan, "optimizer_nit", -1))

                    summary[lbl]["n"] += 1
                    summary[lbl]["ok"] += int(ok)
                    # "reach" = predicted terminal within the goal success radius.
                    summary[lbl]["reach"] += int(term <= float(base_cfg.get("goal_success_radius", 0.25)))
                    summary[lbl]["solve"].append(solve_s)

                    print(f"{t:<24}{c:<5}{lbl:<22}{str(ok):<4}"
                          f"{term:<9.3f}{route:<20}{risk:<10.1f}{amb:<9.1f}{obst:<9.1f}"
                          f"{nit:<5}{solve_s:<9.2f}")
                except Exception as e:
                    print(f"{t:<24}{c:<5}{lbl:<22}ERROR: {type(e).__name__}: {str(e)[:60]}")
            print()

    print("=" * len(hdr))
    print(f"{'setting':<22}{'converged':<12}{'pred_reach':<12}{'median_solve_s':<16}")
    for (lbl, _, _) in SETTINGS:
        d = summary[lbl]
        med = float(np.median(d["solve"])) if d["solve"] else float("nan")
        print(f"{lbl:<22}{d['ok']}/{d['n']:<10}{d['reach']}/{d['n']:<10}{med:<16.2f}")

    for p in patched.values():
        try:
            p.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
