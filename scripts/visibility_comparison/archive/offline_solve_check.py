#!/usr/bin/env python3
"""Offline global-solve check for every task x condition, no Gazebo, no
localization (uses the TRUE start pose). Isolates the planner: if a task that
fails online also fails here, it is a planner/landscape issue; if it converges
here, the online failure is a runtime (localization/latency/control) issue.
"""
import sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "visibility_comparison"))
from efe_offline_lab import load_setup, run_optimizer  # noqa: E402

CONFIG = REPO / "scripts" / "visibility_comparison" / "aws_f31b1_v3_config.yaml"
cfg = yaml.safe_load(open(CONFIG))
tasks = list(cfg["tasks"].keys())
conds = ["C1", "C2"]

print(f"config={CONFIG.name}  gp={cfg['gp_artifact']}")
print(f"{'task':<28}{'cond':<5}{'ok':<4}{'term_gd':<9}{'route':<26}{'risk':<11}{'ambig':<9}{'obst':<8}{'nit':<5}")
print("-" * 110)
for t in tasks:
    for c in conds:
        try:
            s = load_setup(CONFIG, condition=c, seed=0, task_override=t)
            m0 = np.array(s.start_xy_yaw, dtype=float)
            plan = run_optimizer(s.planner, m0, s.S0, s.goal_xy)
            ok = bool(getattr(plan, "optimizer_success", False))
            term = float(getattr(plan, "terminal_goal_distance_pred", np.nan))
            route = str(getattr(plan, "selected_source", "?"))[:24]
            risk = float(getattr(plan, "risk_cost", np.nan))
            amb = float(getattr(plan, "ambiguity_cost", np.nan))
            obst = float(getattr(plan, "obstacle_cost", np.nan))
            nit = int(getattr(plan, "optimizer_nit", -1))
            print(f"{t:<28}{c:<5}{str(ok):<4}{term:<9.3f}{route:<26}{risk:<11.1f}{amb:<9.1f}{obst:<8.1f}{nit:<5}")
        except Exception as e:
            print(f"{t:<28}{c:<5}ERROR: {type(e).__name__}: {str(e)[:70]}")
