#!/usr/bin/env python3
"""V6 selector: exact objective with camera-free duplicate diagnostics.

The v5 selector evaluates every fixed rollout three times with the camera model:
once for the selection objective and twice for reporting-only trajectory summaries.
For the final protocol, belief no-go costs are disabled, so those reporting passes
cannot affect feasibility or selection.  V6 preserves the exact objective and all
swept-footprint checks while records mark the omitted per-candidate summaries null.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
V5_PATH = REPO / "experiments/thesis_pipeline_lock/select_stage09_routes_v5.py"


def _load_v5():
    spec = importlib.util.spec_from_file_location("final_route_selector_v5_base", V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v5 selector from {V5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load_v5()
ORIGINAL_EVALUATE = V5.CORE.UnicyclePlannerBase.evaluate_rollout_controls


def _selection_evaluate(self, *args, **kwargs):
    if self.use_belief_nogo_cost:
        raise RuntimeError(
            "v6 reporting-pass acceleration requires use_belief_nogo_cost=false"
        )
    original_diagnostics = self.planning_visibility_diagnostics
    self.planning_visibility_diagnostics = lambda _m, _s: {
        "p_vis": math.nan,
        "p_vis_eff": math.nan,
        "R_plan": np.eye(2, dtype=float),
        "r_plan_u_std": math.nan,
        "r_plan_v_std": math.nan,
    }
    try:
        return ORIGINAL_EVALUATE(self, *args, **kwargs)
    finally:
        del self.planning_visibility_diagnostics
        assert self.planning_visibility_diagnostics.__func__ is original_diagnostics.__func__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    V5.CORE.UnicyclePlannerBase.evaluate_rollout_controls = _selection_evaluate
    V5.__file__ = str(Path(__file__).resolve())
    V5.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
