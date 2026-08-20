#!/usr/bin/env python3
"""Materialise a runner-valid campaign config from the E4 plan plus E3's routes.

WHY THIS EXISTS. `campaign.yaml` is the human-readable plan: it carries task roles,
the design block and `optimizer_route_seed_source`, none of which
`run_visibility_campaign.py` understands. The runner wants
`optimizer_initial_routes_json` — a JSON string of concrete waypoints. This script
does that substitution, so the plan stays readable and the executed config stays
exactly derivable from it.

The seeded candidates per task are the availability-BLIND route and the CAD-planned
route. Seeding both is deliberate: the planner still solves online, but its multistart
is given the short option and the well-observed option rather than being asked to
discover a 15 m route from a cold start, which is how the 2026-08-15 pilot failed.

Usage:
    python3 make_campaign_config.py --out <path> [--tasks mc_blind_L] \
        [--conditions C1] [--seeds 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "availability_paper"))
import common as C  # noqa: E402

PLAN = HERE / "campaign.yaml"
ROUTES = C.OUT_ROOT / "e3_route_discrimination/e3_selected_routes.json"
#: Keys that belong to the plan, not to the runner.
PLAN_ONLY_TOP = ("design",)
PLAN_ONLY_TASK = ("role", "optimizer_route_seed_source")
#: Which E3 arms become multistart seeds, in order.
SEED_ARMS = ("availability_blind", "cad_reference")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--subset", default="four", help="camera subset whose routes seed the planner")
    args = ap.parse_args()

    cfg = yaml.safe_load(PLAN.read_text())
    if not ROUTES.is_file():
        raise SystemExit(f"missing {ROUTES}; run E3 first")
    routes = json.loads(ROUTES.read_text())["routes"][args.subset]

    for key in PLAN_ONLY_TOP:
        cfg.pop(key, None)

    tasks = {}
    for name, task_cfg in cfg["tasks"].items():
        if args.tasks and name not in args.tasks:
            continue
        task_cfg = {k: v for k, v in task_cfg.items() if k not in PLAN_ONLY_TASK}
        if args.conditions:
            task_cfg["conditions"] = [c for c in task_cfg["conditions"] if c in args.conditions]
            if not task_cfg["conditions"]:
                continue
        if args.seeds is not None:
            task_cfg["seeds"] = list(args.seeds)

        if name not in routes:
            raise SystemExit(f"E3 has no routes for task {name!r} (subset {args.subset})")
        seeds = []
        for arm in SEED_ARMS:
            pts = routes[name].get(arm)
            if not pts:
                continue
            seeds.append({"name": arm, "waypoints": [[float(x), float(y)] for x, y in pts]})
        if not seeds:
            raise SystemExit(f"no seed routes for {name}")
        task_cfg["optimizer_initial_routes_json"] = json.dumps(seeds)
        tasks[name] = task_cfg

    if not tasks:
        raise SystemExit("no tasks selected")
    cfg["tasks"] = tasks
    cfg["optimizer_route_seed_mode"] = "explicit"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=10**6))
    print(f"wrote {out}")
    for name, t in tasks.items():
        n = len(json.loads(t["optimizer_initial_routes_json"]))
        print(f"  {name}: conditions {t['conditions']}, seeds {t['seeds']}, {n} seeded routes")


if __name__ == "__main__":
    main()
