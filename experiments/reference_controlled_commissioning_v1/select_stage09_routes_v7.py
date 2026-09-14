#!/usr/bin/env python3
"""V7 selector: V6 scoring with complete physical collision geometry.

V6 checked the swept rectangular body against the driveable-lane union, while
the rollout collision scene omitted the separately included ``bin_office``
model.  V7 requires a collision scene containing that prop and applies the same
swept body check to both the driveable union and every physical collision prism.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
V6_PATH = REPO / "experiments/reference_controlled_commissioning_v1/select_stage09_routes_v6.py"


def _load_v6():
    spec = importlib.util.spec_from_file_location("final_route_selector_v6_base", V6_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v6 selector from {V6_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V6 = _load_v6()
V5 = V6.V5
CORE = V5.CORE
ORIGINAL_RESOLVE = CORE.ROUTE_PROBE.resolve
ORIGINAL_CLEARANCE = CORE._minimum_static_body_clearance
COLLISION_SCENE = None
REQUIRED_COLLISION_NAMES = {"bin_office"}


def _resolve_with_collision_scene(*args, **kwargs):
    global COLLISION_SCENE
    resolved = ORIGINAL_RESOLVE(*args, **kwargs)
    raw = str(resolved["settings"].get("collision_geometry_json", "") or "")
    scene = CORE.scene_from_json(raw)
    names = {str(prism.name).split("/", 1)[0] for prism in scene.prisms}
    missing = REQUIRED_COLLISION_NAMES - names
    if missing:
        raise RuntimeError(
            f"physical collision scene is missing required models: {sorted(missing)}"
        )
    COLLISION_SCENE = scene
    return resolved


def _minimum_complete_body_clearance(
    driveable_scene, route, *, start_yaw: float, length_m: float,
    width_m: float, step_m: float,
) -> float:
    driveable = ORIGINAL_CLEARANCE(
        driveable_scene, route, start_yaw=start_yaw, length_m=length_m,
        width_m=width_m, step_m=step_m,
    )
    if COLLISION_SCENE is None:
        raise RuntimeError("collision geometry was not resolved before route checking")
    body = CORE.RectangularFootprint(
        COLLISION_SCENE.prisms, length_m, width_m, keep_in=False,
    )
    headings = [
        math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(route, route[1:])
    ]
    clearances = []
    for (start, end), heading in zip(zip(route, route[1:]), headings):
        clearances.extend(
            body.clearance([x, y, heading])
            for x, y in CORE.sample_polyline([start, end], maximum_step_m=step_m)
        )
    rotations = [(route[0], start_yaw, headings[0])]
    rotations.extend(
        (route[index], headings[index - 1], headings[index])
        for index in range(1, len(route) - 1)
    )
    for point, before, after in rotations:
        clearances.extend(
            body.clearance([point[0], point[1], yaw])
            for yaw in CORE._angle_sweep(before, after, math.radians(5.0))
        )
    collision = float(min(clearances)) if clearances else -math.inf
    return min(driveable, collision)


def main() -> int:
    global REQUIRED_COLLISION_NAMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()
    protocol = CORE._read_json(args.protocol)
    for label, record in protocol.get("selector_dependencies", {}).items():
        CORE._verify((REPO / record["path"]).resolve(), record["sha256"], label)
    for label in ("launch_resolution", "world_profile"):
        record = protocol[label]
        CORE._verify((REPO / record["path"]).resolve(), record["sha256"], label)
    REQUIRED_COLLISION_NAMES = set(
        protocol["common_feasibility"].get("required_collision_prisms", ())
    )
    CORE.ROUTE_PROBE.resolve = _resolve_with_collision_scene
    CORE._minimum_static_body_clearance = _minimum_complete_body_clearance
    CORE.UnicyclePlannerBase.evaluate_rollout_controls = V6._selection_evaluate
    V5.__file__ = str(Path(__file__).resolve())
    V5.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
