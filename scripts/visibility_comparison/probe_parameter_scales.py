#!/usr/bin/env python3
"""Deterministic model scales; no run selection, observations or performance claims."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for package in ('planning', 'unav_common'):
    sys.path.insert(0, str(ROOT / 'src' / package))

from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel


def probe():
    geometry = json.dumps({'prisms': [dict(xmin=-10, xmax=10, ymin=-.9, ymax=.9,
                                         zmin=0, zmax=1)]})
    model = NogoZoneCostModel(NogoCostConfig(
        mode='keep_in', geometry_json=geometry, weight=40, safe_distance=.55))
    clearances = (.1, .05, .025, 0., -.001, -.005, -.01, -.05)
    return {
        'scope': 'analytic configuration/model probe; not observed navigation',
        'nogo_penalties': [dict(clearance_m=c, cost=model._penalty_from_clearance_np(c))
                           for c in clearances],
        'lane_width_scales': [dict(width_m=w, centre_clearance_m=w/2-.55,
                                  zero_penalty_centre=w/2-.55 >= .05-1e-12)
                              for w in (1., 1.1, 1.2, 1.8)],
        'discount_scales': [dict(dt_s=dt, half_weight_time_s=dt*math.log(.5)/math.log(.98),
                                weight_after_100_steps=.98**100,
                                weight_after_200_steps=.98**200)
                            for dt in (.25, 1.)],
        'speed_scales': [dict(v_max_mps=v, nominal_capture_spacing_m=v/5,
                             local_step_m=v*.25, local_3s_reach_m=v*3,
                             global_200s_reach_m=v*200,
                             watchdog_0_5s_travel_m=v*.5)
                         for v in (.22, .44)],
    }


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2, allow_nan=False))
