#!/usr/bin/env python3
"""The introduction's setup figure: the simulated warehouse seen from above.

A real Gazebo frame from the world's own overhead camera, so the reader sees the
simulation rather than a schematic of it. The five wall-mounted cameras are visible in the
render as the grey cones on the walls; they are not annotated, because the plan camera is a
perspective view and projecting world coordinates into it would need its full calibration
rather than an eyeballed fit.

Recapture the frame with the simulation running:

    ros2 launch sim bringup_sim.launch.py world:=warehouse_v2.world.sdf
    ign topic -e -t /plan_view_camera/image_raw -n 1 --json-output > plan.json

then decode it to gazebo_plan_view.png. See scripts/paper_figures/capture_overview_frame.py.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
PLAN = OUT / 'gazebo_plan_view.png'



def main() -> None:
    if not PLAN.exists():
        raise SystemExit(f'missing {PLAN}; capture it from a running simulation first')

    image = mpimg.imread(PLAN)
    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(7.4, 7.4 * height / width))
    ax.imshow(image)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')

    fig.tight_layout(pad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'thesis_setup.pdf', OUT / 'thesis_setup.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'thesis_setup.pdf')


if __name__ == '__main__':
    main()
