#!/usr/bin/env python3
"""The introduction's setup figure: the simulated warehouse seen from above.

A real Gazebo frame from the world's own overhead camera, so the reader sees the
simulation rather than a schematic of it. The five wall-mounted cameras are labelled and
their viewing directions drawn.

The label positions are read by hand from this particular frame. The overhead camera is a
perspective view whose rendered pose does not match the world file's include pose, so
projecting world coordinates onto it gave markers that missed the camera bodies. Rather
than fit an offset until the markers looked right, the five pixel positions below were
measured off the render and checked against it. Recapturing the frame invalidates them.

Recapture, if the world changes:

    ros2 launch sim bringup_sim.launch.py world:=warehouse_v2.world.sdf
    ign topic -e -t /plan_view_camera/image_raw -n 1 --json-output > plan.json

then decode it to gazebo_plan_view.png with scripts/paper_figures/capture_overview_frame.py
and re-measure CAMERA_PX.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
PLAN = OUT / 'gazebo_plan_view.png'

# Camera body centres in gazebo_plan_view.png, measured from the render, and the direction
# each camera faces in image coordinates (dx right, dy down), from its world yaw.
CAMERA_PX = {
    'A': ((374, 926), (0.72, -0.70)),
    'B': ((830, 908), (-0.42, -0.91)),
    'C': ((830, 212), (0.50, 0.87)),
    'D': ((1198, 496), (-0.76, 0.65)),
    'E': ((1084, 926), (-0.67, -0.74)),
}
# Content bounds of the render: everything outside is empty background.
CROP = (185, 85, 1415, 1115)
ARROW_PX = 118
CAM = '#1f6fb8'


def main() -> None:
    if not PLAN.exists():
        raise SystemExit(f'missing {PLAN}; capture it from a running simulation first')

    left, top, right, bottom = CROP
    image = mpimg.imread(PLAN)[top:bottom, left:right]
    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(7.2, 7.2 * height / width))
    ax.imshow(image)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')

    for name, ((ux, uy), (dx, dy)) in sorted(CAMERA_PX.items()):
        px, py = ux - left, uy - top
        ax.annotate('', xy=(px + ARROW_PX * dx, py + ARROW_PX * dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.34,head_length=0.7',
                                    lw=3.0, color=CAM,
                                    shrinkA=13, shrinkB=0))
        ax.plot(px, py, marker='o', ms=13, color=CAM, mec='white', mew=2.0, zorder=5)
        # label on the far side of the camera from its arrow, so the two never overlap
        ax.text(px - 44 * dx, py - 44 * dy, name, color='white', fontsize=13,
                fontweight='bold', ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='circle,pad=0.26', fc=CAM, ec='white', lw=1.8))

    fig.tight_layout(pad=0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'thesis_setup.pdf', OUT / 'thesis_setup.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    print('wrote', OUT / 'thesis_setup.pdf')


if __name__ == '__main__':
    main()
