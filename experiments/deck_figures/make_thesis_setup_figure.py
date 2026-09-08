#!/usr/bin/env python3
"""The introduction's setup figure: the simulated warehouse from above.

Drawn from the world file itself. Rack footprints come from the occluder collision boxes,
camera positions and headings from their model poses, and the drivable floor from the
frozen capture grid. Nothing is hand-placed.

What each camera sees is a separate appendix figure, make_camera_views_figure.py.
"""
from __future__ import annotations

import pathlib
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
CAPTURE = REPO / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'

FRAME = 'camera_B/images/pose_001211_r00.png'
CAM_MODELS = {
    'external_camera': 'A', 'external_camera_b': 'B', 'external_camera_c': 'C',
    'external_camera_d': 'D', 'external_camera_e': 'E',
}
RACK, FLOOR, CAM, ROBOT, INK = '#8c96a3', '#ccd4dd', '#2367a2', '#c23d36', '#1d2530'


def read_world() -> tuple[list[tuple[float, float, float, float]], dict[str, tuple[float, float, float]]]:
    text = WORLD.read_text()

    racks = []
    block = text[text.index('warehouse_v2_occluders'):]
    block = block[:block.index('</model>')]
    for pose, size in re.findall(
            r'<pose>([^<]+)</pose>.*?<box><size>([^<]+)</size></box>', block, flags=re.S):
        px, py = (float(v) for v in pose.split()[:2])
        sx, sy = (float(v) for v in size.split()[:2])
        racks.append((px, py, sx, sy))

    cameras = {}
    for name, model, pose in re.findall(
            r'<include><name>([^<]+)</name><uri>model://([^<]+)</uri><pose>([^<]+)</pose>', text):
        if model in CAM_MODELS:
            parts = [float(v) for v in pose.split()]
            cameras[CAM_MODELS[model]] = (parts[0], parts[1], parts[5])
    return racks, cameras


def main() -> None:
    racks, cameras = read_world()
    rows = pd.read_csv(CAPTURE / 'observation_interpretations.csv')
    poses = rows.drop_duplicates('position_id')[['robot_x', 'robot_y']].to_numpy()
    row = rows[(rows.image == FRAME) & (rows.detected == 1)].iloc[0]

    fig, plan = plt.subplots(figsize=(5.6, 4.9))

    # ---- left: the warehouse from above
    plan.scatter(poses[:, 0], poses[:, 1], s=7, color=FLOOR, marker='s',
                 label='drivable floor', zorder=1)
    for px, py, sx, sy in racks:
        plan.add_patch(patches.Rectangle((px - sx / 2, py - sy / 2), sx, sy,
                                         facecolor=RACK, edgecolor='none', zorder=2))
    for name, (cx, cy, yaw) in sorted(cameras.items()):
        plan.plot(cx, cy, marker='o', ms=9, color=CAM, zorder=4)
        plan.annotate('', xy=(cx + 2.6 * np.cos(yaw), cy + 2.6 * np.sin(yaw)), xytext=(cx, cy),
                      arrowprops=dict(arrowstyle='-|>', lw=1.9, color=CAM), zorder=4)
        # push the label away from the arrow, then clamp inside the axes
        lx = float(np.clip(cx + 1.5 * np.sign(cx), -11.6, 11.6))
        ly = float(np.clip(cy + 1.4 * np.sign(cy), -10.6, 10.6))
        if name == 'A':  # bottom-left corner: keep the label clear of the axis frame
            lx, ly = cx + 1.5, cy - 0.9
        plan.text(lx, ly, name, color=CAM, fontsize=11, fontweight='bold',
                  ha='center', va='center', zorder=5)

    plan.plot(row.robot_x, row.robot_y, marker='*', ms=17, color=ROBOT, zorder=6)
    plan.annotate('robot', xy=(row.robot_x, row.robot_y), xytext=(row.robot_x + 1.2, row.robot_y - 2.6),
                  color=ROBOT, fontsize=10.5, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', lw=1.6, color=ROBOT), zorder=6)

    plan.add_patch(patches.Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor=RACK, ec='none'))
    plan.set_aspect('equal')
    plan.set(xlim=(-12.6, 12.6), ylim=(-11.4, 11.4))
    plan.set(xlabel='x (m)', ylabel='y (m)')
    plan.set_title('Five fixed cameras watch the warehouse floor\n'
                   'Each marker points where that camera looks',
                   fontsize=10.5, fontweight='bold')
    plan.grid(alpha=.18)
    handles = [patches.Patch(facecolor=RACK, label='storage rack'),
               patches.Patch(facecolor=FLOOR, label='drivable floor')]
    plan.legend(handles=handles, fontsize=8, loc='upper center',
                bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'thesis_setup.pdf', OUT / 'thesis_setup.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'thesis_setup.pdf')
    print(f'{len(racks)} rack boxes, {len(cameras)} cameras, {len(poses)} drivable positions')


if __name__ == '__main__':
    main()
