#!/usr/bin/env python3
"""Contact sheet of the five camera views, in the appendix-figure layout.

Reads the frames ``grab_frames.py`` wrote and lays them out two per row with the
camera name above and the robot's range from that camera below, matching
``figures/make_camera_views.py``.

That appendix script reads a FROZEN bbox-characterisation dataset, so it cannot
show a world that has just been edited. This one reads live frames instead, so
it is the right tool after a world change.

The robot box is drawn from GEOMETRY -- the spawn pose projected through each
camera's pinhole model -- not from a detector run, and a panel is only boxed
when the spawn is inside that camera's frustum with clear line of sight. A
camera that cannot see the spawn is labelled as such rather than boxed wrongly.

    python3 world/sheet.py <tag> [--spawn X Y]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / p) for p in ('src/unav_common', 'src/experiments')]

FRAMES = REPO / 'world/frames'
WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
ROBOT, INK = '#c23d36', '#1d2530'
TARGET_Z = 0.35          # the visibility model's target height on the robot
ROBOT_L, ROBOT_W, ROBOT_H = 0.80, 0.55, 0.40


def camera_rig(profile):
    """(label, x, y, z, pitch_rad, yaw_rad) per camera, from the world file."""
    labels = {inc: str(cid).replace('camera_', '')
              for inc, cid in zip(profile.get('camera_model_includes') or (),
                                  profile.get('camera_ids') or ())}
    rig = []
    for m in re.finditer(r'<include><name>(external_camera[a-z_]*)</name>'
                         r'<uri>model://[^<]*</uri>(.*?)</include>', WORLD.read_text()):
        pose = re.search(r'<pose>([^<]+)</pose>', m.group(2))
        if not pose:
            continue
        v = [float(t) for t in pose.group(1).split()]
        rig.append((labels.get(m.group(1), m.group(1)), v[0], v[1], v[2], v[4], v[5]))
    return sorted(rig)


def project(cam, point, width, height, fov_h):
    """Pinhole projection into image pixels; None when behind the camera.

    Gazebo's camera looks along +x of its own frame after yaw then pitch, so the
    point is rotated by -yaw about z and then by +pitch about the new y.
    """
    _label, cx, cy, cz, pitch, yaw = cam
    dx, dy, dz = point[0] - cx, point[1] - cy, point[2] - cz
    cy_, sy_ = math.cos(-yaw), math.sin(-yaw)
    fx, fy = dx * cy_ - dy * sy_, dx * sy_ + dy * cy_
    cp, sp = math.cos(pitch), math.sin(pitch)
    forward = fx * cp - dz * sp
    up = fx * sp + dz * cp
    if forward <= 1e-6:
        return None
    f_px = (width / 2) / math.tan(fov_h / 2)
    return (width / 2 - f_px * fy / forward, height / 2 - f_px * up / forward)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tag', nargs='?', default='iter')
    ap.add_argument('--spawn', nargs=2, type=float, default=[0.0, -5.0],
                    metavar=('X', 'Y'))
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    import yaml
    from unav_common.occlusion_geometry import parse_occlusion_scene_from_world

    profile = yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf']
    intr = yaml.safe_load(PROFILES.read_text())['camera_intrinsics']
    fov_h = float(intr['fov_h_rad'])
    fov_v = 2 * math.atan(math.tan(fov_h / 2) * intr['img_height'] / intr['img_width'])
    rig = camera_rig(profile)

    scene = parse_occlusion_scene_from_world(
        str(WORLD), model_name=('warehouse_v2_occluders',),
        geometry_tags=('collision',), robot_z_range=None)
    prisms = list(scene.prisms)

    sx, sy = args.spawn

    def clear(cam):
        _l, cx, cy, cz, _p, _y = cam
        a, b = np.array([cx, cy, cz]), np.array([sx, sy, TARGET_Z])
        for s in np.linspace(0.02, 0.98, 240):
            q = a + (b - a) * s
            for pr in prisms:
                if (pr.xmin <= q[0] <= pr.xmax and pr.ymin <= q[1] <= pr.ymax
                        and pr.zmin <= q[2] <= pr.zmax):
                    return False
        return True

    fig, axes = plt.subplots(3, 2, figsize=(9.6, 7.6))
    for ax in axes.ravel():
        ax.axis('off')

    for ax, cam in zip(axes.ravel(), rig):
        label, cx, cy, cz, pitch, yaw = cam
        path = FRAMES / f'{args.tag}_{label}.png'
        if not path.is_file():
            ax.text(.5, .5, f'missing {path.name}', ha='center', va='center')
            continue
        image = mpimg.imread(path)
        h, w = image.shape[:2]
        ax.imshow(image)
        ax.set_title(f'Camera {label}', fontsize=13, weight='bold', color=INK, pad=6)

        rng = math.hypot(sx - cx, sy - cy)
        dep = math.degrees(math.atan2(cz - TARGET_Z, rng))
        bearing = math.degrees(math.atan2(sy - cy, sx - cx))
        off = abs((bearing - math.degrees(yaw) + 180) % 360 - 180)
        in_fov = (abs(dep - math.degrees(pitch)) <= math.degrees(fov_v) / 2
                  and off <= math.degrees(fov_h) / 2)
        visible = in_fov and clear(cam)

        if visible:
            corners = []
            for ddx in (-ROBOT_L / 2, ROBOT_L / 2):
                for ddy in (-ROBOT_W / 2, ROBOT_W / 2):
                    for ddz in (0.02, ROBOT_H):
                        p = project(cam, (sx + ddx, sy + ddy, ddz), w, h, fov_h)
                        if p:
                            corners.append(p)
            if corners:
                us = [p[0] for p in corners]
                vs = [p[1] for p in corners]
                x0, x1 = max(min(us), 0), min(max(us), w)
                y0, y1 = max(min(vs), 0), min(max(vs), h)
                if x1 > x0 and y1 > y0:
                    ax.add_patch(patches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False, ec=ROBOT, lw=2.0))
            ax.set_xlabel(f'{rng:.1f} m away', fontsize=10, color=INK, labelpad=4)
        else:
            why = 'outside field of view' if not in_fov else 'occluded'
            ax.set_xlabel(f'robot not visible ({why})', fontsize=10, color='#8d2f26')
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    pitches = ', '.join(f'{lbl} {math.degrees(pit):.0f}\u00b0' for lbl, *_r, pit, _y in
                        [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in rig])
    fig.suptitle(f'warehouse_v2 after the 2026-09-18 edits — pitches {pitches}',
                 fontsize=12, weight='bold', color=INK, y=.995)
    fig.text(.5, .043,
             f'robot at ({sx:.2f}, {sy:.2f});  boxes are the projected 0.80 × 0.55 m body, '
             'drawn from geometry, not from a detector run',
             ha='center', fontsize=9, color='#555')
    fig.tight_layout(rect=(0, .055, 1, .975))

    out = args.out or (FRAMES / f'{args.tag}_sheet.png')
    fig.savefig(out, dpi=150)
    print(f'written: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
