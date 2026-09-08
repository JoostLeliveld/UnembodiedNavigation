#!/usr/bin/env python3
"""Problem-statement figure: how a camera image becomes a position measurement.

Two panels following one real detection from the frozen capture:

  (a) the whole camera frame, uncropped, with the detected box and its bottom pixel,
  (b) how that pixel becomes a position on the warehouse map.

The figure answers one question only, what the measurement z_k is and where it comes
from. What the filter then does with it belongs to the localization-example figure,
where the update has a place in the temporal story.

Panel (a) is documentary: the frame at its native aspect, so the reader sees how small
the robot is in the camera's view and how much of the hall the camera covers. Panel (b)
is explanatory rather than a strict perspective construction. It stacks the three things
that happen, camera to image, image pixel to world, world point on the map, and the
homography is drawn as its own step rather than as a ray pretending to be it. The image
plane carries the real frame and the ground level is the planner's real driveable map,
so both levels are the actual data even though the stack is a diagram.

Every geometric element is computed, not drawn by hand. Panel (b) places the real
camera_B optical centre, its real viewing pyramid and the real image plane in world
coordinates, then draws the ray through the detected pixel to its floor intersection;
that intersection reproduces the dataset's recorded back-projection to seven decimals.
The image plane carries a crop of the same real frame shown in (a), warped onto the
plane, so the reader sees that it is literally the image from (a). Panel (c) is the
world's own plan-view render with the real robot pose, the real measurement and the
real innovation on it. Only the covariances are chosen for legibility: the capture is
a static pose grid, so it has no filter belief to read a prior from, and the point of
the panel is the mechanism of the update.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src' / 'unav_common'))
from paths import repo_root  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

REPO = repo_root()
CAPTURE = REPO / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
FRAME = 'camera_B/images/pose_001211_r00.png'
CAMERA = 'camera_B'
FOV_H_RAD = 1.5708  # external_camera_b/model.sdf

DETECT, MEAS, BELIEF, INK = '#c23d36', '#c23d36', '#1f6fb8', '#1d2530'
DRIVE, ZONE, ZONE_EDGE = '#cfe3f5', '#f0e2c8', '#b08a4a'
MAP_SPAN = (25.2, 21.2)   # the map's drawn extent, metres


def camera_geometry() -> tuple[ObliqueCameraModel, np.ndarray]:
    """The real camera_B, built from its captured pose. Returns the model and its centre."""
    entry = next(c for c in json.loads((CAPTURE / 'capture_manifest.json').read_text())['cameras']
                 if c['camera_id'] == CAMERA)
    x, y, z, _, pitch, yaw = entry['pose_xyz_rpy']
    centre = np.array([x, y, z])
    # the optical axis, marched from the centre to where it meets the floor
    axis = np.array([math.cos(pitch) * math.cos(yaw),
                     math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    look_at = centre + axis * (z / -axis[2])
    model = ObliqueCameraModel(cam_pos=centre, look_at=look_at,
                               img_width=entry['image_width'], img_height=entry['image_height'],
                               fov_h_rad=FOV_H_RAD)
    return model, centre


def panel_image(ax, row) -> None:
    """(a) the whole frame the camera delivered, with the detector's box on it.

    Uncropped and at its native aspect: the point is how small the robot is in the
    camera's view and how much of the hall one camera covers.
    """
    image = mpimg.imread(CAPTURE / row.image)
    x0, y0, x1, y1 = (float(row[k]) for k in ('x0', 'y0', 'x1', 'y1'))

    ax.imshow(image)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           ec=DETECT, lw=2.0, zorder=3))
    ax.plot((x0 + x1) / 2, y1, 'o', color=DETECT, ms=6, mec='white', mew=1.2, zorder=4)
    ax.annotate('bottom-centre pixel $u_k$', xy=((x0 + x1) / 2, y1),
                xytext=((x0 + x1) / 2 - 250, y1 + 105), color=DETECT, fontsize=9.5,
                fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=1.6, color=DETECT,
                                shrinkA=4, shrinkB=3))
    ax.text(x1 + 10, y0, f'$B_k$   {row.confidence:.2f}', color='white', fontsize=8.5,
            fontweight='bold', va='top', ha='left', zorder=4,
            bbox=dict(boxstyle='square,pad=0.24', fc=DETECT, ec='none'))
    ax.set_axis_off()
    # the 16:9 frame is shorter than the panel, which is sized for (b)'s stack; sit it
    # at the top so the two panels start on the same line instead of floating centred
    ax.set_anchor('N')


def driveable_map():
    """The planner's own driveable map, so panel (b)'s floor is the real one."""
    here = pathlib.Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[2] / 'experiments' / 'warehouse_v2_sketches'))
    import route_tasks as rt  # noqa: E402
    from experiments.core.world_profiles import load_world_profiles  # noqa: E402
    xs, ys, mask, _ = rt.driveable()
    regions = load_world_profiles(str(rt.PROFILES))['worlds'][rt.WORLD_KEY].get(
        'known_2d_regions', []) or []
    zones = [r for r in regions
             if str(r.get('type', '')).strip().lower() == 'non_driveable_obstacle']
    return xs, ys, mask, zones, rt.RES


def panel_projection(ax, row, model, centre) -> None:
    """(b) how the detected pixel becomes a position on the warehouse map.

    Three stacked levels rather than one perspective construction: the camera and the
    image it forms, then the homography as its own labelled step, then the world map
    the resulting point lands on. The image is the real frame and the map is the
    planner's real driveable map; the stack between them is a diagram.
    """
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect='auto')

    image = mpimg.imread(CAPTURE / row.image)
    height, width = image.shape[:2]
    u_k = np.array([(row.x0 + row.x1) / 2, row.y1])

    # ---- level 1: the camera, and the image it forms ------------------------------
    # The three levels are laid out from one height budget rather than by hand, so the
    # frame and the map each keep their own proportions and never overlap. Both are
    # drawn in axes coordinates with aspect='auto', so a given width fraction implies
    # the height fraction through the axes' own inch dimensions.
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    ax_w, ax_h = fig_w * box.width, fig_h * box.height

    def height_for(width_frac, aspect):
        # height fraction that renders this aspect (height/width) at this width
        return width_frac * ax_w * aspect / ax_h

    plane_w, map_w = 0.80, 0.55
    plane_h = height_for(plane_w, height / width)
    map_h = height_for(map_w, MAP_SPAN[1] / MAP_SPAN[0])
    cam_band, gap = 0.085, 0.075          # the camera above, the homography between
    needed = cam_band + plane_h + gap + map_h
    if needed > 1.0:
        raise SystemExit(f'panel (b) does not fit: needs {needed:.3f} of its height')
    plane_y1 = 1.0 - cam_band
    plane = dict(x0=0.5 - plane_w / 2, x1=0.5 + plane_w / 2,
                 y0=plane_y1 - plane_h, y1=plane_y1)
    cam = np.array([0.50, 1.0 - cam_band / 2])

    ax.imshow(image, extent=(plane['x0'], plane['x1'], plane['y0'], plane['y1']),
              aspect='auto', zorder=2, alpha=.92)
    ax.add_patch(Rectangle((plane['x0'], plane['y0']), plane['x1'] - plane['x0'],
                           plane['y1'] - plane['y0'], fill=False, ec=BELIEF, lw=1.6,
                           zorder=4))

    def on_plane(u: float, v: float) -> np.ndarray:
        """A pixel of the frame, placed on the drawn image plane."""
        return np.array([plane['x0'] + (plane['x1'] - plane['x0']) * u / width,
                         plane['y1'] - (plane['y1'] - plane['y0']) * v / height])

    # the same detection as panel (a), on that image
    box_lo, box_hi = on_plane(row.x0, row.y1), on_plane(row.x1, row.y0)
    ax.add_patch(Rectangle(box_lo, box_hi[0] - box_lo[0], box_hi[1] - box_lo[1],
                           fill=False, ec=DETECT, lw=1.8, zorder=5))
    page_u = on_plane(*u_k)
    ax.plot(*page_u, 'o', color=DETECT, ms=6.5, mec='white', mew=1.2, zorder=6)

    # two thin lines only: enough to say this is the image this camera sees
    for corner in (plane['x0'], plane['x1']):
        ax.plot([cam[0], corner], [cam[1], plane['y1']], color=INK, lw=0.8,
                alpha=.35, zorder=1)

    # the camera body, pointing down at the image it forms
    body_w, body_h = 0.055, 0.032
    ax.add_patch(Rectangle((cam[0] - body_w, cam[1] - body_h / 2), 1.5 * body_w, body_h,
                           fc=INK, ec=INK, zorder=7))
    ax.add_patch(Polygon([(cam[0] + 0.5 * body_w, cam[1] - body_h / 2),
                          (cam[0] + 1.3 * body_w, cam[1] - body_h),
                          (cam[0] + 1.3 * body_w, cam[1] + body_h),
                          (cam[0] + 0.5 * body_w, cam[1] + body_h / 2)],
                         closed=True, fc=INK, ec=INK, zorder=7))
    ax.text(cam[0] - 1.4 * body_w, cam[1], 'camera $c$', color=INK, fontsize=9.5,
            fontweight='bold', ha='right', va='center', zorder=7)

    ax.text(plane['x0'] - 0.012, (plane['y0'] + plane['y1']) / 2, 'image plane',
            color=BELIEF, fontsize=9, ha='right', va='center', rotation=90, zorder=6)
    ax.annotate('box $B_k$, pixel $u_k$', xy=page_u,
                xytext=(page_u[0] - 0.10, plane['y0'] - 0.030), color=DETECT,
                fontsize=9.2, fontweight='bold', ha='center', va='top', zorder=8,
                arrowprops=dict(arrowstyle='-', lw=0.9, color=DETECT, shrinkA=2,
                                shrinkB=4))

    # ---- level 2: the homography, as its own step ---------------------------------
    arrow_x = 0.26
    band_hi, band_lo = plane['y0'] - 0.012, plane['y0'] - gap + 0.012
    ax.annotate('', xy=(arrow_x, band_lo), xytext=(arrow_x, band_hi),
                arrowprops=dict(arrowstyle='-|>,head_width=0.34,head_length=0.8',
                                lw=3.0, color=DETECT), zorder=8)
    ax.text(arrow_x + 0.035, (band_hi + band_lo) / 2,
            r'$\tilde z_k\propto H_c^{-1}\tilde u_k$', color=DETECT, fontsize=11,
            fontweight='bold', ha='left', va='center', zorder=8)

    # ---- level 3: the driveable map the point lands on ----------------------------
    xs, ys, mask, zones, res = driveable_map()
    span_x, span_y = MAP_SPAN
    map_y1 = plane['y0'] - gap
    world = dict(x0=0.5 - map_w / 2, x1=0.5 + map_w / 2,
                 y0=map_y1 - map_h, y1=map_y1)

    def on_map(x: float, y: float) -> np.ndarray:
        return np.array([world['x0'] + (world['x1'] - world['x0']) * (x + span_x / 2) / span_x,
                         world['y0'] + (world['y1'] - world['y0']) * (y + span_y / 2) / span_y])

    lo, hi = on_map(-span_x / 2, -span_y / 2), on_map(span_x / 2, span_y / 2)
    ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fc='white', ec='none',
                           zorder=2))
    edge_x = np.concatenate([xs - res / 2, [xs[-1] + res / 2]])
    edge_y = np.concatenate([ys - res / 2, [ys[-1] + res / 2]])
    grid_x = np.interp(edge_x, [-span_x / 2, span_x / 2], [lo[0], hi[0]])
    grid_y = np.interp(edge_y, [-span_y / 2, span_y / 2], [lo[1], hi[1]])
    ax.pcolormesh(grid_x, grid_y, np.ma.masked_where(~mask, mask.astype(float)),
                  cmap=matplotlib.colors.ListedColormap([DRIVE]), vmin=0, vmax=1,
                  shading='flat', zorder=3, rasterized=True)
    for zone in zones:
        z_lo, z_hi = on_map(zone['xmin'], zone['ymin']), on_map(zone['xmax'], zone['ymax'])
        ax.add_patch(Rectangle(z_lo, z_hi[0] - z_lo[0], z_hi[1] - z_lo[1],
                               fc=ZONE, ec=ZONE_EDGE, lw=0.6, zorder=4))
    ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fill=False, ec='#8d949c',
                           lw=1.0, zorder=6))

    # the camera that took the frame, on the map it looks at
    cam_xy = on_map(centre[0], centre[1])
    look = model.look_at - centre
    look_dir = look[:2] / np.linalg.norm(look[:2])
    ax.plot(*cam_xy, marker='s', ms=5, color=BELIEF, mec='white', mew=1.0, zorder=8)
    ax.annotate('', xy=on_map(*(centre[:2] + 2.6 * look_dir)), xytext=cam_xy,
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.5',
                                lw=1.4, color=BELIEF, alpha=.85, shrinkA=3, shrinkB=0),
                zorder=8)
    ax.text(cam_xy[0] - 0.018, cam_xy[1] - 0.004, 'c', color=BELIEF, fontsize=8,
            fontweight='bold', ha='right', va='center', zorder=8)

    # the measurement itself
    ground = np.array(model.pixel_to_world(*u_k))
    z_xy = on_map(*ground)
    ax.plot(*z_xy, '*', color=MEAS, ms=16, mec='white', mew=1.0, zorder=9)
    ax.annotate('camera position\nmeasurement $z_k$', xy=z_xy,
                xytext=(z_xy[0] + 0.115, z_xy[1] - 0.055), color=MEAS, fontsize=9.2,
                fontweight='bold', ha='left', va='center', zorder=10,
                bbox=dict(boxstyle='square,pad=0.16', fc='white', ec='none', alpha=.85),
                arrowprops=dict(arrowstyle='-', lw=0.9, color=MEAS, shrinkA=3,
                                shrinkB=5))
    ax.text(world['x0'] - 0.012, (world['y0'] + world['y1']) / 2, 'warehouse map',
            color=INK, fontsize=9, ha='right', va='center', rotation=90, zorder=6)


def main() -> None:
    rows = pd.read_csv(CAPTURE / 'observation_interpretations.csv')
    row = rows[(rows.image == FRAME) & (rows.detected == 1)].iloc[0]
    model, centre = camera_geometry()

    # the drawing must reproduce the dataset's own back-projection, or the geometry lies
    recomputed = np.array(model.pixel_to_world((row.x0 + row.x1) / 2, row.y1))
    recorded = np.array([row.raw_x, row.raw_y])
    if not np.allclose(recomputed, recorded, atol=1e-4):
        raise SystemExit(f'camera model disagrees with the capture: {recomputed} vs {recorded}')

    # (a) documentary and (b) explanatory, at roughly 45/55: the stack in (b) needs
    # the height, the frame in (a) needs the width of its own 16:9 aspect
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.0),
                             gridspec_kw=dict(width_ratios=(0.45, 0.55)))
    panel_image(axes[0], row)
    panel_projection(axes[1], row, model, centre)

    fig.tight_layout(pad=0.4, w_pad=1.6)

    # One title row for both panels: they have different data aspects, so an
    # axes-relative title would sit at a different page height in each.
    titles = ('(a) camera detection', '(b) ground-plane projection')
    title_y = max(ax.get_position().y1 for ax in axes) + 0.045
    for ax, title in zip(axes, titles):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, title_y, title, fontsize=10.5,
                 fontweight='bold', ha='center', va='bottom')
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'measurement_chain.pdf', OUT / 'measurement_chain.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'measurement_chain.pdf')
    print(f'frame {FRAME}, range {row.camera_range_m:.2f} m, confidence {row.confidence:.3f}')
    print(f'back-projection matches capture: {recomputed} == {recorded}')


if __name__ == '__main__':
    main()
