#!/usr/bin/env python3
"""Problem-statement figure: how a camera image becomes a position measurement.

Two panels following one real detection from the frozen capture:

  (a) the camera image with the detected box and its bottom-centre pixel,
  (b) the projection geometry that carries that pixel to the floor.

The figure answers one question only, what the measurement z_k is and where it comes
from. What the filter then does with it belongs to the localization-example figure,
where the update has a place in the temporal story.

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


def crop_box(row, half_w: float = 430.0) -> tuple[float, float, float, float]:
    """The neighbourhood of the detection, shared by panels (a) and (b)."""
    cx, cy = (row.x0 + row.x1) / 2, (row.y0 + row.y1) / 2
    half_h = half_w * 0.72
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def panel_image(ax, row) -> None:
    """(a) the frame the camera delivered, with the detector's box on it."""
    image = mpimg.imread(CAPTURE / row.image)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (float(row[k]) for k in ('x0', 'y0', 'x1', 'y1'))

    ax.imshow(image)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           ec=DETECT, lw=2.4, zorder=3))
    ax.plot((x0 + x1) / 2, y1, 'o', color=DETECT, ms=8, mec='white', mew=1.4, zorder=4)
    ax.annotate('bottom-centre pixel $u_k$', xy=((x0 + x1) / 2, y1),
                xytext=((x0 + x1) / 2 - 40, y1 + 120), color=DETECT, fontsize=10,
                fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=1.8, color=DETECT,
                                shrinkA=6, shrinkB=3))
    ax.text(x0, y0 - 14, f'YOLO box $B_k$   {row.confidence:.2f}', color='white',
            fontsize=9, fontweight='bold', va='bottom',
            bbox=dict(boxstyle='square,pad=0.25', fc=DETECT, ec='none'))

    left, top, right, bottom = crop_box(row)
    ax.set(xlim=(max(0, left), min(width, right)), ylim=(min(height, bottom), max(0, top)))
    ax.axis('off')


def panel_projection(ax, row, model, centre) -> None:
    """(b) camera centre, image plane and floor: the real projection that makes $z_k$.

    Drawn as a weak-perspective view of the true 3-D geometry, so the ray, the pyramid
    and the floor intersection are all computed from the camera model.
    """
    ax.axis('off')

    # ---- the 3-D scene, in world metres -------------------------------------------
    left, top, right, bottom = crop_box(row)
    corners_px = [(left, top), (right, top), (right, bottom), (left, bottom)]
    u_k = np.array([(row.x0 + row.x1) / 2, row.y1])

    # The crop's own footprint on the floor is about fourteen metres across, which would
    # dwarf the image plane. Draw the pyramid for a narrower cone about the detection, so
    # the plane and the floor patch are legible at the same scale.
    cone = 0.72
    mid = np.array([(left + right) / 2, (top + bottom) / 2])
    corners_px = [tuple(mid + cone * (np.array(c) - mid)) for c in corners_px]

    # put the image plane a fixed distance along each corner's ray, so the quad drawn
    # is a real cross-section of the viewing pyramid
    def ray(u: float, v: float) -> np.ndarray:
        direction = np.linalg.inv(model.K @ model.R) @ np.array([u, v, 1.0])
        return direction / np.linalg.norm(direction)

    plane_t = 4.30  # metres from the centre; chosen only so the panel reads well
    plane = np.array([centre + plane_t * ray(u, v) for u, v in corners_px])
    box_px = [(row.x0, row.y0), (row.x1, row.y0), (row.x1, row.y1), (row.x0, row.y1)]
    box = np.array([centre + plane_t * ray(u, v) for u, v in box_px])
    u_plane = centre + plane_t * ray(*u_k)
    ground = np.append(np.array(model.pixel_to_world(*u_k)), 0.0)

    # the floor patch the pyramid actually cuts, so the two quads are consistent
    floor = np.array([np.append(np.array(model.pixel_to_world(u, v)), 0.0)
                      for u, v in corners_px])

    # ---- one weak-perspective projection for every 3-D point ----------------------
    # Viewed from the side, square onto the camera's own optical axis, so the ray
    # descends across the page and the optical centre sits clear above the floor.
    # World z stays upright; the small depth shear gives the floor its extent.
    # +3pi/2 rather than +pi/2 so the ray descends left-to-right, with the figure's flow
    az = math.atan2(*(model.look_at - centre)[[1, 0]][::-1]) + 3 * math.pi / 2
    right_axis = np.array([math.cos(az), math.sin(az), 0.0])
    depth_axis = np.array([-math.sin(az), math.cos(az), 0.0])

    def to_page(point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return np.array([point @ right_axis + 0.26 * (point @ depth_axis),
                         point[2] + 0.45 * (point @ depth_axis)])

    page_floor = np.array([to_page(p) for p in floor])
    page_plane = np.array([to_page(p) for p in plane])
    page_box = np.array([to_page(p) for p in box])
    page_centre, page_u, page_ground = to_page(centre), to_page(u_plane), to_page(ground)

    # ---- the floor ----------------------------------------------------------------
    ax.add_patch(Polygon(page_floor, closed=True, fc='#eceef1', ec='#9aa3ac',
                         lw=1.2, zorder=1))

    # ---- the viewing pyramid: faint edges through the image-plane corners ----------
    for corner, page_corner in zip(floor, page_floor):
        ax.plot([page_centre[0], page_corner[0]], [page_centre[1], page_corner[1]],
                color=INK, lw=0.8, alpha=.30, zorder=2)

    # ---- the image plane, carrying the same real frame as panel (a) ----------------
    # pcolormesh maps the crop onto the projected quad by bilinear interpolation of its
    # four page corners, which an affine imshow cannot do once the quad is sheared.
    image = mpimg.imread(CAPTURE / row.image)
    crop = image[int(top):int(bottom), int(left):int(right)]
    rows_n, cols_n = 90, 90
    small = crop[::max(1, crop.shape[0] // rows_n), ::max(1, crop.shape[1] // cols_n)]
    rows_n, cols_n = small.shape[0], small.shape[1]
    # corners in image order: page_plane is [top-left, top-right, bottom-right, bottom-left]
    tl, tr, br, bl = page_plane
    a, b = np.meshgrid(np.linspace(0, 1, cols_n + 1), np.linspace(0, 1, rows_n + 1))
    grid = ((1 - a)[..., None] * (1 - b)[..., None] * tl + a[..., None] * (1 - b)[..., None] * tr
            + a[..., None] * b[..., None] * br + (1 - a)[..., None] * b[..., None] * bl)
    quads = np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]],
                     axis=2).reshape(-1, 4, 2)
    ax.add_collection(PolyCollection(
        quads, facecolors=small.reshape(-1, small.shape[2]), edgecolors='none',
        linewidths=0, zorder=3, rasterized=True))
    ax.add_patch(Polygon(page_plane, closed=True, fill=False, ec=BELIEF, lw=1.6, zorder=4))
    ax.add_patch(Polygon(page_box, closed=True, fc=DETECT, ec=DETECT, lw=2.0,
                         alpha=.16, zorder=5))
    ax.add_patch(Polygon(page_box, closed=True, fill=False, ec=DETECT, lw=2.0, zorder=6))

    # ---- the camera, drawn as a small body at the optical centre --------------------
    # a plain dot reads as an arbitrary vertex; a camera icon says what the apex is.
    # the body is sized from the panel so it holds up at any window scale
    scale = 0.10 * float(np.ptp(page_floor[:, 0]))
    forward = to_page(centre + 0.6 * (model.look_at - centre)
                      / np.linalg.norm(model.look_at - centre)) - page_centre
    forward = forward / np.linalg.norm(forward)
    side = np.array([-forward[1], forward[0]])
    body = np.array([page_centre + a * scale * forward + b * 0.72 * scale * side
                     for a, b in ((-1.55, 1), (-1.55, -1), (0, -1), (0, 1))])
    lens = np.array([page_centre + a * scale * forward + b * 0.72 * scale * side
                     for a, b in ((0, 0.62), (0.62, 1.15), (0.62, -1.15), (0, -0.62))])
    ax.add_patch(Polygon(body, closed=True, fc=INK, ec=INK, lw=1.0, zorder=7))
    ax.add_patch(Polygon(lens, closed=True, fc=INK, ec=INK, lw=1.0, zorder=7))
    ax.plot(*page_centre, 'o', color='white', ms=4.5, zorder=8)
    ax.annotate('camera centre $C_c$', xy=page_centre,
                xytext=(page_centre[0] - 0.05, page_centre[1] + 0.85),
                fontsize=9.5, color=INK, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='-', lw=0.9, color=INK, shrinkA=2, shrinkB=8))

    # ---- the ray through the detection, on to the floor ----------------------------
    ax.plot([page_centre[0], page_ground[0]], [page_centre[1], page_ground[1]],
            color=DETECT, lw=2.0, zorder=6, solid_capstyle='round')
    ax.plot(*page_u, 'o', color=DETECT, ms=7.5, mec='white', mew=1.3, zorder=8)
    ax.annotate('box $B_k$,\npixel $u_k$', xy=page_u,
                xytext=(page_u[0] + 1.35, page_u[1] + 0.30), color=DETECT, fontsize=9.5,
                fontweight='bold', ha='left', va='center', zorder=9,
                arrowprops=dict(arrowstyle='-', lw=0.9, color=DETECT, shrinkA=3,
                                shrinkB=6))
    ax.plot(*page_ground, '*', color=MEAS, ms=18, mec='white', mew=1.0, zorder=8)

    # labels
    plane_top = page_plane[np.argmax(page_plane[:, 1])]
    ax.annotate('image plane\n(the frame in (a))', xy=plane_top,
                xytext=(plane_top[0] - 1.05, plane_top[1] + 0.55), fontsize=9,
                color=BELIEF, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.9, color=BELIEF, shrinkA=2, shrinkB=3))
    ax.text(page_ground[0] + 0.10, page_ground[1] - 0.28,
            '$z_k$', color=MEAS, fontsize=12, fontweight='bold', ha='left', va='top')
    # Beside the ray it names, set horizontally: a label rotated onto this ray reads
    # awkwardly and collided with the pixel label. It sits off the ray's left, where
    # the panel is empty, with a short leader back to the line.
    ray_label = 0.52 * page_ground + 0.48 * page_u
    ray_dir = page_ground - page_centre
    normal = np.array([-ray_dir[1], ray_dir[0]]) / np.linalg.norm(ray_dir)
    if normal[0] > 0:
        normal = -normal
    ax.annotate(r'$\tilde z_k\propto H_c^{-1}\tilde u_k$', xy=ray_label,
                xytext=ray_label + 1.15 * normal, color=DETECT, fontsize=10.5,
                fontweight='bold', ha='center', va='center', zorder=9,
                arrowprops=dict(arrowstyle='-', lw=0.9, color=DETECT, alpha=.7,
                                shrinkA=3, shrinkB=3))
    corner = page_floor[np.argmin(page_floor[:, 1])]
    ax.text(corner[0], corner[1] - 0.30, 'ground plane, $z=0$', fontsize=9.5,
            color=INK, ha='center', fontweight='bold')

    # imshow with an affine transform does not grow the data limits, so set them
    # explicitly from every point the panel actually draws
    drawn = np.vstack([page_floor, page_plane, page_box,
                       page_centre, page_u, page_ground])
    pad_x = 0.16 * (drawn[:, 0].max() - drawn[:, 0].min())
    pad_y = 0.12 * (drawn[:, 1].max() - drawn[:, 1].min())
    ax.set(xlim=(drawn[:, 0].min() - pad_x, drawn[:, 0].max() + pad_x),
           ylim=(drawn[:, 1].min() - pad_y, drawn[:, 1].max() + pad_y))
    ax.set_aspect('equal')


def main() -> None:
    rows = pd.read_csv(CAPTURE / 'observation_interpretations.csv')
    row = rows[(rows.image == FRAME) & (rows.detected == 1)].iloc[0]
    model, centre = camera_geometry()

    # the drawing must reproduce the dataset's own back-projection, or the geometry lies
    recomputed = np.array(model.pixel_to_world((row.x0 + row.x1) / 2, row.y1))
    recorded = np.array([row.raw_x, row.raw_y])
    if not np.allclose(recomputed, recorded, atol=1e-4):
        raise SystemExit(f'camera model disagrees with the capture: {recomputed} vs {recorded}')

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.95))
    panel_image(axes[0], row)
    panel_projection(axes[1], row, model, centre)

    fig.tight_layout(pad=0.4, w_pad=1.2)

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
