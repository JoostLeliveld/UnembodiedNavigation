#!/usr/bin/env python3
"""Problem-statement figure: how a camera image becomes a position measurement.

Three panels following one real detection from the frozen capture:

  (a) the camera image with the detected box and its bottom-centre pixel,
  (b) the projection geometry that carries that pixel to the floor,
  (c) where the resulting measurement lands in the warehouse, and the filter update.

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
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src' / 'unav_common'))
from paths import repo_root  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

REPO = repo_root()
CAPTURE = REPO / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
PLAN = OUT / 'gazebo_plan_view.png'
FRAME = 'camera_B/images/pose_001211_r00.png'
CAMERA = 'camera_B'
FOV_H_RAD = 1.5708  # external_camera_b/model.sdf

DETECT, MEAS, BELIEF, INK = '#c23d36', '#c23d36', '#1f6fb8', '#1d2530'
TRUTH = '#1d2530'

# the world's plan-view camera, as documented in make_thesis_setup_figure.py
PLAN_HEIGHT_M, PLAN_F, PLAN_PRINCIPAL = 42.0, 1656.1258, (800.0, 600.0)


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
    ax.annotate('$u_k$', xy=((x0 + x1) / 2, y1), xytext=((x0 + x1) / 2 - 150, y1 + 88),
                color=DETECT, fontsize=12, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=1.8, color=DETECT,
                                shrinkA=6, shrinkB=3))
    ax.text(x0, y0 - 14, f'{row.confidence:.2f}', color='white', fontsize=9,
            fontweight='bold', va='bottom',
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
    cone = 0.60
    mid = np.array([(left + right) / 2, (top + bottom) / 2])
    corners_px = [tuple(mid + cone * (np.array(c) - mid)) for c in corners_px]

    # put the image plane a fixed distance along each corner's ray, so the quad drawn
    # is a real cross-section of the viewing pyramid
    def ray(u: float, v: float) -> np.ndarray:
        direction = np.linalg.inv(model.K @ model.R) @ np.array([u, v, 1.0])
        return direction / np.linalg.norm(direction)

    plane_t = 3.55  # metres from the centre; chosen only so the panel reads well
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
    ax.add_patch(Polygon(page_box, closed=True, fill=False, ec=DETECT, lw=1.8, zorder=5))

    # ---- the camera centre ---------------------------------------------------------
    ax.plot(*page_centre, 'o', color=INK, ms=9, mec='white', mew=1.4, zorder=7)
    ax.annotate('camera centre $C_c$', xy=page_centre,
                xytext=(page_centre[0] - 0.15, page_centre[1] + 0.72),
                fontsize=9.5, color=INK, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='-', lw=0.9, color=INK, shrinkA=2, shrinkB=5))

    # ---- the ray through the detection, on to the floor ----------------------------
    ax.plot([page_centre[0], page_ground[0]], [page_centre[1], page_ground[1]],
            color=DETECT, lw=2.0, zorder=6, solid_capstyle='round')
    ax.plot(*page_u, 'o', color=DETECT, ms=7.5, mec='white', mew=1.3, zorder=8)
    ax.annotate('$u_k$', xy=page_u, xytext=(page_u[0] - 0.92, page_u[1] - 0.42),
                color=DETECT, fontsize=12, fontweight='bold', ha='center', va='center',
                zorder=9, arrowprops=dict(arrowstyle='-', lw=0.9, color=DETECT,
                                          shrinkA=3, shrinkB=5))
    ax.plot(*page_ground, '*', color=MEAS, ms=18, mec='white', mew=1.0, zorder=8)

    # labels
    plane_top = page_plane[np.argmax(page_plane[:, 1])]
    ax.annotate('image plane\n(the frame in (a))', xy=plane_top,
                xytext=(plane_top[0] - 1.05, plane_top[1] + 0.55), fontsize=9,
                color=BELIEF, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.9, color=BELIEF, shrinkA=2, shrinkB=3))
    ax.text(page_ground[0] + 0.10, page_ground[1] - 0.28,
            '$z_k$', color=MEAS, fontsize=12, fontweight='bold', ha='left', va='top')
    ray_label = 0.68 * page_ground + 0.32 * page_centre
    ax.text(ray_label[0] + 0.30, ray_label[1] + 0.30,
            r'$\tilde z_k\propto H_c^{-1}\tilde u_k$', color=INK, fontsize=10.5,
            fontweight='bold', ha='left', va='bottom')
    corner = page_floor[np.argmin(page_floor[:, 1])]
    ax.text(corner[0], corner[1] - 0.30, 'floor, $z=0$', fontsize=9, color=INK, ha='center')

    # imshow with an affine transform does not grow the data limits, so set them
    # explicitly from every point the panel actually draws
    drawn = np.vstack([page_floor, page_plane, page_box,
                       page_centre, page_u, page_ground])
    pad_x = 0.16 * (drawn[:, 0].max() - drawn[:, 0].min())
    pad_y = 0.12 * (drawn[:, 1].max() - drawn[:, 1].min())
    ax.set(xlim=(drawn[:, 0].min() - pad_x, drawn[:, 0].max() + pad_x),
           ylim=(drawn[:, 1].min() - pad_y, drawn[:, 1].max() + pad_y))
    ax.set_aspect('equal')


def panel_update(ax, row, camera_centre):
    """(c) the same measurement in the warehouse, and the update it drives."""
    image = mpimg.imread(PLAN)

    def to_pixel(x: float, y: float, z: float = 0.0) -> np.ndarray:
        cx, cy = PLAN_PRINCIPAL
        depth = PLAN_HEIGHT_M - z
        return np.array([cx + PLAN_F * x / depth, cy - PLAN_F * y / depth])

    metres_per_pixel = PLAN_HEIGHT_M / PLAN_F

    truth = np.array([row.robot_x, row.robot_y])
    measurement = np.array([row.raw_x, row.raw_y])
    # A prior displaced from the truth, and the posterior its Kalman gain produces.
    # The capture is a static pose grid, so there is no logged belief to read here:
    # these covariances are chosen for legibility and the update itself is computed.
    prior = truth + np.array([-1.05, 0.78])
    prior_cov = np.array([[0.115, 0.040], [0.040, 0.052]])
    meas_cov = np.array([[0.030, -0.008], [-0.008, 0.017]])
    gain = prior_cov @ np.linalg.inv(prior_cov + meas_cov)
    post = prior + gain @ (measurement - prior)
    post_cov = prior_cov - gain @ prior_cov

    # Window the crop on everything the panel draws, with a fixed 4:3 shape, so the
    # three estimates spread across the panel and the racks either side still show.
    drawn = np.vstack([truth, measurement, prior, post])
    centre_xy = (drawn.min(axis=0) + drawn.max(axis=0)) / 2
    half_h = max(1.70, 0.95 * float(np.ptp(drawn, axis=0).max()))
    half_w = half_h * 4 / 3
    top_left = to_pixel(centre_xy[0] - half_w, centre_xy[1] + half_h)
    bottom_right = to_pixel(centre_xy[0] + half_w, centre_xy[1] - half_h)

    ax.imshow(image)
    ax.set(xlim=(top_left[0], bottom_right[0]), ylim=(bottom_right[1], top_left[1]))
    ax.axis('off')

    def ellipse(centre, cov, colour, style, zorder):
        values, vectors = np.linalg.eigh(cov)
        # world +y runs up the page while pixel +y runs down, so the angle flips
        angle = -math.degrees(math.atan2(vectors[1, -1], vectors[0, -1]))
        ax.add_patch(Ellipse(to_pixel(*centre),
                             2 * math.sqrt(values[-1]) / metres_per_pixel,
                             2 * math.sqrt(values[0]) / metres_per_pixel, angle=angle,
                             fc=colour, alpha=.22, ec=colour, lw=1.8, ls=style,
                             zorder=zorder))

    ellipse(prior, prior_cov, BELIEF, '--', 3)
    ellipse(measurement, meas_cov, MEAS, '-', 6)
    ellipse(post, post_cov, BELIEF, '-', 4)

    truth_px = to_pixel(*truth)

    # the sightline from the camera that produced the reading, so the panel says which
    cam_px = to_pixel(camera_centre[0], camera_centre[1], 5.0)
    direction = cam_px - truth_px
    direction = direction / np.linalg.norm(direction)
    # stop the sightline at the panel edge: an arrow drawn past it would enlarge the
    # figure's bounding box, since bbox_inches='tight' measures every artist
    edge = truth_px + direction * (1.25 * half_h / metres_per_pixel)
    ax.plot([truth_px[0], edge[0]], [truth_px[1], edge[1]], color=BELIEF, lw=1.3,
            alpha=.42, ls=(0, (5, 4)), zorder=2, clip_on=True)
    label_at = truth_px + direction * (0.78 * half_h / metres_per_pixel)
    ax.text(label_at[0], label_at[1], 'to camera $c$', color=BELIEF, fontsize=8.8,
            fontweight='bold', ha='center', va='center', zorder=6, alpha=.95,
            bbox=dict(boxstyle='square,pad=0.16', fc='white', ec='none', alpha=.72))

    # the real robot, at the pose the capture recorded, with its true 0.80 x 0.55 m body
    heading = np.array([math.cos(row.robot_yaw), math.sin(row.robot_yaw)])
    across = np.array([-heading[1], heading[0]])
    body = np.array([to_pixel(*(truth + sl * 0.40 * heading + sw * 0.275 * across))
                     for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    ax.add_patch(Polygon(body, closed=True, fc='white', ec=TRUTH, lw=2.0, alpha=.42,
                         zorder=5))
    ax.annotate('', xy=to_pixel(*(truth + 0.80 * heading)), xytext=truth_px,
                arrowprops=dict(arrowstyle='-|>,head_width=0.30,head_length=0.62',
                                lw=2.3, color=TRUTH, shrinkA=1, shrinkB=0))
    ax.plot(*truth_px, marker='o', ms=5.5, color=TRUTH, mec='white', mew=1.1, zorder=7)

    # the innovation, from the predicted reading to the actual one
    ax.annotate('', xy=to_pixel(*measurement), xytext=to_pixel(*prior),
                arrowprops=dict(arrowstyle='-|>,head_width=0.28,head_length=0.6',
                                lw=2.1, color=MEAS, shrinkA=3, shrinkB=3, zorder=6))

    # Labels sit off the geometry and are tied to it with a short leader, so a change
    # of window scale can never leave one stranded away from the thing it names. The
    # offsets are in metres, so they track the window rather than the page.
    halo = dict(boxstyle='square,pad=0.16', fc='white', ec='none', alpha=.82)
    for anchor, colour, label, offset, ha, va, leader in (
            (prior, BELIEF, 'predicted\n$m_k^-,S_k^-$', (-0.30, 0.62), 'center',
             'bottom', True),
            (measurement, MEAS, 'reading\n$z_k,R_c$', (0.62, -0.30), 'left', 'top',
             True),
            (post, BELIEF, 'updated\n$m_k^+,S_k^+$', (-0.78, -0.30), 'right', 'top',
             True),
            (truth, TRUTH, 'true pose $s_k$', (0.72, 0.46), 'left', 'bottom', True),
            ((prior + measurement) / 2, MEAS, '$z_k-h(m_k^-)$', (-0.30, -0.42),
             'center', 'top', False)):
        px = to_pixel(*anchor)
        text_px = to_pixel(anchor[0] + offset[0], anchor[1] + offset[1])
        ax.annotate(label, xy=px, xytext=text_px, color=colour, fontsize=9.2,
                    fontweight='bold', ha=ha, va=va, zorder=11, bbox=halo,
                    arrowprops=dict(arrowstyle='-', lw=0.8, color=colour, alpha=.65,
                                    shrinkA=3, shrinkB=6) if leader else None)

    for centre, colour, marker, size in (
            (prior, BELIEF, 'o', 7), (measurement, MEAS, '*', 17)):
        ax.plot(*to_pixel(*centre), marker=marker, ms=size, color=colour,
                mec='white', mew=1.2, zorder=8)
    ax.plot(*to_pixel(*post), marker='o', ms=8, color=BELIEF, mec='white', mew=1.3,
            zorder=8)

    return truth, measurement, prior, post


def main() -> None:
    if not PLAN.exists():
        raise SystemExit(f'missing {PLAN}; capture it from a running simulation first')
    rows = pd.read_csv(CAPTURE / 'observation_interpretations.csv')
    row = rows[(rows.image == FRAME) & (rows.detected == 1)].iloc[0]
    model, centre = camera_geometry()

    # the drawing must reproduce the dataset's own back-projection, or the geometry lies
    recomputed = np.array(model.pixel_to_world((row.x0 + row.x1) / 2, row.y1))
    recorded = np.array([row.raw_x, row.raw_y])
    if not np.allclose(recomputed, recorded, atol=1e-4):
        raise SystemExit(f'camera model disagrees with the capture: {recomputed} vs {recorded}')

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.85))
    panel_image(axes[0], row)
    panel_projection(axes[1], row, model, centre)
    truth, measurement, prior, post = panel_update(axes[2], row, centre)

    fig.tight_layout(pad=0.4, w_pad=1.4)

    # One title row for all three panels: the panels have different data aspects, so
    # an axes-relative title would sit at a different page height in each.
    titles = ('(a) camera detection', '(b) ground-plane projection',
              '(c) filter update, in the warehouse')
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
    print(f'reading error {np.linalg.norm(measurement - truth) * 100:.1f} cm, '
          f'posterior error {np.linalg.norm(post - truth) * 100:.1f} cm, '
          f'prior error {np.linalg.norm(prior - truth) * 100:.1f} cm')


if __name__ == '__main__':
    main()
