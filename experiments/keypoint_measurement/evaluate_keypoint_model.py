#!/usr/bin/env python3
"""Score a keypoint detector as a *measurement*, not as a detector.

The question is not mAP. It is whether the reading it produces is unbiased, and
if not, what the bias depends on. So for every captured pose this reports:

  pixel residual   predicted keypoint minus the analytically projected marker
  world residual   both keypoints back-projected onto the marker plane, then
                   compared with the pose the robot was actually teleported to
  heading residual the front-to-rear vector's bearing, against the true yaw

and then splits each of those by range, heading, image position and apparent
size, because a reading whose mean error is zero overall can still be wrong in
a way that depends on where the robot is — which is what a filter would inherit.

Reads a dataset written by scripts/perception/capture_projected_keypoint_dataset.py:
its capture_diagnostics.csv carries the true pose and the ground-truth pixels,
and its capture_manifest.json carries the camera the labels were made with.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

FRONT, REAR = 0, 1


def wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def load_capture(dataset: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset / 'capture_manifest.json').read_text(encoding='utf-8'))
    with (dataset / 'capture_diagnostics.csv').open(encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r['accepted'] == '1']
    return rows, manifest


def camera_from_manifest(manifest: dict, img_w: int, img_h: int) -> ObliqueCameraModel:
    pose = [float(v) for v in manifest['camera_pose']]
    look_at = compute_look_at_from_pose(pose[:3], pose[3], pose[4], pose[5])
    return ObliqueCameraModel(
        cam_pos=pose[:3], look_at=look_at,
        img_width=img_w, img_height=img_h, fov_h_rad=1.5708,
    )


def cameras_from_manifest(manifest: dict, img_w: int, img_h: int) -> dict[str, ObliqueCameraModel]:
    """One camera model per camera in the capture.

    A four-camera capture interleaves cameras row by row, so back-projecting a
    reading through the wrong one silently produces a plausible, wrong position.
    Single-camera captures predate the ``cameras`` key and fall back to it.
    """
    entries = manifest.get('cameras')
    if not entries:
        return {'external_camera': camera_from_manifest(manifest, img_w, img_h)}
    out = {}
    for name, spec in entries.items():
        pose = [float(v) for v in spec['pose']]
        look_at = compute_look_at_from_pose(pose[:3], pose[3], pose[4], pose[5])
        out[name] = ObliqueCameraModel(
            cam_pos=pose[:3], look_at=look_at,
            img_width=int(spec.get('img_width', img_w)),
            img_height=int(spec.get('img_height', img_h)),
            fov_h_rad=float(spec.get('fov_h_rad', 1.5708)),
        )
    return out


def predict(model, image_paths: list[Path], imgsz: int, conf: float, batch: int, device: str):
    """Return one (keypoints, confidence) per image; keypoints is (2,2) or None."""
    out: list[tuple[np.ndarray | None, float]] = []
    for start in range(0, len(image_paths), batch):
        chunk = [str(p) for p in image_paths[start:start + batch]]
        results = model.predict(chunk, imgsz=imgsz, conf=conf, verbose=False, device=device)
        for res in results:
            boxes = getattr(res, 'boxes', None)
            kpts = getattr(res, 'keypoints', None)
            if boxes is None or kpts is None or len(boxes) == 0:
                out.append((None, 0.0))
                continue
            confs = boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            xy = kpts.xy.cpu().numpy()[best]
            out.append((np.asarray(xy, dtype=float), float(confs[best])))
    return out


def summarise(name: str, residual: np.ndarray) -> dict:
    """residual is (N,2) in pixels or (N,2) in metres."""
    n = int(len(residual))
    if n == 0:
        return {'split': name, 'n': 0}
    mag = np.hypot(residual[:, 0], residual[:, 1])
    return {
        'split': name, 'n': n,
        'mean_u': float(residual[:, 0].mean()), 'mean_v': float(residual[:, 1].mean()),
        'std_u': float(residual[:, 0].std(ddof=1)) if n > 1 else 0.0,
        'std_v': float(residual[:, 1].std(ddof=1)) if n > 1 else 0.0,
        'median_mag': float(np.median(mag)),
        'p90_mag': float(np.percentile(mag, 90)),
        'mean_mag': float(mag.mean()),
    }


def banded(values: np.ndarray, residual: np.ndarray, edges: list[float], label: str) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (values >= lo) & (values < hi)
        if not sel.any():
            continue
        rows.append(summarise(f'{label} {lo:g}..{hi:g}', residual[sel]))
    return rows


def write_table(handle, title: str, rows: list[dict], unit: str) -> None:
    handle.write(f'\n### {title}\n\n')
    handle.write(f'| split | n | mean along u ({unit}) | mean along v ({unit}) | '
                 f'spread u | spread v | median miss | 90th pct miss |\n')
    handle.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in rows:
        if not r.get('n'):
            continue
        handle.write(
            f"| {r['split']} | {r['n']} | {r['mean_u']:+.2f} | {r['mean_v']:+.2f} | "
            f"{r['std_u']:.2f} | {r['std_v']:.2f} | {r['median_mag']:.2f} | {r['p90_mag']:.2f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True, help='Output folder for the report + csv.')
    parser.add_argument('--split', default='val', choices=('val', 'train', 'all'))
    parser.add_argument('--imgsz', type=int, default=960,
                        help='Below 960 the two markers collapse into one blob; do not lower.')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--label', default='', help='Name for this evaluation in the report.')
    args = parser.parse_args()

    from ultralytics import YOLO

    dataset = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest = load_capture(dataset)
    if args.split != 'all':
        rows = [r for r in rows if r['split'] == args.split]
    if not rows:
        raise RuntimeError(f'No {args.split} rows in {dataset}')

    import cv2
    first = cv2.imread(str(dataset / rows[0]['image']))
    if first is None:
        raise RuntimeError(f"Cannot read {dataset / rows[0]['image']}")
    img_h, img_w = first.shape[:2]
    cameras = cameras_from_manifest(manifest, img_w, img_h)
    default_camera = next(iter(cameras))
    marker_z = float(manifest.get('keypoint_marker_world_z', 0.210))
    geom = manifest['marker_geometry']
    front_x, rear_x = float(geom['front_x']), float(geom['rear_x'])

    model = YOLO(str(Path(args.weights).expanduser().resolve()))
    preds = predict(model, [dataset / r['image'] for r in rows],
                    int(args.imgsz), float(args.conf), int(args.batch), str(args.device))

    records = []
    for row, (xy, conf) in zip(rows, preds):
        x, y, yaw = float(row['x']), float(row['y']), float(row['yaw_rad'])
        camera_name = row.get('camera') or default_camera
        camera = cameras[camera_name]
        gt_front = np.array([float(row['front_u']), float(row['front_v'])])
        gt_rear = np.array([float(row['rear_u']), float(row['rear_v'])])
        rec = {
            'x': x, 'y': y, 'yaw_rad': yaw, 'camera': camera_name,
            'range_m': float(math.dist((x, y, marker_z), tuple(camera.cam_pos))),
            'gt_front_u': gt_front[0], 'gt_front_v': gt_front[1],
            'gt_rear_u': gt_rear[0], 'gt_rear_v': gt_rear[1],
            'gt_separation_px': float(np.linalg.norm(gt_front - gt_rear)),
            'front_labelled_visible': int(row['front_visible']),
            'rear_labelled_visible': int(row['rear_visible']),
            'detected': int(xy is not None), 'conf': conf,
        }
        # New commissioning captures carry grouping metadata. Preserve it so an
        # evaluator can hold out whole sites and simulator sessions rather than
        # randomly splitting nearly identical frames.
        for key in (
            'session_id', 'anchor_id', 'x_idx', 'y_idx', 'yaw_idx', 'repeat_idx',
            'nominal_x', 'nominal_y', 'nominal_yaw_rad', 'split', 'sample_idx',
        ):
            if key in row and row[key] != '':
                rec[key] = row[key]
        if xy is not None:
            pf, pr = xy[FRONT], xy[REAR]
            rec.update({
                'pred_front_u': pf[0], 'pred_front_v': pf[1],
                'pred_rear_u': pr[0], 'pred_rear_v': pr[1],
                'res_front_u': pf[0] - gt_front[0], 'res_front_v': pf[1] - gt_front[1],
                'res_rear_u': pr[0] - gt_rear[0], 'res_rear_v': pr[1] - gt_rear[1],
                'pred_separation_px': float(np.linalg.norm(pf - pr)),
            })
            # Back-project both keypoints onto the marker plane, then read the
            # robot's position and heading off the reconstructed marker pair.
            wf = np.asarray(camera.pixel_to_world_at_z(pf[0], pf[1], marker_z), dtype=float)
            wr = np.asarray(camera.pixel_to_world_at_z(pr[0], pr[1], marker_z), dtype=float)
            heading = math.atan2(wf[1] - wr[1], wf[0] - wr[0])
            # base_link is where the two marker offsets say it is, given that heading
            centre = 0.5 * (wf[:2] + wr[:2])
            mid_offset = 0.5 * (front_x + rear_x)
            base = centre - mid_offset * np.array([math.cos(heading), math.sin(heading)])
            rec.update({
                'est_x': base[0], 'est_y': base[1], 'est_yaw_rad': heading,
                'err_x_m': base[0] - x, 'err_y_m': base[1] - y,
                'err_yaw_rad': float(wrap_pi(heading - yaw)),
            })
        records.append(rec)

    fields = sorted({k for r in records for k in r})
    with (out_dir / 'per_sample.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    det = [r for r in records if r['detected']]
    n_all, n_det = len(records), len(det)
    if n_det == 0:
        raise RuntimeError('The model detected nothing at all on this split.')

    rng = np.array([r['range_m'] for r in det])
    yaws = np.array([r['yaw_rad'] for r in det])
    sep = np.array([r['gt_separation_px'] for r in det])
    res_front = np.array([[r['res_front_u'], r['res_front_v']] for r in det])
    res_rear = np.array([[r['res_rear_u'], r['res_rear_v']] for r in det])
    res_both = np.vstack([res_front, res_rear])
    err_xy = np.array([[r['err_x_m'], r['err_y_m']] for r in det]) * 100.0  # cm
    err_yaw_deg = np.degrees([r['err_yaw_rad'] for r in det])
    flips = float(np.mean(np.abs(err_yaw_deg) > 90.0))

    label = args.label or Path(args.weights).parent.name
    with (out_dir / 'RESULTS.md').open('w', encoding='utf-8') as handle:
        handle.write(f'# Keypoint reading quality: {label}\n\n')
        handle.write(f'- dataset: `{dataset}` (`{args.split}` split)\n')
        handle.write(f'- weights: `{args.weights}`, inference at {args.imgsz} px, conf {args.conf}\n')
        handle.write(f"- camera the labels were built with: {manifest['camera_pose']}, "
                     f"world `{manifest.get('world')}`\n")
        handle.write(f'- marker plane: z = {marker_z:.3f} m\n\n')
        handle.write(f'**Found the robot in {n_det} of {n_all} images ({100.0*n_det/n_all:.1f}%).**\n\n')
        handle.write('## Where the reading lands, in metres on the floor\n\n')
        handle.write(f'- east-west error: mean {err_xy[:,0].mean():+.2f} cm, '
                     f'spread {err_xy[:,0].std(ddof=1):.2f} cm\n')
        handle.write(f'- north-south error: mean {err_xy[:,1].mean():+.2f} cm, '
                     f'spread {err_xy[:,1].std(ddof=1):.2f} cm\n')
        handle.write(f'- distance from truth: median {np.median(np.hypot(*err_xy.T)):.2f} cm, '
                     f'90th percentile {np.percentile(np.hypot(*err_xy.T), 90):.2f} cm\n')
        handle.write(f'- heading: median error {np.median(np.abs(err_yaw_deg)):.1f} deg, '
                     f'signed mean {err_yaw_deg.mean():+.1f} deg, '
                     f'front/rear swapped {100.0*flips:.1f}% of the time\n')

        handle.write('\n## Pixel residual (predicted keypoint minus projected marker)\n')
        write_table(handle, 'Overall', [
            summarise('front marker', res_front),
            summarise('rear marker', res_rear),
            summarise('both', res_both),
        ], 'px')
        write_table(handle, 'By range to camera (both markers)',
                    banded(np.concatenate([rng, rng]), res_both,
                           [0, 3, 5, 7, 9, 20], 'range m'), 'px')
        write_table(handle, 'By heading (both markers)',
                    banded(np.degrees(np.concatenate([yaws, yaws])), res_both,
                           [0, 45, 90, 135, 180, 225, 270, 315, 361], 'yaw deg'), 'px')
        write_table(handle, 'By apparent marker separation (both markers)',
                    banded(np.concatenate([sep, sep]), res_both,
                           [0, 6, 10, 15, 25, 200], 'separation px'), 'px')

        # A marker the renderer never drew is a marker the model had to guess at.
        vis_front = np.array([r['front_labelled_visible'] for r in det], dtype=bool)
        vis_rear = np.array([r['rear_labelled_visible'] for r in det], dtype=bool)
        vis_both = np.concatenate([vis_front, vis_rear])
        write_table(handle, 'By whether that marker actually rendered', [
            summarise('marker was visible', res_both[vis_both]),
            summarise('marker was hidden', res_both[~vis_both]),
        ], 'px')
        both_seen = vis_front & vis_rear
        write_table(handle, 'Floor error when both markers rendered vs not', [
            summarise('both markers visible', err_xy[both_seen]),
            summarise('at least one hidden', err_xy[~both_seen]),
        ], 'cm')
        if len(cameras) > 1:
            cam_of = np.array([r['camera'] for r in det])
            write_table(handle, 'Floor error per camera', [
                summarise(name, err_xy[cam_of == name]) for name in cameras
            ], 'cm')
        if both_seen.any():
            handle.write(
                f'\nWith both markers visible ({int(both_seen.sum())} readings): heading median '
                f'{np.median(np.abs(err_yaw_deg[both_seen])):.1f} deg, '
                f'front/rear swapped {100.0*np.mean(np.abs(err_yaw_deg[both_seen]) > 90.0):.1f}%.\n'
            )

        handle.write('\n## Floor error in cm, by the same splits\n')
        write_table(handle, 'By range to camera',
                    banded(rng, err_xy, [0, 3, 5, 7, 9, 20], 'range m'), 'cm')
        write_table(handle, 'By heading',
                    banded(np.degrees(yaws), err_xy,
                           [0, 45, 90, 135, 180, 225, 270, 315, 361], 'yaw deg'), 'cm')

    print((out_dir / 'RESULTS.md').read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
