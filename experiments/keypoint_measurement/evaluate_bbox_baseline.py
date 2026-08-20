#!/usr/bin/env python3
"""Score today's reading — the box-bottom pixel through plain IPM — on the same poses.

This is the arm the keypoint reading has to beat, measured on exactly the images
the keypoint model is scored on, so the comparison is not across two different
captures. It reproduces the runtime convention from
``perception.core.yolo_selection``: the observation pixel is the horizontal
centre of the box and its bottom edge, back-projected onto the floor plane.

One honest caveat: these frames were rendered with the two marker disks on the
robot, because that is what the keypoint capture needs. The box-bottom detector
was trained without them. The disks sit on top of the robot, so they move the
box's *top* edge and not the bottom one this reading uses, but the box the
detector draws is not bit-identical to the one it would draw on a bare robot.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_keypoint_model import (  # noqa: E402
    banded, cameras_from_manifest, load_capture, summarise, write_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='val', choices=('val', 'train', 'all'))
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    dataset = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest = load_capture(dataset)
    if args.split != 'all':
        rows = [r for r in rows if r['split'] == args.split]
    if not rows:
        raise RuntimeError(f'No {args.split} rows in {dataset}')

    first = cv2.imread(str(dataset / rows[0]['image']))
    if first is None:
        raise RuntimeError(f"Cannot read {dataset / rows[0]['image']}")
    img_h, img_w = first.shape[:2]
    cameras = cameras_from_manifest(manifest, img_w, img_h)
    default_camera = next(iter(cameras))

    model = YOLO(str(Path(args.weights).expanduser().resolve()))
    records = []
    for start in range(0, len(rows), int(args.batch)):
        chunk = rows[start:start + int(args.batch)]
        results = model.predict([str(dataset / r['image']) for r in chunk],
                                imgsz=int(args.imgsz), conf=float(args.conf),
                                verbose=False, device=str(args.device))
        for row, res in zip(chunk, results):
            x, y, yaw = float(row['x']), float(row['y']), float(row['yaw_rad'])
            camera_name = row.get('camera') or default_camera
            camera = cameras[camera_name]
            rec = {
                'x': x, 'y': y, 'yaw_rad': yaw, 'camera': camera_name,
                'range_m': float(math.dist((x, y, 0.0), tuple(camera.cam_pos))),
                'detected': 0,
            }
            boxes = getattr(res, 'boxes', None)
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best = int(np.argmax(confs))
                x0, y0, x1, y1 = boxes.xyxy.cpu().numpy()[best]
                u, v = 0.5 * (float(x0) + float(x1)), float(y1)
                world = camera.pixel_to_world(u, v)
                if world is not None:
                    rec.update({
                        'detected': 1, 'conf': float(confs[best]),
                        'obs_u': u, 'obs_v': v,
                        'box_height_px': float(y1 - y0), 'box_width_px': float(x1 - x0),
                        'est_x': world[0], 'est_y': world[1],
                        'err_x_m': world[0] - x, 'err_y_m': world[1] - y,
                    })
            records.append(rec)

    fields = sorted({k for r in records for k in r})
    with (out_dir / 'per_sample.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    det = [r for r in records if r['detected']]
    if not det:
        raise RuntimeError('The box detector found nothing on this split.')
    rng = np.array([r['range_m'] for r in det])
    yaws = np.degrees([r['yaw_rad'] for r in det])
    err = np.array([[r['err_x_m'], r['err_y_m']] for r in det]) * 100.0

    with (out_dir / 'RESULTS.md').open('w', encoding='utf-8') as handle:
        handle.write('# Box-bottom reading through plain IPM, on the keypoint capture\n\n')
        handle.write(f'- dataset: `{dataset}` (`{args.split}` split)\n')
        handle.write(f'- weights: `{args.weights}`, inference at {args.imgsz} px, conf {args.conf}\n')
        handle.write(f'- reading: horizontal box centre at the box bottom edge, '
                     f'back-projected to the floor plane\n\n')
        handle.write(f'**Found the robot in {len(det)} of {len(records)} images '
                     f'({100.0*len(det)/len(records):.1f}%).**\n\n')
        handle.write(f'- east-west error: mean {err[:,0].mean():+.2f} cm, '
                     f'spread {err[:,0].std(ddof=1):.2f} cm\n')
        handle.write(f'- north-south error: mean {err[:,1].mean():+.2f} cm, '
                     f'spread {err[:,1].std(ddof=1):.2f} cm\n')
        handle.write(f'- distance from truth: median {np.median(np.hypot(*err.T)):.2f} cm, '
                     f'90th percentile {np.percentile(np.hypot(*err.T), 90):.2f} cm\n')
        write_table(handle, 'Floor error by range to camera',
                    banded(rng, err, [0, 3, 5, 7, 9, 20], 'range m'), 'cm')
        write_table(handle, 'Floor error by heading',
                    banded(yaws, err, [0, 45, 90, 135, 180, 225, 270, 315, 361], 'yaw deg'), 'cm')
        write_table(handle, 'Overall', [summarise('box bottom', err)], 'cm')

    print((out_dir / 'RESULTS.md').read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
