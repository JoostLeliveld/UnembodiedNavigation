#!/usr/bin/env python3
"""Score BOTH readings on the same drive frames, so the only difference is the reading.

The capture (`capture_drive_with_markers.sh`) records 1280x720 frames stamped in simulated
time, the noisy odometry the filter is allowed to use, and the truth it is not. This script
reads every frame twice:

  box bottom     the frozen detector `warehouse_yolo_detector_v1` at 960 px, bottom-centre
                 pixel of the best box, straight down the ray onto the floor -- the
                 deployed reading, exactly as `yolo_robot_detector_node` produces it
  marked point   `yolo_pose_aws_v4` at 960 px, both marker keypoints back-projected onto
                 the marker plane (z = 0.21 m), base_link read off the pair

and writes one row per frame with both, plus the truth at that stamp for scoring later.

Nothing here filters or learns; that is `drive_filter.py`. Ground truth is written to the
output but never used to produce a reading.

Run: python3 experiments/localization_reading_story/score_drive_frames.py <drive dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keypoint_geometry as kg  # noqa: E402
import reading_data as rd  # noqa: E402

BOX_MODEL = rd.REPO_ROOT / 'logs/perception_models/warehouse_yolo_detector_v1/model.pt'
KEYPOINT_MODEL = rd.REPO_ROOT / 'logs/perception_models/yolo_pose_aws_v4/model.pt'
IMGSZ = 960
BOX_CONF = 0.05
KEYPOINT_CONF = 0.25


def moving_window(stamps, xy, *, still_m: float = 0.05, horizon_s: float = 10.0):
    """The stretch of the recording in which the robot is actually driving.

    A route driver that keeps commanding after the robot has stopped -- because it steers
    on odometry, and odometry keeps integrating a wheel that is turning against a rack --
    leaves a tail in which the truth does not move and the odometry runs away. That tail is
    not a drive, and nothing in it may be filtered. This finds it from the truth alone.
    """
    last_moving = stamps[0]
    for i, stamp in enumerate(stamps):
        j = min(int(np.searchsorted(stamps, stamp + horizon_s)), len(stamps) - 1)
        if float(np.hypot(*(xy[j] - xy[i]))) >= still_m:
            last_moving = stamps[j]
    return float(stamps[0]), float(last_moving)


def load_truth(drive: Path):
    with (drive / 'evaluation_only/ground_truth.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    stamps = np.array([float(r['stamp']) for r in rows])
    xy = np.column_stack([[float(r['gt_x']) for r in rows], [float(r['gt_y']) for r in rows]])
    yaw = np.unwrap(np.array([float(r['gt_yaw']) for r in rows]))
    order = np.argsort(stamps)
    return stamps[order], xy[order], yaw[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('drive', type=Path)
    parser.add_argument('--camera', default='camera_A')
    args = parser.parse_args()
    drive = args.drive.expanduser().resolve()

    from ultralytics import YOLO

    with (drive / 'views' / args.camera / 'index.csv').open(encoding='utf-8') as handle:
        frames = [(float(r['stamp']), drive / 'views' / args.camera / r['file'])
                  for r in csv.DictReader(handle)]
    frames.sort()
    print(f'{len(frames)} frames, {frames[0][0]:.1f} to {frames[-1][0]:.1f} s')

    t_stamps, t_xy, t_yaw = load_truth(drive)
    start_s, end_s = moving_window(t_stamps, t_xy)
    before = len(frames)
    frames = [f for f in frames if start_s <= f[0] <= end_s]
    if len(frames) < before:
        print(f'trimmed to the moving part of the recording: {start_s:.1f}-{end_s:.1f} s, '
              f'{len(frames)} of {before} frames (the robot is stationary after that, and '
              f'its odometry is not)')
    camera = rd.camera()          # same world, same camera as the static capture
    print(f'camera at {camera.cam_pos}')

    box_model = YOLO(str(BOX_MODEL))
    kp_model = YOLO(str(KEYPOINT_MODEL))
    paths = [str(p) for _, p in frames]
    (drive / 'moving_window.json').write_text(
        json.dumps({'start_s': start_s, 'end_s': end_s, 'frames_kept': len(frames),
                    'frames_recorded': before}, indent=2) + '\n', encoding='utf-8')

    def predict(model, conf, keypoints):
        out = []
        for start in range(0, len(paths), 16):
            chunk = paths[start:start + 16]
            for res in model.predict(chunk, imgsz=IMGSZ, conf=conf, verbose=False, device=0):
                boxes = getattr(res, 'boxes', None)
                if boxes is None or len(boxes) == 0:
                    out.append(None)
                    continue
                best = int(np.argmax(boxes.conf.cpu().numpy()))
                item = {'conf': float(boxes.conf.cpu().numpy()[best]),
                        'xyxy': boxes.xyxy.cpu().numpy()[best]}
                if keypoints:
                    kp = getattr(res, 'keypoints', None)
                    if kp is None:
                        out.append(None)
                        continue
                    item['kp'] = np.asarray(kp.xy.cpu().numpy()[best], dtype=float)
                out.append(item)
        return out

    print('running the box detector ...')
    box = predict(box_model, BOX_CONF, keypoints=False)
    print('running the keypoint model ...')
    kp = predict(kp_model, KEYPOINT_CONF, keypoints=True)

    rows = []
    for (stamp, path), b, k in zip(frames, box, kp):
        row = {
            'stamp': stamp, 'frame': path.name,
            'gt_x': float(np.interp(stamp, t_stamps, t_xy[:, 0])),
            'gt_y': float(np.interp(stamp, t_stamps, t_xy[:, 1])),
            'gt_yaw': float(np.interp(stamp, t_stamps, t_yaw)),
            'box_detected': 0, 'kp_detected': 0,
        }
        if b is not None:
            x0, y0, x1, y1 = b['xyxy']
            u, v = 0.5 * (float(x0) + float(x1)), float(y1)
            world = camera.pixel_to_world(u, v)
            if world is not None:
                row.update({'box_detected': 1, 'box_conf': b['conf'],
                            'box_u': u, 'box_v': v,
                            'box_x': float(world[0]), 'box_y': float(world[1])})
        if k is not None and 'kp' in k and len(k['kp']) >= 2:
            pixels = [k['kp'][0][0], k['kp'][0][1], k['kp'][1][0], k['kp'][1][1]]
            if all(np.isfinite(pixels)) and not all(p == 0 for p in pixels):
                base = kg.read_base(camera, pixels)
                front = np.asarray(camera.pixel_to_world_at_z(pixels[0], pixels[1],
                                                              kg.MARKER_Z), float)
                rear = np.asarray(camera.pixel_to_world_at_z(pixels[2], pixels[3],
                                                             kg.MARKER_Z), float)
                row.update({'kp_detected': 1, 'kp_conf': k['conf'],
                            'kp_front_u': pixels[0], 'kp_front_v': pixels[1],
                            'kp_rear_u': pixels[2], 'kp_rear_v': pixels[3],
                            'kp_x': float(base[0]), 'kp_y': float(base[1]),
                            'kp_yaw': float(math.atan2(front[1] - rear[1],
                                                       front[0] - rear[0]))})
        rows.append(row)

    fields = sorted({key for row in rows for key in row})
    out = drive / 'readings.csv'
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')

    for name, flag, xk, yk in (('box bottom', 'box_detected', 'box_x', 'box_y'),
                               ('marked point', 'kp_detected', 'kp_x', 'kp_y')):
        seen = [r for r in rows if r[flag]]
        if not seen:
            print(f'  {name}: nothing detected')
            continue
        err = 100 * np.array([[r[xk] - r['gt_x'], r[yk] - r['gt_y']] for r in seen])
        print(f'  {name}: {len(seen)}/{len(rows)} frames ({100 * len(seen) / len(rows):.0f}%), '
              f'mean ({err[:, 0].mean():+.2f}, {err[:, 1].mean():+.2f}) cm, '
              f'median miss {np.median(np.hypot(*err.T)):.2f} cm')


if __name__ == '__main__':
    main()
