#!/usr/bin/env python3
"""Compare native CPU/CUDA inference on exact recorded images; no ROS or driving."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/unav_performance_yolo')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/perception'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml')
    parser.add_argument('--capture', type=Path, default=ROOT / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831')
    parser.add_argument('--poses', type=int, nargs='+', default=[0, 1000, 2000])
    parser.add_argument('--devices', nargs='+', default=['cpu', '0'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    import cv2
    import numpy as np
    import torch
    import yaml
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from perception.core.yolo_selection import select_best_detection
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    cfg = yaml.safe_load(args.config.read_text())
    weights = ROOT / cfg['yolo_model']
    cameras = cfg['manager_camera_ids'].split(',')
    capture_meta = json.loads((args.capture / 'capture_manifest.json').read_text())
    index = args.capture / 'capture_index.csv'
    if digest(index) != capture_meta['capture_index_sha256']:
        raise ValueError('capture index no longer matches its manifest')
    with index.open() as handle:
        selected = {(int(r['pose_id']), r['camera_id']): r for r in csv.DictReader(handle)
                    if int(r['pose_id']) in args.poses and int(r['repetition_id']) == 0}
    batches, inputs = [], []
    for pose in args.poses:
        images = []
        for camera in cameras:
            row = selected[(pose, camera)]
            path = args.capture / row['image']
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'cannot decode {path}')
            decoded = hashlib.sha1()
            decoded.update(str(image.shape).encode('utf-8'))
            decoded.update(str(image.dtype).encode('utf-8'))
            decoded.update(image.tobytes())
            if row['capture_status'] != 'ok' or decoded.hexdigest() != row['image_sha1']:
                raise ValueError(f'changed or failed decoded image: {path}')
            images.append(image)
            inputs.append(dict(pose=pose, camera=camera, path=str(path), sha256=digest(path)))
        batches.append(images)
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(kind='offline_native_detector_performance_probe',
                    config=str(args.config), config_sha256=digest(args.config),
                    weights=str(weights), weights_sha256=digest(weights), inputs=inputs,
                    devices=args.devices, repeats=args.repeats, chunk=2, imgsz=cfg['yolo_imgsz'],
                    predict_conf_floor=cfg.get('yolo_predict_conf_floor', .05),
                    reporting_threshold=cfg['yolo_conf_threshold'], iou=cfg['yolo_iou_threshold'],
                    torch=torch.__version__, source_sha256=digest(__file__),
                    box_atol_px=.02, score_atol=1e-4,
                    limitation='Recorded-image probe; live rendering, transport and batch accounting are not reproduced.')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = {}
    # Ultralytics' CPU device selection sets CUDA_VISIBLE_DEVICES=-1 for the
    # remainder of the process. Exercise accelerators first so a requested
    # CPU/GPU comparison cannot accidentally hide CUDA before its own probe.
    execution_devices = [device for device in args.devices if device != 'cpu']
    execution_devices.extend(device for device in args.devices if device == 'cpu')
    for device in execution_devices:
        cuda = device != 'cpu'
        if cuda and not torch.cuda.is_available():
            results[device] = dict(status='unavailable', error='CUDA failed to initialize on this host')
            print(device, json.dumps(results[device]), flush=True)
            continue
        model = YOLO(str(weights))
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        def cycle(images):
            result = []
            for offset in range(0, len(images), 2):
                group = images[offset:offset+2]
                result.extend(model.predict(source=group, imgsz=cfg['yolo_imgsz'],
                              conf=protocol['predict_conf_floor'], iou=cfg['yolo_iou_threshold'],
                              device=device, batch=len(group), stream=False, verbose=False))
            return result
        for _ in range(3):
            cycle(batches[0])
        durations, selections = [], []
        for repeat in range(args.repeats):
            for images in batches:
                if cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                output = cycle(images)
                if cuda:
                    torch.cuda.synchronize()
                durations.append((time.perf_counter() - started)*1000.)
                if repeat == 0:
                    for result in output:
                        item = select_best_detection(result, target_ids={cfg['yolo_class_id']},
                            confidence_threshold=cfg['yolo_conf_threshold'], use_masks=False,
                            mask_min_area=12., mask_bottom_band_px=3.)
                        selections.append(dict(detected=bool(item['detected']),
                            box=np.asarray(item['bbox_xyxy']).tolist() if item['detected'] else None,
                            score=float(item['selected_score']) if item['detected'] else None))
        results[device] = dict(status='measured', median_batch_ms=float(np.median(durations)),
            p95_batch_ms=float(np.percentile(durations, 95)), timings_ms=durations,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if cuda else None,
            selected=selections)
        print(device, json.dumps({k:v for k,v in results[device].items()
                                  if k not in ('selected', 'timings_ms')}), flush=True)
        del output, result, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    comparison = None
    if 'cpu' in results and len(results) > 1:
        comparison = {}
        for device, result in results.items():
            if device == 'cpu' or result.get('status') != 'measured':
                continue
            changed_hits, max_box, max_score = 0, 0., 0.
            for cpu, gpu in zip(results['cpu']['selected'], result['selected'], strict=True):
                changed_hits += cpu['detected'] != gpu['detected']
                if cpu['detected'] and gpu['detected']:
                    max_box = max(max_box, float(np.max(np.abs(np.array(cpu['box'])-gpu['box']))))
                    max_score = max(max_score, abs(cpu['score']-gpu['score']))
            comparison[device] = dict(changed_hit_decisions=changed_hits, max_box_delta_px=max_box,
                max_score_delta=max_score, checked_images=len(inputs),
                within_declared_tolerance=changed_hits == 0 and max_box <= .02 and max_score <= 1e-4,
                median_inference_speedup=results['cpu']['median_batch_ms']/result['median_batch_ms'])
    report = dict(results=results, comparison=comparison, protocol_sha256=digest(args.out/'protocol.json'))
    (args.out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(comparison, indent=2))
    return int(any(r['status'] != 'measured' for r in results.values()))


if __name__ == '__main__':
    sys.exit(main())
