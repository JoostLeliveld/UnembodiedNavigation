#!/usr/bin/env python3
"""Generate a Markdown report for the refreshed indoor YOLO pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import yaml


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) if path.is_file() else {}


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _count_images(root: Path, split: str) -> int:
    image_dir = root / 'images' / split
    if not image_dir.is_dir():
        return 0
    return len([p for p in image_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])


def _format_float(value, digits: int = 3) -> str:
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return 'n/a'


def _training_summary(training_dir: Path) -> dict:
    rows = _read_csv_rows(training_dir / 'results.csv')
    if not rows:
        return {}
    last = rows[-1]
    best_mask_map50 = max(
        (float(row.get('metrics/mAP50(M)', 'nan')) for row in rows),
        default=float('nan'),
    )
    best_box_map50 = max(
        (float(row.get('metrics/mAP50(B)', 'nan')) for row in rows),
        default=float('nan'),
    )
    return {
        'epochs_recorded': len(rows),
        'last_epoch': last.get('epoch', ''),
        'best_mask_map50': best_mask_map50,
        'best_box_map50': best_box_map50,
        'last_precision_mask': last.get('metrics/precision(M)', ''),
        'last_recall_mask': last.get('metrics/recall(M)', ''),
    }


def _benchmark_summary(benchmark_dir: Path) -> dict:
    rows = _read_csv_rows(benchmark_dir / 'summary.csv')
    return rows[0] if rows else {}


def _gp_summary(gp_dir: Path) -> dict:
    return _read_json(gp_dir / 'capture_config.json')


def _report_lines(args) -> list[str]:
    bbox_dir = Path(args.bbox_dataset).expanduser().resolve()
    seg_dir = Path(args.seg_dataset).expanduser().resolve()
    train_dir = Path(args.training_run).expanduser().resolve()
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    gp_dir = Path(args.gp_fit_dir).expanduser().resolve()
    experiment_dirs = [Path(p).expanduser().resolve() for p in args.experiment_dir]

    capture_manifest = _read_json(bbox_dir / 'capture_manifest.json')
    seg_manifest = _read_json(seg_dir / 'conversion_manifest.json')
    bbox_data = _read_yaml(bbox_dir / 'data.yaml')
    seg_data = _read_yaml(seg_dir / 'data.yaml')
    train_summary = _training_summary(train_dir)
    bench_summary = _benchmark_summary(benchmark_dir)
    gp_summary = _gp_summary(gp_dir)

    lines = [
        f'# Warehouse Occ Light Mainline YOLO Pipeline Report',
        '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        '',
        '## Canonical Environment',
        '',
        '- World: `warehouse_occ_light.world.sdf`',
        '- Lighting: fixed overhead indoor lighting, no rendered cast shadows, no colored floor markers.',
        '- Note: earlier shadow-era YOLO artifacts remain on disk as historical/non-canonical outputs.',
        '',
        '## Artifact Paths',
        '',
        f'- Raw bbox dataset: `{bbox_dir}`',
        f'- Segmentation dataset: `{seg_dir}`',
        f'- Candidate diagnostics CSV: `{seg_dir / "candidate_masks.csv"}`',
        f'- Segmentation previews: `{seg_dir / "previews"}`',
        f'- YOLO training run: `{train_dir}`',
        f'- YOLO best weights: `{train_dir / "weights" / "best.pt"}`',
        f'- Validation benchmark: `{benchmark_dir}`',
        f'- GP fit directory: `{gp_dir}`',
        f'- GP artifact: `{gp_dir / "empirical_visibility_gp.npz"}`',
        f'- GP aggregates: `{gp_dir / "aggregated_detection_samples.csv"}`',
        f'- GP preview plot: `{gp_dir / "empirical_visibility_gp.png"}`',
    ]

    if experiment_dirs:
        lines.extend([
            '',
            '## Runtime Experiment Logs',
            '',
        ])
        for exp_dir in experiment_dirs:
            lines.append(f'- `{exp_dir}`')

    lines.extend([
        '',
        '## Dataset Summary',
        '',
        f'- Raw capture frames: `{capture_manifest.get("n_frames", "n/a")}`',
        f'- Raw train images: `{_count_images(bbox_dir, "train")}`',
        f'- Raw val images: `{_count_images(bbox_dir, "val")}`',
        f'- Raw dataset task: `{bbox_data.get("task", "unknown")}`',
        f'- Seg train images: `{_count_images(seg_dir, "train")}`',
        f'- Seg val images: `{_count_images(seg_dir, "val")}`',
        f'- Seg dataset task: `{seg_data.get("task", "unknown")}`',
        '',
        '## Segmentation Conversion Summary',
        '',
        f'- Total images processed: `{seg_manifest.get("results", {}).get("total_images", "n/a")}`',
        f'- Total prompts: `{seg_manifest.get("results", {}).get("total_prompts", "n/a")}`',
        f'- Accepted masks: `{seg_manifest.get("results", {}).get("total_masks_accepted", "n/a")}`',
        f'- Rejected masks: `{seg_manifest.get("results", {}).get("total_masks_rejected", "n/a")}`',
        f'- Acceptance rate: `{_format_float(seg_manifest.get("results", {}).get("acceptance_rate", "nan"))}`',
        f'- Accepted by box: `{seg_manifest.get("results", {}).get("accepted_by_box", "n/a")}`',
        f'- Accepted by point: `{seg_manifest.get("results", {}).get("accepted_by_point", "n/a")}`',
        f'- Rejected for low overlap: `{seg_manifest.get("results", {}).get("rejected_for_low_overlap", "n/a")}`',
        f'- Rejected for centroid distance: `{seg_manifest.get("results", {}).get("rejected_for_centroid_distance", "n/a")}`',
        f'- Rejected for area ratio: `{seg_manifest.get("results", {}).get("rejected_for_area_ratio", "n/a")}`',
        f'- Rejected for small area: `{seg_manifest.get("results", {}).get("rejected_for_small_area", "n/a")}`',
        f'- Rejected for no viable candidate: `{seg_manifest.get("results", {}).get("rejected_for_no_viable_candidate", "n/a")}`',
        '',
        '## Training Summary',
        '',
        f'- Recorded epochs: `{train_summary.get("epochs_recorded", "n/a")}`',
        f'- Last epoch index: `{train_summary.get("last_epoch", "n/a")}`',
        f'- Best mask mAP50: `{_format_float(train_summary.get("best_mask_map50", "nan"))}`',
        f'- Best box mAP50: `{_format_float(train_summary.get("best_box_map50", "nan"))}`',
        f'- Last mask precision: `{_format_float(train_summary.get("last_precision_mask", "nan"))}`',
        f'- Last mask recall: `{_format_float(train_summary.get("last_recall_mask", "nan"))}`',
        '',
        '## Validation Benchmark Summary',
        '',
        f'- Split: `{bench_summary.get("split", "n/a")}`',
        f'- Images: `{bench_summary.get("n_images", "n/a")}`',
        f'- Detection rate: `{_format_float(bench_summary.get("detection_rate", "nan"))}`',
        f'- Mean score: `{_format_float(bench_summary.get("mean_score", "nan"))}`',
        f'- Mean IoU: `{_format_float(bench_summary.get("mean_iou", "nan"))}`',
        f'- Mean bbox IoU: `{_format_float(bench_summary.get("mean_bbox_iou", "nan"))}`',
        f'- Mean time [ms]: `{_format_float(bench_summary.get("mean_time_ms", "nan"), digits=1)}`',
        f'- TP / FP / FN / TN: `{bench_summary.get("tp", "n/a")} / {bench_summary.get("fp", "n/a")} / {bench_summary.get("fn", "n/a")} / {bench_summary.get("tn", "n/a")}`',
        f'- Benchmark previews: `{benchmark_dir / "previews"}`',
        '',
        '## GP Fit Summary',
        '',
        f'- Label mode: `{gp_summary.get("label_mode", "n/a")}`',
        f'- Capture mode: `{gp_summary.get("capture_mode", "n/a")}`',
        f'- Raw samples: `{gp_summary.get("n_raw_samples", "n/a")}`',
        f'- Fresh samples: `{gp_summary.get("n_fresh_samples", "n/a")}`',
        f'- Occupied cells: `{gp_summary.get("n_occupied_cells", "n/a")}`',
        f'- Mean raw YOLO score: `{_format_float(gp_summary.get("mean_yolo_score_raw", "nan"))}`',
        f'- Mean cell YOLO score: `{_format_float(gp_summary.get("mean_yolo_score_cells", "nan"))}`',
        f'- GP length scale: `{_format_float(gp_summary.get("gp_length_scale", "nan"))}`',
        f'- GP noise var: `{_format_float(gp_summary.get("gp_noise_var", "nan"))}`',
        f'- Beta: `{_format_float(gp_summary.get("beta", "nan"))}`',
        '',
        '## Notes',
        '',
        '- Runtime perception stays image-only YOLO-seg with mask-bottom pixel to homography.',
        '- SAM and geometric prompts are offline-only pseudo-label helpers.',
        '- Theta in `/state/bev` remains odometry-backed in the current YOLO path.',
    ])
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate a Markdown report for the indoor YOLO pipeline outputs.')
    parser.add_argument('--bbox-dataset', required=True)
    parser.add_argument('--seg-dataset', required=True)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--benchmark-dir', required=True)
    parser.add_argument('--gp-fit-dir', required=True)
    parser.add_argument('--experiment-dir', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(_report_lines(args)) + '\n', encoding='utf-8')
    print(f'Wrote pipeline report to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
