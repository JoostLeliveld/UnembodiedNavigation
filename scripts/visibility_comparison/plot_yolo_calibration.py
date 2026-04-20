#!/usr/bin/env python3
"""Fit YOLO temperature scaling and plot calibration diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import CURRENT_CAPTURE_DIR, CURRENT_TARGETS_DIR, LOGS_ROOT, parse_float, parse_bool01, read_csv_rows, write_manifest
from yolo_calibration_utils import apply_temperature_scaling, clip_probability, fit_temperature_scaler


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _reliability_bins(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    conf = np.full(centers.shape[0], math.nan, dtype=float)
    acc = np.full(centers.shape[0], math.nan, dtype=float)
    counts = np.zeros(centers.shape[0], dtype=int)
    for idx in range(centers.shape[0]):
        lo = bins[idx]
        hi = bins[idx + 1]
        mask = (scores >= lo) & (scores < hi)
        if idx == centers.shape[0] - 1:
            mask = (scores >= lo) & (scores <= hi)
        counts[idx] = int(np.sum(mask))
        if counts[idx] == 0:
            continue
        conf[idx] = float(np.mean(scores[mask]))
        acc[idx] = float(np.mean(labels[mask]))
    return centers, conf, acc, counts


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    _centers, conf, acc, counts = _reliability_bins(scores, labels, n_bins=n_bins)
    total = max(int(np.sum(counts)), 1)
    ece = 0.0
    for c, a, n in zip(conf, acc, counts):
        if n <= 0 or not (math.isfinite(c) and math.isfinite(a)):
            continue
        ece += abs(c - a) * (float(n) / float(total))
    return float(ece)


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores, kind='mergesort')
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    pos = max(int(np.sum(labels_sorted >= 0.5)), 1)
    neg = max(int(np.sum(labels_sorted < 0.5)), 1)
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    thresholds = [1.0]
    for score, label in zip(scores_sorted, labels_sorted):
        if label >= 0.5:
            tp += 1
        else:
            fp += 1
        tpr.append(float(tp) / float(pos))
        fpr.append(float(fp) / float(neg))
        thresholds.append(float(score))
    return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float), np.asarray(thresholds, dtype=float)


def _precision_recall_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores, kind='mergesort')
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    pos = max(int(np.sum(labels_sorted >= 0.5)), 1)
    tp = 0
    fp = 0
    precision = [1.0]
    recall = [0.0]
    thresholds = [1.0]
    for score, label in zip(scores_sorted, labels_sorted):
        if label >= 0.5:
            tp += 1
        else:
            fp += 1
        precision.append(float(tp) / float(max(tp + fp, 1)))
        recall.append(float(tp) / float(pos))
        thresholds.append(float(score))
    return np.asarray(recall, dtype=float), np.asarray(precision, dtype=float), np.asarray(thresholds, dtype=float)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _view_bins(x: np.ndarray, y: np.ndarray, theta: np.ndarray, camera_pos_xy: np.ndarray | None) -> np.ndarray:
    if camera_pos_xy is None:
        return np.full(x.shape[0], 'unknown', dtype=object)
    bearings = np.arctan2(camera_pos_xy[1] - y, camera_pos_xy[0] - x)
    relative = np.abs(_wrap_angle(bearings - theta))
    labels = np.full(relative.shape[0], 'rear', dtype=object)
    labels[relative <= (math.pi / 4.0)] = 'front'
    side_mask = (relative > (math.pi / 4.0)) & (relative <= (3.0 * math.pi / 4.0))
    labels[side_mask] = 'side'
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description='Fit YOLO score calibration and write calibration plots/artifact.')
    parser.add_argument('--perception-targets', default=str(CURRENT_TARGETS_DIR / 'perception_targets.csv'))
    parser.add_argument('--capture-manifest', default=str(CURRENT_CAPTURE_DIR / 'capture_manifest.json'))
    parser.add_argument('--out-dir', default=str(CURRENT_TARGETS_DIR / 'calibration'))
    parser.add_argument('--artifact-out', default=str(CURRENT_TARGETS_DIR / 'yolo_score_calibration.json'))
    parser.add_argument('--n-bins', type=int, default=10)
    args = parser.parse_args()

    perception_targets = Path(args.perception_targets).expanduser().resolve()
    if not perception_targets.is_file():
        raise RuntimeError(f'Perception targets CSV not found: {perception_targets}')
    out_dir = Path(args.out_dir).expanduser().resolve()
    artifact_out = Path(args.artifact_out).expanduser().resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in out_dir.parents and out_dir != allowed_root:
        raise RuntimeError(f'Calibration outputs must stay under {allowed_root}: {out_dir}')
    if allowed_root not in artifact_out.parents and artifact_out != allowed_root:
        raise RuntimeError(f'Calibration artifact must stay under {allowed_root}: {artifact_out}')
    out_dir.mkdir(parents=True, exist_ok=True)

    capture_manifest = _load_json(Path(args.capture_manifest).expanduser().resolve())
    camera_pos_xy = None
    if isinstance(capture_manifest.get('camera_pos'), list) and len(capture_manifest['camera_pos']) >= 2:
        camera_pos_xy = np.asarray(capture_manifest['camera_pos'][:2], dtype=float)

    rows = read_csv_rows(perception_targets)
    if not rows:
        raise RuntimeError(f'Perception targets CSV is empty: {perception_targets}')

    raw_scores = []
    labels = []
    xs = []
    ys = []
    headings = []
    detected_after_threshold = []
    for row in rows:
        raw = parse_float(row.get('yolo_score_raw', ''), math.nan)
        label = parse_bool01(row.get('oracle_visible', '0'))
        x = parse_float(row.get('x', ''), math.nan)
        y = parse_float(row.get('y', ''), math.nan)
        theta = parse_float(row.get('theta', ''), math.nan)
        if not math.isfinite(raw):
            continue
        raw_scores.append(float(np.clip(raw, 0.0, 1.0)))
        labels.append(float(label))
        xs.append(float(x))
        ys.append(float(y))
        headings.append(float(theta))
        detected_after_threshold.append(float(parse_bool01(row.get('yolo_detected_after_threshold', '0'))))

    if not raw_scores:
        raise RuntimeError('No finite yolo_score_raw values were found in perception_targets.csv')

    raw_scores_arr = clip_probability(np.asarray(raw_scores, dtype=float))
    labels_arr = np.asarray(labels, dtype=float)
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    theta_arr = np.asarray(headings, dtype=float)
    detected_arr = np.asarray(detected_after_threshold, dtype=float)

    fit = fit_temperature_scaler(raw_scores_arr, labels_arr)
    calibrated_scores = apply_temperature_scaling(raw_scores_arr, float(fit['temperature']))
    raw_ece = _ece(raw_scores_arr, labels_arr, n_bins=int(args.n_bins))
    calibrated_ece = _ece(calibrated_scores, labels_arr, n_bins=int(args.n_bins))
    view_bins = _view_bins(x_arr, y_arr, theta_arr, camera_pos_xy)

    centers, raw_conf, raw_acc, raw_counts = _reliability_bins(raw_scores_arr, labels_arr, n_bins=int(args.n_bins))
    _centers2, cal_conf, cal_acc, cal_counts = _reliability_bins(calibrated_scores, labels_arr, n_bins=int(args.n_bins))
    roc_fpr_raw, roc_tpr_raw, _ = _roc_curve(raw_scores_arr, labels_arr)
    roc_fpr_cal, roc_tpr_cal, _ = _roc_curve(calibrated_scores, labels_arr)
    pr_recall_raw, pr_precision_raw, _ = _precision_recall_curve(raw_scores_arr, labels_arr)
    pr_recall_cal, pr_precision_cal, _ = _precision_recall_curve(calibrated_scores, labels_arr)

    reliability_path = out_dir / 'yolo_reliability.png'
    fig_rel, axes_rel = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, conf, acc, counts, title in (
        (axes_rel[0], raw_conf, raw_acc, raw_counts, 'Raw Reliability'),
        (axes_rel[1], cal_conf, cal_acc, cal_counts, 'Calibrated Reliability'),
    ):
        ax.plot([0.0, 1.0], [0.0, 1.0], 'k--', linewidth=1.2, label='perfect')
        valid = np.isfinite(conf) & np.isfinite(acc) & (counts > 0)
        ax.plot(conf[valid], acc[valid], 'o-', linewidth=2.0, label='empirical')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed visible fraction')
        ax.set_title(title)
        ax.legend(loc='upper left')
    fig_rel.savefig(reliability_path, dpi=160)
    plt.close(fig_rel)

    curves_path = out_dir / 'yolo_pr_roc.png'
    fig_curves, axes_curves = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes_curves[0].plot(pr_recall_raw, pr_precision_raw, label='raw')
    axes_curves[0].plot(pr_recall_cal, pr_precision_cal, label='calibrated')
    axes_curves[0].set_xlabel('Recall')
    axes_curves[0].set_ylabel('Precision')
    axes_curves[0].set_title('Precision-Recall')
    axes_curves[0].legend(loc='lower left')
    axes_curves[1].plot([0.0, 1.0], [0.0, 1.0], 'k--', linewidth=1.2)
    axes_curves[1].plot(roc_fpr_raw, roc_tpr_raw, label='raw')
    axes_curves[1].plot(roc_fpr_cal, roc_tpr_cal, label='calibrated')
    axes_curves[1].set_xlabel('False positive rate')
    axes_curves[1].set_ylabel('True positive rate')
    axes_curves[1].set_title('ROC')
    axes_curves[1].legend(loc='lower right')
    fig_curves.savefig(curves_path, dpi=160)
    plt.close(fig_curves)

    hist_path = out_dir / 'yolo_score_histograms.png'
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    visible_mask = labels_arr >= 0.5
    hidden_mask = ~visible_mask
    axes_hist[0].hist(raw_scores_arr[hidden_mask], bins=20, alpha=0.65, label='oracle hidden')
    axes_hist[0].hist(raw_scores_arr[visible_mask], bins=20, alpha=0.65, label='oracle visible')
    axes_hist[0].set_title('Raw Score Histogram')
    axes_hist[0].set_xlabel('Raw score')
    axes_hist[0].set_ylabel('Count')
    axes_hist[0].legend(loc='upper center')
    axes_hist[1].hist(calibrated_scores[hidden_mask], bins=20, alpha=0.65, label='oracle hidden')
    axes_hist[1].hist(calibrated_scores[visible_mask], bins=20, alpha=0.65, label='oracle visible')
    axes_hist[1].set_title('Calibrated Score Histogram')
    axes_hist[1].set_xlabel('Calibrated score')
    axes_hist[1].set_ylabel('Count')
    axes_hist[1].legend(loc='upper center')
    fig_hist.savefig(hist_path, dpi=160)
    plt.close(fig_hist)

    view_path = out_dir / 'yolo_view_angle_bias.png'
    fig_view, axes_view = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, name in zip(axes_view, ('front', 'side', 'rear')):
        mask = view_bins == name
        if np.any(mask):
            ax.hist(raw_scores_arr[mask], bins=15, alpha=0.7, label='raw')
            ax.hist(calibrated_scores[mask], bins=15, alpha=0.5, label='calibrated')
            det_rate = float(np.mean(detected_arr[mask]))
            vis_rate = float(np.mean(labels_arr[mask]))
            ax.set_title(f'{name} view\nvisible={vis_rate:.2f}, detect={det_rate:.2f}')
        else:
            ax.set_title(f'{name} view\nno samples')
        ax.set_xlabel('score')
        ax.set_ylabel('count')
        ax.legend(loc='upper center')
    fig_view.savefig(view_path, dpi=160)
    plt.close(fig_view)

    artifact = {
        'calibration_type': 'temperature_scaling',
        'temperature': float(fit['temperature']),
        'sample_count': int(raw_scores_arr.shape[0]),
        'visible_count': int(np.sum(labels_arr >= 0.5)),
        'hidden_count': int(np.sum(labels_arr < 0.5)),
        'raw_brier': float(fit['raw_brier']),
        'calibrated_brier': float(fit['calibrated_brier']),
        'raw_nll': float(fit['raw_nll']),
        'calibrated_nll': float(fit['calibrated_nll']),
        'raw_ece': float(raw_ece),
        'calibrated_ece': float(calibrated_ece),
        'optimizer_success': bool(fit['optimizer_success']),
        'perception_targets': str(perception_targets),
        'capture_manifest': str(Path(args.capture_manifest).expanduser().resolve()) if Path(args.capture_manifest).expanduser().resolve().is_file() else '',
        'camera_pos_xy': [] if camera_pos_xy is None else [float(v) for v in camera_pos_xy],
        'view_bin_summary': {
            name: {
                'count': int(np.sum(view_bins == name)),
                'mean_raw_score': float(np.mean(raw_scores_arr[view_bins == name])) if np.any(view_bins == name) else math.nan,
                'mean_calibrated_score': float(np.mean(calibrated_scores[view_bins == name])) if np.any(view_bins == name) else math.nan,
                'mean_detected_after_threshold': float(np.mean(detected_arr[view_bins == name])) if np.any(view_bins == name) else math.nan,
            }
            for name in ('front', 'side', 'rear')
        },
        'assets': {
            'reliability_plot': str(reliability_path),
            'curve_plot': str(curves_path),
            'histogram_plot': str(hist_path),
            'view_angle_plot': str(view_path),
        },
        'notes': [
            'Raw scores are fitted against oracle_visible with scalar temperature scaling.',
            'Calibrated scores are monotone in raw scores via sigmoid(logit(score) / temperature).',
            'View-angle bins are derived from robot heading relative to the camera bearing.',
        ],
    }
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    artifact_out.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    write_manifest(out_dir / 'manifest.json', artifact)
    print(f'Wrote YOLO calibration artifact to {artifact_out}')
    print(f'Wrote YOLO calibration plots to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
