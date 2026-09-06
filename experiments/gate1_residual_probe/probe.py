#!/usr/bin/env python3
"""Gate 1's hard stop: is the signed residual STILL predictable after the NN correction?

WHY THIS EXISTS.  Before any R is fitted, the measurement mean must be right. A covariance
fitted on top of a residual that still has predictable structure absorbs that structure as
if it were noise -- which is what the 2026-09-03 live campaign already paid for once.

"Global bias = 0.7 cm" is not the test. Least squares drives the in-sample mean to zero by
construction, and per-camera leans of opposite sign cancel when pooled (measured: pooled
6.8 cm hiding 10-18 cm per camera on the far half). The real test is adversarial:

    fit a SIMPLE probe g(runtime-available features) -> signed residual
    on one split, score it on physically disjoint places.

If a probe with a handful of parameters can still predict the sign and size of what the NN
left behind, the mean is not done and R must not be fitted yet. If it cannot, the residual
is unpredictable from what the runtime knows, which is what "zero-mean" has to mean
operationally.

WHAT IS SCORED.  The frozen paired-crop model `C_real_crop_seed0` (experiment 31), on its
development split -- the same predictions the audit retained, not a re-run. Residual is in
PIXELS, the domain the correction acts in:

    leftover = target - prediction        (target = gt contact pixel - raw box pixel)

Features are runtime-available only: box height/width/aspect, detector confidence, camera
one-hot. NO ground truth, no height ratio (that is evaluation-only), no place identity.

The split is by PHYSICAL PLACE, so a probe cannot win by memorising a location.

DECISION RULE, DECLARED BEFORE RUNNING.  Held-out R^2 of the probe against the leftover:
    R^2 <= 0.05   the mean is done; proceed to R
    0.05 - 0.15   marginal; report and decide explicitly
    R^2 > 0.15    STOP -- structure remains, fix the mean before fitting R
Reported per axis (du, dv) and pooled, plus the worst per-camera conditional mean, because
a probe that is globally weak can still be strong inside one camera.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / 'logs/studies/perception_bayesian_gaussian'
PIXELS = STUDY / 'data/pixels/pixels.npz'
FROZEN = STUDY / 'results/31_image_gate_audit/strict_seed0/C_real_crop_seed0_development.npz'

STOP_ABOVE = 0.15
MARGINAL_ABOVE = 0.05


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Least squares with a whisper of ridge, so a collinear one-hot cannot blow up."""
    xtx = x.T @ x + lam * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def design(scalars: np.ndarray, degree: int) -> np.ndarray:
    """Runtime-available features. degree=1 linear; degree=2 adds squares of the continuous ones."""
    columns = [np.ones(len(scalars))]
    continuous = scalars[:, :4]                    # box_h, box_w, aspect, confidence
    onehot = scalars[:, 4:]                        # camera identity
    # Standardise the continuous block so ridge treats the axes comparably.
    mu, sd = continuous.mean(0), continuous.std(0) + 1e-9
    z = (continuous - mu) / sd
    columns.append(z)
    if degree >= 2:
        columns.append(z ** 2)
    columns.append(onehot)
    return np.column_stack([c if c.ndim > 1 else c[:, None] for c in columns])


def r2(truth: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(((truth - predicted) ** 2).sum())
    total = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float('nan')


def main() -> int:
    data = np.load(PIXELS, allow_pickle=True)
    frozen = np.load(FROZEN, allow_pickle=True)
    index = frozen['source_index']
    leftover = data['targets'][index] - frozen['prediction']
    scalars = data['scalars'][index]
    place = data['place'][index]
    camera = data['camera'][index]
    edge = data['box_at_edge'][index]
    defined = data['target_defined'][index]

    # D017: boundary-censored rows have no defined contact point and carried 74% of the
    # squared target. They are refused operationally, so they are not the mean's job.
    keep = defined & ~edge
    leftover, scalars, place, camera = leftover[keep], scalars[keep], place[keep], camera[keep]

    # Split by PHYSICAL PLACE so the probe cannot memorise a location.
    places = np.unique(place)
    rng = np.random.default_rng(20260906)
    shuffled = rng.permutation(places)
    cut = len(shuffled) // 2
    fit_places = set(shuffled[:cut].tolist())
    is_fit = np.array([p in fit_places for p in place])

    results = {}
    for degree, label in ((1, 'linear'), (2, 'quadratic')):
        x = design(scalars, degree)
        entry = {'n_fit': int(is_fit.sum()), 'n_held_out': int((~is_fit).sum())}
        for axis, name in ((0, 'du'), (1, 'dv')):
            y = leftover[:, axis]
            beta = ridge_fit(x[is_fit], y[is_fit])
            held = ~is_fit
            entry[name] = {
                'held_out_r2': r2(y[held], x[held] @ beta),
                'in_sample_r2': r2(y[is_fit], x[is_fit] @ beta),
                'leftover_mean_px': float(y[held].mean()),
                'leftover_sd_px': float(y[held].std()),
            }
        # A globally weak probe can still be strong inside one camera.
        worst = {}
        for cam in np.unique(camera):
            mask = (camera == cam) & (~is_fit)
            if mask.sum() < 30:
                continue
            worst[str(cam)] = {
                'n': int(mask.sum()),
                'mean_du_px': float(leftover[mask, 0].mean()),
                'mean_dv_px': float(leftover[mask, 1].mean()),
            }
        entry['per_camera_leftover_mean'] = worst
        results[label] = entry

    best_r2 = max(results[k][a]['held_out_r2'] for k in results for a in ('du', 'dv'))
    verdict = ('STOP: structure remains, fix the mean before fitting R' if best_r2 > STOP_ABOVE
               else 'MARGINAL: report and decide explicitly' if best_r2 > MARGINAL_ABOVE
               else 'PASS: leftover is not predictable from runtime features')

    manifest = {
        'status': 'complete',
        'schema': 'gate1_residual_probe.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'After the frozen NN correction, can a simple runtime-available probe '
                    'still predict the signed residual on unseen places?',
        'frozen_model': 'C_real_crop_seed0 (experiment 31 strict audit), development split',
        'features': ['box_h_px', 'box_w_px', 'aspect', 'confidence', 'camera one-hot'],
        'excluded': ['boundary-censored rows (D017)', 'height_ratio (evaluation-only)',
                     'place identity', 'any ground-truth quantity'],
        'split': 'by physical place, half/half, seed 20260906',
        'decision_rule': {'stop_above': STOP_ABOVE, 'marginal_above': MARGINAL_ABOVE,
                          'declared': 'before running'},
        'n_rows_scored': int(keep.sum()),
        'best_held_out_r2': best_r2,
        'verdict': verdict,
        'results': results,
    }
    out = REPO / 'logs/studies/gate1_residual_probe'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'probe_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'rows scored: {keep.sum()}  (fit {is_fit.sum()} / held out {(~is_fit).sum()} by place)')
    for label, entry in results.items():
        print(f'\n--- {label} probe ---')
        for name in ('du', 'dv'):
            e = entry[name]
            print(f"  {name}: held-out R2 {e['held_out_r2']:+.4f}  "
                  f"(in-sample {e['in_sample_r2']:+.4f})  "
                  f"leftover {e['leftover_mean_px']:+.3f} +- {e['leftover_sd_px']:.3f} px")
        print('  per-camera leftover mean (held out):')
        for cam, v in entry['per_camera_leftover_mean'].items():
            print(f"    {cam}: du {v['mean_du_px']:+.3f}  dv {v['mean_dv_px']:+.3f} px  (n={v['n']})")
    print(f'\nbest held-out R2 = {best_r2:+.4f}\nVERDICT: {verdict}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
