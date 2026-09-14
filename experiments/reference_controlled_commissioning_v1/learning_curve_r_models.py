#!/usr/bin/env python3
"""How do the R-study conclusions move as fit drives are added?

Refits the static ladder and D1 on nested subsets of the fit drives and rescores
everything on the same eight held-out development drives.
"""
import itertools, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path('/home/joostleliveld/Thesis/UnembodiedNavigation/experiments/reference_controlled_commissioning_v1')
sys.path.insert(0, str(HERE))
import analyze_r_models as R

campaign = Path('/home/joostleliveld/Thesis/UnembodiedNavigation/logs/commissioning/reference_controlled_v1_scientific_20260911_v3')
bias = Path('/home/joostleliveld/Thesis/UnembodiedNavigation/logs/studies/reference_controlled_commissioning_v1/bias_v2_20260911')

execution, camera_xy, drives = R.load_campaign(campaign)
corrections = R.load_frozen_corrections([bias/'fit_leave_one_drive_out_predictions.csv',
                                         bias/'development_predictions.csv'])
records = R.load_measurements(drives, camera_xy, corrections)
rounds = R.load_rounds(drives)

partition = np.asarray([r['partition'] for r in records])
drive_of = np.asarray([r['drive'] for r in records])
dev = np.flatnonzero(partition == 'development')
scale = float(np.mean([r['range_m'] for r in records]))
fit_drives = sorted(set(drive_of[partition == 'fit'].tolist()))
dev_resid = np.stack([records[i]['residual_ray'] for i in dev])
dev_drives = drive_of[dev]

rng = np.random.default_rng(0)
out = []
for k in (2, 3, 4, 5, 6, 7, 8):
    # average over several random subsets of size k so the curve is not one lucky draw
    subsets = [tuple(sorted(rng.choice(fit_drives, size=k, replace=False))) for _ in range(6)] \
        if k < 8 else [tuple(fit_drives)]
    subsets = list(dict.fromkeys(subsets))
    rows = []
    for subset in subsets:
        train = np.flatnonzero(np.isin(drive_of, list(subset)) & (partition == 'fit'))
        models = R.fit_static_ladder(records, train, scale)
        entry = {'k': k, 'n_train': int(len(train))}
        for name in ('S0', 'S1', 'S2a', 'S2b'):
            cov = R.predict_static(models, name, records, dev, scale)
            mu = R.static_mean(models, name, records, dev)
            s = R.covariance_scores(cov, dev_resid - mu, dev_drives)
            entry[f'{name}_nll'] = s['equal_drive_mean_nll']
            entry[f'{name}_cov95'] = s['containment']['95']
            entry[f'{name}_area'] = s['median_95_ellipse_area_m2']
        rows.append(entry)
    agg = {'k': k, 'n_subsets': len(rows),
           'n_train': int(np.mean([r['n_train'] for r in rows]))}
    for name in ('S0', 'S1', 'S2a', 'S2b'):
        agg[f'{name}_nll'] = float(np.mean([r[f'{name}_nll'] for r in rows]))
        agg[f'{name}_nll_sd'] = float(np.std([r[f'{name}_nll'] for r in rows]))
        agg[f'{name}_cov95'] = float(np.mean([r[f'{name}_cov95'] for r in rows]))
        agg[f'{name}_area'] = float(np.mean([r[f'{name}_area'] for r in rows]))
    out.append(agg)
    print(f"k={k} n={agg['n_train']:5d}  " + "  ".join(
        f"{n} {agg[f'{n}_nll']:+.3f}+-{agg[f'{n}_nll_sd']:.3f}" for n in ('S0','S1','S2a','S2b')))

Path('/tmp/claude-1000/-home-joostleliveld-Thesis/b1ff578a-092e-472f-bd4b-27cd958a4119/scratchpad/learning_curve.json').write_text(json.dumps(out, indent=2))
