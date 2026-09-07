#!/usr/bin/env python3
"""Exploratory camera-subset comparison on the registered six-run thesis pilot.

All singles, all pairs and the five-camera network are reported. No winning subset
is selected as a confirmatory baseline. This uses the complete opportunity loader
and a common propagation/scoring grid. It is not arrival-time or navigation replay.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/icra_mpl')
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import joblib
from study import OUT, REPO, digest, writejson as save
from commissioned_field import CAMERAS

DEFAULT_SELECTION = OUT / 'thesis_evidence/selection.json'
DEFAULT_OUT = OUT / 'thesis_network_pilot'
KINDS = ('constant', 'confidence')


def camera_subsets():
    return [(c,) for c in CAMERAS] + list(itertools.combinations(CAMERAS, 2)) + [tuple(CAMERAS)]


def subset_name(cameras):
    return '+'.join(c.removeprefix('camera_') for c in cameras)


def main(selection, out):
    from field_driving import load_run, run_filter
    selected = json.loads(selection.read_text())
    registry = json.loads((REPO / 'docs/localization_metrics_registry.json').read_text())
    registered = registry['thesis_commissioning_pilot']
    if selection.resolve() != (REPO / registered['selection']).resolve():
        raise ValueError('selection is not the registered thesis pilot')
    if selected['status'] != 'complete_pilot_diagnostic' or selected['pending'] or selected['invalid']:
        raise ValueError('complete diagnostic pilot required')
    if len(selected['runs']) != registered['runs']:
        raise ValueError('pilot run count differs')
    subsets = camera_subsets()
    protocol = dict(
        status='exploratory_pilot_not_confirmatory', selection=str(selection.relative_to(REPO)),
        selection_sha256=digest(selection), model_sha256=digest(OUT / 'models.joblib'),
        arms=[dict(name=subset_name(c), cameras=list(c)) for c in subsets], covariance=list(KINDS),
        mean='frozen bbox-feature NN and the same fitted per-camera mean offset',
        motion='measured /odom_noisy, supplied Q xy=0.01 theta=0.02; unchanged recursion',
        prediction_grid='union of measured odometry and ALL simultaneous camera-opportunity timestamps',
        scoring_grid='same odometry timestamps against time-interpolated simulator GT in each run',
        observations='same valid detector outputs with finite ground projection; no replay innovation gate',
        timing='capture-time idealization; live processing/refusals are not replayed',
        inference='per-run comparisons only; exhaustive subsets are exploratory, not a selected best-camera test',
        sources={str(p.relative_to(REPO)): digest(p) for p in [Path(__file__),
            REPO/'experiments/icra_commissioning/replay.py',
            REPO/'experiments/icra_commissioning/field_driving.py',
            REPO/'experiments/icra_commissioning/model.py',
            REPO/'experiments/fusion_on_fixed_routes/aligned.py']})
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('existing protocol differs; choose a new output directory')
    save(path, protocol)  # Freeze all subset choices before computing their outcomes.
    models = joblib.load(OUT / 'models.joblib')
    results = []
    for entry in selected['runs']:
        m, summary, truth, odom, readings, batches, accounting = load_run(entry)
        grid = [b['t'] for b in batches]
        expected_stamps = None
        for cameras in subsets:
            mask = [CAMERAS.index(c) for c in cameras]
            received = [b['t'] for b in batches if b['hits'][mask].any()]
            start, end = float(summary['first_cmd_stamp']), float(summary['stop_stamp'])
            gap = float(np.diff(sorted([start, *received, end])).max())
            for kind in KINDS:
                score, records, innovations = run_filter(m, truth, odom, readings, models,
                    kind, list(cameras), prediction_times=grid)
                stamps = np.array([r['t'] for r in records])
                if expected_stamps is None: expected_stamps = stamps
                np.testing.assert_array_equal(stamps, expected_stamps)
                expected = sum(r['camera'] in cameras for r in readings)
                if score['updates'] != expected or len(innovations) != expected:
                    raise ValueError('camera-mask observation count differs')
                if any(np.linalg.eigvalsh(r['P']).min() <= 0 for r in records):
                    raise ValueError('non-positive robot covariance')
                results.append(dict(run=entry['key'], task=entry['task'], seed=entry['seed'],
                    cameras=list(cameras), subset=subset_name(cameras), kind=kind, score=score,
                    opportunity_batches=len(batches), batches_with_reading=len(received),
                    longest_no_reading_gap_s=gap, collection_outcome=accounting['outcome'],
                    collection_live_dropped_fraction=accounting['dropped_fraction'],
                    collection_live_longest_correction_gap_s=accounting['longest_gap_s']))
        print('Completed', entry['key'], len(expected_stamps), 'common score timestamps', flush=True)
    save(out/'results.json', dict(status=protocol['status'], protocol_sha256=digest(path),
        runs=len(selected['runs']), scores=results,
        limitations=['Six previously inspected runs: three seeds per route; no significance claim.',
            'Traverse trials stop at recorded contact; paths are fixed and results do not establish navigation gains.',
            'Availability is pre-innovation-gate detector/projection support, not successful live assimilation.',
            'Calibration, remaining bias, reference error and temporal/cross-camera dependence remain potential limits.']))
    plot(results, subsets, out)


def plot(results, subsets, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'svg.fonttype':'none', 'pdf.fonttype':42})
    tasks = list(dict.fromkeys(r['task'] for r in results))
    names = [subset_name(c) for c in subsets]
    colors = {'constant':'#356184', 'confidence':'#b15c25'}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True, layout='constrained')
    for col, task in enumerate(tasks):
        for j, name in enumerate(names):
            for kind, dx in [('constant', -.14), ('confidence', .14)]:
                rows = [r for r in results if r['task']==task and r['subset']==name and r['kind']==kind]
                for row in (0, 1):
                    values = [r['score']['median_cm'] if row==0 else
                        100*r['score']['coverage']['0.95'] for r in rows]
                    axes[row,col].scatter(np.full(len(values), j+dx), values, s=18,
                        color=colors[kind], alpha=.8, label=kind if j==0 else None)
        axes[0,col].set_yscale('log')
        axes[0,col].set_title(task.replace('fusion_', '').replace('_', ' ')+' (3 runs)')
        axes[0,col].set_ylabel('Run median position error (cm; log axis)')
        axes[1,col].set_ylabel('Run 95% ellipse containment (%)')
        axes[1,col].axhline(95, color='#666666', lw=1, ls='--')
        axes[1,col].set_ylim(-2, 102)
        for ax in axes[:,col]:
            ax.set_xticks(range(len(names)), names, rotation=65, ha='right')
            ax.axvline(4.5, color='#cccccc', lw=.8)
            ax.axvline(14.5, color='#cccccc', lw=.8)
            ax.grid(axis='y', alpha=.2)
        axes[0,col].legend(frameon=False, fontsize=8)
    fig.suptitle('Camera subsets: identical-log capture-time replay\nExploratory six-run pilot; all singles, all pairs, full network', fontsize=12)
    fig.supxlabel('Available camera subset (each dot is one run; no pooled-frame confidence interval)')
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(out/f'camera_subsets.{ext}', dpi=180)
    plt.close(fig)
    save(out/'figure_manifest.json', {f'camera_subsets.{ext}':digest(out/f'camera_subsets.{ext}')
        for ext in ('svg', 'pdf', 'png')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.selection.resolve(), args.out.resolve())
