#!/usr/bin/env python3
"""Rebuild every figure the thesis cites, and copy the ones that succeed into it.

Run this after changing the warehouse. It regenerates each figure from the world
file and the frozen artifacts, reports which ones a world change invalidates, and
installs the results into ../papers/Thesis/figures/.

    python3 experiments/deck_figures/rebuild_thesis_figures.py            # build and install
    python3 experiments/deck_figures/rebuild_thesis_figures.py --check    # report only
    python3 experiments/deck_figures/rebuild_thesis_figures.py --only measurement_chain

Not every figure survives a world change on its own. Each entry below records what
it reads, so the report can say what still needs doing by hand:

  live      reads the world file or the planner's map at build time, so a world
            change is picked up by rerunning this script and nothing else.
  capture   draws real camera frames from a commissioning capture. The geometry is
            recomputed, but the pixels are of the old warehouse: a world change
            needs a fresh capture before the figure tells the truth.
  drive     draws a recorded run. A world change invalidates the run itself, so the
            drive has to be repeated before the figure means anything.
  render    needs a frame captured from a running simulator, which this script
            cannot produce.
  missing   no generator is known. The figure cannot be rebuilt at all.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
THESIS = REPO.parent / 'papers' / 'Thesis' / 'figures'
BUILT = REPO / 'logs/studies/thesis_setup_figure_20260908'

# name -> (generator relative to REPO, where it writes, kind, note)
FIGURES = {
    'thesis_setup': (
        'experiments/deck_figures/make_thesis_setup_figure.py', BUILT, 'render',
        'needs logs/studies/thesis_setup_figure_20260908/gazebo_plan_view.png, '
        'captured from a running simulation; see the script docstring'),
    'measurement_chain': (
        'experiments/deck_figures/make_measurement_chain_figure.py', BUILT, 'capture',
        'camera frame from the bbox characterization capture; the map and the '
        'projection are recomputed live'),
    'localization_example': (
        'experiments/deck_figures/make_localization_example_figure.py', BUILT, 'drive',
        'the icra_commissioning drive; racks and camera poses are read from the '
        'world file, but the drive itself would have to be repeated'),
    'camera_views': (
        'experiments/deck_figures/make_camera_views_figure.py', BUILT, 'capture',
        'one real frame per camera from the characterization capture'),
    'driveable_map': (
        'experiments/deck_figures/make_driveable_map_figure.py', BUILT, 'live',
        'reads the world file and world_profiles.yaml through route_tasks.driveable()'),
    'reading_independence': (
        'experiments/reading_independence/plot_independence.py',
        REPO / 'logs/studies/reading_independence_20260907', 'drive',
        'plots logs/studies/reading_independence_20260907/results.json; rerun '
        'experiments/reading_independence/measure_independence.py to refresh it'),
    'single_camera_route_choice': (
        None, None, 'missing',
        'no generator in either repository; the PDF entered papers/Thesis at its '
        'initial commit. It has to be found or rewritten before it can be rebuilt'),
}


def build(name: str) -> tuple[bool, str]:
    script, out_dir, _, _ = FIGURES[name]
    if script is None:
        return False, 'no generator'
    result = subprocess.run([sys.executable, str(REPO / script)],
                            capture_output=True, text=True, cwd=REPO)
    if result.returncode != 0:
        tail = (result.stderr.strip().splitlines() or ['(no stderr)'])[-1]
        return False, tail
    produced = out_dir / f'{name}.pdf'
    if not produced.exists():
        return False, f'ran, but {produced} was not written'
    return True, str(produced)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true',
                        help='report what each figure needs; build nothing')
    parser.add_argument('--only', metavar='NAME', help='rebuild just this figure')
    args = parser.parse_args()

    names = [args.only] if args.only else list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f'unknown figure(s): {unknown}; known: {sorted(FIGURES)}')

    width = max(len(n) for n in names)
    if args.check:
        print(f'{"figure":<{width}}  kind      what a world change needs')
        print('-' * (width + 60))
        for name in names:
            _, _, kind, note = FIGURES[name]
            print(f'{name:<{width}}  {kind:<8}  {note}')
        return 0

    installed, failed = [], []
    for name in names:
        _, _, kind, note = FIGURES[name]
        ok, detail = build(name)
        if not ok:
            failed.append((name, kind, detail, note))
            print(f'FAIL  {name:<{width}}  {detail}')
            continue
        THESIS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(detail, THESIS / f'{name}.pdf')
        installed.append((name, kind))
        print(f'ok    {name:<{width}}  -> {THESIS / (name + ".pdf")}')

    print(f'\n{len(installed)} installed, {len(failed)} failed')
    stale = [(n, k) for n, k in installed if k in ('capture', 'drive')]
    if stale:
        print('\nRebuilt, but the underlying data is from the old warehouse. These'
              '\nredraw correctly and still show the world they were recorded in:')
        for name, kind in stale:
            print(f'  {name} ({kind}): {FIGURES[name][3]}')
    if failed:
        print('\nStill to do by hand:')
        for name, kind, detail, note in failed:
            print(f'  {name} ({kind}): {note}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
