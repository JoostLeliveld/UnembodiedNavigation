#!/usr/bin/env python3
"""Appendix figure: the view from each of the five cameras, with the robot boxed.

One frame per camera, taken at that camera's own median detection range so each panel is
a typical view rather than its best one.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
# The 20260831 characterization capture predates the current world and has been
# deleted.  The v5 recapture is the current-world capture behind the thesis.
CAPTURE = REPO / 'logs/thesis_final_pipeline_v1/recapture_v5/master_capture'
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
ROBOT, INK = '#c23d36', '#1d2530'


def main() -> None:
    rows = pd.read_csv(
        CAPTURE / 'capture_index.csv',
        usecols=['camera_id', 'image', 'capture_status', 'nominal_in_frame',
                 'camera_range_m', 'line_of_sight',
                 'expected_x0', 'expected_y0', 'expected_x1', 'expected_y1'])
    # A usable panel needs the frame to have been captured, the robot to be in
    # frame with a known box, and the camera to actually see it.
    hits = rows[(rows.capture_status == 'ok')
                & (rows.nominal_in_frame == 1)
                & (rows.line_of_sight == 1)
                & rows.expected_x0.notna()].copy()
    hits = hits.rename(columns={'expected_x0': 'x0', 'expected_y0': 'y0',
                                'expected_x1': 'x1', 'expected_y1': 'y1'})

    cameras = sorted(hits.camera_id.unique())
    # 2x3 with the last cell blank; the appendix is single column, so it has the
    # full page width to fill.
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 7.6))

    for ax, camera in zip(axes.ravel(), cameras):
        subset = hits[hits.camera_id == camera]
        median_range = subset.camera_range_m.median()
        # the frame closest to this camera's own median range: a typical view, not its best
        row = subset.iloc[(subset.camera_range_m - median_range).abs().argsort()[:1]].iloc[0]

        image = mpimg.imread(CAPTURE / row.image)
        height, width = image.shape[:2]
        ax.imshow(image)
        ax.set(xlim=(0, width), ylim=(height, 0))

        x0, y0, x1, y1 = (float(row[k]) for k in ('x0', 'y0', 'x1', 'y1'))
        pad = max(16.0, 0.35 * max(x1 - x0, y1 - y0))
        ax.add_patch(patches.Rectangle((x0 - pad, y0 - pad), (x1 - x0) + 2 * pad,
                                       (y1 - y0) + 2 * pad, fill=False,
                                       edgecolor=ROBOT, linewidth=1.9))
        ax.set_title(f'Camera {camera[-1]}', fontsize=12, fontweight='bold', pad=3)
        ax.set_xlabel(f'{float(row.camera_range_m):.1f}~m away'.replace('~', ' '),
                      fontsize=9, labelpad=2)
        ax.xaxis.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Inset: the robot region enlarged, since at these ranges the box is a
        # few tens of pixels across in a 1280x720 frame.
        half = 0.5 * max(x1 - x0, y1 - y0) * 3.2
        centre = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        left = min(max(centre[0] - half, 0.0), width - 2 * half)
        top = min(max(centre[1] - half, 0.0), height - 2 * half)
        inset = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
        inset.imshow(image)
        inset.set(xlim=(left, left + 2 * half), ylim=(top + 2 * half, top))
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor(INK)
            spine.set_linewidth(0.9)
        inset.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                          edgecolor=ROBOT, linewidth=1.4))

    axes.ravel()[len(cameras)].axis('off')
    fig.tight_layout(pad=0.35)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'camera_views.pdf', OUT / 'camera_views.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'camera_views.pdf')


if __name__ == '__main__':
    main()
