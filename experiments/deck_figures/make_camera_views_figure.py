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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
CAPTURE = REPO / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
ROBOT, INK = '#c23d36', '#1d2530'


def main() -> None:
    rows = pd.read_csv(CAPTURE / 'observation_interpretations.csv')
    hits = rows[rows.detected == 1].copy()
    hits['area'] = (hits.x1 - hits.x0) * (hits.y1 - hits.y0)

    cameras = sorted(hits.camera_id.unique())
    # 3x2 with the last cell blank: five panels stay large across a two-column page.
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 4.3))

    for ax, camera in zip(axes.ravel(), cameras):
        subset = hits[hits.camera_id == camera]
        median_range = subset.camera_range_m.median()
        # the frame closest to this camera's own median range: a typical view, not its best
        row = subset.iloc[(subset.camera_range_m - median_range).abs().argsort()[:1]].iloc[0]

        image = mpimg.imread(CAPTURE / row.image)
        height, width = image.shape[:2]
        ax.imshow(image)
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.axis('off')

        x0, y0, x1, y1 = (float(row[k]) for k in ('x0', 'y0', 'x1', 'y1'))
        pad = max(16.0, 0.35 * max(x1 - x0, y1 - y0))
        ax.add_patch(patches.Rectangle((x0 - pad, y0 - pad), (x1 - x0) + 2 * pad,
                                       (y1 - y0) + 2 * pad, fill=False,
                                       edgecolor=ROBOT, linewidth=1.9))
        ax.set_title(f'Camera {camera[-1]}', fontsize=12, fontweight='bold', pad=3)

    axes.ravel()[len(cameras)].axis('off')
    fig.tight_layout(pad=0.35)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'camera_views.pdf', OUT / 'camera_views.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'camera_views.pdf')


if __name__ == '__main__':
    main()
