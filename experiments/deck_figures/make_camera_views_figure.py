#!/usr/bin/env python3
"""Appendix figure: what each of the five cameras sees.

One frame per camera, chosen at that camera's own median detection range so the panel
shows a typical view rather than its best one. The robot is boxed in each. Put side by
side, the panels show why the cameras are not interchangeable: camera B works close in
and returns a large robot, while camera D looks across the hall and its robot falls near
the size at which the detector starts to fail.
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
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.0))

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
        ax.set_title(f'Camera {camera[-1]}: robot {row.camera_range_m:.0f} m away, '
                     f'{row.area:.0f} px', fontsize=9.5, fontweight='bold', pad=4)

    # The sixth cell explains the boxes rather than sitting empty.
    spare = axes.ravel()[len(cameras)]
    spare.axis('off')
    spare.text(0.5, 0.55,
               'Each panel is one camera\'s view\nof the robot at that camera\'s\n'
               'typical working distance.\n\nThe red box marks the robot.',
               ha='center', va='center', fontsize=10, color=INK, linespacing=1.6)
    spare.text(0.5, 0.16, 'Detection quality falls off\nbelow about 800 px.',
               ha='center', va='center', fontsize=9.5, color=ROBOT,
               fontweight='bold', linespacing=1.5)

    fig.tight_layout(pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'camera_views.pdf', OUT / 'camera_views.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'camera_views.pdf')


if __name__ == '__main__':
    main()
