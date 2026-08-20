"""Load the three position readings that were scored on the SAME 423 held-out images.

Everything in `localization_reading_story` recomposes an existing, dated measurement.
This module is the only place that reads those measurements, so a figure can never
quietly invent a number:

  box bottom            `logs/studies/keypoint_measurement/bbox_baseline/per_sample.csv`
  marked point, old cam `logs/studies/keypoint_measurement/v3_on_current_camera/per_sample.csv`
  marked point, here    `logs/studies/keypoint_measurement/v4_retrained/per_sample.csv`

All three come from one capture, `projected_keypoint_dataset_aws_v4` (run 2026-08-17,
single-camera `warehouse_aws`, camera at (0, -5.50, 4.80)). The rows of each
per_sample.csv are the accepted `val` rows of the capture's own
`capture_diagnostics.csv`, in order -- which is how a reading is joined back to the
image it was read from. `load_reading` checks that join on the true pose and refuses
to guess.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')   # figures are written to files; cv2's Qt libs fight any GUI backend

REPO_ROOT = Path(__file__).resolve().parents[2]
for _rel in ('src/experiments', 'src/unav_common'):
    _p = str((REPO_ROOT / _rel).resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)

STUDY = REPO_ROOT / 'logs/studies/keypoint_measurement'
DATASET = REPO_ROOT / 'logs/perception_datasets/projected_keypoint_dataset_aws_v4'
OUT_DIR = REPO_ROOT / 'logs/studies/localization_reading_story/figures'

READINGS = {
    'box_bottom': ('bbox_baseline', 'bottom of the detected box, projected onto the floor'),
    'keypoint_old_camera': ('v3_on_current_camera', 'marked point, model trained for the old camera'),
    'keypoint_retrained': ('v4_retrained', 'marked point, model retrained for this camera'),
}

# Provenance line every figure carries.
SOURCE = ('423 held-out poses, one capture (projected_keypoint_dataset_aws_v4, 2026-08-17), '
          'single-camera warehouse_aws -- all readings scored on the same images. '
          'Evidence: logs/studies/keypoint_measurement/')


@dataclass
class Reading:
    key: str
    label: str
    rows: list[dict]          # every val pose, detected or not
    images: list[Path]        # image the pose was read from, same order as rows

    @property
    def detected(self) -> np.ndarray:
        return np.array([int(r['detected']) for r in self.rows], dtype=bool)

    def col(self, name: str, detected_only: bool = True) -> np.ndarray:
        vals = np.array([float(r[name]) if r.get(name) not in (None, '') else np.nan
                         for r in self.rows])
        return vals[self.detected] if detected_only else vals

    @property
    def err_cm(self) -> np.ndarray:
        """(N,2) east-west / north-south error in cm, detected readings only."""
        return 100.0 * np.column_stack([self.col('err_x_m'), self.col('err_y_m')])

    @property
    def miss_cm(self) -> np.ndarray:
        e = self.err_cm
        return np.hypot(e[:, 0], e[:, 1])


def _capture_val_rows() -> list[dict]:
    with (DATASET / 'capture_diagnostics.csv').open(encoding='utf-8') as handle:
        return [r for r in csv.DictReader(handle)
                if r['accepted'] == '1' and r['split'] == 'val']


def manifest() -> dict:
    return json.loads((DATASET / 'capture_manifest.json').read_text(encoding='utf-8'))


def camera(img_w: int = 1280, img_h: int = 720):
    from experiments.core.world_profiles import compute_look_at_from_pose
    from unav_common.camera_model import ObliqueCameraModel
    pose = [float(v) for v in manifest()['camera_pose']]
    look_at = compute_look_at_from_pose(pose[:3], pose[3], pose[4], pose[5])
    return ObliqueCameraModel(cam_pos=pose[:3], look_at=look_at,
                              img_width=img_w, img_height=img_h, fov_h_rad=1.5708)


def load_reading(key: str) -> Reading:
    folder, label = READINGS[key]
    path = STUDY / folder / 'per_sample.csv'
    with path.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    capture = _capture_val_rows()
    if len(rows) != len(capture):
        raise RuntimeError(f'{path} has {len(rows)} rows, capture val split has {len(capture)}')
    for row, cap in zip(rows, capture):
        for field in ('x', 'y', 'yaw_rad'):
            if abs(float(row[field]) - float(cap[field])) > 1e-9:
                raise RuntimeError(f'{path} does not line up with the capture on {field}')
    images = [DATASET / cap['image'] for cap in capture]
    for row, cap in zip(rows, capture):
        row['front_rendered'] = cap['front_visible']
        row['rear_rendered'] = cap['rear_visible']
    return Reading(key=key, label=label, rows=rows, images=images)


def load_all() -> dict[str, Reading]:
    return {k: load_reading(k) for k in READINGS}


def summarise(reading: Reading) -> dict:
    """The bias/spread split, computed here rather than copied from a markdown table."""
    err = reading.err_cm
    mean = err.mean(axis=0)
    centred = err - mean
    cov = np.cov(centred, rowvar=False, ddof=1)
    return {
        'label': reading.label,
        'n_poses': len(reading.rows),
        'n_read': int(reading.detected.sum()),
        'found_pct': 100.0 * reading.detected.mean(),
        'mean_east': float(mean[0]), 'mean_north': float(mean[1]),
        'systematic_cm': float(np.hypot(*mean)),
        'spread_cm': float(math.sqrt(np.trace(cov))),
        'sigma_major_cm': float(math.sqrt(max(np.linalg.eigvalsh(cov)))),
        'sigma_minor_cm': float(math.sqrt(min(np.linalg.eigvalsh(cov)))),
        'median_miss_cm': float(np.median(reading.miss_cm)),
        'p90_miss_cm': float(np.percentile(reading.miss_cm, 90)),
    }


# ------------------------------------------------------------------ figure house style

INK = '#22333b'
BOX_BOTTOM = '#9b2226'      # the deployed reading
KEYPOINT_OLD = '#e9c46a'    # marked point, stale camera
KEYPOINT = '#2a9d8f'        # marked point, retrained
MUTED = '#6b705c'
COLOUR = {'box_bottom': BOX_BOTTOM, 'keypoint_old_camera': KEYPOINT_OLD,
          'keypoint_retrained': KEYPOINT}


def style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 160, 'font.size': 10,
        'axes.titlesize': 11.5, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'legend.frameon': False, 'figure.autolayout': False,
    })


def note(fig, text: str = SOURCE, width: int = 175) -> None:
    """The provenance line every figure carries, so it stands alone.

    Wrapped, because `savefig(bbox_inches='tight')` widens the whole canvas to fit one
    long line of text and quietly ruins the layout.
    """
    import textwrap
    wrapped = '\n'.join(textwrap.fill(line, width) if line.strip() else line
                        for line in text.split('\n'))
    fig.text(0.005, 0.004, wrapped, fontsize=7, color=MUTED, va='bottom', linespacing=1.5)


def save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')
    return path
