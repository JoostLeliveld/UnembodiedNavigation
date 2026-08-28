"""Reading the frozen commissioning artifacts.  No drawing, no fitting, no re-deriving.

Everything here comes out of ``logs/studies/measurement_commissioning/``, written once by
``experiments/measurement_commissioning/commission.py``.  Nothing in the deck recomputes a
commissioning number -- if a figure disagrees with ``calibration.json``, the figure is wrong.

**Never match a row in one file to a row in another by coordinate.**  The capture stores
full floats, every written file stores formatted ones, and the sample grid lands exactly on
rounding ties, so rounding, lattice-snapping and decimal formatting each fail somewhere.
That silently corrupted results three times -- once inflating "no camera can help here" from
15% to 24%.  Every position carries an integer ``position_id``; joins use it.
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COMMISSIONING = REPO / "logs/studies/measurement_commissioning"


def calibration():
    """The frozen result: the offset coefficients, sigma, the detector's characterization."""
    return json.loads((COMMISSIONING / "calibration.json").read_text())


def availability_rows():
    """Every (camera, pose) trial with its outcome, straight from the commissioning table.

    Read as one file rather than reconstructed by joining files on coordinates: the sample
    grid lands exactly on rounding ties, so any coordinate join silently detaches a slice of
    the usable sightings and makes well-covered places look blind.
    """
    path = COMMISSIONING / "availability.csv"
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append((r["camera"], float(r["x"]), float(r["y"]), float(r["yaw"]),
                     int(r["line_of_sight"]), int(r["usable"])))
    return rows


def _by_position(pick):
    per = collections.defaultdict(list)
    for cam, x, y, yaw, los, use in availability_rows():
        per[(x, y, yaw)].append((cam, los, use))
    out = collections.defaultdict(list)
    for (x, y, _yaw), v in per.items():
        out[(x, y)].append(pick(v))
    return {k: float(np.mean(v)) for k, v in out.items()}


def support_field(min_cameras=1):
    """Per floor position: the fraction of headings with at least `min_cameras` usable views."""
    f = _by_position(lambda v: sum(u for _c, _l, u in v) >= min_cameras)
    return f, {k: 6 for k in f}


def geometric_field():
    """Per floor position: fraction of headings where at least one camera had a clear view.

    The simulator's own answer to whether the line of sight is open, with no detector and no
    admission checks -- the honest version of what a field-of-view model would predict.
    """
    return _by_position(lambda v: any(l for _c, l, _u in v))


def per_camera_fields():
    """Availability per camera per floor position: fraction of headings that were usable."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for cam, x, y, _yaw, _los, use in availability_rows():
        per[cam][(x, y)].append(use)
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in per.items()}


def sightings():
    """The usable sightings and their pixel residuals, as commissioning recorded them."""
    return list(csv.DictReader(open(COMMISSIONING / "sightings.csv")))


def offset_positions():
    """The ids of the spots the commissioned offset was fitted on."""
    return {int(r["position_id"])
            for r in csv.DictReader(open(COMMISSIONING / "offset_positions.csv"))}
