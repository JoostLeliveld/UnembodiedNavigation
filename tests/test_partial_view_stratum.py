"""The P1 selector must be able to ASK for partial views, not collect them by accident.

Experiment 37's spatial-correlation pilot could not freeze its spacing because four of five
cameras came back with 4-20 partial-view position pairs against the 20 its own rule requires.
The cause was in the selector: its only occlusion stratum, `sightline_transition`, is built
from a ray-cast to ONE point at MARKER_Z, which answers "is the centre visible" rather than
"is part of the robot hidden". See logs/studies/gate1b_partial_view_targeting/.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
# `experiments/warehouse_v2_sketches/coverage.py` collides with the installed `coverage`
# package (pytest's own coverage tool). Site-packages wins on a bare import, so the local
# directory has to come FIRST on sys.path and any already-imported copy has to be dropped.
for rel in ('experiments/camera_observation_characterization',
            'src/unav_common', 'src/experiments', 'experiments/warehouse_v2_sketches'):
    value = str((REPO / rel).resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
sys.modules.pop('coverage', None)

_sketches = str((REPO / 'experiments/warehouse_v2_sketches/coverage.py').resolve())
if not Path(_sketches).is_file():
    pytest.skip('warehouse_v2_sketches coverage engine not present', allow_module_level=True)


def _fields(half_extent_m=0.35):
    from make_icra_p1_geometry_plan import _footprint_partial_fields
    import coverage as cov
    from warehouse_v2 import build

    xs, ys = cov.grid()
    return _footprint_partial_fields(build(), xs, ys, half_extent_m=half_extent_m)


def test_every_camera_has_some_partial_geometry():
    """A stratum no camera can reach would make the selector unsatisfiable."""
    fields = _fields()
    assert set(fields) == set('ABCDE')   # CAMERA_NAMES is bare letters
    for name, field in fields.items():
        assert field.any(), f'{name} has no partial-view cells at all'


def test_partial_share_brackets_the_observed_rate():
    """Detections in the frozen capture are 14-33% partial per camera.

    These are different denominators -- floor area here, detections there -- so this is a
    magnitude check, not a calibrated prediction. It exists to catch a change that pushes
    the geometry an order of magnitude away, which is what the centre-only two-height test
    did (0.3-1.8%).
    """
    fields = _fields()
    for name, field in fields.items():
        share = float(field.mean())
        assert 0.02 < share < 0.25, f'{name} partial share {share:.3f} is implausible'


def test_a_centre_only_test_would_miss_them():
    """Pins the reason the stratum is footprint-aware rather than two-height-at-the-centre."""
    import coverage as cov
    from warehouse_v2 import build
    from make_icra_p1_geometry_plan import ROBOT_BASE_Z_M

    xs, ys = cov.grid()
    layout = build()
    height_map = cov.height_map(layout, 'A', xs, ys)

    def visible_at(model, z):
        previous = cov.MARKER_Z
        cov.MARKER_Z = z
        try:
            return cov.visible_from(model, height_map, xs, ys)
        finally:
            cov.MARKER_Z = previous

    model = cov.make_cam(layout.cameras[0])
    centre_only = visible_at(model, 0.35) & ~visible_at(model, ROBOT_BASE_Z_M)
    footprint = _fields()['A']
    # The footprint test must find substantially more than the centre-only one; that gap is
    # the whole reason the stratum exists.
    assert footprint.sum() > 3 * centre_only.sum()


def test_wider_footprint_finds_at_least_as_much():
    """Monotonicity: a larger body cannot be clipped in strictly fewer places."""
    narrow = _fields(half_extent_m=0.20)
    wide = _fields(half_extent_m=0.45)
    for name in narrow:
        assert wide[name].sum() >= narrow[name].sum()
