"""The cross-bearing degree of freedom in the projection calibration.

Evidence: logs/studies/external_camera_bias_model/exp2_two_dof_bias/RESULTS.md.

The load-bearing test here is the backward-compatibility one: the deployed
along-only calibration must project bit-identically to before this DOF existed,
because every locked multicam artifact was produced with it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reliability.contracts import CameraObservation
from reliability.projection import (
    _project_pixel_to_world,
    camera_model_from_world,
    load_projection_calibration,
    project_observation_to_world,
    project_observation_to_world_with_covariance,
    projection_kwargs_for_camera,
)

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DEPLOYED = (
    ROOT / "logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v2"
    / "projection_calibration.json"
)
CONTACT_Z_M = 0.05
PIXELS = ((640.0, 500.0), (300.0, 620.0), (980.0, 430.0), (200.0, 700.0))


@pytest.fixture(scope="module")
def camera():
    if not WORLD.is_file():
        pytest.skip(f"world SDF not present: {WORLD}")
    return camera_model_from_world(WORLD, include_name="external_camera_c")


def _observation(u: float, v: float) -> CameraObservation:
    return CameraObservation(
        camera_id="camera_C", timestamp_s=0.0, pixel_uv=(u, v), detection_valid=True
    )


# ------------------------------------------------------- backward compatibility


def test_zero_cross_term_is_a_no_op(camera):
    """The new arguments default to 0.0 and must change nothing when unset."""

    for u, v in PIXELS:
        without = _project_pixel_to_world(
            u, v, camera, contact_z_m=CONTACT_Z_M,
            along_bearing_offset_m=0.12, along_bearing_slope_per_m=0.004,
        )
        with_zero = _project_pixel_to_world(
            u, v, camera, contact_z_m=CONTACT_Z_M,
            along_bearing_offset_m=0.12, along_bearing_slope_per_m=0.004,
            cross_bearing_offset_m=0.0, cross_bearing_slope_per_m=0.0,
        )
        assert with_zero == without


def test_deployed_calibration_loads_with_zero_cross_terms():
    if not DEPLOYED.is_file():
        pytest.skip(f"deployed calibration not present: {DEPLOYED}")
    calibrations = load_projection_calibration(DEPLOYED)
    assert calibrations
    for camera_id, entry in calibrations.items():
        assert entry["cross_intercept_m"] == 0.0, camera_id
        assert entry["cross_slope_per_m"] == 0.0, camera_id


def test_deployed_calibration_projects_identically_to_the_one_dof_path(camera):
    """Locked multicam artifacts were produced with the along-only path."""

    if not DEPLOYED.is_file():
        pytest.skip(f"deployed calibration not present: {DEPLOYED}")
    entry = load_projection_calibration(DEPLOYED)["camera_C"]
    kwargs = projection_kwargs_for_camera(
        load_projection_calibration(DEPLOYED), "camera_C", contact_z_m=CONTACT_Z_M
    )
    for u, v in PIXELS:
        one_dof = _project_pixel_to_world(
            u, v, camera, contact_z_m=CONTACT_Z_M,
            along_bearing_offset_m=entry["intercept_m"],
            along_bearing_slope_per_m=entry["slope_per_m"],
        )
        through_helper = _project_pixel_to_world(u, v, camera, **kwargs)
        assert through_helper == one_dof


# ------------------------------------------------------------------- geometry


def test_cross_offset_displaces_perpendicular_to_the_bearing(camera):
    """A pure cross offset moves the point sideways, leaving range unchanged."""

    cam_x, cam_y = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    for u, v in PIXELS:
        raw = _project_pixel_to_world(
            u, v, camera, contact_z_m=CONTACT_Z_M,
            along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
        )
        shifted = _project_pixel_to_world(
            u, v, camera, contact_z_m=CONTACT_Z_M,
            along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
            cross_bearing_offset_m=0.25,
        )
        assert raw is not None and shifted is not None
        displacement = (shifted[0] - raw[0], shifted[1] - raw[1])
        assert math.hypot(*displacement) == pytest.approx(0.25)
        bearing = (raw[0] - cam_x, raw[1] - cam_y)
        # Perpendicular to the bearing: zero dot product.
        assert displacement[0] * bearing[0] + displacement[1] * bearing[1] == pytest.approx(
            0.0, abs=1e-9
        )


def test_cross_offset_is_positive_to_the_left_of_the_bearing(camera):
    """Sign convention must match the one the calibration is fitted in.

    Left of the bearing means the 2-D cross product bearing x displacement is
    positive. exp2 decomposes residuals with e_cross = (-b_y, b_x)/|b| and stores
    the correction as the negation of the measured error, so a sign flip here
    would double the lateral bias instead of removing it.
    """

    cam_x, cam_y = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    u, v = PIXELS[0]
    raw = _project_pixel_to_world(
        u, v, camera, contact_z_m=CONTACT_Z_M,
        along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
    )
    shifted = _project_pixel_to_world(
        u, v, camera, contact_z_m=CONTACT_Z_M,
        along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
        cross_bearing_offset_m=0.3,
    )
    assert raw is not None and shifted is not None
    bearing = (raw[0] - cam_x, raw[1] - cam_y)
    displacement = (shifted[0] - raw[0], shifted[1] - raw[1])
    cross_product = bearing[0] * displacement[1] - bearing[1] * displacement[0]
    assert cross_product > 0.0


def test_along_and_cross_terms_are_orthogonal_and_additive(camera):
    """Each correction is a translation, so applying both must be their vector sum."""

    u, v = PIXELS[1]
    base = dict(contact_z_m=CONTACT_Z_M, along_bearing_offset_m=0.0,
                along_bearing_slope_per_m=0.0)
    raw = _project_pixel_to_world(u, v, camera, **base)
    along_only = _project_pixel_to_world(
        u, v, camera, **{**base, "along_bearing_offset_m": 0.15}
    )
    cross_only = _project_pixel_to_world(u, v, camera, **base, cross_bearing_offset_m=0.08)
    both = _project_pixel_to_world(
        u, v, camera, **{**base, "along_bearing_offset_m": 0.15}, cross_bearing_offset_m=0.08
    )
    assert raw is not None and along_only is not None
    assert cross_only is not None and both is not None
    for axis in (0, 1):
        expected = raw[axis] + (along_only[axis] - raw[axis]) + (cross_only[axis] - raw[axis])
        assert both[axis] == pytest.approx(expected, abs=1e-12)


def test_cross_slope_scales_with_ground_distance(camera):
    """cross = c0 + c1*d, with d the ground distance to the RAW projected point."""

    cam_x, cam_y = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    slope = 0.01
    for u, v in PIXELS:
        raw = _project_pixel_to_world(
            u, v, camera, contact_z_m=CONTACT_Z_M,
            along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
        )
        shifted = _project_pixel_to_world(
            u, v, camera, contact_z_m=CONTACT_Z_M,
            along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
            cross_bearing_slope_per_m=slope,
        )
        assert raw is not None and shifted is not None
        distance = math.hypot(raw[0] - cam_x, raw[1] - cam_y)
        magnitude = math.hypot(shifted[0] - raw[0], shifted[1] - raw[1])
        assert magnitude == pytest.approx(slope * distance)


# -------------------------------------------------------------- plumbing paths


def test_observation_and_covariance_paths_accept_the_cross_term(camera):
    observation = _observation(*PIXELS[2])
    point = project_observation_to_world(
        observation, camera, contact_z_m=CONTACT_Z_M, cross_bearing_offset_m=0.05
    )
    direct = _project_pixel_to_world(
        PIXELS[2][0], PIXELS[2][1], camera, contact_z_m=CONTACT_Z_M,
        along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
        cross_bearing_offset_m=0.05,
    )
    assert point == direct

    with_covariance = project_observation_to_world_with_covariance(
        _observation(*PIXELS[2]), camera, contact_z_m=CONTACT_Z_M,
        cross_bearing_offset_m=0.05,
    )
    assert with_covariance is not None
    centre, covariance = with_covariance
    assert centre == pytest.approx(direct)

    # The cross correction is NOT a constant translation in pixel space: its
    # direction is the bearing normal at the RAW projected point, which rotates
    # across the image, so the numerically differentiated Jacobian shifts a little
    # and the propagated covariance shifts with it. That is correct behaviour --
    # the same thing the along-bearing SLOPE term already does. What matters is
    # that a bias term does not materially restructure the uncertainty.
    baseline = project_observation_to_world_with_covariance(
        _observation(*PIXELS[2]), camera, contact_z_m=CONTACT_Z_M
    )
    assert baseline is not None
    for i in range(2):
        for j in range(2):
            assert covariance[i][j] == pytest.approx(baseline[1][i][j], rel=0.02)
    # Still a valid covariance.
    assert covariance[0][1] == pytest.approx(covariance[1][0], rel=1e-12)
    determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2
    assert covariance[0][0] > 0.0 and determinant > 0.0


def test_loader_accepts_the_candidate_alias_and_defaults(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"cameras": {
        "camera_A": {"intercept_m": 0.1},
        "camera_B": {"intercept_m": 0.1, "cross_bearing_offset_m": -0.02},
        "camera_C": {"intercept_m": 0.1, "cross_intercept_m": -0.078,
                     "cross_slope_per_m": 0.001},
        "camera_D": 0.05,
    }}), encoding="utf-8")
    calibrations = load_projection_calibration(path)
    assert calibrations["camera_A"]["cross_intercept_m"] == 0.0
    assert calibrations["camera_B"]["cross_intercept_m"] == pytest.approx(-0.02)
    assert calibrations["camera_C"]["cross_slope_per_m"] == pytest.approx(0.001)
    assert calibrations["camera_D"] == {
        "intercept_m": 0.05, "slope_per_m": 0.0,
        "cross_intercept_m": 0.0, "cross_slope_per_m": 0.0,
    }


def test_kwargs_helper_covers_the_whole_projection_signature(camera):
    """The helper must supply every correction argument, or a DOF silently stays off."""

    import inspect

    signature = inspect.signature(_project_pixel_to_world)
    correction_parameters = {
        name for name in signature.parameters
        if name not in ("u", "v", "camera")
    }
    kwargs = projection_kwargs_for_camera({}, "camera_A", contact_z_m=CONTACT_Z_M)
    assert set(kwargs) == correction_parameters
    assert all(value == 0.0 for name, value in kwargs.items() if name != "contact_z_m")


def test_no_node_reimplements_the_projection():
    """Nodes must call the library, not carry their own copy of the correction.

    ``scheduled_camera_detector_node`` used to hold a hand-copied version with one
    along-bearing degree of freedom. A copy like that does not fail loudly when the
    library gains a degree of freedom — it just keeps projecting differently from
    the camera manager.
    """

    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "projection.py":
            continue
        source = path.read_text(encoding="utf-8")
        # Raw ground-plane projection without any bearing correction is fine (the
        # paper-1 single-camera BEV path does exactly that). The rule is narrower:
        # anything that speaks of the bearing CORRECTION must get it from the
        # library rather than carry its own arithmetic.
        mentions_correction = "along_bearing" in source or "slope_per_m" in source
        imports_library = "reliability.projection" in source
        if mentions_correction and not imports_library:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "these modules apply a bearing correction without going through "
        f"reliability.projection: {offenders}"
    )


def test_contact_plane_travels_with_the_calibration_artifact(tmp_path) -> None:
    """The contact plane and the along-bearing term are one quantity, not two knobs.

    Intersecting the ray at height ``z`` instead of the floor shortens every estimate
    by ``z·d/(H−z)`` — exactly the form of ``slope_per_m``. While the plane was a node
    parameter and the slope was fitted per camera, the fit absorbed a constant the
    operator had chosen: measured on the four-camera captures, moving the plane from
    the historical 0.05 m to the floor cuts held-out along-bearing bias from 13.1 cm to
    4.0 cm with **no** fitted correction at all. So the artifact owns both, and an
    artifact that predates the field keeps the historical default.
    """

    from reliability.projection import load_projection_contact_z

    v2_style = tmp_path / "v2.json"
    v2_style.write_text(
        json.dumps({"contact_z_m": 0.05, "cameras": {"camera_A": {"intercept_m": 0.1}}}),
        encoding="utf-8",
    )
    assert load_projection_contact_z(v2_style) == pytest.approx(0.05)

    v4_style = tmp_path / "v4.json"
    v4_style.write_text(
        json.dumps({"contact_z_m": 0.0, "cameras": {"camera_A": {"intercept_m": 0.0}}}),
        encoding="utf-8",
    )
    assert load_projection_contact_z(v4_style) == pytest.approx(0.0)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"camera_A": 0.1}), encoding="utf-8")
    assert load_projection_contact_z(legacy, default=0.05) == pytest.approx(0.05)


def test_shipped_v4_artifact_carries_the_floor_plane_and_no_along_bearing_term() -> None:
    """v4 is the measured replacement: floor plane, zero along-bearing, gated cross."""

    payload = json.loads(
        (
            ROOT
            / "logs/studies/multicamera_commissioning_bigwarehouse"
            / "projection_calibration_v4/projection_calibration.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["contact_z_m"] == pytest.approx(0.0)
    for camera_id, entry in payload["cameras"].items():
        assert entry["intercept_m"] == pytest.approx(0.0), camera_id
        assert entry["slope_per_m"] == pytest.approx(0.0), camera_id
        # the cross-bearing term stays per-camera and gated: it is the one component
        # that does not transfer between cameras.
        assert entry["cross_term_fitted"] == (entry["cross_bias_to_sigma"] >= 1.2)
        if not entry["cross_term_fitted"]:
            assert entry["cross_intercept_m"] == pytest.approx(0.0), camera_id
