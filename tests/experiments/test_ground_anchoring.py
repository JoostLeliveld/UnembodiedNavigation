"""Acceptance suite for the ground-anchoring / visibility-inference method.

Scope note, because this repo has a hard rule against synthetic evidence: the
depth images below are produced by an exact ray/box intersection, hand-checkable
line by line, and they exist to test *algebra* -- does an exact plane recover an
exact affine, does a mislabelled convention raise, does an obstacle of known
size cast the shadow trigonometry says it must. They are unit-test fixtures, not
an experiment. Operational evidence for this method comes from the dynamic
Gazebo scenarios (agent 1) run through real monocular predictions (agent 2), and
nothing here may be quoted as a result.

Covers the agreed acceptance criteria:

1. exact plane -> exact recovery
2. scale/shift perturbations recovered, in every declared convention
3. wrong floor masks handled robustly
4. insufficient floor evidence returns unknown
5. a known obstacle produces the correct visibility shadow
6. depth-convention mismatches fail loudly

plus determinism, the oracle boundary, and the output contract itself.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "experiments" / "mono_depth_visibility", _ROOT / "src" / "unav_common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ground_anchoring as ga  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402


# ---------------------------------------------------------------------------
# Exact ray/box renderer -- UNIT-TEST FIXTURE ONLY (see module docstring)
# ---------------------------------------------------------------------------
class Box:
    """Axis-aligned box in world coordinates, metres."""

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, name="box"):
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.zmin, self.zmax = float(zmin), float(zmax)
        self.name = name

    @property
    def lo(self):
        return np.array([self.xmin, self.ymin, self.zmin])

    @property
    def hi(self):
        return np.array([self.xmax, self.ymax, self.zmax])


def _ray_box_t(origin: np.ndarray, dirs: np.ndarray, box: Box) -> np.ndarray:
    """Slab intersection. Returns entry ``t`` per ray (NaN when missed).

    ``dirs`` is ``(3, N)`` and carries a z-row of 1 for camera rays, so ``t``
    comes out directly as optical-axis depth.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box.lo[:, None] - origin[:, None]) / dirs
        t2 = (box.hi[:, None] - origin[:, None]) / dirs
    tmin = np.nanmax(np.minimum(t1, t2), axis=0)
    tmax = np.nanmin(np.maximum(t1, t2), axis=0)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax > 0.0)
    t_entry = np.where(tmin > 0.0, tmin, tmax)
    return np.where(hit, t_entry, np.nan)


def render_depth(calib, boxes=(), plane=None, floor: bool = True) -> np.ndarray:
    """Exact optical-axis depth image of a floor plane plus axis-aligned boxes."""
    plane = plane or ga.FloorPlane()
    u, v = calib.pixel_grid(1)
    dirs = calib.rays_world(u, v)
    depth = np.full(u.size, np.inf)
    if floor:
        d_floor, _ = ga.analytic_plane_depth(calib, plane, u, v)
        depth = np.fmin(depth, np.where(np.isfinite(d_floor), d_floor, np.inf))
    for b in boxes:
        t = _ray_box_t(calib.cam_pos, dirs, b)
        depth = np.fmin(depth, np.where(np.isfinite(t), t, np.inf))
    depth = np.where(np.isfinite(depth), depth, np.nan)
    return depth.reshape(calib.height, calib.width)


def blocked_by_boxes(calib, xs, ys, boxes, target: ga.TargetVolume) -> np.ndarray:
    """Exact geometric shadow of ``boxes`` for the robot volume, per grid cell.

    Independent of the method: the test's own trigonometry, used to check the
    method's answer. It is the test that knows the boxes, never the method.
    """
    gx, gy = np.meshgrid(xs, ys)
    offsets = target.sample_offsets()
    blocked = np.ones(gx.shape + (offsets.shape[0],), dtype=bool)
    for m, off in enumerate(offsets):
        tgt = np.stack([gx + off[0], gy + off[1], np.full_like(gx, off[2])], -1).reshape(-1, 3)
        seg = tgt - calib.cam_pos
        length = np.linalg.norm(seg, axis=1)
        dirs = (seg / length[:, None]).T
        hit_any = np.zeros(tgt.shape[0], dtype=bool)
        for b in boxes:
            t = _ray_box_t(calib.cam_pos, dirs, b)
            hit_any |= np.isfinite(t) & (t > 1e-6) & (t < length - 1e-6)
        blocked[..., m] = hit_any.reshape(gx.shape)
    return blocked.all(axis=-1)


def any_offset_blocked(calib, xs, ys, boxes, target: ga.TargetVolume) -> np.ndarray:
    """Penumbra-inclusive shadow: at least one point of the body is blocked."""
    gx, gy = np.meshgrid(xs, ys)
    hit = np.zeros(gx.shape, dtype=bool)
    for off in target.sample_offsets():
        tgt = np.stack([gx + off[0], gy + off[1], np.full_like(gx, off[2])], -1).reshape(-1, 3)
        seg = tgt - calib.cam_pos
        length = np.linalg.norm(seg, axis=1)
        dirs = (seg / length[:, None]).T
        for b in boxes:
            t = _ray_box_t(calib.cam_pos, dirs, b)
            hit |= (np.isfinite(t) & (t > 1e-6) & (t < length - 1e-6)).reshape(gx.shape)
    return hit


def dilate(mask: np.ndarray, cells: int = 1) -> np.ndarray:
    """Grow a boolean mask by ``cells`` in each direction (Chebyshev)."""
    out = mask.copy()
    for _ in range(cells):
        grown = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = grown
    return out


def erode(mask: np.ndarray, cells: int = 1) -> np.ndarray:
    """Shrink a boolean mask by ``cells``: its interior, away from the edge."""
    return ~dilate(~mask, cells)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
CAMERAS = {
    "A": dict(cam_pos=(-3.0, -3.0, 6.0), look_at=(1.5, 1.5, 0.0)),
    "B": dict(cam_pos=(9.0, -3.0, 6.0), look_at=(3.0, 1.5, 0.0)),
    "C": dict(cam_pos=(-3.0, 9.0, 6.0), look_at=(1.5, 3.0, 0.0)),
    "D": dict(cam_pos=(9.0, 9.0, 6.0), look_at=(3.0, 3.0, 0.0)),
    # optical axis along +x, so lines of constant depth on the floor run along
    # y and an axis-aligned strip can be made almost iso-depth
    "X": dict(cam_pos=(-3.0, 2.5, 6.0), look_at=(4.0, 2.5, 0.0)),
}
BOX = Box(2.0, 2.8, 2.0, 2.8, 0.0, 1.2, "pallet")


def make_calib(name: str = "A", width: int = 320, height: int = 240) -> ga.CameraCalibration:
    cam = ObliqueCameraModel(
        img_width=width, img_height=height, fov_h_rad=1.2, **CAMERAS[name]
    )
    return ga.CameraCalibration.from_oblique(cam, camera_id=name)


def make_grid(res: float = 0.25):
    return np.arange(-1.0, 8.0 + res, res), np.arange(-1.0, 8.0 + res, res)


DRIVABLE = [ga.Footprint(-1.0, 8.0, -1.0, 8.0, "aisle")]


def prediction(values, convention=ga.DepthConvention.METRIC_Z, **kw):
    return ga.DepthPrediction(values=values, convention=convention, **kw)


# ---------------------------------------------------------------------------
# 1. Exact plane -> exact recovery
# ---------------------------------------------------------------------------
def test_exact_floor_plane_recovers_identity_affine():
    calib = make_calib()
    depth = render_depth(calib)
    fit = _fit_only(calib, depth, ga.DepthConvention.METRIC_Z)

    assert fit.status is ga.FrameStatus.OK
    assert fit.scale == pytest.approx(1.0, abs=1e-9)
    assert fit.shift == pytest.approx(0.0, abs=1e-9)
    assert fit.residual_rms_m == pytest.approx(0.0, abs=1e-9)
    assert fit.inlier_fraction == pytest.approx(1.0)
    assert fit.n_beyond_floor == 0 and fit.n_shorter_than_floor == 0


def test_exact_plane_end_to_end_leaves_no_obstacle_and_no_shadow():
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib)
    result = ga.estimate_visibility(
        prediction(depth), calib, xs, ys, drivable=DRIVABLE, scenario_id="unit"
    )

    assert result.status is ga.FrameStatus.OK
    f = result.visibility
    seen = f.observed & f.in_fov
    assert seen.sum() > 100
    # an empty floor rasterises to zero height everywhere it was seen
    assert float(np.max(f.height_map_m[seen])) < 1e-6
    # and every seen cell is confidently visible
    assert float(np.min(f.p_visible[seen])) > 0.99


def _fit_only(calib, values, convention, config=None, drivable=None, valid_mask=None):
    """Anchor selection + affine fit, without the raycast (unit-level check)."""
    cfg = config or ga.MethodConfig()
    anchors = ga.select_floor_anchors(
        calib, ga.FloorPlane(), drivable if drivable is not None else DRIVABLE,
        config=cfg.anchors, valid_mask=valid_mask,
    )
    pred = ga.to_optical_axis(
        np.asarray(values)[anchors.v.astype(int), anchors.u.astype(int)],
        convention, calib, u=anchors.u, v=anchors.v,
    )
    return ga.fit_ground_affine(
        pred, anchors.depth_m, convention, config=cfg.fit,
        anchor_depth_span_m=anchors.depth_span_m,
    )


# ---------------------------------------------------------------------------
# 2. Scale / shift perturbations recovered, per convention
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "convention, scale, shift",
    [
        (ga.DepthConvention.METRIC_Z, 1.0, 0.0),
        (ga.DepthConvention.METRIC_Z, 0.80, 1.30),
        (ga.DepthConvention.METRIC_Z, 1.45, -2.10),
        (ga.DepthConvention.RELATIVE_DEPTH, 0.04, 3.0),
        (ga.DepthConvention.RELATIVE_DEPTH, 22.0, -5.0),
        (ga.DepthConvention.INVERSE_DEPTH, 1.0, 0.0),
        (ga.DepthConvention.INVERSE_DEPTH, 0.35, 0.02),
        (ga.DepthConvention.EUCLIDEAN_RANGE, 1.0, 0.0),
    ],
)
def test_scale_and_shift_are_recovered(convention, scale, shift):
    """Build a prediction that needs a known affine, then demand it back."""
    calib = make_calib()
    depth = render_depth(calib)
    truth = ga.to_optical_axis(depth, ga.DepthConvention.METRIC_Z, calib)

    if convention is ga.DepthConvention.EUCLIDEAN_RANGE:
        u, v = calib.pixel_grid(1)
        norms = calib.ray_norms(u, v).reshape(depth.shape)
        values = (depth * norms - shift) / scale
    else:
        y = ga.target_in_fit_space(truth, convention)
        values = (y - shift) / scale

    fit = _fit_only(calib, values, convention)
    assert fit.status is ga.FrameStatus.OK
    assert fit.scale == pytest.approx(scale, rel=1e-6)
    assert fit.shift == pytest.approx(shift, abs=1e-6 * max(1.0, abs(shift)))

    recovered = fit.apply(ga.to_optical_axis(values, convention, calib))
    ok = np.isfinite(recovered) & np.isfinite(depth)
    assert np.allclose(recovered[ok], depth[ok], rtol=1e-6, atol=1e-6)


def test_relative_model_recovers_metric_depth_end_to_end():
    """A model with no idea of scale still yields metres after anchoring."""
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib, boxes=[BOX])
    values = (depth - 4.0) / 0.03  # arbitrary affine-invariant units

    result = ga.estimate_visibility(
        prediction(values, ga.DepthConvention.RELATIVE_DEPTH), calib, xs, ys, drivable=DRIVABLE
    )
    assert result.status is ga.FrameStatus.OK
    md = result.metric_depth
    err = np.abs(md.depth_m[md.valid] - depth[md.valid])
    # not exact: the inlier band is a deliberate 5% noise allowance, so the
    # obstacle pixels closest to the floor sit inside it and enter the fit
    assert float(np.nanmax(err)) < 0.01
    # the box top ends up at its true height in the sensed map
    top = result.visibility.height_map_m.max()
    assert top == pytest.approx(BOX.zmax, abs=0.15)


# ---------------------------------------------------------------------------
# 3. Wrong floor masks handled robustly
# ---------------------------------------------------------------------------
def test_obstacle_pixels_inside_the_declared_floor_do_not_bias_the_fit():
    """The drivable map says the aisle is clear; a box is standing in it.

    Those pixels are anchors by construction and lie in front of the plane.
    They must be rejected, not averaged in -- and their count is the evidence
    that something is standing there.
    """
    calib = make_calib()
    depth = render_depth(calib, boxes=[BOX])
    fit = _fit_only(calib, depth, ga.DepthConvention.METRIC_Z)

    assert fit.status is ga.FrameStatus.OK
    # the obstacle pixels deep inside the box are rejected outright; the sliver
    # at its base lies within the 5% inlier band, which is what keeps this a
    # near-identity rather than an exact one
    assert fit.scale == pytest.approx(1.0, abs=1e-3)
    assert fit.shift == pytest.approx(0.0, abs=1e-2)
    assert fit.n_shorter_than_floor > 0, "the box should show up as short anchors"
    assert fit.n_beyond_floor == 0, "nothing can be behind the floor plane"


@pytest.mark.parametrize("corrupt_fraction", [0.1, 0.3, 0.5])
def test_fit_survives_a_corrupted_floor_mask(corrupt_fraction):
    """A segmentation that wrongly labels wall/rack pixels as floor."""
    calib = make_calib()
    depth = render_depth(calib)
    rng = np.random.default_rng(7)
    corrupted = depth.copy()
    n = corrupted.size
    idx = rng.choice(n, size=int(corrupt_fraction * n), replace=False)
    flat = corrupted.ravel()
    flat[idx] = flat[idx] * rng.uniform(0.2, 0.7, size=idx.size)  # much nearer surfaces

    fit = _fit_only(calib, corrupted, ga.DepthConvention.METRIC_Z)
    assert fit.status is ga.FrameStatus.OK
    assert fit.scale == pytest.approx(1.0, abs=1e-3)
    assert fit.shift == pytest.approx(0.0, abs=1e-2)
    assert fit.inlier_fraction == pytest.approx(1.0 - corrupt_fraction, abs=0.05)


def test_majority_corruption_is_refused_not_fitted():
    """Past the robustness budget the method must decline, not guess."""
    calib = make_calib()
    depth = render_depth(calib)
    rng = np.random.default_rng(11)
    corrupted = depth * rng.uniform(0.2, 0.7, size=depth.shape)
    keep = rng.random(depth.shape) < 0.2
    corrupted = np.where(keep, depth, corrupted)  # only 20% real floor

    fit = _fit_only(calib, corrupted, ga.DepthConvention.METRIC_Z)
    assert fit.status is not ga.FrameStatus.OK
    assert fit.status in (
        ga.FrameStatus.LOW_INLIER_FRACTION,
        ga.FrameStatus.HIGH_RESIDUAL,
    )


def test_explicit_floor_segmentation_narrows_the_anchor_set():
    calib = make_calib()
    seg = np.zeros((calib.height, calib.width), dtype=bool)
    seg[calib.height // 2 :, :] = True
    wide = ga.select_floor_anchors(calib, ga.FloorPlane(), DRIVABLE)
    narrow = ga.select_floor_anchors(
        calib, ga.FloorPlane(), DRIVABLE, floor_segmentation=seg
    )
    assert 0 < len(narrow) < len(wide)
    assert narrow.stage_counts["inside_segmentation"] == len(narrow)


def test_enhanced_anchors_erode_the_floor_segmentation_boundary():
    """Object/floor boundary pixels must not become trusted anchors."""
    calib = make_calib()
    seg = np.zeros((calib.height, calib.width), dtype=bool)
    boundary = calib.height // 2
    seg[boundary:, :] = True
    cfg = ga.AnchorConfig(quality_filter=True, segmentation_erosion_px=5)

    anchors = ga.select_floor_anchors(
        calib, ga.FloorPlane(), DRIVABLE, config=cfg, floor_segmentation=seg
    )

    assert len(anchors) > 0
    assert np.min(anchors.v) >= boundary + cfg.segmentation_erosion_px
    assert anchors.stage_counts["inside_eroded_segmentation"] == len(anchors)
    assert np.allclose(anchors.weights.mean(), 1.0)


def test_enhanced_anchors_prefer_native_high_confidence_pixels():
    """Native confidence is used by rank, without assuming its raw units."""
    calib = make_calib()
    confidence = np.broadcast_to(
        np.linspace(0.0, 1000.0, calib.width), (calib.height, calib.width)
    )
    baseline = ga.select_floor_anchors(calib, ga.FloorPlane(), DRIVABLE)
    enhanced = ga.select_floor_anchors(
        calib,
        ga.FloorPlane(),
        DRIVABLE,
        config=ga.AnchorConfig(
            quality_filter=True,
            confidence_keep_fraction=0.5,
            depth_edge_dilation_px=0,
        ),
        native_confidence=confidence,
    )

    assert 0 < len(enhanced) < len(baseline)
    assert enhanced.stage_counts["high_confidence"] == len(enhanced)
    assert np.median(enhanced.u) > np.median(baseline.u)
    assert np.all(np.isfinite(enhanced.weights))
    assert enhanced.weights.mean() == pytest.approx(1.0)


def test_flat_depth_prediction_is_not_misclassified_as_all_edges():
    calib = make_calib()
    baseline = ga.select_floor_anchors(calib, ga.FloorPlane(), DRIVABLE)
    enhanced = ga.select_floor_anchors(
        calib,
        ga.FloorPlane(),
        DRIVABLE,
        config=ga.AnchorConfig(quality_filter=True),
        prediction_values=np.ones((calib.height, calib.width)),
    )
    assert len(enhanced) == len(baseline)


def test_quality_weights_reduce_the_influence_of_weak_floor_pixels():
    """Low-confidence, near-inlier pixels should not pull the affine fit."""
    pred = np.linspace(0.0, 10.0, 500)
    truth = 2.0 * pred + 1.0
    weak = pred > 6.0
    corrupted_prediction = pred.copy()
    corrupted_prediction[weak] += 0.15
    weights = np.where(weak, 0.05, 1.0)
    cfg = ga.FitConfig(
        min_anchor_pixels=20,
        inlier_rel_tol=0.10,
        inlier_abs_tol_m=0.10,
    )

    unweighted = ga.fit_ground_affine(
        corrupted_prediction, truth, ga.DepthConvention.RELATIVE_DEPTH, config=cfg
    )
    weighted = ga.fit_ground_affine(
        corrupted_prediction,
        truth,
        ga.DepthConvention.RELATIVE_DEPTH,
        config=cfg,
        weights=weights,
    )

    assert unweighted.status is ga.FrameStatus.OK
    assert weighted.status is ga.FrameStatus.OK
    unweighted_error = abs(unweighted.scale - 2.0) + abs(unweighted.shift - 1.0)
    weighted_error = abs(weighted.scale - 2.0) + abs(weighted.shift - 1.0)
    assert weighted_error < 0.40 * unweighted_error


# ---------------------------------------------------------------------------
# 4. Insufficient floor evidence -> unknown
# ---------------------------------------------------------------------------
def test_no_valid_pixels_returns_all_unknown():
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib)
    dead = np.zeros_like(depth, dtype=bool)

    result = ga.estimate_visibility(
        prediction(depth, valid_mask=dead), calib, xs, ys, drivable=DRIVABLE
    )
    assert result.status is ga.FrameStatus.INSUFFICIENT_FLOOR_PIXELS
    assert not result.is_valid
    f = result.visibility
    assert np.all(f.p_unknown == 1.0)
    assert np.all(f.p_visible == 0.0) and np.all(f.p_occluded == 0.0)
    assert np.all(f.unknown_mask)
    assert np.all(np.isnan(f.p_los)), "an unknown cell must not report a made-up p_LOS"


def test_anchors_at_one_range_are_refused_as_unidentifiable():
    """Scale needs depth diversity, exactly as bias needs bearing diversity."""
    calib = make_calib("X")
    xs, ys = make_grid()
    depth = render_depth(calib)
    # camera X looks along +x, so this strip is a band of near-constant depth:
    # plenty of pixels, almost no depth spread
    patch = [ga.Footprint(4.4, 4.8, -1.0, 6.0, "iso_depth_strip")]
    cfg = ga.MethodConfig(fit=ga.FitConfig(min_anchor_pixels=20, min_depth_span_m=1.0))

    anchors = ga.select_floor_anchors(calib, ga.FloorPlane(), patch, config=cfg.anchors)
    assert len(anchors) >= 20, "fixture must supply enough pixels for the span gate to be the cause"
    assert anchors.depth_span_m < 1.0

    result = ga.estimate_visibility(
        prediction(depth), calib, xs, ys, drivable=patch, config=cfg
    )
    assert result.status is ga.FrameStatus.INSUFFICIENT_DEPTH_SPAN
    assert np.all(result.visibility.p_unknown == 1.0)
    assert "not identifiable" in result.ground_fit.notes


def test_too_few_anchors_is_reported_with_the_stage_that_emptied_them():
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib)
    result = ga.estimate_visibility(
        prediction(depth), calib, xs, ys,
        drivable=[ga.Footprint(50.0, 51.0, 50.0, 51.0, "elsewhere")],
    )
    assert result.status is ga.FrameStatus.INSUFFICIENT_FLOOR_PIXELS
    counts = result.provenance["anchor_stage_counts"]
    assert counts["within_range"] > 0 and counts["inside_drivable"] == 0


def test_require_drivable_without_a_map_is_a_contract_violation():
    calib = make_calib()
    with pytest.raises(ga.ContractViolation, match="undeclared floor"):
        ga.select_floor_anchors(calib, ga.FloorPlane(), [])


# ---------------------------------------------------------------------------
# 5. A known obstacle produces the correct visibility shadow
# ---------------------------------------------------------------------------
def test_known_obstacle_casts_the_shadow_the_geometry_predicts():
    calib = make_calib()
    xs, ys = make_grid()
    cfg = ga.MethodConfig()
    depth = render_depth(calib, boxes=[BOX])

    clear = ga.estimate_visibility(
        prediction(render_depth(calib)), calib, xs, ys, drivable=DRIVABLE, config=cfg
    )
    blocked_result = ga.estimate_visibility(
        prediction(depth), calib, xs, ys, drivable=DRIVABLE, config=cfg
    )
    assert clear.status is ga.FrameStatus.OK and blocked_result.status is ga.FrameStatus.OK

    truth_shadow = blocked_by_boxes(calib, xs, ys, [BOX], cfg.target)
    f = blocked_result.visibility
    # the fixture must actually discriminate: some cells shadowed, many not
    assert 20 < int(truth_shadow.sum()) < int(0.4 * truth_shadow.size)

    # Inside the shadow -- away from its one-cell boundary, where a body point
    # can land on a pixel just past the obstacle's silhouette -- visibility is
    # gone, and it is accounted for rather than silently dropped.
    core = erode(truth_shadow, 1)
    assert int(core.sum()) > 10, "the shadow must have an interior to test"
    assert float(np.max(f.p_visible[core])) < 0.02
    assert float(np.min((f.p_occluded + f.p_unknown)[core])) > 0.98
    # taken as a whole, including its graded edge, the shadow is dark
    assert float(np.mean(f.p_visible[truth_shadow])) < 0.05

    # Nothing outside the obstacle's geometric reach may darken. The tolerance
    # is exactly the representation's resolution: a 2.5-D raster snaps the box
    # to whole cells, so its footprint is up to half a cell wide on each side
    # and the shadow boundary moves with it. One cell of dilation covers that;
    # anything darkened beyond it would be a real error.
    reach = dilate(any_offset_blocked(calib, xs, ys, [BOX], cfg.target), cells=1)
    was_visible = (clear.visibility.p_visible > 0.99) & ~reach
    lost = was_visible & (f.p_visible < 0.5)
    assert int(lost.sum()) == 0, (
        f"{int(lost.sum())} cells darkened that the obstacle does not stand between"
    )


def test_removing_the_obstacle_restores_the_original_field():
    """The event that matters for the dynamic scenario: box in, box out."""
    calib = make_calib()
    xs, ys = make_grid()
    before = ga.estimate_visibility(
        prediction(render_depth(calib)), calib, xs, ys, drivable=DRIVABLE
    )
    during = ga.estimate_visibility(
        prediction(render_depth(calib, boxes=[BOX])), calib, xs, ys, drivable=DRIVABLE
    )
    after = ga.estimate_visibility(
        prediction(render_depth(calib)), calib, xs, ys, drivable=DRIVABLE
    )

    assert np.array_equal(before.visibility.p_visible, after.visibility.p_visible)
    assert during.visibility.p_visible.sum() < before.visibility.p_visible.sum() - 5.0
    assert during.visibility.height_map_m.max() > 1.0
    assert after.visibility.height_map_m.max() < 1e-6


def test_a_taller_obstacle_casts_a_longer_shadow():
    """Monotonicity: the shadow must respond to the thing casting it."""
    calib = make_calib()
    xs, ys = make_grid()
    short = Box(2.0, 2.8, 2.0, 2.8, 0.0, 0.6, "short")
    tall = Box(2.0, 2.8, 2.0, 2.8, 0.0, 2.0, "tall")
    areas = []
    for box in (short, tall):
        r = ga.estimate_visibility(
            prediction(render_depth(calib, boxes=[box])), calib, xs, ys, drivable=DRIVABLE
        )
        areas.append(float((r.visibility.p_visible < 0.5).sum()))
    assert areas[1] > areas[0]


def test_ground_hidden_by_a_seen_box_is_occluded_not_unknown():
    """Hidden *by something we can see* is a known reason, so it is occlusion.

    The region behind the box gets no depth return of its own -- it appears in
    the height map as unobserved -- but a robot standing there would be
    invisible because the box is in the way, and the method must say so instead
    of shrugging.
    """
    calib = make_calib()
    xs, ys = make_grid()
    r = ga.estimate_visibility(
        prediction(render_depth(calib, boxes=[BOX])), calib, xs, ys, drivable=DRIVABLE
    )
    f = r.visibility
    box_cells = (
        (np.abs(f.xs[None, :] - 2.4) < 0.45) & (np.abs(f.ys[:, None] - 2.4) < 0.45)
    )
    assert bool(np.any(f.observed & box_cells)), "the box surface itself is seen"
    assert float(np.max(f.p_occluded)) > 0.95, "a seen obstacle must produce occluded mass"
    # the ground behind it has no return of its own ...
    hidden = f.in_fov & ~f.observed
    assert int(hidden.sum()) > 10
    # ... and away from the silhouette edge it is reported as occluded, with no
    # unknown mass anywhere in the frame
    core = erode(hidden, 1)
    assert int(core.sum()) > 5
    assert float(np.max(f.p_visible[core])) < 0.05
    assert float(np.max(f.p_unknown)) == 0.0


def test_pixels_the_model_could_not_read_become_unknown():
    """The one thing that is genuinely unknown: directions with no answer."""
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib)
    dropout = np.ones_like(depth, dtype=bool)
    dropout[:, : calib.width // 3] = False  # the adapter marked a third invalid

    r = ga.estimate_visibility(
        prediction(depth, valid_mask=dropout), calib, xs, ys, drivable=DRIVABLE
    )
    assert r.status is ga.FrameStatus.OK
    f = r.visibility
    assert float(np.max(f.p_unknown)) == pytest.approx(1.0)
    unknown_cells = int((f.p_unknown > 0.5).sum())
    assert unknown_cells > 20
    # and the readable part of the frame is still confidently usable
    assert float(np.max(f.p_visible)) > 0.99


# ---------------------------------------------------------------------------
# 6. Depth-convention mismatches fail loudly
# ---------------------------------------------------------------------------
def test_unknown_convention_string_raises():
    with pytest.raises(ga.DepthConventionError, match="unknown depth convention"):
        ga.DepthPrediction(values=np.ones((4, 4)), convention="furlongs")


def test_inverse_depth_labelled_as_metric_raises():
    calib = make_calib()
    depth = render_depth(calib)
    disparity = 1.0 / depth
    with pytest.raises(ga.DepthConventionError, match="non-positive slope"):
        _fit_only(calib, disparity, ga.DepthConvention.METRIC_Z)


def test_metric_depth_labelled_as_inverse_raises():
    calib = make_calib()
    depth = render_depth(calib)
    with pytest.raises(ga.DepthConventionError, match="non-positive slope"):
        _fit_only(calib, depth, ga.DepthConvention.INVERSE_DEPTH)


def test_metric_model_in_the_wrong_units_raises():
    """Centimetres declared as metres: the fit works, the claim does not."""
    calib = make_calib()
    depth_cm = render_depth(calib) * 100.0
    with pytest.raises(ga.DepthConventionError, match="plausible band"):
        _fit_only(calib, depth_cm, ga.DepthConvention.METRIC_Z)


def test_range_labelled_as_optical_axis_depth_is_caught_by_the_scale_band():
    calib = make_calib()
    u, v = calib.pixel_grid(1)
    norms = calib.ray_norms(u, v).reshape(calib.height, calib.width)
    euclid = render_depth(calib) * norms
    cfg = ga.MethodConfig(fit=ga.FitConfig(metric_scale_band=(0.95, 1.05)))
    with pytest.raises(ga.DepthConventionError, match="plausible band"):
        _fit_only(calib, euclid, ga.DepthConvention.METRIC_Z, config=cfg)
    # declared correctly, the same numbers fit perfectly
    ok = _fit_only(calib, euclid, ga.DepthConvention.EUCLIDEAN_RANGE, config=cfg)
    assert ok.status is ga.FrameStatus.OK and ok.scale == pytest.approx(1.0, abs=1e-9)


def test_non_strict_mode_downgrades_a_mismatch_to_unknown_instead_of_raising():
    calib = make_calib()
    xs, ys = make_grid()
    disparity = 1.0 / render_depth(calib)
    cfg = ga.MethodConfig(fit=ga.FitConfig(strict_convention=False))
    result = ga.estimate_visibility(
        prediction(disparity, ga.DepthConvention.METRIC_Z), calib, xs, ys,
        drivable=DRIVABLE, config=cfg,
    )
    assert result.status is ga.FrameStatus.CONVENTION_MISMATCH
    assert np.all(result.visibility.p_unknown == 1.0)


def test_prediction_shape_must_match_the_camera():
    calib = make_calib()
    xs, ys = make_grid()
    with pytest.raises(ga.ContractViolation, match="does not match camera"):
        ga.estimate_visibility(
            prediction(np.ones((120, 160))), calib, xs, ys, drivable=DRIVABLE
        )


# ---------------------------------------------------------------------------
# Determinism, batching, and the output contract
# ---------------------------------------------------------------------------
def test_same_frame_twice_is_bitwise_identical():
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib, boxes=[BOX])
    runs = [
        ga.estimate_visibility(prediction(depth), calib, xs, ys, drivable=DRIVABLE)
        for _ in range(2)
    ]
    a, b = runs
    assert a.ground_fit.scale == b.ground_fit.scale
    assert a.ground_fit.shift == b.ground_fit.shift
    for name in ("p_visible", "p_occluded", "p_unknown", "height_map_m", "height_sigma_m"):
        assert np.array_equal(getattr(a.visibility, name), getattr(b.visibility, name)), name
    assert a.provenance["config_fingerprint"] == b.provenance["config_fingerprint"]


def test_four_cameras_run_independently_on_one_grid():
    xs, ys = make_grid()
    calibs = [make_calib(name) for name in ("A", "B", "C", "D")]
    preds = [prediction(render_depth(c, boxes=[BOX])) for c in calibs]
    results = ga.estimate_visibility_batch(preds, calibs, xs, ys, drivable=DRIVABLE)

    assert len(results) == 4
    assert [r.camera_id for r in results] == ["A", "B", "C", "D"]
    assert all(r.status is ga.FrameStatus.OK for r in results)
    assert all(r.visibility.p_visible.shape == (ys.size, xs.size) for r in results)
    # four viewpoints must not agree everywhere -- otherwise the geometry is wrong
    stack = np.stack([r.visibility.p_visible for r in results])
    assert float(np.max(np.ptp(stack, axis=0))) > 0.5


def test_probabilities_partition_every_cell():
    calib = make_calib()
    xs, ys = make_grid()
    r = ga.estimate_visibility(
        prediction(render_depth(calib, boxes=[BOX])), calib, xs, ys, drivable=DRIVABLE
    )
    f = r.visibility
    total = f.p_visible + f.p_occluded + f.p_unknown
    assert np.allclose(total, 1.0, atol=1e-9)
    for name in ("p_visible", "p_occluded", "p_unknown"):
        arr = getattr(f, name)
        assert arr.min() >= 0.0 and arr.max() <= 1.0, name


def test_output_contract_fields_are_all_present_and_shaped():
    calib = make_calib()
    xs, ys = make_grid()
    pred = prediction(
        render_depth(calib, boxes=[BOX]),
        model_name="unit-fixture", checkpoint="none", inference_time_s=0.0,
    )
    r = ga.estimate_visibility(
        pred, calib, xs, ys, drivable=DRIVABLE, scenario_id="s0", frame_id="f0", timestamp=1.5
    )
    grid_shape, img_shape = (ys.size, xs.size), (calib.height, calib.width)

    assert r.metric_depth.depth_m.shape == img_shape
    assert r.metric_depth.sigma_m.shape == img_shape
    assert r.metric_depth.valid.shape == img_shape
    assert np.all(r.metric_depth.sigma_m[r.metric_depth.valid] >= 0.0)

    fit = r.ground_fit
    assert fit.n_inlier > 0 and 0.0 <= fit.inlier_fraction <= 1.0
    assert np.isfinite([fit.residual_rms_m, fit.residual_p95_m, fit.anchor_depth_span_m]).all()

    f = r.visibility
    for name in ("p_visible", "p_occluded", "p_unknown", "unknown_mask", "in_fov",
                 "height_map_m", "height_sigma_m", "observed"):
        assert getattr(f, name).shape == grid_shape, name
    assert r.summary()["status"] == "ok"
    assert r.provenance["depth_convention"] == "metric_z"
    assert 0.0 <= r.provenance["unobserved_in_fov_fraction"] <= 1.0


def test_uncertainty_grows_where_the_model_is_extrapolating():
    """Anchors constrain a depth range; beyond it the fit says so by itself."""
    calib = make_calib()
    depth = render_depth(calib)
    fit = _fit_only(calib, depth, ga.DepthConvention.METRIC_Z)
    # a fit on exact data has zero scatter, so use a noisy one to exercise sigma
    rng = np.random.default_rng(3)
    noisy = depth + rng.normal(0.0, 0.05, size=depth.shape)
    fit = _fit_only(calib, noisy, ga.DepthConvention.METRIC_Z)
    assert fit.status is ga.FrameStatus.OK

    probe = np.array([[8.0, 200.0]])
    sig = ga.predicted_depth_sigma(fit, probe, fit.apply(probe))
    assert sig[0, 1] > sig[0, 0], "extrapolated depth must carry more uncertainty"
    assert sig[0, 0] < 0.5


def test_adapter_uncertainty_is_carried_through():
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib)
    quiet = ga.estimate_visibility(
        prediction(depth, uncertainty=np.full_like(depth, 0.01)), calib, xs, ys,
        drivable=DRIVABLE,
    )
    noisy = ga.estimate_visibility(
        prediction(depth, uncertainty=np.full_like(depth, 0.50)), calib, xs, ys,
        drivable=DRIVABLE,
    )
    assert float(np.nanmedian(noisy.metric_depth.sigma_m)) > float(
        np.nanmedian(quiet.metric_depth.sigma_m)
    )


def test_native_confidence_is_not_mistaken_for_metric_uncertainty():
    """A larger-is-better score must not inflate the depth covariance."""
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib)
    reference = ga.estimate_visibility(
        prediction(depth), calib, xs, ys, drivable=DRIVABLE
    )
    confidence = ga.estimate_visibility(
        prediction(
            depth,
            uncertainty=np.full_like(depth, 1000.0),
            uncertainty_kind="native_confidence",
        ),
        calib,
        xs,
        ys,
        drivable=DRIVABLE,
    )
    assert np.allclose(
        reference.metric_depth.sigma_m,
        confidence.metric_depth.sigma_m,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Optional temporal Bayesian anchoring
# ---------------------------------------------------------------------------
def _temporal_fit(scale: float, shift: float, seed: int) -> ga.GroundFit:
    rng = np.random.default_rng(seed)
    pred = np.linspace(0.0, 10.0, 600)
    truth = scale * pred + shift + rng.normal(0.0, 0.08, pred.size)
    return ga.fit_ground_affine(
        pred,
        truth,
        ga.DepthConvention.RELATIVE_DEPTH,
        config=ga.FitConfig(min_anchor_pixels=20),
    )


def test_temporal_bayesian_update_reduces_parameter_uncertainty():
    temporal = ga.TemporalGroundAnchorFilter()
    first = _temporal_fit(2.0, 1.0, 1)
    second = _temporal_fit(2.002, 1.004, 2)

    posterior1, p1 = temporal.update(
        first, camera_id="A", model_name="relative", timestamp_s=0.0
    )
    posterior2, p2 = temporal.update(
        second, camera_id="A", model_name="relative", timestamp_s=10.0
    )

    assert p1["mode"] == "initialised"
    assert p2["mode"] == "updated" and p2["accepted"]
    assert np.trace(posterior2.parameter_covariance) < np.trace(
        second.parameter_covariance
        + np.diag([second.scale * 0.005, 0.01]) ** 2
    )
    # With correlated scale/shift covariance a joint Gaussian posterior need
    # not be component-wise bracketed by both measurements.  It should remain
    # close to their common scale while becoming sharper.
    assert abs(posterior2.scale - 0.5 * (first.scale + second.scale)) < 1e-3


def test_temporal_filter_rejects_a_jump_and_expires_stale_evidence():
    temporal = ga.TemporalGroundAnchorFilter(
        ga.TemporalAnchorConfig(max_stale_s=30.0)
    )
    first = _temporal_fit(2.0, 1.0, 3)
    temporal.update(first, camera_id="A", model_name="relative", timestamp_s=0.0)

    jump = _temporal_fit(4.0, -2.0, 4)
    reused, jump_info = temporal.update(
        jump, camera_id="A", model_name="relative", timestamp_s=10.0
    )
    assert jump_info["mode"] == "innovation_rejected_prior"
    assert reused.status is ga.FrameStatus.OK
    assert abs(reused.scale - first.scale) < 0.05

    refused = replace(first, status=ga.FrameStatus.INSUFFICIENT_FLOOR_PIXELS)
    recent, recent_info = temporal.update(
        refused, camera_id="A", model_name="relative", timestamp_s=20.0
    )
    expired, expired_info = temporal.update(
        refused, camera_id="A", model_name="relative", timestamp_s=40.1
    )
    assert recent.status is ga.FrameStatus.OK and recent_info["mode"] == "stale_prior"
    assert expired.status is ga.FrameStatus.INSUFFICIENT_FLOOR_PIXELS
    assert expired_info["mode"] == "stale_limit_exceeded"


def test_temporal_filter_keeps_camera_states_separate():
    temporal = ga.TemporalGroundAnchorFilter()
    temporal.update(_temporal_fit(2.0, 1.0, 5), camera_id="A", model_name="m", timestamp_s=0)
    temporal.update(_temporal_fit(3.0, -1.0, 6), camera_id="B", model_name="m", timestamp_s=0)
    snapshot = temporal.snapshot()
    assert set(snapshot) == {"A/m/relative_depth", "B/m/relative_depth"}
    assert abs(snapshot["A/m/relative_depth"]["mean"][0] - 2.0) < 0.02
    assert abs(snapshot["B/m/relative_depth"]["mean"][0] - 3.0) < 0.02


def test_enhanced_and_temporal_modes_are_explicit_in_pipeline_provenance():
    calib = make_calib()
    xs, ys = make_grid()
    depth = render_depth(calib, boxes=[BOX])
    cfg = ga.MethodConfig(anchors=ga.AnchorConfig(quality_filter=True))
    temporal = ga.TemporalGroundAnchorFilter()
    values1 = (depth - 1.0) / 2.0
    values2 = (depth - 1.002) / 2.002

    first = ga.estimate_visibility(
        prediction(values1, ga.DepthConvention.RELATIVE_DEPTH, model_name="m"),
        calib,
        xs,
        ys,
        drivable=DRIVABLE,
        config=cfg,
        temporal_filter=temporal,
        timestamp=0.0,
    )
    second = ga.estimate_visibility(
        prediction(values2, ga.DepthConvention.RELATIVE_DEPTH, model_name="m"),
        calib,
        xs,
        ys,
        drivable=DRIVABLE,
        config=cfg,
        temporal_filter=temporal,
        timestamp=10.0,
    )

    assert first.status is ga.FrameStatus.OK and second.status is ga.FrameStatus.OK
    assert first.provenance["enhanced_anchor_selection"] is True
    assert first.provenance["temporal_anchor"]["mode"] == "initialised"
    assert second.provenance["temporal_anchor"]["mode"] == "updated"


# ---------------------------------------------------------------------------
# The oracle boundary
# ---------------------------------------------------------------------------
def test_method_visible_record_drops_the_oracle_fields():
    record = {
        "scenario_id": "aisle_box_01", "timestamp": 12.5, "camera_id": "A",
        "rgb_path": "rgb/A_000125.png", "oracle_depth_path": "depth/A_000125.npy",
        "camera_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5,
                              "width": 2, "height": 2},
        "camera_extrinsics": {"camera_position": [0, 0, 1], "look_at": [1, 0, 0]},
        "obstacle_state": {"pallet": [2.4, 2.4, 0.0]},
        "oracle_visibility_grid": "vis/A_000125.npy",
    }
    visible = ga.method_visible_record(record)
    assert set(visible) - {"_withheld"} <= ga.METHOD_VISIBLE_KEYS
    assert visible["_withheld"] == [
        "obstacle_state", "oracle_depth_path", "oracle_visibility_grid"
    ]
    assert ga.ORACLE_ONLY_KEYS.isdisjoint(ga.METHOD_VISIBLE_KEYS)
    with pytest.raises(ga.OracleAccessError):
        ga.assert_no_oracle_access(["rgb_path", "oracle_depth_path"])


def test_package_source_never_mentions_an_oracle_field():
    """Structural guard: the method cannot read what it does not name."""
    pkg = _ROOT / "experiments" / "mono_depth_visibility" / "ground_anchoring"
    forbidden = ("oracle_depth", "oracle_visibility_grid", "ground_truth", "gt_pose")
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        if path.name == "io_contract.py":
            continue  # the one place that names them, in order to refuse them
        text = path.read_text()
        offenders += [f"{path.name}:{tok}" for tok in forbidden if tok in text]
    assert offenders == []


def test_camera_from_record_ignores_oracle_keys_and_matches_the_repo_model():
    cam = ObliqueCameraModel(img_width=320, img_height=240, fov_h_rad=1.2, **CAMERAS["A"])
    record = {
        "scenario_id": "s", "timestamp": 0.0, "camera_id": "A",
        "camera_intrinsics": {"K": cam.K.tolist(), "width": 320, "height": 240},
        "camera_extrinsics": {
            "camera_position": list(CAMERAS["A"]["cam_pos"]),
            "look_at": list(CAMERAS["A"]["look_at"]),
        },
        "oracle_depth_path": "nope.npy",
        "obstacle_state": {"box": [1, 1, 0]},
    }
    calib = ga.camera_from_record(record)
    assert np.allclose(calib.R, cam.R, atol=1e-12)
    assert np.allclose(calib.K, cam.K, atol=1e-12)
    assert calib.camera_id == "A"


@pytest.mark.parametrize(
    "extrinsics",
    [
        {"R_world_to_cam": np.eye(3).tolist(), "t_world_to_cam": [0.0, 0.0, -6.0]},
        {"T_cam_world": np.block(
            [[np.eye(3), np.array([[0.0], [0.0], [-6.0]])], [np.zeros((1, 3)), np.ones((1, 1))]]
        ).tolist()},
    ],
)
def test_extrinsics_forms_agree(extrinsics):
    intr = {"fx": 200.0, "fy": 200.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
    calib = ga.calibration_from_parts(intr, extrinsics)
    assert np.allclose(calib.cam_pos, [0.0, 0.0, 6.0])


def test_unrecognised_calibration_blocks_fail_loudly():
    with pytest.raises(ga.ContractViolation, match="unrecognised intrinsics"):
        ga.calibration_from_parts({"focal": 200, "width": 32, "height": 24}, {})
    with pytest.raises(ga.ContractViolation, match="unrecognised extrinsics"):
        ga.calibration_from_parts(
            {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "width": 32, "height": 24}, {"pose": "??"}
        )


# ---------------------------------------------------------------------------
# Persistence + reuse parity
# ---------------------------------------------------------------------------
def test_saved_result_round_trips_the_contract(tmp_path):
    calib = make_calib()
    xs, ys = make_grid()
    r = ga.estimate_visibility(
        prediction(render_depth(calib, boxes=[BOX])), calib, xs, ys, drivable=DRIVABLE,
        scenario_id="aisle_box_01", frame_id="000125",
    )
    npz_path = ga.save_result(r, tmp_path)
    data = np.load(npz_path)
    for key in ("p_visible", "p_occluded", "p_unknown", "unknown_mask", "in_fov",
                "height_map_m", "height_sigma_m", "observed", "depth_m", "depth_valid"):
        assert key in data.files, key
    summary = json.loads((tmp_path / f"{npz_path.stem}.json").read_text())
    assert summary["scenario_id"] == "aisle_box_01"
    assert summary["ground_fit"]["status"] == "ok"


def test_prediction_npz_must_declare_its_convention(tmp_path):
    path = tmp_path / "pred.npz"
    np.savez(path, values=np.ones((4, 4)))
    with pytest.raises(ga.ContractViolation, match="does not declare a depth convention"):
        ga.load_prediction(path)

    np.savez(path, values=np.ones((4, 4)))
    path.with_suffix(".json").write_text(json.dumps({"convention": "metric_z",
                                                     "model_name": "m"}))
    pred = ga.load_prediction(path)
    assert pred.convention is ga.DepthConvention.METRIC_Z and pred.model_name == "m"


def test_loads_the_depth_adapter_schema(tmp_path):
    """The peer adapter's own layout: depth/valid npz + metadata sidecar."""
    depth = np.full((6, 8), 3.0, dtype=np.float32)
    np.savez_compressed(tmp_path / "frame7__dav2_large.npz", depth=depth,
                        valid=np.ones_like(depth, dtype=bool),
                        uncertainty=np.full_like(depth, 0.1),
                        native_confidence=np.full_like(depth, 0.8))
    (tmp_path / "frame7__dav2_large.json").write_text(json.dumps({
        "schema_version": 1,
        "image_id": "frame7",
        "convention": "euclidean_range",
        "npz_file": "frame7__dav2_large.npz",
        "model": {"model_name": "dav2_large", "checkpoint": "hf://depth-anything/x"},
        "timing": {"preprocess_s": 0.01, "forward_s": 0.2, "postprocess_s": 0.01,
                   "total_s": 0.22},
        "uncertainty_kind": "flip_consistency",
    }))

    from_json = ga.load_prediction(tmp_path / "frame7__dav2_large.json")
    from_npz = ga.load_prediction(tmp_path / "frame7__dav2_large.npz")
    for pred in (from_json, from_npz):
        assert pred.convention is ga.DepthConvention.EUCLIDEAN_RANGE
        assert pred.model_name == "dav2_large"
        assert pred.checkpoint.startswith("hf://")
        assert pred.inference_time_s == pytest.approx(0.22)
        assert pred.frame_id == "frame7"
        assert pred.uncertainty is not None and pred.valid_mask.all()
        assert pred.uncertainty_kind == "flip_consistency"
        assert pred.native_confidence is not None

    index = ga.prediction_index(tmp_path)
    assert index["frame7"] == [tmp_path / "frame7__dav2_large.json"]


def test_convention_aliases_are_accepted_but_nonsense_is_not():
    assert ga.DepthConvention.parse("metric_range") is ga.DepthConvention.EUCLIDEAN_RANGE
    assert ga.DepthConvention.parse("disparity") is ga.DepthConvention.INVERSE_DEPTH
    assert ga.DepthConvention.parse(" Metric_Z ") is ga.DepthConvention.METRIC_Z
    with pytest.raises(ga.DepthConventionError):
        ga.DepthConvention.parse("depth_ish")


def test_the_generators_calibration_spelling_is_accepted():
    """The dynamic-world generator writes img_width/img_height and cam_pos."""
    calib = ga.calibration_from_parts(
        {"img_width": 1280, "img_height": 720, "fov_h_rad": 1.5708,
         "fx": 640.0, "fy": 640.0, "cx": 640.0, "cy": 360.0},
        {"sdf_pose_xyz_rpy": [-3, -3, 6, 0, 0.6, 0.8], "cam_pos": [-3.0, -3.0, 6.0],
         "look_at": [1.5, 1.5, 0.0]},
        camera_id="camera_A",
    )
    assert (calib.width, calib.height) == (1280, 720)
    assert np.allclose(calib.cam_pos, [-3.0, -3.0, 6.0])
    cam = ObliqueCameraModel(cam_pos=(-3.0, -3.0, 6.0), look_at=(1.5, 1.5, 0.0),
                             img_width=1280, img_height=720, fov_h_rad=1.5708)
    assert np.allclose(calib.R, cam.R) and np.allclose(calib.K, cam.K)


def test_height_rasterisation_matches_the_existing_repo_helper():
    """Reuse check against ``geometry_visibility.height_map_from_points``.

    They must agree on cell-centred points; this module additionally carries a
    per-cell sigma, which is why it is not simply a call into that helper.
    """
    # Load the file directly under a private module name. `geometry_visibility`
    # exists both as a module and as a package on different sys.path entries, so
    # a plain import resolves to whichever one an earlier test happened to cache
    # -- this test passed alone and failed in the full suite until it stopped
    # going through sys.path at all.
    import importlib.util

    gv_file = _ROOT / "scripts" / "geometry_visibility" / "geometry_visibility.py"
    spec = importlib.util.spec_from_file_location("_gv_for_parity_check", gv_file)
    assert spec is not None and spec.loader is not None
    gv = importlib.util.module_from_spec(spec)
    # registered while executing: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for an unregistered module
    sys.modules[spec.name] = gv
    try:
        spec.loader.exec_module(gv)
    finally:
        sys.modules.pop(spec.name, None)

    xs = np.arange(0.0, 5.0, 0.25)
    ys = np.arange(0.0, 5.0, 0.25)
    rng = np.random.default_rng(5)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack(
        [gx.ravel(), gy.ravel(), rng.uniform(0.0, 2.0, size=gx.size)]
    )
    mine = ga.rasterize_heights(pts, np.zeros(pts.shape[0]), xs, ys)
    theirs = gv.height_map_from_points(pts, xs, ys)
    assert np.allclose(mine.h_max, theirs["h_max"])
    assert np.array_equal(mine.observed, theirs["observed"])
