"""Depth-convention handling: get into a common fit space, and fail loudly.

Every monocular model here is corrected by one affine fit, but *which space*
that affine lives in is set by the model's declared convention:

    metric_z / euclidean_range / relative_depth  ->  z      = a * p + b
    inverse_depth                              ->   1 / z  = a * p + b

Getting this wrong does not produce a slightly worse fit, it produces a
confidently wrong depth map, so the declared convention is also *checked*
against the data: in the correct space the slope ``a`` must be positive,
because the prediction and the target both increase with the same physical
quantity. A negative slope means the numbers are the other kind, and that
raises :class:`DepthConventionError` rather than being fitted around.
"""

from __future__ import annotations

import numpy as np

from .contracts import CameraCalibration, DepthConvention, DepthConventionError

#: name of the space each convention's affine fit is performed in
FIT_SPACE = {
    DepthConvention.METRIC_Z: "depth",
    DepthConvention.EUCLIDEAN_RANGE: "depth",
    DepthConvention.RELATIVE_DEPTH: "depth",
    DepthConvention.INVERSE_DEPTH: "inverse_depth",
}


def fit_space_for(convention: DepthConvention) -> str:
    return FIT_SPACE[DepthConvention.parse(convention)]


def to_optical_axis(
    values: np.ndarray,
    convention: DepthConvention,
    calib: CameraCalibration,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Normalise a prediction so a single affine fit is enough.

    Only ``EUCLIDEAN_RANGE`` needs work: Euclidean range ``r`` relates to
    optical-axis depth by ``z = r / ||K^-1 [u,v,1]||``, a purely geometric
    per-pixel factor. Every other convention passes through untouched -- the
    unknown scale and shift are the fit's job, not this function's.

    ``u``/``v`` are the pixel coordinates of ``values``; omit them when
    ``values`` is a full image and they will be generated.
    """
    convention = DepthConvention.parse(convention)
    values = np.asarray(values, dtype=float)
    if convention is not DepthConvention.EUCLIDEAN_RANGE:
        return values
    if u is None or v is None:
        if values.ndim != 2:
            raise ValueError("full-image conversion needs a 2-D array or explicit u, v")
        vv, uu = np.mgrid[0 : values.shape[0], 0 : values.shape[1]]
        u_px, v_px = uu.ravel().astype(float), vv.ravel().astype(float)
    else:
        u_px, v_px = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    norms = calib.ray_norms(u_px, v_px).reshape(values.shape)
    return values / norms


def target_in_fit_space(depth_m: np.ndarray, convention: DepthConvention) -> np.ndarray:
    """Map known metric depth into the space the affine is fitted in."""
    depth_m = np.asarray(depth_m, dtype=float)
    if DepthConvention.parse(convention).is_inverse:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(depth_m > 1e-9, 1.0 / depth_m, np.nan)
    return depth_m


def depth_from_fit_space(y: np.ndarray, convention: DepthConvention) -> np.ndarray:
    """Inverse of :func:`target_in_fit_space`; non-positive ``y`` -> NaN."""
    y = np.asarray(y, dtype=float)
    if DepthConvention.parse(convention).is_inverse:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y > 1e-9, 1.0 / y, np.nan)
    return y


def sigma_to_depth(
    sigma_y: np.ndarray, depth_m: np.ndarray, convention: DepthConvention
) -> np.ndarray:
    """Propagate a 1-sigma from fit space to metres.

    For inverse space ``y = 1/z`` so ``dz/dy = -z^2`` and the depth sigma grows
    quadratically with range -- which is the honest statement that a disparity
    model knows very little about far surfaces.
    """
    sigma_y = np.asarray(sigma_y, dtype=float)
    if DepthConvention.parse(convention).is_inverse:
        return sigma_y * np.square(np.asarray(depth_m, dtype=float))
    return sigma_y


def assert_slope_consistent(
    slope: float, convention: DepthConvention, *, correlation: float | None = None
) -> None:
    """Raise when the data says the declared convention is the wrong one.

    In the declared fit space the relationship is monotonically increasing by
    construction, so a non-positive slope is a labelling error -- most often
    an inverse-depth map handed over as ``metric_z`` (or the reverse).
    """
    convention = DepthConvention.parse(convention)
    if np.isfinite(slope) and slope > 0.0:
        return
    other = (
        DepthConvention.INVERSE_DEPTH.value
        if not convention.is_inverse
        else f"{DepthConvention.METRIC_Z.value}/{DepthConvention.RELATIVE_DEPTH.value}"
    )
    extra = "" if correlation is None else f" (anchor correlation {correlation:+.3f})"
    raise DepthConventionError(
        f"prediction declared as {convention.value!r} fits with a non-positive slope "
        f"{slope:+.4g} in {FIT_SPACE[convention]!r} space{extra}: the numbers move "
        f"opposite to depth, which is what {other!r} looks like. Refusing to fit -- "
        f"fix the declared convention in the depth adapter."
    )


def assert_metric_scale_plausible(
    slope: float, convention: DepthConvention, band: tuple[float, float]
) -> None:
    """A model that claims metres must not need a 5x rescale to become metres.

    Only applied to metric conventions; relative and inverse models are
    *expected* to need an arbitrary scale, so no band is meaningful for them.
    """
    convention = DepthConvention.parse(convention)
    if not convention.is_metric:
        return
    lo, hi = float(band[0]), float(band[1])
    if not (lo <= slope <= hi):
        raise DepthConventionError(
            f"prediction declared as {convention.value!r} needed scale {slope:.3f} to reach "
            f"metres, outside the plausible band [{lo:g}, {hi:g}]. Either the model is not "
            f"metric, the units are not metres, or the extrinsics do not match this image."
        )
