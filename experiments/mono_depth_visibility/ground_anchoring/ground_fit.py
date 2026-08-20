"""Robustly fitting a monocular prediction onto the known floor depths.

One affine map per frame, fitted only on floor anchors, in the space the
model's convention demands. Two things matter as much as the fit itself:

**It must be robust.** A box standing in the aisle occludes floor the
deployment map calls drivable, so a minority of anchors report the box, not the
floor. Least squares would split the difference and quietly shrink the very
obstacle we are looking for. RANSAC on a range-scaled tolerance rejects them,
and the count of rejected-short anchors is reported because it is evidence that
something is standing there.

**It must be allowed to refuse.** Too few anchors, anchors bunched at one
range, a badly conditioned design, too many outliers, or residuals that stay
large after fitting all mean the frame cannot be anchored. The method then
reports a status and emits an all-unknown field. It never returns a
best-effort scale it does not believe.
"""

from __future__ import annotations

import numpy as np

from .contracts import (
    DepthConvention,
    DepthConventionError,
    FitConfig,
    FrameStatus,
    GroundFit,
)
from .conventions import (
    assert_metric_scale_plausible,
    assert_slope_consistent,
    depth_from_fit_space,
    fit_space_for,
    sigma_to_depth,
    target_in_fit_space,
)


def _tolerance_m(z_true: np.ndarray, cfg: FitConfig) -> np.ndarray:
    """Inlier tolerance in metres, scaled with range.

    A fixed tolerance is the wrong shape: 25 cm is generous at 3 m and strict
    at 30 m, so a constant threshold silently re-weights the fit toward
    whichever range happens to dominate the image.
    """
    return cfg.inlier_rel_tol * np.abs(z_true) + cfg.inlier_abs_tol_m


def _residual_m(
    slope: float, shift: float, pred: np.ndarray, z_true: np.ndarray, convention: DepthConvention
) -> np.ndarray:
    """Signed depth error in metres for a candidate affine (NaN where invalid)."""
    z_hat = depth_from_fit_space(slope * pred + shift, convention)
    return z_hat - z_true


def _least_squares(
    pred: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, float]:
    """Ordinary LS of ``y_true ~ a*pred + b``; also returns ``(AᵀA)^-1`` and cond.

    The fit is done on a standardised predictor and then mapped back. That
    matters for the reported condition number: on the raw predictor it is
    dominated by the *units* -- a model emitting millimetres looks catastrophi-
    cally ill-conditioned while an identical model emitting metres looks fine.
    Standardised, the condition number measures the only thing worth gating on,
    which is whether the anchors actually spread the predictor out.
    """
    p = np.asarray(pred, dtype=float)
    if weights is None:
        # Preserve the original arithmetic path for frozen studies.
        mu = float(np.mean(p))
        sd = float(np.std(p))
        weight = None
    else:
        weight = np.asarray(weights, dtype=float).ravel()
        if weight.shape != p.shape:
            raise ValueError(f"weights shape {weight.shape} != predictor shape {p.shape}")
        weight = np.clip(weight, 1e-12, np.inf)
        weight = weight / float(np.mean(weight))
        mu = float(np.average(p, weights=weight))
        sd = float(np.sqrt(np.average((p - mu) ** 2, weights=weight)))
    if not np.isfinite(sd) or sd <= 0.0:
        return float("nan"), float("nan"), np.full((2, 2), np.inf), float("inf")

    A = np.column_stack([(p - mu) / sd, np.ones_like(p)])
    if weight is None:
        Aw, yw = A, y_true
    else:
        sqrt_weight = np.sqrt(weight)
        Aw = A * sqrt_weight[:, None]
        yw = y_true * sqrt_weight
    ata = Aw.T @ Aw
    cond = float(np.linalg.cond(ata)) if np.isfinite(ata).all() else np.inf
    if not np.isfinite(cond) or cond > 1.0 / np.finfo(float).eps:
        return float("nan"), float("nan"), np.full((2, 2), np.inf), cond
    coeffs, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    a_std, b_std = float(coeffs[0]), float(coeffs[1])

    # undo the standardisation: [p, 1] = [(p-mu)/sd, 1] @ M, M = [[sd, 0], [mu, 1]]
    m_inv = np.array([[1.0 / sd, 0.0], [-mu / sd, 1.0]])
    ata_inv = m_inv @ np.linalg.inv(ata) @ m_inv.T
    return a_std / sd, b_std - a_std * mu / sd, ata_inv, cond


def _ransac(
    pred: np.ndarray, y_true: np.ndarray, z_true: np.ndarray,
    convention: DepthConvention, cfg: FitConfig, weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Deterministic RANSAC seed for the affine; returns ``(a, b, inliers)``."""
    rng = np.random.default_rng(cfg.ransac_seed)
    n = pred.size
    weight = (
        np.ones(n, dtype=float)
        if weights is None
        else np.clip(np.asarray(weights, float), 0.0, np.inf)
    )
    sampling = (
        weight / weight.sum()
        if weights is not None and float(weight.sum()) > 0.0
        else None
    )
    tol = _tolerance_m(z_true, cfg)
    spread = float(np.ptp(pred))
    min_sep = max(1e-9, 1e-3 * spread)

    best_a, best_b, best_score, best_count = 1.0, 0.0, -1.0, -1
    best_inliers = np.zeros(n, dtype=bool)
    for _ in range(max(1, int(cfg.ransac_iters))):
        if sampling is None:
            i, j = rng.integers(0, n, 2)
        else:
            i, j = rng.choice(n, size=2, replace=True, p=sampling)
        dp = pred[i] - pred[j]
        if abs(dp) < min_sep:
            continue
        a = float((y_true[i] - y_true[j]) / dp)
        b = float(y_true[i] - a * pred[i])
        resid = _residual_m(a, b, pred, z_true, convention)
        inliers = np.isfinite(resid) & (np.abs(resid) <= tol)
        count = int(inliers.sum())
        score = float(weight[inliers].sum())
        if score > best_score or (score == best_score and count > best_count):
            best_a, best_b, best_score, best_count, best_inliers = a, b, score, count, inliers
    if best_count <= 0:  # every hypothesis degenerate -> fall back to plain LS
        a, b, _, _ = _least_squares(pred, y_true, weights)
        if np.isfinite(a):
            resid = _residual_m(a, b, pred, z_true, convention)
            best_a, best_b = a, b
            best_inliers = np.isfinite(resid) & (np.abs(resid) <= tol)
    return best_a, best_b, best_inliers


def fit_ground_affine(
    pred_anchor: np.ndarray,
    depth_anchor_m: np.ndarray,
    convention: DepthConvention | str,
    *,
    config: FitConfig | None = None,
    anchor_depth_span_m: float | None = None,
    weights: np.ndarray | None = None,
    notes: str = "",
) -> GroundFit:
    """Fit model output -> metres on the floor anchors, with validity gating.

    ``pred_anchor`` must already be in the model's declared convention with
    ``EUCLIDEAN_RANGE`` converted to optical-axis depth (see
    :func:`conventions.to_optical_axis`). ``depth_anchor_m`` is the analytic
    floor depth for the same pixels.

    Raises :class:`DepthConventionError` when the data contradicts the declared
    convention and ``config.strict_convention`` is set (the default).
    """
    cfg = config or FitConfig()
    convention = DepthConvention.parse(convention)
    space = fit_space_for(convention)

    pred = np.asarray(pred_anchor, dtype=float).ravel()
    z_true = np.asarray(depth_anchor_m, dtype=float).ravel()
    if pred.size != z_true.size:
        raise ValueError(f"anchor arrays disagree: {pred.size} predictions vs {z_true.size} depths")

    raw_weights = (
        np.ones_like(pred)
        if weights is None
        else np.asarray(weights, dtype=float).ravel()
    )
    if raw_weights.size != pred.size:
        raise ValueError(f"weights contain {raw_weights.size} values for {pred.size} anchors")
    finite = (
        np.isfinite(pred)
        & np.isfinite(z_true)
        & (z_true > 0.0)
        & np.isfinite(raw_weights)
        & (raw_weights > 0.0)
    )
    pred, z_true = pred[finite], z_true[finite]
    fit_weights = raw_weights[finite]
    fit_weights = fit_weights / float(np.mean(fit_weights)) if fit_weights.size else fit_weights
    n_anchor = int(pred.size)

    span = (
        float(anchor_depth_span_m)
        if anchor_depth_span_m is not None
        else (float(np.ptp(np.percentile(z_true, [5.0, 95.0]))) if n_anchor >= 2 else 0.0)
    )

    def refused(status: FrameStatus, why: str) -> GroundFit:
        return GroundFit(
            scale=float("nan"), shift=float("nan"), fit_space=space, convention=convention,
            status=status, n_anchor=n_anchor, n_inlier=0, inlier_fraction=0.0,
            residual_rms_m=float("nan"), residual_p95_m=float("nan"),
            anchor_depth_span_m=span, condition_number=float("nan"), sigma_fit=float("nan"),
            ata_inv=np.full((2, 2), np.inf), notes="; ".join(x for x in (notes, why) if x),
        )

    if n_anchor < max(2, cfg.min_anchor_pixels):
        return refused(
            FrameStatus.INSUFFICIENT_FLOOR_PIXELS,
            f"{n_anchor} usable floor anchors < required {cfg.min_anchor_pixels}",
        )
    if span < cfg.min_depth_span_m:
        return refused(
            FrameStatus.INSUFFICIENT_DEPTH_SPAN,
            f"anchor depth span {span:.2f} m < required {cfg.min_depth_span_m:.2f} m; "
            "the scale is not identifiable from anchors at one range",
        )

    y_true = target_in_fit_space(z_true, convention)
    ok = np.isfinite(y_true)
    pred, z_true, y_true, fit_weights = pred[ok], z_true[ok], y_true[ok], fit_weights[ok]

    a, b, inliers = _ransac(
        pred,
        y_true,
        z_true,
        convention,
        cfg,
        None if weights is None else fit_weights,
    )
    if inliers.sum() >= 2:  # refit on the consensus set, then re-score once
        a_ls, b_ls, ata_inv, cond = _least_squares(
            pred[inliers], y_true[inliers], None if weights is None else fit_weights[inliers]
        )
        if np.isfinite(a_ls):
            a, b = a_ls, b_ls
            resid = _residual_m(a, b, pred, z_true, convention)
            inliers = np.isfinite(resid) & (np.abs(resid) <= _tolerance_m(z_true, cfg))
    if inliers.sum() >= 2:
        a_ls, b_ls, ata_inv, cond = _least_squares(
            pred[inliers], y_true[inliers], None if weights is None else fit_weights[inliers]
        )
        if np.isfinite(a_ls):
            a, b = a_ls, b_ls
    else:
        ata_inv, cond = np.full((2, 2), np.inf), float("inf")

    n_inlier = int(inliers.sum())
    inlier_fraction = n_inlier / n_anchor if n_anchor else 0.0

    # --- convention self-check: in the declared space the slope must be positive
    corr = (
        float(np.corrcoef(pred, y_true)[0, 1])
        if pred.size >= 2 and np.ptp(pred) > 0 and np.ptp(y_true) > 0
        else float("nan")
    )
    try:
        assert_slope_consistent(a, convention, correlation=corr)
    except DepthConventionError:
        if cfg.strict_convention:
            raise
        return refused(FrameStatus.CONVENTION_MISMATCH, f"slope {a:+.4g} in {space} space")

    if not np.isfinite(cond) or cond > cfg.max_condition_number:
        return refused(
            FrameStatus.ILL_CONDITIONED,
            f"design condition number {cond:.3g} > {cfg.max_condition_number:.3g}",
        )

    resid_all = _residual_m(a, b, pred, z_true, convention)
    resid_in = resid_all[inliers]
    rms = float(np.sqrt(np.mean(np.square(resid_in)))) if n_inlier else float("nan")
    p95 = float(np.percentile(np.abs(resid_in), 95.0)) if n_inlier else float("nan")
    inlier_weights = fit_weights[inliers]
    y_resid = y_true[inliers] - (a * pred[inliers] + b)
    if weights is None:
        dof = max(1, n_inlier - 2)
        sigma_fit = (
            float(np.sqrt(np.sum(np.square(y_resid)) / dof))
            if n_inlier >= 2 else float("nan")
        )
    else:
        dof = max(1.0, float(inlier_weights.sum()) - 2.0)
        sigma_fit = (
            float(np.sqrt(np.sum(inlier_weights * np.square(y_resid)) / dof))
            if n_inlier >= 2 else float("nan")
        )

    tol = _tolerance_m(z_true, cfg)
    n_short = int(np.count_nonzero(np.isfinite(resid_all) & (resid_all < -tol)))
    n_beyond = int(np.count_nonzero(np.isfinite(resid_all) & (resid_all > tol)))

    fit = GroundFit(
        scale=float(a), shift=float(b), fit_space=space, convention=convention,
        status=FrameStatus.OK, n_anchor=n_anchor, n_inlier=n_inlier,
        inlier_fraction=float(inlier_fraction), residual_rms_m=rms, residual_p95_m=p95,
        anchor_depth_span_m=span, condition_number=float(cond), sigma_fit=sigma_fit,
        ata_inv=np.asarray(ata_inv, dtype=float), n_shorter_than_floor=n_short,
        n_beyond_floor=n_beyond, notes=notes,
    )

    if inlier_fraction < cfg.min_inlier_fraction:
        return refused(
            FrameStatus.LOW_INLIER_FRACTION,
            f"only {100 * inlier_fraction:.0f}% of anchors agree with the fitted plane "
            f"(need {100 * cfg.min_inlier_fraction:.0f}%)",
        )
    if not np.isfinite(rms) or rms > cfg.max_residual_rms_m:
        return refused(
            FrameStatus.HIGH_RESIDUAL,
            f"inlier residual RMS {rms:.3f} m > {cfg.max_residual_rms_m:.3f} m",
        )
    try:
        assert_metric_scale_plausible(a, convention, cfg.metric_scale_band)
    except DepthConventionError:
        if cfg.strict_convention:
            raise
        return refused(FrameStatus.NON_PHYSICAL_SCALE, f"metric model needed scale {a:.3f}")

    return fit


def predicted_depth_sigma(
    fit: GroundFit,
    pred: np.ndarray,
    depth_m: np.ndarray,
    model_sigma: np.ndarray | None = None,
) -> np.ndarray:
    """1-sigma on the corrected depth of each pixel, in metres.

    Three contributions, all in fit space before being pushed through to
    metres:

    * ``sigma_fit`` -- scatter of the floor anchors about the fitted line, i.e.
      how well this model tracks a surface it was anchored on;
    * ``[p,1] Cov [p,1]ᵀ`` -- uncertainty of the two fitted parameters. This is
      what makes extrapolation honest: a pixel whose prediction lies far
      outside the anchored range inherits a large sigma automatically, with no
      extra tuning knob;
    * the adapter's own per-pixel uncertainty, if it supplied one, scaled by
      the fitted slope.
    """
    p = np.asarray(pred, dtype=float)
    cov = fit.parameter_covariance
    var_y = fit.sigma_fit ** 2 + (
        np.square(p) * cov[0, 0] + 2.0 * p * cov[0, 1] + cov[1, 1]
    )
    if model_sigma is not None:
        var_y = var_y + np.square(fit.scale * np.asarray(model_sigma, dtype=float))
    with np.errstate(invalid="ignore"):
        sigma_y = np.sqrt(np.maximum(var_y, 0.0))
    return sigma_to_depth(sigma_y, depth_m, fit.convention)
