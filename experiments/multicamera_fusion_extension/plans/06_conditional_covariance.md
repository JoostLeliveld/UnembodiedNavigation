# Plan 06 — Conditional measurement covariance R_cond (§9, E2)

## Purpose
Anisotropic per-camera covariance of usable observations — the quantity the
filter actually consumes. Kept strictly separate from reliability (the GP is
never re-labelled as covariance — thesis architecture rule).

## Tiers
1. **MVP (paper-sufficient):** one anisotropic 2×2 per camera from LOO
   residuals with shrinkage toward the global pooled matrix:
   `R̂_i = (1−λ)S_i + λ R_global`, λ from sample count (Ledoit-Wolf-style or
   fixed-count rule, pre-registered).
2. **Toro-compatible:** nearest-calibration-point covariance (plan 01 — shared code).
3. **Stretch (only after MVP passes):** spatial Cholesky-parameterized
   `R(s)=L(s)L(s)ᵀ` fitted by residual likelihood. Not on the critical path.

## New code
`src/reliability/reliability/conditional_covariance.py` + tests.

```python
def estimate_conditional_covariance(residuals, min_count, shrinkage_target) -> CovarianceEstimate
    # per camera; sample cov + shrinkage + PSD floor; stores count + frame ('uv'|'xy')
def matrix_nll(residuals, covariances) -> float          # MNLL (§ E2)
def chi2_coverage(residuals, covariances, q=0.95) -> float  # C95 vs χ²_{2,q}
def sharpness(covariances) -> float                      # mean log|R|
```

Metrics stay here (they're covariance-specific, not in shared metrics.py —
but Brier/NLL of probabilities still import shared).

## Frames & units
Prefer image-space (px²) residuals with propagation through the projection
Jacobian to world when the runtime needs xy (§4):
`R_xy = J R_uv Jᵀ + J_θ Σ_θ J_θᵀ` — calibration uncertainty as an explicit
second term (do not silently absorb calibration error into confidence). The
Jacobian/propagation lives in `reliability.projection` (extend).

## Gates (= E2)
- Coverage C95 within pre-registered band AND competitive sharpness (a huge-R
  model passes coverage trivially — always report both).
- MNLL beats constant-global and Toro nearest-point on held-out runs.
- Per-camera x/y bias reported (feeds health monitor bias term).
- Unit tests: shrinkage limits (λ=0 → sample cov, λ=1 → target), PSD always,
  MNLL matches hand-computed values, coverage on synthetic Gaussian residuals
  ≈ nominal, Jacobian propagation vs finite differences.
