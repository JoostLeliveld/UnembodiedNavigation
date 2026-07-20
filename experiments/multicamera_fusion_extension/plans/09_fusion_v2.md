# Plan 09 — Fusion v2: Joseph updates, robust weighting, information selection (§12)

## Purpose
Upgrade the fusion layer to the paper's estimator: sequential timestamped
updates with Joseph form, Student-t robust reweighting, information-aware
selection, and the fuse-vs-select consistency logic (RQ2, E3/E7).

## What exists / reuse
- `reliability.fusion` — selection policies (primary/zone/score/freshest/
  static/conservative-best) = baselines B0/B3-family; `sequential_kalman_update_2d`;
  2×2 helpers. Extend the same file/style (pure Python, SPD-validated).
- `reliability.handover` — source-switch inflation stays as-is, applied before
  the update.

## New code (extend `fusion.py`, new tests)
```python
def joseph_update_2d(state, cov, observation, R) -> (state, cov, nis)
    # (I−KH)P(I−KH)ᵀ + KRKᵀ — numerical stability at small R
def robust_reweight_covariance(R, nis, dof=4.0, w_min) -> R_robust
    # w = (ν+2)/(ν+d²); R/max(w, w_min) — MUST be ablatable (flag off = plain update)
def expected_information_gain(cov_prior, H, R) -> float
    # Δ = log|P⁻| − log|P⁺|
def select_information_best(observations, cov_prior) -> observation
    # B8: geometry-aware selection — high reliability with poor geometry can lose
def fuse_or_select(observations, taus, healths, consistency) -> decision
    # C_t = {i: τ≥τ_min, h≥h_min}; mutually consistent → fuse all;
    # one inconsistent → isolate/select among consistent subset (§12.5)
```
Constant-velocity propagation between per-observation timestamps (shared with
plan 01 so every condition uses the same process model/Q).

## Gates
- Unit: Joseph ≡ standard form on well-conditioned cases; PSD retained at
  extreme R ratios; Student-t weight limits (d²→0 ⇒ w→(ν+2)/ν, d²→∞ ⇒ floor);
  information gain matches log-det arithmetic; fuse_or_select truth table.
- Robust term ablation is mandatory in E3/E7 (otherwise reviewers can't
  attribute gains GP-vs-gating).
- Expected finding treated as hypothesis: fusion wins when errors are unbiased
  & consistent; selection/robust-subset wins under persistent bias (E7).
