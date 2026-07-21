# Plan 03 — Per-camera availability & quality GPs (factorised, ch.04)

## Purpose
Two GP classifiers per camera (§5.1/5.2, §7): availability `a_i(s)` and
conditional usability `q_i(s)`. This IS research_story ch.04 ("are availability
and conditional accuracy different spatial fields?") executed at 4-cam scale —
candidate second contribution A.

## What exists / reuse (do NOT reimplement)
- `scripts/visibility_comparison/fit_belief_aware_gp.py` — canonical GP code
  (naive / uncertainty-weighted / belief-spread / expected-kernel modes). The
  §7.3 belief-uncertainty options map 1:1 onto these modes; mean-only is the
  mandatory baseline.
- `reliability.providers` grid `.npz` provider — serving format stays unchanged.
- Geometry day-zero prior (`warehouse_full_4cam_dayzero_v1`) = the range/FOV
  baseline of RQ6/H1 and the GP mean-function candidate.
- `scripts/shared/metrics.py` for Brier/NLL/ECE/AUROC.

## New code (wrappers only)
- `tools/train_availability_gp.py`, `tools/train_quality_gp.py` — thin CLIs:
  load plan-02 dataset → select label column (A or G) → call the canonical
  fitter per camera → save `.npz` grids (mean + std) + model card JSON
  (kernel, lengthscales, inducing summary, train-route list, hashes).
- Feature config: baseline `s=[x,y]`; augmented camera-centric
  `φ_i(s)=[x,y,r_i,β_i,û,v̂]` computed from calibration (operational, allowed).
  Report both; if range/FOV-augmented ≈ spatial-only, say so (honest-null rule).
- Conservative planning heads (§7.4): `ã = clip(μ_a − λ_a σ_a, a_min, 1)`
  (same for q̃) implemented in the provider layer, λ pre-registered.

## Splits
Grouped by route/run (never random frames): train routes = single-camera
passes; held-out = handover traverses (matches frozen `paper_protocol.yaml`
D0/D1 design); plus one leave-region-out mask. Inducing points cover the
drivable region (k-means or masked grid), never chosen from test routes.

## Gates (= E1 acceptance)
- GP beats global-rate, range-only, range+edge, nearest-calibration-point and
  raw/calibrated confidence on held-out Brier + NLL (H1), grouped bootstrap CIs.
- Factorised (a,q) beats single undifferentiated field on NIS-prediction /
  false-high-trust — else ch.04 stays closed and we keep the scalar field
  (registry gate: "promote only if both components learned + validated + beat
  scalar + change R_plan/navigation").
- GP σ high in unexplored regions (leave-region-out check).
- Four-panel figure per camera: empirical rate / GP mean / GP σ / held-out
  calibration residual.
- Blocked on plan 02 data (hence commissioning M1/M2).
