# Plan 11 — Experiment campaign E0–E8, baselines, statistics (§14–18)

## Baselines (all share detections, timestamps, calibration, CV model, one Q)
B0 best-single | B1 constant-R (=R0) | B2a/B2b Toro (plan 01) | B3 calibrated-
confidence-only | B4 GP-only | B5 GP+confidence | B6 full (health-aware robust)
| B7 GP-selection | B8 information-selection | B9 oracle (evaluation-only,
firewall; never in headline tables). Map onto existing replay IDs R0–R4/M5–M8
in `reliability.benchmark`; add missing conditions there.

## Experiments (each = one `logs/studies/multicamera_fusion_extension/<expN>/` with RESULTS.md)
- **E0 component validation** — reuse commissioning GT tools + projection
  audits. Gate: frames/timestamps/projection residual/coverage pass BEFORE any
  fusion run. Largely done by commissioning M1/M2; E0 is the freeze record.
- **E1 reliability prediction** — plans 03/04/05 evaluation. Grouped
  route/region splits; Brier/NLL/ECE/AUPRC/false-trust; data-fraction curve (RQ6).
- **E2 covariance calibration** — plan 06 evaluation: MNLL, χ² C95, sharpness,
  per-camera bias.
- **E3 nominal localization** — offline replay of B0–B8 over identical
  detections (detector stochasticity eliminated); ATE RMSE/median/p95/p99/max,
  RPE 1s/5s, NIS mean+CDF, NEES (eval-only), jitter, displacement-std (Toro's
  metric, reported not headline).
- **E4 camera subsets** — 4 cameras ⇒ 15 non-empty subsets (paper draft says
  7 of 3 cameras; we have A–D). Track N_installed / N_covering(s) /
  N_healthy(t) / N_observing(t) separately. Plots: error vs healthy count, per
  exact subset, vs viewpoint diversity, fusion gain vs best constituent.
- **E5 dropout & latency** — offline masking sweeps (permanent, sudden at
  25/50/75%, p_drop ∈ {0,.1,.25,.5,.75}, bursts {0.5,1,2,5}s, delays
  {0,50,100,200,500,1000}ms) via new `tools/run_dropout_sweep.py`; selected
  cases re-run live to verify replay transfers. camera_A's real ~1 Hz CPU path
  is a latency condition — measured, not simulated (commissioning M1d).
- **E6 calibration drift** — `tools/run_calibration_drift_sweep.py`: perturb
  the estimator's calibration copy (yaw/pitch/roll/translation/principal-point/
  focal, one-factor-at-a-time levels per §15-E6 table), images untouched =
  "controlled calibration-ablation evidence"; plus ≥1 physical (in-sim: move
  the camera model by a measured amount) run = deployment evidence. Methods:
  constant, Toro, GP+conf (no health), full, hard-NIS, oracle removal.
  Health-monitor critical-failure rule applies (plan 08).
- **E7 selection vs fusion** — conditions: all-good / one-noisy-unbiased /
  one-biased / correlated-viewpoints / complementary / intermittent
  false-high-confidence. Methods incl. random-camera control.
- **E8 closed-loop navigation** — P0 const/const, P1 Toro/const, P2 GP+conf/
  const, P3 full/R_plan, P4 full/reliability-blind-shortest. 4 tasks × 5 seeds
  × 5 conditions = 100 nominal + 4 selected failure campaigns. Frozen aws
  planner config; safety-framed metrics (breach-free clean-goal, GT clearance,
  belief tr(P) time-above-threshold, camera handovers, health transitions).

## Statistics (§16)
Unit = run / route-subset replay / fault episode (never frames). Paired design
on identical detections; paired Δ vs Toro with hierarchical bootstrap
(route→seed→episode) 95% CIs + proportion-improved; Wilson intervals for
binary outcomes. Pre-registered primaries: full-vs-Toro, full-vs-GP-only,
fusion-vs-selection; all else exploratory. The 0.45-threshold decision
(derive-from-aws vs keep-as-convention + sensitivity curve) is made BEFORE
runs (commissioning M5 rule). Positive control required: ≥1 regime with
released corrections and sane NIS/NEES.

## Data plan (§18)
Reuse the frozen `paper_protocol.yaml` route/seed structure; target ≈60
recorded runs (routes × seeds × sessions varying lighting/clutter/timing/
direction) → 30 train / 10 calibration / 20 held-out with spatial novelty.
Offline replay expands to thousands of method-subset evaluations without
recollecting.
