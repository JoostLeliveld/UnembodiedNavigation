# Experiment B — real logged reliability prediction (ch.03 + ch.04)

**This is the discriminating, reportable single-camera result.** NEW capture in
`warehouse_aws`; no reuse of `belief_aware_gp_score_v1` / `belief_gp_events`.

## Question
On real operational detector logs and real belief `(μ_t, P_t)`, does U5 predict
held-out camera hit/miss (and usable-observation) better than U0–U4, on
route-disjoint splits?

## Collection design (NEW data — DEDICATED COMMISSIONING DRIVE)
**Data source = `DATA_SOURCE_commissioning_drive.md`**, NOT the EFE navigation
campaign. A coverage drive systematically traverses the whole drivable region
("as if teleoperated"), decoupled from the planner, so sampling is unbiased and
the map is not coupled to the planner that consumes it.
- Serpentine coverage of all aisles/lanes/connectors; ≥2 directions, repeat passes.
- Central + image-edge + far + shelf-occluded regions; genuine successes AND misses.
- Log per opportunity: `state_source`, `state_mean_xy`, `state_cov_xy`,
  `camera_live`, `frame_age_ms`, `support_probability`, `detection_received`,
  `association_valid`, `raw_confidence`, `bbox_xyxy`, `selected_pixel_uv`,
  `training_label_availability`, GT (`gt_*`, **evaluation-only**), hashes.
- Opportunity = belief-ellipse sigma points project into validated image region
  (`support_probability ≥ 0.8`), stream live, frame not stale. A miss is only a
  label when an opportunity existed — do NOT label out-of-support frames as fails.

## Factorisation (ch.04 second contribution, optional promote)
- Availability `a(s) = P(detection | opportunity, s)`;
- Quality `q(s) = P(usable | detection, s)` with usable ≡ `|e| ≤ δ` (GT-scored,
  eval-only) — a camera can detect reliably yet localize poorly.
- Report `a`, `q` separately; promote ch.04 only if factorised beats the scalar
  field on NIS-prediction / false-high-trust (registry gate).

## Splits (mandatory)
Grouped by route/run — NEVER random frames (adjacent frames leak). Train routes /
calibration routes / fully held-out test routes + one leave-region-out contiguous
mask. Inducing points cover the drivable region (k-means or masked grid), never
chosen from test routes.

## Methods & metrics
U0–U5 (+U6 ceiling). Metrics via `scripts/shared/metrics.py`: Brier, NLL, ECE,
AUROC, AUPRC, false-trust `P(Y=0|p≥0.9)`, false-distrust `P(Y=1|p≤0.1)`; by
region / range / image-edge distance / reported-covariance bin. Bin predictions
by GP σ and plot actual error vs σ. Grouped bootstrap CIs (run-level unit).

## P_t calibration prerequisite (document §11 — genuinely new)
Uncertain-input GP is only meaningful if `P_t` is calibrated. On eval runs
compute NEES over `(μ_t − x_t^GT)`; report 50/90/95% ellipse coverage + median
NEES; if overconfident, fit ONE scalar `κ_P` on the calibration split and FREEZE
it before test. Compare raw vs κ_P-calibrated vs mean-only. (New helper — see
plan; reuse `campaign_metrics` loaders + `scripts/shared/metrics.py`.)

## Gate (= ch.03 acceptance)
U5 beats U0–U4 on held-out Brier + NLL (grouped CIs); GP σ high in unexplored
regions (leave-region-out); result does NOT depend on GT training labels
(firewall). A null result is acceptable and reported honestly. Four-panel
figure per camera: empirical rate / GP mean / GP σ / held-out residual.
Outputs → `logs/studies/single_camera_uigp_reliability/expB_real_prediction/RESULTS.md`.
