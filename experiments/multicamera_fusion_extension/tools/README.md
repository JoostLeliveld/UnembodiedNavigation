# Operational data builders

These builders create the training surface for the factorised reliability
models. They only accept `operational/` data. Ground truth, oracle labels, and
paths under `evaluation_only/` are rejected before a row is written.

`build_opportunity_dataset.py` combines exported operational samples with an
operational prediction sidecar. Each sidecar row is keyed by `sample_id` and
contains the camera ID, predicted image position/covariance, predicted robot
height, stream-liveness flag, and belief-to-image association delta. It emits a
row only when the expected robot support lies in the validated image region,
the predicted scale is sufficient, the stream is live, and association timing
is within the frozen bound. An out-of-support pose is omitted, not relabelled as
a detector miss.

`build_loo_labels.py` then joins an operational leave-one-camera-out reference.
The reference must name the labelled camera in `excluded_camera_id`; otherwise
the command fails. It writes the image-space residual and usability label, but
leaves a miss unlabelled for conditional usability rather than inventing a
residual.

The actual sidecar/reference producers will be enabled only after the locked
four-camera detector and projection commissioning gate passes. The schema and
firewall are implemented now so that collection has a stable target.

`train_availability_gp.py` and `train_quality_gp.py` are thin wrappers around
the canonical belief-aware GP fitter. They preserve `run_id` and require a
complete held-out run, so random-frame validation cannot be introduced at the
last minute. Both write an input/model provenance card beside the fitted
artifact.

## Offline experiment evaluators (the fixed measurement apparatus)

`experiment_evaluators.py` is the data-independent scoring apparatus for
experiments **E1** (reliability prediction) and **E2** (conditional covariance
calibration), per `plans/11_experiment_campaign.md`. It deliberately lives here
and NOT in the `reliability` package: the leakage firewall pre-registers
`reliability.evaluation` as a forbidden planner-facing import, because
evaluators may score residuals derived from evaluation ground truth.

- `evaluate_reliability_prediction(...)` → Brier / NLL / ECE / AUROC / AUPRC /
  false-high-trust-rate for one method's predicted trust probabilities, with a
  clustered-bootstrap Brier CI over runs. `reliability_prediction_gate(...)`
  enforces the §21 E1 gate: combined GP+confidence must beat BOTH GP-only and
  confidence-only on Brier and NLL.
- `evaluate_covariance_calibration(...)` → MNLL / χ² C95 coverage / sharpness /
  bias / RMSE / median / p95, with a clustered-bootstrap MNLL CI.
  `compare_covariance_methods(...)` renders a coverage-and-sharpness-together
  table and flags over-inflated covariances (high coverage bought with useless
  width — the §E2 failure mode).

All scoring reuses the canonical scorers (`scripts/shared/metrics.py`), the
canonical covariance primitives (`reliability.conditional_covariance`), and the
canonical run-level inference (`reliability.campaign_statistics`); nothing is
re-derived. The evaluators are validated in
`tests/reliability/test_experiment_evaluators.py` against synthetic data with
known ground truth, so they run unchanged the moment real fitted models and
recorded runs exist.

## Offline replay sweep drivers (E4 / E5 / E6)

`replay_sweeps.py` drives the degradation experiments by transforming recorded
frames and re-running the SAME offline replay pipeline
(`reliability.replay.run_replay`) over identical detections, so any difference
between conditions is attributable to the transform, not to a retuned filter
(§21 fusion gate). It loads no truth: the caller passes evaluation-only
`EvaluationFrame`s from an evaluation harness.

- `run_camera_subset_sweep(...)` (**E4**) — every non-empty camera subset through
  the same fusion mode; a size-1 subset equals the single-camera result, so
  `fusion_gain_p95` (best single-camera p95 − full-set p95) is like-for-like.
- `run_dropout_sweep(...)` / `run_latency_sweep(...)` (**E5**) — intermittent
  dropout and delayed-association transforms over a severity axis, each with an
  error-severity AUC.
- `run_calibration_drift_sweep(...)` (**E6**) — a controlled constant
  world-position bias on one camera (images untouched = "controlled
  calibration-ablation evidence"), over a bias-magnitude axis.

The frame transforms (`filter_frames_to_subset`, `drop_camera_*`,
`delay_camera`, `bias_camera_position`) are pure. Cross-run paired CIs are the
caller's job: run each recorded run through a sweep, collect per-run
`ReplayMetrics`, then use `reliability.campaign_statistics.summarize_paired` /
`hierarchical_bootstrap`. Validated in `tests/reliability/test_replay_sweeps.py`
against synthetic runs with known monotone behaviour — including the honest
finding that a NIS gate makes error-vs-drift non-monotone because the gate
rejects a grossly biased camera (the protective behaviour E6 studies).
