# network_commissioning_realism — would this commission in a real warehouse?

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-NET-COMMISSION
status: LOCKED
claim_ids:
- C1
- C6
assumption_ids:
- A01
- A10
- A12
- A13
- A15
reviewer_question_ids:
- RQ07
- RQ11
- RQ12
figure_ids:
- F03
- F09
dependencies:
- ASSET-RUNTIME
operational_inputs:
- camera_residuals
- leave_one_out_reference
- commissioning_baseline
evaluation_only_inputs:
- ground_truth_pose
primary_metric: agreement between operational and truth-based commissioning decisions
promotion_gate: Recover the actionable camera decision without operational truth leakage.
evidence_paths:
- logs/studies/network_commissioning_realism/exp1_gate_without_truth/summary.json
archive_rule: Preserve summary provenance and negative camera decisions.
next_action: Use as supporting commissioning evidence; do not claim end-to-end truth-free
  calibration.
```

<!-- RESEARCH-METADATA:END -->


**Claims served:** C1 and C6; paper scope in
[`research/papers/correlated_error_icra.md`](../../research/papers/correlated_error_icra.md).

## Question

The gated cross-bearing bias correction passed the belief-calibration gate
([`operational_residual_rcond/exp3`](../../logs/studies/operational_residual_rcond/exp3_two_dof_rcond/RESULTS.md):
NEES 8.51 → 1.06 on a held-out capture). But it was *selected* with Gazebo ground truth,
on four cameras with 125–530 detections each. A real warehouse has no ground truth and,
at 50–200 cameras, few detections per camera. This study asks whether the method survives
either condition.

- **Q1** Can the gate `|b_cross|/σ_cross` be computed **without truth**?
- **Q2** How many detections does a camera need before the gate is **trustworthy**?

## Answers (exp1)

**Q1 — yes, with two caveats.** The raw operational (GT-free, self-held-out) gate agrees
with the oracle on 3 of 4 cameras and is conservative by construction. The textbook state
correction — subtracting the belief's own uncertainty — is **degenerate** here (ratios of
170 and 767 from PSD flooring) and must not be used.

**Q2 — only outliers are decidable.** Camera C (ratio 4.63) is decided correctly from
**20 detections**. Every marginal camera stays below 60 % correct even with all available
data, and the small-sample failure mode is *false calibrate*, which exp2 measured at
−27 mm of held-out harm.

**Deployment consequence: correct the outliers, leave the rest raw.**

## Run

```bash
python3 experiments/network_commissioning_realism/exp1_gate_without_truth.py
```

Outputs → [`logs/studies/network_commissioning_realism/exp1_gate_without_truth/`](../../logs/studies/network_commissioning_realism/exp1_gate_without_truth/).
Offline, no Gazebo, no new capture.

## Reuse map

| need | reused from |
|---|---|
| GT-free residuals, self-held-out reference | `experiments/operational_residual_rcond/estimate_rcond.collect` |
| oracle residuals (evaluation only) | `estimate_rcond.collect_oracle` |
| captures, camera models, calibration in force | `experiments/operational_residual_rcond/rcond_common` |
| the gate itself | `experiments/external_camera_bias_model/exp2_two_dof_bias_fit.py` (definition), `tools/fit_projection_calibration.py` (deployed) |

## Method notes

- Subsamples are **contiguous windows**, not random draws: detections 0.2 s apart are not
  independent, and "this camera has seen N detections" means one or a few passes. Random
  subsampling would overstate sample efficiency by the autocorrelation factor.
- Every camera is held out of its own reference trajectory; the `R_cond` study measured a
  4.2× self-confirmation error for camera A without that.
- Ground truth appears only in the oracle column, as the thing the deployable estimators
  are scored against.
