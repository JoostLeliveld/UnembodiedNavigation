# calibration_drift_lifecycle — does the commissioning gate survive into service?

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-DRIFT
status: LOCKED
claim_ids:
- C1
- C6
assumption_ids:
- A01
- A03
- A12
- A14
reviewer_question_ids:
- RQ04
- RQ07
- RQ14
figure_ids:
- F04
dependencies:
- ASSET-RUNTIME
operational_inputs:
- camera_residual_change
- commissioning_baseline
evaluation_only_inputs:
- injected_pose_fault
- ground_truth_pose
primary_metric: drift detected before stale correction becomes harmful
promotion_gate: Detection threshold precedes harm threshold across controlled faults.
evidence_paths:
- logs/studies/calibration_drift_lifecycle/exp1_stale_correction/drift_lifecycle.json
archive_rule: Preserve summary fault ladder and figure.
next_action: None; state controlled-injection scope explicitly.
```

<!-- RESEARCH-METADATA:END -->


**Question.** The 2-DOF per-camera bias correction is fitted once at commissioning
(assumption **A2**) and then deployed frozen. Cameras drift. Does a *stale* correction
become harmful, and can the drift be detected with the same GT-free statistic the
commissioning gate already uses?

**Why it matters.** This is the register's runway item #1: it upgrades **C5** (the
dominant error is per-camera systematic bias; correcting it is a gated GT-free decision)
from a commissioning result into the **lifecycle capability C3 already claims**. Without
it, "correct the outliers, leave the rest raw" is a policy with no expiry date.

**Chapter served.** `research_story/09_multicamera_handover_fusion`; ICRA addendum C3/C5.

## Fault model (A4 + A8)

The **physical camera moves; the estimator's calibration copy does not.** The recorded
pixel is re-imaged through the drifted pose, then projected through the stale calibration:

```
xy_true = true_cam.pixel_to_ground(u, v)        # what the recorded pixel meant
u', v'  = drifted_cam.world_to_pixel(xy_true)   # where it lands after the camera moves
xy_est  = _project_pixel_to_world(u', v', true_cam, **stale_kwargs)
```

This is the *dual* of perturbing the estimator's calibration copy
(`calibration_perturbation.reproject_world`) — same induced world bias, opposite sign. The
physical direction is modelled because the **sign is what decides whether a stale
correction helps or hurts**.

Identity drift must reproduce the deployed pipeline bit-for-bit. Asserted every run:
**0.00e+00 m over 230 detections**, which also proves `PinholeGroundCamera` is an exact
re-parameterisation of the deployed `ObliqueCameraModel`, not an approximation.

Real detections, real pixels, real recorded capture. The only injected quantity is the
calibration perturbation itself.

## Result

| | |
|---|---|
| **A stale correction does become harmful** | camera C, the largest commissioned correction: at rest it halves error (0.043 vs 0.088 m raw); at **0.25° yaw** it is *worse than not correcting* (0.100 vs 0.094 m) |
| **The commissioning gate is NOT an in-service monitor** | its absolute form fires at rest on cameras A and B (ratios 10.2 and 5.0), and camera B's drift is **masked** at 0.25° (ratio *drops* to 0.31) when the induced bias cancels the resident one |
| **The change form of the same statistic is** | `\|b_cross(δ) − b_cross(0)\| / σ_cross(0)` is monotone on **all 4 cameras × both fault types**, and detects at **0.1° yaw / 0.025 m** |
| **Detection precedes harm** | 0.1° detected vs 0.25° harmful — one ladder rung of margin |

Full write-up: `logs/studies/calibration_drift_lifecycle/exp1_stale_correction/RESULTS.md`.

## What is NOT ours

- The E6 injector (`reliability.calibration_perturbation`) is pre-existing.
- The gate statistic (`|b_cross|/σ_cross`, threshold 1.2) is
  `experiments/network_commissioning_realism/exp1_gate_without_truth.py`, imported, not
  reimplemented.
- The smoother / operational reference is `experiments/operational_residual_rcond`.
- The v3 calibration constants are `projection_calibration_v3`, loaded and applied exactly
  as the runtime applies them. Never refitted here.

## Reproduce

```bash
python3 experiments/calibration_drift_lifecycle/drift_lifecycle.py
```

Offline. No Gazebo run, no new capture. Uses `fusion_handover_20260721` — the capture
**held out of the v3 calibration fit**, which is the only honest place to ask what a
frozen constant does to unseen data.

## Reuse map

| need | reused from |
|---|---|
| repo root | `scripts/shared/paths.repo_root` |
| calibration perturbation | `reliability.calibration_perturbation` (`PinholeGroundCamera`, `perturb`) |
| gate statistic | `network_commissioning_realism/exp1_gate_without_truth` (`cross_axis_series`, `gate_ratio`, `GATE`) |
| smoothed operational reference (camera held out) | `operational_residual_rcond/estimate_rcond.smooth_capture` |
| residual records | `reliability.operational_residual.build_operational_residuals` |
| projection + deployed correction | `reliability.projection` (`projection_kwargs_for_camera`, `_project_pixel_to_world`) |
| capture loading, truth join (eval only) | `operational_residual_rcond/rcond_common` |
