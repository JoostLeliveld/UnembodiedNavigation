# external_camera_bias_model — audit of the deployed along-bearing projection correction

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-BIAS
status: LOCKED
claim_ids:
- C1
assumption_ids:
- A01
- A02
- A08
- A13
- A15
reviewer_question_ids:
- RQ07
- RQ11
- RQ14
figure_ids:
- F01
- F03
dependencies:
- ASSET-RUNTIME
operational_inputs:
- camera_id
- pixel_measurement
- calibration
evaluation_only_inputs:
- ground_truth_pose
primary_metric: held-out residual bias and coverage
promotion_gate: Bias mechanism repeats across held-out captures.
evidence_paths:
- logs/studies/external_camera_bias_model/exp1_residual_characterization/audit.json
- logs/studies/external_camera_bias_model/exp2_two_dof_bias/summary.json
- logs/studies/external_camera_bias_model/exp3_projection_pipeline_minimal/summary.json
archive_rule: Preserve summaries provenance residual tables and decisive figures permanently.
next_action: None; use as locked paper evidence.
```

<!-- RESEARCH-METADATA:END -->


**Question.** After the *already-deployed* per-camera along-bearing projection
calibration is applied, what residual remains in an external camera network —
and is that remainder homogeneous enough for the planner's single scalar `R`?

**Claim served.** C1. This is locked evidence for
[`research/papers/correlated_error_icra.md`](../../research/papers/correlated_error_icra.md).

## What is NOT ours

The along-bearing bias model is **pre-existing, fitted, and deployed work**:

- artifact: `logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v2/projection_calibration.json`
- model: `correction_m = intercept_m + slope_per_m * ground_distance_m`, applied
  along the camera→point bearing, positive away from the camera
- fitted by `experiments/multicamera_commissioning_bigwarehouse/tools/fit_projection_calibration.py`
  from *both* `gt_validation_smoke*_20260716` captures
- applied by `src/reliability/reliability/projection.py`
  (`_project_pixel_to_world`) and wired into every multicam campaign config via
  `manager_projection_calibration:`

This study does **not** refit or re-derive that model. It loads it and applies it
exactly as the runtime does.

## What IS ours (the audit)

1. **Post-correction conditional covariance** `R_cond,c(x)` — the residual that
   survives the deployed fix. This is the honest noise floor.
2. **Network heterogeneity** — how much `R_cond` varies *between* cameras and
   *within* a camera across range. The planner currently collapses this to one
   scalar blended with detection probability; the size of that modelling gap is
   the candidate contribution.
3. **Leftover structure** — does bias survive correction (cross-bearing terms,
   range curvature the linear model misses, per-region effects)?
4. **Extrapolation risk** — each camera's calibration was fitted over a limited
   range window; how much of the reachable floor lies outside it.
5. **Handover step, post-correction** — the discontinuity injected when the
   manager switches source, versus `pixel_max_correction_jump_m` (0.5 m).

## exp2 — fitting the missing degree of freedom (2026-08-04)

exp1 audits; **exp2 fixes**. Three independent lines converged on the same next step
(exp1's untouched cross-bearing bias, `projection_amplification/exp1`'s bias-transfer
result, `operational_residual_rcond/exp2`'s bias-bound `R_cond`), so exp2 fits and
selects among 2-DOF candidates under leave-region-out CV.

Result: `MD_plus_cross_const` — keep the frozen deployed along-bearing correction, add
**one** cross-bearing parameter — wins on B/C/D and loses on A, and one measurable
quantity predicts which: `|b_cross|/σ_cross` (A 0.16 → −27 mm, B 1.05 → +0 mm,
D 1.30 → +10 mm, C 4.66 → +42 mm). Camera C's remaining bias drops 77→4 mm and its
conditional covariance improves by 2.4 nats, which is the loop closing on the camera
whose bias was the blocker. **Nothing is wired into the runtime** — the projection has
one along-bearing DOF, so deployment is a separate deliberate change.

## Reproduce

```bash
python3 experiments/external_camera_bias_model/residual_audit.py       # exp1 data + stats
python3 experiments/external_camera_bias_model/make_figures.py         # exp1 fig_b1..b6
python3 experiments/external_camera_bias_model/exp2_two_dof_bias_fit.py  # exp2 fig_x1..x3
```

Outputs land in `logs/studies/external_camera_bias_model/exp1_residual_characterization/`
and `.../exp2_two_dof_bias/`.

Pure offline analysis — no Gazebo, no ROS launch. Ground truth is used for
**evaluation only** (measuring a residual *is* evaluation); it never enters a
projection, a correction, or a covariance.

## Reuse map

| need | reused from |
|---|---|
| repo root | `scripts/shared/paths.repo_root` |
| Spearman, binning | `scripts/shared/metrics` (`spearman`, `binned`) |
| camera models from SDF | `reliability.projection.camera_model_from_world` |
| pixel→world + deployed correction | `reliability.projection._project_pixel_to_world` |
| deployed calibration loader | `reliability.projection.load_projection_calibration` |
| nearest-stamp truth join | `attach_evaluation_truth._nearest` |
| along-bearing fit (for model comparison only) | mirrors `fit_projection_calibration._fit_line` |
