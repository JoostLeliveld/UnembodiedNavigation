# 07 — Cold start: how much trust knowledge is free before driving?

**Question.** Before a single commissioning metre is driven, how much of the trust field can
be predicted from what a facility realistically has — camera calibration (07a), and possibly
sensed/maintained geometry (07b)? And how much driving does a good prior save?

**Status: ACTIVE.** Two sub-tracks with different ambitions — keep them separate so the
cheap prior isn't judged by the expensive one's standard:

- [07a_fov_range_prior/](07a_fov_range_prior/) — calibration-derived FOV/range prior.
  **Supporting ablation.** Claim ceiling: "reduces early commissioning burden."
- [07b_geometry_occlusion_model/](07b_geometry_occlusion_model/) — geometry/occlusion
  visibility model. **Candidate separate contribution ONLY IF** it uses realistically
  available geometry AND clearly beats range/FOV. Everything stays `MODEL ONLY` until
  checked against detector evidence.

## The results we're aiming for

- **Fig 07A** — prior maps: distance-only, +image-edge/FOV, combined.
- **Fig 07B (decision figure for 07a)** — learning curves: held-out NLL vs frames / driven
  metres / commissioning minutes for {no prior, weak prior, prior+updates}. **Aim:
  prior+updates reaches a target NLL with the fewest metres.** exp34's budget curves are the
  prototype.
- **Fig 07C** — incorrect-prior recovery: detector evidence overrides a wrong FOV/range
  prediction (safety of the prior, not just its accuracy).
- **Fig 07D** — geometry prediction vs empirical miss map, with false-visible /
  false-occluded regions made explicit.
- **Fig 07E (verdict table for 07b)** — range-only vs range+FOV vs geometry vs geometry+GP
  on held-out NLL, miss recall, calibration. **Aim: an honest verdict either way** — the S1
  finding that the Jacobian *range term is co-dominant* already suggests "range and
  obliquity dominate" is a live outcome, which demotes 07b to an ablation without hurting
  the thesis.

## Implemented now

| Item | Tag | Note |
|---|---|---|
| Geometry visibility module (`scripts/geometry_visibility/`, 9/9 tests, VALIDATION.md, 10-fig chain) | model_plumbing | S1 explanation gate **PASSED**: Spearman 0.730 / R² 0.51 vs empirical GP over 22,917 cells, robust to F_std gate |
| `calibrated_prior_v1` | measured_in_sim | the 07a prior artifact on AWS |
| exp34 init/budget curves | measured_in_sim | Fig 07B prototype |
| Day-zero prior at 4-cam scale (`paper_artifacts/gp/warehouse_full_4cam_dayzero_v1`) | model_plumbing | calibration-only, no training data/GT; proves the prior pipeline runs on `warehouse_full_4cam` (feeds ch.08) |
| `depth_sensed_initial_gp_v1` | model_plumbing | viewpoint-matched sensed structure; sim-sensor benchmark AUROC 0.968 vs 0.782 camera-only — sensing realism must stay explicit |
| Survey of realistic sensing options (`docs/reliability_prior_sensing_survey.md`) | reference | what geometry a real facility actually has |

## Gap → next experiments

07a: run the five-condition data-efficiency comparison (constant / distance / distance+edge /
GP-no-prior / prior+updates) on existing capture data → Figs 07A–07C.
07b: S3 zero-shot transfer via the existing teleport-capture scripts, then Fig 07D/07E.

## Gate

07b promotion requires realistic geometry inputs (sensed LiDAR/RGB-D/stereo, maintained CAD
declared as such, approximate calibration — never perfect Gazebo shelf heights as an
operational input) AND a clear 07E win. Otherwise both tracks publish as ablations inside
Contribution 1's data-efficiency story.
