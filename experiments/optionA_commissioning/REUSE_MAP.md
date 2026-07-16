# Option A — what was reused, moved, or is now removable

Audit of existing repo assets against the Option-A plan ("realistic commissioning of
external-camera trust maps under uncertain robot poses"), done before building the suite.
Verdict per asset. Date: 2026-07-15.

## Reused as-is (the plan was already ~60% built)

| asset | role in Option A |
|---|---|
| `scripts/visibility_comparison/build_belief_gp_events.py` | THE dataset builder: (μ⁻, P⁻, c, det_hit) per detection, GT quarantined as `eval_*`. Used unchanged for honest + whitenoise campaigns. |
| `scripts/visibility_comparison/fit_belief_aware_gp.py` | THE model code: naive / uncertainty-weighted / belief-spread / expected-kernel (uncertain-input GP), logit-residual prior seam. Imported as a library by every experiment — not reimplemented. |
| `scripts/geometry_visibility/campaign_metrics.py` + `docs/campaign_log_metrics.md` | Canonical column rules (planner_belief/gt only). All experiments comply; the documented retraction ("naive ≈ oracle at real σ") is *replicated*, not contradicted, by Exp2. |
| `logs/geometry_visibility_prior/calibrated_prior_v1/calibrated_prior.npz` | RQ3's I1 calibration-only prior (`prior_logit_mean_map`) and Exp7's px/m map. |
| `paper_artifacts/gp/warehouse_visibility_gp_v1/` (locked GP) | I2 historical prior, Exp6 matched-history setting, grid/geometry metadata, warehouse prism overlay for all figures. |
| `paper_artifacts/gp/archive/aws_gp_v7b_superseded/` | **Promoted from archive**: the stale-generation map is exactly the RQ5 stress asset (Exp6 setting B). Do not delete. |
| `logs/visibility_comparison/honest_campaign_v1` (43 runs) | All operational training/eval data (8 539 events). |
| `logs/visibility_comparison/whitenoise_campaign_v1` (41 runs) | Exp6 dynamics-change context (8 984 events, built into `logs/visibility_comparison/optionA_whitenoise_events/`). |
| `logs/visibility_comparison/warehouse_visibility_targets_v1` (556-sample teleport survey) | Provenance of the locked GP = the "earlier commissioning phase" narrative. |
| `geometry_visibility.trust_to_r_plan` + planner artifact schema | Exp7's unchanged planner seam. |
| `side_projects/mini_relifusion` showcase style | Plot conventions (palette, badges, panel style) via `optA_common.py`. |

## Moved / wrapped (small deltas, no duplication of logic)

- `optA_common.py` wraps `fit_belief_aware_gp` internals and **adds only the two missing
  analysis baselines** the plan requires: `tuned_point` (CV/pose-inflated length scale) and
  `fixed_blur` (shared mean covariance). Everything else delegates to the existing module.
- Leave-one-route-out validation added on top of the existing 5-fold run-group CV
  (stricter spatial split; the old CV in `validation_summary.csv` remains valid for its
  original interpolation claim).

## Removable / superseded for the Option-A storyline

- `logs/visibility_comparison/belief_aware_gp_expected_kernel_smoke/` — smoke run, safe to delete.
- `showcase_belief_aware_update.py` / `showcase_gp_before_after_updates.py` *outputs*
  (`belief_aware_gp_showcase/`, `depth_sensed_trajectory_gp_showcase/`) — superseded by
  Exp2/Exp3 figures; keep the scripts, the old PNGs can go.
- The headline use of run-group 5-fold numbers in `BELIEF_AWARE_GP.md` ("expected_kernel
  best Brier") — superseded by Exp2's route-level result (all modes tie at the real
  operating point); the doc should point at `logs/studies/optionA_commissioning/exp2_*/RESULTS.md`.
- NOT removable despite looking similar: `depth_sensed_initial_gp_v1` (an alternative
  deployable I1-class prior), `stack_capture2` (used by `hard_evidence.py`).

## Loose ends surfaced by this suite

1. **r_miss_uv mismatch**: offline tooling uses 40 px, runtime default 120 px
   (`unicycle_planner_node.py`). Exp7 uses the runtime value; reconcile before quoting
   R_plan magnitudes in the thesis.
2. **Hard NIS gate runaway** reproduced offline (Exp5): a soft gate (R·NIS/9.21) avoids
   the death spiral — candidate runtime improvement.
3. `build_belief_gp_events.py` and `campaign_metrics.py` implement the same stamp-join
   twice; unifying them would remove a divergence risk.
