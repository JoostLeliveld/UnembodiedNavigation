# Belief-Aware GP Module

This module extends the locked offline visibility GP with detector-event updates
that use planner belief means and belief covariance from campaign logs.

## Command Chain

First verify that the baseline paper GP is the current locked artifact:

```bash
python3 scripts/visibility_comparison/audit_visibility_gp_artifacts.py
```

Build detector events from campaign logs:

```bash
python3 scripts/visibility_comparison/build_belief_gp_events.py \
  --campaign logs/visibility_comparison/honest_campaign_v1 \
  --out logs/visibility_comparison/belief_gp_events
```

Fit the planner-compatible expected-kernel method and its ablation baselines:

```bash
python3 scripts/visibility_comparison/fit_belief_aware_gp.py \
  --events logs/visibility_comparison/belief_gp_events/events.csv \
  --out logs/visibility_comparison/belief_aware_gp_v1 \
  --grid-from paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz
```

For an apples-to-apples continuation of the original `yolo_score_raw` GP, use:

```bash
python3 scripts/visibility_comparison/fit_belief_aware_gp.py \
  --events logs/visibility_comparison/belief_gp_events/events.csv \
  --out logs/visibility_comparison/belief_aware_gp_score_v1 \
  --grid-from paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz \
  --target score
```

Create the explanatory showcase figure:

```bash
python3 scripts/visibility_comparison/showcase_belief_aware_update.py --target score
```

Compare the locked initial GP against the after-update maps:

```bash
python3 scripts/visibility_comparison/showcase_gp_before_after_updates.py --target score
```

Build a deployable depth-sensed initial GP, then update it from trajectory
events as residual evidence:

```bash
python3 scripts/visibility_comparison/build_depth_sensed_initial_gp.py

python3 scripts/visibility_comparison/fit_belief_aware_gp.py \
  --events logs/visibility_comparison/belief_gp_events/events.csv \
  --out logs/visibility_comparison/depth_sensed_trajectory_gp_score_v1 \
  --grid-from logs/visibility_comparison/depth_sensed_initial_gp_v1/depth_sensed_initial_gp.npz \
  --prior-gp logs/visibility_comparison/depth_sensed_initial_gp_v1/depth_sensed_initial_gp.npz \
  --prior-map-key P_mean_map \
  --target score

python3 scripts/visibility_comparison/showcase_gp_before_after_updates.py \
  --target score \
  --artifact-dir logs/visibility_comparison/depth_sensed_trajectory_gp_score_v1 \
  --baseline-gp logs/visibility_comparison/depth_sensed_initial_gp_v1/depth_sensed_initial_gp.npz \
  --baseline-label "depth-sensed initial GP" \
  --out logs/visibility_comparison/depth_sensed_trajectory_gp_showcase
```

## Outputs

The fitter writes one planner-compatible artifact per mode:

- `<target>_naive_gp.npz`: deterministic GP over belief means.
- `<target>_uncertainty_weighted_gp.npz`: same inputs, but high belief covariance
  inflates observation noise.
- `<target>_belief_spread_gp.npz`: each aggregate event is spread into covariance
  sigma points while conserving total observation precision.
- `<target>_expected_kernel_gp.npz`: the paper-style uncertain-input GP. The
  RBF kernel is analytically integrated under each pose belief `N(m, S)`.
- `fit_summary.csv`: final fit statistics.
- `validation_summary.csv`: run-group held-out Brier, log loss, and AUC.

The showcase script writes:

- `logs/visibility_comparison/belief_aware_gp_showcase/belief_aware_update_showcase_score.png`
- `logs/visibility_comparison/belief_aware_gp_showcase/belief_aware_update_showcase_score.pdf`
- `logs/visibility_comparison/belief_aware_gp_showcase/belief_aware_update_showcase_score_summary.csv`
- `logs/visibility_comparison/belief_aware_gp_showcase/gp_before_after_updates_score.png`
- `logs/visibility_comparison/belief_aware_gp_showcase/gp_before_after_updates_score.pdf`
- `logs/visibility_comparison/belief_aware_gp_showcase/gp_before_after_updates_score_summary.csv`

The depth-prior workflow writes:

- `logs/visibility_comparison/depth_sensed_initial_gp_v1/depth_sensed_initial_gp.npz`
- `logs/visibility_comparison/depth_sensed_initial_gp_v1/depth_sensed_initial_gp_showcase.png`
- `logs/visibility_comparison/depth_sensed_trajectory_gp_score_v1/yolo_score_raw_<mode>_gp.npz`
- `logs/visibility_comparison/depth_sensed_trajectory_gp_showcase/gp_before_after_updates_score.png`
- `logs/visibility_comparison/depth_sensed_trajectory_gp_showcase/gp_before_after_updates_score_summary.csv`

All `.npz` GP artifacts keep the planner-facing schema:

- `xs`
- `ys`
- `P_mean_map`
- `P_conservative_plan_map`

## Current First-Pass Result

The first honest-campaign run produced 8,539 usable detector events and 309
aggregate belief points at `0.20 m` resolution. After adding the analytic
expected-kernel mode, 5-fold run-group validation on the trajectory-only update
selects `expected_kernel` as the best Brier mode for both the `yolo_score_raw`
target and the binary `det_hit` target.

With the depth-sensed initial GP used as a prior mean function, the same
trajectory events are fit as logit-space residuals. The current score-target
run has:

| mode | Brier | conservative map mean |
| --- | ---: | ---: |
| `prior_only` | 0.09671 | 0.29704 |
| `naive` | 0.01017 | 0.44636 |
| `uncertainty_weighted` | 0.01013 | 0.44645 |
| `belief_spread` | 0.00946 | 0.42523 |
| `expected_kernel` | 0.01018 | 0.43596 |

This means the depth prior supplies the structural shadows and the trajectories
then lift or lower local reliability where detector evidence disagrees with the
one-frame depth estimate. In this depth-prior run, `belief_spread` is the best
Brier ablation, while `expected_kernel` remains the paper-style uncertain-input
implementation.

For the trajectory-only module, use `expected_kernel` as the main thesis method.
For the depth-prior continuation, report both `expected_kernel` and the
empirically strongest residual ablation (`belief_spread` in the current run).
Treat `naive`, `uncertainty_weighted`, and `belief_spread` as ablations that
show what changes when uncertain observations are pinned, downweighted, or
discretely spread instead of integrated in the kernel.

## Data Hygiene

Training input uses `planner_belief_x/y` and `planner_cov_*`. It intentionally
does not use stale `state_x/y`. Ground-truth columns in `events.csv` are prefixed
with `eval_` and are carried only for auditing and evaluation.
