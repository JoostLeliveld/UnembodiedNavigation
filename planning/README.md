# EFE Route Planning

[Back to repository overview](../README.md)

This module shows how constant and learned predictive camera covariance change
route planning under the same warehouse geometry and task seeds.

## Story

Only predictive observation covariance changes. Route behavior changes because
that covariance changes future belief growth and belief-tube keep-in
feasibility.

## Visual Demonstration

![Robustness spread](../paper_artifacts/figures/robustness_spread.png)

Red trajectories are C1 constant-covariance runs. Blue trajectories are C2
visibility-aware runs. The background is the learned conservative reliability
field used by the planner.

Planned media is listed in [`demos/`](demos/): paired C1/C2 route stills,
rollout GIFs, a covariance-along-route plot, and a side-by-side comparison
video.

## Inputs And Outputs

| Input | Output |
| --- | --- |
| `/state/bev` and planner belief | `/cmd_vel_raw` and `/cmd_vel` |
| `/goal_bev` | planner preview and diagnostics |
| GP artifact for C2 | global route, waypoint tracker targets, run metrics |
| Shared driveable/no-go geometry | collision, reach, path, and localization outcomes |

## Method

C1 and C2 share the same world, route seeds, driveable layer, execution tracker,
and optimizer budget. They differ in predictive camera covariance:

```text
C1: R_plan = constant camera covariance
C2: R_plan = GP-derived state-dependent camera covariance
```

The GP is not a direct visibility reward. It changes the predicted observation
covariance used in the EFE risk/ambiguity terms and in the belief-tube keep-in
feasibility term.

## Performance And Diagnostics

Current packaged campaign outcome:

| Condition | Clean reaches | Collisions | Other outcomes |
| --- | ---: | ---: | --- |
| C1 `constant_R_efe` | 12/20 | 8/20 | none |
| C2 `visibility_aware_efe` | 16/20 | 2/20 | 1 near-success, 1 infrastructure-invalid |

Single-run mechanism figure:

- [`../paper_artifacts/figures/paired_mechanism_taskA.pdf`](../paper_artifacts/figures/paired_mechanism_taskA.pdf)
- [`../paper_artifacts/figures/paired_mechanism_taskA_data/`](../paper_artifacts/figures/paired_mechanism_taskA_data/)

## Reproduce

Regenerate the robustness spread visualization from the packaged campaign data:

```bash
python3 scripts/paper_figures/make_robustness_spread.py
```

Regenerate the paired mechanism figure:

```bash
PAIRED_CAMP=_paper_runs/paired_mechanism_clean_verify \
PAIRED_TASK=F31_b1_apron_a3_mid \
PAIRED_SEED=0 \
python3 scripts/paper_figures/make_paired_mechanism.py
```

## Relevant Implementation Files

| File | Role |
| --- | --- |
| [`../src/planning/planning/planners/base_planner.py`](../src/planning/planning/planners/base_planner.py) | Main planner logic. |
| [`../src/planning/planning/core/casadi_efe.py`](../src/planning/planning/core/casadi_efe.py) | Symbolic EFE objective. |
| [`../src/planning/planning/core/visibility_gp_map.py`](../src/planning/planning/core/visibility_gp_map.py) | GP map loading and querying. |
| [`../src/planning/planning/core/nogo_cost.py`](../src/planning/planning/core/nogo_cost.py) | Keep-in/no-go cost support. |

## Limitations

- The baseline is not true dead reckoning; it is an EFE planner with constant
  detector-observation covariance.
- The GP field is setup-specific.
- The optimizer is route-seed sensitive, so claims are tied to the locked
  matched-seed campaign protocol.

See planned visual media in [`demos/`](demos/).
