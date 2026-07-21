# `planning`

This package contains the planner runtime and the planner math. It is the core method package for the thesis milestone.

The demo role of this folder is to show how a learned observation-quality field
changes route choice without changing the known obstacle map.

For the current thesis-facing implementation, the planning question is:

\[
\hat s_t = [\hat x_t,\hat y_t]^\top \mapsto p_{\mathrm{vis}}(\hat x_t,\hat y_t) \mapsto R_{\mathrm{plan}}.
\]

For the current implementation, `R_{\mathrm{plan}}` is built through
visibility-aware precision blending rather than a simple linear covariance mix.
That is the central mechanism implemented in this package.

![GP reliability and induced covariance](../../paper_artifacts/figures/gp_pipeline_aws.png)

## Why This Folder Exists

This package answers the main research question: how does planning change when future observation quality depends on robot state?

## Inputs And Outputs

- **Inputs**
  - `/state/bev`
  - `/goal_bev`
  - pixel observations and detection diagnostics
  - fixed GP visibility artifact
- **Outputs**
  - `/cmd_vel`
  - planner belief
  - plan preview
  - planner metrics and diagnostics

## Central Files

| File | Role |
| --- | --- |
| [`planning/nodes/unicycle_planner_node.py`](planning/nodes/unicycle_planner_node.py) | ROS planner wrapper and runtime correction/planning loop |
| [`planning/nodes/efe_agent_node.py`](planning/nodes/efe_agent_node.py) | thin runtime entry point used by launches |
| [`planning/planners/base_planner.py`](planning/planners/base_planner.py) | main planner logic |
| [`planning/core/casadi_efe.py`](planning/core/casadi_efe.py) | symbolic ET1/ET2 EFE objective construction |
| [`planning/core/visibility_gp_map.py`](planning/core/visibility_gp_map.py) | loads the empirical GP visibility artifact |
| [`planning/core/dynamics.py`](planning/core/dynamics.py) | unicycle dynamics helpers |
| [`planning/core/rollout.py`](planning/core/rollout.py) | rollout helpers |
| [`planning/core/efe_utils.py`](planning/core/efe_utils.py) | EFE-related utility math |

## Support Files

| File | Role |
| --- | --- |
| `planning/core/nogo_cost.py` | obstacle/no-go support cost |

## What To Read First

1. `planning/nodes/unicycle_planner_node.py`
2. `planning/planners/base_planner.py`
3. `planning/core/casadi_efe.py`
4. `planning/core/visibility_gp_map.py`
5. `planning/core/dynamics.py`
6. `planning/core/rollout.py`

## Implemented Now

- ET1-based `visibility_aware_efe` as the main thesis path
- `constant_R_efe` under the same planner wrapper
- planner-side loading of a fixed empirical visibility field

## Demonstrated Artifact

The paper-facing GP artifact is
[`../../paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz`](../../paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz).
Its fit summary records 238 training targets with raw detector-score targets
from `0.0016` to `0.9232` and mean target `0.5498`.

In the locked comparison, C1 uses a constant observation covariance. C2 uses the
GP-derived reliability to build state-dependent camera covariance inside the EFE
horizon. The route-choice summary is visualized in
[`../../paper_artifacts/figures/paper_snapshot/robustness_spread.png`](../../paper_artifacts/figures/paper_snapshot/robustness_spread.png).

## Caveats

- The baseline is not true dead reckoning.
- `base_planner.py` is still monolithic.
- ET1 is the primary claim path; broader planner variants are diagnostic, not paper conditions.
- The GP artifact is setup-specific and changes observation modeling indirectly rather than encoding a full geometric theory.
