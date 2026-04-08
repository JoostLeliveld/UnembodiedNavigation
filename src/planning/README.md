# `planning`

This package contains the planner runtime and the planner math. It is the core method package for the thesis milestone.

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
| [`planning/core/visibility_gp_map.py`](planning/core/visibility_gp_map.py) | loads the empirical GP visibility artifact |
| [`planning/core/dynamics.py`](planning/core/dynamics.py) | unicycle dynamics helpers |
| [`planning/core/rollout.py`](planning/core/rollout.py) | rollout helpers |
| [`planning/core/efe_utils.py`](planning/core/efe_utils.py) | EFE-related math utilities |

## Support Files

| File | Role |
| --- | --- |
| `planning/core/casadi_efe.py` | CasADi support for the cleaned ET1/ET2 notebook-style planner path |
| `planning/core/gp_visibility_helpers.py` | shared lightweight GP math used by the empirical visibility fit/load path |
| `planning/core/nogo_cost.py` | obstacle/no-go support cost |

## What To Read First

1. `planning/nodes/unicycle_planner_node.py`
2. `planning/planners/base_planner.py`
3. `planning/core/visibility_gp_map.py`
4. `planning/core/dynamics.py`
5. `planning/core/rollout.py`

## Implemented Now

- ET1-based `efe1` as the main thesis path
- `visibility_unaware_baseline` under the same planner wrapper
- retained `efe2`, `efer`, and `mpc`
- planner-side loading of a fixed empirical visibility field

## Caveats

- The baseline is not true dead reckoning.
- `base_planner.py` is still monolithic.
- ET1 is the primary claim path; `efe2`, `efer`, and `mpc` are retained but secondary.
- The GP artifact is setup-specific and changes observation modeling indirectly rather than encoding a full geometric theory.
