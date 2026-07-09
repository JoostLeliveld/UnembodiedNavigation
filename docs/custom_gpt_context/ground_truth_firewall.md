# Ground-Truth Firewall

The thesis must remain realistic for a real warehouse. Gazebo ground truth is
useful, but it must stay behind a firewall.

## Rule

Ground truth can judge the method, but it cannot be part of the method.

## Allowed Uses

Ground truth may be used for:

- evaluation metrics,
- diagnostics,
- calibration checks,
- controlled ablations,
- debugging odometry/state drift,
- detector dataset generation or YOLO training if clearly disclosed,
- offline audits of whether a result was real or an artifact.

## Forbidden Uses

Ground truth must not be used as:

- online robot state,
- planner input,
- reliability-learning label in the contribution,
- hidden correction source,
- route-choice input,
- substitute for odometry/belief in runtime,
- evidence for an operational signal that would not exist in a real warehouse.

## Required Labels

Every plot/table should identify the source:

| Label | Meaning |
| --- | --- |
| GT | Gazebo ground truth, evaluation only. |
| ODOM | wheel/robot odometry, operationally available. |
| BELIEF | planner/filter belief, operationally available if produced by the system. |
| STATE | current state estimate, operationally available but may be noisy. |
| PIXEL | camera detection or pixel-derived measurement. |
| MODEL | analytic prediction or prior, not measured evidence. |

## Good Wording

Use:

> Ground truth is used to evaluate whether the robot reached the goal and
> whether it breached geometry. It is not used by the planner or reliability
> learner.

Use:

> The reliability GP is trained/refined from detector outcomes and operational
> belief/state estimates. Ground truth is reserved for scoring the result.

Avoid:

> The robot learns visibility from ground truth.

Avoid:

> The planner knows where it really is.

Avoid:

> The GP was validated because it matched Gazebo truth, unless the statement is
> explicitly framed as an evaluation-only diagnostic.

