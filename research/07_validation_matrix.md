# Validation matrix

| Stage | Question | Inputs held fixed | Primary result | Promotion gate |
|---|---|---|---|---|
| Offline prediction | Does a source predict usable observations? | Splits, labels, candidate poses, detector | Brier score and calibration; PR-AUC secondary | Beat or explain the simple FOV/range baseline on held-out routes. |
| Failure audit | Where does it fail? | Same split and budget | Error by occlusion, range, support, and layout shift | At least one documented failure case and fallback. |
| Offline planning | Does it alter expected belief or route ranking? | Planner, prior belief, route library, weights | Route discrimination and expected terminal trace | Nontrivial paired route difference on a prespecified task. |
| Closed-loop navigation | Does the changed belief improve action? | Controller, seeds, starts/goals, runtime | Clean-goal rate; breach and calibration diagnostics | Prespecified campaign complete or documented null. |
| Deployment | Is the gain worth its assumptions? | Reporting protocol | Samples/time/runtime/transfer matrix | No unsupported operational input or hidden oracle. |

## Reliability-source arms

Run constant, FOV/range, sensed-depth/raycast, GP, and hybrid. The DL challenger is admitted
only after it passes feature-availability, probability-calibration, held-out prediction,
and route-discrimination gates.

## Camera-management arms

After freezing the winning fields, compare nearest camera, maximum availability, achievable
precision, hysteretic selection, and conservative fusion. This is a separate experiment;
the estimator is not refitted for each policy.
