# F44 Route-Tracking Root-Cause Diagnostic

This figure reads the existing F34/F35/F37/F43 logs only. It is a diagnostic artifact, not a new Gazebo run.

## Files

- Figure PNG: `timing_presentation/figures/F44/F44_route_tracking_diagnosis.png`
- Figure PDF: `timing_presentation/figures/F44/F44_route_tracking_diagnosis.pdf`

## Summary Table

| Fig | Cond | Outcome | path m | min goal m | first plan m | first end | last local end | final truth | detect frac | max pixel age s | loc mean/max m | yaw state mean/p90/max rad | exec idx max |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: |
| F34 | C1 | collision | 6.93 | 1.51 | 4.68 | (1.03,1.75) | (2.93,6.67) | (3.79,0.79) | 0.46 | 15.91 | 1.00/2.36 | 0.78/2.38/3.14 | n/a |
| F34 | C2 | goal_reached | 7.15 | 0.11 | 6.07 | (1.02,1.72) | (1.03,1.73) | (0.88,1.88) | 1.00 | 0.78 | 0.22/0.58 | 0.14/0.43/0.85 | n/a |
| F35 | C1 | collision | 4.34 | 0.86 | 4.64 | (1.03,1.72) | (2.61,1.73) | (1.77,1.36) | 0.73 | 5.28 | 0.48/1.78 | 0.37/1.73/2.80 | n/a |
| F35 | C2 | goal_reached | 7.00 | 0.06 | 6.14 | (1.01,1.59) | (1.02,1.72) | (1.06,1.76) | 1.00 | 0.99 | 0.22/0.55 | 0.14/0.44/1.19 | n/a |
| F37 | C1 | collision | 4.45 | 0.78 | 4.75 | (1.04,1.73) | (2.88,-0.26) | (1.71,2.08) | 0.76 | 5.07 | 0.44/1.90 | 0.38/1.62/2.17 | n/a |
| F37 | C2 | stuck | 7.18 | 0.29 | 6.23 | (1.08,1.50) | (1.03,1.62) | (1.56,1.73) | 1.00 | 0.79 | 0.39/0.88 | 0.14/0.40/0.75 | n/a |
| F43 | C1 | collision | 1.64 | 2.25 | 4.77 | (1.03,1.74) | (3.07,0.26) | (2.39,-0.02) | 0.99 | 0.84 | 0.12/0.50 | 0.06/0.22/0.73 | 16 |
| F43 | C2 | collision | 4.21 | 3.26 | 6.08 | (1.03,1.58) | (1.02,-0.81) | (1.67,-1.44) | 1.00 | 0.86 | 0.13/0.53 | 0.45/1.82/3.10 | 17 |

## What Did The Initial Planner Choose?

The first global plans are mostly sensible. Across F34/F35/F37/F43, C1's first plan is consistently the shorter upper/direct candidate, while C2's first plan is consistently the longer lower-visible sweep. That means the most confusing path shapes are not primarily caused by the initial route-choice optimization choosing random routes.

## Where Did Execution Diverge?

The divergence appears during local execution/replanning. In successful C2 runs such as F35, the last local endpoint remains near the true goal and the executed path follows the lower visible sweep. In F43, the initial C2 plan still points toward the lower visible route, but the later local endpoint collapses back near the lower aisle/apron while the truth path never progresses north. This makes F43 a tracking/runtime failure, not a clean visibility-vs-shortest-path comparison.

## Was Yaw Wrong At The Divergence?

Yaw is a plausible contributor but not the whole explanation. Current YOLO-seg runs do not use visual heading as a paper-facing measurement; the state estimator mostly uses odometry heading or held previous heading. F43 C2 shows much larger state-yaw error than F35 C2, so heading handling should be inspected before claiming the route behavior is planner-optimal.

Heading source counts:
- F34 C1: `odom_heading_fallback:239, held_previous_heading:52, unknown:42`
- F34 C2: `odom_heading_fallback:213, held_previous_heading:140, unknown:34`
- F35 C1: `odom_heading_fallback:143, unknown:35, held_previous_heading:32`
- F35 C2: `held_previous_heading:153, odom_heading_fallback:138, unknown:31`
- F37 C1: `odom_heading_fallback:130, held_previous_heading:50, unknown:41`
- F37 C2: `odom_heading_fallback:362, held_previous_heading:217, unknown:32`
- F43 C1: `held_previous_heading:139, odom_heading_fallback:54, unknown:37`
- F43 C2: `held_previous_heading:386, odom_heading_fallback:51, unknown:42`

## Was Perception Stale Or Missing?

F35 shows the desired visibility/localization story: C1 loses detections and localization quality, while C2 stays visually locked and reaches the goal. F43 does not show the same mechanism: perception availability and localization error are good for both conditions, yet both crash. Therefore F43 should be treated as a runtime/local-tracking diagnostic rather than visibility-method evidence.

## Did F43 Fail For A Different Reason Than F35?

Yes. F35 is mainly a perception/localization contrast: C1 takes the risky route and loses visual updates; C2 stays observable and succeeds. F43 is mainly a tracking/timing/collision issue: both conditions remain localized, but the local plan endpoints and executed paths disagree with the initial global route.

## Next Fix Target

The next fix should target local waypoint tracking, heading-state consistency, no-go clearance during local execution, and command/update timing. YOLO heading should be a future controlled ablation, not an immediate assumption, because the current segmentation setup provides `(x,y)` camera localization while heading comes from odometry/held previous state.

## Source Runs

- F34 C1: `logs/visibility_comparison/f34_b1_route_choice_v2/F31_b1_apron_a3_mid/C1/seed1/experiment_20260529_122054`
- F34 C2: `logs/visibility_comparison/f34_b1_route_choice_v2/F31_b1_apron_a3_mid/C2/seed1/experiment_20260529_122223`
- F35 C1: `logs/visibility_comparison/f35_b1_route_choice_v1/F31_b1_apron_a3_mid/C1/seed1/experiment_20260529_130507`
- F35 C2: `logs/visibility_comparison/f35_b1_route_choice_v1/F31_b1_apron_a3_mid/C2/seed1/experiment_20260529_130641`
- F37 C1: `logs/visibility_comparison/f37_b1_route_choice_v3/F31_b1_apron_a3_mid/C1/seed1/experiment_20260529_142846`
- F37 C2: `logs/visibility_comparison/f37_b1_route_choice_v3/F31_b1_apron_a3_mid/C2/seed1/experiment_20260529_142953`
- F43 C1: `logs/visibility_comparison/f43_b1_timing_architecture_v2/F31_b1_apron_a3_mid/C1/seed1/experiment_20260601_140554`
- F43 C2: `logs/visibility_comparison/f43_b1_timing_architecture_v2/F31_b1_apron_a3_mid/C2/seed1/experiment_20260601_140725`
