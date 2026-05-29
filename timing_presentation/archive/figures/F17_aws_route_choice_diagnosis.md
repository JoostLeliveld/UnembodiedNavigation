# F17 AWS Route-Choice Diagnosis

Generated after correcting the offline diagnostic to pass the same `keep_in`
driveable-region no-go layer used by Gazebo launches.

## Files

- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F17_aws_route_choice_diagnosis.png`
- Corrected diagnostic CSVs:
  - `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/initial_rollout_diagnostics/aws_b1_v2_goal8_w8_vmax10_keepin_v1/initial_rollout_sweep.csv`
  - `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/initial_rollout_diagnostics/aws_b1_goal8_w8_vmax10_keepin_v1/initial_rollout_sweep.csv`
  - `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/initial_rollout_diagnostics/aws_visible_cross_goal8_w8_vmax10_keepin_v1/initial_rollout_sweep.csv`

## Findings

1. The previous offline route-choice diagnostic was too optimistic because
   `efe_offline_lab.py` did not pass `nogo_mode=keep_in` and the serialized
   driveable-region geometry to `UnicyclePlannerBase`. That is now fixed.

2. One earlier cross-aisle probe is not a clean route-choice task in the
   current known driveable layer. Its goal was in A3 at `y=0.5`, but the legal
   A4-to-A3 crossing is the mid cross-aisle near `y=1.72`. A nominal "A4 then
   cross" route at the goal latitude cuts through rack/non-driveable geometry.

3. The mid-cross B1 family was geometrically cleaner because both the direct A4
   route and A3-detour route could connect through the mid cross-aisle. It has
   since been superseded by the single active diagnostic
   `B1_apron_a4_to_uppermid_a3`.

4. The current AWS GP does not make the A3 detour cheaper than the direct/mid
   route for B1. Along the mid-cross goal latitude, both A4 and R4-adjacent
   regions are low-reliability in the planner-facing map, while A3 is only
   moderately reliable. Therefore the task is stable, but not yet a clean
   "visible detour wins" demonstration.

5. The Gazebo C2 smoke run on the cross task failed mainly because H120 solve
   time was too slow relative to `v_max=1.0`, causing a near-open-loop execution
   segment. This is a runtime/replanning cadence issue, not evidence that the
   visibility objective is wrong.

## Recommendation

Do not run more Gazebo on the earlier cross-aisle probe as paper evidence. Use
it only as historical diagnostic context.

Superseded task cleanup:

- The active AWS route-choice diagnostic is now
  `B1_apron_a4_to_uppermid_a3`, start `(3.20, -1.00)`, yaw `0.0` east,
  goal `(1.00, 1.75)`.
- The active diagnostic plot for that cleaned task is
  `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F18_pick_east_plan_alternatives.png`.

For the next serious AWS iteration:

- keep the mid-cross B1 geometry family;
- adjust world/GP geometry so A4 is distinctly lower reliability while A3 and
  the mid-cross return are visibly safer;
- keep the final goal visible;
- consider a long first solve followed by shorter receding-horizon replans:
  the first solve may use H120/H200 to expose the route-scale visibility
  tradeoff, while subsequent solves use a smaller horizon to keep closed-loop
  timing realistic. This is not mission waypoint forcing if it remains
  condition-neutral and only acts as optimizer warm-start / basin selection.
- rerun offline initial-rollout diagnostics first;
- only then run one Gazebo smoke with a solve-time-aware speed setting.

Until that is done, the compact benchmark remains the paper-facing result and
AWS remains exploratory.
