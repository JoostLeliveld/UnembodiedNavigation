# F30 - R01 Gazebo Timing Smoke

Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F30/F30_r01_timing_smoke.png`
PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F30/F30_r01_timing_smoke.pdf`
Stats: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F30/F30_r01_timing_smoke.csv`

## Diagnosis

- Gazebo spawn/init is clean for both conditions: truth start and yaw errors are essentially zero.
- Both C1 and C2 reached the visible goal with command and encoder noise active.
- The stop-on-exhausted-plan guard plus less overconfident pixel noise prevents the earlier obstacle penetration.
- The remaining weakness is timing: local solves are still around 1.7-2.6 s for the median/p90 range.
- This is a valid smoke pass, not yet evidence of a GP advantage, because this R01 endpoint is largely visible and `p_vis_plan` stays high.

## Summary Table

| condition | outcome | path [m] | min obstacle [m] | mean state error [m] | median solve [ms] | p90 solve [ms] | optimizer success |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | goal_reached | 7.00 | 0.319 | 0.240 | 1754 | 2421 | 55% |
| C2 | goal_reached | 6.64 | 0.348 | 0.236 | 1676 | 2588 | 55% |

## Interpretation

F30 shows that the runtime stack can complete R01 in Gazebo when local execution is made more conservative. The next experiment should keep these runtime safeguards and move back to a route where C2 has a real learned-observation-reliability reason to differ from C1.

## Next Lock-In Decision

Keep the F30 runtime hygiene as the current Gazebo smoke baseline, but do not claim a visibility-aware advantage from R01 alone. Next, test a task where the final goal is visible but the short route spends enough time in a weak-observation region for C2 to prefer a safer visible route.
