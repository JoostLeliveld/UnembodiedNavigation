# F35 vs F43 - Why F35 Looked Good

## Files

- F35 dashboard: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F35/F35_dashboard.png`
- F43 dashboard: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F43/F43_dashboard.png`
- F35 logs: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f35_b1_route_choice_v1`
- F43 logs: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f43_b1_timing_architecture_v2`

## F35 Outcome

F35 had the desired qualitative result:

- C1 constant-R: collision, path `4.34 m`, min goal distance `0.859 m`.
- C2 GP-visibility: goal reached, path `7.00 m`, min goal distance `0.056 m`.

The important evidence is not only that C2 reached the goal. It is that C1 lost localization while C2 maintained it:

- F35 C1 detection fraction: `0.735`.
- F35 C1 pixel pose age: mean `1.22 s`, max `5.28 s`.
- F35 C1 localization error: mean `0.477 m`, max `1.783 m`.
- F35 C2 detection fraction: `1.000`.
- F35 C2 pixel pose age: mean `0.615 s`, max `0.990 s`.
- F35 C2 localization error: mean `0.215 m`, max `0.550 m`.

This is the paper story in miniature: the non-visibility-aware route loses visual localization and crashes, while the visibility-aware route stays observable enough to finish.

## Why F43 Did Not Reproduce It

F43 kept good perception for both methods:

- F43 C1 detection fraction: `0.990`.
- F43 C2 detection fraction: `1.000`.
- F43 C1 localization error: mean `0.122 m`.
- F43 C2 localization error: mean `0.127 m`.

So F43 did not recreate the F35 failure mechanism. It became mostly an execution / collision / route-following problem, not a localization-failure contrast.

F43 also ran slower:

- F43 C1 local solve mean: `3454 ms` in dashboard summary.
- F43 C2 local solve mean: `4335 ms` in dashboard summary.
- F43 C2 global solve: `66.1 s`.

The new execution diagnostics show that command tapes do advance when they exist:

- F43 C1 `exec_control_index`: mean `6.67`, max `16`.
- F43 C2 `exec_control_index`: mean `6.83`, max `17`.

That means the earlier F42 suspicion that the tape was constantly reset was too strong. The true issue is more subtle: solves are slow and sometimes sparse, but the active command tape does execute across multiple controls.

## Interpretation

F35 worked because the chosen C1 path entered a region where detector updates dropped and localization drift grew, while C2 stayed in a path with uninterrupted detections. F43 failed because both methods maintained good localization, so the run no longer tested the intended visibility/localization failure contrast.

The next experiment should not just tune speed or ambiguity weight. It should deliberately preserve the F35 mechanism:

1. C1 should have a plausible shorter path with poor detection coverage.
2. C2 should have a longer route with reliable detection coverage.
3. Both routes should be physically traversable.
4. Collision should arise from localization/control consequences, not from a hidden route-forcing waypoint.
5. Runtime should avoid multi-second local EFE solves in the inner loop, because they make behavior sensitive to timing.

