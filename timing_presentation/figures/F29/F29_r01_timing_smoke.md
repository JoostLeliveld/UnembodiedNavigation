# F29 - R01 Gazebo Timing Smoke

Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F29/F29_r01_timing_smoke.png`
PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F29/F29_r01_timing_smoke.pdf`
Stats: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F29/F29_r01_timing_smoke.csv`

## Diagnosis

- Gazebo spawn/init is now clean for both conditions: truth start and yaw errors are essentially zero.
- YOLO availability is high, so this run is not mainly a detector-dropout failure.
- Both C1 and C2 still end in geometry obstacle penetration.
- The local tracker is the bottleneck: most nonzero local solves take about 1.1-2.1 s while the local loop asks for 2 Hz replanning.
- Most failed local replans hit `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`; the robot then executes solver-returned controls from non-converged local plans.

## Interpretation

This points away from spawn, world-frame, or GP choice as the immediate blocker. The global route can be generated, but the closed-loop local EFE tracker is too slow and too often non-converged for tight warehouse execution.

## Next Lock-In Decision

Decide whether the paper runtime is: (1) global EFE route choice plus a conventional local tracker, or (2) EFE for both global and local layers with a lower-rate, better-converged local planner. The current 2 Hz / H10 / 15-iteration local EFE setting is not stable enough.
