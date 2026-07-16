# 07 — Real commissioning execution

## Show 1 — `figures/01_real_routes_and_observations.png`

“These are not route sketches. The two paths were actually driven in the new
Gazebo warehouse while all four detector topics and the independent noisy
odometry stream were recorded. Each camera produced 60–62 aligned records over
the two runs; misses remain in the log as evidence.”

## Show 2 — `figures/02b_actual_gp_updates_wide.png`

“Every camera begins from its own day-zero calibration prior. We then fitted
four separate expected-kernel GPs from the records associated with that camera.
The updates are visibly local to the collected paths. The companion
`02_actual_per_camera_gp_updates.png` adds the posterior-uncertainty column;
the pooled map is only a commissioning diagnostic and does not replace the four
source-specific maps.”

## Show 3 — `figures/03_actual_gp_learning_progress.png`

“This is the update story in numbers: an early GP checkpoint after the first
traverse, then a final refit after the dedicated overlap pass. The bars show
that these fields changed because new observations arrived—not because we drew
a different prior.”

## Show 4 — `figures/04_actual_overlap_gate_C_D.png`

“The long traverse produced three truly synchronized Camera C/D pairs. Their
mean projected disagreement was 0.247 m, the largest was 0.274 m, and all
three passed the configured 0.30 m gate with no outliers. This is genuine pilot
evidence of a C/D overlap edge, not yet the 30-pair campaign threshold for a
general fusion claim.”

## Show 5 — `figures/05_actual_algorithm_execution.png`

“The entire pipeline was executed: collection, uncertainty-stamped GP fitting,
overlap validation, and hysteretic handover replay. At the configured 0.45
spatial-trust release threshold the newly fitted pilot maps released zero
corrections. That is the intended fail-safe outcome: the system deferred rather
than turning sparse, weakly trusted data into a false localization correction.”

## Evidence boundary

- Two executed pilot routes; not a route-disjoint closed-loop evaluation.
- Input covariance is a declared 0.10 m noisy-odometry floor plus alignment
  age because the current encoder-noise publisher leaves covariance entries at
  zero. It is not a calibrated covariance-estimation result.
- C/D passes the pilot agreement gate with three pairs. More independent pairs
  are required before enabling or claiming general measurement fusion.
- The policy replay demonstrates safe withholding, not localization
  improvement. Closed-loop corrections remain the next experiment.

## Transition

“The upgrade is now demonstrated as a functioning evidence pipeline. The next
campaign step is not more diagrams: it is enough repeated, route-disjoint
evidence to move from safe defer to calibrated source release and then measure
closed-loop benefit.”
