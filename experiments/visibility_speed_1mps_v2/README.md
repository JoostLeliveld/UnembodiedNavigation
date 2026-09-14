# 1 m/s visibility commissioning — v2

This is the requested new operating configuration. It does not modify the frozen
`visibility_pilot_v1` or the previous navigation campaigns.

`speed_profile.json` is the shared configuration. The planner and tracker permit
1 m/s and the actuator clips delivered linear commands at 1 m/s. `sweep.launch.py`,
`planner_worker.py`, `prelaunch_plan.py` and `navigation_driver.py` reject missing
or inconsistent profile evidence. Global plans are regenerated and bound to the
profile hash. No 0.16 m/s plan is reused.

The first speed candidate increased spatial preview to preserve preview time.
Its live Uniform attempt stopped at the uncertainty-clearance guard. It remains
under `../visibility_speed_1mps_v1` with its original source and data.

V2 retains 0.20 m tracker lookahead, 0.65 m local-goal distance and 1.50 m forward
projection search. The tracker target-distance gain is 5/s, allowing 1 m/s on a
straight segment. Curvature limits linear speed to preserve the requested turn
within the existing 0.4 rad/s angular bound. This controller change is explicit
and common to all three arms. Goal slowdown, 5 s local horizon, optimizer budget,
physics step, RTF, camera/NN models and devices, estimator, uncertainty guard and
success criteria remain unchanged. Global horizon is ceil((longest route / 1 m/s
+ 65 s) / 0.25 s), giving 383 steps for this task.

Commands (from a ROS Humble environment with this repository's install sourced):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python3 -m pytest -q test_speed_profile.py test_pilot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python3 prepare.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python3 campaign_runner.py Uniform
```

Preparation and acquisition refuse to overwrite existing artifacts. The runner
accepts arm names for staged commissioning. Do not run the remaining arms after
a failed integration gate merely to label the configuration validated.

Artifacts are under
`/home/joostleliveld/Thesis/2026-09-07/i-w/outputs/bayesian_navigation/visibility_speed_1mps_v2_reporting`.
Runs use exact IDs from `pilot.json` in the adjacent `navigation` directory.
Read the validation report and repository metrics registry before citing results.
Static preflight alone does not approve this configuration for the final campaign.

## Recorded validation outcome (2026-09-08)

Nine tests and all three preflights passed. Uniform passed the full navigation
checks; IWAI timed out near the first bend following starting-belief-footprint
rejections. Commissioned was not launched after the failed gate. This candidate
is not final-campaign-ready. Keep both v1 and v2 failures in the evidence record.
The canonical selection, `validation.json` and `timing.json` are in the reporting
directory above. Repository `docs/visibility_speed_commissioning.md` and the
localization metrics registry define the claim boundary. Evaluate from the repo
with `python3 experiments/validate_visibility_speed.py v2`; summarize timing with
`python3 experiments/visibility_speed_timing.py v2`. Acquisition source files
are frozen and must not be edited for further tuning; create a new version.
