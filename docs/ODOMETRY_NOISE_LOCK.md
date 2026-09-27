# Odometry, noise, Q and updates — locked 2026-09-27

Authority for the simulated odometry, the process noise and the estimator update.
Heading mode decided: coupled. Supersedes the 2026-09-14 process-noise lock (constant 0.02/0.08 PSDs), which is kept
only as the legacy `process_noise_model: constant_psd`. Changing anything here is a
method change: record it in METHOD.md first.

## 1. World: simulated odometry
- Wheel odometry is used. The encoder starts from the TRUE body velocity
  (`/ground_truth_tf`, decimated to 50 Hz), not the Gazebo DiffDrive wheel odometry,
  which carries Gazebo's own undescribed wheel-floor slip.
- The true-motion encoder initializes and integrates `/odom_noisy` in the map frame, so its
  yaw already includes the spawn yaw and receives no additional launcher offset. Raw `/odom`
  begins at yaw zero and retains the spawn-yaw transform. This is test-guarded; applying both
  transforms was caught by the final pre-campaign qualification on task A.
- Declared encoder noise, set explicitly in both campaign templates (test-guarded):
  multiplicative AR(1) slip, alpha 0.80, std 0.125 (v) / 0.075 (w), mean 0;
  additive white per 50 Hz sample, 0.004 m/s (v) / 0.050 rad/s (w).
- Systematic wheel errors, fixed for the robot (not per seed), from UMBmark on a TRC
  LabMate (Borenstein & Feng): diameter ratio D_R/D_L = 1.00121, wheelbase 337.2/340 mm;
  wheel separation 0.44 m. w_enc = rho w - (e/b) v, v_enc = v - (e b/4) w.
- Command (actuation) noise stays on the plant; the encoders measure its effect.

## 2. Process noise Q: SET, not fitted
- `process_noise_model: encoder` (default). The IWAI continuous-time unicycle
  derivation (exact nilpotent integration) with input-dependent PSDs equal to the
  white-noise equivalent of the injected encoder noise:
  white std s -> s^2 D; AR(1) slip on rate r -> s^2 D (1+alpha)/(1-alpha) r^2;
  systematic bias rate -> bias^2 T with T = 12 s (the longest camera-free stretch).
- Declared assumptions: slip treated as white (correlation 0.1 s < filter step);
  systematic bias as noise with the same variance at T.
- Command noise is not in Q. Planner and estimator use the same Q: the planner predicts
  the estimator's belief, which grows with the encoder noise.
- Check: `tests/planning/test_encoder_process_noise.py` (Monte Carlo of the injected
  noise, 95 % region covers 0.965-0.995 at 1/3/10 s; NumPy = CasADi; templates = model).

## 3. Estimator update
- EKF, fused camera position measurement with the spatial R from the survey, NIS gate
  9.21, Joseph form, one committed covariance (the record's heading variance equals the
  reported one).
- In camera_xy_only the heading variance is the process-noise heading PSD integrated
  along the odometry since start.
- HEADING MODE: `coupled` (decided 2026-09-27). Standard EKF: the position measurement
  corrects heading through its correlation with position, from the declared task heading
  prior. This is what the thesis text describes. Known cost, reported as a limitation: at
  coverage edges the camera error drifts with the view and bends the heading (offline on
  the new-world pilot: roughly double the cross-track error after coverage ends compared
  with heading from odometry). `camera_xy_only` remains available and was the better
  estimator after coverage loss; it is not the method.

## 4. Accepted limitations (reported, not modelled)
- With an honest Q the position belief is overconfident (pilot: cross-track 95 %
  coverage 0.77 while seen, 0.64-0.71 after): R from a static survey is overconfident and
  camera errors are correlated frame to frame and drift with the view.
- Heading error, fed by drifting camera errors at coverage edges, contributes to failures
  after coverage loss. Mitigations: a gyro/IMU or better odometry; on the perception side a
  filter over each camera's correlated error (per-camera bias state) or a view-dependent R.

## Provenance
Commits 10aa3a29, 02f0a5a3, 38044a06, 0e65d8c3, 8e8ee77b. Analyses in
`pipeline/process_noise/`. Pilot: logs/thesis/revisions/new_world_pilot (seed 91600).
