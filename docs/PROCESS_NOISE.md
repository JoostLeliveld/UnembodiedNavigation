# Process noise Q — locked 2026-09-14 (PROVISIONAL, pending drift verification)

    process_noise_xy    = 0.02      (sigma_v,     m/s, actuation PSD)
    process_noise_theta = 0.08      (sigma_omega, rad/s, actuation PSD)

These are the two free numbers in the planner's process noise. Everything else
about Q follows from the closed form below.

## What they are

The **power spectral densities of the assumed actuation noise** on forward speed
and yaw rate. They are NOT the encoder-noise specification and NOT fitted to
data. The belief uses them to grow covariance during prediction.

## Where Q comes from

Continuous unicycle with zero-mean Gaussian disturbances on the actuation
signals. The motion Jacobian is nilpotent (F^2 = 0), so the matrix exponential
truncates exactly and the discrete process noise integrates in closed form:

    Q_d = Q_c dt + 0.5 (F Q_c + Q_c F^T) dt^2 + (1/3) F Q_c F^T dt^3
    Q_c = L(theta) diag(sigma_v^2, sigma_omega^2) L(theta)^T

Implemented identically in NumPy (`dynamics.py`) and CasADi
(`casadi_efe.py: unicycle_process_noise_ca`). Q therefore scales with **speed**
(v^2 in the cross-track terms), **heading** (L(theta) rotates it) and **dt**
(first, second and third order).

## Why these values

The design rule is that Q must **conservatively bound** the odometry drift of
the simulated condition - overpredicting drift is the safe direction for a
filter and is the explicit intent here.

The encoder noise was raised to a low-traction warehouse condition targeting
~4% position drift per metre travelled (see
`visibility_launch_common.py: _ENCODER_NOISE_*`). Over a 4 m unobserved stretch
at 0.84 m/s that is roughly 16 cm of real cross-track error. At
sigma_omega = 0.08 the predicted cross-track growth is about 22 cm, so the
prediction bounds the drift with margin.

**History.** The values were briefly set to the camera-ready IWAI PSDs
(0.012 / 0.05), which matched the measured along-track drift exactly at the OLD
encoder-noise level. Raising the encoder noise made those values non-conservative,
so they were raised in proportion. The superseded originals (0.01 / 0.02) made Q
nearly isotropic, contradicting the observed ordering that cross-track drift
exceeds along-track.

## PROVISIONAL — what is still owed

The ~4% drift target is the INTENT of the encoder-noise setting; the realised
drift has **not been measured** at that setting. Verify by running a few drives
and measuring odometry-minus-ground-truth in the GT heading frame over windows,
then confirm the predicted growth still exceeds it. Adjust once and remove this
section.

Do NOT predict the drift analytically from the encoder parameters. That was
attempted on 2026-09-14 and was 4.5x too high on along-track and 22% too low on
cross-track: the slip is multiplicative and AR(1)-correlated, and the EKF
predicts from the same corrupted velocity, so part of a constant scale error
cancels in a way a per-step formula does not capture.

## The lock

- Defaults set in `src/experiments/experiments/core/visibility_launch_common.py`
  and `src/planning/planning/nodes/unicycle_planner_node.py`.
- `UnicyclePlannerBase.__init__` raises a `RuntimeWarning` if either value is
  overridden, naming this file.

Changing either number is a method change. Record the reason here first.
