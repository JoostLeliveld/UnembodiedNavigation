# Perception-side filter cascade

## The question this study answers

Does filtering the bounding box over time, before the ground projection, turn imperfect
per-frame detections into a better localization measurement — and does the covariance that
filter reports describe the error that is actually left?

The proposed cascade is:

```text
image -> YOLO box -> NN box correction -> temporal box filter -> ground projection
      -> (z, R) per camera -> one robot EKF
```

This folder tests the temporal-box-filter link in that chain. It does not test the NN
correction, which is a separate stage, and it does not touch the planner.

## Why the existing captures cannot answer it

The error in a camera reading splits into

```text
e_t = b(s_t) + q_t
```

`b(s)` is the smooth perception/projection bias. On the frozen 0.67 m characterization grid
it is the whole story: the error field's structure function rises from 8.4 cm at 0.5 m
separation to 20.2 cm at 6 m against a field spread of 30.3 cm, which puts the correlation
length at roughly 3–5 m. A temporal filter cannot average that down, because every frame in
any usable window shares nearly the same value of it.

`q_t` is the fast component. The simulator has no motion blur and no timing jitter, and the
detector is deterministic — 40 repeats of one static pose return a bit-identical box. But a
moving robot renders to a different pixel raster at every step, so the detector response
changes slightly from sample to sample. That component is what a temporal filter could
legitimately suppress.

The frozen grid is sampled at 0.67 m, which is far coarser than the scale on which `q`
varies, so on that grid `q` cannot be separated from the curvature of `b`. Estimates made
from it are upper bounds only. This study captures at 4 cm instead.

## Stage 1 — measure `q`

`make_dense_line_poses.py` writes straight lines sampled at 4 cm. `probe_line_validity.py`
first reports the longest collision-valid line at each centre, using the same pose filter
the capture enforces, so the capture's all-or-nothing pose-file validation cannot reject the
result. Centres are positions the frozen grid already captured, spread along the camera-range
axis from about 2 m to about 16 m, because range is the variable that separates the
near-range consistency tail.

`separate_bias_and_fast_noise.py` fits a low-order polynomial in arc length along each line,
treats the fit as `b` and the residual as `q`, and reports both in pixels (where a box filter
would act) and in centimetres on the floor (what the EKF receives). Two guards decide whether
the split is trustworthy:

- **Order sweep.** If the reported `q` keeps shrinking as the polynomial order rises, the fit
  is absorbing real signal and the number means nothing.
- **Lag-1 autocorrelation of the residual.** A genuinely fast, near-white `q` sits near zero.
  A residual that is still strongly correlated means `b` was not removed and the split failed.

## Stage 2 — compare the arms

`compare_perception_filters.py` runs five arms on the same dense-line detections:

| arm | what it is |
|---|---|
| `A_raw` | per-frame observation, no temporal processing — the current pipeline |
| `B_kf` | constant-velocity Kalman filter on the box-bottom pixel |
| `C_robust_kf` | same filter, but a surprising observation inflates its own covariance instead of being discarded (soft rejection) |
| `D_smoother` | fixed-lag smoother over the following few observations, as a Rauch-Tung-Striebel backward pass |
| `E_static_R` | no temporal filter; per-frame observation paired with an offline-measured covariance |

**Arm E is the arm to beat.** It is the alternative in which the covariance comes from a
commissioned per-camera field rather than from a filter. Without it, arms B–D could only be
compared against bare per-frame noise, which flatters them.

The filter runs in pixel space; its covariance reaches the floor through the local Jacobian
of the same `pixel_to_world` homography the frozen interpretations use, so arm A reproduces
the existing `raw` reading and nothing is re-calibrated here.

Two details are worth stating because they decide whether the arms mean anything:

- **Soft rejection is not an arbitrary curve.** Arm C scales a surprising observation's own
  noise by the factor that brings its normalised innovation back to the gate, which is the
  Huber-style rescaling used for heavy-tailed observations. Below the gate nothing changes,
  so a well-behaved reading is untouched. Resistance grows with how wrong the reading is: a
  mild disagreement is absorbed, a wild one is nearly ignored, and neither is discarded.
- **The smoother is a real smoother.** Arm D runs the Rauch-Tung-Striebel backward
  recursion. Back-propagating the filtered mean with the motion model instead would leave
  the covariance at its filtered value, so the arm would claim the wrong uncertainty and
  the calibration comparison — the whole decision variable — would be meaningless.

The estimator machinery is checked against inputs whose answer is known, in
`tests/test_perception_filter_cascade.py`. On white noise the arms order themselves
raw > filtered > smoothed and every one reports a mean normalised squared error near 2.0.
That control matters: if the arms come out badly calibrated on the real captures, the cause
is the error structure and not a broken filter.

## Decision rule, written before the results

Calibration is the decision variable, not accuracy. A filter that merely widens its ellipse
passes any consistency test and is useless, so accuracy and honesty are always read together.

- A temporal arm **earns its place** only if it beats arm E on median error *without* getting
  worse on `mean_NSE` and on the share of readings beyond 4 sigma.
- `mean_NSE` of 2.0 is consistent for a two-dimensional measurement. Above 2.0 is
  overconfident, below is conservative.
- The pre-registered prediction is that arms B–D improve median error slightly and get
  **worse** on calibration, because the filter's posterior shrinks like `1/N` while the
  shared `b(s)` in the window does not. If that is what happens, the finding is that
  cascaded box filtering is not the right instrument for this error structure, and the
  covariance belongs in a pose-indexed field instead.
- The gain a filter can win is bounded by `q`, so stage 1 sets the ceiling before stage 2 is
  interpreted. A filter cannot beat a component that is not there.

## Running it

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export ROS_DOMAIN_ID=82 IGN_PARTITION=<unique token>

# 1. longest valid line per centre
python3 experiments/perception_filter_cascade/probe_line_validity.py \
  --out experiments/perception_filter_cascade/line_probe.json

# 2. pose file at the probed spans
python3 experiments/perception_filter_cascade/make_dense_line_poses.py \
  --out experiments/perception_filter_cascade/dense_lines.json \
  --auto-span experiments/perception_filter_cascade/line_probe.json

# 3. capture (needs Gazebo up with all five cameras bridged)
python3 experiments/camera_observation_characterization/capture_bbox_grid.py \
  --world warehouse_v2.world.sdf \
  --out logs/perception_datasets/warehouse_v2_dense_lines_20260902 \
  --pose-file experiments/perception_filter_cascade/dense_lines.json

# 4. frozen detector, then the same three interpretations as the grid study
python3 experiments/camera_observation_characterization/run_bbox_detector.py \
  --capture logs/perception_datasets/warehouse_v2_dense_lines_20260902 --device 0
python3 experiments/camera_observation_characterization/derive_interpretations.py \
  --capture logs/perception_datasets/warehouse_v2_dense_lines_20260902

# 5. the two analyses
python3 experiments/perception_filter_cascade/separate_bias_and_fast_noise.py \
  --capture logs/perception_datasets/warehouse_v2_dense_lines_20260902 \
  --out-dir logs/studies/perception_filter_cascade_20260902
python3 experiments/perception_filter_cascade/compare_perception_filters.py \
  --capture logs/perception_datasets/warehouse_v2_dense_lines_20260902 \
  --out-dir logs/studies/perception_filter_cascade_20260902 \
  --bq-split logs/studies/perception_filter_cascade_20260902/bq_split.json

# 6. figures, then the verdict against the pre-registered rule
python3 experiments/perception_filter_cascade/plot_cascade.py \
  --study-dir logs/studies/perception_filter_cascade_20260902
python3 experiments/perception_filter_cascade/write_results.py \
  --study-dir logs/studies/perception_filter_cascade_20260902
```

Gazebo must be launched with every camera bridged, otherwise only camera A appears:

```bash
ros2 launch sim bringup_sim.launch.py world:=warehouse_v2.world.sdf \
  world_name:=warehouse_v2 headless:=true use_lidar:=false bridge_scan:=false \
  bridge_camera_a:=true bridge_camera_b:=true bridge_camera_c:=true \
  bridge_camera_d:=true bridge_camera_e:=true
```

## Outputs

Everything lands in `logs/studies/perception_filter_cascade_20260902/`:

- `bq_split.json` — the `b`/`q` split, the order sweep and the autocorrelation guard
- `line_residuals.csv` — per-sample trend and residual, for plotting
- `arm_comparison.json` — accuracy and calibration per arm, overall and per camera
- `figures/01_smooth_versus_fast.png` — how much of the error a filter can even reach
- `figures/02_is_the_leftover_fast.png` — whether the leftover is white or bias in disguise
- `figures/03_do_the_arms_earn_their_place.png` — accuracy and honesty side by side
- `RESULTS.md` — the verdict, generated by `write_results.py` applying the rule above

`write_results.py` is the only thing that turns numbers into a verdict, so the conclusion
cannot drift to fit the outcome. `tests/test_perception_filter_cascade.py` pins the rule
itself, and the figure headlines are derived from the data rather than written in advance.

## Scope

Commanded pose is evaluation-only, as everywhere else in this repo. The detector and the
projection are the frozen ones; this study adds no calibration of its own. The captures are
static placements sampled densely along a path, which isolates the raster effect but does not
reproduce closed-loop timing — so a positive result here is a necessary condition for the
cascade, not a demonstration of it in the live loop.
