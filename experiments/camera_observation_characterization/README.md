# Camera observation characterization

One frozen sensor is studied here: the warehouse YOLO detector's bounding box. Every camera
sees the same commanded robot pose before it moves again. Raw box/IPM, fixed offset, and
analytic hull are derived later from that same box; they are not separate captures or
detectors.

## Capture

The capture is a controlled grid over the drivable warehouse:

```text
386 floor positions x 8 headings x 5 cameras x 1 shared robot placement
```

The frozen 2026-08-31 capture is complete at
`logs/perception_datasets/warehouse_v2_bbox_characterization_20260831`: 3,088 robot poses,
15,440 camera opportunities, and zero failed five-camera batches. The frozen detector returns
6,412 boxes (41.5% of all camera opportunities) before any post-detection admission gate.

`capture_bbox_grid.py` teleports once, waits for a fresh RGB frame from all five cameras, and
writes one row per camera. Frames are retained even when the robot is occluded or outside a
camera's useful view, because those frames are required to measure detector misses and false
positives. Commanded pose is evaluation-only. Semantic labels are deliberately absent from
the main capture; `--with-semantic` exists only for a small diagnostic capture.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 experiments/camera_observation_characterization/capture_bbox_grid.py \
  --world warehouse_v2.world.sdf \
  --out logs/perception_datasets/warehouse_v2_bbox_characterization_20260831 \
  --sample-nx 33 --sample-ny 28 --yaw-samples 8
```

Use `--plan-only` before launching Gazebo. Use `--max-poses` only for a transport pilot; a
limited capture is marked diagnostic in its manifest.

For fixed-pose repeat experiments only, `--rgb-noise-stddev-dn` applies seeded independent
Gaussian read noise after ROS image transport and before hashing, saving, and YOLO inference.
The value is in 8-bit digital-number units and the manifest records the level and seed. This
isolates detector sensitivity to camera read noise; it does not represent scene, lighting,
pose, calibration, or renderer variation and must not be described as the full camera noise.

After the capture is complete, run the frozen detector once and derive all three
interpretations from its selected bounding box:

```bash
python3 experiments/camera_observation_characterization/run_bbox_detector.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831 \
  --device 0

python3 experiments/camera_observation_characterization/derive_interpretations.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831

python3 experiments/camera_observation_characterization/fit_bias_updates.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831

python3 experiments/camera_observation_characterization/plot_characterization.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831

python3 experiments/camera_observation_characterization/plot_error_fields_by_correction.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831
```

Pass `--overwrite` only when intentionally refreshing this script's named PNGs in an existing
figure directory.

The figures are written to `logs/studies/camera_observation_characterization_20260831/`,
whose folders are one argument read in number order — see its own `README.md` and
`00_what_the_data_is/README.md`. The camera and heading audits expand a numbered argument into
explicit correction/camera/heading subfolders, so always name those figures with their folder.
No figure silently removes a detector return.

| folder | question | script |
|---|---|---|
| `00_what_the_data_is/` | what was captured, and what can it answer? | `plot_characterization.py` |
| `01_where_cameras_see/` | where does a camera return a box, and how sure is it? | `plot_characterization.py` |
| `02_the_error/` | how wrong is the reading, in which direction? | `plot_error_fields_by_correction.py`, `plot_bias_updates.py` |
| `03_why_the_error/` | does it depend on heading, and by how much? | `plot_heading_conditioned_bias.py` |
| `04_why_readings_fail/` | **why** is it wrong, and why is camera D worst? | `plot_failure_anatomy.py`, `plot_failure_gallery.py` |
| `05_the_fixes/` | what does each correction do on unseen floor? | `plot_bias_updates.py` |
| `06_what_a_gate_costs/` | what does refusing readings buy and cost? | `plot_gate_sensitivity.py` |
| `07_along_a_virtual_route/` | what bias would a traverse meet? | `plot_route_bias_profile.py` |
| `08_on_a_real_drive/` | what happens on a moving robot? | `plot_real_run_bias.py` |
| `09_learned_fixes_replayed/` | what would the learned fixes have done there? | `replay_learned_on_actual_run.py` |
| `10_learning_R/` | how far does a learned covariance get, and where does it break? | `learn_measurement_covariance.py`, `plot_why_r_fitting_fails.py` |
| `11_a_better_gate/` | can the missing variable be recovered from the box? | `plot_box_shape_gate.py` |

Figure `01` carries two rows: detection coverage per floor position, and the median YOLO
confidence over the headings that returned a box there. Confidence is read only where
`detected == 1` — undetected rows still hold a sub-threshold candidate score from the 0.001
prediction floor — and its colour scale is anchored on the detector's own frozen
`confidence_threshold`, read from `bbox_detector_manifest.json` rather than chosen by eye.
Positions with no box at any heading are drawn as a grey ×, never as low confidence.

`plot_characterization.py` now owns only the teaching example in folder 00 and the coverage
sheet in folder 01. `plot_error_fields_by_correction.py` writes the field maps in
`02_the_error/`: one folder for
each of the five correction rungs and one full-page sheet per camera. The three interpretations
that fit nothing use every captured position; the learned linear and neural rungs use held-out
tiles only. The strict five-rung like-for-like version lives in `05_the_fixes/`. Both are kept
because a gap-free baseline map and a held-out comparison are different claims. The plotter
reads `bias_update_interpretations.csv`, so `fit_bias_updates.py` must run before it.

**Warehouse arrow maps are scaled per panel.** `_fieldscale.py` gives every panel its own arrow
gain and colour cap, derived from that panel's own residuals, printed in its title, and backed
by a scale bar drawn inside the panel. One shared scale is unusable across a ladder whose rungs
differ tenfold: a gain suited to the ~30 cm raw rung draws the ~3 cm neural rung as dots pinned
to the bottom of the colour bar. Arrows are clipped in *drawn* length, not world length, so an
amplified residual cannot run off the warehouse, and the clipped count is printed.

## Why readings fail

```bash
python3 experiments/camera_observation_characterization/plot_failure_anatomy.py --overwrite
python3 experiments/camera_observation_characterization/plot_failure_gallery.py --overwrite
```

`_sightline.py` defines the two geometric quantities these use, once: whether a rack stands
between a camera and a floor point (the racks are 5.226 m and the cameras 5.00 m, so a crossed
footprint blocks outright and a 2D test is exact), and how many centimetres of floor one pixel
of box-bottom error is worth, `h / (f sin^2 theta)`. Both come from world CAD and camera pose
only — no detector output, no ground truth — so a deployment could precompute either. The
capture ran without the semantic pass, so `line_of_sight` in `capture_index.csv` is empty and
this geometric test is the only occlusion evidence there is; it is blind to the low crates.

The gallery picks, for each diagnosed cause, the reading closest to that group's median error,
so no panel is selected for effect. Its reference box is the analytic hull projected from the
TRUE pose — an oracle, shown to expose what the detector's box got wrong.

## A runtime gate on box size

```bash
python3 experiments/camera_observation_characterization/plot_box_shape_gate.py --overwrite
```

Projects the hull at the RAW back-projection, averaged over eight headings because runtime
heading is unknown, and compares the expected box width with the detected one. Inputs are the
box, the calibration and CAD — never the true pose or an error column.

**Thresholds are set on TRAIN availability, never scanned against held-out error**: each
retains a declared 90% of TRAIN detections, and the whole trade-off curve is drawn so the
operating point is visible. The result is that accuracy and calibration want *different*
gates — box-size consistency cuts the 90th-percentile error hardest, a plain size floor is the
one that improves NEES. Read the folder README before quoting either.

## Stage 2: learning the measurement covariance

```bash
python3 experiments/camera_observation_characterization/learn_measurement_covariance.py \
  --mean-model nn --overwrite
```

Runs the R0–R4 ladder from PLAN.md on the residual the frozen mean model leaves behind, and
writes `10_learning_R/`. Two things in it are easy to get wrong and are handled explicitly:

- The covariance is fitted to **out-of-fold** residuals. The mean model scores 4.2 cm RMSE on
  its own tiles and 12.0 cm off them, so fitting a covariance to in-sample residuals yields
  NEES 30–60 instead of 4–10. Folds are cut by floor position, so all eight headings of a
  position move together.
- A regression on log-variance is unbounded. Predictions are clamped to the range the data can
  actually resolve, measured from camera × range-quartile strata of roughly 160 readings each —
  not from single-sample `log(e²)` values, which cross zero and would license a 0.1 mm sigma.

The result is that the ladder stops at **R0**, one constant for all five cameras, and that even
R0 is about twice as confident as it should be.

`plot_why_r_fitting_fails.py` takes that apart into three separate problems: fitting to
in-sample residuals (a procedure mistake, fixed by the out-of-fold step above), a missing
variable, and too little data to split finely. Its key control is an oracle told which decile
of error a reading will land in: it reaches NEES 2.15 with a *sharper* sigma than any real
rung, which shows a Gaussian `R` is achievable here and that the obstacle is the missing
variable — how much of the robot a rack hides — rather than intractable noise. See the folder
README before quoting any of it.

## Evidence boundary

- The detector never sees semantic labels or commanded pose.
- Every downstream box interpretation must join by `(pose_id, repetition_id, camera_id)` and
  use the same `image_sha1` and YOLO box.
- A single capture at a state maps deterministic spatial structure. It does not estimate a
  repeated-sampling covariance at that state; that needs a separately frozen repeat panel.
- Final plots belong in the numbered folders under this study's output directory; do not
  scatter PNGs directly under `logs/studies` or the capture dataset.

## Two learned bias updates on the same frozen box

`fit_bias_updates.py` adds two more rungs to the same ladder. Both correct the *same* raw
back-projected point that `raw` and `fixed` start from, so nothing about the sensor changes:

| rung | what it is | what it may look at |
|---|---|---|
| `raw` | no correction | — |
| `fixed` | one constant radial shift everywhere | — |
| `learned` | per-camera ridge regression, two outputs (along-ray and across-ray correction) | box, camera, raw back-projection |
| `nn` | one pooled `MLPRegressor(64, 64)` over the same inputs plus a five-way camera indicator | the same |
| `hull` | analytic silhouette residual around the offline reference pose | **needs the reference pose — not operational** |

Both learned rungs see only what the runtime holds: box corners, bottom-centre pixel,
confidence, the camera identity, and the range/bearing of the raw back-projection. Commanded
pose, robot heading and true range are never inputs; truth is the regression target and only
on training tiles.

Package the current fitted models and create an auditable model card, exact reproduction
check, learning curve, held-out CDF, generalisation plot and per-camera summary with:

```bash
python3 experiments/camera_observation_characterization/package_bias_model.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831
```

The canonical artifact and summaries are written to
`logs/perception_models/box_feature_bias_correction_20260831/`. This box-feature correction
is the only learned bias model used by this characterization study; it must not be confused
with image-based or oracle-heading detector experiments.

Although heading is not an input, box shape and projected box location can encode it
implicitly. Audit that possibility without changing the frozen model:

```bash
python3 experiments/camera_observation_characterization/analyze_implicit_heading.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831
```

This writes a held-out heading-decoding probe, predicted-versus-required correction curves
by camera and heading, box-shape curves, and a machine-readable diagnostic beside the current
model artifact. Ground-truth heading is only the offline probe/stratification label.

**Holdout.** The warehouse is cut into a 2 m checkerboard and alternate tiles are held out;
all eight headings of a position stay on the same side. Every figure is scored on held-out
tiles only, and all five rungs are scored on exactly the same rows.

```bash
python3 experiments/camera_observation_characterization/fit_bias_updates.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831

python3 experiments/camera_observation_characterization/plot_bias_updates.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831
```

### Candidate admission-gate sensitivity

Keep the all-return figures as the sensor baseline, then generate a separate diagnostic set
showing what a runtime-plausible gate buys and what availability it costs:

```bash
python3 experiments/camera_observation_characterization/plot_gate_sensitivity.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831
```

The displayed candidate gate retains the detector's frozen confidence threshold, requires
the selected bbox bottom-centre to be clear of the image edge, and limits the range of the
raw floor back-projection to 16 m. It never reads commanded pose, true range, or an error
column. The learned methods are not refitted after gating. The 16 m rule remains a diagnostic
proposal rather than a frozen runtime decision.
Every gate plot reports retained camera opportunities beside conditional reading error so a
gate cannot appear better merely by hiding hard observations. Outputs go to
`logs/studies/camera_observation_characterization_20260831/06_what_a_gate_costs/`.

### Virtual route bias profile

The field capture has no meaningful elapsed-time axis because poses were teleported. To show
how bias would be encountered during traversal without inventing time, project held-out field
positions onto the frozen 30.6 m route and select the captured heading nearest its local
direction of travel:

```bash
python3 experiments/camera_observation_characterization/plot_route_bias_profile.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831
```

The resulting `07_along_a_virtual_route/18_virtual_route_bias.png` plots signed forward and
sideways camera-reading error against distance travelled. Every returned reading is drawn:
filled where the candidate gate admits it, hollow where the gate rejects it, so the cost of
the gate is visible in the same panel. It is a spatial route profile, not dynamic-drive,
timing, motion-blur, or filter evidence.

### Actual-run bias over elapsed time

Use an exact schema-4+ drive directory to plot genuinely timestamped camera-reading errors:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
B=logs/studies/fusion_on_fixed_routes/diagnostic_schema5_20260831/fusion_network_traverse
python3 experiments/camera_observation_characterization/plot_real_run_bias.py \
  --run $B/O1/seed0/experiment_20260831_110742 \
  --run $B/O2/seed0/experiment_20260831_111253 \
  --run $B/F4/seed0/experiment_20260831_110019
```

Repeat `--run` once per column; each column must carry a distinct observation model and every
run must share one frozen route. The loader deduplicates each `(camera, obs_stamp)` and scores
it against ground truth at that capture stamp. Each row of the resulting
`08_on_a_real_drive/19_raw_fixed_hull_over_time.png` shares one scale so the interpretations
compare, but the columns are separate closed-loop drives with different observation streams:
the sheet remains one-seed diagnostic evidence until replicated runs are explicitly frozen
under the localization metrics contract.

This script writes into two folders of the study's output directory. `02_the_error/` gets the
signed-component sheet (`03`) and the distance-error histograms (`04`); `05_the_fixes/` gets
the uncluttered median/spread ladder (`09`), the fitted-versus-held-out comparison (`10`), and
the per-camera warehouse maps (`11`).

Pass `--skip-fields` to skip the five map sheets, and `--overwrite` to refresh an existing
figure directory.

Because neither learned update receives robot heading, never use the heading-marginalized
maps alone to claim that conditional bias was removed. Generate the held-out
camera-by-heading audit as well:

```bash
python3 experiments/camera_observation_characterization/plot_heading_conditioned_bias.py \
  --capture logs/perception_datasets/warehouse_v2_bbox_characterization_20260831
```

The `03_why_the_error/` output contains a camera-by-heading scorecard reporting coverage,
typical error, tails and signed residuals (`07`); heading-induced spread sheets (`06`); and,
under `08_residual_by_heading/`, all five corrections split into camera folders containing an
overview plus eight full-page heading maps. Folder `09` holds the paired neural-minus-linear
maps. The CSV/JSON summary contains all 200 correction × camera × heading cells. Each field
arrow is one held-out observation. With one capture per state it is not a conditional
mean-bias estimate; the separately frozen repeat panel is still required to separate
systematic bias from repeated-sampling spread.
