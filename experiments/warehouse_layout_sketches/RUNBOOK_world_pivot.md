# Runbook — pivoting the 4-camera world to an asymmetric, occluded layout

Precise steps with real parameters. Every tool already exists and has been run before;
nothing here is new pipeline. Times are measured where known and **marked as unmeasured
where they are not** — the one genuine unknown is training wall-clock, and it is
measurable in 20 minutes rather than guessable.

Prerequisite decision (not a time cost, but it must be made first): every artifact fitted
against the current world — `projection_calibration_v2/v3`, the coverage npz, the
recorded captures, every locked campaign — describes the OLD camera poses. After the
pivot they describe nothing. Either the new world becomes the evaluation world and the
old comparisons are retired, or it becomes a second world and the two-world rule is
restated. **Decide this before step 1, not after step 8.**

## Step 0 — snapshot (1 min)

```bash
git status --short          # confirm a clean-enough tree
git rev-parse HEAD          # record what the old artifacts belong to
```

The old world's artifacts stay valid *as a record of the old world*; they must not be
silently reused against the new one.

## Step 1 — author the world (~5 min)

`scripts/geometry_visibility/make_warehouse_full.py` generates the world and already
holds the camera list and shelf rows as data:

- `CAMERA_ITEMS` — one entry per camera (name, topic, pose). Add/move/aim here.
- the shelf/rack row table — add blocks to create genuine occlusion and misaligned
  north/south aisles.
- `SITE_X0/X1/Y0/Y1` — keep the current 22.7 × 17.2 m footprint unless there is a reason
  not to. **Enlarging the floor multiplies step 3**, because the capture grid scales with
  area; the layout sketches show the interesting structure comes from occlusion and
  aiming, not from floor area.

Use the sketch geometry as the source:
`logs/studies/warehouse_layout_sketches/exp1_layout_candidates/summary.json`.

```bash
python3 scripts/geometry_visibility/make_warehouse_full.py     # regenerates the .world.sdf
colcon build --packages-select sim_gazebo_worlds reliability perception --symlink-install
```

## Step 2 — capture the detector dataset (~25–35 min, real Gazebo)

Launch with the semantic-label bridge, then capture. `capture_yolo_dataset.py` already
maps `external_camera{,_b,_c,_d}` → `camera_A/B/C/D` and auto-labels from the segmentation
mask per camera — **no hand annotation**, which is what makes this fast.

```bash
DATASET=logs/perception_datasets/warehouse_yolo_dataset_asym_v1 \
  scripts/perception/train_yolo_detector.sh          # launches gazebo, captures, audits, trains
```

Defaults that set the clock: `--sample-nx 16 --sample-ny 14 --yaw-samples 8` = **1792
poses** at `--settle-s 0.80` ⇒ ~24 min of settle plus sync/image wait. Raise the grid only
if the floor grew.

**Gate:** per-camera acceptance in `label_diagnostics.csv`. The v2 attempt had A/B smoke
failures — do not train on an under-covered camera.

## Step 3 — train (**UNMEASURED — the one real unknown**)

The shipped recipe is `--epochs 30 --imgsz 960 --batch 4` from
`local_artifacts/base_models/yolo11n-seg.pt`. On a P2000 that is the long pole.

**Do not run 30 epochs from the generic base for a pivot.** Only the viewpoint changed;
the robot's appearance did not. Fine-tune from the existing four-camera detector:

```bash
BASE_MODEL=logs/perception_models/warehouse_yolo_detector_4cam_v3_960/weights/best.pt \
DATASET=logs/perception_datasets/warehouse_yolo_dataset_asym_v1 \
MODELOUT=logs/perception_models/warehouse_yolo_detector_asym_v1 \
  scripts/perception/train_yolo_detector.sh
```

with `--epochs` cut to 3–6 in the script's train invocation.

**Measure before committing:** run 1 epoch, read the per-epoch time from
`/tmp/yolo_detector_train.log`, multiply. That converts the only guess in this runbook
into a number in ~20 minutes.

## Step 4 — detector gate (~2 min)

Range-conditioned per-camera detection rate against `config/detector_4cam_v1.yaml`:
`detection_rate_le_12m ≥ 0.90`, `detection_rate_12_to_16m_unoccluded ≥ 0.75`, on a
held-out traverse. Metrics from `scripts/shared/metrics.py` only.

**This gate has failed before** (`v2_640_diag`). If it fails, the pivot stops here and
nothing downstream is trustworthy.

## Step 5 — projection calibration (~10 min)

Drive a GT-validation traverse in the new world, then refit — now with the gated
cross-bearing term:

```bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/fit_projection_calibration.py \
  --audit-dir logs/studies/<new_capture>/evaluation_inputs \
  --output logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_asym_v1/projection_calibration.json
```

The `--cross-bias-sigma-gate 1.2` default decides per camera whether the cross term is
fitted. Read the printed `GATED OFF (ratio …)` lines — in a new geometry the gate
outcomes will differ from A/B off, C/D on.

## Step 6 — coverage / GP artifacts (~10–15 min)

Regenerate the per-camera availability maps that everything downstream consumes
(`P_camera_*_map`, `P_best_4cam_map`, `coverage_count`). `fit_belief_aware_gp.py` is the
canonical fit — import it, do not reimplement.

## Step 7 — re-run every offline study (~2 min total)

All nine studies read artifacts and re-run in seconds. Point them at the new calibration
and coverage artifact:

| study | what it re-derives |
|---|---|
| `external_camera_bias_model` exp1/exp2 | per-camera bias, the 2-DOF gate outcomes |
| `projection_amplification` | geometric amplification in the new geometry |
| `operational_residual_rcond` exp2/exp3 | belief calibration, NEES |
| `network_commissioning_realism` | gate without truth, sample efficiency |
| `calibration_drift_lifecycle` | staleness thresholds |
| `bayesian_filter_showcase` exp1/exp2 | the honest-belief result and its ablations |
| `achievable_precision_map` | precision-vs-coverage — **expected to get much more interesting** with real occlusion |
| `planner_covariance_branching` | unaffected (analytic) |

## Honest total

| step | wall-clock |
|---|---|
| 0–1 world + build | ~6 min |
| 2 capture | ~25–35 min |
| 3 train | **unmeasured** — 1 epoch tells you |
| 4 gate | ~2 min |
| 5 calibration | ~10 min |
| 6 coverage/GP | ~10–15 min |
| 7 studies | ~2 min |

Everything except step 3 totals **~55–70 min**. Whether the whole pivot lands under an
hour is entirely a question of the fine-tune length, which is why step 3 says measure
one epoch first.

## What would make this NOT worth it

- The detector gate (step 4) fails on the new viewpoints and needs a real retrain rather
  than a fine-tune. That is the scenario that turns an hour into a day.
- The decision at the top has not been made, and the pivot silently orphans the locked
  comparisons.
