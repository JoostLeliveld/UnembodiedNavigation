# Commissioning comparison and presentable results

Current account: [../../docs/ICRA_STATUS.md](../../docs/ICRA_STATUS.md).
This directory adds a controlled **development** study; it does not replace historical
campaign selections or claim a paper-facing fusion result. The current task maps this
evidence into a 12-page, two-column AIES thesis; see the current status above.

The [planner implementation plan](planner_implementation_plan.md) and
[whole-paper map](../../../papers/master_thesis/planning/paper_map.md) describe the current
IWAI network extension. `export_network_planner.py` produces uniform, geometry and GP
score fields; `network_route_probe.py` resolves and exercises the full live-planner
configuration. `network_navigation_pilot.yaml` executes the three matched fields with
frozen post-NN mean/R calibration. The separate `network_navigation_tracking_pilot.yaml`
tests a reproduced waypoint handoff failure. Both remain one-seed integration pilots;
`network_navigation_analysis.py` preserves and scores their explicit campaign ledgers.

The approved staged plan, GP decision and evidence gates are maintained in
[`docs/ICRA_STATUS.md`](../../docs/ICRA_STATUS.md).
Earlier conceptual diagrams are preserved below. Current implementation diagrams are in
`../papers/master_thesis/planning/` relative to the repository root and are regenerated
from the workspace root with `python3 papers/master_thesis/planning/build_maps.py`.
Generate the earlier conceptual thesis block diagrams (vector PDF, SVG, PNG and combined plan) with
`MPLCONFIGDIR=/tmp/icra_mpl python3 experiments/icra_commissioning/thesis_plan.py`.
Outputs are under `logs/studies/icra_commissioning_20260905/thesis_plan/` and contain no
experimental-result claims.

## Reproduce the recorded results

Run from the repository root, with the existing Python environment. Limit BLAS/OpenMP
threads to avoid starving simulation. Each script reads explicit manifests and rejects
changed required inputs. `freeze` is first-time only; do not overwrite a frozen manifest.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/icra_mpl
python3 experiments/icra_commissioning/image_uncertainty.py
python3 experiments/icra_commissioning/generalization.py
python3 experiments/icra_commissioning/replay.py run
python3 experiments/icra_commissioning/future.py
python3 experiments/icra_commissioning/baseline_report.py
python3 experiments/icra_commissioning/replay.py run --selection logs/studies/icra_commissioning_20260905/validation_manifest.json --output logs/studies/icra_commissioning_20260905/validation_replay
python3 experiments/icra_commissioning/present_results.py
```

To refit calibration without altering the frozen evidence, use a new output directory:

```bash
python3 experiments/icra_commissioning/study.py freeze --output logs/studies/icra_commissioning_refit
python3 experiments/icra_commissioning/study.py fit --output logs/studies/icra_commissioning_refit
```

The validation replay uses a byte-identical copy of `models.joblib` in `validation_replay/`.
If a refit changes the artifact, create a new versioned validation output and manifest;
do not present it as the old frozen model. The crop-statistics cache is ordered by the
manifest-bound source rows and is specific to this study.

A new live baseline execution (requires local ROS/Gazebo sockets):

```bash
source install/setup.bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config experiments/icra_commissioning/baseline_cpu.yaml \
  --log-root logs/studies/icra_commissioning_20260905/another_baseline \
  --run-timeout 1100 --first-cmd-timeout 150
```

This is the existing NN mean plus existing pixel-derived covariance and joint-network
fusion. The new covariance models and latent-bias ablation remain **replay implementations**
for measurement updates. A frozen constant R also enters the new planner cost proxy;
this does not install the same mean/R or bias state in the online camera manager.

## Measurement interface

- Quantity: estimated planar robot model origin / ground footprint reference, `map_bev`, m.
- Raw statistic: bbox bottom centre in the original 1280×720 camera image. It is not a
  physical contact-point label. Ground projection uses the existing calibrated camera.
- Existing NN target: reference XY minus raw XY in the **raw camera-ray basis**, in metres.
  The output is a metric reference-position measurement. It is not a pixel-keypoint network.
- Observation function: `h([x,y,theta])=[x,y]`; the correction is already in z, so it is
  not applied again to h or its Jacobian. No estimated heading or GT is a correction input.
- Replay and the new `commissioned_reference_r` runtime mean: existing NN output minus
  the **same fitted per-camera mean offset**. The `camera_reference_calibration.v1`
  artifact binds that offset and full camera R to the exact mean checkpoint hash.
  The wrapper applies after the NN, once, without changing pixel/crop coordinates.
  Its camera outputs are checked against recorded offline predictions and live logs.
- Covariance: centred conditional scatter in m², followed by separately recorded scalar
  calibration and isotropic shrinkage chosen on selection tiles. Uncentred second moments
  and residual cell means are separate fields. Working Gaussian scores are not white-noise
  or exact-generative-likelihood claims.
- Identity: live capture stamp and `(run, camera, source_batch_id)`; replay additionally
  deduplicates by camera/capture time through `aligned.py`. Static v1 has a row identity
  `(capture, camera, pose, repetition)` but no source batch ID or independently verified
  settle time. `records.jsonl` preserves all misses and states the reference limitation.
- Acceptance: static raw-valid detection, with cross-role duplicate-hit-image exclusion.
  Replay fixes the legacy logged population, no new NIS gate. Legacy logs omit some manager
  refusals; do not use them to fit full availability. Static misses train the forecast.
- Current images: five crop statistics, with only the pixel extraction rectangle bounded
  to the image. No reference target is clamped. Runtime RGB was not retained in these drives.

## What the new experiments isolate

`model.py` fits full constant, diagonal, isotropic, observable ray/range, spatial and
confidence-regime covariance with SPD regularization. The original mean checkpoint is frozen.
`study.py` separates 2 m tiles into covariance fitting, selection, and evaluation; all original
NN training rows are excluded from uncertainty fitting. These tiles had been inspected before
this task, so evaluation is development-only. Selection tuning includes the strong constant
baseline. Budget curves explicitly show the untuned constant estimator and exclude NN training
only in the panel whose caption says so.

`replay.py` uses noisy measured wheel velocities and the production unicycle dynamics and Q.
It scores all arms on the same odometry-stamp grid. It is capture-time replay, not live delay
reproduction. The optional 2 s bias state keeps robot Q unchanged; its parameters were chosen
as a development diagnostic and were frozen for the new CPU execution replay.

`future.py` queries commissioning with the predicted robot pose, averages twelve joint
camera outcomes and compares expected information. It has no future image/GT input. Its
multi-step moment matching assumes independent opportunities and its cadence sweep does not
change the reference estimator's cadence. It cannot establish route ranking from one route.

## Correctness repairs in this task

1. Register existing N1 campaign in `CONDITION_PLANNER`.
2. Repair `planA_nn.yaml` ROS domain 251 → 151 (valid supported range 0..232).
3. Replace inconsistent scaled-gain covariance arithmetic by the joint-moment Joseph form
   and stable solve in `belief_correction.py`. Unit-gain camera fusion retains equivalent
   mathematics. The regression suite separately checks legacy unit gain and scaled Joseph.
4. During development of the new replay, fix scoring to a common odometry grid; otherwise
   camera rate would change metric weights. This was a new-script repair, not a runtime gain.

Meaningful checks: camera projection round-trip over the actual installation and headings,
runtime/offline NN agreement, covariance versus bias, complementary-view fusion, hit/miss
branch averaging, Joseph update, timestamp alignment and fusion accounting. Final focused
suite: **79 passed, 3 skipped**; the skips are old locked-data tests whose artifacts are absent.
