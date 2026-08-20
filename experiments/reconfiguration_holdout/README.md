# Reconfiguration holdout — does the observation model survive a changed warehouse?

**Question this study answers:** a fixed camera network's usable-observation
probability can be estimated from the cameras' own images without a surveyed 3-D
model. That was measured in one warehouse configuration, with the fields fitted and
scored in that same configuration. Warehouses get restocked. **Which parts of the
field survive a change it was never shown, and does the component that makes the field
sharp enough to plan with also make it stale?**

The analysis plan is frozen in [`PREREGISTRATION.md`](PREREGISTRATION.md), written
before any reconfigured-layout detector outcome existed. Read that first; this file is
how to run it.

> **Post-analysis correction (2026-08-19).** The first E1 implementation leaked L0
> outcomes into its GP spatial field and did not actually freeze the hybrid residual.
> Those E1 numbers are withdrawn. [`ANALYSIS_CORRECTION.md`](ANALYSIS_CORRECTION.md)
> documents the errors, the fold-clean replacement, and the corrected null raw-Brier
> interactions. The routing statistics now enumerate all 25 cells and apply Holm
> correction; do not reuse the former “15 cells / 8 significant” manuscript text.
> E3 also now uses an independent-heading protocol: all learned fields are fitted on
> L0's four diagonal headings, GP/hybrid links use six-block out-of-fold predictions
> from those diagonal outcomes, and route truth uses only the disjoint cardinal
> headings shared by L0 and L1.

## The four environments

Same cameras, same calibration, same lanes, same robot, same detector, same
thresholds. Only the world file differs.

| key | layout | lighting | world |
|---|---|---|---|
| `L0` | nominal | nominal | `warehouse_full_4cam` (frozen flagship, untouched) |
| `L1` | 12 of 27 rack segments carry one extra 0.40 m layer of stock | nominal | `warehouse_full_4cam_recfg` |
| `L0_lit` | nominal | two lamps out, third dimmed, low-angle light from the west | `warehouse_full_4cam_lit` |
| `L1_lit` | restocked | changed | `warehouse_full_4cam_recfg_lit` |

**Why restocking rather than a pallet in an aisle.** Measured, not assumed. Three
obstacles placed in driveable aisles, chosen greedily to maximise newly blind
*reachable* floor, buy +46 newly blind cells out of 3397 covered — 1.4 %. Four cameras
on opposite corners cover each other's shadows, and an obstacle big enough to darken
an aisle also severs it, at which point every planner detours and the comparison stops
being about visibility. Restocking costs the network **191 of 3397 covered cells
(5.6 %)** and changes **575 camera-cell visibility pairs**, with the driveable network
bit-identical — no aisle is touched, so obstacle avoidance and observation modelling
stay separate. Per camera, coverage of eligible driveable ground falls 11–13 %.

## What the pilot already settled, before the main captures

Two findings from a 24-position pilot, both of which changed the design:

**The changed-lighting condition does not move this detector.** At any threshold from
0.05 up, `L0_lit` and `L0` agree on detection to within 0.02 (hit given a clear
sight-line: 0.982 against 1.000). The images really did change — frame mean grey
163 → 112, standard deviation 53 → 36 — so this is detector robustness, not a world
that failed to change. A conditional-detection term conditioned on appearance has
nothing to model here, and the study says so rather than escalating the lighting until
an effect appears.

**The `L0` reference capture's 0.01 confidence threshold is the real defect.** At 0.01
the detector fires at **60 %** of `L0_lit` poses that no camera has a sight-line to,
against 12.5 % in `L0`. It vanishes by 0.05. So the frozen 30,144-sample reference
dataset contains roughly 12 % detections at poses no camera can see. Every headline
number here uses **threshold 0.25**, the middle of the plateau where both environments
agree, re-derived offline from `yolo_raw_best_score` so no re-detection is needed. At
0.25 the per-camera detection rate (0.30–0.33) lines up with the oracle-visible
fraction (0.29–0.32) almost exactly; at 0.01 it did not.

## Running it

```bash
# 1. choose the reconfiguration (geometry only, no simulator)
python3 experiments/reconfiguration_holdout/choose_layout.py --n-segments 12

# 2. generate the three variant worlds + their profile entries
python3 experiments/reconfiguration_holdout/make_variant_worlds.py

# 3. capture each changed environment: real Gazebo, four cameras, detector offline
bash experiments/reconfiguration_holdout/capture_environment.sh \
     L1 warehouse_full_4cam_recfg.world.sdf 4
bash experiments/reconfiguration_holdout/capture_environment.sh \
     L0_lit warehouse_full_4cam_lit.world.sdf 4
bash experiments/reconfiguration_holdout/capture_environment.sh \
     L1_lit warehouse_full_4cam_recfg_lit.world.sdf 4

# 4. the adaptive arm: monocular depth from each environment's own frames.
#    L0 FIRST -- it fits the floor anchor every other environment then reuses.
python3 experiments/reconfiguration_holdout/mono_depth_field.py --env L0
python3 experiments/reconfiguration_holdout/mono_depth_field.py --env L1

# 5. score every arm in every environment
python3 experiments/reconfiguration_holdout/e1_reconfiguration_holdout/run_experiment.py

# 6. solve and score every registered route cell. This also writes the 25-cell
#    paired summary and applies Holm correction across the complete family.
python3 experiments/reconfiguration_holdout/e3_availability_routing/run_experiment.py \
    --environments L0 L1

# The summary can be regenerated from e3_routes.csv without solving routes again:
python3 experiments/reconfiguration_holdout/e3_availability_routing/summarize_cells.py \
    --changed-environment L1

# 7. release the disk once an environment is scored (keeps the depth source frames)
python3 experiments/reconfiguration_holdout/prune_images.py --env L1
```

`L0` needs no capture: it is the existing
`logs/visibility_comparison/commissioning_grid_20260807` reference, re-thresholded.

## Recovering a capture that died at the end

The `L1` capture rendered 15,033 of 15,072 frames and then raised inside its pose loop,
three positions from the end, so `samples.csv` and the manifest were never written.
Re-rendering costs ninety minutes. It is not necessary:
[`rebuild_capture_metadata.py`](rebuild_capture_metadata.py) recomputes every row from
the capture's *own* position and heading samplers and its own oracle calls, keyed by the
frame filenames, which carry sample id, position index, heading index and camera. No
image is re-rendered and no value is interpolated; a frame absent from disk simply has
no row, and the missing count goes into the manifest.

Do not trust it without the check that comes with it. `--validate-against` reruns the
reconstruction over a capture that *succeeded* and diffs every field:

```bash
python3 experiments/reconfiguration_holdout/rebuild_capture_metadata.py \
    --validate-against logs/visibility_comparison/commissioning_grid_20260807
# 30144 real rows, 30144 reconstructed; 7 fields each; PASS
```

## Three traps this study hit, worth not rediscovering

- **This repo runs Ignition Fortress**, launched as `ign gazebo -r -s ... --force-version 6`
  through a Ruby wrapper — not `gz sim`. `pkill -f "gz sim"` finds nothing and leaves a
  live server that the next run silently adopts. Teardown must also kill
  `ros_gz_bridge/parameter_bridge`: killing only the server leaves ~8 bridges still
  advertising the old world's topics.
- **A `pgrep` guard loose enough to match the calling shell's own command line** reports
  a false "already running" and refuses to start. Match the simulator's argv.
- **Never write a log inside the capture's output directory.** The capture tool calls
  `safe_reset_generated_dir` on its `--out` *after* a driver script has opened a log
  there, so the file is unlinked and the shell writes into a deleted inode. The
  traceback from a ninety-minute failure went to nowhere exactly once. Logs now live in
  `logs/visibility_comparison/recfg_holdout_logs/`.
- **Captures must live under `logs/visibility_comparison/`.** The capture tool refuses
  any output root outside it, by design — raw captures are append-only there, and a
  study folder holds only what analysis derives.

## Where the outputs go

| path | what |
|---|---|
| `logs/visibility_comparison/recfg_holdout_<env>/` | raw capture: frames, `samples.csv`, `perception_targets.csv`, `appearance_features.csv` |
| `logs/studies/reconfiguration_holdout/layout/` | the selected reconfiguration and every candidate scored |
| `logs/studies/reconfiguration_holdout/mono_depth/` | per-environment monocular-depth fields and the commissioning anchor |
| `logs/studies/reconfiguration_holdout/work/e3_independent_heading_v1/` | E3-only diagonal events, priors, full deployment fields, spatially OOF link predictions, and their hashes |
| `logs/studies/reconfiguration_holdout/e1_reconfiguration_holdout/` | the results tables |
| `logs/studies/reconfiguration_holdout/e3_availability_routing/e3_routes.csv` | all task-level route outcomes |
| `logs/studies/reconfiguration_holdout/e3_availability_routing/e3_cell_summary_<env>.csv` | one environment-keyed 25-cell paired summary, with CIs and Holm-adjusted p-values |
| `logs/studies/reconfiguration_holdout/e3_availability_routing/e3_cell_summary_<env>_manifest.json` | exact selected environment, contrasts, statistical families, seeds, and input/output hashes |
| `logs/studies/reconfiguration_holdout/e3_availability_routing/e3_cell_summary.csv` | compatibility alias refreshed only from the independent-heading L1 analysis |

Reuse map: [`REUSE_MAP.md`](REUSE_MAP.md).
