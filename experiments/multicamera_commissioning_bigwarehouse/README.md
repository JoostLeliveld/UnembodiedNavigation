# Full-Warehouse Four-Camera Commissioning

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-COMMISSION
status: LOCKED
claim_ids:
- C1
- C6
assumption_ids:
- A01
- A03
- A08
- A10
- A14
- A15
- A16
reviewer_question_ids:
- RQ07
- RQ09
- RQ12
figure_ids:
- F01
- F03
dependencies:
- ASSET-RUNTIME
operational_inputs:
- camera_frames
- detections
- calibration
evaluation_only_inputs:
- ground_truth_pose
primary_metric: held-out calibration and commissioning readiness
promotion_gate: Frozen detector calibration and provenance pass the readiness contract.
evidence_paths:
- logs/studies/multicamera_commissioning_bigwarehouse/paper_readiness_v1/readiness_status.json
- logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v3/projection_calibration.json
- experiments/multicamera_commissioning_bigwarehouse/GATE_PROVENANCE.md
archive_rule: Keep chosen captures manifests calibrations and detector; cold-store
  superseded raw captures.
next_action: Preserve commissioning evidence; WS05 owns the superseding v2-v3-v4 campaign-arm
  decision.
```

<!-- RESEARCH-METADATA:END -->


> **Retargeted 2026-07-15 — world consolidation.** This study commissions the
> four fixed external cameras in `warehouse_full_4cam.world.sdf`. The thesis
> keeps only `warehouse_aws` and `warehouse_full_4cam`; the former auxiliary
> two-camera testbed and its data are archived development history, not current
> evidence.

## Scope

```text
four isolated camera streams
  -> independent camera-specific trust maps
  -> overlap validation and conservative source selection
  -> handover covariance adjustment
```

The study uses cameras A–D:

| Camera | Mount | RGB topic |
| --- | --- | --- |
| A | south wall, west dock column | `/external_camera/image_raw` |
| B | north wall, west dock column | `/external_camera_b/image_raw` |
| C | south wall, east dock column | `/external_camera_c/image_raw` |
| D | north wall, east dock column | `/external_camera_d/image_raw` |

`config/study.yaml` is the contract for camera IDs, calibration IDs, gates,
collection routes, and artifact locations. The four-camera day-zero prior is a
calibration-only starting point; learned per-camera maps require new operational
detector evidence.

The manager receives only detector validity, projection validity, timestamp/age,
camera-specific trust, association quality, and pairwise consistency. Ground
truth and oracle labels are evaluation-only.

## Current status

- Four-camera world, launch, camera contracts, recorder, replay/export split,
  selection baselines, hysteretic manager, and handover covariance adjustment:
  **implemented**.
- Route geometry, documented spawn poses, offsets, and two speeds are now
  defined in `config/study.yaml` for all four source regions, both handover
  directions, and three overlap passes.
- A two-route operational pilot produced per-camera GP posteriors and three
  strict C/D pairs. It proves the pipeline, but is deliberately not presented
  as route-disjoint mapping, qualified fusion, or closed-loop evidence.
- `config/paper_protocol.yaml` is the frozen paper contract: route-disjoint
  fitting, all three overlap edges, paired replay policies, disturbances, and
  the gates that must pass before active handover can be enabled.

## Laptop commissioning successor

The locked 1280×720 paper configuration is not changed. A separate
640×360-per-camera laptop commissioning world and v3 configuration are now
available for throughput diagnosis on the P2000. It is explicitly blocked from
evidence collection until it has fresh calibration, detector validation, and
strict 3 Hz/100 ms timing evidence. See
[`LAPTOP_640X360_COMMISSIONING.md`](LAPTOP_640X360_COMMISSIONING.md).

The v4 asynchronous shared-model microbatch profile is a separate diagnostic
to isolate strict timestamp-admission cost. It has no runtime contract and is
therefore intentionally barred from recording and evidence; see
[`LAPTOP_ASYNC_MICROBATCH_COMMISSIONING.md`](LAPTOP_ASYNC_MICROBATCH_COMMISSIONING.md).

The v6 direct-Gazebo / fixed-shape TorchScript candidate keeps 640×360 source
quality but **fails** the 3 Hz gate: its 3.39 Hz probe was retracted (the export
had lost the segmentation task metadata), and with `task=segment` restored the
integrated runtime does ~1.2 Hz per camera. It stays blocked; see
[`LAPTOP_DIRECT_COMPILED_COMMISSIONING.md`](LAPTOP_DIRECT_COMPILED_COMMISSIONING.md).

## Start the passive four-camera stack

```bash
source install/setup.bash
ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
  yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
  headless:=true
```

This launches the current four-camera Gazebo world, bridges cameras B–D, starts
four isolated detector nodes, publishes one camera-observation contract per
source, and publishes `/odom_noisy` as a passive operational recording stream.
The collection driver continues to use `/odom`, so logged encoder noise cannot
alter the driven path. `/odom_noisy` is a passive operational stream and now
carries a propagated planar covariance; the recorder rotates it into warehouse
coordinates before the GP-input adapter sees it.

## Frozen paper protocol

Materialise the route/task matrix and audit current evidence with:

```bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/paper_campaign.py \
  --run-root logs/studies/multicamera_commissioning_bigwarehouse/actual_commissioning_20260715 \
  --validation-root logs/visibility_comparison/fourcam_actual_20260715/paper_route_disjoint \
  --out-dir logs/studies/multicamera_commissioning_bigwarehouse/paper_protocol_v1
```

This creates a separate folder containing `campaign_plan.csv`, a collection
checklist, gate status, route/qualification figures, and talking points. It
does not allow the sparse pilot to pass any gate.

## Confirmatory paper readiness

`config/paper_analysis_plan.yaml` is the paper claim contract. It fixes the
simulation-only scope, allowed operational inputs, D0 projection and odometry
calibration gates, the route-disjoint D1 GP endpoint, D2 edge qualification,
the paired M8-versus-M7 D3 decision rule, and the final D4 closed-loop
confirmation. It also states what the paper must *not* claim (physical-world
performance, a pooled deployment GP, or oracle selection).

Run the independent gate audit after each batch:

```bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/paper_readiness.py \
  --run-root logs/studies/multicamera_commissioning_bigwarehouse/<campaign> \
  --validation-root logs/visibility_comparison/<four-camera-d1-validation> \
  --replay-root logs/studies/multicamera_commissioning_bigwarehouse/<d3-replay-root> \
  --out-dir logs/studies/multicamera_commissioning_bigwarehouse/paper_readiness_v1
```

It writes `PAPER_EVIDENCE.md`, machine-readable gate status, and SHA-256 hashes
of the frozen protocol/method files. A missing or failed gate means the result
is not eligible for a confirmatory paper claim.

Each D3 replay case is deterministic and manifest-backed. Pass one posterior
per camera; the tool rejects a pooled-only GP setup. The `low_light_proxy`
condition is a clearly labelled replay observation-thinning proxy, not an
illumination experiment.

```bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/run_paired_replay.py \
  --export-dir <evaluation-only-split-export> --task-id T1_long_handover \
  --condition camera_dropout --seed 0 \
  --camera-gp camera_A=<A_expected_kernel.npz> \
  --camera-gp camera_B=<B_expected_kernel.npz> \
  --camera-gp camera_C=<C_expected_kernel.npz> \
  --camera-gp camera_D=<D_expected_kernel.npz> \
  --out-dir logs/studies/multicamera_commissioning_bigwarehouse/<d3-root>/T1_long_handover/camera_dropout/seed0
```

Fresh recordings must use the current encoder covariance model. It propagates
both random encoder error and a declared residual scale-bias term; the D0 audit
checks its calibration against evaluation-only truth without exposing truth to
the detector, GP, manager, or planner.

## Collection protocol

For each frozen plan row, start Gazebo at the documented **offset-adjusted**
route spawn, pass the same detector settings and seed to every process, record
both streams against the driver's success-only completion artifact, and then
run the fixed route driver. Do not add or move a route after inspecting
evaluation-only results. Every failed attempt remains immutable; never delete
partial files and retry in the same run directory.

```bash
# Create this once, before collecting or inspecting evaluation truth. The two
# --config arguments must be repeated unchanged on both recorders below.
python3 experiments/multicamera_commissioning_bigwarehouse/tools/campaign_ledger.py plan \
  --campaign-root logs/studies/multicamera_commissioning_bigwarehouse/<campaign> \
  --model logs/perception_models/<frozen-model>/model.pt \
  --calibration logs/studies/multicamera_commissioning_bigwarehouse/<frozen-calibration>/projection_calibration.json \
  --config experiments/multicamera_commissioning_bigwarehouse/config/paper_analysis_plan.yaml \
  --config experiments/multicamera_commissioning_bigwarehouse/config/detector_4cam_v2.yaml

# Shared values (use the exact row_id/run_dir from campaign_ledger.json)
campaign_root=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>
row_id=<row_id>
attempt_id=attempt_001
run_id=<unique-run-id>
run_dir="$campaign_root/runs/$row_id/$attempt_id"
route_completion="$run_dir/raw/route_completion.json"
model=logs/perception_models/<frozen-model>/model.pt
projection=logs/studies/multicamera_commissioning_bigwarehouse/<frozen-calibration>/projection_calibration.json
analysis=experiments/multicamera_commissioning_bigwarehouse/config/paper_analysis_plan.yaml
runtime_config=experiments/multicamera_commissioning_bigwarehouse/config/detector_4cam_v2.yaml
study_config=experiments/multicamera_commissioning_bigwarehouse/config/study.yaml
protocol_config=experiments/multicamera_commissioning_bigwarehouse/config/paper_protocol.yaml
seed=<seed-used-by-encoder-and-both-recorders>
analysis_split=<expected-analysis-split-from-ledger>
evidence_role=<expected-evidence-role-from-ledger>
ros_domain_id=<ros_domain_id-from-attempt>
ign_partition=<ign_partition-from-attempt>

# These values are part of the attempt contract. Export them in every shell
# (simulator, readiness barrier, both recorders, and driver); only one attempt
# may be active for a campaign at a time.
export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export ROS_DOMAIN_ID=$ros_domain_id IGN_PARTITION=$ign_partition

# Terminal 1 — exact frozen runtime at the offset-adjusted spawn pose
source install/setup.bash
ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
  headless:=true reset_world:=true shadow_manager:=false \
  spawn_x:=<route-start-x> spawn_y:=<route-start-y> spawn_yaw:=<route-start-yaw> \
  use_encoder_noise:=true encoder_noise_seed:=$seed \
  yolo_model:=$model yolo_imgsz:=<frozen-image-size> \
  yolo_conf_threshold:=0.05 yolo_iou_threshold:=0.45 \
  yolo_use_masks:=false yolo_use_torchscript:=false \
  yolo_batched_four_camera:=true yolo_batched_device:=0 \
  yolo_batched_cpu_num_threads:=2 yolo_cpu_num_interop_threads:=1 \
  yolo_opencv_num_threads:=1 yolo_max_batch_stamp_skew_s:=0.10 \
  yolo_max_pending_wall_s:=0.50 \
  camera_observation_r_visible_uv:=2.5 camera_observation_r_miss_uv:=40.0

# Before any evidence run, this must pass in enforce mode. Pilot mode is
# timing-diagnostic only and is never evidence-eligible. This exact immutable
# report is a required attempt artifact: run it before invoking the driver,
# which writes raw/route_manifest.json immediately before route motion.
python3 experiments/multicamera_commissioning_bigwarehouse/tools/runtime_readiness.py \
  --mode enforce --timeout-s 180 \
  --campaign-ledger "$campaign_root/campaign_ledger.json" \
  --run-id "$run_id" --plan-row-id "$row_id" --attempt-id "$attempt_id" \
  --analysis-split "$analysis_split" --seed "$seed" \
  --evidence-role "$evidence_role" \
  --study-config "$study_config" --protocol "$protocol_config" --analysis-plan "$analysis" \
  --detector-model "$model" --detector-runtime-config "$runtime_config" \
  --detector-imgsz 640 --projection-calibration "$projection" \
  --frozen-config "$analysis" --frozen-config "$runtime_config" \
  --min-camera-messages 30 --min-unique-camera-stamps 30 --sample-window 30 \
  --output "$run_dir/runtime_readiness.json"

# Terminal 2 — operational record only; completion, not duration, owns stop
source install/setup.bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/record_operational_logs.py \
  --out-dir "$run_dir/raw" --completion-manifest "$route_completion" \
  --wall-timeout-s 720 \
  --campaign-ledger "$campaign_root/campaign_ledger.json" \
  --run-id "$run_id" --plan-row-id "$row_id" --attempt-id "$attempt_id" \
  --analysis-split "$analysis_split" --seed "$seed" \
  --evidence-role "$evidence_role" \
  --frozen-config "$analysis" --frozen-config "$runtime_config" \
  --odom-origin-x-m=<route-start-x> \
  --odom-origin-y-m=<route-start-y> \
  --odom-origin-yaw-rad=<route-start-yaw> \
  --detector-model "$model" --detector-runtime-config "$runtime_config" \
  --detector-imgsz 640 \
  --detector-conf 0.05 --detector-iou 0.45 --no-detector-use-masks \
  --detector-runtime-mode batched_four_camera \
  --detector-executable batched_four_camera_yolo_node \
  --detector-model-format native_ultralytics \
  --detector-model-instances 1 --detector-batch-size 4 \
  --detector-camera-order camera_A camera_B camera_C camera_D \
  --detector-shared-device 0 --detector-cpu-threads 2 \
  --detector-interop-threads 1 --detector-opencv-threads 1 \
  --detector-max-batch-stamp-skew-s 0.10 \
  --detector-max-pending-wall-s 0.50 \
  --detector-frame-policy strict_new_unique_stamp_latest_only_no_reuse \
  --detector-yolo-batched-four-camera \
  --projection-calibration "$projection"

# Terminal 2b — evaluation only; uses the identical completion and identity
source install/setup.bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/record_evaluation_truth.py \
  --out-dir "$run_dir/evaluation_only" \
  --completion-manifest "$route_completion" --wall-timeout-s 720 \
  --campaign-ledger "$campaign_root/campaign_ledger.json" \
  --run-id "$run_id" --plan-row-id "$row_id" --attempt-id "$attempt_id" \
  --analysis-split "$analysis_split" --seed "$seed" \
  --evidence-role "$evidence_role" \
  --frozen-config "$analysis" --frozen-config "$runtime_config"

# Terminal 3 — start only after every recorder .part stream has received data
source install/setup.bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/drive_study_route.py \
  --route <route-name> --lateral-offset-m <offset> --speed-mps <speed> \
  --manifest-path "$run_dir/raw/route_manifest.json" \
  --completion-path "$route_completion"
```

If the driver fails, it writes no route-completion artifact; both recorders
therefore fail on their wall deadman and leave `.part`/failed evidence. If any
recorder exits before driver success, interrupt the driver immediately. Do not
use process-name matching for supervision: retain the three exact PIDs/process
groups and terminate them with `SIGINT` so their fail-closed shutdown runs.

After the driver and both recorders exit zero, publish the campaign completion
contract. The finalizer refuses stale `.part`/failed files, mismatched run IDs,
plan-row IDs, seeds, frozen hashes, route identity, or completion hashes. It
also independently hashes `runtime_readiness.json` and `raw/route_completion.json`.
The readiness report must be `enforce`/passing/evidence-eligible and timestamped
before the pre-run route manifest. At finalization its hash is bound to the
recorded detector runtime, model, and frozen configuration hashes. The finalizer
also recomputes route-wide camera health from every raw camera CSV: each source
must cover both route ends, keep frame ages fresh, and stay below the readiness
wall-gap limit for the whole route. A healthy startup window alone cannot
complete an evidence attempt.

```bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/finalize_campaign_run.py \
  --campaign-root "$campaign_root" --row-id "$row_id" --attempt-id "$attempt_id"
python3 experiments/multicamera_commissioning_bigwarehouse/tools/campaign_ledger.py validate \
  --campaign-root "$campaign_root" --rows
```

After both recorders finish, attach truth to *copies* of the camera CSVs for
the evaluation export. The operational CSVs stay unchanged.

```bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/attach_evaluation_truth.py \
  --raw-dir "$run_dir/raw" \
  --truth-csv "$run_dir/evaluation_only/ground_truth.csv" \
  --out-dir "$run_dir/evaluation_inputs" \
  --projection-role qualification
```

Export one CSV per camera into the shared replay timeline:

```bash
reliability_tools export-multicamera \
  --camera-csv camera_A=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/evaluation_inputs/camera_A_perception.csv \
  --camera-csv camera_B=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/evaluation_inputs/camera_B_perception.csv \
  --camera-csv camera_C=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/evaluation_inputs/camera_C_perception.csv \
  --camera-csv camera_D=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/evaluation_inputs/camera_D_perception.csv \
  --experiment-csv logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/raw/experiment.csv \
  --output-dir logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/export
```

The recorder derives projected world coordinates from known camera calibration;
it records no ground truth topic. Attach simulation truth only later in the
separate `evaluation_only/` export.

For a four-source replay, pass one frozen posterior for each camera rather than
a pooled map:

```bash
reliability_tools benchmark --export-dir <export-dir> --include-multicamera \
  --camera-gp camera_A=<A-posterior.npz> \
  --camera-gp camera_B=<B-posterior.npz> \
  --camera-gp camera_C=<C-posterior.npz> \
  --camera-gp camera_D=<D-posterior.npz>
```

## Detector capture provenance and merging

Every fresh detector capture is training-eligible only when its manifest pins
the capture-script SHA-256, exact command line and resolved defaults, and a
canonical inventory of the world, profile, route exclusions, recursively
referenced local Gazebo model packages (including meshes/textures), standard
simulation launch files, and robot-description assets. The capture hashes that
inventory again at the end and writes `.complete` only after all quality gates
pass. An interrupt or failure preserves the files by atomically moving the
directory to `<requested-output>.failed_<timestamp>` with
`.capture_failed.json`; that directory is diagnostic only.

Capture also fails closed unless the local transport boundary is explicit:
`ROS_LOCALHOST_ONLY=1`, `IGN_IP=127.0.0.1`, `GZ_IP=127.0.0.1`, an explicit
`ROS_DOMAIN_ID` (0–232), and a unique nonempty `IGN_PARTITION`. These values
are recorded in the manifest. Launch the simulator and its matching capture
client from shells inheriting the *same* values; for example set
`ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1 ROS_DOMAIN_ID=79
IGN_PARTITION=fourcam_capture_A_001` before starting both processes for camera
A. Start a fresh isolated partition for B/C/D rather than reusing that token.
The
`--allow-unisolated-transport` escape hatch is retained solely to preserve
diagnostic files: it stamps both manifest and completion marker as
non-training-eligible, so the merger rejects it.

`tools/merge_fourcam_yolo_dataset.py` requires this provenance and matching
capture-script/simulation-asset hashes across cameras by default. This is the
only merge mode eligible for training. The immutable v2 smoke captures predate
the contract and may be combined only to audit yield/geometry with the explicit
`--allow-legacy-diagnostic-provenance` flag. That override stamps both the
dataset card and merged manifest `diagnostic_legacy_non_training`; never pass
such a merge to `train_yolo_seg.py`, calibration, model selection, or paper
evidence.

## Historical pilot

The archived two-camera static pilot remains provenance only. It failed its D2
gate honestly: it was underpowered and exceeded the configured outlier limit.
It defines the measurement discipline—synchronised capture pairs, enough overlap
samples, and held-out disagreement checks—but cannot support a claim about this
four-camera world. Use `tools/make_pilot_showcase.py` only to reproduce that
archived historical artifact.

## Artifact layout

```text
logs/multicamera_commissioning_bigwarehouse/<run-id>/
  raw/                         # per-camera and experiment CSVs
  export/{operational,evaluation_only}/
  commissioning/               # camera audits and overlap graph
  replay/                      # selection / handover / failure summaries
  manifest.yaml
```

`operational/` may contain only manager-eligible information. Ground truth,
selection regret, NEES, and outcome labels remain evaluation-only.
