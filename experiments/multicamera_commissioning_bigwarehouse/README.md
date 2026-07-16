# Full-Warehouse Four-Camera Commissioning

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
  --out-dir logs/studies/multicamera_commissioning_bigwarehouse/paper_protocol_v1
```

This creates a separate folder containing `campaign_plan.csv`, a collection
checklist, gate status, route/qualification figures, and talking points. It
does not allow the sparse pilot to pass any gate.

## Collection protocol

For each frozen plan row, start Gazebo at the documented route spawn, record
operational data, and run the fixed route driver. Do not add or move a route
after inspecting evaluation-only results.

```bash
# Terminal 1 — start at the route's documented spawn pose
source install/setup.bash
ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
  headless:=true \
  spawn_x:=<route-start-x> spawn_y:=<route-start-y> spawn_yaw:=<route-start-yaw> \
  yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt

# Terminal 2 — operational record only
source install/setup.bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/record_operational_logs.py \
  --out-dir logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/raw \
  --duration-s 120 \
  --odom-origin-x-m=<route-start-x> \
  --odom-origin-y-m=<route-start-y> \
  --odom-origin-yaw-rad=<route-start-yaw>

# Terminal 3 — only after the named route exists in study.yaml
source install/setup.bash
python3 experiments/multicamera_commissioning_bigwarehouse/tools/drive_study_route.py \
  --route <route-name> --lateral-offset-m 0.0 --speed-mps 0.12 \
  --manifest-path logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/raw/route_manifest.json
```

Export one CSV per camera into the shared replay timeline:

```bash
reliability_tools export-multicamera \
  --camera-csv camera_A=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/raw/camera_A_perception.csv \
  --camera-csv camera_B=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/raw/camera_B_perception.csv \
  --camera-csv camera_C=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/raw/camera_C_perception.csv \
  --camera-csv camera_D=logs/studies/multicamera_commissioning_bigwarehouse/<campaign>/<run-id>/raw/camera_D_perception.csv \
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
