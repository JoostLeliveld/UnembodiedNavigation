# Multicamera Fusion Extension

This namespace is for the focused extension:

```text
camera-specific reliability
-> robust two-camera selection/fusion
-> continuity through occlusion and handover
```

It is not a license to turn the thesis into full CCTV graph optimization, LiDAR
mapping, transformer fusion, fleet coordination, or energy-aware scheduling.

## Research Question

How can camera-specific reliability models support robust localization from
multiple external cameras under occlusion, handover, delayed measurements, and
camera failure?

Planning remains a downstream consumer of the fused belief. The extension should
not introduce a new planner objective, active camera scheduling, dynamic obstacles,
or a new controller while testing localization robustness.

## Ordered Gates

1. Represent the current camera as `camera_A` through `CameraObservation` and
   `CameraQuality`.
2. Prove the camera-interface runtime reproduces the current single-camera
   behavior before adding camera B.
3. Build offline replay with separate `operational/` and `evaluation_only/`
   records.
4. Test single-camera loss/recovery before multi-camera fusion.
5. Use an evaluation-only oracle second-camera feasibility test before building a
   real second camera.
6. Add exactly one real second external camera only after the oracle test shows
   useful redundancy.
7. Compare simple policies before learned reliability: primary camera, fixed
   zones, detector score, freshness, static reliability, sequential updates, and
   conservative best-camera selection.
8. Train per-camera GP reliability before any temporal network.

## Module Status

| Module | Status | Notes |
| --- | --- | --- |
| 1. Single-camera adapter | Implemented | `reliability.single_camera_adapter` converts current detector/planner diagnostics into camera contracts without runtime rewiring. |
| 2. Runtime camera-interface pass-through | Implemented, opt-in | `yolo_robot_detector_node` can publish `/perception/camera_observation/<camera_id>` JSON when `publish_camera_observation_json=true`; default is off. |
| 3. Offline replay exporter/runner | Implemented | `reliability_tools export-run` writes split `operational/` and `evaluation_only/`; `reliability_tools replay` runs R0-R4 replay configs. |
| 4. Single-camera health state | Implemented | `CameraHealthMachine` covers `TRACKING`, `DEGRADED`, `LOST`, and `REACQUIRING` from operational misses, age, NIS, and reliability. |
| 5. Simple camera selection/fusion | Implemented core | Primary, fixed-zone, detector-score, freshest, static-reliability, conservative-best, and sequential 2D Kalman fusion policies are available in `reliability.fusion`. |
| 6. Evaluation-only camera-B feasibility | Implemented offline | `reliability.oracle` labels hypothetical camera availability and dropout coverage as `evaluation_only_oracle`; it is blocked from planner-facing imports. |
| 7. Per-camera reliability provider interface | Implemented offline | `reliability.providers` supports fixed quality, current-GP-shaped `.npz` grid maps, and multi-camera dispatch. |
| 8. Extension world and two-detector launch | Implemented, not claimed as result | `warehouse_multicamera_extension.world.sdf` includes `external_camera` and isolated `external_camera_b`; `warehouse_multicamera_extension.launch.py` starts two YOLO detector instances with isolated topics. |
| 9. Calibration/overlap validation | Implemented offline | `reliability_tools validate-overlap` checks A/B map-estimate disagreement from multicamera replay frames without GT. |
| 10. Real multi-camera replay export | Implemented offline | `reliability_tools export-multicamera` writes split operational/evaluation records and shared replay frames for M5/M6 benchmarking. |
| 11. Real camera-B GP reliability | Pending | Requires real camera-B logs with operational detector evidence before fitting. |

## Working Commands

Export a completed current-style run directory:

```bash
reliability_tools export-run \
  --run-dir logs/experiments/experiment_XXXXX \
  --output-dir logs/reliability_exports/experiment_XXXXX
```

Run the required replay baselines on the split export:

```bash
reliability_tools replay \
  --export-dir logs/reliability_exports/experiment_XXXXX \
  --gp-artifact paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz
```

Run the wider benchmark suite. R0-R4 are always included; M5/M6 are included
automatically when replay frames contain multiple camera observations:

```bash
reliability_tools benchmark \
  --export-dir logs/reliability_exports/experiment_XXXXX \
  --gp-artifact paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz
```

Opt into the runtime camera-observation pass-through:

```text
publish_camera_observation_json:=true
camera_id:=camera_A
```

This only adds a JSON observation topic. It does not replace the existing
`/perception/pixel_pose` or EKF/planner correction path.

Launch the extension-only two-camera detector stack:

```bash
ros2 launch experiments warehouse_multicamera_extension.launch.py \
  yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
  headless:=true
```

Export real camera-A/B perception logs into shared replay frames:

```bash
reliability_tools export-multicamera \
  --camera-csv camera_A=logs/multicamera/run_001/camera_A_perception.csv \
  --camera-csv camera_B=logs/multicamera/run_001/camera_B_perception.csv \
  --experiment-csv logs/multicamera/run_001/experiment.csv \
  --output-dir logs/reliability_exports/run_001_multicamera
```

Run the A/B overlap calibration gate:

```bash
reliability_tools validate-overlap \
  --export-dir logs/reliability_exports/run_001_multicamera \
  --max-disagreement-m 0.30
```

## Claim Discipline

Multi-camera claims remain planned or hypothetical until real camera-B logs,
matched replay/campaign results, and evaluation-only metrics exist. The current
single-camera result remains the core thesis evidence until this extension earns
its own chain of artifacts.
