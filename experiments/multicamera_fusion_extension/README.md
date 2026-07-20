# Multicamera Fusion Extension

**Paper-extension plans (2026-07-17):** `plans/` holds the module-by-module
implementation plan for the *Spatial and Instantaneous Reliability-Aware
Multi-Camera Fusion and Planning* paper (Toro-Diz-baseline extension). Start
at `plans/00_REPO_MAPPING.md` (where everything lives — no repo restructure)
and `plans/ROADMAP.md` (dependency spine, baselines B0–B9, experiments E0–E8,
pre-registered "beats Toro" criteria). Library modules land in
`src/reliability/reliability/`, study CLIs in `tools/`, outputs in
`logs/studies/multicamera_fusion_extension/`.

This namespace is for the focused extension:

```text
camera-specific reliability
-> robust multi-camera selection/fusion
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
| 8. Canonical 4-camera world and detector launch | Implemented, not claimed as result | `warehouse_full_4cam.world.sdf` is the forward world; `warehouse_multicamera_extension.launch.py` starts four YOLO detector instances with isolated camera topics. |
| 9. Calibration/overlap validation | Implemented offline | `reliability_tools validate-overlap` checks A/B map-estimate disagreement and overlap trust diagnostics from multicamera replay frames without GT. |
| 10. Real multi-camera replay export | Implemented offline | `reliability_tools export-multicamera` writes split operational/evaluation records and shared replay frames for M5/M6 benchmarking. |
| 11. Calibrated multi-camera day-zero prior | Implemented offline | `reliability.prior` builds per-camera known-calibration priors and fuses them with union/best-camera maps. |
| 12. Per-camera reliability learning | Implemented offline | `reliability.learning` updates one Beta field per camera before fusing posteriors, so camera-specific failures do not contaminate other camera maps. |
| 13. Occlusion-aware BEV reliability network | Implemented offline | `reliability.bev_reliability` scores calibrated per-camera BEV tokens and fuses them with camera-set attention, while preserving the navigation/reliability framing. |
| 14. Handover uncertainty | Implemented offline | `reliability.handover` scores source-switch uncertainty from A/B disagreement, staleness, quality drop, and missing overlap confirmation, then inflates map-observation covariance. |
| 15. Real camera-B GP reliability | Pending | Requires real camera-B logs with operational detector evidence before fitting. |

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

Run the wider benchmark suite. R0-R4 are always included; M5/M6/M7 are included
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

Launch the extension-only four-camera detector stack:

```bash
ros2 launch experiments warehouse_multicamera_extension.launch.py \
  yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
  headless:=true
```

Build the day-zero planner/research prior for the canonical four-camera world:

```bash
python3 scripts/geometry_visibility/build_full4cam_planner_prior.py
```

The planner-facing field is camera-A-compatible until a fused runtime
observation topic replaces `/perception/pixel_pose`; the same artifact stores
four-camera union/best maps for offline fusion and handover studies.

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

Assess source-switch uncertainty before feeding a selected camera observation to
fusion/replay:

```python
from reliability import handover_adjusted_observation

adjusted_observation, diagnostic = handover_adjusted_observation(
    previous_camera_id="camera_A",
    selected_observation=camera_b_map_observation,
    candidate_observations=(camera_a_map_observation, camera_b_map_observation),
)
```

The diagnostic is operational-only. It asks: did the camera source change, did
the old and new cameras agree in the overlap, were their timestamps close, did
quality drop, and was the selected measurement stale? The returned observation
keeps the same mean but inflates covariance until the handover is well supported.

Record synchronized camera-view clips for a live four-camera Gazebo run:

```bash
source install/setup.bash
python3 scripts/reliability/record_multicamera_views.py \
  --out-dir logs/warehouse_full_4cam/videos/live_camera_views \
  --camera camera_A=/external_camera/image_raw \
  --camera camera_B=/external_camera_b/image_raw \
  --camera camera_C=/external_camera_c/image_raw \
  --camera camera_D=/external_camera_d/image_raw \
  --duration-s 30 \
  --every 2 \
  --write-mp4
```

## Offline Research Steps 1-3

The extension starts as an experiment framework, not a planner replacement.

1. Build a calibrated day-zero prior from known camera intrinsics/extrinsics:

```python
from reliability import (
    CameraCalibration,
    PriorGridSpec,
    build_multicamera_geometry_prior,
)

grid = PriorGridSpec.from_bounds(x_min=-5.5, x_max=5.5, y_min=-5.0, y_max=5.0, nx=220, ny=200)
prior = build_multicamera_geometry_prior([camera_a, camera_b], grid)
```

This produces per-camera maps plus `union_probability`, `best_probability`, and
`best_camera_id`. The important claim is not "more cameras are always better";
it is "known calibration gives a measurable day-zero coverage/reliability prior
before any detector learning."

2. Learn reliability per camera, then fuse:

```python
from reliability import LearningConfig, ReliabilityObservation, learn_per_camera_reliability

result = learn_per_camera_reliability(
    prior,
    [
        ReliabilityObservation("camera_A", (1.0, 0.5), 1.0),
        ReliabilityObservation("camera_B", (1.0, 0.5), 0.0),
    ],
    LearningConfig(prior_strength={"camera_A": 3.0, "camera_B": 3.0}),
)
```

Camera observations update only their own field. Fusion happens after learning
through the same union/best-camera maps.

3. Use overlap as a calibration/trust signal:

```python
from reliability import overlap_trust_from_frames

trust = overlap_trust_from_frames(frames, disagreement_gate_m=0.30)
```

The overlap summary reports p90 disagreement, outlier rate, systematic
B-minus-A bias, and a pair-trust score. It does not use ground truth and it does
not decide which camera is absolutely correct; it quantifies whether the pair is
consistent enough for fusion ablations.

4. Score candidate navigation states with a small BEV reliability network:

```python
from reliability import BEVReliabilityModel, bev_tokens_from_prior_maps

tokens = bev_tokens_from_prior_maps(
    prior,
    (1.0, 0.5),
    camera_measurements={
        "camera_A": {"detector_score": 0.1, "detection_valid": False},
        "camera_B": {"detector_score": 0.8, "detection_valid": True},
    },
)
prediction = BEVReliabilityModel().predict_set(tokens)
```

This is the thesis-sized network baseline: calibration projects each camera into
the same BEV navigation frame, operational detector features modulate reliability,
and attention exposes which camera the model trusts. It can produce
planner-facing `CameraQuality` or a BEV grid provider, but it does not consume
raw images or evaluation-only labels.

## Handover Uncertainty Question

The next thesis-sized question is narrower than "does multi-camera help?":

> During an A-to-B camera handover, how much temporary covariance inflation is
> needed to avoid overconfident localization when the two cameras disagree, are
> asynchronous, or lack an overlap confirmation?

This isolates the new failure mode introduced by multiple cameras. More cameras
reduce blind regions, but they also create source-switch uncertainty: the robot
may jump between two biased map estimates exactly at the aisle boundary where
single-camera reliability was already weak. The first offline answer should
compare naive immediate switching against handover-inflated switching on replay
frames, using overlap disagreement and NIS/NEES as the calibration metrics.

## Claim Discipline

Multi-camera claims remain planned or hypothetical until real camera-B logs,
matched replay/campaign results, and evaluation-only metrics exist. The current
single-camera result remains the core thesis evidence until this extension earns
its own chain of artifacts.
