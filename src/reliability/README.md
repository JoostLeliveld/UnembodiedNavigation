# Reliability Contracts

This package defines ROS-independent Phase 0 contracts for future external-camera
reliability learning. It does not change planner, controller, EKF, no-go geometry,
GP artifacts, or active campaign behavior.

## Claim

A reliability model may estimate whether an external-camera detection is available
and how much covariance inflation should be used from fields that would be
available in a warehouse deployment: image/detector diagnostics, selected pixel
metadata, projection status, timing, odometry/state/belief summaries, camera-relative
features, recent detector history, and configuration or artifact hashes.

Ground truth, oracle visibility, collisions, clearance, and final task outcomes are
evaluation labels only. They may be used to audit or train later experiments, but
they must not enter planner-facing features or normal runtime configuration.

## Realistic Assumptions

- The robot receives camera detections through the selected pixel path used by the
  current YOLO detector and BEV projection.
- Odometry, planner belief, detector diagnostics, and measurement age are available
  online.
- Configuration and artifact hashes identify the runtime surface without exposing
  oracle labels.
- Evaluation jobs may read Gazebo truth and collision/clearance metrics when clearly
  labeled as evaluation or controlled oracle contexts.

## Non-Assumptions

- No planner-facing reliability provider may depend on `/ground_truth_tf`.
- No normal reliability source may use Gazebo dynamic pose, oracle visibility, GT
  pixel errors, collisions, clearances, geometry breaches, or final outcomes.
- No Phase 0 code assumes learned reliability models already exist.
- No Phase 0 code changes the active warehouse campaign configuration or GP map.

## Literature Anchor

The contracts follow the common separation between deployment features and
evaluation labels: covariate inputs must be observable at runtime, while labels and
oracle diagnostics are isolated for validation, calibration, and reporting. Later
phases can attach specific papers or model families to these stable records.

## Inputs

`OperationalReliabilitySample` contains only online-observable inputs:

- run/task/seed identifiers and timestamps
- image reference and content hash
- detector diagnostics and selected pixel
- projection validity and staleness
- odometry, state estimate, and planner belief summaries
- camera-relative range and normalized image-location features
- recent detector history
- config and artifact hashes

`CameraObservation` makes the current external camera look like one member of a
future camera network:

- `camera_id`, timestamp, calibration ID, and image frame ID
- selected pixel, detector validity, detector score, and optional bbox
- measurement age
- conditional pixel covariance
- availability and association probabilities

`CameraQuality` is the compact per-camera quality summary that selection/fusion
policies can consume without knowing whether the source was fixed-R, detector
score, GP reliability, or a future temporal model.

`reliability.single_camera_adapter` is Module 1 of the extension. It converts
the current single-camera detector diagnostics and planner visibility diagnostics
into `CameraObservation` and `CameraQuality` without changing runtime behavior.
It preserves the existing scalar-trust precision blend for pixel covariance.

`reliability.providers` defines the operational camera reliability provider
interface. It includes fixed camera quality, current-GP-style `.npz` grid-map
quality, and multi-camera dispatch providers. These providers read only runtime
state such as `belief_xy`; they do not read GT, oracle, clearance, collision, or
final-outcome labels.

`reliability.prior` builds day-zero per-camera reliability priors from known
calibration only: FOV, projected pixel location, image-border margin,
camera-relative range, ground incidence, and projection scale. It fuses multiple
cameras with union-of-success and best-camera maps while preserving the
per-camera maps for later learning.

`reliability.learning` updates one Beta-Bernoulli reliability field per camera
from detector hit/miss evidence. Observations never update another camera's map;
multi-camera union/best maps are recomputed only after per-camera posteriors are
formed.

`reliability.bev_reliability` adds an offline occlusion-aware BEV reliability
network surface. It turns each calibrated camera into an operational feature
token, scores those tokens with a small linear model, and fuses the camera set
with soft attention plus union/best-camera probabilities. It is deliberately not
a raw-image transformer; it keeps the thesis framing on navigation reliability,
visibility, and planner-facing covariance.

`reliability.handover` quantifies temporary source-switch uncertainty when a
selected observation changes from one camera to another. It uses only operational
signals: previous/selected camera IDs, near-synchronous A/B map disagreement,
timestamp gap, staleness, and per-camera quality. Its output is a diagnostic plus
a covariance inflation factor for the selected map observation.

`EvaluationOnlySample` contains labels and audit metrics:

- Gazebo ground-truth pose and projected pixel
- GT localization error
- clearance, collision, and geometry-breach fields
- final task outcome and aggregate metrics

## Outputs

`ReliabilityPrediction` reports availability probability and optional quality or
uncertainty summaries.

`CameraObservation` and `CameraQuality` report camera-specific operational
evidence for future selection/fusion work. The current camera should be treated as
`camera_A` when adapting existing logs or runtime nodes.

`CameraHealthMachine` implements the single-camera loss/recovery states:
`TRACKING`, `DEGRADED`, `LOST`, and `REACQUIRING`.

`reliability.fusion` implements simple camera baselines before learned fusion:
primary camera, fixed zones, highest detector score, freshest valid observation,
best static reliability, conservative best camera, and sequential 2D Kalman
updates.

`reliability.replay` runs R0-R4-style offline estimator comparisons from split
operational/evaluation records.

`reliability.benchmark` wraps replay into deterministic suites. By default it
runs the required R0-R4 baselines; when frames contain multiple cameras it can
also include M5 sequential fusion, M6 conservative best-camera selection, and
M7 handover-aware selection.

`reliability.overlap` validates two-camera calibration/overlap agreement from
operational map estimates. It reports A/B disagreement, systematic B-minus-A
bias, outlier rate, and pair-trust diagnostics without consuming GT.

`reliability.handover` bridges overlap diagnostics to estimator behavior. During
an A-to-B handover it can leave an agreeing, near-synchronous switch almost
unchanged, or inflate covariance when the switch has no overlap confirmation,
large disagreement, stale selected data, or a quality drop.

`reliability.bev_reliability` emits `CameraQuality` and BEV grid-provider
outputs from calibrated priors plus operational detector/overlap features, so it
can be ablated against the current GP reliability map before runtime planner
integration.

`reliability.oracle` is evaluation-only feasibility tooling for hypothetical
second-camera coverage. It consumes truth trajectories or hand-labeled oracle
regions and must remain outside normal planner-facing providers and feature
loaders.

`UpdateCovariance` reports the covariance used by an estimator update when an image
observation is actually consumed.

`PlanningCovariance` reports planner-facing covariance and availability terms for
risk/ambiguity scoring.

## Units

- Timestamps and ages are seconds.
- Pixel coordinates are image pixels, ordered `(u, v)`.
- Bounding boxes are image pixels, ordered `(x_min, y_min, x_max, y_max)`.
- Covariance matrices are 2x2 and in `px^2`.
- Range and clearance metrics are meters.
- Yaw fields, when present inside nested summaries, are radians.
- Probabilities are in `[0, 1]`.

## Validation Gate

The leakage firewall fails when:

- operational feature columns include GT, oracle, collision, clearance, geometry, or
  final-outcome fields
- normal training loaders name GT/oracle topics or paths
- planner-facing modules import evaluation-only loaders
- normal runtime configs enable truth localization or GT/oracle reliability sources
- covariance records are not finite, 2x2, symmetric positive definite matrices in
  `px^2`

Run the Phase 0 smoke gate with:

```bash
python3 -m pytest tests/reliability/test_contracts.py tests/reliability/test_leakage_firewall.py tests/visibility_comparison/test_current_runtime_contract.py
```

Export and replay a completed run directory with:

```bash
reliability_tools export-run \
  --run-dir logs/experiments/experiment_XXXXX \
  --output-dir logs/reliability_exports/experiment_XXXXX

reliability_tools replay \
  --export-dir logs/reliability_exports/experiment_XXXXX \
  --gp-artifact paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz

reliability_tools benchmark \
  --export-dir logs/reliability_exports/experiment_XXXXX \
  --gp-artifact paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz

reliability_tools export-multicamera \
  --camera-csv camera_A=logs/multicamera/run_001/camera_A_perception.csv \
  --camera-csv camera_B=logs/multicamera/run_001/camera_B_perception.csv \
  --experiment-csv logs/multicamera/run_001/experiment.csv \
  --output-dir logs/reliability_exports/run_001_multicamera

reliability_tools validate-overlap \
  --export-dir logs/reliability_exports/run_001_multicamera \
  --max-disagreement-m 0.30
```

Example handover covariance inflation for an offline fusion/replay step:

```python
from reliability import handover_adjusted_observation

adjusted_observation, diagnostic = handover_adjusted_observation(
    previous_camera_id="camera_A",
    selected_observation=camera_b_map_observation,
    candidate_observations=(camera_a_map_observation, camera_b_map_observation),
)
```

Offline multi-camera initialization and learning can be tested without planner
integration:

```python
from reliability import (
    CameraCalibration,
    LearningConfig,
    PriorGridSpec,
    ReliabilityObservation,
    build_multicamera_geometry_prior,
    learn_per_camera_reliability,
)

grid = PriorGridSpec.from_bounds(x_min=-5.5, x_max=5.5, y_min=-5.0, y_max=5.0, nx=220, ny=200)
prior = build_multicamera_geometry_prior([camera_a, camera_b], grid)
learned = learn_per_camera_reliability(
    prior,
    [ReliabilityObservation("camera_A", (1.0, 0.5), 1.0)],
    LearningConfig(prior_strength=3.0),
)
```

A lightweight camera-set attention baseline can score the same calibrated BEV
state without raw images:

```python
from reliability import BEVReliabilityModel, bev_tokens_from_prior_maps

tokens = bev_tokens_from_prior_maps(
    prior,
    (1.0, 0.5),
    camera_measurements={
        "camera_A": {"detector_score": 0.2, "detection_valid": False},
        "camera_B": {"detector_score": 0.9, "detection_valid": True},
    },
)
prediction = BEVReliabilityModel().predict_set(tokens)
```

## Baselines

The package preserves the existing baseline surfaces:

- constant image covariance from the current planner configuration
- GP visibility map output already used by planning diagnostics
- per-camera GP-grid reliability providers that load current artifact-shaped
  `.npz` files without importing planner code
- current YOLO detector diagnostics and selected bbox-bottom pixel source
- current `warehouse_visibility_campaign.yaml` as the GT-free normal runtime config

Later phases may add real camera-B detector/calibration logs and learned adapters
against these contracts.
