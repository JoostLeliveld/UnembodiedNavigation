# Scripts

Offline tooling around the ROS/Gazebo runtime: capture data, train the detector, run
campaigns, fit models.

**Organised by the stage each family serves.** The active stage is camera/YOLO
characterization. A script's existence does not make its output a current result; `PLAN.md`
defines the gates between characterization, covariance estimation, availability, fusion,
and planning. This file was rewritten on 2026-08-29 because six of the seven scripts it used
to list had been deleted.

## Fusion on a fixed route  (downstream; diagnostic until the earlier gates pass)

| purpose | file |
|---|---|
| the campaign runner | `visibility_comparison/run_visibility_campaign.py` |
| the live campaign | `visibility_comparison/fusion_realistic_speed_n1_campaign.yaml` |
| the five-seed design | `visibility_comparison/fusion_on_fixed_routes_campaign.yaml` |
| covariance ablation (K0–K2) | `visibility_comparison/measurement_covariance_ablation_campaign.yaml` |
| heading ablation (H0/H1) | `visibility_comparison/heading_update_ablation_campaign.yaml` |
| watch a campaign while it runs | `visibility_comparison/monitor_campaign.py` |

Scoring never happens here — it happens in `experiments/fusion_on_fixed_routes/`, through
`aligned.py` against a frozen run manifest.

## Frozen sensor  (shared by every direction)

The detector is trained once and never retrained after anything is measured against it.

| purpose | file |
|---|---|
| dataset capture and labelling | `perception/capture_yolo_dataset.py`, `relabel_from_masks.py`, `extract_yolo_contours.py` |
| detector training | `perception/train_yolo_detector.sh`, `train_yolo_seg.py` |
| the pipeline, written down | `perception/YOLO_DATASET_PIPELINE.md` |

## Learned bias / shape correction  (decision-gated)

A network that maps a detection to a position better than projecting the robot's shape does.
Best measured result is a set-pose held-out study with an oracle commanded heading, marked
provisional: analytic hull 3.11 cm mean against 2.04 cm for `mlp_without_shape`.

| purpose | file |
|---|---|
| residual-bias model | `perception/{build_residual_bias_dataset,train_residual_bias,residual_bias_model,evaluate_residual_bias}.py` |
| shape-conditioned update | `perception/{build_shape_update_dataset,train_shape_update,shape_conditioned_update_model,evaluate_shape_update}.py` |
| centre keypoint | `perception/{build_center_keypoint_dataset,train_center_keypoint,evaluate_center_keypoint}.py` |
| contour update | `perception/{build_contour_update_dataset,contour_update_model}.py` |

**Nothing here is wired into the runtime.** A learned method may enter the characterization
registry only when its artifact, held-out split, required online inputs, and runtime output
are frozen. It becomes a correction candidate only if the bias gate justifies correction.

## Availability-aware planning  (downstream; waiting on operational labels)

| purpose | file |
|---|---|
| export the observation dataset | `reliability/export_observation_dataset.py` |
| fit the availability field | `reliability/run_observation_gp.py` |
| build the planner-facing artifact | `reliability/build_planner_p_use_artifacts.py` |
| belief-aware GP variants | `visibility_comparison/fit_belief_aware_gp.py` |

**Blocked on data that does not exist**: driven operational logs whose usable-sighting labels
are computed the way the runtime computes them, with no ground truth. Ground-truth
characterization may evaluate this stage but may not supply its online labels; see `PLAN.md`,
"Stage 4".

## Shared

| purpose | file |
|---|---|
| scoring functions — never re-implement these inline | `shared/metrics.py` |
| locating the checkout | `shared/paths.py` |
| per-timestep campaign-CSV column safety (diagnostic only; no time alignment, no dedupe) | `geometry_visibility/campaign_metrics.py` |
| world assets fetch | `sim/fetch_external_models.sh` |
| camera-stream recording for figures | `paper_figures/record_camera_stream.py` |
