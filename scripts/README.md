# `scripts`

This folder contains the offline and post-run scripts that support the current thesis milestone.

![Offline visibility artifact tutorial](../docs/figures/visibility_capture_tutorial.png)

![Planner timeseries tutorial](../docs/figures/planner_run_timeseries.png)

## Why This Folder Exists

The runtime ROS graph does not do everything. This folder covers:

- offline GP visibility artifact generation
- post-run summaries
- qualitative plotting
- calibration and diagnostic utilities

## Script Catalog

### High-Level Orchestration

| Script | Classification | When to use it | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| [`build_supervisor_aligned_seg_pipeline.py`](build_supervisor_aligned_seg_pipeline.py) | core | run the preferred supervisor-aligned segmentation pipeline (geometric boxes -> SAM masks -> YOLO11n-seg training) | live simulator or existing YOLO bbox dataset, SAM model | SAM-labeled YOLO-seg dataset, optional trained `best.pt`, deployment command |

### Dataset Capture & Conversion

| Script | Classification | When to use it | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| [`capture_yolo_dataset.py`](capture_yolo_dataset.py) | core | capture simulator frames and auto-label a one-class YOLO robot dataset with geometric bboxes | live Gazebo image stream, set-pose service, world profile | YOLO `images/`, `labels/`, `data.yaml` |
| [`capture_yolo_seg_dataset.py`](capture_yolo_seg_dataset.py) | ablation | capture simulator frames and auto-label visible red robot segmentation masks for comparison against the SAM path | live Gazebo image stream, set-pose service, world profile | YOLO-seg `images/`, polygon `labels/`, `data.yaml`, previews |
| [`convert_yolo_bbox_to_seg_by_red_mask.py`](convert_yolo_bbox_to_seg_by_red_mask.py) | ablation | reuse an existing bbox dataset and convert visible red robot pixels into segmentation polygons for legacy/red-mask comparisons | YOLO bbox dataset | YOLO-seg dataset copy |
| [`convert_yolo_labels_to_seg_by_sam.py`](convert_yolo_labels_to_seg_by_sam.py) | core | use existing YOLO bbox labels as full-image SAM prompts and generate validated segmentation pseudo-labels (recommended for supervisor-aligned pipeline) | YOLO bbox dataset, SAM model | YOLO-seg dataset copy with SAM masks |
| [`filter_yolo_dataset_by_red_visibility.py`](filter_yolo_dataset_by_red_visibility.py) | core | clean geometric bbox labels by blanking boxes with little visible red robot evidence | YOLO bbox dataset | filtered dataset copy |

### Model Training & Evaluation

| Script | Classification | When to use it | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| [`train_yolo_robot.py`](train_yolo_robot.py) | core | train YOLO11n detect or YOLO11n-seg segment robot models | YOLO `data.yaml` | Ultralytics training run with `best.pt` |
| [`inspect_yolo_dataset.py`](inspect_yolo_dataset.py) | utility | analyze dataset structure, labels, and metadata | dataset directory (with `data.yaml`) | formatted statistics and file listings |
| [`benchmark_yolo_models.py`](benchmark_yolo_models.py) | core | compare YOLO model refs on the same saved frames | YOLO dataset or image directory, model refs | prediction CSV, summary CSV, bbox/mask previews |

### Visibility Estimation

| Script | Classification | When to use it | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| [`fit_empirical_visibility_gp.py`](fit_empirical_visibility_gp.py) | core | collect a simulated capture dataset and fit the per-world empirical visibility artifact | sampled robot poses or live `/state/bev`, detection diagnostics, world profile | `empirical_visibility_gp.npz`, raw/aggregated capture CSVs, fit plot |

### Analysis & Evaluation

| Script | Classification | When to use it | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| [`evaluate_occlusion_comparison.py`](evaluate_occlusion_comparison.py) | core | summarize logged runs for the main comparison | `logs/experiments/*` | run/group summary CSV and JSON |
| [`plot_visibility_run.py`](plot_visibility_run.py) | core | generate qualitative figures for one run | run log directory + visibility artifact | trajectory/visibility plots |
| [`generate_docs_figures.py`](generate_docs_figures.py) | core | regenerate the tutorial figures embedded in README/docs | latest capture artifact + latest logged run | `docs/figures/*.png` |

### Calibration

| Script | Classification | When to use it | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| [`capture_noise_data.py`](capture_noise_data.py) | calibration | collect streams for Q/R calibration | live ROS topics | capture CSV |
| [`estimate_noise_from_capture.py`](estimate_noise_from_capture.py) | calibration | estimate process/observation noise from a capture CSV | capture CSV, world profile, world SDF | JSON summary or console estimates |

## Support And Archive Material

- `artifacts/visibility_prior/`
  - local scratch/output area if you generate one manually; not part of the active thesis path
- `figures/`
  - currently not part of the active scripted story

## What To Read First

**For the supervisor-aligned segmentation pipeline:**
1. `build_supervisor_aligned_seg_pipeline.py --stage all` - Run the geometric-box -> SAM-mask -> YOLO11n-seg pipeline
2. `convert_yolo_labels_to_seg_by_sam.py` - Inspect the SAM pseudo-label conversion stage
3. `train_yolo_robot.py` - Train or retrain the runtime YOLO-seg model

**For inspecting the training process:**
1. `inspect_yolo_dataset.py` - Analyze any generated YOLO dataset
2. `benchmark_yolo_models.py` - Compare trained YOLO models on the same saved frames

**For visibility estimation:**
1. `fit_empirical_visibility_gp.py`
2. `evaluate_occlusion_comparison.py`
3. `plot_visibility_run.py`

## How This Folder Connects To The Rest Of The Repository

- reads world/profile information from `src/experiments/config`
- loads or writes visibility artifacts in `src/experiments/data/visibility_gp`
- consumes logs written by `experiment_logger`
- uses planning and geometry helpers during offline fitting and plotting

## Important Caveats

- The GP-fitting script generates a learned visibility prior for the current simulated camera-detector stack.
- The GP input is `/state/bev` x-y, and the fitter supports `normalized_blob_area`, `binary_usable_detection`, and YOLO `yolo_soft_score` target modes.
- The fitter can run live in `teleport` or `driving` capture mode, or offline from an existing capture directory via `--from-capture-dir`.
- The YOLO score is a confidence-derived soft target, not a calibrated probability unless a later calibration step is added.
- Red-mask segmentation scripts are legacy/ablation utilities; they are not the preferred training path.
- The SAM-prompted segmentation converter uses simulator-projected robot boxes as offline prompts, but those masks are still pseudo-labels and should be spot-checked in the generated previews.
- The preferred supervisor-aligned path is geometric robot boxes -> SAM pseudo-masks -> YOLO11n-seg training -> runtime mask-bottom homography with `yolo_score` and `confidence_logit` logged for analysis.
- The evaluator is useful for milestone summaries, but not yet a complete thesis-final analysis suite.
- Calibration scripts are secondary to the main comparison and should not dominate the repository story.

## Recommended Future Reorganization

The current folder is readable enough for the milestone, but a cleaner split would be:

- `scripts/core/`
- `scripts/calibration/`
- `scripts/exploratory/`
- `scripts/archive/`

That split is recommended for future cleanup, but not required for the current documentation pass.
