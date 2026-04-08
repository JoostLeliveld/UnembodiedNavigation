# `scripts`

This folder contains the offline and post-run scripts that support the current thesis milestone.

## Why This Folder Exists

The runtime ROS graph does not do everything. This folder covers:

- offline GP visibility artifact generation
- post-run summaries
- qualitative plotting
- calibration and diagnostic utilities

## Script Catalog

| Script | Classification | When to use it | Main inputs | Main outputs |
| --- | --- | --- | --- | --- |
| [`fit_empirical_visibility_gp.py`](fit_empirical_visibility_gp.py) | core | build the per-world empirical visibility artifact | selected world, captured detections, world profile | `empirical_visibility_gp.npz`, capture CSVs, fit plot |
| [`evaluate_occlusion_comparison.py`](evaluate_occlusion_comparison.py) | core | summarize logged runs for the main comparison | `logs/experiments/*` | run/group summary CSV and JSON |
| [`plot_visibility_run.py`](plot_visibility_run.py) | core | generate qualitative figures for one run | run log directory + visibility artifact | trajectory/visibility plots |
| [`capture_noise_data.py`](capture_noise_data.py) | calibration | collect streams for Q/R calibration | live ROS topics | capture CSV |
| [`estimate_noise_from_capture.py`](estimate_noise_from_capture.py) | calibration | estimate process/observation noise from a capture CSV | capture CSV, world profile, world SDF | JSON summary or console estimates |

## Support And Archive Material

- `artifacts/visibility_prior/`
  - local scratch/output area if you generate one manually; not part of the active thesis path
- `figures/`
  - currently not part of the active scripted story

## What To Read First

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
- The evaluator is useful for milestone summaries, but not yet a complete thesis-final analysis suite.
- Calibration scripts are secondary to the main comparison and should not dominate the repository story.

## Recommended Future Reorganization

The current folder is readable enough for the milestone, but a cleaner split would be:

- `scripts/core/`
- `scripts/calibration/`
- `scripts/exploratory/`
- `scripts/archive/`

That split is recommended for future cleanup, but not required for the current documentation pass.
