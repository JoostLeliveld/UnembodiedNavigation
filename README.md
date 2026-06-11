# Unembodied Navigation

ROS 2 / Gazebo experiments for visibility-aware navigation with an external
camera. The repository is being prepared as the code companion to a thesis /
paper, so the public surface is intentionally narrow:

- paper-facing world: `warehouse_aws.world.sdf`
- paper-facing campaign: four AWS route-choice tasks with five seeds per condition
- main comparison: `constant_R_efe` vs `visibility_aware_efe`

The method uses a learned, state-dependent detector reliability field to choose
the observation covariance used inside the EFE planner. This is not a direct
visibility reward; the known obstacle/no-go map remains a separate feasibility
layer shared by all compared planners. In the locked campaign, covariance also
enters the shared keep-in feasibility term through a belief tube.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/planning` | EFE planner, GP reliability loading, no-go cost, unicycle rollout |
| `src/perception` | YOLO-based external-camera robot detection |
| `src/state` | pixel-to-BEV projection and state publication |
| `src/sim` | Gazebo worlds, models, camera, robot description, simulator launch |
| `src/experiments` | tasks, world profiles, launch wiring, run logging |
| `scripts/visibility_comparison` | campaign, GP fitting, metrics, and paper-figure tooling |
| `scripts/perception` | detector dataset capture and training support |
| `docs` | paper/code alignment notes and publication checks |
| `archive` | legacy material kept for interpreting old runs only |

See `docs/active_research_state.md` and `docs/experiment_registry.md` for the
current paper-facing vs exploratory split.

## Main Paper Run

Build the workspace:

```bash
colcon build
source install/setup.bash
```

Run the locked AWS robustness campaign:

```bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/aws_f31b1_final_config.yaml \
  --log-root logs/visibility_comparison/aws_f31b1_final_v1
```

Generate paper metrics from a completed campaign:

```bash
python3 scripts/visibility_comparison/compute_paper_metrics.py \
  --campaign-log logs/visibility_comparison/aws_f31b1_final_v1/campaign_log.json \
  --gp-artifact logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz \
  --out logs/visibility_comparison/aws_f31b1_final_v1/paper_metrics.csv \
  --summary-out logs/visibility_comparison/aws_f31b1_final_v1/paper_summary.txt
```

Paper-packaged figures and metrics are kept under `logs/paper_figures/`; the
maintained figure scripts live in `scripts/paper_figures/`.

## Current Evidence Status

The paper-facing robustness campaign is packaged as:

- `logs/paper_figures/robustness_metrics.csv`
- `logs/paper_figures/robustness_summary.txt`
- `logs/paper_figures/robustness_spread.png`
- `logs/paper_figures/f31b1_markeroff_v2/paired_mechanism_taskA.pdf`

Current headline: C2 reaches `18/20` runs with `2/20` collisions; C1 reaches
`12/20` with one near-success and `7/20` collisions. b2 remains the hard case.
Continuous localization metrics are pooled over clean successes only.
## Publication Notes

- Do not cite generated logs or local weights unless they are included in the
  release artifact or linked from an archival store.
- Keep TeX claims aligned with `docs/paper_alignment.md`.
- Keep local caches, model weights, build directories, and raw logs out of the
  source release.
- A final release still needs an explicit license, citation metadata, and an
  artifact/data availability statement.
