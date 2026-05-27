# Unembodied Navigation

ROS 2 / Gazebo experiments for visibility-aware navigation with an external
camera. The repository is being prepared as the code companion to a thesis /
paper, so the public surface is intentionally narrow:

- compact reported benchmark: `warehouse_occ_light.world.sdf`
- main paper task: `shadow_tradeoff_a`
- main comparison: `constant_R_efe` vs `visibility_aware_efe`
- exploratory extension: `warehouse_aws.world.sdf` for Experiment B

The method uses a learned, state-dependent detector reliability field to choose
the observation covariance used inside the EFE planner. This is not a direct
visibility reward; the known obstacle/no-go map remains a separate feasibility
layer shared by all compared planners.

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

Run the locked compact benchmark campaign:

```bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/paper_campaign_config.yaml \
  --log-root logs/visibility_comparison/paper_campaign_rawgp_v1
```

Generate paper metrics from a completed campaign:

```bash
python3 scripts/visibility_comparison/compute_paper_metrics.py \
  --campaign-log logs/visibility_comparison/paper_campaign_rawgp_v1/campaign_log.json \
  --gp-artifact logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz \
  --out logs/visibility_comparison/paper_campaign_rawgp_v1/paper_metrics.csv \
  --summary-out logs/visibility_comparison/paper_campaign_rawgp_v1/paper_summary.txt
```

Generate thesis/paper figures:

```bash
python3 scripts/visibility_comparison/thesis_plots/make_thesis_figures.py \
  --campaign-log logs/visibility_comparison/paper_campaign_rawgp_v1/campaign_log.json \
  --metrics-csv logs/visibility_comparison/paper_campaign_rawgp_v1/paper_metrics.csv \
  --gp-artifact logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz
```

## Experiment B Status

`warehouse_aws.world.sdf` is the active AWS/JDeRobot-style extension world. It
is designed to test whether the same visibility-aware mechanism survives in a
more realistic warehouse with racks, loading apron, staged boxes, and route
choices.

Treat it as exploratory until these artifacts exist for that world:

- AWS-specific detector
- AWS-specific visibility capture
- AWS-specific GP artifact with `P_conservative_plan_map`
- smoke run on B1
- seeded C1/C2/C3 campaign logs
- figures and metrics generated from those logs

## Publication Notes

- Do not cite generated logs or local weights unless they are included in the
  release artifact or linked from an archival store.
- Keep TeX claims aligned with `docs/paper_alignment.md`.
- Keep local caches, model weights, build directories, and raw logs out of the
  source release.
- A final release still needs an explicit license, citation metadata, and an
  artifact/data availability statement.
