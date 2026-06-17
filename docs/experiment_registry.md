# Experiment Registry

Separates paper evidence from exploratory diagnostics. A run is not paper
evidence unless the full artifact chain is present. Aligned 2026-06-12.

Only the current paper-ready artifact bundle is kept in the public tree. Raw
campaign logs, training runs, and historical diagnostics are local/private
material unless explicitly packaged under `paper_artifacts/`.

## Evidence Chain

| Link | Required Record |
| --- | --- |
| World | fixed world file and camera pose |
| Detector | detector artifact trained/validated for that world |
| Visibility data | capture directory and manifest from that world |
| GP | fitted artifact with `P_conservative_plan_map` |
| Config | campaign config with explicit detector and GP paths |
| Logs | seeded run directories with manifests and summaries |
| Figures | generated from logs, not hand-drawn behavior claims |
| Paper wording | claims match the logs and metrics |

## Current Paper-Facing Artifacts (in repo, KEEP)

| Status | Asset | Notes |
| --- | --- | --- |
| CURRENT | `src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf` | Locked AWS warehouse geometry and external-camera pose `(0.0, -5.5, 4.8)`. |
| LOCAL | `logs/perception_models/warehouse_yolo_detector_v1/model.pt` | Paper detector checkpoint trained from simulator semantic-segmentation labels. Not tracked in git; runtime localization uses the selected bounding-box bottom centre. |
| CURRENT | `paper_artifacts/perception/warehouse_yolo_detector_v1/` | Public detector metadata: manifest, validation metrics, training plot, and representative validation image. |
| CURRENT | `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz` | Paper GP on the locked camera, fitted with length scale `0.90`, noise variance `0.05`, beta `0.5`, grid `220 x 200`, `P_conservative_plan_map`. |
| CURRENT | `scripts/visibility_comparison/warehouse_visibility_campaign.yaml` | Locked robustness-campaign config: four tasks, five seeds, C1/C2, `ambiguity_weight=1.0`, `visibility_weight=0.0`, `use_belief_nogo_cost=true`, `nogo_belief_kappa=1.0`, `yolo_use_masks=false`. |
| TO REGENERATE | `paper_artifacts/metrics/robustness_metrics.csv` | Previous generated metrics were archived with the old campaign bundle. Regenerate from a completed canonical campaign with `scripts/visibility_comparison/build_paper_outputs.sh`. |
| TO REGENERATE | `paper_artifacts/figures/robustness_spread.png` | Regenerate together with metrics from a completed canonical campaign. |
| TO REGENERATE | `paper_artifacts/figures/paired_mechanism_taskA.pdf` | Previous generated copies were archived because their provenance used superseded artifact names. Regenerate with `scripts/paper_figures/make_paired_mechanism.py` after the canonical campaign/log source is selected. |

## Current Paper Evidence Status

**Robustness campaign:** canonical protocol ready; generated metrics need rerun.

The canonical run matrix is four tasks, two conditions, and five seeds per
condition. Old generated metrics and partial campaign bundles are archived until
the canonical detector/GP/config names are used end to end.

## Superseded / Archived

| Asset family | Reason |
| --- | --- |
| `warehouse_occ_light`, `shadow_tradeoff_*`, `current_gp` | Retired compact-benchmark line. Useful history only, not current paper evidence. |
| `aws_gp_v5`, `aws_gp_v6`, `aws_gp_v6b`, `aws_gp_v7`, `aws_gp_v7b` | Superseded GPs or pre-clean artifacts. |
| `aws_capture_*`, `aws_targets_*`, old perception datasets | Superseded or raw capture/training data. Current named copies use `warehouse_visibility_*` and `warehouse_yolo_*`. |
| `localEFE_paper_v1` | Abandoned local belief-space EFE variant (`use_simple_local_controller=false`). |
| f24-f85 smokes/route-choice/timing, probes, dry runs | Exploratory tuning history. |

## Invalid / Rejected Lines (do not revive as evidence)

| Line | Reason |
| --- | --- |
| visible-goal AWS route-choice probe | baseline already took the detour route; learned condition stalled at high ambiguity weight |
| dark-final-goal AWS route-choice probe | final goal itself camera-poor, confounding route-choice |
| waypoint-driven mission behavior | route externally imposed, not emergent from EFE |
| oversized-ambiguity-only demonstration | not persuasive if it only appears by overwhelming the objective |
| direct-visibility-reward explanation | locked config has `visibility_weight=0.0`; GP enters through observation covariance and belief-tube feasibility |

## Adding A New Evidence Line

Add a row only after the chain is complete. Until then mark `exploratory` or
`diagnostic` and avoid paper-result language. Update this registry and
`docs/paper_runtime_contract.yaml` together.
