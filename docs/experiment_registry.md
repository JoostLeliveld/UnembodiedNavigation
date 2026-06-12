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
| LOCAL | `local_artifacts/perception_models/aws_yolo_simseg_v2/model.pt` | Paper detector checkpoint trained from simulator semantic-segmentation labels. Not tracked in git; runtime localization uses the selected bounding-box bottom centre. |
| CURRENT | `paper_artifacts/perception/aws_yolo_simseg_v2/` | Public detector metadata: manifest, validation metrics, training plot, and representative validation image. |
| CURRENT | `paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz` | Paper GP on the locked camera. v7b = v7 capture plus final target-table cleanup/augmentation, fitted with length scale `0.90`, noise variance `0.05`, beta `0.5`, grid `220 x 200`, `P_conservative_plan_map`. |
| CURRENT | `scripts/visibility_comparison/aws_f31b1_final_config.yaml` | Locked robustness-campaign config: four tasks, five seeds, C1/C2, `ambiguity_weight=1.0`, `visibility_weight=0.0`, `use_belief_nogo_cost=true`, `nogo_belief_kappa=1.0`, `yolo_use_masks=false`. |
| CURRENT | `paper_artifacts/metrics/robustness_metrics.csv` | Per-task/condition campaign metrics used for the paper table. Continuous localization metrics are clean-success pooled. |
| CURRENT | `paper_artifacts/figures/robustness_spread.png` | Robustness spread map generated from seeded runs. |
| CURRENT | `paper_artifacts/figures/paired_mechanism_taskA.pdf` | Single-run mechanism figure. Source data and provenance live beside the figure in `paper_artifacts/figures/paired_mechanism_taskA_data/` and `paired_mechanism_taskA.provenance.json`. Use as mechanism illustration, not as the whole robustness claim. |

## Current Paper Evidence Status

**Robustness campaign:** paper-ready with caveats.

Five seeds were run for each condition on four tasks: three discriminating
route-choice tasks and one control. Aggregate outcome:

- C2 (`visibility_aware_efe`): `18/20` clean goal reaches, `2/20` collisions.
- C1 (`constant_R_efe`): `12/20` clean goal reaches, one near-success,
  `7/20` collisions.

Per task:

| Task | C1 | C2 | Interpretation |
| --- | --- | --- | --- |
| `F31_b1_apron_a3_mid` | `4/5` clean, `1/5` collision | `5/5` clean | discriminator |
| `b5_a4_apron_to_a2_mid` | `3/5` clean, `2/5` collision | `5/5` clean | discriminator |
| `b2_a0_west_to_a1_upper` | `1/5` clean, `4/5` collision | `3/5` clean, `2/5` collision | hard discriminator |
| `b6_a0_west_to_a1_low_control` | `4/5` clean + one near-success, `0/5` collision | `5/5` clean | control |

The mechanism evidence is route observability: C2 spends less time in low
reliability regions and has higher detection fraction on the discriminating
tasks. The result should be framed as improved robustness, not a clean sweep.

## Superseded / Archived

| Asset family | Reason |
| --- | --- |
| `warehouse_occ_light`, `shadow_tradeoff_*`, `current_gp` | Retired compact-benchmark line. Useful history only, not current paper evidence. |
| `aws_gp_v5`, `aws_gp_v6`, `aws_gp_v6b`, `aws_gp_v7` | Superseded GPs or pre-clean artifacts. |
| `aws_capture_v6/v7`, `aws_targets_*`, perception datasets | Raw capture/training data; fitted v7b GP and detector are kept. |
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
