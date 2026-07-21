# Current snapshot (the "after")

Regenerated artifacts from the honest re-run pipeline, to compare against
`../paper/`. Runtime: retrained detector `warehouse_yolo_detector_v1`, refit GP
`warehouse_visibility_gp_v1` on detection rate, low-latency YOLO inference at
640 for the 960-trained detector, `keep_in` honoured, standard chi-squared NIS
gate, 30 s global horizon, ground-truth outcome metrics, and a working
physics-contact channel.

## Figures

All media lives in `paper_artifacts/figures/` (single source of truth,
consolidated 2026-07-15): canonical paired PDFs + `.provenance.json` + `_data/`
source bundles at the root, current-side renders in `current_surface/`.
`CS = ../../../paper_artifacts/figures/current_surface`,
`PA = ../../../paper_artifacts/figures`.

| artifact | purpose |
| --- | --- |
| [`CS/gp_pipeline_current.{pdf,png}`](../../../paper_artifacts/figures/current_surface/gp_pipeline_current.png) | Current detection-rate GP and induced covariance map. |
| [`CS/robustness_spread_current.png`](../../../paper_artifacts/figures/current_surface/robustness_spread_current.png) | 40-run current spread: all seeds over the current GP field. |
| [`PA/paired_mechanism_taskA_current.pdf`](../../../paper_artifacts/figures/paired_mechanism_taskA_current.pdf) (+ png/gif in `CS/`) | Current task-A representative pair from `paired_mechanism_current_taskA`. |
| [`PA/paired_mechanism_west_current.pdf`](../../../paper_artifacts/figures/paired_mechanism_west_current.pdf) (+ png/gif in `CS/`) | Current hard west-route pair from `paired_mechanism_current_west`. |
| `PA/paired_mechanism_{taskA,a2mid,west,control}_lowlat.pdf` (+ png/gif in `CS/`) | Representative seed pairs from `honest_campaign_v1`. |
| [`PA/yolo_training_clarification.png`](../../../paper_artifacts/figures/yolo_training_clarification.png) | Clean detector training/validation clarification (shared with the paper side). |

Each paired PDF has a `.provenance.json` and its source bundle under
`paper_artifacts/figures/<figure-name>_data/`. Current paired plots use
`gt_x/gt_y` and `belief_error_gt_m`; the paper paired plot is explicitly
labelled as the legacy paper truth column for the frozen baseline.

## 40-run headline

Source: `logs/visibility_comparison/honest_campaign_v1`, 4 routes x 2
conditions x 5 seeds.

| route | C1 goal | C1 geom / phys collisions | C2 goal | C2 collisions |
| --- | ---: | ---: | ---: | ---: |
| `route_apron_to_a3_mid` | 4/5 | 0 / 0 | 5/5 | 0 |
| `route_apron_to_a2_mid` | 5/5 | 0 / 0 | 5/5 | 0 |
| `route_west_to_a1_upper` | 1/5 | 4 / 0 | 5/5 | 0 |
| `control_west_to_a1_low` | 5/5 | 0 / 0 | 5/5 | 0 |
| **total** | **15/20** | **4 / 0** | **20/20** | **0** |

Read: C2 succeeds on every current run. C1 fails only on the camera-poor west
route, where the constant-observability planner takes the short blind lane and
breaches the GT geometric safety envelope. The physics-contact channel remains
zero, so those failures are near-wall safety breaches rather than hard impacts.

## Regenerate

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
python3 scripts/paper_figures/remake_paper_vs_current.py
```
