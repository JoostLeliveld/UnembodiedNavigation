# Paper snapshot (the "before")

Frozen artifacts from the IWAI-paper campaign, for side-by-side comparison with
the honest re-run under `../current/`. These use the paper detector/config/GP
line: `aws_yolo_simseg_v2`, archived `aws_gp_v7b`, and the runner that silently
defaulted to `keep_out`.

## Figures

All media lives in `paper_artifacts/figures/` (single source of truth,
consolidated 2026-07-15): canonical PDFs/PNGs + `_data/` bundles at the root,
paper-side renders in `paper_snapshot/`.
`PS = ../../../paper_artifacts/figures/paper_snapshot`,
`PA = ../../../paper_artifacts/figures`.

| artifact | purpose |
| --- | --- |
| [`PA/gp_pipeline_aws.png`](../../../paper_artifacts/figures/gp_pipeline_aws.png) (pdf in `PS/`) | Paper GP pipeline regenerated with archived `aws_gp_v7b`. |
| [`PA/problem_setup_camera.png`](../../../paper_artifacts/figures/problem_setup_camera.png) (pdf in `PS/`) | External-camera setup panel. |
| [`PA/problem_setup_snapshots.png`](../../../paper_artifacts/figures/problem_setup_snapshots.png) (pdf in `PS/`) | Problem-statement uncertainty snapshots. |
| [`PA/localization_pathway.png`](../../../paper_artifacts/figures/localization_pathway.png) (pdf/png variants in `PS/`) | YOLO bottom-centre to BEV localization pathway. |
| [`PA/paired_mechanism_taskA_PAPER.pdf`](../../../paper_artifacts/figures/paired_mechanism_taskA_PAPER.pdf) (+ png/gif in `PS/`) | Frozen task-A C1/C2 mechanism pair. |
| [`PS/robustness_spread.png`](../../../paper_artifacts/figures/paper_snapshot/robustness_spread.png) | Paper robustness spread over all four tasks and five seeds. |
| [`PA/yolo_training_clarification.png`](../../../paper_artifacts/figures/yolo_training_clarification.png) | Detector-training clarification figure. |

The paired mechanism figure has a `.provenance.json` and source bundle under
`paper_artifacts/figures/paired_mechanism_taskA_PAPER_data/`. The paper
robustness run summaries live at
`paper_artifacts/campaigns/robustness_campaign_headline/`. Because the frozen paired logs predate the new
GT schema, the plotter is run with `PAIRED_ALLOW_LEGACY_TRUTH=1` and the
provenance labels the plotted source as `truth_x/truth_y,
truth_belief_error_m`.

## Robustness headline

Source: `logs/visibility_comparison/_paper_runs/robustness_campaign_keepout_lanegraph_v1`
and `paper_artifacts/metrics/archive/robustness_metrics.csv`.

| route | C1 clean goal | C1 collisions | C2 clean goal | C2 collisions |
| --- | ---: | ---: | ---: | ---: |
| apron to A3 | 3/5 | 2 | 3/5 | 1 |
| apron to A2 | 4/5 | 1 | 4/5 | 1 |
| west to A1 upper | 0/5 | 5 | 4/5 | 0 |
| visible control | 5/5 | 0 | 5/5 | 0 |
| **total** | **12/20** | **8** | **16/20** | **2** |

Read with the caveat from `../README.md`: the apparent paper improvement was
partly contaminated by the `keep_out` runner bug and odom-as-truth collision
scoring. The current folder is the corrected comparison.

## Regenerate

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
python3 scripts/paper_figures/remake_paper_vs_current.py
```
