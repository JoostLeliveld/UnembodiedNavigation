# Paper Figure Scripts

Paper-ready figure generation lives here. Outputs used by the paper are curated
under `paper_artifacts/` and copied into `thesis-report/figures/` when needed.

## Paper-Facing Scripts

| Figure/artifact | Script |
| --- | --- |
| GP reliability pipeline | `make_aws_gp_pipeline_figure.py` |
| Problem setup panels | `make_aws_problem_setup_figure.py` |
| Localization pathway | `make_localization_pathway_figure.py` |
| YOLO training clarification | `make_yolo_training_clarification.py` |
| Paired F31 mechanism figure | `make_paired_mechanism.py` |
| Robustness spread map | `make_robustness_spread.py` |
| Cost/route decomposition diagnostics | `make_suite_decomposition.py`, `plot_efe_decomposition.py` |

## Diagnostic Scripts

Files named `diag_*`, `validate_*`, `make_f88_*`, or `make_f31b1_*` are
diagnostic/provenance tools unless their output is explicitly listed in
`docs/experiment_registry.md` or `paper_artifacts/README.md`.
