# Visibility Legacy Archive

This folder keeps the older monolithic visibility-comparison scripts that were replaced by the cleaner shared backbone under [`scripts/visibility_comparison/`](../../scripts/visibility_comparison/).

Archived here:

- `fit_empirical_visibility_gp.py`
- `plot_visibility_run.py`
- `showcase_yolo_gp_performance.py`
- `generate_docs_figures.py`

Why they were archived:

- they mixed capture, target construction, GP fitting, plotting, and showcase logic in ways that made cross-method comparisons harder to reason about
- they encoded older comparison stories that were no longer the active thesis-facing path
- the new framework separates:
  - raw teleport capture
  - perception target extraction
  - GP target construction
  - GP fitting
  - GP/ambiguity plotting
  - planner-run plotting
  - final reporting

These archived scripts are not part of the active comparison framework anymore.
