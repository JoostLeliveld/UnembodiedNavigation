# `docs/figures`

This folder stores the tutorial figures used by the active README and the canonical docs.

## Why This Folder Exists

The repository now teaches the method with actual visuals, not only prose:

- an example empirical visibility field artifact
- the planner-side observation-noise mapping
- the current state-estimation composition
- an example planning field overlay
- an example run-timeseries panel

## Main Figures

| Figure | Role |
| --- | --- |
| [`visibility_capture_tutorial.png`](visibility_capture_tutorial.png) | example empirical visibility artifact and its supporting sample tables |
| [`observation_model_tutorial.png`](observation_model_tutorial.png) | how `p_vis(x,y)` becomes `R_plan` in the planner |
| [`state_pipeline_tutorial.png`](state_pipeline_tutorial.png) | current estimator composition: camera `x,y` plus odometry-backed `theta` |
| [`planner_field_story.png`](planner_field_story.png) | example top-down planning/visibility overlay from a logged run |
| [`planner_run_timeseries.png`](planner_run_timeseries.png) | example runtime evolution of visibility, belief, EFE terms, and planner timing |

## Provenance

- figure sources are recorded in [`figure_sources.json`](figure_sources.json)
- the current figure set is illustrative documentation support, not the canonical output surface of the new visibility-comparison framework

## Important Caveat

The current stored example visibility artifact comes from an older capture run and is used here as an illustrative learned field. It is useful for explaining the planner-facing artifact format, but it should not be mistaken for fresh evidence of the current teleport-first capture workflow.
