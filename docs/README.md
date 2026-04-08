# Docs Overview

This folder contains the **canonical deeper documentation** for the active thesis milestone. These documents are secondary to the root [`README.md`](../README.md), which is the main onboarding entry point.

## Figure Set

The docs now use a shared tutorial figure set under [`figures/`](figures/README.md):

![Visibility field tutorial](figures/visibility_capture_tutorial.png)

![Observation model tutorial](figures/observation_model_tutorial.png)

![Planner run tutorial](figures/planner_run_timeseries.png)

## Read These First

1. [`architecture_overview.md`](architecture_overview.md)
2. [`runtime_dataflow.md`](runtime_dataflow.md)
3. [`planner_method.md`](planner_method.md)
4. [`evaluation_and_plots.md`](evaluation_and_plots.md)
5. [`limitations.md`](limitations.md)

## Canonical Docs

| File | Why it exists |
| --- | --- |
| [`architecture_overview.md`](architecture_overview.md) | Package map, main files, and minimum reading path |
| [`runtime_dataflow.md`](runtime_dataflow.md) | Offline preparation, runtime ROS graph, and topic flow |
| [`planner_method.md`](planner_method.md) | What the planner is actually doing and what differs between methods |
| [`evaluation_and_plots.md`](evaluation_and_plots.md) | Logged outputs, scripts, and presentation-grade figures |
| [`limitations.md`](limitations.md) | Caveats that should constrain claims in reports, slides, and papers |

These files are the only active documentation surface under `docs/`. Older planning notes and reference PDFs were removed so they stop competing with the current milestone story.

To regenerate the figures used here, run:

```bash
source install/setup.bash
python3 ../scripts/generate_docs_figures.py
```
