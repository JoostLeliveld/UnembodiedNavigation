# `experiments/config`

This folder holds the two YAML files that define the experiment before anything is launched.

![Visibility artifact tutorial figure](../../../docs/figures/visibility_capture_tutorial.png)

The `world_profiles.yaml` bounds and artifact paths directly shape both the driving-based capture pipeline and the online planner.

## Why This Folder Exists

The current milestone is a controlled comparison. These files fix:

- which worlds are supported
- where the camera is placed
- which GP artifact belongs to which world
- which start/goal tasks are benchmark, diagnostic, or exploratory

## Main Files

| File | Role |
| --- | --- |
| [`world_profiles.yaml`](world_profiles.yaml) | supported worlds, camera intrinsics, default planner, visibility artifact path |
| [`tasks.yaml`](tasks.yaml) | benchmark and exploratory tasks for each world |
| `../../archive/experiments/tasks_legacy.yaml` | archived alias map for older `T*` task names; not part of the active launch surface |

## What To Read First

1. `world_profiles.yaml`
2. `tasks.yaml`

## Important Caveats

- `warehouse_occ_light.world.sdf` is the primary benchmark world
- `warehouse_open_shelves.world.sdf` is a secondary support-world configuration
- the current thesis-facing story is not “many maps equally validated”; it is one primary benchmark plus one exploratory extension
