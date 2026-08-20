# Figure slots — constant/distance

- `01_begin_state.png`: Gazebo view plus “knows only prevalence/camera distance.”
- `02_planning_field.png`: constant and distance maps on identical `[0,1]` scales.
- `03_update_sequence.png`: identical before/after maps with `NO FIELD UPDATE` badge; show a
  camera observation updating only the belief inset.
- `04_route_grid.png`: R1/R2/R3/R6 selected plans, with missed occlusion shadows outlined.

Status: all four figures were generated deterministically by the package renderer. The
recorded Gazebo view is context only, not evidence for the baseline.
