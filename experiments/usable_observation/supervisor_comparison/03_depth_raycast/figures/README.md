# Figure slots — depth/raycast

- `01_begin_state.png`: Gazebo RGB, depth frame, back-projected point cloud/height map.
- `02_planning_field.png`: height map, clear/blocked example rays, final `p_use` shadow map.
- `03_update_sequence.png`: commissioning scan, stale layout, rescan and unknown-cell
  fallback.
- `04_route_grid.png`: common R1/R2/R3/R6 plans over the depth-derived field.

Additional method diagnostic: `05_depth_provenance_ladder.png`, showing sensed depth as the
operational source and complete CAD as a separately labelled reference.

Status: all five figures were generated deterministically. D2 is an explicitly labelled
sensor-realism model over the actual SDF, not a captured RGB-D frame.
