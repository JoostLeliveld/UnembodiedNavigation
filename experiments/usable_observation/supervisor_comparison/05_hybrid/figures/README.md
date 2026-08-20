# Figure slots — hybrid

- `01_begin_state.png`: depth prior + zero residual/high uncertainty + combined cold-start
  field.
- `02_planning_field.png`: aligned prior/residual/final maps with identical coordinates.
- `03_update_sequence.png`: hit/miss residual update, plus separate rescan/reset path.
- `04_route_grid.png`: common R1/R2/R3/R6 plans over the combined field.

Additional diagnostic: `05_boundary_cross_section.png`, comparing FOV, depth, GP and hybrid
across one rack-shadow boundary. This should make “sharp prior, smooth residual” immediately
visible.

Status: all five figures were generated deterministically using the declared candidate
combination `logit(D2) + GP residual`.
