# Figure contract

## Common canvas

- World: `warehouse_full_4cam.world.sdf` for the final comparison.
- Top-down extent: x = `[-11.7, 11.7] m`, y = `[-9, 9] m`.
- Reliability color scale: fixed `[0, 1]`; dark purple = low trust, yellow = high trust.
- Obstacles: opaque grey or rack yellow; unknown map cells: hatched, never silently colored
  as reliable.
- Cameras: A blue, B green, C purple, D orange, matching the four-camera layout.
- Routes: selected explanatory plan thick; start green circle; goal red star.
- A reliability map always labels the represented camera or explicitly says `fused`/
  `four-camera noisy-OR`.
- GP epistemic uncertainty is a separate panel/contour, not encoded by changing the
  reliability color scale.

## Required filenames in every method `figures/` folder

| Filename | Contents |
|---|---|
| `01_begin_state.png` | Recorded Gazebo view where informative, plus only the operational inputs available initially |
| `02_planning_field.png` | Top-down `p_use` field actually passed to planning |
| `03_update_sequence.png` | Before/observations/after; or explicit static-method panel |
| `04_route_grid.png` | R1, R2, R3 and R6 route plans on the same scales |

Generators may write PDF/SVG companions, but the PNG is the supervisor-preview contract.

## Required annotations

Across each method's four-panel set, the figures state:

- method and variant;
- cold-start versus commissioned/updated state;
- operational inputs;
- fallback for unsupported or missing cells;
- map age where geometry is stored;
- `EXPLORATORY`, `REFERENCE`, or `CONFIRMATORY` status.

Do not place AUROC/Brier/navigation numbers on a mechanism figure unless its manifest binds
the dataset, split and artifact hashes. The supervisor package first explains the methods;
results can later occupy a separate row without redrawing the mechanism.
