# Geometry Visibility Prior Implementation Plan

This plan defines the first validation-first module for the geometry-derived
observability extension. The goal is to prove the offline chain before any
planner integration, depth camera work, or online GP updating.

## First Goal

Build an offline `geometry_visibility_prior` module that reuses the existing
warehouse setup:

```text
current warehouse geometry + current camera model
-> 2.5D height map
-> raycast visibility labels
-> visibility-to-R_plan map
-> validation figures and summary
```

The module is complete only when we can point to a cell in the warehouse and
explain why the expected camera measurement covariance is low, medium, or high.

## Non-Goals For This First Module

- Do not use a depth camera yet.
- Do not parse arbitrary SDF geometry yet if the existing packaged
  `geometry_json` is sufficient.
- Do not update the live planner.
- Do not perform online GP updates.
- Do not claim route improvement.
- Do not replace the existing YOLO-derived GP artifact.

This first module is a validated observability-prior generator, not a planner
result.

## Existing Systems To Reuse

| Need | Existing source |
| --- | --- |
| World identity | `warehouse_aws.world.sdf` |
| Camera intrinsics/profile | `src/experiments/config/world_profiles.yaml` |
| Camera projection | `src/unav_common/unav_common/camera_model.py` |
| Current campaign config | `scripts/visibility_comparison/warehouse_visibility_campaign.yaml` |
| Driveable region | `driveable_geometry_json` in the campaign config |
| Occlusion geometry | `geometry_json` in `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz` |
| Current YOLO-derived GP comparison | `P_conservative_plan_map` in the same GP artifact |
| Existing grid | `xs`, `ys` in the current GP artifact |
| Existing covariance settings | `r_visible_uv: 2.5`, `r_miss_uv: 40.0` in campaign config |
| Existing plotting style | `scripts/paper_figures/make_readme_visuals.py` |

Reusing these keeps the first test aligned with the current thesis setup and
avoids debugging a new world/camera configuration at the same time.

## Proposed Files

Add a small offline script package under `scripts/geometry_visibility/`:

```text
scripts/geometry_visibility/
  README.md
  build_geometry_visibility_prior.py
  geometry_visibility.py
  test_geometry_visibility.py
```

Suggested responsibilities:

| File | Responsibility |
| --- | --- |
| `geometry_visibility.py` | Pure functions: load geometry, build height map, project points, raycast clearance, compute visibility score, map trust to `R_plan`. |
| `build_geometry_visibility_prior.py` | CLI entrypoint for the existing warehouse artifact. Writes CSV/NPZ/figures/summary. |
| `test_geometry_visibility.py` | Deterministic toy-scene tests for height map, raycasting, FOV, and `R_plan` monotonicity. |
| `README.md` | Module contract, command, outputs, validation gates, and known limits. |

Generated outputs should go under `logs/geometry_visibility_prior/`, not under
tracked artifacts until the module is stable.

## CLI Contract

Primary command:

```bash
python3 scripts/geometry_visibility/build_geometry_visibility_prior.py \
  --gp-artifact paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz \
  --campaign-config scripts/visibility_comparison/warehouse_visibility_campaign.yaml \
  --world-profile src/experiments/config/world_profiles.yaml \
  --world warehouse_aws.world.sdf \
  --out logs/geometry_visibility_prior/warehouse_aws_v0
```

The command should not require ROS, Gazebo, a YOLO checkpoint, or a live
simulator.

## Output Contract

The output directory should contain:

```text
height_map.npz
raycast_visibility.csv
geometry_visibility_prior.npz
VALIDATION.md
figures/
  01_geometry_overlay.png
  02_height_map.png
  03_fov_mask.png
  04_clearance_map.png
  05_raw_visibility.png
  06_visibility_components.png
  07_geometry_r_plan_std.png
  08_r_plan_matrix_examples.png
  09_current_yolo_gp_vs_geometry_prior.png
```

### `height_map.npz`

Required fields:

```text
xs, ys
h_max
known
occ_conf
resolution
origin_xy
geometry_json_sha256
```

### `raycast_visibility.csv`

One row per evaluated candidate cell:

```text
x
y
in_driveable
in_fov
u
v
depth_m
range_m
min_clearance_m
unknown_fraction
f_fov
f_occ
f_range
f_boundary
visibility_score
label_noise
r_plan_std_px
r_plan_var_px2
reason
```

### `geometry_visibility_prior.npz`

Required fields:

```text
xs
ys
visibility_score_map
label_noise_map
clearance_map
unknown_fraction_map
fov_mask
r_plan_std_map
r_plan_var_map
current_gp_trust_map
```

## Step-By-Step Implementation

### Step 1: Load Current Artifact And Config

Read:

- `xs`, `ys`, `geometry_json`, `P_conservative_plan_map` from the current GP
  artifact.
- `driveable_geometry_json`, `r_visible_uv`, `r_miss_uv` from the campaign
  config.
- camera intrinsics and camera pose from `world_profiles.yaml`.

Validation:

- Print artifact shape and bounds.
- Assert `xs` and `ys` are strictly increasing.
- Assert `P_conservative_plan_map.shape == (len(ys), len(xs))`.
- Assert `r_visible_uv < r_miss_uv`.
- Plot `01_geometry_overlay.png` with geometry rectangles, driveable
  rectangles, camera position, and grid bounds.

Pass condition:

- The overlay visually matches the existing warehouse layout and current GP
  figures.

### Step 2: Build 2.5D Height Map

For each grid cell, set `h_max` to the maximum `zmax` of any occlusion prism
covering the cell. Use the current GP grid first. Mark known cells as true
because version 1 uses known geometry, not depth.

Validation:

- Toy fixture: one cuboid covering a known cell range produces exactly those
  high cells.
- Warehouse figure: rack and crate footprints appear in the expected places.
- Summary counts: number of obstacle cells, free cells, and max height.

Pass condition:

- `02_height_map.png` shows rack/crate footprints aligned with
  `geometry_json`.

### Step 3: Camera Projection And FOV Mask

Use `ObliqueCameraModel` from `src/unav_common/unav_common/camera_model.py`.
Evaluate candidate marker points:

```text
p_marker = [x, y, z_marker]
```

Initial `z_marker` should use the existing visibility target height from the
world profile if available; otherwise start with `0.35 m`.

Validation:

- Known open apron points project inside the image.
- Clearly outside-map points or behind-camera points fail.
- `03_fov_mask.png` shows which driveable cells are inside the camera image.

Pass condition:

- FOV rejection is explainable independently from occlusion.

### Step 4: Raycast Clearance

For each in-FOV candidate position, sample the ray from camera center to
`p_marker`. At every ray sample:

```text
clearance = z_ray - h_max(x_ray, y_ray)
```

The minimum clearance determines occlusion risk.

Validation with toy fixtures:

| Fixture | Expected result |
| --- | --- |
| Empty map | Positive clearance; visible. |
| Wall directly between camera and target | Negative clearance; occluded. |
| Obstacle beside ray | Positive clearance; visible. |
| Ray grazing obstacle top | Near-zero clearance; uncertain. |
| Target outside FOV | FOV failure, not occlusion failure. |

Warehouse validation:

- Pick 5 hand-labelled points: open apron, lower aisle, behind rack, mid
  connector, far corner.
- Save their clearance values in `VALIDATION.md`.

Pass condition:

- Toy tests pass exactly and hand-labelled warehouse points are plausible.

### Step 5: Visibility Components

Compute separate components:

```text
f_fov       = 0 or 1
f_occ       = sigmoid(clearance / tau_clearance)
f_range     = marker-size or range score
f_boundary  = image-boundary score
```

Then:

```text
visibility_score = f_fov * f_occ * f_range * f_boundary
```

Start conservative and simple:

- `tau_clearance = 0.10 m`
- range score disabled or gentle in v0 if it makes interpretation harder
- boundary score plotted but not over-weighted

Validation:

- Plot each component separately in `06_visibility_components.png`.
- Confirm no single component silently dominates all others.
- Confirm the final score is in `[0, 1]`.

Pass condition:

- A human can explain why three selected cells have high, medium, and low
  score by looking at the components.

### Step 6: Label Noise

Assign pseudo-label noise, even before fitting a GP:

```text
low noise: clear visible or clear blocked
high noise: near occlusion boundary, near image boundary, or unknown geometry
```

For v0 with known geometry:

```text
label_noise = sigma_min^2 + a_boundary * boundary_uncertainty
```

Validation:

- Noise is high near clearance around zero.
- Noise is lower in confident open/blocked regions.
- Noise stays bounded.

Pass condition:

- `raycast_visibility.csv` records both `visibility_score` and `label_noise`
  for every candidate cell.

### Step 7: Map Visibility To `R_plan`

Use the same precision blend as the current planner docs:

```text
1 / var_plan = trust / r_visible_uv^2 + (1 - trust) / r_miss_uv^2
R_plan = [[var_plan, 0],
          [0, var_plan]]
```

Validation:

- `trust = 1` gives `r_plan_std_px ~= r_visible_uv`.
- `trust = 0` gives `r_plan_std_px ~= r_miss_uv`.
- `r_plan_std_px` is monotonic decreasing with trust.
- The matrix is shown explicitly as 2x2, symmetric, diagonal, in `px^2`.

Pass condition:

- `08_r_plan_matrix_examples.png` shows visible, uncertain, and occluded
  examples with actual matrix values.

### Step 8: Compare Against Current YOLO-Derived GP

Plot geometry prior next to the current empirical GP field:

```text
geometry visibility score
current P_conservative_plan_map
difference map
```

Interpretation:

- Agreement means geometry explains learned reliability.
- Disagreement is not automatically failure. It may indicate detector
  appearance, perspective, range, calibration bias, YOLO localization error, or
  sparse empirical data.

Validation:

- `09_current_yolo_gp_vs_geometry_prior.png`
- `VALIDATION.md` lists at least three agreement regions and three
  disagreement regions.

Pass condition:

- Differences are documented rather than hidden or tuned away.

## Tests To Add

Run with:

```bash
python3 scripts/geometry_visibility/test_geometry_visibility.py
```

Minimum deterministic tests:

| Test | Assertion |
| --- | --- |
| `test_height_map_single_box` | Cells inside cuboid footprint get `zmax`; outside cells stay zero. |
| `test_world_to_grid_roundtrip` | Known points map to expected cell indices and back within resolution. |
| `test_empty_scene_visible` | In-FOV target in empty map has positive clearance and high visibility. |
| `test_wall_blocks_ray` | Cuboid intersecting ray gives negative clearance and low visibility. |
| `test_side_obstacle_does_not_block` | Cuboid beside ray does not reduce clearance below threshold. |
| `test_outside_fov_zero_visibility` | Outside-FOV target has zero final visibility and reason `outside_fov`. |
| `test_visibility_bounds` | Every score is finite and in `[0, 1]`. |
| `test_r_plan_monotonic` | Higher trust gives lower or equal `r_plan_std_px`. |
| `test_r_plan_endpoints` | Trust 0 and 1 recover `r_miss_uv` and `r_visible_uv`. |

## First Acceptance Criteria

The first module is accepted only when all are true:

1. The CLI runs without ROS/Gazebo.
2. Toy tests pass.
3. Warehouse outputs are generated under `logs/geometry_visibility_prior/`.
4. The geometry overlay matches the current warehouse.
5. FOV, clearance, visibility, and `R_plan` maps are all plotted separately.
6. `R_plan` is shown as an explicit 2x2 matrix with units.
7. Geometry prior is compared to the current YOLO-derived GP.
8. `VALIDATION.md` explains known agreement/disagreement regions.
9. No planner code consumes this artifact yet.
10. The module README states exactly what downstream modules may and may not
    assume.

## Suggested Implementation Order

1. Add pure utility functions and toy tests.
2. Add warehouse artifact loader.
3. Add height-map builder.
4. Add raycast clearance.
5. Add component visibility scoring.
6. Add `R_plan` conversion.
7. Add plotting and `VALIDATION.md` writer.
8. Compare to current GP artifact.
9. Write module README.
10. Only then consider a planner integration plan.

## Stop Conditions

Stop and fix the module before moving on if:

- geometry overlay does not match the warehouse,
- FOV and occlusion failures are mixed together,
- raycast toy tests fail,
- `R_plan` endpoints do not match `r_visible_uv` and `r_miss_uv`,
- the geometry prior is tuned to imitate the current GP without explaining
  disagreements,
- any figure hides whether it uses geometry prior, current empirical GP,
  odometry, belief, or ground truth.

## Later Extensions

After this module passes:

1. Fit a GP to the geometry pseudo-labels.
2. Compare `raw raycast -> GP-smoothed geometry prior`.
3. Add a planner experiment using the geometry-derived `R_plan`.
4. Add depth-derived height map as Plan 1B.
5. Add belief-weighted online updates as Plan 1C.

These are intentionally later. The first validated artifact is the offline
warehouse geometry visibility prior.
