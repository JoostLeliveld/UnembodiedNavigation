# Presentation figures

Built from the frozen commissioning artifacts. No simulator, a few minutes each.
Every figure writes to `logs/studies/deck_figures/<subject>/` under its own name.

## Shared, at the top level

| module | what it does |
|---|---|
| `style.py` | the visual vocabulary and the warehouse drawing. No data. |
| `sources.py` | reads the frozen commissioning artifacts. No drawing, no fitting. |
| `rollout.py` | camera support along a candidate route |

**Nothing here recomputes a commissioning number.** If a figure disagrees with
`calibration.json`, the figure is wrong.

---

## `observation/` — the box is not the robot

Six figures: three on why it matters, three on what the fix does and costs.

| | figure | what it shows |
|---|---|---|
| **why** | `01_two_points` | real frames from three cameras: the box bottom-centre and the robot are **24–36 cm apart** |
| | `02_how_big_and_how_it_moves` | the gap over a full turn, and that it **swings ~7 cm with heading** — so it is not a constant you could subtract |
| | `03_cost_of_ignoring` | the error field if you treat the bottom-centre as the robot: **30 cm typical, 14× the detector's own scatter** |
| **fix** | `04_predict_dont_convert` | project the robot's shape, box it, take the bottom-centre — the same operation the detector performs, so the gap is on both sides and never appears |
| | `05_what_is_left` | 0.53 → 0.29 cm on 313 held-out positions, and how few spots it takes |
| | `05b_leftover_next_to_the_scatter` | the same, against 2.2 cm of random scatter — the appendix half |
| | `06_the_price_is_heading` | the fix needs a heading, which comes from odometry. Break-even at **14°** |

**This is not a bias correction.** Nothing in `01`–`04` is fitted. Only `05` fits anything,
and it is worth half a centimetre.

## `uncertainty/` — how much to trust a sighting

Same shape: three on why one number cannot be right, three on the fix.

| | figure | what it shows |
|---|---|---|
| **why** | `01_error_grows_with_distance` | 0.9 cm close, 4.0 cm far — **fivefold across one building** |
| | `02_not_a_circle` | the error is a **stretched ellipse along the line of sight**, not a circle |
| | `03_constant_R_fails` | one covariance in centimetres: **100% coverage near, 61% far**; eight times too large close in, four times too small far out |
| **fix** | `04_pixels_are_stationary` | the same error in **pixels does not move** — it was never a property of where the robot stood |
| | `05_geometry_does_the_rest` | one number tracks a sixfold change in spread **to within 15%**, with nothing about distance fitted |
| | `06_more_parameters_dont_help` | the ladder: **one number wins**; a 41-number map fitted over the floor is the least honest of all |

## `availability/`, `planning/`, `confidence/`

| | |
|---|---|
| `availability/01_promise_vs_reality` | every position is geometrically covered; 18% still get nothing usable |
| `availability/02_where_sightings_are_lost` | only 29% of chances become a measurement, and why |
| `availability/03_three_places` | three real frames: clear, half hidden, buried |
| `planning/01_two_routes` | two routes, camera handovers, how far the robot drives unseen |
| `planning/02_camera_density` | blind driving against camera count, and what the detour costs |
| `confidence/01_why_not_confidence` | why the planner cannot route on detector confidence |
| `confidence/02_confidence_vs_covariance` | post-geometry pixel residual spread and held-out covariance models |
| `confidence/03_confidence_role_split` | confidence is stronger for admission than as a direct covariance dial |

## `fusion/` — the route the fusion arms drive

| | |
|---|---|
| `fusion/01_the_route` | the 30.4 m traverse every fusion arm drives: which cameras watch it, where they hand over, and how many watch at once (none to four) |

Route candidates come from the lane geometry alone (`route_tasks.py`, task
`dock_w__xaisle_e`); the shortest is driven. Support is the commissioned per-camera
usable-sighting rate. It is what the model expects the route to meet, not a recorded drive.
The six arms it introduces are in
[`experiments/fusion_on_fixed_routes/README.md`](../fusion_on_fixed_routes/README.md).

## The fusion study's own figures

They live with the study, not here: `logs/studies/fusion_on_fixed_routes/`, one folder per
method plus `compare/`. Built by `experiments/fusion_on_fixed_routes/story/` and `compare.py`,
in the same house style as this deck.

| | |
|---|---|
| `00_hull_observation/` | what a camera reading is, why a box cannot be converted into a position, and what the four rules do with one real moment |
| `01…06_*/` | one arm each: where it drove, whether it was honest, its worst moment |
| `compare/` | error and claim against camera count; the six arms side by side; what the box was taken to mean |

## `warehouse/` — the new environment at a glance

| | |
|---|---|
| `warehouse/01_warehouse_showcase` | representative clear views from all five warehouse_v2 cameras, plus the new AMR |
| `warehouse/02_new_robot_detail` | full-scene and detail views of the floor-scale warehouse AMR |

The five-camera showcase is visual orientation, not a synchronized observation. The robot
pose differs between panels so that every viewpoint contains a clear, useful view.

---

## Two words that are not interchangeable

**The observation model** is the box-versus-centre problem — 30 cm, not fitted, removed by
predicting rather than converting. Because it needs a heading, heading error feeds straight
into any error measured against the true robot centre. `observation/`.

**The offset** is the half-centimetre lean left over. Six commissioned numbers, a calibration
detail. `observation/05*`.

Do not call the first one a bias.

## One trap, already paid for three times

**Never join files on coordinates.** The capture stores full floats, every written file
stores formatted ones, and the sample grid lands exactly on rounding ties — rounding,
lattice-snapping and decimal formatting each fail somewhere. It silently detached 10% of the
usable sightings from their attempt records (inflating "no camera can help here" from 15% to
24%) and twice miscounted which spots the offset was fitted on. Every floor position carries
an integer `position_id`, written into every file by `commission.py`. Joins use it.
