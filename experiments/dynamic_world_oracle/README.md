# Dynamic world and visibility oracle

**Question this answers:** when the warehouse changes — a pallet is set down in an
aisle, a forklift drives through — *which floor cells can each camera still see, at
each instant?* This study owns the changing world and the ground-truth answer, so
that a learned or geometric visibility model has something honest to be scored
against.

It produces a dataset, not a method. Everything here is measurement apparatus.

## The boundary that makes this useful

Oracle depth, Gazebo object poses and ray-cast visibility are **evaluation-only**.

A method may read, from a run of this study:

- `rgb_path` — the image the camera rendered
- `camera_intrinsics`, `camera_extrinsics` — calibration, which a deployed system has

A method may **not** read:

- `oracle_depth_path` — the renderer's depth buffer
- `oracle_visibility_grid` — line-of-sight computed from CAD prisms and exact poses
- `obstacle_state[*].pose` / `world_aabb` — where the simulator put the obstacle

Scoring a visibility model against a signal it was allowed to read is not a
result. `verify_acceptance.py` greps `src/` for references to this study's ground
truth and fails if any runtime package has picked one up; `tests/experiments/`
holds the same check so the full suite enforces it.

## Where the world comes from

The four-camera warehouse generator was extended rather than forked:

```bash
python3 scripts/geometry_visibility/make_warehouse_full.py --variant dynamic
```

writes, all generated, never hand-edited:

| file | what it is |
|---|---|
| `src/sim/gazebo_worlds/worlds/warehouse_full_4cam_dynamic.world.sdf` | the flagship four-camera warehouse under a different world name; structurally identical, so a dynamic run stays comparable to a static one |
| `…/warehouse_full_4cam_dynamic.stage.json` | camera intrinsics/extrinsics, site bounds, the scenario aisle, and the obstacle catalogue |
| `src/sim/models/dyn_pallet_box/` | loaded euro-pallet, 1.20 × 0.80 × 1.49 m |
| `src/sim/models/dyn_forklift/` | forklift silhouette, 2.72 × 1.02 × 2.09 m including forks |

`--variant static` (the default) still writes the frozen flagship world
byte-for-byte unchanged, so regenerating cannot disturb the evaluation world.

**Every aisle starts clear.** Obstacles are not baked into the world; they arrive
through runtime spawn events. That makes "clear aisle at t=0" true by
construction, and it is checked at generation time — the generator refuses to
write the dynamic world if any rack or pillar intrudes on the scenario aisle.

The obstacles are one rigid link with several collision/visual parts,
`static=false` (Gazebo will not accept pose commands otherwise) with
`gravity=false` (so a commanded pose is held exactly, with no settling and no
run-to-run solver drift).

## Running a scenario

```bash
python3 experiments/dynamic_world_oracle/run_scenario.py \
    --scenario experiments/dynamic_world_oracle/scenarios/s01_box_in_aisle.yaml \
    --run-tag run01
```

Output lands in `logs/studies/dynamic_world_oracle/<scenario_id>/<run_tag>/`.

To **watch** a scenario instead of just recording it, add `--gui`:

```bash
python3 experiments/dynamic_world_oracle/run_scenario.py \
    --scenario experiments/dynamic_world_oracle/scenarios/s02_forklift_transit.yaml \
    --run-tag gui_watch --gui
```

A Gazebo window opens and the world advances in visible 0.2 s steps — obstacles
appear, move and vanish on the timeline. The GUI is a viewer on the same gz
partition; it does not touch the clock, but it does compete for the GPU with the
sensor renders the loop waits on. Watch a run with it; take datasets from runs
without it.

Scenarios are YAML timelines of four event kinds — `spawn`, `move`, `stop`,
`remove` — on simulated time, plus a capture schedule. `move` with a
`duration_s` interpolates the pose every sensor tick, which is how the forklift
drives; `duration_s: 0` teleports. The loader rejects timelines a simulator could
not act out (moving something that was never spawned, spawning it twice) and
snaps every instant onto the sensor tick so a scenario cannot ask for a frame at
an instant no camera renders.

| scenario | what happens |
|---|---|
| `s01_box_in_aisle` | clear aisle → pallet set down on camera A's sight-line → driven 10 m up the aisle into camera B's → stopped → removed |
| `s02_forklift_transit` | forklift drives 18 m along the south perimeter aisle, handing its occlusion shadow from camera A to camera C |

Obstacle poses are not guessed. `choose_obstacle_pose.py` sweeps a lane and
prints how many visible cells each camera would lose with the obstacle at each
position, using the same geometry the run will use:

```bash
python3 experiments/dynamic_world_oracle/choose_obstacle_pose.py \
    --model dyn_pallet_box --lane rack_aisle_W2 --along y
```

`s01`'s spawn pose is the row where camera A loses 44 cells and B, C and D lose
none — which is what makes "this obstacle intersects a *selected* camera's rays"
a checkable statement.

## The output contract

`records.jsonl`, one record per capture instant per camera:

| field | |
|---|---|
| `scenario_id` | which scenario produced this |
| `timestamp` | simulated seconds; identical across the four cameras of one instant |
| `camera_id` | `external_camera`, `…_b`, `…_c`, `…_d` |
| `rgb_path` | PNG, 1280×720 RGB |
| `oracle_depth_path` | `.npy` float32, 1280×720, **planar depth** along the optical axis (not Euclidean range) |
| `camera_intrinsics` | `fx`, `fy`, `cx`, `cy`, image size, horizontal FOV |
| `camera_extrinsics` | the SDF pose, plus the `cam_pos`/`look_at` pair the repo's `ObliqueCameraModel` takes |
| `obstacle_state` | per obstacle: model, pose read back from the simulator, world AABB, whether that AABB is exact, and whether it was moving |
| `oracle_visibility_grid` | path to a `uint8` grid plus its extent, resolution, target height and cell counts |

Alongside: `manifest.json` (scenario, world/model/code SHA-256, Gazebo version,
grid spec), `events.csv` (requested vs applied simulated time per event),
`checksums.sha256`, and `EVALUATION_ONLY.md`.

Visibility grid cells are `0` blocked sight-line, `1` visible, `2` outside the
camera's image, `3` inside solid geometry. "Visible" is `== 1`; the other three
codes record *why* a cell is not visible, which is what makes an occlusion event
legible instead of just a number going down. `oracle_visibility/any_camera/`
holds the union — the cells at least one camera can see.

## How determinism is achieved

A normal `gz sim -r` run advances on wall-clock, so two runs of one scenario see
the spawn at different simulated times and render at different instants. Instead
the server is started **paused** and never runs freely: every advance is an
explicit `multi_step` of a fixed number of 1 ms physics steps, every event is
applied at an exact step boundary, and a spawn or remove is followed by a *fixed*
settle step count rather than "keep stepping until it shows up".

Three findings worth not rediscovering:

- **`gz` is a Ruby script, so a Gazebo server's process name is `ruby`.**
  `pgrep -x gz` finds nothing and leaves it running; use `pgrep -af "gz sim"`.
  A server orphaned this way kept advertising on its partition, and the next run
  of the same scenario adopted it — inheriting a clock already at t=5.2 s, with
  no error anywhere. Run partitions now carry the process id so a name can never
  be reused, the session refuses to start if its partition is already taken, and
  the runner aborts if the world is not at t=0 before the first step.

- **Do not subscribe to `/world/<world>/pose/info`.** It carries all ~555 entities
  of this warehouse on every one of the 200 iterations in a step burst; decoding
  that in Python starved the interpreter badly enough that Gazebo *skipped camera
  renders* and *dropped step requests*. `dynamic_pose/info` carries only the
  non-static models — the obstacles — and the problem disappears.
- **Wait for each tick's frames even when you are not keeping them.** Gazebo
  renders sensors on its own thread and silently skips an update that comes due
  while the previous render is in flight, so firing the next burst the instant the
  clock arrives is what makes frames vanish. Blocking until the current tick's
  frames are in hand keeps the renderer in step.

Obstacle poses are recorded rounded to nanometres: a gravity-free commanded body
reads back with ~1e-13 m of solver noise that differs between runs, nine orders of
magnitude below the 0.25 m grid cell it feeds.

## Acceptance

```bash
python3 experiments/dynamic_world_oracle/verify_acceptance.py \
    --run    logs/studies/dynamic_world_oracle/s01_box_in_aisle/run01 \
    --repeat logs/studies/dynamic_world_oracle/s01_box_in_aisle/run02
```

Checks, each stated as the claim it defends:

1. **output contract** — every record carries the agreed fields
2. **reproducible** — two runs of one scenario produce byte-identical artifacts
3. **partial occlusion** — the obstacle hides some cells and leaves others visible
4. **event timing** — visibility changes at the event instants and nowhere else
5. **camera synchronisation** — all four cameras deliver frames at the same simulated stamp
6. **removal restores** — taking the obstacle away returns the original oracle map
7. **oracle is trustworthy** — the fast ray cast agrees with `unav_common.segment_occluded`
8. **depth agrees with the ray cast** — Gazebo's own depth buffer confirms every obstacle occlusion
9. **boundary holds** — nothing under `src/` reads this study's ground truth

`make_sanity_montage.py` draws the rendered frame and the oracle map side by side
for the same instants, which is the check a person should do before trusting any
of it.

### What check 8 caught, and what it still reports

Checks 1–7 all compare the oracle with itself. Check 8 asks the simulator: for
every cell an obstacle newly hides from a camera, the rendered depth along that
pixel must come back *shorter* than the distance to the cell.

It failed the first time it ran — 71% agreement — and the oracle was wrong, not
the check. Obstacles were being bounded by a single box, so the pallet's narrower
load and the forklift's thin mast were modelled as solid slabs and rays that
really did pass were called blocked. Obstacle geometry now comes from the
collision parts of the generated model SDF (`parts_from_model_sdf`), which is the
same geometry Gazebo simulates. Obstacle-caused occlusions are now confirmed by
the renderer **100%** of the time.

The same check reports a second number it does not assert on: how many cells the
oracle calls *visible* have a genuinely unobstructed rendered depth. That started
at 96.3%, and chasing the gap found a second real defect. Reconstructing the 3-D
point where the renderer stopped showed three quarters of the disagreements
sitting between 2.09 m and 2.40 m — exactly rack-top height. The generator draws
blue rails and tan shelf boxes on top of every rack with **no collision**, so
they were invisible to a collision-only prism parse while being perfectly visible
to the camera. A camera does not care whether a thing has physics: the static
prism set now takes the visual boxes of the wall and rack models as well, and
agreement went to **98.85%** (camera A: 94.1% → 98.4%).

The residual ~1% is ShelfD/E mesh overhang. Those shelves are DAE meshes, not
boxes, so no prism can represent their gaps and protrusions; the oracle treats
the rack as its box, the renderer draws the mesh, and near a rack edge the two
disagree. That is a bound on the oracle's fidelity, and it is reported on every
acceptance run rather than being asserted away.

## Known limits

- **The oracle is a geometric line-of-sight test, not a detectability model.** A
  cell it calls visible may still be one a detector fails on — long range, glare,
  a robot half-hidden behind a rack upright. That is the gap a learned model is
  supposed to fill, which is why the oracle must not be one of its inputs.
- **Rack occlusion is prism-exact, not mesh-exact.** Measured: 98.85% of cells the
  oracle calls visible are confirmed clear by Gazebo's own depth buffer; the
  residual is ShelfD/E mesh detail (see check 8 above).
- **Target height is a parameter, not a fact.** The scenarios use 0.35 m, matching
  the four-camera world profile's `visibility_target_height_m`; it is recorded in
  every record so a different choice can be re-derived rather than argued about.
- **Obstacles hold their commanded pose** rather than resting under gravity. That
  is a deliberate trade for determinism; a scenario that needs a falling or
  colliding obstacle would need this revisited.
- **AWS clutter meshes near the walls are outside the site boundary** and are not
  in the static prism set. Sight-lines from a 6.1 m mount to floor cells inside
  the operating field do not pass through them.

## Reuse

| need | use |
|---|---|
| camera geometry from a world SDF | `reliability.projection.camera_model_from_world` |
| CAD prisms from a world SDF | `unav_common.occlusion_geometry.parse_collision_scene_from_world` |
| single-ray occlusion | `unav_common.occlusion_geometry.segment_occluded` |
| many rays at once | `oracle.segments_hit_any_prism` — the vectorised twin, cross-checked against the scalar one in the acceptance run |
| repo root | `scripts/shared/paths.repo_root` |

Nothing here reimplements a projection, a prism parser or an occlusion test.
