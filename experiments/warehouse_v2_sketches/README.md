# warehouse_v2 — one layout, three storage sections

A replacement for `warehouse_full_4cam`. One building, three sections that store goods
three different ways, and five cameras placed by measurement rather than by taste.

![plan](figures/warehouse_v2.png)

```
   +--------------------------------------------------------------+
   |  || || ||        |  ====================================      |
   |  || || ||        |  ====================================      |   SECTION B
   |  || || ||   A    |  ------------- cross aisle ----------      |   racking, turned 90°
   |  || || ||        |  [ 1 high ]  [ 2 high ]  [ 3 high ]        |   SECTION C
   |  || || ||        |  [        ]  [        ]  [        ]        |   boxes on the floor
   |                  artery                                       |
   |         INBOUND DOCK APRON                                    |
   +===[]==============[]====================[]===================+
```

`python3 render_v2.py` rebuilds the figure. Geometry lives in `warehouse_v2.py`.

## Three sections, none a copy of another

| | what it is | how big |
|---|---|---|
| **A** west | selective racking, runs **north–south**, **blue** steel | 3 back-to-back pairs, runs 11.75 m, top beam **5.23 m** |
| **B** north-east | racking **turned 90°**, lower, **orange** steel | one pair against the north wall, runs 7.83 m, top beam **2.61 m** |
| **C** south-east | **no racking** — floor storage, **yellow** bay paint | **three 8.05 m rows** with **two 1.50 m aisles running their full length**, plus a 1.50 m east perimeter lane; three stacks per row at **1, 2 or 3 pallets high**: 1.06 / 2.12 / 3.17 m |

A real DC does not store everything the same way: fast full-pallet movers go on the floor,
case-pick stock goes in racking, and a second family often sits in its own block turned the
other way. That is also the honest way to get asymmetry — the building is genuinely
different in different places, rather than one block scattered irregularly.

Nothing is mirrored. The two racking sections have different run lengths and different top
beams, the three block-stack bays are different widths, the dock doors are unevenly spaced
(x = −8.5, −1.5, +5.5), and the north cross aisle is 2.90 m in the west and does not exist
in the east.

Every dimension is a measured mesh dimension or a standard racking dimension:
back-to-back pairs are 0.88 + 0.14 flue + 0.88 = **1.90 m**, runs are whole ShelfD/E
modules at their native 3.917 m **tiled, never stretched**, so the camera sees the shelf the
detector was trained on. Walls are 9.0 m (`WallB_01` native); cameras hang at 5.2–6.0 m,
below the wall top.

## Cameras — chosen by measurement

Seven placements were ray-cast against this geometry before picking one.

| variant | seen by ≥1, peak | by ≥2, peak | stock lever |
|---|---|---|---|
| low 3.6 m mount on the dock-office roof | 77.5 % | **40.4 %** | 31.9 % |
| four corners + a dock camera (mirrored) | 82.8 % | 51.0 % | 41.0 % |
| **chosen: SW corner, dock door, NW corner, east wall, SE corner** | **81.3 %** | **52.0 %** | **31.7 %** |
| two aisle-end column cameras | 79.1 % | 24.7 % | 45.4 % |

Two things fell out of that. The low dock-office mount looked plausible and cost **12 points
of two-camera cover** — too grazing. And a pair of cameras at the feet of two aisles halves
two-camera cover, because they see the same aisle rather than different ones.

The chosen set has **no north-east camera**, so it is not a mirrored pair, and it beats the
mirrored four-corner set on two-camera cover anyway.

## What it is for

The map declares zones — three racking pairs in A, two in B, three block-stack bays in C,
the dock office and two staging pads. It says nothing about what is in them.

The figure shows the same warehouse in two stock states. **Peak**: racking full to the top
beam, floor stacked three high. **After the post-peak ship-out**: racking picked down to the
bottom beam with one bay out for a beam repair, floor stacks down to one high.

| | peak | after ship-out |
|---|---|---|
| drivable map | identical, to the cell | identical, to the cell |
| lanes seen by ≥1 camera | 84 % | 96 % |
| lanes seen by ≥2 cameras | 53 % | 75 % |
| lanes whose 2-camera cover changes | — | **24 %** |

## The 5 m mount cap, and what it costs

Cameras are capped at **5.0 m**. That is a constraint, not an optimum, and the price is
measured: against an 8.0 m mount on the same geometry it costs about **8 points of
single-camera and 18 points of two-camera cover** (65 % → 47 % at peak).

Three things follow that are worth knowing rather than rediscovering:

- **At 5 m the racking is taller than the cameras, and rack height then stops mattering.**
  A 2.61 m rack blocks an oblique view into a 1.80 m aisle exactly as completely as a
  5.23 m one — coverage is identical to the decimal at 2.61, 3.50 and 5.23 m. So the stock
  lever no longer comes from the difference between a full and a half-full rack; it comes
  from the floor storage and from bays picked down **below** the camera line.
- **Shallower pitch pays down here.** −4° on every camera is worth **11 points** of
  two-camera cover: once the camera is low, reach matters more than look-down angle.
- **Aisle-end mounts are much worse at this height**, not better as one might expect —
  2 % two-camera cover. A 5 m camera looking along a 1.80 m aisle sees rack faces, not
  floor. The corner-and-dock arrangement survives the height change; camera C turned to
  −50° is worth another 3 points.

Two more things were fixed by opening the east half rather than by moving a camera:

- **Section C used to run straight into the east wall**, so camera D had a wall of pallets in
  its face and saw 14 % of the floor. Pulling the rows 1.50 m back for an east perimeter lane
  raised the whole world by 3 points of single-camera cover.
- **Camera D was still aimed at the rack pair section B had just given up.** Moving it into
  the east cross aisle at (+11.45, +7.20), yaw −140°, raised its own share 14.7 % → 21.2 %
  *and* combined two-camera cover 48.5 % → 52.9 %. That is the one move in this whole exercise
  where a camera's own share and the combined figure went up together.

Camera C was swept again after all of it and is already at its best; every alternative is worse
on both counts.

The current world's equivalent number is **0 %**: its zones *are* its objects, so nothing can
change without moving an obstacle the map declares. It also only reaches two cameras on 26 %
of its floor, against 52 % here at peak and 81 % after.

Note what peak season costs: 16 % of the lanes have no camera at all when the racking is
full. Section A's aisles become canyons only seen from their ends. That is a real property of
a full warehouse and it is the thing a planner should be avoiding.

## The dimension audit, and what it changed

Every number was checked against what the equipment implies, and three of them were wrong.

| | before | verdict | after |
|---|---|---|---|
| section A top beam | 4.20 m | **too low** — 47 % of a 9 m clear height | **5.23 m**, two whole modules, 58 % |
| dock apron | 4.05 m | **too thin** — a turning space, not a staging area | **4.95 m** |
| section C | one 7.20 m deep slab, reachable only from its ends | **not how floor storage is laid out** | three long rows, **two full-length aisles** and an east perimeter lane. Section B gave up a rack pair to pay for it |
| picking aisle | 1.80 m | fine, but it implies an AMR or very-narrow-aisle operation, not a counterbalance forklift | unchanged, assumption stated |
| main artery | 2.20 m | single lane | 2.35 m — still single lane, and that is a deliberate limit of a 24 m building |
| clear height | 9.0 m | right | unchanged |

Two of those were settled by measurement rather than taste:

- **Raising the racking is nearly free.** Going from 4.20 m to 6.27 m costs **0.7 points** of
  two-camera cover, because a rack that already blocks an aisle at 4.2 m does not block more
  floor by being taller. The limit on rack height here is what keeps the camera views usable,
  not coverage. 5.23 m was chosen because it is two whole ShelfD modules — no squashed
  part-tier in the render.
- **Lowering the building instead would have been expensive.** A 7.0 m roof with 6.3 m
  cameras costs **10 points** (64.5 % → 54.8 %). So the 9 m roof stays and the racking rises
  to meet it, not the other way round.

The cameras were re-swept on the new geometry: +4° of pitch was worth 1.6 points, and camera
C's placement was re-tested against seven alternatives. C looks like the weak one — it sees
only 21 % of the floor alone — but every move that raises its own share **lowers** combined
two-camera cover (35.9 % alone but 60.3 % combined, against 21.5 % / 65.5 % where it is).
What a second camera is for is seeing cells another camera already sees.

## Colour, and the permanent steel

The AWS pack cannot supply colour variety, so it is painted on, the way this repo's older
world paints its blue rails:

- **section A** blue racking steel, **section B** orange — two racking sections that no longer
  read as the same thing;
- **section C** yellow painted floor bays, so the block-storage grid is legible as marked
  floor rather than as scattered pallets;
- yellow kick plates at rack ends and yellow dock-door sills.

The painted frames stand at the section's **design** height in every stock state, and are
**visual-only**. That is deliberate and more honest than the alternative: real racking steel
does not come and go, only the goods on it do, and open steel is not much of an occluder. So
the frame is permanent scenery and the *goods* are the collision box the sight-line model
sees. The check that `warehouse_v2_rack_frames` is link-for-link identical between the two
stock states is in the build.

## How the numbers are made

- Geometry: measured bounding box of each `*_visual.DAE` after the COLLADA `<unit meter>`
  factor, `<up_axis>` and the visual-scene transforms (`mesh_library.py`). Validated — it
  returns 3.917 × 0.880 × 2.613 m for `ShelfD_01`, exactly the constant
  `make_warehouse_full.py` measured independently.
- Cameras: the project's own `unav_common.camera_model.ObliqueCameraModel`, 1280 × 720,
  90° horizontal FOV. The runtime projection, not a re-derivation.
- Visibility: marker at 0.35 m is seen when it projects inside the image and 48 samples
  along the sight-line all clear the height map. Grid 0.10 m.
- Lanes are **derived**, not typed: site field, minus zones plus 0.32 m, largest 4-connected
  component, greedy maximal-rectangle cover, 0.70 m minimum width. 15 rectangles capture
  98.7 % of reachable free space and **0 cells** of a lane fall inside a keep-out envelope —
  a property of the construction, not a check that happened to pass.
- Contents are asserted to stay inside their zone; the build fails otherwise.
- Resolution is reported (54 px/m median at covered cells), not used as a gate: a 40 px/m cut
  would exclude cells past ~9 m, and this detector hits 98.2 % of sight-line poses across the
  whole grid at confidence 0.25.

## Two defects in the current world, worth fixing either way

Both measured, not inferred:

1. **The cameras float above the walls.** Mounts at z = 6.10 m; all four walls are 4.50 m
   boxes. warehouse_v2 uses 9.0 m walls and mounts at 5.2–6.0 m.
2. **The hanging lamps are at 18 m.** `Lamp_01`'s geometry sits at z ∈ [12.50, 14.02] in its
   own DAE and the world includes it at pose z = 5.95 with no offset, putting the fixtures at
   z ∈ [18.45, 19.97]. To hang one at 4.2 m the include pose needs z = −8.30.

## About "more colours"

Measured from the textures, not judged by eye: `ShelfD_01` and `ShelfE_01` share **all four**
texture files, and one cardboard texture is shared by **seven** models. Every dominant colour
in the pack is ochre `#c59653`, yellow steel `#d2bc00`, or dark frame `#383331`. The only
breakouts are the trash can (green), the walls (off-white), the lamp/roof (grey), and
`GroundB_01`, which is the one coloured asset — teal floor with yellow lane paint.

**Picking different AWS meshes will not make the warehouse more colourful.** Colour has to be
added: painted `<material>` on the rack collision boxes (the current world already does this
for its blue rails), floor-zone paint in the `GroundB_01` idiom, or re-tinted copies of the
shared texture. None of that touches geometry.

## Files

| file | what it is |
|---|---|
| `warehouse_v2.py` | the layout: sections, zones, stock states, cameras. Asserts contents stay inside their zone |
| `render_v2.py` | the figure |
| `mesh_library.py` | measures every AWS DAE and texture → `mesh_inventory.json` |
| `coverage.py` | height map, derived lanes, ray-cast visibility, crossing angles |
| `baseline.py` | parses the current world's SDF + `world_profiles.yaml` for comparison |
| `layouts.py`, `render_sketches.py`, `compare.py` | five earlier candidates, rejected as over-designed. Kept because `coverage.py` and the baseline numbers came out of them; figures in `figures/rejected/` |

## Built in Gazebo

```bash
python3 make_world.py                 # -> src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf
python3 make_world.py --state B --out .../warehouse_v2_shipout.world.sdf
python3 verify_world.py               # 182 + 146 numeric checks, both states
colcon build --packages-select sim --symlink-install
bash sim_up.sh <tag> [world]          # headless, 7 streams, frames + contact sheet
python3 overlay.py <tag>              # planned geometry drawn on the render
python3 compare_sketch.py             # sketch beside both built states
```

![sketch vs world](figures/sketch_vs_world.png)

Two worlds are generated from the same layout: `warehouse_v2.world.sdf` (peak stock) and
`warehouse_v2_shipout.world.sdf` (after the ship-out). Their `warehouse_shell` and
`known_driveable_boundary` models are **link-for-link identical** — checked, not asserted —
so the building and the painted map are the same object in both, and only the 38 vs 34
contents boxes differ.

**The world is checked two ways, because neither is enough alone.**

`verify_world.py` checks what a render cannot show — 182 checks in the peak state, 146 in
the ship-out state, all passing:

- every fill object has a collision box at the planned centre, size and height;
- no collision box escapes its declared zone;
- every box is dressed by a mesh visual, and **no mesh is stretched** — horizontal scale is
  1:1 and vertical scale never exceeds 1.0, because a 4.2 m rack is a native 2.613 m ShelfD
  module with a shorter one stacked on it, not one module scaled 1.6×;
- every camera pose in the SDF equals the layout, sits inside the building, and stays clear
  of the wall top;
- the shell, the floor paint and the rack frames are link-for-link identical between the two
  stock states, so only the goods differ;
- the south wall is closed from floor to roof except at the three dock openings (this caught
  a real defect: the first lintels stopped at 7.1 m and left a 1.9 m slot above every door);
- every lamp hangs above the sight-lines and below the roof (this caught the second: the
  fixtures poked 0.9 m through the roof).

`overlay.py` projects the planned rectangles onto the near-orthographic plan render. The
projection is verified against two independent features of the render — the inner face of the
south wall lands at v = 994 px and the east wall at u = 1273 px — so anything that does not
sit on the object it describes is a real mismatch, not perspective.

### What the iterations actually fixed

| | found by | fix |
|---|---|---|
| cameras at 6.0 m were eye-to-eye with a 4.2 m rack 4 m away; the views were nearly all rack face | rendering camera A | mount at 8.0 m under the 9.0 m roof — **+12 points of two-camera cover**, 54 % → 64 % |
| a 1.9 m slot above every dock door, daylight visible in camera A | the wall check | lintels run from the 4.50 m door head to the roof |
| lamps 0.9 m through the roof | the lamp check | hang at 7.00–8.52 m; the include pose has to be **negative** (−5.50) because `Lamp_01`'s geometry sits at z ∈ [12.50, 14.02] in its own DAE |
| the floor was blown out white | rendering | directional light 0.85 → 0.62, point lights 0.55 → 0.30 with heavier attenuation |
| the overview camera was rotated 90° and framed the hall at a third of the frame | rendering | 19 m with yaw 1.5708 — north up, east right |

Re-aiming was tested and rejected: fifteen yaw combinations for cameras A and C were swept and
the planned aim (A +45°, C −38°) was already the best two-camera coverage available.

The robot spawns at (0.55, −7.50) on the apron and is seen by **four** of the five cameras at
peak stock. It renders 34 px across at 6.5 m from camera B — the same target scale the existing
detector already works on.

`world_profiles.yaml` gained a `warehouse_v2.world.sdf` entry: the spawn, the 15 derived lane
rectangles, and the 11 storage zones with their 0.32 m envelopes. The lanes in the profile are
the same rectangles `coverage.py` derived, so the map the planner is handed and the map the
sketch was scored against are one object.

## Not done yet

- No campaign has been run in it: the detector has never been pointed at these frames, so
  there is no measured detection rate for this world yet, only ray-cast geometry.
- Peak and post-peak are one designed pair, not a distribution over stock states.
- `plan_view_camera` is a checking aid bridged by `sim_up.sh`, deliberately outside the
  launch's camera registry so it can never be mistaken for a sixth localisation camera.
- Camera pitch was held at the values above during the placement sweep; only position and
  yaw were varied.
