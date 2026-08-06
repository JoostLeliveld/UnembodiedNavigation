# warehouse_full_4cam Layout

Canonical world: `src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf`

![Top-down layout](assets/warehouse_full_4cam_map.svg)

PNG preview: `docs/assets/warehouse_full_4cam_map.png`

## Map Decision

- Footprint: ground plane `24.5 x 20.5 m`, walls at `x = +/-12 m`, `y = +/-10 m`.
- Structure: two four-row rack blocks copied from the old warehouse style, spaced so the inner rack faces bound a `4.50 m` central aisle instead of a broad empty hall. Rack segments were shortened so both interior cross-aisles retain `0.96 m` between conservative no-go envelopes.
- One-sided shelf: an extra ShelfD/E row is backed directly against the west wall, so it can only be reached from the aisle side.
- Floor semantics: beige is the physical floor, cyan is the exact planner driveable union from `world_profiles.yaml`, blue is the site boundary, and green is a conservative no-go envelope expanded by 0.32 m beyond collision geometry. Cyan and green regions are contract-checked not to overlap.
- Aisle cleanup: the dropped west-aisle stack, four loose south-apron crates, and east-service pallet-box island were removed. The fixed building pillar remains and is represented as a local two-sided bypass instead of removing the whole center strip.
- Camera placement: four high wall-mounted cameras sit at `(-6.0, -10)`, `(-6.0, 10)`, `(6.0, -10)`, and `(6.0, 10)` and look perpendicular to their wall. Bringing the columns closer to the central aisle widens adjacent-camera overlap for handover validation.
- Presentation overview: a separate top-down camera at `(0.0, 0.0, 26.0)` produces a full-facility Gazebo view for media only. It is not a fifth localization camera and is excluded from the GP/fusion/planner interfaces.
- Occlusion/handover: W-block inner north racks and E-block inner south racks are tall, creating different blind regions for the north/south camera pairs.
- Styling: box collision/contact sensors remain the operational geometry; ShelfD/E AWS meshes are visual overlays so rendered cameras see the same shelf footprint.
- Added AWS props: bucket, trash can, clutter piles, pallet jack, lamps, and a `DeskC_01` quality-control desk near the north-east wall.

## Cameras

- Camera A: include `external_camera`, mount `south wall, west dock column`, pose `-6.00 -10.00 6.10 0 0.92 1.5708`, RGB topic `/external_camera/image_raw`
- Camera B: include `external_camera_b`, mount `north wall, west dock column`, pose `-6.00 10.00 6.10 0 0.92 -1.5708`, RGB topic `/external_camera_b/image_raw`
- Camera C: include `external_camera_c`, mount `south wall, east dock column`, pose `6.00 -10.00 6.10 0 0.92 1.5708`, RGB topic `/external_camera_c/image_raw`
- Camera D: include `external_camera_d`, mount `north wall, east dock column`, pose `6.00 10.00 6.10 0 0.92 -1.5708`, RGB topic `/external_camera_d/image_raw`

Approximate FOV wedges in the SVG are visual planning aids only. Runtime projection still comes from the camera model and calibration.

## Planner Driveable Aisles (cyan)

- `west_service_lane`: x `[-10.95, -9.42]`, y `[-8.35, +8.35]` - lane between the west wall-backed shelf and rack column W1
- `rack_aisle_W3`: x `[-8.23, -7.32]`, y `[-8.35, +8.35]` - aisle between rack columns W1 and W2
- `rack_aisle_W2`: x `[-6.13, -5.22]`, y `[-8.35, +8.35]` - continuous west aisle 2; loose mid-aisle box stacks were removed
- `rack_aisle_W1`: x `[-4.03, -3.12]`, y `[-8.35, +8.35]` - aisle between rack columns W3 and W4
- `central_aisle_south`: x `[-1.93, +1.93]`, y `[-8.35, -1.47]` - full central aisle south of the support pillar
- `central_pillar_bypass_west`: x `[-1.93, -0.57]`, y `[-1.47, -0.33]` - west-side bypass around the support-pillar envelope
- `central_pillar_bypass_east`: x `[+0.57, +1.93]`, y `[-1.47, -0.33]` - east-side bypass around the support-pillar envelope
- `central_aisle_north`: x `[-1.93, +1.93]`, y `[-0.33, +8.35]` - full central aisle north of the support pillar
- `rack_aisle_E1`: x `[+3.12, +4.03]`, y `[-8.35, +8.35]` - aisle between rack columns E1 and E2
- `rack_aisle_E2`: x `[+5.22, +6.13]`, y `[-8.35, +8.35]` - aisle between rack columns E2 and E3
- `rack_aisle_E3`: x `[+7.32, +8.23]`, y `[-8.35, +8.35]` - aisle between rack columns E3 and E4
- `east_service_lane`: x `[+9.42, +11.25]`, y `[-8.35, +8.35]` - continuous east service lane after removal of the artificial pallet-box island
- `south_perimeter_aisle`: x `[-10.95, +11.25]`, y `[-8.35, -6.82]` - cross aisle south of the lower rack band
- `lower_cross_aisle`: x `[-10.95, +11.25]`, y `[-2.48, -1.52]` - widened cross aisle between the lower and middle rack bands
- `upper_cross_aisle`: x `[-10.95, +11.25]`, y `[+1.92, +2.88]` - widened cross aisle between the middle and upper rack bands
- `north_perimeter_aisle`: x `[-10.95, +11.25]`, y `[+7.22, +8.35]` - cross aisle north of the upper rack band

## Conservative No-Go Envelopes

- `central_support_pillar`: center `(+0.00, -0.90)`, footprint `0.50 x 0.50 m` - fixed building column inside the 4.5 m central aisle
- `west_wall_backed_shelf`: center `(-11.63, +0.20)`, footprint `0.55 x 13.40 m` - ShelfD/E row backed directly against the west wall; reachable only from the east aisle side

## Tall Handover Segments

- `W2_north` at `(-6.72, +5.05)`, height `2.61 m`
- `W3_north` at `(-4.62, +5.05)`, height `2.61 m`
- `E2_south` at `(+4.62, -4.65)`, height `2.61 m`
- `E3_south` at `(+6.72, -4.65)`, height `2.61 m`

## AWS Props

- `bucket_dock`: AWS `Bucket_01` at `11.2 -8.9 0 0 0 0` (bucket)
- `trashcan_nw`: AWS `TrashCanC_01` at `-11.3 9.0 0 0 0 0.6` (trash)
- `clutterA_sw`: AWS `ClutteringA_01` at `-10.8 -9.2 0 0 0 0` (clutter A)
- `clutterC_ne`: AWS `ClutteringC_01` at `10.6 9.0 0 0 0 3.1416` (clutter C)
- `clutterD_dock`: AWS `ClutteringD_01` at `-2.5 -9.3 0 0 0 0` (clutter D)
- `palletjack_apron`: AWS `PalletJackB_01` at `-4.6 -9.1 0 0 0 0` (pallet jack)
- `desk_qc_ne`: AWS `DeskC_01` at `8.75 8.85 0 0 0 3.1416` (QC desk)

## Launch

```bash
ros2 launch sim bringup_sim.launch.py world:=warehouse_full_4cam.world.sdf bridge_camera_b:=true bridge_camera_c:=true bridge_camera_d:=true use_lidar:=false bridge_scan:=false
```

Generated by `scripts/geometry_visibility/make_warehouse_full.py`; edit the generator, not the SDF.
