#!/usr/bin/env python3
"""ONE warehouse, three storage sections, nothing mirrored.

    SECTION A   west, selective pallet racking, runs north-south, 11.75 m long
    SECTION B   north-east, pallet racking turned 90 deg, shorter runs, lower
    SECTION C   south-east, block-stacked boxes on the floor -- no racking at
                all -- in three different stacking patterns

    plus an inbound dock apron along the south wall with a dock office in the
    corner, a main north-south artery between the west and east halves, and a
    cross aisle that only exists on the east side.

Why three sections. A real DC does not store everything the same way: fast full-
pallet movers go on the floor in block stacks, case-pick stock goes in selective
racking, and a second product family often sits in its own block turned the other
way. It is also the honest way to get asymmetry -- the building is different in
different places, rather than the same block scattered irregularly.

Nothing is mirrored: section widths differ, the two racking sections have
different run lengths (3 modules west, 2 modules north-east) and different top
beams, the dock doors are unevenly spaced, and the north cross aisle is 2.90 m in
the west and does not exist in the east.

Realism decisions, each with a reason:

- Racking is back-to-back PAIRS: 0.88 + 0.14 flue + 0.88 = 1.90 m, as built.
- Runs are whole ShelfD/E modules at their native 3.917 m, tiled, never stretched,
  so the camera sees the shelf the detector was trained on.
- Block stacks are whole Cluttering pallets at native size, stacked 1, 2 or 3 high.
- Walls are 9.0 m (WallB_01's native height). Cameras hang at 3.6-6.0 m, below the
  wall top. The current world mounts them at 6.10 m on 4.50 m walls.

The map declares the section zones. It says nothing about what is in them, and
the two stock states below prove it: same map to the cell, different building to
every camera.
"""
from __future__ import annotations

import math

from layouts import Camera, Layout, Obstacle, Zone
from mesh_library import footprint, height

# --------------------------------------------------------------------------- #
MODULE = 3.917           # ShelfD/E native length
DEPTH = 0.88             # ShelfD/E native depth
FLUE = 0.14
PAIR = 2 * DEPTH + FLUE  # 1.90 m rack zone

SITE = (-11.35, 11.35, -9.35, 9.35)
APRON_TOP = -4.40        # dock apron is everything south of this (4.95 m deep)
DOCK_DOORS = [-8.5, -1.5, 5.5]      # unevenly spaced on purpose

# --- section A: west, north-south racking, three back-to-back pairs -----------
A_X = [-8.90, -5.20, -1.50]         # pitch 3.70 -> 1.80 m picking aisles
A_Y0, A_Y1 = APRON_TOP, APRON_TOP + 3 * MODULE          # -4.40 .. 7.35
# Top beam is TWO whole ShelfD modules, 5.23 m: 58 % of the 9.0 m clear height
# instead of 47 %, and no squashed part-tier in the render. Going taller still
# was measured and costs only 0.7 points of two-camera cover, so the limit here
# is what keeps the camera views usable, not the coverage.
A_TOP = {"full": 2 * 2.613, "half": 2.613, "picked": 1.10}

# --- main artery and the east half -------------------------------------------
ARTERY = (-0.55, 1.80)                                  # 2.35 m main aisle
B_X0, B_X1 = 1.80, 1.80 + 2 * MODULE                    # 1.80 .. 9.63
# Section B gives up its second rack pair so section C can have three rows and
# two full-length aisles. One pair still reads as "racking turned 90 degrees",
# and the floor it releases is what the east cameras were short of.
B_Y = [8.35]
B_Y0, B_Y1 = 7.40, 9.30
B_TOP = {"full": 2.613, "half": 1.90, "picked": 1.10}   # one module, lower

CROSS_E = (5.05, 7.40)                                  # east-side cross aisle, 2.35 m

# --- section C: block-stacked boxes, three patterns --------------------------
# Floor storage as two LONG east-west rows with one long aisle between them,
# rather than a grid of little bays. The aisle then runs the full 9.55 m width of
# the section instead of stopping every 2.35 m, which is how block storage is
# actually laid out and gives the robot a proper run. Stack height varies ALONG
# each row, so one, two and three high are all visible from one aisle.
# The rows stop 1.50 m short of the site edge so there is an east perimeter lane.
# Without it the stacks ran into the wall camera's face and it saw 14 % of the floor.
C_X0, C_X1 = 1.80, 9.85                  # 8.05 m of row, east lane 9.85 .. 11.35
C_ROWS = [("Cs", -4.40, -2.25), ("Cm", -0.75, 1.40), ("Cn", 2.90, 5.05)]
C_AISLES = [(-2.25, -0.75), (1.40, 2.90)]               # two 1.50 x 9.55 m aisles
C_PALLET = "ClutteringA_01"

# Peak vs post-peak. Rows are the racking runs, columns the bays along each run.
A_PEAK = [["full", "full", "full"],
          ["full", "half", "full"],
          ["full", "full", "half"]]
A_AFTER = [["picked", "picked", "half"],
           ["picked", "picked", "picked"],
           ["half", "empty", "picked"]]      # middle bay out for a beam repair
B_PEAK = [["full", "full"]]
B_AFTER = [["picked", "half"]]

# Four stacks along each row, and how many pallets high each one is. Peak is a
# full floor; after the ship-out two positions are cleared and the rest come down.
C_PEAK = {"Cs": [1, 2, 3], "Cm": [2, 3, 1], "Cn": [3, 1, 2]}
C_AFTER = {"Cs": [0, 1, 2], "Cm": [1, 0, 1], "Cn": [1, 1, 0]}


# --------------------------------------------------------------------------- #
def _rack_pair(name, x, y0, y1, bays, tops, mesh_off=0):
    """Two faces of a back-to-back pair, one entry per bay along the run."""
    out = []
    for b, level in enumerate(bays):
        if level == "empty":
            continue
        yc = y0 + MODULE * (b + 0.5)
        for face, dx in (("w", -(FLUE / 2 + DEPTH / 2)), ("e", +(FLUE / 2 + DEPTH / 2))):
            mesh = "ShelfD_01" if (b + mesh_off) % 2 == 0 else "ShelfE_01"
            out.append(Obstacle(f"{name}b{b+1}{face}", x + dx, yc, DEPTH, MODULE,
                                0.0, tops[level], mesh, name))
    return out


def _rack_pair_ew(name, y, x0, x1, bays, tops, mesh_off=0):
    """Same, for a run turned 90 degrees."""
    out = []
    for b, level in enumerate(bays):
        if level == "empty":
            continue
        xc = x0 + MODULE * (b + 0.5)
        for face, dy in (("s", -(FLUE / 2 + DEPTH / 2)), ("n", +(FLUE / 2 + DEPTH / 2))):
            mesh = "ShelfD_01" if (b + mesh_off) % 2 == 0 else "ShelfE_01"
            out.append(Obstacle(f"{name}b{b+1}{face}", xc, y + dy, MODULE, DEPTH,
                                0.0, tops[level], mesh, name))
    return out


def _block_row(z: Zone, mesh: str, highs: list[int]):
    """Whole pallets set out along a storage row, each stacked its own number
    high. Positions are evenly spaced across the row so the gaps read as the
    working clearance between stacks."""
    sx, sy = footprint(mesh)
    inset = 0.06
    if sy > z.sy - 2 * inset:               # turn the pallet to fit the row depth
        sx, sy = sy, sx
    if sy > z.sy - 2 * inset:
        raise AssertionError(f"pallet {sy:.2f} m deep does not fit row {z.name} "
                             f"({z.sy:.2f} m)")
    n = len(highs)
    span = z.sx - 2 * inset
    if n * sx > span:
        raise AssertionError(f"{n} pallets ({n*sx:.2f} m) do not fit row {z.name} "
                             f"({span:.2f} m)")
    pitch = span / n
    out = []
    for k, nh in enumerate(highs):
        if nh < 1:
            continue                        # this position is empty
        cx = z.xmin + inset + pitch * (k + 0.5)
        out.append(Obstacle(f"{z.name}_p{k}", cx, z.cy, sx, sy, 0.0,
                            height(mesh) * nh, mesh, z.name))
    return out


def build() -> Layout:
    L = Layout(
        "V2", "warehouse_v2",
        "Three storage sections that work differently: north-south selective "
        "racking in the west, a lower rack block turned 90 degrees in the "
        "north-east, and floor block-stacking in the south-east where the boxes "
        "are stacked one, two and three high in three separate patterns.",
        "Asymmetry comes from the building being genuinely different in "
        "different places, which is also what makes the stock change bite: the "
        "block-stack section can lose two metres of height without the map "
        "moving a cell.")

    # ---- zones: this, and only this, is what the map declares ----------------
    for i, x in enumerate(A_X):
        L.zones.append(Zone(f"A{i+1}", x - PAIR / 2, x + PAIR / 2, A_Y0, A_Y1))
    for i, y in enumerate(B_Y):
        L.zones.append(Zone(f"B{i+1}", B_X0, B_X1, y - PAIR / 2, y + PAIR / 2))
    for name, y0, y1 in C_ROWS:
        L.zones.append(Zone(name, C_X0, C_X1, y0, y1, kind="block_stack"))
    L.zones += [
        Zone("DOCK_OFFICE", -11.35, -8.85, -9.35, -7.55, kind="structure"),
        Zone("STAGE_MID", -4.60, -2.20, -9.35, -8.15, kind="staging"),
        Zone("STAGE_E", 2.60, 5.00, -9.35, -8.15, kind="staging"),
    ]

    # ---- contents: not declared anywhere ------------------------------------
    for tag, a_state, b_state, c_state in (("fill_a", A_PEAK, B_PEAK, C_PEAK),
                                           ("fill_b", A_AFTER, B_AFTER, C_AFTER)):
        objs = []
        for i, x in enumerate(A_X):
            objs += _rack_pair(f"A{i+1}", x, A_Y0, A_Y1, a_state[i], A_TOP, i)
        for i, y in enumerate(B_Y):
            objs += _rack_pair_ew(f"B{i+1}", y, B_X0, B_X1, b_state[i], B_TOP, i + 1)
        for name, _y0, _y1 in C_ROWS:
            z = [q for q in L.zones if q.name == name][0]
            objs += _block_row(z, C_PALLET, c_state[name])
        # dock office and staged pallets on the apron
        oz = [q for q in L.zones if q.name == "DOCK_OFFICE"][0]
        objs.append(Obstacle("dock_office", oz.cx, oz.cy, oz.sx - 0.06, oz.sy - 0.06,
                             0.0, 2.80, "WallB_01", "DOCK_OFFICE"))
        peak = tag == "fill_a"
        for nm, h in (("STAGE_MID", 1.72 if peak else 1.06),
                      ("STAGE_E", 1.06 if peak else 1.72)):
            z = [q for q in L.zones if q.name == nm][0]
            for k in range(2):
                objs.append(Obstacle(f"{nm}{k}", z.xmin + 0.70 + 1.05 * k, z.cy,
                                     0.95, 1.00, 0.0, h, "ClutteringC_01", nm))
        setattr(L, tag, objs)

    # ---- cameras: four mounts, four heights, no mirrored pair ---------------
    # Chosen by measurement, not taste: seven placements were ray-cast against
    # this geometry. A low 3.6 m mount on the dock-office roof looked plausible
    # and cost 12 points of two-camera cover -- too grazing. Four corners plus a
    # dock camera scored the same but is the mirrored set this layout exists to
    # avoid. This set has no north-east camera, so it is not a mirrored pair.
    # MOUNT HEIGHT IS CAPPED AT 5.0 m -- a constraint, not an optimum. Measured
    # against this geometry it costs about 8 points of single-camera and 18 points
    # of two-camera cover against an 8.0 m mount. Three consequences worth knowing
    # rather than rediscovering:
    #
    #  * At 5 m the racking is TALLER than the cameras, and rack height above
    #    ~2.6 m then makes no difference at all to what floor is visible: a 2.61 m
    #    rack blocks an oblique view into a 1.80 m aisle exactly as completely as
    #    a 5.23 m one. The stock lever therefore comes from the floor storage and
    #    from bays picked down BELOW the camera line, not from the difference
    #    between a full and a half-full rack.
    #  * Shallower pitch pays down here: -4 degrees on every camera is worth 11
    #    points of two-camera cover, because reach matters more than look-down
    #    angle once the camera is low.
    #  * Aisle-end mounts are much worse at this height (2 % two-camera cover): a
    #    5 m camera looking along a 1.80 m aisle sees rack faces, not floor. The
    #    corner-and-dock arrangement survives the height change; camera C is
    #    turned to -50 deg, which is worth 3 points.
    L.cameras = [
        Camera("A", -11.45, -9.45, 5.00, 45.0, 44.0,
               "south-west corner, apron and the foot of section A", "#2a78d6"),
        Camera("B", -1.50, -9.72, 5.00, 96.0, 48.0,
               "south wall above the centre dock door, up the artery", "#eb6834"),
        # C is an AISLE-END mount, not a corner mount, and that is a deliberate
        # trade measured on this geometry (21 placements, both stock states).
        # From the north-west corner it saw only 3.2 % of the two section-A
        # picking aisles, leaving aisle A1|A2 at 17.8 % single-camera cover
        # against 84.2 % for the floor as a whole -- the aisles were the one
        # place the network was effectively blind, and re-aiming the corner mount
        # does not help at all (A1|A2 stays at 17.8 % at every yaw from -30 to
        # -70). Moving it to the head of aisle A1|A2 and looking straight down it
        # takes that aisle to 100 % and both aisles together from 34.2 % to
        # 76.9 %. The price, stated because the earlier note warned about exactly
        # this: whole-floor single cover falls 84.2 -> 82.0 and TWO-camera cover
        # falls 52.9 -> 42.6. Pitch 38 deg rather than the 24 deg that would put
        # the far aisle end on the horizon -- 38 measured better on both aisles.
        # KNOWN RESIDUAL: two-camera cover inside the aisles stays near 10 %, so
        # fusion is still barely testable there. The alternative mount at
        # (-3.35, 9.45), down aisle A2|A3, reaches 27.8 % two-camera aisle cover
        # instead, because camera B already covers that aisle so a second view
        # there creates overlap rather than new floor. That is the placement to
        # switch to if aisle FUSION matters more than aisle blindness.
        Camera("C", -7.05, 9.45, 5.00, -90.0, 38.0,
               "north wall at the head of aisle A1|A2, straight down the aisle",
               "#1baf7a"),
        Camera("D", 11.45, 7.20, 5.00, -140.0, 38.0,
               "east wall in the cross aisle, over the box rows", "#4a3aa7"),
        Camera("E", 11.45, -9.45, 5.00, 132.0, 42.0,
               "south-east corner, over the block stacks", "#e34948"),
    ]
    L.notes = [
        "Section A: 3 back-to-back pairs, 1.80 m picking aisles, runs 11.75 m "
        "(three native modules), 3-level racking to 4.2 m.",
        "Section B: one pair turned 90 deg against the north wall, runs 7.83 m "
        "(two modules), top beam 2.61 m -- a different product family, not a "
        "copy of A.",
        "Section C: no racking. Three long floor-storage rows with two 1.50 m "
        "aisles running their full 9.55 m length, four stacks per row at one, "
        "two or three pallets high: 1.06 / 2.12 / 3.17 m.",
    ]
    bad = L.check()
    if bad:
        raise AssertionError(f"contents escape their zone: {bad[:5]}")
    return L


if __name__ == "__main__":
    L = build()
    print(f"{len(L.zones)} zones | {len(L.fill_a)} objects at peak | {len(L.fill_b)} after")
    print(f"apron        {SITE[2]:+.2f} .. {APRON_TOP:+.2f}  ({APRON_TOP - SITE[2]:.2f} m deep)")
    print(f"section A    x {A_X[0]-PAIR/2:+.2f} .. {A_X[-1]+PAIR/2:+.2f}, "
          f"y {A_Y0:+.2f} .. {A_Y1:+.2f}   aisles {A_X[1]-A_X[0]-PAIR:.2f} m")
    print(f"artery       x {ARTERY[0]:+.2f} .. {ARTERY[1]:+.2f}  ({ARTERY[1]-ARTERY[0]:.2f} m)")
    print(f"section B    x {B_X0:+.2f} .. {B_X1:+.2f}, y {B_Y0:+.2f} .. {B_Y1:+.2f}   "
          f"{len(B_Y)} pair against the north wall")
    print("section C    three long floor-storage rows")
    for name, y0, y1 in C_ROWS:
        print(f"   {name:<3} x {C_X0:+.2f}..{C_X1:+.2f} ({C_X1-C_X0:.2f} m long)  "
              f"y {y0:+.2f}..{y1:+.2f} ({y1-y0:.2f} m deep)  stacks {C_PEAK[name]} high")
    for a, b in C_AISLES:
        print(f"   box aisle y {a:+.2f}..{b:+.2f}: {b-a:.2f} m wide, {C_X1-C_X0:.2f} m long")
    print(f"   east cross aisle y {CROSS_E[0]:+.2f}..{CROSS_E[1]:+.2f}: "
          f"{CROSS_E[1]-CROSS_E[0]:.2f} m")
    print(f"north aisle  west {SITE[3]-A_Y1:.2f} m, east cross aisle "
          f"{CROSS_E[1]-CROSS_E[0]:.2f} m")
