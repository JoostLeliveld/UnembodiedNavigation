# Oriented rectangular collision model

The planner and local tracker now use an **0.80 m long × 0.55 m wide rectangle**,
rotated by the predicted heading. These defaults match the AMR chassis collision box
in `src/sim/robot_description/urdf/warehouse_amr.urdf.xacro`. The wheel and caster
footprints lie within that chassis outline. The model is planar and assumes the
robot remains upright.

This supersedes the circular-body decisions in `parameter_rationale.md`. The
`robot_collision_radius_m` parameter remains for existing logger diagnostics and
historical interfaces. It no longer decides planner or tracker body clearance.
`robot_length_m` and `robot_width_m` are declared, forwarded to local/global planners,
included in planner cache identity, validated in campaigns and recorded in manifests.

## Geometry and motion

`unav_common/rectangular_footprint.py` builds the union of the supplied axis-aligned
obstacle rectangles using Shapely. A pose transforms all four chassis corners. For
obstacles, any intersection is refused, including contact. For keep-in lanes, the
whole rectangle must be covered by the lane union. Shared rectangle edges inside a
lane union are not obstacles, and holes under the body cannot be hidden by checking
only its corners.

Positive clearance is the Euclidean distance between the body and forbidden geometry.
Negative values indicate overlap; their magnitude is an overlap-area-derived indicator,
**not a penetration distance**. They are suitable for validity decisions, not physical
penetration-depth reporting. Empty obstacle scenes are unconstrained; empty declared
keep-in scenes refuse motion.

Checking endpoints is insufficient during both translation and rotation. Each control
interval now checks:

1. the planner's linear position interpolation with unwrapped heading;
2. the exact constant-command unicycle arc for that interval.

The estimator's existing Euler propagation is unchanged. Both paths must pass, so
the geometry check does not silently assume they are identical. Exact-command
endpoints are carried forward across the tape as well, so accumulated turning
displacement is not reset to the Euler path at every control boundary.

The sweep uses adaptive interval certification. Any chassis point moves no farther
than centre travel plus the half-diagonal times angular travel. If midpoint clearance
exceeds the corresponding half-interval travel bound, the entire interval is clear.
Otherwise it subdivides. Observed intersections and intervals still unresolved after
16 subdivisions are refused. This is a conservative clearance certificate over the
modeled paths, rather than a fixed grid that can step over a thin obstacle. The
half-diagonal is used only to bound motion, not as the accepted body shape.

Global trajectory validity and the local prefix gate call these same sweep methods.
In-place rotation recovery must pass them too. A body already overlapping forbidden
geometry does not get an allowance to move out through the obstacle: the interval
contains an invalid initial state. This may stop a run that previously proceeded on
the assumption that a predicted overlap was merely belief error.

## No-go cost versus body fit

The existing no-go objective retains its centre standoff, warning band and belief-cost
options. These remain **soft route preferences**, not a surrogate for the body shape.
The hard gate now requires rectangle fit instead of requiring the centre to maintain
the old circular standoff, or permitting the global mean to leave the lane by 5 cm.
The enabled no-go geometry is still enforced when its penalty weight is zero; disabling
the entire no-go model remains a separate configuration choice.

Consequently an aligned chassis fits a 70 cm straight aisle with 7.5 cm per-side
clearance, while the sideways chassis does not. That aisle may still carry a high
soft no-go cost under the existing 0.55 m centre standoff. This change establishes
feasibility; it does not retune the objective to prefer narrow routes.

## Verification and limits

Regression cases cover aligned/sideways aisle fit, corner contact during rotation,
thin walls between endpoints, the difference between an Euler segment and a turning
arc, lane-union seams, holes, concave lane corners, heading wraparound and explicit
full rotations, invalid dimensions, empty scenes and unresolved sweep refusal.
Integration tests exercise the actual global diagnostics and local execution gate.
The dimensions are also checked against the URDF.

The focused geometry/planner/tracker/configuration/logger suite passed **137 tests**.
A broader scan of the concurrently edited planning suite returned **324 passed,
3 skipped and 13 failures**, all in `test_timing_publication_invariants.py`; those
publication/epoch/transaction failures are outside this footprint implementation.
The earlier global-route installation failures no longer appeared in that scan.
This does not certify the entire workspace. Syntax and diff-whitespace checks pass.

This is a source change for future runs, not a reclassification of historical results.
Manifests identify `planner_collision_model=oriented_rectangle_swept_v1`. Existing
logger geometry columns remain the legacy circular diagnostic and are explicitly
identified by `legacy_geometry_diagnostic_model=circle`; simulator contacts remain
the physical collision evidence. Do not interpret old radius-based diagnostic columns
as measurements from the new rectangular checker.

No live navigation or stopping test was run for this change. The checker covers only
the supplied static planar geometry and modeled control paths. Missing physical props,
actuation noise, pose uncertainty, delayed commands, braking and three-dimensional
contacts are not made safe by this model. The missing-prop/contact issues in audit 14
remain separate. Runtime command and global-route installation validity also require
their own checks.
