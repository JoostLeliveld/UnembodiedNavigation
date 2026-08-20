# Complete-CAD raycast reference

## Role in the comparison

This is the complete geometric map available from the Gazebo/SDF world: all relevant
occluder footprints and exact heights. It answers how well line-of-sight geometry could work
with complete, perfectly registered structure. It is an evaluation reference, not a
deployable sensor arm.

## Begin state

The complete map is available immediately because it is read from simulator truth. The
begin-state panel must carry a visible `EVALUATION-ONLY ORACLE/REFERENCE` banner and contrast
its inputs with the depth method’s sensed surfaces.

## Map used in planning

For diagnostic/reference planning, exact prisms are raycast to produce a full static
occlusion field with no unknown cells. It may be used to interpret false-clear and
false-occluded regions in operational methods. It cannot be allowed to supply hidden
geometry to an arm labelled “sensed depth.”

## Updates

There is no operational update or staleness clock: the world file is treated as exact. That
is precisely why its commissioning and maintenance costs are not comparable to deployed
methods. If a changed-layout SDF is loaded, that is new oracle truth, not a rescan.

## Expected plans

The reference plan shows what the common planner does under complete static geometry. It is
useful for explaining R1/R2 route opportunities and diagnosing other fields, but it cannot
“win” deployment.

## Important nuance

Complete CAD is geometrically complete but need not best match the detector. A viewpoint-
matched sensed surface can encode what that camera actually sees more faithfully. Present
CAD as a geometric reference, not automatically as an empirical upper bound on `p_use`.
