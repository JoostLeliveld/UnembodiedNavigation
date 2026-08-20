# FOV and range geometry

## What this method asks

What does calibrated camera geometry provide before any driving or scene sensing? The field
uses camera intrinsics/extrinsics, image boundaries, projection geometry and range. It knows
where a target should project into the image, but it does not know whether a rack blocks that
ray.

## Begin state

Known: calibrated camera pose, intrinsics, image size and target height. No detector samples
or depth map are required. The begin-state figure should project the driveable floor into one
camera and mark in-frustum versus out-of-frustum cells.

## Map used in planning

The planner receives a smooth analytic or calibrated score inside the frustum and the
declared fallback outside it. Range/obliquity can lower reliability toward the footprint
edge. Crucially, the map remains bright through tall racks; overlaying the physical rack and
its true shadow makes the missing information obvious.

## Updates

The field is static while calibration is valid. It is recomputed after a camera calibration
change, not after detector hits or misses. A calibration-age label belongs on the figure.

## Expected plans

- R1: can detour away from footprint edges but may route through an internal rack shadow.
- R2: treats equal-length mirror routes similarly if their frustum/range profiles match.
- R3: understands overlap geometrically but not which overlapping view is occluded.
- R6: should preserve the normal short route.

## Key comparison

The direct comparison to depth/raycast isolates the value of sensed 3-D occlusion because
both methods share the same calibration and projection terms. Any difference between their
maps should be concentrated in line-of-sight shadows and unknown-depth cells.
