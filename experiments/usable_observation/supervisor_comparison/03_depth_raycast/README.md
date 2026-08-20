# Depth and raycast geometry

## What this method asks

How much does explicit sensed 3-D structure improve the cold-start reliability field? A
commissioning depth observation is back-projected into world coordinates, rasterized as a
2.5-D height map, and raycast from the camera to each candidate target position.

`depth image → point cloud → height map → line-of-sight clearance → p_use map`

## Begin state

Known: camera calibration, target height, one declared depth source and its capture age. No
route-specific detector results are needed. The primary visual should pair the RGB/Gazebo
view with its depth image and resulting top-down height map so it is clear how a front rack
surface generates a blind volume behind it.

The final operational rung is still to be selected. The clearest primary candidate is a
sensor-realistic commissioning scan; perfect depth is a mechanism check, not the default.

## Map used in planning

For each candidate point, the raycaster tests whether sensed structure rises above the line
from camera to target. Clear rays retain the FOV/range score; blocked rays create sharp dark
shadows. Unknown depth is explicitly hatched and invokes the selected fallback:

- unavailable;
- FOV/range fallback; or
- conservative low reliability.

The supervisor panel should show the height map and resulting reliability map side by side,
with two example rays drawn in both views.

## Updates

For a commissioned static map, ordinary detector hits/misses do not alter the depth field.
A rescan replaces or merges sensed geometry and resets its age. Live depth would be a
different operational variant and must not be silently mixed into the commissioned-map
story.

The update panel should show `initial scan → stored map ages → layout changes → stale false
clearance → rescan restores shadow`, plus the missing-cell fallback.

## Expected plans

- R1: avoids the short rack shadow if the detour provides meaningfully better predicted
  belief.
- R2: separates equal-length north/south routes when only one is occluded.
- R3: identifies geometrically visible handover cameras, subject to missing depth.
- R6: should not detour where all rays remain clear.

## Failure mode to make visible

Stale, missing or misregistered heights can be unsafe: an unknown or moved rack can produce a
false bright corridor. The failure figure is part of the method explanation, not an optional
appendix.
