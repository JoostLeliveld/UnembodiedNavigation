# Constant and distance baselines

## What this method asks

How far can we get without modelling field of view, occlusion or operational experience?
These are two distinct simple baselines presented together because neither knows the scene.

- **Constant:** `p_use(s) = p0` everywhere the method is queried.
- **Distance:** `p_use,c(s) = f(||s - camera_c||)`, with a declared monotone calibration.

## Begin state

Known: camera position for distance, and optionally a calibration-set prevalence. Unknown:
camera orientation/frustum, racks, line of sight, past detections and layout changes. The
begin-state figure must show that two equally distant cells receive the same prediction even
when one is behind a rack.

## Map used in planning

The constant field is uniform. The distance field forms smooth rings around each camera.
For multiple cameras, show both the four per-camera fields and the declared fused field. The
collision map remains visible underneath so the audience can see that obstacles affect
motion but do not cast reliability shadows in this method.

## Updates

There is no route-time reliability update. If `p0` or the distance curve is fitted, it changes
only during an explicit recommissioning step. Runtime observations update the robot belief,
not this reliability map.

## Expected plans

- R1: may prefer the short blind branch because both alternatives look similarly reliable.
- R2: predicts the mirror routes similarly when their range profiles match.
- R3: can prefer the nearest camera but cannot reason about rack-blocked handover.
- R6: should take the normal short route and acts as the no-spurious-detour control.

## Figures to produce

See [`figures/README.md`](figures/README.md). The central teaching figure is a three-column
top-down view: constant field, distance field, and the same field with two rack shadows that
the method fails to represent outlined in red.

## Interpretation boundary

This is not a deliberately weak straw man. It measures whether more complicated sources
earn their additional inputs and commissioning cost. A strong distance result would mean
occlusion-aware complexity is unnecessary in that tested regime.
