# F76 - Camera-off exact failure

Figure files:

- `timing_presentation/figures/F76/F76_camera_off_exact_failure.png`
- `timing_presentation/figures/F76/F76_camera_off_exact_failure.pdf`

## What fails

The camera-off run does not fail because the initial route plot is uninterpretable. It fails because the execution pipeline becomes internally inconsistent:

1. `use_pixel_correction=False` disables planner belief correction.
2. YOLO and `/state/bev` still exist in the log, but after missed detections `/state/bev` becomes stale near the early route segment.
3. The local tracker continues to compute waypoint distance against that stale `/state`, not against the physical truth pose.
4. The tracker therefore believes it is about `0.26 m` from waypoint 2 for several seconds while the real robot is more than `2 m` away from that waypoint.
5. It keeps commanding forward motion and the truth trajectory collides.

## Numbers

Successful C2 correction ON:

- outcome: `goal_reached`
- path: `5.837 m`
- min goal distance: `0.131 m`
- min obstacle distance: `0.190 m`

C2 camera OFF:

- outcome: `collision`
- path: `3.881 m`
- min goal distance: `1.193 m`
- min obstacle distance: `-0.032 m`
- mean truth-belief error after first command: `1.935 m`

## Interpretation

This is not the same as C1 passing through a low-visibility segment. C1 still has camera correction enabled and can reacquire. Camera-off removes the global `(x,y)` correction completely, while stale `/state` remains available to the local tracker. The result is a false sense of waypoint progress: the controller thinks it is tracking the route, but the physical robot has drifted far away from the state used for control.

This diagnostic should be treated as an ablation/failure-mode figure, not paper evidence for the final C1/C2 comparison.
