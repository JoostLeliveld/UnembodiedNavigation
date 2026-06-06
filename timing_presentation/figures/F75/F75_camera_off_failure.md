# F75 Camera-Off Failure Comparison

This figure compares F73 C1/C2 with Run A, where C2 used the same route-choice setup but `use_pixel_correction:=False`.

## Files

- Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F75/F75_camera_off_failure.png`
- PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F75/F75_camera_off_failure.pdf`

## Runtime-Only Summary

### F73 C1: direct route, correction ON
- Run: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1/probe_a4_boxside_north_to_a3top/C1/seed0/experiment_20260604_144614`
- Outcome: `goal_reached`
- Path length: `4.835 m`
- Min goal distance: `0.077 m`
- Min obstacle distance: `0.136 m`
- Truth-belief error mean/median/max: `0.308` / `0.170` / `1.951 m`
- YOLO detection rate: `0.452`

### F73 C2: visible route, correction ON
- Run: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1/probe_a4_boxside_north_to_a3top/C2/seed0/experiment_20260604_144802`
- Outcome: `goal_reached`
- Path length: `5.837 m`
- Min goal distance: `0.131 m`
- Min obstacle distance: `0.190 m`
- Truth-belief error mean/median/max: `0.107` / `0.105` / `0.567 m`
- YOLO detection rate: `0.975`

### Run A: same C2 route, correction OFF
- Run: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/ablation_corrOFF/probe_a4_boxside_north_to_a3top/C2/seed0/experiment_20260604_155318`
- Outcome: `collision`
- Path length: `3.881 m`
- Min goal distance: `1.193 m`
- Min obstacle distance: `-0.032 m`
- Truth-belief error mean/median/max: `1.935` / `1.859` / `3.934 m`
- YOLO detection rate: `0.415`

## Interpretation

C1 is not equivalent to camera-off. C1 still has pixel correction enabled and gets enough camera updates before and after the weak-visibility segment to keep the planner belief bounded. Its direct route has a blackout, but not a complete removal of camera corrections.

Run A disables pixel correction entirely. The robot still has YOLO detections in the logs, but the planner belief cannot use them; it relies on noisy dead reckoning. Belief error grows to multiple metres and the truth trajectory collides before reaching the goal.

So the correct distinction is: C1 tests a camera-poor route with correction still available when detections exist; Run A tests no camera correction at all. The latter is much harsher and fails because drift accumulates globally, not only in one occluded segment.