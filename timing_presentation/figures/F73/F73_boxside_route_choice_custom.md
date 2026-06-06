# F73 Boxside Route-Choice Probe

Log root: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1`

Config: `scripts/visibility_comparison/aws_probe_boxside_north_route_choice_config.yaml`

Change: post-reboot CUDA restored, YOLO forced on `device=0`, start moved beside A4 box pads and facing north.

All reported means below are computed after the first non-trivial command, i.e.,
while the robot is actually driving. Pre-command global-solve / warm-up rows are
excluded.

- C1 constant-R: `goal_reached`, path=4.84 m, min_goal=0.077 m, mean truth-state error=1.169 m, mean truth-belief error=0.308 m, y-range while driving=[-1.52, 1.73].

- C2 GP-aware: `goal_reached`, path=5.84 m, min_goal=0.131 m, mean truth-state error=0.336 m, mean truth-belief error=0.107 m, y-range while driving=[-2.06, 1.75].


Interpretation: C1 reaches by the shorter/northern route but has substantially larger external-camera state error while driving. C2 reaches by a longer lower/front-camera sweep and keeps both state and planner-belief error lower. This is a promising single-seed diagnostic, not paper evidence yet.

Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F73/F73_boxside_route_choice_custom.png`
