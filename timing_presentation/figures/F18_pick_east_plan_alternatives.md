# F18 Pick-East Plan Alternatives

Task: `B1_apron_a4_to_uppermid_a3`

- start: `(3.20, -1.00)`, yaw `0.0` (facing east toward R5 shelf)
- goal: `(1.00, 1.75)`
- condition shown: C2 visibility-aware EFE
- config: `goal_prior_final=8`, `ambiguity_weight=8`, `v_max=1.0`, keep-in no-go layer

Files:

- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F18_pick_east_plan_alternatives.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F18_pick_east_plan_alternatives.pdf`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F18_pick_east_plan_alternatives.csv`

Key result:

- H80 best: `aws_a3_detour_seed` with J=2101.4, terminal distance=0.28 m, mean p_vis=0.57.
- H120 best: `direct_goal_seed` with J=2341.4, terminal distance=0.08 m, mean p_vis=0.55.

Interpretation: H80 exposes the A3-detour basin, but the original/cold start does not find it. At H120 the direct-goal basin wins. This supports investigating condition-neutral long-first-solve / multistart behavior, not route-forcing waypoints.
