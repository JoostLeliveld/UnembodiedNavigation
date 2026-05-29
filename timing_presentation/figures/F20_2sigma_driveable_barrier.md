# F20 2-Sigma Driveable-Region Barrier

- figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F20_2sigma_driveable_barrier.png`
- samples: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F20_2sigma_driveable_barrier.csv`
- source run: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/initial_rollout_diagnostics/aws_pick_east_2sigma_barrier_posterior_v1/`

## What Changed

The known 2D driveable/forbidden-zone layer is now treated as an objective-level
driveable-region barrier, rather than only a soft no-go preference.

The planner-facing clearance is:

```text
c_t = d_driveable(mean_xy_t) - r_clearance - 2 sigma_max(S_xy,t)
```

where `d_driveable` is positive inside the known driveable floor. The no-go term
uses a log barrier on this clearance, and invalid rollouts are still guarded
internally as execution hygiene.

Crucially, the covariance used for the tube is the expected posterior covariance
after the planner-facing camera update. This keeps the rule aligned with the
visibility-aware mechanism: visible routes can keep a narrower belief tube,
while low-visibility routes carry a larger tube near forbidden regions.

## Interpretation

This is the cleaner scientific framing:

- non-driveable floor is not a negotiable visibility tradeoff;
- visibility affects the route through planner-facing covariance;
- a route is attractive only if the predicted belief tube remains inside the
  known driveable floor.

In the current diagnostic, the H80 AWS A3-detour seed is feasible and reaches
near the goal. Cold starts still safe-stop, and some long-horizon seeded runs
fail because the GP query leaves the artifact support. That means the next
debugging target is the long-horizon/GP-support boundary, not the core
driveable-region barrier.

