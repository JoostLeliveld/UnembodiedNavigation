# Results — static four-camera probability planning

## What ran

- Four frozen per-camera detector-hit GP fields from 8,808 spawn-grid events.
- Four declared warehouse tasks and 400 matched stochastic-replay seeds per task and method.
- One selected camera per cell; no simultaneous fusion.
- Runtime replay always used a realised hit or no update. Only route planning differed.

## Selected-route results

| Task | Method | Length [m] | Mean planner p(use) | Exact expected terminal sigma [m] |
|---|---|---:|---:|---:|
| `mc_blind_L` | `availability_blind_shortest` | 15.05 | 0.771 | 0.0250 |
| `mc_blind_L` | `r_over_p_shortcut` | 15.18 | 0.943 | 0.0246 |
| `mc_blind_L` | `explicit_hit_miss` | 15.18 | 0.943 | 0.0246 |
| `mc_m2_w2e_traverse` | `availability_blind_shortest` | 22.20 | 0.691 | 0.1014 |
| `mc_m2_w2e_traverse` | `r_over_p_shortcut` | 22.20 | 0.691 | 0.1014 |
| `mc_m2_w2e_traverse` | `explicit_hit_miss` | 22.20 | 0.691 | 0.1014 |
| `full_traverse_handover` | `availability_blind_shortest` | 16.33 | 0.998 | 0.0238 |
| `full_traverse_handover` | `r_over_p_shortcut` | 16.33 | 0.998 | 0.0238 |
| `full_traverse_handover` | `explicit_hit_miss` | 16.33 | 0.998 | 0.0238 |
| `route_tall_shadow_west` | `availability_blind_shortest` | 14.31 | 0.863 | 0.0253 |
| `route_tall_shadow_west` | `r_over_p_shortcut` | 14.41 | 1.000 | 0.0243 |
| `route_tall_shadow_west` | `explicit_hit_miss` | 14.41 | 1.000 | 0.0243 |

## Model-based stochastic replay

Values below are means over runs. The experimental unit is one task/method/seed run.

| Task | Method | n | Belief RMSE [m] | Longest dropout [s] | Terminal sigma [m] | 95% coverage |
|---|---|---:|---:|---:|---:|---:|
| `mc_blind_L` | `availability_blind_shortest` | 400 | 0.0464 | 2.30 | 0.0250 | 0.950 |
| `mc_blind_L` | `r_over_p_shortcut` | 400 | 0.0374 | 0.87 | 0.0246 | 0.951 |
| `mc_blind_L` | `explicit_hit_miss` | 400 | 0.0374 | 0.87 | 0.0246 | 0.951 |
| `mc_m2_w2e_traverse` | `availability_blind_shortest` | 400 | 0.0551 | 3.48 | 0.0997 | 0.951 |
| `mc_m2_w2e_traverse` | `r_over_p_shortcut` | 400 | 0.0551 | 3.48 | 0.0997 | 0.951 |
| `mc_m2_w2e_traverse` | `explicit_hit_miss` | 400 | 0.0551 | 3.48 | 0.0997 | 0.951 |
| `full_traverse_handover` | `availability_blind_shortest` | 400 | 0.0356 | 0.04 | 0.0238 | 0.950 |
| `full_traverse_handover` | `r_over_p_shortcut` | 400 | 0.0356 | 0.04 | 0.0238 | 0.950 |
| `full_traverse_handover` | `explicit_hit_miss` | 400 | 0.0356 | 0.04 | 0.0238 | 0.950 |
| `route_tall_shadow_west` | `availability_blind_shortest` | 400 | 0.0437 | 2.60 | 0.0253 | 0.952 |
| `route_tall_shadow_west` | `r_over_p_shortcut` | 400 | 0.0356 | 0.00 | 0.0243 | 0.955 |
| `route_tall_shadow_west` | `explicit_hit_miss` | 400 | 0.0356 | 0.00 | 0.0243 | 0.955 |

## Conditional measurement covariance provenance

`R_cond` is a planning input, not a probability. It was constructed from current
camera-measurement residual component SDs versus commanded ground truth, zero-parameter
floor IPM, balanced set-pose dataset `PG-IPM-CURRENT`. Ground truth was offline only.

| Camera | detections | radial SD [m] | lateral SD [m] | isotropic conditional sigma [m] |
|---|---:|---:|---:|---:|
| camera_A | 455 | 0.0404 | 0.0521 | 0.0466 |
| camera_B | 476 | 0.0430 | 0.0532 | 0.0484 |
| camera_C | 449 | 0.0412 | 0.0508 | 0.0462 |
| camera_D | 464 | 0.0397 | 0.0505 | 0.0454 |

## Interpretation boundary

This experiment establishes route discrimination and the model-level consequence of
using an explicit Bernoulli observation model. It does **not** establish held-out
probability calibration, a closed-loop Gazebo navigation advantage, a real-robot result,
or a simultaneous four-camera fusion result. Those require separate registered campaigns.
