# F24 Visible-To-Visible Route Battery

- figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F24/F24_visible_route_battery.png`
- csv: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F24/F24_visible_route_battery.csv`

## Contract

- Offline only; this is not Gazebo evidence.
- Ten harder routes have GP conservative endpoint reliability above the visible threshold.
- Compared with the first F24 pass, short lower-lane-only tasks were replaced by longer diagonal/cross-aisle tasks.
- visible endpoint threshold: `0.6`
- C1 and C2 receive the same known driveable layer and same neutral lane-graph route seeds.
- Reduced-time global solve: `H=50`, `dt=0.40`, `maxiter=70`, preserving about 20 s lookahead.

## Summary

| route | C1 class | C1 d | C1 t | C2 class | C2 d | C2 t |
|---|---:|---:|---:|---:|---:|---:|
| R01 | lower_lane | 0.25 | 9.5s | lower_lane | 0.09 | 9.5s |
| R02 | lower_lane | 0.06 | 8.6s | lower_lane | 0.22 | 14.1s |
| R03 | lower_lane | 0.13 | 6.8s | lower_lane | 0.10 | 10.0s |
| R04 | lower_lane | 0.14 | 7.0s | lower_lane | 0.17 | 8.2s |
| R05 | lower_lane | 0.12 | 8.3s | lower_lane | 0.21 | 10.5s |
| R06 | lower_lane | 0.14 | 11.7s | lower_lane | 4.07 | 12.4s |
| R07 | lower_lane | 0.14 | 8.3s | lower_lane | 0.12 | 10.9s |
| R08 | lower_lane | 0.06 | 7.8s | lower_lane | 0.08 | 10.7s |
| R09 | lower_lane | 0.17 | 7.8s | lower_lane | 0.18 | 9.7s |
| R10 | lower_lane | 0.16 | 9.1s | lower_lane | 0.14 | 11.7s |

## Aggregate

- C1 reached/valid: `10/10`
- C2 reached/valid: `9/10`
- C1 median solve time: `8.29s`
- C2 median solve time: `10.62s`

Interpretation should focus on whether the reduced global solve remains stable across visible endpoints.
Route differences are diagnostics only until confirmed by Gazebo runs with command and encoder noise.
