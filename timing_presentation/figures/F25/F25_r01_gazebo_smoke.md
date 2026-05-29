# F25 - R01 Gazebo Smoke Diagnostic

Source C1 run: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f24_r01_gazebo_smoke_v2/F24_R01_a4_lower_to_a3_mid/C1/seed1/experiment_20260528_201658`
Source C2 run: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f24_r01_gazebo_smoke_v2/F24_R01_a4_lower_to_a3_mid/C2/seed1/experiment_20260528_201931`

Both C1 and C2 completed infrastructure-valid runs but ended by geometry collision.

| condition | outcome | path m | min goal m | mean truth-state err m | min obstacle margin m | mean solve ms | mean p_vis_plan |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | collision | 3.57 | 2.62 | 0.286 | -0.046 | 1897 | 1.00 |
| C2 | collision | 3.20 | 3.20 | 0.271 | -0.037 | 2164 | 1.00 |

Interpretation: this smoke test is not yet a visibility tradeoff result. During the local closed-loop phase, both conditions report high planner-facing visibility (`p_vis_plan ~= 1`). The failure is currently a tracking/traversability-margin problem: the local planner cuts close enough to the forbidden layer that command/encoder noise and belief error create a small geometry penetration.

Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F25/F25_r01_gazebo_smoke.png`
PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F25/F25_r01_gazebo_smoke.pdf`
