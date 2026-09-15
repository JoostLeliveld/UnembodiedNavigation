# Corrected global route objective -- offline validation

Generated: 2026-09-15T22:37:17

Objective under test:

```
minimize   travel_time
         + localization_weight * INTEGRAL c_loc(q(t), R(t)) dt
subject to collision-free, inside the driveable region,
           the complete body fits (straights, corners, in-place turns),
           kinematically feasible, reaches the goal

c_loc = max( 0.5*log( det R_plan(q,R) / det R_ref ), 0 )   [nats, >= 0]
```

Declared localization weight: 1.0 s per nat-second (not fitted; see breakeven.csv for the sensitivity of every decision).
Reference measurement quality R_ref: r = 2.5 px.

## Candidate tables (corrected objective)

### footprint = spec_0.80x0.55, gate set = full

| task | condition | route | safety | len [m] | T [s] | min body clr [m] | q mean | q min | loc cost [s] | total | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| occlusion_transit_a4 | q=unity/R=constant_global | below_main_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.28 | 18.7 | -0.100 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=constant_global | above_connector | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.20 | 15.4 | -0.100 | 1.000 | 1.000 | 0.00 | 15.40 | no |
| occlusion_transit_a4 | q=unity/R=commissioned | below_main_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.28 | 18.7 | -0.100 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=commissioned | above_connector | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.20 | 15.4 | -0.100 | 1.000 | 1.000 | 0.00 | 15.40 | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | below_main_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.28 | 18.7 | -0.100 | 0.574 | 0.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | above_connector | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.20 | 15.4 | -0.100 | 0.236 | 0.000 | 0.00 | 15.40 | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | below_main_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.28 | 18.7 | -0.100 | 0.574 | 0.000 | 32.80 | 51.50 | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | above_connector | UNSAFE:body_obstacle_clearance,body_inside_driveable | 7.20 | 15.4 | -0.100 | 0.236 | 0.000 | 58.75 | 74.15 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 7.93 | 18.1 | -0.110 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | above_connector | UNSAFE:body_inside_driveable | 5.05 | 13.4 | -0.110 | 1.000 | 1.000 | 0.00 | 13.40 | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 7.93 | 18.1 | -0.110 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | above_connector | UNSAFE:body_inside_driveable | 5.05 | 13.4 | -0.110 | 1.000 | 1.000 | 0.00 | 13.40 | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 7.93 | 18.1 | -0.110 | 0.958 | 0.670 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | above_connector | UNSAFE:body_inside_driveable | 5.05 | 13.4 | -0.110 | 0.420 | 0.000 | 0.00 | 13.40 | no |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 7.93 | 18.1 | -0.110 | 0.958 | 0.670 | 0.86 | 18.96 | no |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | above_connector | UNSAFE:body_inside_driveable | 5.05 | 13.4 | -0.110 | 0.420 | 0.000 | 37.02 | 50.42 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 9.93 | 21.4 | -0.110 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | above_connector | UNSAFE:body_inside_driveable | 7.05 | 16.7 | -0.110 | 1.000 | 1.000 | 0.00 | 16.70 | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 9.93 | 21.4 | -0.110 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | above_connector | UNSAFE:body_inside_driveable | 7.05 | 16.7 | -0.110 | 1.000 | 1.000 | 0.00 | 16.70 | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 9.93 | 21.4 | -0.110 | 0.964 | 0.665 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | above_connector | UNSAFE:body_inside_driveable | 7.05 | 16.7 | -0.110 | 0.423 | 0.000 | 0.00 | 16.70 | no |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 9.93 | 21.4 | -0.110 | 0.964 | 0.665 | 0.88 | 22.28 | no |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | above_connector | UNSAFE:body_inside_driveable | 7.05 | 16.7 | -0.110 | 0.423 | 0.000 | 40.19 | 56.89 | no |
| route_west_to_a1_upper | q=unity/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 9.88 | 21.4 | -0.085 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=unity/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 8.65 | 19.3 | -0.160 | 1.000 | 1.000 | 0.00 | 19.30 | no |
| route_west_to_a1_upper | q=unity/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 9.88 | 21.4 | -0.085 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=unity/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 8.65 | 19.3 | -0.160 | 1.000 | 1.000 | 0.00 | 19.30 | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 9.88 | 21.4 | -0.085 | 0.774 | 0.028 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 8.65 | 19.3 | -0.160 | 0.272 | 0.000 | 0.00 | 19.30 | no |
| route_west_to_a1_upper | q=commissioned/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 9.88 | 21.4 | -0.085 | 0.774 | 0.028 | 10.26 | 31.66 | no |
| route_west_to_a1_upper | q=commissioned/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 8.65 | 19.3 | -0.160 | 0.272 | 0.000 | 55.93 | 75.23 | no |
| control_west_to_a1_low | q=unity/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 5.63 | 14.4 | -0.085 | 1.000 | 1.000 | 0.00 | 14.40 | no |
| control_west_to_a1_low | q=unity/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 12.90 | 26.4 | -0.160 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=unity/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 5.63 | 14.4 | -0.085 | 1.000 | 1.000 | 0.00 | 14.40 | no |
| control_west_to_a1_low | q=unity/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 12.90 | 26.4 | -0.160 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 5.63 | 14.4 | -0.085 | 0.975 | 0.833 | 0.00 | 14.40 | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 12.90 | 26.4 | -0.160 | 0.293 | 0.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 5.63 | 14.4 | -0.085 | 0.975 | 0.833 | 0.39 | 14.79 | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance,body_inside_driveable | 12.90 | 26.4 | -0.160 | 0.293 | 0.000 | 67.07 | 93.47 | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 9.12 | 20.1 | -0.185 | 1.000 | 1.000 | 0.00 | 20.10 | no |
| sanity_visible_apron_crossing | q=unity/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 9.12 | 20.1 | -0.185 | 1.000 | 1.000 | 0.00 | 20.10 | no |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | below_main_aisle | UNSAFE:body_inside_driveable | 9.12 | 20.1 | -0.185 | 0.977 | 0.757 | 0.00 | 20.10 | no |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | below_main_aisle | UNSAFE:body_inside_driveable | 9.12 | 20.1 | -0.185 | 0.977 | 0.757 | 0.50 | 20.60 | no |

### footprint = deployed_burger, gate set = full

| task | condition | route | safety | len [m] | T [s] | min body clr [m] | q mean | q min | loc cost [s] | total | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| occlusion_transit_a4 | q=unity/R=constant_global | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=constant_global | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 1.000 | 1.000 | 0.00 | 15.40 | yes |
| occlusion_transit_a4 | q=unity/R=commissioned | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=commissioned | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 1.000 | 1.000 | 0.00 | 15.40 | yes |
| occlusion_transit_a4 | q=commissioned/R=constant_global | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 0.574 | 0.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 0.236 | 0.000 | 0.00 | 15.40 | yes |
| occlusion_transit_a4 | q=commissioned/R=commissioned | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 0.574 | 0.000 | 32.80 | 51.50 | yes |
| occlusion_transit_a4 | q=commissioned/R=commissioned | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 0.236 | 0.000 | 58.75 | 74.15 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 1.000 | 1.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=unity/R=commissioned | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 1.000 | 1.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 0.958 | 0.670 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 0.420 | 0.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 0.958 | 0.670 | 0.86 | 18.96 | yes |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 0.420 | 0.000 | 37.02 | 50.42 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 1.000 | 1.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=unity/R=commissioned | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 1.000 | 1.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 0.964 | 0.665 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 0.423 | 0.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 0.964 | 0.665 | 0.88 | 22.28 | yes |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 0.423 | 0.000 | 40.19 | 56.89 | no |
| route_west_to_a1_upper | q=unity/R=constant_global | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=unity/R=constant_global | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 1.000 | 1.000 | 0.00 | 19.30 | yes |
| route_west_to_a1_upper | q=unity/R=commissioned | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=unity/R=commissioned | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 1.000 | 1.000 | 0.00 | 19.30 | yes |
| route_west_to_a1_upper | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 0.774 | 0.028 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 0.272 | 0.000 | 0.00 | 19.30 | yes |
| route_west_to_a1_upper | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 0.774 | 0.028 | 10.26 | 31.66 | yes |
| route_west_to_a1_upper | q=commissioned/R=commissioned | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 0.272 | 0.000 | 55.93 | 75.23 | no |
| control_west_to_a1_low | q=unity/R=constant_global | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 1.000 | 1.000 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=unity/R=constant_global | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=unity/R=commissioned | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 1.000 | 1.000 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=unity/R=commissioned | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 0.975 | 0.833 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=commissioned/R=constant_global | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 0.293 | 0.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 0.975 | 0.833 | 0.39 | 14.79 | yes |
| control_west_to_a1_low | q=commissioned/R=commissioned | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 0.293 | 0.000 | 67.07 | 93.47 | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 1.000 | 1.000 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=unity/R=commissioned | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 1.000 | 1.000 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 0.977 | 0.757 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 0.977 | 0.757 | 0.50 | 20.60 | yes |

### footprint = spec_0.80x0.55, gate set = deployment

| task | condition | route | safety | len [m] | T [s] | min body clr [m] | q mean | q min | loc cost [s] | total | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| occlusion_transit_a4 | q=unity/R=constant_global | below_main_aisle | UNSAFE:body_obstacle_clearance | 7.28 | 18.7 | -0.100 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=constant_global | above_connector | UNSAFE:body_obstacle_clearance | 7.20 | 15.4 | -0.100 | 1.000 | 1.000 | 0.00 | 15.40 | no |
| occlusion_transit_a4 | q=unity/R=commissioned | below_main_aisle | UNSAFE:body_obstacle_clearance | 7.28 | 18.7 | -0.100 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=commissioned | above_connector | UNSAFE:body_obstacle_clearance | 7.20 | 15.4 | -0.100 | 1.000 | 1.000 | 0.00 | 15.40 | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | below_main_aisle | UNSAFE:body_obstacle_clearance | 7.28 | 18.7 | -0.100 | 0.574 | 0.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | above_connector | UNSAFE:body_obstacle_clearance | 7.20 | 15.4 | -0.100 | 0.236 | 0.000 | 0.00 | 15.40 | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | below_main_aisle | UNSAFE:body_obstacle_clearance | 7.28 | 18.7 | -0.100 | 0.574 | 0.000 | 32.80 | 51.50 | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | above_connector | UNSAFE:body_obstacle_clearance | 7.20 | 15.4 | -0.100 | 0.236 | 0.000 | 58.75 | 74.15 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | below_main_aisle | SAFE | 7.93 | 18.1 | -0.110 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | above_connector | SAFE | 5.05 | 13.4 | -0.110 | 1.000 | 1.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=unity/R=commissioned | below_main_aisle | SAFE | 7.93 | 18.1 | -0.110 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | above_connector | SAFE | 5.05 | 13.4 | -0.110 | 1.000 | 1.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | below_main_aisle | SAFE | 7.93 | 18.1 | -0.110 | 0.958 | 0.670 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | above_connector | SAFE | 5.05 | 13.4 | -0.110 | 0.420 | 0.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | below_main_aisle | SAFE | 7.93 | 18.1 | -0.110 | 0.958 | 0.670 | 0.86 | 18.96 | yes |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | above_connector | SAFE | 5.05 | 13.4 | -0.110 | 0.420 | 0.000 | 37.02 | 50.42 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | below_main_aisle | SAFE | 9.93 | 21.4 | -0.110 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | above_connector | SAFE | 7.05 | 16.7 | -0.110 | 1.000 | 1.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=unity/R=commissioned | below_main_aisle | SAFE | 9.93 | 21.4 | -0.110 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | above_connector | SAFE | 7.05 | 16.7 | -0.110 | 1.000 | 1.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.93 | 21.4 | -0.110 | 0.964 | 0.665 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | above_connector | SAFE | 7.05 | 16.7 | -0.110 | 0.423 | 0.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.93 | 21.4 | -0.110 | 0.964 | 0.665 | 0.88 | 22.28 | yes |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | above_connector | SAFE | 7.05 | 16.7 | -0.110 | 0.423 | 0.000 | 40.19 | 56.89 | no |
| route_west_to_a1_upper | q=unity/R=constant_global | below_main_aisle | SAFE | 9.88 | 21.4 | -0.085 | 1.000 | 1.000 | 0.00 | 21.40 | yes |
| route_west_to_a1_upper | q=unity/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance | 8.65 | 19.3 | -0.160 | 1.000 | 1.000 | 0.00 | 19.30 | no |
| route_west_to_a1_upper | q=unity/R=commissioned | below_main_aisle | SAFE | 9.88 | 21.4 | -0.085 | 1.000 | 1.000 | 0.00 | 21.40 | yes |
| route_west_to_a1_upper | q=unity/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance | 8.65 | 19.3 | -0.160 | 1.000 | 1.000 | 0.00 | 19.30 | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.88 | 21.4 | -0.085 | 0.774 | 0.028 | 0.00 | 21.40 | yes |
| route_west_to_a1_upper | q=commissioned/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance | 8.65 | 19.3 | -0.160 | 0.272 | 0.000 | 0.00 | 19.30 | no |
| route_west_to_a1_upper | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.88 | 21.4 | -0.085 | 0.774 | 0.028 | 10.26 | 31.66 | yes |
| route_west_to_a1_upper | q=commissioned/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance | 8.65 | 19.3 | -0.160 | 0.272 | 0.000 | 55.93 | 75.23 | no |
| control_west_to_a1_low | q=unity/R=constant_global | below_main_aisle | SAFE | 5.63 | 14.4 | -0.085 | 1.000 | 1.000 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=unity/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance | 12.90 | 26.4 | -0.160 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=unity/R=commissioned | below_main_aisle | SAFE | 5.63 | 14.4 | -0.085 | 1.000 | 1.000 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=unity/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance | 12.90 | 26.4 | -0.160 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | below_main_aisle | SAFE | 5.63 | 14.4 | -0.085 | 0.975 | 0.833 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=commissioned/R=constant_global | above_cross_aisle | UNSAFE:body_obstacle_clearance | 12.90 | 26.4 | -0.160 | 0.293 | 0.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | below_main_aisle | SAFE | 5.63 | 14.4 | -0.085 | 0.975 | 0.833 | 0.39 | 14.79 | yes |
| control_west_to_a1_low | q=commissioned/R=commissioned | above_cross_aisle | UNSAFE:body_obstacle_clearance | 12.90 | 26.4 | -0.160 | 0.293 | 0.000 | 67.07 | 93.47 | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | below_main_aisle | SAFE | 9.12 | 20.1 | -0.185 | 1.000 | 1.000 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=unity/R=commissioned | below_main_aisle | SAFE | 9.12 | 20.1 | -0.185 | 1.000 | 1.000 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.12 | 20.1 | -0.185 | 0.977 | 0.757 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.12 | 20.1 | -0.185 | 0.977 | 0.757 | 0.50 | 20.60 | yes |

### footprint = deployed_burger, gate set = deployment

| task | condition | route | safety | len [m] | T [s] | min body clr [m] | q mean | q min | loc cost [s] | total | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| occlusion_transit_a4 | q=unity/R=constant_global | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=constant_global | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 1.000 | 1.000 | 0.00 | 15.40 | yes |
| occlusion_transit_a4 | q=unity/R=commissioned | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 1.000 | 1.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=unity/R=commissioned | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 1.000 | 1.000 | 0.00 | 15.40 | yes |
| occlusion_transit_a4 | q=commissioned/R=constant_global | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 0.574 | 0.000 | 0.00 | 18.70 | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 0.236 | 0.000 | 0.00 | 15.40 | yes |
| occlusion_transit_a4 | q=commissioned/R=commissioned | below_main_aisle | SAFE | 7.28 | 18.7 | +0.230 | 0.574 | 0.000 | 32.80 | 51.50 | yes |
| occlusion_transit_a4 | q=commissioned/R=commissioned | above_connector | SAFE | 7.20 | 15.4 | +0.230 | 0.236 | 0.000 | 58.75 | 74.15 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 1.000 | 1.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=unity/R=commissioned | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 1.000 | 1.000 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 1.000 | 1.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 0.958 | 0.670 | 0.00 | 18.10 | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 0.420 | 0.000 | 0.00 | 13.40 | yes |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | below_main_aisle | SAFE | 7.93 | 18.1 | +0.262 | 0.958 | 0.670 | 0.86 | 18.96 | yes |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | above_connector | SAFE | 5.05 | 13.4 | +0.262 | 0.420 | 0.000 | 37.02 | 50.42 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 1.000 | 1.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=unity/R=commissioned | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 1.000 | 1.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 0.964 | 0.665 | 0.00 | 21.40 | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 0.423 | 0.000 | 0.00 | 16.70 | yes |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.93 | 21.4 | +0.262 | 0.964 | 0.665 | 0.88 | 22.28 | yes |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | above_connector | SAFE | 7.05 | 16.7 | +0.262 | 0.423 | 0.000 | 40.19 | 56.89 | no |
| route_west_to_a1_upper | q=unity/R=constant_global | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=unity/R=constant_global | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 1.000 | 1.000 | 0.00 | 19.30 | yes |
| route_west_to_a1_upper | q=unity/R=commissioned | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 1.000 | 1.000 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=unity/R=commissioned | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 1.000 | 1.000 | 0.00 | 19.30 | yes |
| route_west_to_a1_upper | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 0.774 | 0.028 | 0.00 | 21.40 | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 0.272 | 0.000 | 0.00 | 19.30 | yes |
| route_west_to_a1_upper | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.88 | 21.4 | +0.287 | 0.774 | 0.028 | 10.26 | 31.66 | yes |
| route_west_to_a1_upper | q=commissioned/R=commissioned | above_cross_aisle | SAFE | 8.65 | 19.3 | +0.212 | 0.272 | 0.000 | 55.93 | 75.23 | no |
| control_west_to_a1_low | q=unity/R=constant_global | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 1.000 | 1.000 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=unity/R=constant_global | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=unity/R=commissioned | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 1.000 | 1.000 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=unity/R=commissioned | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 1.000 | 1.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 0.975 | 0.833 | 0.00 | 14.40 | yes |
| control_west_to_a1_low | q=commissioned/R=constant_global | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 0.293 | 0.000 | 0.00 | 26.40 | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | below_main_aisle | SAFE | 5.63 | 14.4 | +0.287 | 0.975 | 0.833 | 0.39 | 14.79 | yes |
| control_west_to_a1_low | q=commissioned/R=commissioned | above_cross_aisle | SAFE | 12.90 | 26.4 | +0.212 | 0.293 | 0.000 | 67.07 | 93.47 | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 1.000 | 1.000 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=unity/R=commissioned | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 1.000 | 1.000 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 0.977 | 0.757 | 0.00 | 20.10 | yes |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | below_main_aisle | SAFE | 9.12 | 20.1 | +0.187 | 0.977 | 0.757 | 0.50 | 20.60 | yes |

## Old vs corrected selection

| task | condition | footprint | gate set | legacy | corrected | changed |
|---|---|---|---|---|---|---|
| control_west_to_a1_low | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=commissioned/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=commissioned/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| control_west_to_a1_low | q=unity/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=unity/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=unity/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=unity/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| control_west_to_a1_low | q=unity/R=constant_global | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=unity/R=constant_global | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=unity/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| control_west_to_a1_low | q=unity/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | <none> | <none> | no |
| occlusion_transit_a4 | q=commissioned/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | deployed_burger | deployment | above_connector | above_connector | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | deployed_burger | full | above_connector | above_connector | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | <none> | <none> | no |
| occlusion_transit_a4 | q=commissioned/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| occlusion_transit_a4 | q=unity/R=commissioned | deployed_burger | deployment | above_connector | above_connector | no |
| occlusion_transit_a4 | q=unity/R=commissioned | deployed_burger | full | above_connector | above_connector | no |
| occlusion_transit_a4 | q=unity/R=commissioned | spec_0.80x0.55 | deployment | <none> | <none> | no |
| occlusion_transit_a4 | q=unity/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| occlusion_transit_a4 | q=unity/R=constant_global | deployed_burger | deployment | above_connector | above_connector | no |
| occlusion_transit_a4 | q=unity/R=constant_global | deployed_burger | full | above_connector | above_connector | no |
| occlusion_transit_a4 | q=unity/R=constant_global | spec_0.80x0.55 | deployment | <none> | <none> | no |
| occlusion_transit_a4 | q=unity/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | deployed_burger | deployment | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | deployed_burger | full | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | deployed_burger | deployment | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | deployed_burger | full | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | spec_0.80x0.55 | deployment | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=unity/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | deployed_burger | deployment | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | deployed_burger | full | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | spec_0.80x0.55 | deployment | above_connector | above_connector | no |
| route_apron_to_a2_mid | q=unity/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | deployed_burger | deployment | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | deployed_burger | full | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | deployed_burger | deployment | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | deployed_burger | full | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | spec_0.80x0.55 | deployment | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=unity/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | deployed_burger | deployment | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | deployed_burger | full | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | spec_0.80x0.55 | deployment | above_connector | above_connector | no |
| route_apron_to_a3_mid | q=unity/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| route_west_to_a1_upper | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| route_west_to_a1_upper | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| route_west_to_a1_upper | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| route_west_to_a1_upper | q=commissioned/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | deployed_burger | deployment | above_cross_aisle | above_cross_aisle | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | deployed_burger | full | above_cross_aisle | above_cross_aisle | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| route_west_to_a1_upper | q=commissioned/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| route_west_to_a1_upper | q=unity/R=commissioned | deployed_burger | deployment | above_cross_aisle | above_cross_aisle | no |
| route_west_to_a1_upper | q=unity/R=commissioned | deployed_burger | full | above_cross_aisle | above_cross_aisle | no |
| route_west_to_a1_upper | q=unity/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| route_west_to_a1_upper | q=unity/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| route_west_to_a1_upper | q=unity/R=constant_global | deployed_burger | deployment | above_cross_aisle | above_cross_aisle | no |
| route_west_to_a1_upper | q=unity/R=constant_global | deployed_burger | full | above_cross_aisle | above_cross_aisle | no |
| route_west_to_a1_upper | q=unity/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| route_west_to_a1_upper | q=unity/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |
| sanity_visible_apron_crossing | q=unity/R=commissioned | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=unity/R=commissioned | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=unity/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=unity/R=commissioned | spec_0.80x0.55 | full | <none> | <none> | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | deployed_burger | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | deployed_burger | full | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | below_main_aisle | no |
| sanity_visible_apron_crossing | q=unity/R=constant_global | spec_0.80x0.55 | full | <none> | <none> | no |

## What actually drove the legacy choice

The legacy total is dominated on some tasks by the soft no-go penalty (a safety-shaping term), not by an observability trade-off. Removing that term shows which route the legacy *observability* accounting preferred on its own.

| task | condition | legacy (total) | legacy (excl. no-go) | no-go share of total |
|---|---|---|---|---|
| control_west_to_a1_low | q=commissioned/R=commissioned | below_main_aisle | below_main_aisle | 1.000 |
| control_west_to_a1_low | q=commissioned/R=constant_global | below_main_aisle | below_main_aisle | 0.232 |
| control_west_to_a1_low | q=unity/R=commissioned | below_main_aisle | below_main_aisle | 0.232 |
| control_west_to_a1_low | q=unity/R=constant_global | below_main_aisle | below_main_aisle | 0.232 |
| occlusion_transit_a4 | q=commissioned/R=commissioned | below_main_aisle | above_connector | 0.999 |
| occlusion_transit_a4 | q=commissioned/R=constant_global | above_connector | above_connector | 0.045 |
| occlusion_transit_a4 | q=unity/R=commissioned | above_connector | above_connector | 0.045 |
| occlusion_transit_a4 | q=unity/R=constant_global | above_connector | above_connector | 0.045 |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | below_main_aisle | above_connector | 0.985 |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | above_connector | above_connector | 0.000 |
| route_apron_to_a2_mid | q=unity/R=commissioned | above_connector | above_connector | 0.000 |
| route_apron_to_a2_mid | q=unity/R=constant_global | above_connector | above_connector | 0.000 |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | below_main_aisle | above_connector | 0.995 |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | above_connector | above_connector | 0.000 |
| route_apron_to_a3_mid | q=unity/R=commissioned | above_connector | above_connector | 0.000 |
| route_apron_to_a3_mid | q=unity/R=constant_global | above_connector | above_connector | 0.000 |
| route_west_to_a1_upper | q=commissioned/R=commissioned | below_main_aisle | above_cross_aisle | 1.000 |
| route_west_to_a1_upper | q=commissioned/R=constant_global | above_cross_aisle | above_cross_aisle | 0.468 |
| route_west_to_a1_upper | q=unity/R=commissioned | above_cross_aisle | above_cross_aisle | 0.468 |
| route_west_to_a1_upper | q=unity/R=constant_global | above_cross_aisle | above_cross_aisle | 0.468 |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | below_main_aisle | below_main_aisle | 0.003 |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | below_main_aisle | below_main_aisle | 0.003 |
| sanity_visible_apron_crossing | q=unity/R=commissioned | below_main_aisle | below_main_aisle | 0.003 |
| sanity_visible_apron_crossing | q=unity/R=constant_global | below_main_aisle | below_main_aisle | 0.003 |

## Break-even localization weight

| task | condition | footprint | gate set | shortest | alternative | dT [s] | dLoc [nat.s] | break-even weight | selected |
|---|---|---|---|---|---|---|---|---|---|
| control_west_to_a1_low | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | above_cross_aisle | 12.0 | -66.68 | inf | below_main_aisle |
| control_west_to_a1_low | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | above_cross_aisle | 12.0 | -66.68 | inf | below_main_aisle |
| control_west_to_a1_low | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| control_west_to_a1_low | q=commissioned/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| control_west_to_a1_low | q=commissioned/R=constant_global | deployed_burger | deployment | below_main_aisle | above_cross_aisle | 12.0 | 0.00 | inf | below_main_aisle |
| control_west_to_a1_low | q=commissioned/R=constant_global | deployed_burger | full | below_main_aisle | above_cross_aisle | 12.0 | 0.00 | inf | below_main_aisle |
| control_west_to_a1_low | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| control_west_to_a1_low | q=commissioned/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| control_west_to_a1_low | q=unity/R=commissioned | deployed_burger | deployment | below_main_aisle | above_cross_aisle | 12.0 | -0.00 | inf | below_main_aisle |
| control_west_to_a1_low | q=unity/R=commissioned | deployed_burger | full | below_main_aisle | above_cross_aisle | 12.0 | -0.00 | inf | below_main_aisle |
| control_west_to_a1_low | q=unity/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| control_west_to_a1_low | q=unity/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| control_west_to_a1_low | q=unity/R=constant_global | deployed_burger | deployment | below_main_aisle | above_cross_aisle | 12.0 | 0.00 | inf | below_main_aisle |
| control_west_to_a1_low | q=unity/R=constant_global | deployed_burger | full | below_main_aisle | above_cross_aisle | 12.0 | 0.00 | inf | below_main_aisle |
| control_west_to_a1_low | q=unity/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| control_west_to_a1_low | q=unity/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=commissioned/R=commissioned | deployed_burger | deployment | above_connector | below_main_aisle | 3.3 | 25.95 | 0.127 | below_main_aisle |
| occlusion_transit_a4 | q=commissioned/R=commissioned | deployed_burger | full | above_connector | below_main_aisle | 3.3 | 25.95 | 0.127 | below_main_aisle |
| occlusion_transit_a4 | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=commissioned/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=commissioned/R=constant_global | deployed_burger | deployment | above_connector | below_main_aisle | 3.3 | 0.00 | inf | above_connector |
| occlusion_transit_a4 | q=commissioned/R=constant_global | deployed_burger | full | above_connector | below_main_aisle | 3.3 | 0.00 | inf | above_connector |
| occlusion_transit_a4 | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=commissioned/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=unity/R=commissioned | deployed_burger | deployment | above_connector | below_main_aisle | 3.3 | -0.00 | inf | above_connector |
| occlusion_transit_a4 | q=unity/R=commissioned | deployed_burger | full | above_connector | below_main_aisle | 3.3 | -0.00 | inf | above_connector |
| occlusion_transit_a4 | q=unity/R=commissioned | spec_0.80x0.55 | deployment | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=unity/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=unity/R=constant_global | deployed_burger | deployment | above_connector | below_main_aisle | 3.3 | 0.00 | inf | above_connector |
| occlusion_transit_a4 | q=unity/R=constant_global | deployed_burger | full | above_connector | below_main_aisle | 3.3 | 0.00 | inf | above_connector |
| occlusion_transit_a4 | q=unity/R=constant_global | spec_0.80x0.55 | deployment | - | - | - | - | nan | - |
| occlusion_transit_a4 | q=unity/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | 39.32 | 0.120 | below_main_aisle |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | deployed_burger | full | above_connector | below_main_aisle | 4.7 | 39.32 | 0.120 | below_main_aisle |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | 39.32 | 0.120 | below_main_aisle |
| route_apron_to_a2_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | deployed_burger | full | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a2_mid | q=unity/R=commissioned | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | -0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=unity/R=commissioned | deployed_burger | full | above_connector | below_main_aisle | 4.7 | -0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=unity/R=commissioned | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | -0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=unity/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a2_mid | q=unity/R=constant_global | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=unity/R=constant_global | deployed_burger | full | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=unity/R=constant_global | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a2_mid | q=unity/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | 36.16 | 0.130 | below_main_aisle |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | deployed_burger | full | above_connector | below_main_aisle | 4.7 | 36.16 | 0.130 | below_main_aisle |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | 36.16 | 0.130 | below_main_aisle |
| route_apron_to_a3_mid | q=commissioned/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | deployed_burger | full | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=commissioned/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a3_mid | q=unity/R=commissioned | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | -0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=unity/R=commissioned | deployed_burger | full | above_connector | below_main_aisle | 4.7 | -0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=unity/R=commissioned | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | -0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=unity/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_apron_to_a3_mid | q=unity/R=constant_global | deployed_burger | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=unity/R=constant_global | deployed_burger | full | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=unity/R=constant_global | spec_0.80x0.55 | deployment | above_connector | below_main_aisle | 4.7 | 0.00 | inf | above_connector |
| route_apron_to_a3_mid | q=unity/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_west_to_a1_upper | q=commissioned/R=commissioned | deployed_burger | deployment | above_cross_aisle | below_main_aisle | 2.1 | 45.67 | 0.046 | below_main_aisle |
| route_west_to_a1_upper | q=commissioned/R=commissioned | deployed_burger | full | above_cross_aisle | below_main_aisle | 2.1 | 45.67 | 0.046 | below_main_aisle |
| route_west_to_a1_upper | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| route_west_to_a1_upper | q=commissioned/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_west_to_a1_upper | q=commissioned/R=constant_global | deployed_burger | deployment | above_cross_aisle | below_main_aisle | 2.1 | 0.00 | inf | above_cross_aisle |
| route_west_to_a1_upper | q=commissioned/R=constant_global | deployed_burger | full | above_cross_aisle | below_main_aisle | 2.1 | 0.00 | inf | above_cross_aisle |
| route_west_to_a1_upper | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| route_west_to_a1_upper | q=commissioned/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_west_to_a1_upper | q=unity/R=commissioned | deployed_burger | deployment | above_cross_aisle | below_main_aisle | 2.1 | -0.00 | inf | above_cross_aisle |
| route_west_to_a1_upper | q=unity/R=commissioned | deployed_burger | full | above_cross_aisle | below_main_aisle | 2.1 | -0.00 | inf | above_cross_aisle |
| route_west_to_a1_upper | q=unity/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| route_west_to_a1_upper | q=unity/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| route_west_to_a1_upper | q=unity/R=constant_global | deployed_burger | deployment | above_cross_aisle | below_main_aisle | 2.1 | 0.00 | inf | above_cross_aisle |
| route_west_to_a1_upper | q=unity/R=constant_global | deployed_burger | full | above_cross_aisle | below_main_aisle | 2.1 | 0.00 | inf | above_cross_aisle |
| route_west_to_a1_upper | q=unity/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| route_west_to_a1_upper | q=unity/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | deployed_burger | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | deployed_burger | full | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=commissioned/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | deployed_burger | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | deployed_burger | full | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=commissioned/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| sanity_visible_apron_crossing | q=unity/R=commissioned | deployed_burger | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=unity/R=commissioned | deployed_burger | full | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=unity/R=commissioned | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=unity/R=commissioned | spec_0.80x0.55 | full | - | - | - | - | nan | - |
| sanity_visible_apron_crossing | q=unity/R=constant_global | deployed_burger | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=unity/R=constant_global | deployed_burger | full | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=unity/R=constant_global | spec_0.80x0.55 | deployment | below_main_aisle | - | - | - | nan | below_main_aisle |
| sanity_visible_apron_crossing | q=unity/R=constant_global | spec_0.80x0.55 | full | - | - | - | - | nan | - |

## Validation gates

- q=1 selects the shortest safe route: **PASS**
- localization cost is never negative (no reward for duration): **PASS**
- q=1 with constant/global R gives exactly zero localization cost: **PASS**
- no unsafe route was ever selected: **PASS**

- 28 (task, condition, footprint, gate set) combinations had NO safe candidate, so no route was selected. See the candidate table for the failing gate.

## Gazebo runs that must be repeated

No route selection changed for the deployed footprint under the full gate set; on that evidence no previously recorded run is invalidated by the route choice itself.

Independently of route choice, every run recorded before this change used the pre-correction objective, so any figure that quotes an EFE cost decomposition (risk / ambiguity split, total cost) must be regenerated: the terms are defined differently now.

