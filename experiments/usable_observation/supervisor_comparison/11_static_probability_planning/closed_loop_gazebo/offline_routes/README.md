# Offline executable routes

These artifacts are the ROS- and Gazebo-free route-selection result for
`route_tall_shadow_west_safe_start` in `warehouse_full_4cam.world.sdf`.

The headline comparison is in `01_executable_routes.png`. C1 takes the shortest complete route
(14.24 m) but crosses a predicted observation shadow where planning `p_use` falls to 0.001. C2
(`R/p`) and C3 (explicit hit/miss) choose a 14.50 m route whose minimum planning `p_use` is
0.994. The 0.26 m difference is the modeled price of remaining observable.

Every selected route:

- contains the exact start and mission goal;
- is complete, with zero endpoint error;
- has at least 0.25 m driveable clearance under 0.04 m segment sampling;
- was rescored after simplification to the controller waypoint polyline;
- comes from the same six-candidate set for all conditions.

`routes.json` is the executable interface. For each condition,
`controller_waypoints_xy` excludes the initial pose and includes the exact mission goal. A later
closed-loop run must bypass continuous global EFE route discovery when consuming these
waypoints. `selected.csv` contains the concise numerical comparison, `candidates.csv` contains
all alternatives, and `route_points.csv` contains dense samples for auditing or plotting.

These values are planning-model predictions based on the frozen static four-camera probability
field. They are not empirical detection rates on these paths, filter localization metrics, or
Gazebo navigation outcomes.
