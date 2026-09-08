# availability_prediction

**Question:** at planning time there is no image, so can a commissioned field predict whether
a camera will return a usable detection at a position the robot has not reached yet?

**Answer:** yes. A spatial Gaussian process reaches Brier 0.0590 on held-out tiles against
0.2352 for a constant per-camera rate and 0.1348 for CAD sightline geometry.

**The finding worth the space:** the gain is uneven and the pattern is mechanical. Camera C
improves 19.6x, camera D only 2.1x, because C's misses are racks blocking a sightline (a
sharp spatial boundary a GP learns) while D's are range-driven (diffuse, no boundary in
position). An availability field inherits the geometry of the failure it models.

`summarize_availability.py` reads the frozen commissioning artifact; it fits nothing.

Results, interpretation and limitations: `logs/studies/availability_prediction_20260907/`.
