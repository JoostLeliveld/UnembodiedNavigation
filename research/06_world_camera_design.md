# World and camera design

Worlds are samples from measurable properties, not anonymous Warehouse A/B/C labels.

## Required world properties

Record occlusion density, aisle length/openness, camera-overlap fraction, symmetry,
candidate-pose range distribution, layout-change frequency, camera-poor floor fraction,
and the number and cost of route alternatives. Each benchmark split must explain which
property changes and which remain fixed.

The principal comparison should include at least:

1. a relatively symmetric, low-to-moderate occlusion world for mechanism checks;
2. an asymmetric world with unequal route alternatives, uneven overlap, and camera-poor
   regions for route discrimination;
3. a changed-layout variant of the asymmetric world for staleness and transfer.

## Required camera properties

Record height, pitch, viewing angle, horizontal/vertical FOV or focal length, resolution,
update rate, overlap type, range distribution, occlusion exposure, calibration bias, and
residual correlation floor.

The present four cameras use nominally identical hardware. Supported diversity is limited
to position, occlusion geometry, handover role, and measured systematic bias. Optical,
resolution, and frame-rate diversity require new evidence.

## Noise contract

Keep camera geometry, detector, threshold, robot, controller, planner weights, route start
and goal, evaluation labels, seeds, and total sample budget identical across reliability
methods. Use paired seeds and apply noise at shared interfaces.

Recommended sensitivity ladder (not a claim of sensor realism until justified):

- pixel noise: `0`, `0.5`, `1.0`, `2.0` px standard deviation;
- calibration yaw drift: `0`, `0.1`, `0.25`, `0.5` degrees;
- calibration translation: `0`, `0.025`, `0.05`, `0.10` m;
- message latency: `0`, `50`, `100`, `200` ms;
- dropout: `0`, `5`, `15`, `30` percent;
- layout state: nominal, locally changed, globally changed.

Run the nominal level for the primary comparison and treat the remaining levels as paired
sensitivity arms. Pilot data may refine the ladder once, before preregistration; never tune
levels per method.
