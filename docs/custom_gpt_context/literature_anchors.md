# Literature Anchors

Use literature as support for a specific module, not as decoration.

## Gaussian Processes For Spatial Reliability

Use to justify:

- reliability as a latent function over `(x, y)`,
- uncertainty-aware predictions,
- kernels/length scales and posterior uncertainty,
- held-out likelihood/calibration metrics.

Core reference:

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, MIT Press,
  2006. Official page: https://gaussianprocess.org/gpml/

## GP Mapping And Spatial Uncertainty

Use to justify:

- continuous spatial map learning,
- map uncertainty,
- fewer observations through spatial correlation,
- active or online map refinement.

Examples:

- Ghaffari Jadidi, Valls Miro, Dissanayake, "Gaussian Process Autonomous Mapping
  and Exploration for Range Sensing Mobile Robots", Autonomous Robots, 2018.
  https://arxiv.org/abs/1605.00335

## Uncertain Inputs And Belief-Aware Updates

Use to justify:

- robot pose uncertainty affecting map learning,
- spreading or weakening datapoint influence,
- not treating estimated position as ground truth.

Example:

- Ghaffari Jadidi, Valls Miro, Dissanayake, "Warped Gaussian Processes
  Occupancy Mapping with Uncertain Inputs", IEEE RA-L, 2017.
  https://arxiv.org/abs/1701.00925

Important caveat:

If the robot's reported covariance is overconfident, uncertainty-aware updates
can become misleading. Validate the covariance before trusting the weighting.

## Perception-Aware Planning

Use to justify:

- planning should account for sensing quality,
- action and perception can conflict,
- perception-aware objectives/costs can improve behavior.

Example:

- Falanga et al., "PAMPC: Perception-Aware Model Predictive Control for
  Quadrotors", IROS 2018. https://arxiv.org/abs/1804.04811

Connection:

This thesis differs by using fixed external cameras and a learned reliability
field, but shares the principle that planning should account for perception
quality.

## 3D Occupancy / Optional Height Priors

Use only for optional stronger priors when sensed geometry exists.

Example:

- OctoMap: probabilistic 3D occupancy mapping with occupied, free, and unknown
  space. https://octomap.github.io/

Connection:

If a real warehouse provides LiDAR/RGB-D/stereo/CAD geometry, it can strengthen
the cold-start prior. If only drivable region is known, do not claim occlusion
is known.

## Kalman / Observation Covariance

Use to justify:

- `R` as measurement-noise covariance in an observation model,
- separation between process noise, measurement noise, and state uncertainty,
- why a covariance matrix has shape, units, and semantics.

Use any standard filtering reference used by the thesis or course material.
The important local rule is:

> The GP predicts reliability. The observation model maps reliability into an
> effective covariance. These are not the same object.

