# Project Story

## One-Sentence Pitch

This thesis studies how a warehouse robot can plan more reliably with fixed
external cameras by learning where those cameras are trustworthy, then using
that reliability to construct a state-dependent observation covariance for
planning.

## Clean Storyline

A warehouse robot may be observed by fixed external cameras. These cameras are
useful, but their reliability is spatially uneven:

- close, central, unobstructed regions are usually easy,
- far regions, image edges, occlusions, glare, and clutter can be unreliable,
- detector confidence alone is not a calibrated measure of localization quality.

The robot should therefore not plan as if the camera has the same quality
everywhere. It should plan with an observation model that depends on where the
robot expects to be.

The thesis story is:

```text
known drivable region + camera calibration
-> weak camera-coverage prior
-> logged detector evidence over robot operation
-> camera reliability field over the drivable region
-> effective state-dependent observation covariance R_plan(s)
-> reliability-aware route behavior
```

## How The Old Paper Fits

The old paper demonstrated the downstream idea:

```text
external camera -> YOLO -> GP visibility map -> R_plan -> EFE planner
```

The lesson from the paper is that the middle part cannot be treated as a black
box. The new thesis should explain and validate:

- where the reliability map comes from,
- what assumptions it uses,
- how robot pose uncertainty affects learning it,
- how it differs from `R`,
- how it changes planning behavior.

The old paper becomes the historical baseline and motivation. The new thesis
contribution is the validated reliability-model pipeline.

## Recommended Contribution Statement

This work develops a reliability-aware observation model for external-camera
robot navigation over a known drivable region. A weak camera-coverage prior is
constructed from camera calibration and drivable-area support, refined from
logged detections, misses, and residuals, and converted through a separate
mapping into an effective state-dependent observation covariance for planning.

## Thesis Framing

The contribution is not:

- a pure EFE-planning contribution,
- a claim that `R` is learned directly,
- a claim that perfect warehouse geometry is known,
- a method that relies on Gazebo ground truth online.

The contribution is:

- a modular reliability model for external cameras,
- a validated bridge from perception evidence to planning covariance,
- a workflow that separates operational inputs from evaluation-only truth,
- an evidence chain showing when reliability-aware planning helps.

