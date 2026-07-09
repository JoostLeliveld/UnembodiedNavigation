# Contribution Claims

This file lists safe claims, caveated claims, and claims to avoid.

## Safe Core Claim

The thesis develops and validates a reliability-aware observation model for
external-camera robot navigation over a known drivable region.

The method:

```text
learns/predicts camera reliability
-> maps reliability to effective observation covariance R_plan(s)
-> uses that covariance in reliability-aware planning
```

## Safe Contribution Bullets

- Model camera reliability over the drivable region rather than treating all
  camera observations as equally trustworthy.
- Keep detector reliability, conditional measurement noise, and effective
  planning covariance separate.
- Convert reliability/trust into a fixed-shape 2x2 `R_plan(s)` matrix with
  clear units.
- Evaluate constant covariance against reliability-aware covariance on matched
  tasks, seeds, world, and no-go geometry.
- Use ground truth only for evaluation and diagnostics.
- Use modular validation gates so perception, projection, GP, covariance, and
  planning claims are not entangled.

## Claims Requiring Caveats

### Geometry Visibility Prior

Use only if assumptions are explicit:

> A geometry/camera model can provide a cold-start observability prior when
> geometry or sensed height is available.

Caveats:

- if only drivable region is known, it cannot know occlusion,
- range/FOV may explain much of the signal,
- synthetic depth/CAD emulations are not real sensor evidence,
- fresh detector logs are needed to make it empirical.

### Belief-Weighted Updates

Use only if current evidence supports it:

> Belief covariance can be used to decide how confidently logged detector
> outcomes should update the reliability map.

Caveats:

- if covariance is overconfident or poorly calibrated, weighting can hurt,
- naive updates may match or beat uncertainty-weighted updates in current logs,
- this should be tested with held-out detection metrics, not assumed.

### Multi-Camera Extension

Use as planned/what-if unless real multi-camera data exists:

> Multiple cameras can each maintain a reliability field over the same drivable
> region and be combined for planning.

Caveat:

- hypothetical second-camera plots are commissioning illustrations, not current
  experimental evidence.

## Claims To Avoid

Do not claim:

- "The GP learns R."
- "The GP is a visibility reward."
- "The GP is an obstacle map."
- "Geometry visibility is known from the drivable region."
- "Gazebo ground truth is part of the operational method."
- "YOLO mAP proves good localization."
- "C2 wins because it directly maximizes visibility" in the locked campaign.

## Preferred `R_plan` Explanation

The GP predicts a scalar reliability/trust value over the drivable region. The
planner then maps that value into a 2x2 image-space measurement covariance:

```text
R_plan(s) =
[ sigma_plan^2(s)  0               ]
[ 0                sigma_plan^2(s) ]  px^2
```

The GP changes the variance value through a separate mapping. The matrix shape,
coordinate meaning, and units are fixed by the observation model.

