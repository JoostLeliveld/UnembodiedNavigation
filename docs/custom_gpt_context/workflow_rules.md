# Modular Evidence-Based Workflow

The old failure mode was moving downstream before upstream modules were fully
validated. The new workflow is slower but safer.

## Default Loop

```text
choose one module
-> write the claim
-> state realistic assumptions
-> define the input/output contract
-> implement the smallest offline version
-> validate the module's real downstream job
-> compare against a simpler baseline
-> document failure modes and evidence
-> only then depend on it downstream
```

## Module Acceptance Template

Every module README should answer:

1. What problem does this module solve?
2. What is the precise contribution?
3. What would a real warehouse know at this point?
4. What is not assumed?
5. What are the inputs, outputs, units, frames, and state sources?
6. What literature supports the method?
7. What is the validation gate?
8. What baseline does it beat?
9. What failure cases remain?
10. What exact command/artifact/figure reproduces the evidence?

## Validation Ladder

1. Static contract check: paths, frames, units, matrix shapes, topics.
2. Single-example visual check: one frame/run by eye.
3. Batch metric check: metric that matters downstream.
4. Stress/failure check: camera-poor, stale, far, edge, ambiguous regions.
5. Baseline/ablation check: beat constant or simpler method.
6. Downstream smoke check: smallest consumer still behaves.
7. Documentation check: README, evidence registry, current/historical labels.

## Module Gates

### Perception

Do not stop at mAP. Validate:

- selected bottom-centre pixel,
- projection/localization residual,
- missed detections by region,
- detector score distribution,
- failure frames.

### Projection And State

Validate:

- homography/BEV projection residuals,
- affine correction before/after,
- stale-measurement handling,
- odometry-driven heading convention,
- state source labels.

### Reliability GP

Validate:

- training coverage over drivable region,
- held-out detection metrics,
- calibration/reliability curve,
- uncertainty map,
- residual map,
- simple baselines such as distance/range/FOV only.

### `R_plan` Mapping

Validate:

- explicit 2x2 matrix shape,
- units in pixel squared,
- monotonic trust-to-covariance curve,
- endpoints match `r_visible_uv` and `r_miss_uv`,
- planner consumes covariance, not a direct visibility reward.

### Planning

Validate:

- C1 and C2 differ only by planner-facing camera covariance,
- matched tasks/seeds,
- no-go/obstacle geometry held fixed,
- ambiguity explained separately from reliability,
- route changes explained by covariance and belief behavior.

### Experiments

Validate:

- current/historical labels,
- no odometry-as-ground-truth fallback,
- physics-contact cross-check where claimed,
- config/log/metric/figure provenance,
- invalid and near-success cases not hidden.

## Stop Conditions

Stop and document before continuing if:

- a module needs ground truth as an operational input,
- the result only beats a weak strawman,
- a simpler baseline performs similarly,
- a visual contradicts the numeric metric,
- downstream behavior changes for a reason other than the claimed mechanism,
- the evidence would not be available in a real warehouse.

