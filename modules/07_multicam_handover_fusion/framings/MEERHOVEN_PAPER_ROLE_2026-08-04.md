# Meerhoven's role in the paper framing

[Back to framings](README.md) · operational plan:
[`MEERHOVEN_COMMISSIONING_PLAN`](../../../experiments/warehouse_layout_sketches/MEERHOVEN_COMMISSIONING_PLAN.md)

## Decision

Meerhoven **fits the assistive-infrastructure framing**, but it does not replace the
compact `warehouse_aws` B1 benchmark or become evidence merely by looking more realistic.
Its paper role is a second, exploratory scale-and-heterogeneity evaluation of the service
contract:

> Can a fixed external-camera localization service state where it is trustworthy, hand
> observations across a heterogeneous inherited network, and expose the resulting spatial
> uncertainty to a receding-horizon belief-space controller in a genuinely occluded
> logistics layout?

This is stronger and more coherent than presenting Meerhoven as “a bigger warehouse.” Its
twelve cameras have distinct operational histories, mounts, heights, pitches, views and
occlusion regimes. That is exactly where one global detector score, one projection bias,
or one covariance stops being a credible service contract.

## What it can support if commissioning passes

- **External localization is a spatially varying service.** The accepted geometric design
  already spans roughly a tenfold range in per-camera floor share and contains blind,
  single-covered and redundant zones. Detector and GP evidence must replace geometry
  before this becomes an empirical claim.
- **Correct the outliers, leave unresolved cameras raw.** Twelve per-camera commissioning
  decisions are a more meaningful deployment test of the gated calibration policy than
  four symmetric wall cameras.
- **Handover is a contract, not just a camera switch.** The planner and estimator need the
  identity, calibration, predicted availability and covariance of the camera supplying a
  correction.
- **Assistive infrastructure is most valuable where onboard self-assessment is weakest.**
  Blind aisle segments, changing block stacks, a mezzanine and awkward inherited mounts
  create externally legible failure modes without claiming to replace onboard safety
  sensing.

## What it cannot support

- It is Gazebo evidence, not a real warehouse deployment.
- It contains one robot, so it cannot support fleet coordination, multi-target association
  or throughput claims.
- It does not measure battery savings or justify duty-cycling onboard sensing.
- It does not prove learned reliability always produces a longer visible route. A safe
  stop, delayed commitment, different handover, or a null route result can all be valid.
- Camera count is not the independent variable. The claim remains about heterogeneous
  regimes and honest uncertainty, not “twelve beats four.”

## Relationship to the existing paper

| Surface | Role |
|---|---|
| `warehouse_aws` B1 | compact paper core; clearest controlled planning comparison |
| `warehouse_full_4cam` | existing four-camera commissioning/fusion evidence; artifacts remain locked to that geometry |
| `warehouse_meerhoven` | exploratory external-validity test for heterogeneous assistive infrastructure; promotable only after the full artifact chain passes |

The safest paper structure is therefore: establish the method and controlled effect in the
compact benchmark, then use Meerhoven as a stress test of commissioning, handover and
safe-operating-envelope transfer. If its detector, calibration, GP or repeated runs fail a
gate, report that failure as a deployment limit rather than weakening the compact result.

## Paste-ready transition paragraph

> We additionally evaluate the frozen pipeline in a larger brownfield logistics layout
> designed around an inherited, heterogeneous network of fixed external cameras. This
> second environment is not used to tune the method or to replace the compact route-choice
> benchmark. It tests whether the same observation-service contract remains auditable when
> camera mounts, ranges, occlusions and overlap vary across operational zones. We report
> Meerhoven results only when the world, detector, per-camera projection calibration,
> learned reliability artifacts and seeded closed-loop logs share one recorded provenance
> chain.

