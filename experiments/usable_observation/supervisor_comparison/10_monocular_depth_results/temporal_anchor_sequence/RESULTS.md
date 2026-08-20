# Temporal ground-anchor sequence results

Status: **longitudinal mechanism evidence**, not a real-time drift experiment.

The frozen Gazebo sequence supplies 21 synchronized updates × four cameras = 84 RGB
frames. DA-V2 relative depth was inferred from RGB only. Each source frame is replayed at
a 10 s operational cadence, giving 21 Bayesian updates across 200 s. Gazebo depth and
visibility are opened only afterwards by the evaluator.

## Untouched sequence

| arm | valid camera updates | median cycle structure MAE | median cycle visibility balanced accuracy | median cycle visible IoU | median four-camera method time |
|---|---:|---:|---:|---:|---:|
| enhanced, fresh affine | 84/84 | 0.921 m | 92.02% | 90.18% | 2.30 s |
| enhanced + Bayesian affine | 84/84 | 0.922 m | 92.02% | 90.18% | 1.95 s |

Median absolute successive scale change ratio, temporal/single:
undefined because the single-frame median change is exactly zero.
The equivalent shift-change ratio is
undefined because the single-frame median change is exactly zero.
These are stability diagnostics in DA-V2 relative-depth parameter units, not localization
errors.

## Predeclared floor-anchor dropout

The external floor mask is set empty at four predeclared synchronized updates, producing
16 camera-frame anchor failures. The selection was frozen in `PROTOCOL.md` and did not use
oracle outcomes.

| arm | valid all-sequence updates | valid injected-dropout updates |
|---|---:|---:|
| enhanced, fresh affine | 68/84 | 0/16 |
| enhanced + Bayesian affine | 84/84 | 16/16 |

The temporal arm retained
100% of the
injected dropout updates by reusing only a recent scale/shift posterior. It still
computed metric depth and visibility from the current RGB-derived depth frame; obstacle
evidence was not carried over.

At the four dropout cycles, stale-prior reuse changed median structure-depth MAE by
+0.7
mm, visibility balanced accuracy by
+0.003
percentage points, and visible IoU by
+0.000
percentage points relative to the untouched temporal replay of those same frames.

## Boundary and interpretation

This demonstrates repeated Bayesian updates and the intended graceful-degradation
mechanism on real Gazebo renders. It does not demonstrate natural floor-segmentation
failure frequency, 200 seconds of illumination drift, or real-camera ageing. The 21
updates are sequential and dependent, so all summaries are descriptive rather than iid
confidence claims.
