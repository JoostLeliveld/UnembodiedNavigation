# Anomaly investigation: why a camera reading is 130 cm wrong

> **CLOSED, and kept for its method.** The defect it found — an admission gate with no caller
> in the runtime — was fixed on 2026-08-27. The gate now runs, so the readings this folder
> examines cannot occur any more, and the numbers below describe a pipeline that no longer
> exists. What is worth keeping is the technique: when a statistic looks wrong, look at the
> frame the detector actually saw.

**A camera reading should not be a metre out. These were.** This folder finds out why, using
the actual detector views rather than statistics.

```bash
python3 experiments/anomaly_investigation/show_the_view.py
```

Offline, on frames already on disk. Writes `logs/studies/anomaly_investigations/`.

## The finding

**The admission gate never runs in the live pipeline.**

`reliability/silhouette_observation.py` defines `plausibility_reasons()` — the four checks that
decide whether a detection is usable: is it tall enough, is it the right width, is its bottom
edge where the prediction says the robot's contact point is, and does it touch the frame edge.
It has the same thresholds as the commissioning gate, it is unit-tested, and **nothing calls
it**. The camera manager imports only `equivalent_position_measurement` from that module.

So every detection the detector emits above its confidence threshold becomes a position
measurement, including the ones commissioning would have thrown away.

## Why that produces a metre

When the robot's feet are hidden — behind a rack leg, a pallet, or the edge of an aisle — the
detected box's bottom edge sits higher in the image than the robot's true contact point.
Back-projecting a higher pixel onto the floor plane puts the robot **further away along the
viewing ray**. It is not noise; it is a geometric consequence, and it always pushes the same
direction.

The worst readings in the campaign show exactly that signature: camera E, robot truly at
(0.95, 0.57), read at (0.27, 1.68) — 1.3 m displaced almost exactly away from the camera.

## The proof, in commissioning data

The nearest commissioned position to that anomaly is (0.67, 0.32), seen by camera E at 14.5 m.
All six headings were photographed. **Every one fails the gate:**

| heading | gate verdict | what the ungated reading would have been |
|---|---|---|
| 0° | too_short, wrong_width, bottom_hidden | 21.9 cm off |
| 60° | wrong_width | 16.3 cm off |
| 120° | wrong_width | 13.5 cm off |
| 180° | too_short, wrong_width, bottom_hidden | 20.1 cm off |
| 240° | wrong_width | 16.5 cm off |
| 300° | wrong_width, bottom_hidden | 13.2 cm off |

Commissioning rejected all six, which is why its numbers are clean. The live pipeline accepted
all six, which is why the drives are not.

## What this explains

- **The 130 cm readings**, and the 58 cm p95 that the latency fix did not touch.
- **Camera C's 17.3 cm median** — it looks south-east across the west aisles, so much of what
  it sees is partly behind racking.
- **Camera A's 87.9 cm below 6 m** — the robot is nearly underneath it, where the box runs
  into the frame edge, which is exactly what the border check exists to catch.
- Why the commissioned rate of "chances that become a usable measurement" was 29%: the gate
  was doing most of the work, and in the live system it was doing none of it.


## There is no third defect

The per-camera "geometry problems" identified before the gate was found are the same bug. Run
the commissioning capture through the gate and score only what it admits:

| camera | ungated | gated | what it was |
|---|---|---|---|
| A below 6 m | 87.9 cm | **every reading refused** | the robot is nearly under the camera, so the box runs into the frame edge — exactly what the border check is for |
| C, all ranges | 17.3 cm | **1.47 cm** | not a bad camera: it looks south-east across the west aisles, so much of what it sees is partly behind racking |
| D beyond 20 m | 31.5 cm | 5.34 cm | long range plus occlusion |
| E, overall | 5.3 cm | 1.38 cm | |
| worst reading anywhere | 122 cm | **27 cm** | |

Gated, every camera reads 0.9–2.5 cm median with a mild and physically sensible growth with
range (0.7 cm close, 2.6–7.0 cm beyond 20 m). **Two defects — the missing gate and the 400 ms
application lag — account for the whole gap** between the commissioned 1.5 cm sensor and the
8 cm one the drives saw, and for the entire tail.

## Consequences for what was already written

- `../fusion_on_fixed_routes/RESULTS.md` and everything under it was measured with **both**
  defects live. The arm ordering may survive; no absolute number should be quoted.
- The "per-camera geometry defect" paragraph in `../fusion_on_fixed_routes/latency/README.md`
  is superseded by this: it is the gate, not the cameras.
- Commissioning was right all along, and for a specific reason worth keeping: it ran the check.
  The gap between commissioning and operation was never the sensor model — it was two things
  the sensor model assumed and the pipeline did not do.


## Both defects fixed, driven, measured

Two runtime changes, each behind a flag and default-on:

- `admission_gate` — the check now runs, fails closed on a detection it cannot check, and
  reports its refusal reasons. Bootstrap readings pass unchecked and are counted, because the
  check needs a predicted box and the prediction needs a belief that the first correction
  creates: refusing there deadlocks the robot, which is what the first attempt did.
- `correction_timestamp_compensation` — each correction is carried forward from the pose it
  describes to the pose it is used on, with odometry.

Scored the honest way — **each correction once, at the moment it is published**, rather than
re-scoring a held message while the robot drives on:

| run | corrections | median | p95 | worst |
|---|---|---|---|---|
| neither fix | 711 | 6.53 cm | 39.0 | 90 |
| timing fix only | 1438 | 2.16 cm | 36.3 | 91 |
| **both** | 1338 | **1.78 cm** | **9.0** | **37** |
| *commissioning, for reference* | *3351* | *1.44 cm* | *9.2* | *27* |

**The operational sensor now matches the commissioned one.** The timing fix takes the median,
the gate takes the tail, and neither substitutes for the other.

### A note on the metric itself

The p95 of ~58 cm quoted throughout the earlier study was partly an artefact. `state_error_gt_m`
scores whatever correction the filter is currently holding, so during an outage it keeps
re-scoring an ageing message against a moving robot and reports the robot's own travel as
measurement error. With the gate on, outages are real and the artefact is severe: an apparent
p95 of 151 cm, of which none is measurement error. Corrections must be scored once, when they
land. Every earlier correction-level number in this study is affected.

### What the gate costs

It converts bad measurements into absent ones, which is exactly the trade `PLAN.md` describes.
Measured, same route, same decision rate:

| | corrections | median gap | worst outage | outages over 1 s |
|---|---|---|---|---|
| timing fix only | 1439 | 100 ms | 5.0 s | 1 |
| gate + timing fix | 1339 (-7%) | 100 ms | **13.0 s** | 3 |

A 7% reduction in corrections and a worst outage that grows from 5 s to 13 s. That is the price
of not fusing a reading of a robot whose feet are hidden, and it is an argument for the
availability side of the work rather than against the gate.


## What is left, and it is one thing

With both defects fixed the measurement is honest — 1.79 cm median claiming ~1.5 — but the
belief is still only ~52% honest. That is now a single, well-posed problem.

**The belief shrinks below a bias that does not shrink.** Measured on the fixed pipeline:

| | |
|---|---|
| systematic part of the fused correction | **-1.72 cm along the direction of travel**, spread 1.89 cm |
| what the belief's ellipse shrinks to | **1.12 cm** on its short axis |
| actual error along that short axis | **2.63 cm** — 2.9 sigma |
| actual error along its long axis (3.5 cm sigma) | 0.66 cm — 0.13 sigma |

The ellipse is not too small on average. It is **mis-oriented**: confident in the direction
where the error lives. A filter fusing N corrections shrinks toward `sigma/sqrt(N)`, and after
a hundred corrections it claims a couple of millimetres while a 1.7 cm systematic sits
untouched. This is the mechanism `CLAUDE.md` already names as the core finding of the thesis,
now measured on the repaired pipeline rather than inferred from a retired one.

Where the 1.7 cm comes from: the correction is propagated to the decision instant but consumed
~50 ms later, and 50 ms at 0.22 m/s is 1.1 cm. It is a *residual* of the lag, not a camera
lean — it points along travel, not along any camera's ray.

**The existing mechanism does not fit it.** `bias_floor_along_slope_m_per_m` exists in the
manager, but it is disabled by default, it is only wired into the legacy fusion path, and it
models a bias along the camera's line of sight. This one is along the direction of travel.

**So the next change is a covariance floor the belief cannot shrink through, oriented along
travel and sized by the residual propagation interval (`v * delta`).** That is one number with
a physical derivation rather than a fitted fudge, and it is the last thing between this
pipeline and an honest belief.


## The floor, driven — and it changes the study's conclusion

Four arms, primary route, one change: the residual propagation interval is declared as
uncertainty along the direction of travel.

| arm | both defects live | + timing fix | + admission check | **+ floor** |
|---|---|---|---|---|
| F1 single best camera | 43.4% | 52.3% | 51.2% | **66.0%** |
| F2 distance and angle | 45.9% | 48.5% | 53.5% | **68.2%** |
| F3 precisions add | 26.0% | 39.2% | 43.0% | **69.4%** |
| F4 network, divided by N | 54.8% | 50.1% | 54.9% | **73.0%** |

(Truth inside the belief's own 95% ellipse. Honest is 95%.)

**Every arm gains 15-26 points, and the arms converge.** The gap between the proposed pooling
rule and the standard one:

| | F4 over F3 |
|---|---|
| both defects live | **+28.7 points** |
| timing fixed | +10.9 |
| and the check on | +11.9 |
| **and the floor** | **+3.6** |

### What that means, stated against our own earlier claim

The study's headline was "dividing by N buys calibration". On the repaired pipeline **that
advantage is 3.6 points, down from 28.7** — most of what looked like a pooling result was a
conservative claim quietly absorbing three modelling failures: a stale correction, an
unenforced admission check, and a bias declared nowhere.

The honest conclusion is now about the covariance, not the rule:

> **Get the measurement model right — apply corrections at the time they describe, refuse
> detections whose contact point is hidden, and declare the residual as uncertainty in the
> direction it acts — and the choice of fusion rule stops mattering much.** Four rules that
> differed by 29 points of calibration differ by 4 once the model is right, and all four land
> at 66-73% where they started at 26-55%.

That is a stronger result than a rule bake-off, and it is the opposite of what this study
concluded three days of driving ago. **None of it is honest yet** — 73% is not 95% — and the
remaining gap is the next question rather than a caveat to bury.

### What the floor did NOT change

Correction accuracy is untouched, as it should be: 1.74 → 1.68 cm median, p95 8.1 → 7.6 cm.
The floor changes what the belief *claims*, not what the network *measures*.


## The remaining gap, accounted for

With all three fixes the belief is 66-73% honest. Splitting by whether a correction is recent
separates two different problems:

| | n | belief honesty |
|---|---|---|
| a correction landed within 0.3 s | 5354 | **74.3%** |
| the last correction is older than 0.3 s | 583 | **21.3%** |

**The outage half is a degenerate process noise.** Between corrections the filter's covariance
does not grow, it *stretches*:

| correction age | major sigma | minor sigma | aspect | error along major | error along minor |
|---|---|---|---|---|---|
| fresh | 3.5 cm | 1.27 cm | 3:1 | 0.6 cm | 2.4 cm |
| 1-3 s | 104 cm | 2.22 cm | 46:1 | 1.8 cm | 7.3 cm |
| 7-20 s | **635 cm** | **3.64 cm** | **174:1** | 3.6 cm | **10.7 cm** |

Six metres of uncertainty one way, three and a half centimetres the other. The error is 10.7 cm
across the tight axis — 2.9 sigma — so the truth escapes an ellipse that is, on average,
enormously conservative. That is why the belief simultaneously claims 4.5 m of uncertainty and
fails its own 95% test 88% of the time during outages.

Two things this is *not*: it is not the process noise being too small (its growth of 7.2 cm over
a 13 s outage exceeds the 1.30 +- 0.32 cm/m of drift odometry actually accumulates), and it is
not heading (1.3 degrees median during outages, correlation +0.15 with the error).

**The fix is the shape, not the size.** A robot that has driven two metres unaided is uncertain
in both directions; this covariance says it is uncertain in one. Floor the minor axis by
distance travelled since the last correction, or derive the growth from the unicycle Jacobian
with heading uncertainty coupled into cross-track.

**The other half — 74.3% honest with a fresh correction — is still open**, and it is now the
only thing left between this pipeline and an honest belief.
