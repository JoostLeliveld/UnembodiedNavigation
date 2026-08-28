# Presentation: localization-aware navigation with infrastructure cameras

Speaking notes and slide plan. Nine main slides plus a status slide, then backups.
Slides 8a-8c are the fusion experiment in detail: the route first, then the six arms.
Slide 8d is what the drives found wrong with the pipeline, and why the earlier
numbers had to be re-earned.
Companion to [`PLAN.md`](PLAN.md), which holds the five sentences the paper has to earn.

**The line to keep coming back to:**

> Availability determines whether a correction is likely to arrive.
> The measurement covariance determines how much to trust it once it has.

**The ladder to climb back onto whenever discussion gets technical:**

```
localization can degrade
        ↓
external cameras can correct it
        ↓
but corrections are patchy and imperfect
        ↓
so model the camera support we expect
        ↓
and plan using it
```

---

## Opening, spoken

> The broad problem is localization during warehouse navigation. The robot always has
> onboard localization, but that estimate drifts. Fixed infrastructure cameras can provide
> global corrections — but not everywhere and not all the time, because of occlusion,
> clutter and viewpoint. So two geometrically valid routes can expose the robot to very
> different localization support.
>
> My previous method already tried to exploit this, using a learned reliability score that
> modified the planner's assumed observation covariance. What I've realized is that this
> combines two separate quantities: the probability that the robot will actually receive a
> camera observation, and the uncertainty of that observation once received.
>
> The new formulation separates them. I learn where usable camera observations are likely
> to occur, calibrate how uncertain successful observations are, and use both to predict
> the robot's future belief during planning. The experimental question is whether that
> produces better localization for a reasonable increase in navigation cost.

---

## Slide 1 — The warehouse problem

**Title:** Localization-aware navigation with infrastructure cameras

Warehouse top view: robot start, goal, two candidate routes, the five cameras, one
camera-poor area. **No equations.**

- Onboard localization is continuous but drifts.
- Infrastructure cameras give global corrections, but only intermittently.

> **Should the robot change its route to improve the localization support it will receive?**

## Slide 2 — Why route choice matters

Almost entirely visual. Two routes with a timeline of when corrections arrive:

```
Route A  short          ● ●                    ●        long gap, uncertainty grows
Route B  slightly longer ● ●  ●   ● ●   ●  ● ●          regular corrections, uncertainty held
```

> The shortest geometric route is not necessarily the best route when localization
> quality matters.

Make the problem understandable before any implementation appears.

## Slide 3 — What makes external cameras difficult

Three boxes, nothing else:

1. **Availability is not one** — occlusion, clutter, detector failure.
2. **A successful detection is not perfectly accurate.**
3. **Both vary with where the robot is and how it sits relative to the camera.**

Measured here, over 2 316 robot poses:

| | a clear line of sight | a **usable** sighting |
|---|---|---|
| from no camera | **0 %** | **18 %** |
| from two or more | 73 % | 43 % |

> Geometry says the building is completely covered. Nearly a fifth of it is not.
> So the planner needs a model of expected localization support, not geometric visibility.

Figure: `logs/studies/deck_figures/fig_map_promise_vs_reality.png` (the same warehouse, twice)
and `fig_funnel.png` (where the sightings are lost).

## Slide 4 — What changed from the previous paper

**Probably the strongest slide. Give it the whole page.**

```
PREVIOUS                              NEW
Camera + YOLO                         Camera + YOLO
      ↓                                  ├──►  will an observation arrive?
confidence / reliability                 │            ↓
      ↓                                  │      availability(s)
     GP                                  │
      ↓                                  └──►  if it does, how accurate?
 effective R_plan                                     ↓
      ↓                                          covariance(s)
   planner                                            ↓
                                          expected belief evolution → planning
```

Previous: one reliability quantity reached the planner by changing an assumed covariance.

> **Separate "will I receive a correction?" from "how accurate is it if I do?"**

Figure: `fig_availability_examples.png` — the same camera, three places: robot in the open
(usable from all 6 headings), half behind a rack leg (3 of 6), buried in stacked cartons
(0 of 6). Real frames, with the true robot position marked.

## Slide 5 — The sensor model

For each camera: the probability of a usable observation at a given robot state, and — when
one occurs — a position measurement with a calibrated covariance.

Figure: `fig_uncertainty.png` — the same camera at 4–7 m and at 17–21 m. The stated
uncertainty goes from 1.2 cm to 4.0 cm, and it is a **stretched ellipse along the line of
sight**, not a circle, because that is the direction in which a pixel is worth the most
centimetres. The measured errors sit inside the stated ellipse in both — and the ellipse
came from one number, before seeing any of them.

Footer, small: *a fixed commissioned correction removes the dominant repeatable sensor
offset before the covariance is estimated.* **Do not explain the correction's parameters
unless asked.**

Put confidence here, briefly:

> YOLO confidence is retained, but its role is tested rather than assumed —
> admission, and possibly covariance refinement. Not treated as a covariance by fiat.

## Slide 6 — How the model is obtained

The commissioning slide.

```
commissioning observations
        ├── hit / miss ─────────► availability map
        └── error when it hit ──► measurement covariance
```

Bias appears as one small box in the chain: raw measurement → fixed commissioned
correction → centred residuals. **No bias comparison table on this slide.**

**Do not call the box-versus-centre problem a bias.** That is the observation model, it is
worth 30 cm rather than 0.5, and nothing about it is fitted — `fig_geometry.png` is a main
deck slide, not a backup.

If bias needs a picture at all, it is `fig_bias.png`: all 3 351 sightings on the left with
the 2.2 cm scatter ring, and a 12x zoom on the right showing the average move from 0.55 cm
to 0.20 cm. One figure, two numbers, move on.

## Slide 7 — What the planner does

The robot has a current uncertainty. For a candidate future place: if a camera observation
arrives, the uncertainty shrinks by an amount the covariance determines; if it does not, the
uncertainty keeps growing. The planner weights those two outcomes by how likely the
observation is.

> Candidate routes are evaluated according to the localization corrections they are
> expected to receive, in addition to normal navigation cost.

Say out loud, because it is the distinction people miss:

> **We are not rewarding visibility. We are predicting the effect of likely future
> localization measurements.**

Obstacle and keep-out costs stay entirely separate.

## Slide 8 — Experiments, as three questions

Three questions, not twenty metrics.

| | question | comparison | headline number |
|---|---|---|---|
| 1 | Can we predict where usable observations occur? | constant / geometry / old reliability model / proposed | Brier score on held-out places |
| 2 | Do camera observations genuinely improve localization, and how should several cameras be combined? | six arms on one route — see the next three slides | position error, cm |
| 3 | Does planning with availability help? | shortest / uniform / geometry / previous method / proposed | localization error **and** extra path length, % |

Supporting for question 3: the longest stretch with no usable observation.

## Slide 8a — The route the fusion runs drive

**Show this before naming a single arm.** Figure: `logs/studies/deck_figures/fusion/01_the_route.png`.

The west dock door, three metres from camera A, to the east cross aisle under camera D —
**30.6 m, and all five cameras get a turn**: A 7 m, B 14 m, C 4 m, D 17 m, E 19 m. The
number of cameras watching at once runs from none to four: 2.4 m with nothing, 9.6 m with
one, 9.2 m with two, 5.2 m with three, 4.4 m with four.

> **Every arm drives this same route.** So the only thing that differs between arms is how
> the cameras' readings are combined.

Two lines to say out loud, because they are the honest part:

- The route came out of the **lane geometry alone** — six corridors, the shortest is driven.
  Route choice is deliberately *not* the treatment here; it becomes the treatment later.
- Camera C only carries 4 m. The corridors that give it 9–21 m spend their length in the
  single-camera west aisles, which would collapse the two-or-more-camera share from 61 % to
  about a quarter — and that share is the axis this experiment is measured along.
- If asked how tight it is: 0.354 m at the worst corner, measured along the polyline. The
  robot is 0.80 x 0.55 m, so it drives that corner rather than turning on the spot — no route
  to this goal clears the 0.486 m a pirouette needs.

## Slide 8b — Six runs: which fusion rule, and what the box means

Start from the same detector noise for every camera — **one commissioned number, in pixels**.
Push it through that camera's own geometry and you get what *that* camera knows about the
robot's position.

```
sigma_px  ──►  R_pix = sigma_px^2 I        the detector's noise, in pixels
               ↓  through camera c's geometry (the hull Jacobian)
               Sigma_c = J_c^-1 R_pix J_c^-T    what camera c knows, in metres
```

> **Distance and viewing angle are already in there** — they are what changes the geometry.
> A far or badly-angled camera gets a big ellipse without anyone penalising it by hand.

So range and angle appear **once**, as the simple baseline, and never as a tuned term in the
proposed method.

| run | what the box means | how the cameras are combined |
|---|---|---|
| 1 | hull | the single best camera — the smallest ellipse |
| 2 | hull | weight by distance and viewing angle — the intuitive answer, untuned |
| 3 | hull | ordinary independent fusion — precisions add |
| **4** | **hull** | **the network as one sensor — precisions add, then divided by N** |
| 5 | the box bottom-centre *is* the robot | as run 4 |
| 6 | the box bottom-centre plus a fixed offset | as run 4 |

Runs 1–4 answer *which fusion rule*. Runs 4–6 answer *what a detector's box means*. Run 4 is
both, so it is six runs rather than eight.

The one sentence that carries run 4:

> The camera's ellipse says **how good that camera is**. Dividing by the number of cameras
> says **we will not claim five independent votes** when all five share a robot, a detector
> and a shelf layout.

Run 1 is the baseline that could end the story: *why fuse at all, if you can just use the
best camera?*

## Slide 8c — The plot to watch

**Error and *claimed* uncertainty, both against how many cameras are contributing at that
moment.** This is why the route was chosen to span none to four.

```
claimed sigma   independent fusion  ●───●───●───●     keeps shrinking like 1/N
                network fusion      ●───●──●──●       shrinks, but conservatively
actual error                        ●───●───●───●     does it shrink too?

                cameras watching:   1    2    3    4
```

**Measured, 2026-08-26, six drives, all six arms reached the goal.** From one camera to four,
the correction the network publishes:

| rule | claimed precision improves | actually improves |
|---|---|---|
| precisions add | **4.15x** | 1.10x |
| network, divided by N | **2.07x** | 1.11x |

> Adding cameras is worth about a tenth off the error. The standard rule claims four times the
> precision for it.

Downstream, the truth sits inside the belief's own 95% ellipse 57% of the time for the network
rule against 39% for precisions-add — while their median errors differ by 0.12 cm. **The rule
you pick barely moves where the robot thinks it is; it moves how much that belief deserves to
be trusted.**

Say the caveat in the same breath: **neither is honest.** 57% is not 95%. And the correction
itself lands ~8 cm from the truth while claiming 0.6–2.5 cm, against 1.49 cm measured on a
static robot — so the missing piece is heading error and motion, not the pooling rule. That is
a covariance-model finding, and it comes before the planner.

Say this before anyone asks how the arms are compared: **six live drives, one per arm, each
written up on its own before any of them are put on the same axes.** The arms do not see the
same detections — better fusion means a better belief, a better predicted box, and more
sightings admitted. That is not a flaw in the comparison, it is the thing being compared: an
arm that only wins when it is handed another arm's detections has not won anything a robot
could use.

One drive each is not a variance claim, and no figure here implies one. Each arm's path length
and how far it wandered from the commanded route are reported next to its error, so nobody has
to take "same route" on trust.

## Slide 8d — Two defects the drives found, and what they cost

**This is the slide that makes the rest defensible.** Everything measured before it was
measured through two faults, both found by asking why a camera reading was a metre wrong.

**1. The admission check was never called.** The four checks — tall enough, right width, contact
point where predicted, not touching the frame edge — exist, are unit-tested, have the same
thresholds commissioning used, and had **no caller in the runtime**. So a third of detections
that should have been refused became corrections.

Show `anomaly_investigations/01_what_a_metre_looks_like.png`: the robot behind a pallet, YOLO
finding a sliver of it, the box bottom sitting high, and the reading thrown a metre along the
viewing ray. Same cameras, clean views, 1–2 cm.

> Gated, those refused readings have a median error of 24 cm and a worst of 122 cm.
> The ones the check keeps: **1.44 cm**.

**2. Corrections were applied 400 ms stale.** A correction describes where the robot *was*. At
0.22 m/s that is 8.8 cm, and it is a **bias** — no covariance can represent it.

| what a published correction is worth | median | p95 |
|---|---|---|
| with both faults | 8.2 cm | 37 |
| carried to the time it is used | 2.1 cm | 33 |
| **and the check switched on** | **1.8 cm** | **9** |
| *commissioning, stationary robot* | *1.44 cm* | *9.2* |

> The sensor was always this good. The pipeline was applying it to the wrong instant, and
> fusing readings of a robot whose feet were hidden.

Two things to say out loud, because they are the honest half:

- **The check is not free.** It converts bad measurements into *absent* ones: 7% fewer
  corrections and a worst outage that grows from 5 s to 13 s. That is the availability side of
  the problem, and it is an argument for modelling availability rather than against the check.
- **A third fault was in the measurement itself.** Scoring a correction the filter is *holding*
  while the robot drives on reports the robot's travel as sensor error. It made a real 9 cm p95
  look like 151 cm. Corrections are scored once, when they land.

## Slide 9 — What would make this a contribution

**Show the criterion, not invented numbers.**

The paper succeeds if:

1. Learned availability predicts held-out usable observations better than constant and
   geometry-only models.
2. External observations measurably improve localization.
3. The planner uses that structure to reduce localization degradation and long observation
   outages.
4. The gain does not require an excessive detour.

> Planning with commissioned infrastructure-camera support improves localization-aware
> navigation compared with uniform, geometry-based, and previous reliability-to-covariance
> formulations.

## Slide 10 — Status and next steps

**Established**

- Detector trained and frozen; it finds the robot 93 % of the time on an independent
  capture, and fails where you would expect — 85 % beyond 20 m, 56 % when most of the robot
  is hidden.
- Independent commissioning data collected: five cameras over one shared set of positions.
- Successful detections locate the robot to a few centimetres.
- The dominant repeatable offset can be commissioned out — half a centimetre down to two
  millimetres, against 2 cm of ordinary scatter.
- **Camera support is clearly not uniform, and geometry alone does not predict it.** Across
  2 316 robot poses every single one had a clear line of sight to at least one camera, yet
  **18 % produced no usable sighting from any camera** and only 43 % produced two or more.
  A field-of-view model calls the building fully covered; it is not.
- **Only 29 % of the chances to see the robot become a usable measurement**: 53 % is lost to
  blocked lines of sight, 5 % to the detector missing, 12 % to sightings whose shape does
  not match what the robot should look like there.

**Next**

- Re-earn the fusion numbers on the repaired pipeline: every arm, every route, both faults out.
- Close the last gap between an honest measurement and an honest belief — the filter shrinks
  through a 1.7 cm systematic that does not shrink. A covariance floor along the direction of
  travel is built and awaiting its own comparison.
- Freeze the usable-observation definition, including the false-positive check.
- Seeds, so the honesty numbers get error bars.
- Then availability, and only then any planning.

> **No further detector or bias modelling unless one of these validation gates fails.**

---

# Backup slides

## B1 — Sensor calibration

The four bias figures live here, not in the main deck. For:
*"How do you know the camera measurement isn't biased?"*

Ready to show: the error across the floor before and after correction; error versus
distance; error versus which way the robot faces; the comparison of correction models.
Headline: 0.48 cm → 0.19 cm, against 2.2 cm of random scatter per sighting.

Also ready, if pressed on mechanism: the residual comes from mildly occluded sightings that
the admission checks let through, and those get commoner with distance — so it is a
property of this stock arrangement rather than of the cameras.

## B2 — What happened to detector confidence?

Expect this question. The answer is not "we dropped it."

**Previously:** confidence → reliability → the planner's assumed covariance.

**Now:** confidence is a property of a detection that has *already happened*. When planning
a route through a place the robot has not reached, there is no confidence value to consult.
So the two quantities are modelled separately: how likely an observation is at a future
place, and how accurate it is once received.

> We test whether confidence adds predictive value for the measurement covariance. If it
> does, it stays in fusion. If it does not, it remains in detection admission and as the
> previous-method baseline.

The data for this test is already collected — confidence is recorded for every commissioned
sighting.

## B3 — Why a camera network rather than one camera

Redundancy, complementary coverage, differing quality, and handover as the robot moves. The
run that has to earn this is **run 1 against run 4** — the single best camera against the
whole network, on the same 30.4 m route.

**Do not claim multiple cameras cancel bias.** Two cameras with offsets fuse to a weighted
combination of those offsets, which does not vanish. Measured here: a heading error is
shared by every camera, and fusing all of them removes 1 % of it.

## B4 — Known limits of the commissioning data

- Commissioned with the true robot pose; operation uses the robot's own estimate. Untested,
  and the largest open risk.
- False positives unmeasured — no pictures of the empty warehouse yet.
- Six robot headings, 60° apart.
- One stock arrangement; the picked-down warehouse is built but not yet photographed.
- Simulation only; no sim-to-real claim is made anywhere.
