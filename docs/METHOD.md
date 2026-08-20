# The method, start to finish

Plain-language walkthrough of what this system actually does, in the order it happens. No
registry IDs, no workstream numbers. Every term is defined where it first appears. If you
read one document to understand the thesis, read this one.

Detail lives elsewhere and is linked at each step. This file is the spine.

---

## The problem in one paragraph

A small robot drives around a warehouse. It cannot localize itself well on its own — all it
has is wheel odometry, which drifts without bound. Fixed cameras mounted high on the walls
watch the floor and can see the robot, so the robot can use those sightings to correct its
own position estimate. But cameras are not equally useful everywhere: shelves block the view,
accuracy degrades with distance and viewing angle, and a camera can be quietly miscalibrated.

So the question is not "can the robot see itself" but **"how good is this particular sighting,
and does knowing that change how the robot should drive?"** That is what the whole system is
built to answer.

---

## The cast

| thing | what it is |
|---|---|
| the robot | a TurtleBot3 Burger in Gazebo simulation |
| the cameras | four fixed cameras, `A`–`D`, mounted high and looking down at an angle, 1280×720 |
| the world | a warehouse with shelf racks; aisles are driveable, racks are not |
| ground truth | the simulator knows the robot's real pose. **The robot never sees it.** It is used only to score results afterwards |
| the belief | the robot's estimate of where it is: a mean position plus an uncertainty ellipse |

**"Belief" is the central object.** It is not just a position — it is a position *and a stated
uncertainty*. Most of this thesis is about whether that stated uncertainty is honest.

---

## Step 1 — The robot drives blind

Wheel encoders are integrated to give odometry. The simulator deliberately corrupts this
(`encoder_noise_node.py`), so the odometry drifts the way real odometry drifts. Left alone,
the robot's idea of where it is slides steadily away from the truth.

This is the baseline the cameras have to rescue.

---

## Evidence convention — read before the numbers

“Camera measurement error,” “belief error,” and “belief honesty” are different objects.
The canonical definitions, exact run IDs, current A–D comparison, and current-versus-historical
boundary live in [`localization_metrics.md`](localization_metrics.md).

In particular, **76.9/77/78 mm is not Camera C's current localization error**. It is a signed
cross-bearing bias measured under the retired v2 projection on three route/yaw-confounded
driving captures. The current fair A–D comparison uses floor-plane IPM on the same balanced
set-pose dataset: mean camera-measurement error is A 64.6, B 68.1, C 66.6, D 67.1 mm.

## Step 2 — A camera sees the robot

Each camera image goes through a YOLO object detector, which returns a **bounding box** around
the robot in pixel coordinates.

Worth knowing: the detector is *not* the weak link. Measured on 1844 real detections, every
box edge landed within ±0.34 pixels of its true edge, and detection succeeded 99.7% of the
time. Deleting the detector entirely and using perfect boxes changes the final position error
by 0 mm. Whenever something goes wrong downstream, it is not the detector.

---

## Step 3 — Turning a pixel into a place on the floor

Take the **bottom-centre pixel of the box** — roughly where the robot touches the floor. Shoot
a ray from the camera through that pixel. Find where the ray hits the floor plane. That point
is the measured position of the robot in warehouse coordinates.

That is it. This is called **inverse perspective mapping (IPM)**, and it has **zero tunable
parameters**. One line of code: `camera.pixel_to_world(u, v)`.

**Why zero parameters, when we could fit corrections?** Because we tried, and fitting made it
worse. Scored on the same 1844 detections:

| approach | parameters fitted | mean error |
|---|---|---|
| **plain IPM** | **0** | **66.6 mm** |
| fitted correction v2 | 8 | 68.2 mm |
| fitted correction v4 | 2 | 70.1 mm |
| fitted correction v3 | 10 | 74.5 mm |

Every fitted correction lost to applying none. The reason is instructive: the corrections were
fitted in one region of the warehouse and did not transfer to another. One of them *inverted*
the very bias it existed to remove — camera C's error went from +18.8 mm to −58.7 mm after
"correcting" it. So all twelve fitted parameters were deleted from the runtime on 2026-08-07.

**Where the remaining 66.6 mm comes from**, since this matters later: the robot's visual body
is not centred on its wheels — it sits 32 mm off. That offset *rotates as the robot turns*, so
the centre of the silhouette a camera sees is not a fixed point on the robot. The camera cannot
see which way the robot is facing, so it cannot correct for this. It is a real, irreducible
geometric limit, not sloppiness.

Code: [`projection.py`](../src/reliability/reliability/projection.py). Evidence:
`logs/studies/pixel_ground_path/`.

---

## Step 4 — How much should the robot trust that sighting?

The filter needs more than a point; it needs a **covariance** — a 2×2 matrix describing the
size and shape of the error ellipse around that point. Call it `R`. Big `R` means "don't lean
on this sighting much".

Two things build it:

**4a. Pixel uncertainty → metre uncertainty.** A sighting's uncertainty starts in pixels
(`R_uv`, units px²) and is pushed through the geometry of the projection:

```
R_xy  =  J · R_uv · Jᵀ
```

`J` is the Jacobian — how many metres on the floor one pixel of image is worth, computed
numerically. This matters a lot, because the same pixel error is worth far more metres at 15 m
range and a shallow viewing angle than at 2 m looking almost straight down. **So the same
detector confidence produces very different ellipses depending on where the robot is.** That
state-dependence is the honest core of the geometry.

**4b. What sets the pixel uncertainty in the first place?** This is the contribution point of
the thesis. The system interpolates between two endpoints — "the camera sees this spot well"
(2.5 px) and "the camera probably cannot see this spot at all" (a much larger number) —
according to a predicted **trust** that the sighting is usable. How that trust should be
predicted (from learned models, from geometry, or from nothing at all) is what the reliability
experiments compare.

Code: `project_observation_to_world_with_covariance` in
[`projection.py`](../src/reliability/reliability/projection.py), and
`covariance_mapping.trust_to_update_covariance`.

---

## Step 5 — Four cameras, one answer

`camera_manager_node` gathers whatever the four cameras reported, decides which sightings to
use and how to combine them, and publishes a single position-with-covariance in warehouse
coordinates.

Having four cameras is not just redundancy — it is the only reason a lying camera is
detectable at all. A camera cannot notice its own bias. Three others disagreeing with it can.
That fact drives Step 7.

Code: [`camera_manager_node.py`](../src/reliability/reliability/nodes/camera_manager_node.py).

---

## Step 6 — The Bayesian filter

This is where odometry and cameras get combined into the belief. It is an **Extended Kalman
Filter (EKF)**, which alternates two moves:

**Predict** — "I drove for 0.1 s, so where am I now, and how much less sure am I?" A unicycle
motion model advances the mean; the covariance grows by `Q`, the **process noise**, which
encodes how fast the belief should spread while driving blind. The system uses the exact
integrated form `Q_d(θ, v, Δt)` rather than a frozen constant.

One structural point that surprises people: **heading is pure dead reckoning.** A single point
on the floor tells you nothing about which way the robot is facing, so no camera ever corrects
heading directly. Heading is only nudged indirectly, through the position–heading correlation
that the motion model builds up. This is realistic — real external-camera deployments have the
same limitation — and it means heading drift in camera-poor regions is a genuine failure mode
the experiments are designed to expose.

**Update** — "a camera says I am *here*; how much should I move my estimate?" The standard
Kalman update, weighting the camera against the prediction by their relative covariances.

Before any update is allowed in, it passes a chain of gates, in this exact order:

| gate | if it fires |
|---|---|
| measurement too old | reject |
| implausible time step | reject |
| no belief snapshot to update | reject |
| the update maths failed | reject |
| **the belief has diverged** | **re-anchor** — snap to the measurement, because the *belief* is what is wrong |
| position jump too large | reject |
| **NIS** too large (the sighting disagrees wildly with the prediction) | reject |

The divergence check deliberately sits *ahead* of the jump and NIS gates: a badly diverged
belief trips both, and rejecting on either would lock the robot out of ever recovering.

And a rejection never freezes the belief — the covariance keeps inflating, so a robot that
rejects everything becomes visibly less sure rather than confidently lost.

Code: [`belief_correction.py`](../src/planning/planning/core/belief_correction.py) (the gate
chain, shared by both the single- and multi-camera paths so they cannot drift apart),
[`unicycle_planner_node.py`](../src/planning/planning/nodes/unicycle_planner_node.py) (the ROS
node). Detail: [`uncertainty_propagation.md`](uncertainty_propagation.md).

---

## Step 7 — The finding: honest uncertainty is harder than accurate position

Everything above assumes each camera's error is **fresh random noise every frame**. It is not.
A miscalibrated camera has a *lean* — the same offset, frame after frame.

That breaks the Kalman filter in a specific way. The filter treats 100 sightings from one
camera as 100 independent votes, so it shrinks that camera's uncertainty like `1/N` toward
zero. But the bias does not shrink at all. **The stated uncertainty collapses while the real
error stays put.**

Measured in the locked **historical v2 belief study** (1,424 update steps over the three
named July captures), the filter states a mean **1.9 cm** 1σ uncertainty while its belief has
**5.3 cm RMSE**, and the truth falls outside its stated 95% ellipse **41.9% of the time**.
That study's largest repeated input residual was Camera C's historical +76.9 mm signed
cross-bearing bias. The filter mechanism remains valid; the 76.9 mm magnitude and camera
ranking are not current-runtime claims.

Four fixes were tried. **All four failed, each for a different and useful reason:**

1. **A sharper per-camera noise model made it worse.** Measuring each camera's noise properly
   tightens the belief — but the historical study's dominant residual is a lean, not noise. Better noise modelling
   plus an unmodelled bias equals *more confident and equally wrong*.
2. **A chi-square outlier gate did nothing** — it rejected 0.2% of updates. That repeated offset sits
   comfortably inside a 95% gate. Classical robust filtering is built for wild outliers, not
   for a camera that is quietly always slightly wrong.
3. **Health-checking each camera against the fused belief made it much worse.** This one is
   backwards in an instructive way: a biased source *captures* the belief, after which its
   own sightings look perfectly consistent and the honest cameras look broken.
4. **Checking each camera against a belief built without that camera fixed the direction but
   not the magnitude.** Judging a camera against a belief that excludes that camera is the
   correct comparison — and it is only possible because other cameras exist. Necessary, not
   sufficient.

Within that historical-input study, what works is to add a **floor** to each camera's covariance that repeated looks cannot shrink
below — because no per-frame noise model, however well calibrated, can represent an error that
is the same every frame. Accuracy stays similar (belief RMSE 5.3 → 5.0 cm), while the
outside-ellipse rate falls 41.9% → 3.3%. Median NEES 0.46 versus the calibrated 2-D reference
1.386 shows the result is conservative. The mean 5.1 cm RMS per-axis 1σ is a sharpness
descriptor, not a quantity that should equal the 5.0 cm radial RMSE.

Evidence: `experiments/bayesian_filter_showcase/`, figures in [`../figures/EXP-BELIEF/`](../figures/EXP-BELIEF/).

---

## Step 8 — What the belief is actually *for*

The belief is not the output. It feeds the planner, and it changes where the robot drives in
two ways:

- **Wider berth when unsure.** Obstacle clearance is computed as
  `clearance − κ · σ_max(belief)`. A fatter uncertainty ellipse shrinks the effective free
  space, so an unsure robot gives racks more room.
- **Route choice.** The planner rolls the belief forward over a long horizon and prefers routes
  where it predicts it will stay well-localized. If the covariance model says a stretch of
  aisle is camera-poor, that stretch looks expensive *before* the robot commits to it.

This is why honest uncertainty is worth the trouble: an overconfident belief does not just
misreport, it plans as though camera-poor regions were safe.

Detail: [`uncertainty_propagation.md`](uncertainty_propagation.md) §6–§7.

---

## Glossary

Read this once and most of the rest of the repo decodes.

| term | plain meaning |
|---|---|
| **belief** | the estimate: mean position + uncertainty ellipse |
| **EKF** | Extended Kalman Filter — the predict/update loop that maintains the belief |
| **IPM** | inverse perspective mapping: shoot a ray through a pixel, intersect the floor |
| **`R`** | measurement covariance — how much the filter distrusts one camera sighting |
| **`Q`** | process noise — how fast the belief spreads while driving on odometry alone |
| **`J`** | Jacobian — here, how many metres of floor one pixel of image is worth |
| **NIS** | "does this incoming sighting agree with my prediction?" Used as an outlier gate |
| **NEES** | "is my stated uncertainty the right size?" Squared error divided by stated variance. For a 2-D position an honest filter sits near **1.39**; higher means overconfident, lower means needlessly vague |
| **unearned confidence** | how often the truth falls outside the robot's own stated 95% ellipse. Should be 5%. |
| **bias / lean** | a repeated, same-direction error — the thing a Kalman filter cannot handle |
| **process noise vs. simulation noise** | `Q` is the robot's *model* of disturbance; the simulator injects the *real* disturbance. They are deliberately not equal — one models the other |

---

## Reading the numbers honestly

Two habits, because both have burned this project:

- **A stated uncertainty and an accuracy are different claims.** "5 cm error" and "honestly
  reports 5 cm of error" are separate results, and a filter can pass one while failing the
  other badly. Always look at them together — the figures deliberately put them side by side,
  because a filter that just widens its ellipse passes any honesty test and is useless.
- **Every number is conditioned on a pipeline version.** The Step 7 numbers were measured with
  the *old* fitted projection from Step 3, which has since been deleted. The mechanism (a
  repeated bias defeats a per-frame noise model) is unaffected — but the specific magnitudes,
  including which camera is the liar, have not been re-measured under plain IPM. Check what a
  number was measured on before quoting it.

---

## What is settled and what is not

**Settled and measured:**
- Pixel-to-ground is plain IPM at 66.6 mm, and fitting corrections makes it worse.
- The detector is not a limiting factor.
- A repeated per-camera bias makes the filter overconfident, and no per-frame `R` can fix it.
- A covariance floor plus leave-one-camera-out checking restores honesty at no accuracy cost —
  **offline**.

**Not settled:**
- The covariance floor **is not in the runtime.** It exists only in the offline study, so no
  closed-loop run has ever used it.
- Its magnitudes come from the deleted projection path and need re-measuring under plain IPM.
- The floor is *conservative* rather than exactly calibrated — it errs toward vagueness.
- Occlusion is untested: every sample behind the Step 3 and Step 7 numbers had a clear view.
- The closed-loop campaign that would tie this to navigation outcomes has not produced a usable
  run yet.
