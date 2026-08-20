# Storyline: a camera that is wrong the same way every time

**Notebook:** `camera_localisation_from_scratch.ipynb`
**Written for a reader with no prior knowledge of this project, this repo, or this thesis.**
No registry IDs, no arm codes, no acronyms that are not defined on first use.

---

## The one-sentence version

> A wall camera can tell a warehouse robot where it is, but it is wrong by about half a
> robot width in the same direction every time; that error cannot be averaged away and
> cannot be hidden inside the robot's uncertainty; so it has to be estimated — and whether
> it *can* be estimated depends on how the robot drives.

---

## The rules this notebook plays by

These are the assumptions, and they are chosen to be ones a real warehouse could actually
satisfy. Anything the notebook does that a warehouse could not do is labelled where it
happens.

| the system may use, at run time | the system may **not** use, at run time |
|---|---|
| the camera image | ground truth of any kind |
| where the camera is bolted and its lens spec | the robot's CAD model or mesh |
| wheel odometry from the robot | any measured-offline correction table |
| the warehouse floorplan | anything fitted on the drive being filtered |

Ground truth appears **only** to score results. The robot's exact 3-D model appears **only**
in Part 1, once, to explain to the reader *why* the error exists — it is never available to
the estimator, and every figure that uses it says so on its face.

---

## Part 0 — Why a robot needs a camera at all

*Nothing assumed. Start from a robot in a room.*

**0.1 A robot that counts its wheels.**
A warehouse robot tracks itself by counting wheel rotations. Every slip, every uneven
floor tile, every turn adds a small error, and nothing ever takes it back off. The error
only grows.

- **Figure 0.1** — one real drive: where the wheels say the robot is, where it actually
  is, and the gap between them opening up over nine metres. Two lines, gap annotated in
  centimetres.

**0.2 A camera can see the robot.**
Warehouses already have cameras on the walls. A camera does not drift — it looks at the
floor and the robot is either there or it is not.

- **Figure 0.2** — a real frame from the wall camera, the whole warehouse floor in view,
  the robot a small object in it. Plus a plan view of where the camera is and what it can
  see.

**0.3 Turning a picture into a position.**
The camera does not output a position. It outputs an image. Three steps get you a number:
a detector draws a box around the robot; take the bottom-centre of that box; shoot a ray
through that pixel and see where it hits the floor.

- **Figure 0.3** — the same real frame with the detector's box drawn, the bottom-centre
  pixel marked, and beside it the side-on geometry of the ray meeting the floor.

*Reader now has: a robot that drifts, a camera that does not, and a recipe for turning one
into the other. Everything so far looks like it works.*

---

## Part 1 — The camera is wrong, and always in the same direction

**1.1 Check it.**
Score the camera's answer against the truth the simulator knows.

- **Figure 1.1 (the hook)** — every camera-reported position minus the true position,
  drawn as arrows. They are about 9 cm long and they nearly all point the same way. This
  is not scatter. This is a lean.

**1.2 Split it into the part that repeats and the part that does not.**
About nine tenths of the error is the same every frame; one tenth is genuine randomness.

- **Figure 1.2** — the same errors, decomposed: the mean arrow, and the small cloud
  around it.

**1.3 Averaging cannot fix it.**
Randomness shrinks when you average — a hundred looks are ten times better than one. A
lean does not shrink at all, because you are adding up the same mistake.

- **Figure 1.3** — error against number of sightings averaged. The scatter falls away as
  one over the square root; the lean sits flat. Two curves, one of which never comes down.

**1.4 Where the lean comes from.** ⚠️ *explanation only*
The camera looks down at an angle. The lowest pixel of the robot in the picture is not the
point where its wheels touch the floor — it is the bottom edge of its outline, which for a
tilted view sits somewhere on the robot's far side. You are back-projecting the wrong point
on the robot.

- **Figure 1.4** — side-on diagram of exactly this, plus a real frame with the robot's
  true 3-D outline drawn over it so the reader can see the gap with their own eyes.
- **Callout box, unmissable:** *this figure used the robot's exact CAD model, which a real
  warehouse does not have. It is here to explain the mechanism. Nothing after this point
  uses it.*

*Reader now has: the camera is biased, not noisy; the bias is geometric, not mysterious;
and the obvious fix — average more — is ruled out.*

---

## Part 2 — Why you cannot hide it in the uncertainty

*The Bayesian core. This part has to be airtight, because everything after it depends on
the fix being genuinely unavailable.*

**2.1 What a belief is.**
A position estimate on its own is not enough for a robot that has to avoid shelves. It
needs a position *and* how sure it is — an ellipse, not a point.

- **Figure 2.1** — one update drawn from real data: what the robot believed before, what
  the camera said, what it believed after. Three ellipses and an arrow.

**2.2 The assumption the filter makes.**
A Kalman filter treats every sighting as an independent vote. Ten votes are better than
one, so it shrinks its ellipse. That is correct when the errors are independent. A lean is
the same vote ten times, and the ellipse shrinks anyway.

- **Figure 2.2** — the stated uncertainty falling steadily while the actual error stays
  flat. The two lines cross, and after they cross the robot is confidently wrong.

**2.3 Measuring dishonesty.**
If the ellipse is right, the truth should sit inside the 95% ellipse about 95% of the time.
Measure how often it actually does.

- **Figure 2.3** — coverage, stated uncertainty and actual error, side by side. *(House
  rule: honesty and sharpness always appear together — a filter that just draws a huge
  ellipse passes any honesty test and is useless.)*

**2.4 The tempting fix, and why it is not one.**
The obvious response is to make the ellipse bigger until the numbers look honest.

- **Table 2.4** — no inflation, ×2, ×5, ×10. Position error does not improve in any of
  them. Only the admission changes.
- **The sentence this part exists to earn:** *inflating the uncertainty buys you an
  admission of being wrong. The robot is still in the wrong place; it just says so now.*

---

## Part 3 — Estimate the lean

**3.1 The idea.**
It cannot be predicted, because that needs the robot's shape and a warehouse has not got
it. It cannot be hidden in the uncertainty, because Part 2. So estimate it: give the filter
one more thing to track, alongside the robot's position — the camera's lean.

- **Figure 3.1** — what the filter now carries, drawn: a position, and an arrow for the
  camera's lean, each with its own uncertainty.
- **Input ledger, restated on the figure:** the pixel, the camera's mounting, wheel
  odometry. No ground truth, no robot model.

**3.2 Does it work?**
Run it on the recorded drives and see whether the estimated lean lands on the real one.

- **Figure 3.2** — the estimated lean converging over the drive, against the real lean
  measured from ground truth *afterwards*, and against what geometry says it should be
  (the CAD prediction from Part 1, used here purely as a yardstick).
- **Table 3.2** — accuracy and honesty, against every arm from Part 2.

**3.3 The catch, and the actual contribution.**
"The camera leans 9 cm north" and "the robot is 9 cm south of where it thinks" produce
*exactly the same data*. Nothing in the measurements separates them. If the robot drives in
a straight line and is only ever seen from one direction, the two cannot be told apart and
the estimate is arbitrary.

What breaks the tie is **angular diversity**: the robot being seen from different bearings,
or turning so its own outline presents differently. Then the two explanations predict
different things and the data can choose.

- **Figure 3.3a** — the confounding, drawn: two completely different worlds, identical
  measurements.
- **Figure 3.3b** — how well the lean is recovered, against how much angular diversity each
  drive contains. The straight aisle drives should do badly; the diagonals, the corner and
  the arc should do well. *This is the plot the paper is built on.*

---

## Part 4 — What a real warehouse throws at it

*Everything before this ran at a walking-pace crawl on clean, unobstructed routes. Two
things a real warehouse has that those drives do not.*

**4.1 Speed.** — *new Gazebo captures*
Every drive so far ran at 0.15 m/s. A real warehouse robot cruises at 1 to 1.5 m/s. The
detector runs at a fixed 5 frames a second regardless, so speed changes how far the robot
travels between looks.

- Captured: 0.15, 0.50, 1.00, 1.50 m/s on the same route, plus 0.50 and 1.00 on a second.
- **Figure 4.1** — sightings per metre driven, and the gap between consecutive sightings,
  against speed. At 1.5 m/s the robot moves further than its own body length between looks.
- **Figure 4.2** — what that does to the lean estimate: less data, and the position moves
  more between updates.
- **Stated limit:** the simulator renders instantaneous frames, so motion blur is exactly
  zero here. On real hardware it would not be, and at 1.5 m/s it would matter. This is
  named as a gap, not measured.

**4.2 Occlusion that displaces rather than removes.** — *new Gazebo captures*
Occlusion is normally treated as a coverage problem: the camera sees the robot or it does
not. That is the harmless half. The dangerous half is **partial** occlusion — a rack hides
where the robot meets the floor while its top is still visible, so a detection still
arrives, the box bottom is now the bottom of the *visible* part, and the measurement is
displaced with nothing in the data to say so.

- Captured: a route chosen by ray-testing the camera against every shelf in the warehouse,
  which holds the robot in exactly that state for about a third of the drive while never
  losing it entirely. Plus the same line driven back, and two mixed routes that lose the
  robot repeatedly as well.
- **Figure 4.3** — real frames from the grazing drive: the same robot, clean and
  half-hidden, with the displacement between them.
- **Figure 4.4** — measurement error split three ways by what the camera could actually
  see: fully visible, contact point hidden, nothing visible.
- **Figure 4.5** — what partial occlusion does to the lean estimate. It is a step change
  the filter has no way of seeing, and that is the honest failure mode.

---

## Part 5 — What is honest to claim

**Demonstrated.** The camera's error is a lean, not noise. It cannot be averaged away and
cannot be covered by widening the uncertainty. It can be estimated online with nothing but
the camera and wheel odometry — but only when the robot's motion supplies enough angular
diversity, and that is a property of the route, not of the estimator.

**Zero by construction, and therefore untested here.**
- **camera calibration error** — the camera model is read from the same world file the
  simulator renders from, so it is exactly right. On real hardware this would be the
  *largest* error source: half a degree of pointing error is 22 cm of floor at ten metres,
  against the 9 cm lean this whole notebook is about.
- **lens distortion** — no distortion term in the sensor.
- **motion blur** — instantaneous frames, no shutter.

**Also unaddressed.** One camera, one warehouse, one robot, one detector. Simulation only.

**What a warehouse should take from it.** Do not buy a better camera and do not widen the
uncertainty. Give the robot a route with some angular variety near each camera and let it
work its own lean out.

---

## Build order

| step | what | state |
|---|---|---|
| 1 | new Gazebo captures: speed sweep, 5 runs | **done** |
| 2 | new Gazebo captures: occlusion set, 4 runs | **done** |
| 3 | visibility labelling from the world SDF (`check_route_clearance.py`, `story_model.visibility_at`) | **done** |
| 4 | Parts 0–2 — the problem and why the obvious fixes fail | **done** |
| 5 | Part 3 — the lean as a state, and the identifiability result | **done** |
| 6 | Part 4 — speed and occlusion, on the new captures | **done** |
| 7 | execute end to end, so the notebook ships with its figures | **done** |

---

## What changed once the data was in

Three hypotheses in the plan above were tested and did not survive. They are recorded here
because the corrected versions are what the notebook actually claims.

**The lean does not ride with the robot's body.** The plan assumed a body-frame lean would
be the physically right model. Measured across nine drives, the body-frame lean varies MORE
between drives (3.7, 4.2 cm) than the world-frame one does (2.1, 1.2 cm). Refuted. What it
actually depends on is the ANGLE between the robot's heading and the camera's sightline:
conditioning on that single number cuts the unexplained spread from 4.26 cm to 1.35 cm,
where no fixed frame gets below 2.1 cm.

**A sightline-frame lean state is worse, not better.** Built it, swept its one parameter
over a 32x range, and it stays at 5.8 cm against 1.9 cm for the plain world-frame lean
state. The world-frame lean state is the method; the elaboration was not earned.

**Partial occlusion mostly does not produce a displaced measurement.** The plan called this
"the dangerous half" and designed a route to maximise it. Measured over every message,
misses included: the detector finds the robot 100% of the time when its contact point is
visible and 7% of the time when it is not. Partial occlusion behaves almost exactly like
total occlusion -- the detector fails rather than lying. The dangerous case is real but
rare. The useful finding that replaced it: the warehouse floorplan predicts camera coverage
almost perfectly, with no learning and no robot model.
