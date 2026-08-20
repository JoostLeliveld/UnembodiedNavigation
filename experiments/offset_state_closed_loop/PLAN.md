# Experiment plan: can the robot work out that a camera is off, and does it help?

Design sketch, 2026-08-10. Not registered, nothing run yet.

> **REDESIGN REQUIRED before execution.** This draft assumes the historical-v2 77 mm signed
> Camera C residual is a current, identifiable camera offset. It is not: the source captures
> confound camera, route, region, yaw, and silhouette, and current balanced-IPM Camera C has
> +18.8 mm lateral bias with 66.6 mm mean measurement error. A valid closed-loop test must
> preregister an explicit injected fault or independently identify an offset on yaw/region-
> diverse training data, then freeze exact runs and scoring. See `docs/localization_metrics.md`.

Serves the open question left by `bayesian_filter_showcase`: the per-camera offset model
(A5) beat the covariance floor (A4) offline, but has never driven a robot.

---

## 1. The idea in plain words

Four cameras watch a robot drive around a warehouse. The robot cannot see itself; it adds up
wheel turns, which slowly goes wrong, and leans on the cameras to correct it.

One of the cameras is off. Not noisy — *off*, by about 7 cm, in the same direction every
time. Think of a bathroom scale that always reads 3 kg heavy. Weigh yourself a hundred times
and the readings barely move, so you conclude the scale is excellent and your weight is known
to the gram. You are 3 kg wrong and extremely sure of yourself.

That is what the robot currently does. It sees consistent readings, decides the camera is
precise, and shrinks its own claimed uncertainty. Then it plans a route through a 1.36 m gap
believing it knows its position to 2 cm, when it is 5 cm off.

The fix is not to trust the camera less. It is to work out *how much that camera reads off to
one side*, and subtract it. And you can only do that because the other cameras exist: if
three cameras agree and one says the robot is 7 cm further left, the robot is not in two
places, so the disagreement must belong to the camera.

**What this experiment has to show:** the robot works the offset out for itself while
driving, its stated uncertainty becomes honest, and it stops shaving shelves it thought it
had cleared.

---

## 2. What already exists, so we do not rebuild it

| piece | where | state |
|---|---|---|
| the world, 4 cameras, real routes | `warehouse_full_4cam.world.sdf`, `src/experiments/config/tasks.yaml` | built |
| the offset-state filter | `experiments/bayesian_filter_showcase/demo_state_space_model.py` | works offline |
| per-camera coverage models | `logs/visibility_comparison/spawn_grid_20260727/gp/camera_{A,B,C,D}/` | fitted |
| constant-offset fault injector | `reliability.calibration_perturbation.bias_camera_position` | built |
| calibration-drift fault injector (E6) | `reliability.calibration_perturbation.perturb_camera_calibration` | built |
| overhead media camera at `(0, 0, 26)` | in the world, excluded from localization | built |
| run replay renderer | `generate_run_replay.py` | built |

---

## 3. The two faults, and why both are needed

This is the part that decides whether the result is honest or flattering.

**Fault 1 — a flat sideways offset** (`bias_camera_position`). The camera's answer is shifted
by a fixed amount in the world, the same everywhere. This is *exactly* what the offset model
assumes. If the model cannot nail this, it is broken.

**Fault 2 — a drifted calibration** (`perturb_camera_calibration`, the established E6 fault).
Nudge the camera's real mounting angle by a fraction of a degree and re-project. Now the error
is small close to the camera and large far away. **The offset model's assumption is wrong
here** — there is no single offset to find, only an average over wherever the robot happened
to drive.

Run both. Fault 1 shows the mechanism working. Fault 2 shows where it runs out. A supervisor
will ask for the second one, and it is more interesting than the first.

**A third source of trouble, not injected — it is already there.** The robot's body is not
centred on its wheels; it sits 32 mm off. That offset *turns with the robot*. The cameras
cannot see which way the robot faces, so on a straight run this looks like a fixed sideways
error the model can absorb, and **on a route with corners it rotates, and the model's
"constant offset" assumption breaks by design**. This is why the route list below deliberately
includes both straight runs and runs with corners.

---

## 4. Three arms, one difference

Same world, same routes, same seeds, same planner, same detector. The only thing that changes
is how a camera sighting becomes a position update.

| arm | rule | what we expect |
|---|---|---|
| **N** — trust everything | one fixed trust level for every camera | confidently wrong; shaves gaps |
| **Q** — loosen the motion model | inflate the process noise until the belief is honest | honest, but less accurate |
| **F** — widen the uncertainty | never claim to be sharper than the camera's known error | safe but vague; drives timidly, wide berth |
| **O** — estimate the offset | track each camera's sideways error as part of the state | honest *and* sharp |

Arm N is what is normally done. Arm F is the current best result in this repo. Arm O is the
candidate.

### Arm Q exists because it nearly wins, and nobody had tried it

Added 2026-08-10 after a direct challenge: *could the offset model's advantage just be a
looser predict step?* Swept on all three offline captures, `exp1` protocol:

| arm N, process sigma | honesty | outside 95 % | error | claimed |
|---|---:|---:|---:|---:|
| 0.04 (deployed) | 4.22 | 41.9 % | 5.3 cm | 1.9 cm |
| 0.16 | 1.48 | 6.0 % | 5.4 cm | 3.4 cm |
| **0.32** | **1.09** | **1.8 %** | 5.6 cm | 4.4 cm |
| 0.64 | 0.91 | 1.3 % | 5.9 cm | 5.3 cm |

**Simply inflating the process noise makes the naive filter more honest than either proposed
method** (1.09 against an ideal 1.39, 1.8 % outside its ellipse, versus arm O's 1.89 / 2.5 %).
Honesty on its own is therefore NOT evidence for the offset model, and any write-up claiming
it is overstates the case.

What inflating the process noise cannot do is stay accurate: error rises monotonically
5.3 → 6.2 cm across the sweep, because it compensates for a bad camera by pretending the
robot's motion is unpredictable — throwing away odometry that was fine. At matched claimed
uncertainty (~3.3 cm) arm O is 5.4 → **4.8 cm** and less than half the unearned confidence
(6.0 % → 2.5 %).

Three checks say arm O's result is not a process-noise effect in disguise:

1. **Turn arm O's extra process noise off completely** (the offsets' slow drift → 0): honesty
   2.05, outside 95 % 2.7 %, error **4.8 cm**. Essentially unchanged.
2. **Crank that drift up** (0.0256): error degrades to **9.7 cm** — the opposite direction
   from the hypothesis.
3. **Sweep the position process noise under arm O** across 64×: honesty stays 1.7–2.9, error
   4.8–5.6 cm. A process-noise explanation predicts a large swing; there isn't one.

Structurally, the process noise describes how the *robot* moves — one value for one robot. It
cannot produce a separate number per camera. The quoted 67/36 mm recovered offsets and
77/33 mm references are all historical-v2, within-study values; they motivate the mechanism
but are not current camera constants.

**Consequence for this experiment:** arm Q must be tuned to match arm O's *claimed*
uncertainty, then compared on accuracy and on the tail. It is the strongest baseline, not arm
N. Reproduce with `experiments/offset_state_closed_loop/q_ablation.py`.

---

## 5. Order of work, cheapest first

Each phase can kill the idea before the expensive one runs.

### Phase 0 — does it survive the current pipeline? *(offline, minutes)*

The offline result was measured through pixel-to-floor maths that was deleted from the robot on
7 August. Re-run it with what the robot actually uses now.

**Stop if:** honesty falls apart. Then the offline result was an artifact of the old maths.

### Phase 1 — where can it possibly work? *(offline, existing data)*

The offset is only findable where **two or more cameras see the robot at once**. Build the map
of how many cameras cover each patch of floor, from the coverage models already fitted.

This picks the routes, and it sets an upper bound on the claim: if the overlap is thin, the
robot has few chances to learn.

**Stop if:** the routes never cross a two-camera region. Then no filter can do this and the
answer is "add a camera", not "change the filter".

### Phase 2 — put in a known error, see what comes out *(offline, recorded runs)*

Inject a lean we choose — 0, 25, 50, 100, 200 mm — on one camera, and ask what the filter
recovers. Repeat for both fault types and each of the four cameras.

This is where the evidence gets strong, because we know the true answer exactly instead of
comparing against a confounded historical residual.

**Stop if:** recovery is not roughly proportional to what went in. Then it is fitting noise.

### Phase 3 — drive it *(real Gazebo, expensive)*

Three arms × routes × seeds. This is the only phase that can say anything about clearance and
collisions.

**Blocked on a known bug:** the last closed-loop attempt (`clv2_pilot`, 6 August) produced
0 of 3 usable runs — 96 s of startup, then a "stuck" with the goal distance never falling, and
two runs dying as "interrupted". That has to be fixed first, and it is not a filter problem.

### Phase 4 — prove it is the network doing it *(real Gazebo, short)*

Repeat one route with **only the leaning camera switched on**. The offset should become
unfindable and arm O should collapse onto arm N.

If arm O still "recovers" an offset with one camera, something is wrong — it cannot know.

---

## 6. Routes, and why each one is in the list

All coordinates are real entries in `src/experiments/config/tasks.yaml`.

Camera geometry that matters: **A `(-6, -10)` and C `(+6, -10)` sit on the south wall looking
north. B `(-6, +10)` and D `(+6, +10)` sit on the north wall looking south.** So the south pair
owns the bottom of the map, the north pair owns the top, and they trade the robot somewhere in
the middle. Tall racks block differently for each pair: the west block's north racks are tall,
the east block's south racks are tall.

### Primary — `mc_central_ns`: `(0, −7.6) → (0, +7.6)`

Straight up the central aisle, 15.2 m. This is the money route because it does all three
things in one drive:

1. **starts where it can learn** — deep in the south pair's view, and at x = 0 both A and C
   can see it, which is the disagreement the offset needs;
2. **hands over in the middle** — the south pair runs out and the north pair picks up;
3. **squeezes a tight gap exactly there** — the central support pillar sits at `(0, −0.90)`
   and leaves two bypasses only **1.36 m wide**. A 7 cm error the robot does not know about
   is a real problem in a 1.36 m gap, and it happens at the worst possible moment.

It is also almost straight, so the robot's rotating body offset stays roughly constant. This
is the route where the model's assumption holds best — the fair "does the mechanism work"
test.

### Secondary — `mc_south_we`: `(−7.77, −7.5) → (+7.77, −7.5)`

Straight along the south perimeter, west to east, 15.5 m. Runs from A's territory into C's,
so it is a **west-to-east handover instead of south-to-north** — a second, independent chance
to see the same effect. It also passes below the tall east-south racks, which is where C's
view is broken up.

### The stress case — `mc_tour_L`: `(−7.77, −7.5) → (+7.77, −7.5) → (+7.77, +7.5)`

Two legs with a **90° corner**. The robot turns, so its body offset rotates, so the "constant
offset per camera" assumption is wrong for part of the run. Expect arm O to be *worse* here
than on the straight routes. That is the honest limit of the method and it should be measured,
not avoided.

### The hard case — `mc_m3_sw2ne_diag`: `(−10, −6) → (10, 6)`

Diagonal across the whole warehouse, through every coverage band and every handover. Constant
turning-free heading but crossing all four cameras. Good for the animation; hardest to
interpret.

### The control — a route that should show nothing

One route kept entirely inside a region several cameras cover well, with no tight gap.
**Prediction: all three arms tie.** Including a case where the method does not matter is what
separates a real claim from a cherry-picked one. Pick it from the Phase 1 coverage map.

### Which camera to break

Inject a preregistered fault into **camera C** first because the east-south tall racks give it
a broken-up view—not because it has a known current lean. Then inject the same fault into
**camera A**, which contributed zero sightings
in one of the offline recordings — a camera nobody sees through is a camera you cannot learn
about, and that should show up as "no answer" rather than "no offset".

Also worth one run: break a camera that **does not** cover the tight gap. Same injected error,
no consequence. Shows the effect is about *where* the bad camera looks, not about the size of
the number.

---

## 7. Plots

Eight figures. The first four are for someone who has never heard of a Kalman filter.

**P1 — the setup.** Top-down view of the warehouse. Four cameras marked with which way they
look, the racks, the route drawn as a line, the 1.36 m pillar gap called out. No data, just
"here is the situation".

**P2 — who can see you where.** The warehouse floor coloured by how many cameras have a view
of each patch. Immediately shows why the middle of the map is the interesting part, and where
the robot is on its own.

**P3 — the lie, made concrete.** A real frame from the broken camera at three points along the
route. On each: the detector's box, a dot for where the camera says the robot is, a dot for
where it really is, and the gap between them labelled in millimetres. This is the single most
useful picture for a non-expert — it turns "a lean" into something you can see.

**P4 — three robots, three stories.** The route drawn three times side by side. Truth as a
solid line. Each arm's belief as a dotted line with its uncertainty drawn as ellipses every
couple of metres. The reader sees it without being told: arm N's ellipses are tiny and the
truth is outside them; arm F's are big and slow; arm O's are small and the truth sits inside.

**P5 — watching it work the offset out.** Distance driven on the bottom, the estimated lean
for the broken camera up the side, with a shaded band for how sure it is, and a flat line for
the value we injected. It starts at zero, wobbles, narrows, and settles on the line. This is
the figure that answers "how can it possibly track that".

**P6 — what we put in against what came out.** Injected lean on the bottom, recovered lean up
the side, a diagonal for perfect. One series per camera, one panel per fault type. Fault 1
should sit on the diagonal. Fault 2 should not, and the gap is the finding.

**P7 — the part that matters.** Distance along the bottom, clearance to the nearest rack up
the side, one line per arm, zero marked as contact. Shade the stretch where the robot is
inside the pillar gap. This is where honesty stops being a statistic.

**P8 — take the network away.** P5 redrawn with only the broken camera switched on. The line
should never leave zero. Proves the claim depends on having other cameras and is not a
filtering trick.

Two rules for all of them, from `CLAUDE.md`: the title says what the figure shows, not the
variable names; and honesty is never shown without sharpness next to it, because a filter that
just claims to be unsure passes any honesty test and is useless.

---

## 8. Gazebo pictures and animations

The world already carries an overhead camera at `(0, 0, 26)` for exactly this, and it is
excluded from localization so using it cannot contaminate anything.

**A1 — side by side, naive against offset-estimating.** The same route, two panels, the
overhead view, with each robot's belief ellipse drawn on the floor as it drives. In the left
panel the ellipse is small and sitting in the wrong place; in the right it tracks. Then the
left one clips the pillar. Nothing explains this faster than watching it happen twice.

**A2 — the offset as an arrow.** Draw an arrow on the broken camera showing the filter's
current guess at its lean, with the arrow's fuzziness shrinking as the robot drives. Pair it
with P5 so the plot and the picture are the same story.

**A3 — the camera's own view, inset.** The broken camera's actual image in the corner with its
detection box, next to the overhead map. Makes clear that the picture looks completely normal
— the fault is invisible from inside that camera, which is the whole point.

**A4 — the handover moment, slowed down.** A few seconds around the middle of
`mc_central_ns`, where the south pair drops the robot and the north pair picks it up. Show
which cameras are contributing as it happens.

**A5 — one still for the thesis.** The overhead view at the tight gap, all three arms' beliefs
overlaid at the same instant, truth marked. One frame that contains the whole argument.

---

## 9. What gets measured

| what | how | why |
|---|---|---|
| honesty | how often the true position is outside the robot's own 95 % ellipse (should be 5 %) | the headline |
| sharpness | the uncertainty it claims, next to the error it has | stops "just be vague" from winning |
| one combined score | log score of the truth under the stated distribution | cannot be gamed by inflating or shrinking |
| accuracy | position error | **a control, not a target** — should stay flat |
| consequence | minimum clearance, contacts, goal reached | the only thing a robotics reviewer cares about |
| recovery | recovered lean against injected lean | shows the mechanism, not just the outcome |

**On accuracy being a control:** across six arms measured offline, position error varied only
1.12×, while honesty varied 11×. Changing which pixel gets projected to the floor moves
accuracy about 6×. So the filter is not what sets accuracy here, and if arm O wins on error it
is luck. Its job is to stay flat and prove the honesty was not bought by driving worse.

---

## 10. How this could fail

Written down in advance so the result cannot be reinterpreted afterwards.

- **Too little overlap.** If the routes rarely put two cameras on the robot, the offset never
  becomes findable. Phase 1 decides this before any Gazebo time is spent.
- **The corner case breaks it.** On `mc_tour_L` the rotating body offset is not a constant, so
  arm O may do no better than arm F. Expected; measure it and say so.
- **Fault 2 defeats it.** A drifted calibration is not a single offset, so the model can only
  find an average. If that average is useless, the honest statement is "this handles a
  consistently offset camera, not a drifting one".
- **It learns the wrong thing.** The offset can absorb anything systematic, including the
  robot's own shape seen from an angle. That is still useful, but it must be called "this
  camera reads off to one side" and not "this camera is miscalibrated".
- **A camera nobody sees through.** Camera A reported nothing in one offline recording. Its
  offset must come back as "unknown", not as zero.
- **Closed loop never runs.** Phase 3 is blocked on the campaign bug. Phases 0–2 and the plots
  from them stand on their own and are worth having regardless.
