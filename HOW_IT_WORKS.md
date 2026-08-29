# How it works, start to finish

The whole chain: what the world is assumed to do, what was collected, what is measured once,
and what happens every frame while the robot drives. Companion to [`PLAN.md`](PLAN.md),
which holds what the paper has to earn.

---

## 1. The generative model — what the world is assumed to do

Everything else in this document is either *fitting* this model or *inverting* it.

For a robot at pose `(x, y, θ)` and a camera `c`:

**Step one — does a sighting happen at all?**

> With probability `p_c(x, y, θ)`, camera `c` produces a usable sighting. Otherwise it
> produces nothing.

That single probability covers three separate ways of getting nothing: racks block the line
of sight, the detector fails to fire, or the box it draws does not match the robot well
enough to be trusted. They are collapsed into one number because from the planner's point of
view they are the same event — no correction arrived.

**Step two — if a sighting happens, what does it say?**

> The detector's box bottom-centre lands at
>
> &nbsp;&nbsp;&nbsp;&nbsp;`z = h_c(x, y, θ)  +  b(range)  +  noise`
>
> where `h_c` projects the robot's own shape and takes its bounding box's bottom-centre,
> `b` is a small commissioned offset, and the noise is zero-mean, the same size in every
> direction, **and measured in pixels**.

**That is the entire model.** Note what lives where:

| part | what it is | numbers fitted |
|---|---|---|
| `h_c` | the robot's shape, projected. Pure geometry. | **0** |
| `b(range)` | the leftover lean | **6** |
| noise | one spread, in pixels | **1** |
| `p_c(x, y, θ)` | the availability field | a field, fitted separately |

**Seven numbers, plus a map.** Everything else is the robot's description and the camera
mounts.

### Why the noise is in pixels, and what follows from that

The detector makes its mistakes in the image, so that is where its error is described. The
error on the *floor* is then a consequence, not a separate thing to fit: near a camera one
pixel is worth a few millimetres, far away it is worth centimetres, and along the viewing ray
it is worth more than across it. The geometry supplies all of that.

So there is no such thing as "fitting the covariance in world units" here. There is one
pixel number, and the world-space ellipse is what comes out when it is pushed through the
camera. **Section 6** is about how that number is estimated and why nothing more elaborate
earned its place.

---

## 2. Every assumption, stated

Marked **tested** where it was actually checked, **assumed** where it was not.

| | assumption | status |
|---|---|---|
| A1 | The floor is flat and at a known height; the robot stands on it. | assumed |
| A2 | Camera positions, angles and lenses are known and do not move. | **tested** — rebuilt cameras reproduce the sealed predictions to 0.03 px, and the run refuses to continue otherwise |
| A3 | The robot's shape is known from its description file and is rigid. | **tested** — the projected shape matches the rendered robot to 0.006 px on unoccluded sightings |
| A4 | There is one robot, so a detection cannot be confused with another. | assumed (true by construction here) |
| A5 | The detector is frozen and never retrained after any of this is measured. | **enforced** — weights hashed into the frozen record |
| A6 | Pixel noise is zero-mean after the offset, and the same size in every direction. | **partly tested** — 0.76 px sideways, 0.74 up-down |
| A7 | Pixel noise is independent between sightings and between cameras. | **weakly tested** — cross-camera correlation 0.17 measured, but with the true pose, so it excludes the shared heading error |
| A8 | The heading used in `h` is good enough to treat as known. | **NOT tested** — this is the open gate |
| A9 | The stock arrangement is fixed. | assumed, and it matters: both the offset and the availability map depend on it |
| A10 | Simulation only. | stated — no claim about real cameras is made anywhere |

**A8 is the one to worry about**, and A7 depends on it: heading error is shared by every
camera, so if it is large the independence assumption fails in the one way that fusion cannot
survive.

---

## 3. The data

| dataset | what it is | what it is for |
|---|---|---|
| `warehouse_v2_yolo_20260821` | 32 × 26 floor positions × 4 headings, per camera | trained the detector — **never used for anything else** |
| `warehouse_v2_yolo_shared_20260822` | 33 × 28 positions × 6 headings, all five cameras over the **same** positions | everything below |

The second gives **11 585 trials** — one per (camera, robot placement). 5 407 had a clear
line of sight, 4 797 produced a detection, **3 351 produced a usable sighting**.

Those 386 floor positions are split, disjointly:

- **20 positions** for the offset, because it needs ground truth and converges by about ten;
- **366 positions** for the availability map, because it needs no truth and was still
  improving at 250.

**Still missing:** any driven trajectory, pictures of the empty warehouse (so false positives
are unmeasured), and the picked-down stock state.

---

## 4. Commissioning — measured once, then frozen

**Check the cameras.** Rebuild them from the capture manifests and confirm they reproduce
predictions already sealed into the data. Refuse to continue otherwise.

**Measure the shape offset with the detector absent.** Compare the projected shape against
the simulator's own segmentation masks. Under 0.03 px — the shape is right, and this is
established without the detector being involved at all.

**Characterize the detector.** Detection rate by distance, by how much of the robot is
visible, at the frame edge. Reported, never tuned.

**Fit the offset.** 20 positions, six numbers. 0.53 cm → 0.29 cm on 313 positions it never
saw.

**Fit the noise.** One number: 0.764 px.

**Build the availability map.** Not yet done — it is the next step, on its own 366 positions.

---

## 5. Runtime — what happens every frame

### The belief

The robot carries a position, a heading, and one joint covariance. The camera measurement
contains position only. In the repaired fusion campaign, a position correction can also
update heading indirectly through position-heading cross-covariance (`coupled` mode); the
camera never publishes or observes heading directly. A separate `camera_xy_only` mode
anchors heading to odometry and removes the corresponding cross-covariances. Planning and
recursive filtering use the same selected mode.

### When a camera frame arrives

```
the robot's current guess (x, y) and heading θ
        │
        ├── project the robot's shape at that guess  ──► predicted box bottom-centre
        │
YOLO ───┴── the box it actually drew  ─────────────► measured box bottom-centre
                                │
                   they disagree by some pixels
                                │
              how many centimetres is a pixel worth here?
              (nudge the guess, re-project, see how far it moved)
                                │
                   move the guess by that much
```

**The measurement is never turned into a position.** Only the *disagreement* is converted.
That is what makes the 30 cm box-versus-centre gap harmless: it is present in the prediction
and in the measurement, so it cancels.

Demonstrated on a real sighting: put the guess 20 cm wrong and one step brings it to 0.7 cm.
Put it **2 m** wrong and one step brings it to 2.4 cm.

**The runtime already works this way.** Its single-camera path is described in the code as
`z = (u, v) pixels, h(x) nonlinear via the camera model` — the same space the generative
model lives in. The change the new method makes is small: where the old method supplied a
covariance that varied from a learned reliability score, the new one supplies a constant, in
pixels.

### The checks a sighting must pass

Before a sighting is allowed to move the belief:

1. it must look like the robot should look there — right height, right width, bottom edge in
   the right place, not cut off by the frame edge;
2. it must be recent enough, and the time step plausible;
3. the correction it implies must not be an implausible jump;
4. it must not disagree with the belief by more than the stated uncertainty allows.

Every one of these compares the detection against a prediction, so all of them run without
ground truth. **They fail safe:** corrupt the pose they predict from and the rate of
sightings wrongly *kept* stays near 1 % while wrongly *dropped* climbs to 15 %. The system
goes quiet rather than converging to the wrong place.

### Obstacle avoidance, and how it couples

Obstacle avoidance is **separate machinery** — a penalty on getting closer than 0.35 m to any
rack, computed from the warehouse map. It does not involve the cameras.

But it reads the **belief**, not the truth. So:

> **Localization error eats directly into clearance.** A belief 5 cm off means the planner
> thinks it has 0.35 m of margin while it really has 0.30 m.

That is the honest reason localization quality matters for safety and not only for
tidiness — and it is the argument for why the planner should care where corrections are
available.

---

## 6. Fitting the noise, and how it relates to everything else

### What is being fitted

One number: the spread of the pixel residuals, after the offset is removed. That is the
`noise` term of the generative model in section 1 and nothing else.

### Why it does not need to vary with position

Because the model says the noise is a property of the **detector**, and the geometry supplies
everything that depends on where the robot is. Measured, that holds: the sideways spread is
flat to within 10 % across a fourfold change in distance, while the error on the floor grows
fivefold over the same range.

Pushed through the camera, one number produces the right ellipse everywhere — 1.2 cm near a
camera, 4.0 cm far, stretched along the viewing ray. It tracks the observed spread to within
15 % while that spread changes sixfold.

### What was tried instead

Fitted on half the floor positions, scored on the other half:

| model | numbers | how well it predicts the errors seen | truth inside the stated 95 % |
|---|---|---|---|
| one covariance in centimetres, everywhere | 3 | −4.411 | 89.9 % |
| one per camera, in centimetres | 15 | −4.519 | 88.1 % |
| **one pixel number, through the geometry** | **1** | **−5.400** | **91.0 %** |
| one pixel number per camera | 5 | −5.404 | 90.1 % |
| pixel noise by distance band | 6 | −5.370 | 90.2 % |
| a fitted map over the floor | 41 | −5.351 | 87.0 % |

The one-number model wins, and the elaborate ones get *worse*. The per-camera version is
0.004 better on likelihood — a tie — so the simpler one is kept.

**What a constant covariance in centimetres actually does**, by distance: 100 % coverage at
0–8 m, 61 % at 20–24 m. Eight times too cautious close in, where it ignores good corrections,
and four times too confident far out, where it trusts bad ones.

### How it relates to the offset

They are fitted from the same residuals but describe different halves of it: the **offset**
is the average, the **noise** is the spread around that average. The offset is removed first,
because a spread measured around a shifted centre is not the spread.

Their sizes say which matters: the offset is 0.29 cm after correction, the noise is 2.2 cm.
**The offset is a calibration detail; the noise is the sensor.**

### What must never be folded into it

Heading error. It does not reach the measurement through the pixel channel, and it is
**shared by every camera** — there is one robot with one heading. Putting it into a
per-camera pixel noise would be the wrong size and, worse, would treat a shared error as
independent, making the filter more overconfident the more cameras are fused. It belongs in
the state.

Its size is known: about 0.23 cm of position error per degree, with break-even against the
detector's own noise at about 14°. What is not known is how far the heading actually drifts —
one measurement, one drive.

---

## 7. What is settled and what is not

**Settled.** The shape model is right to 0.006 px. The box-versus-centre problem is removed
by construction, not by fitting. One pixel number beats every alternative for the covariance.
The offset is a calibration detail worth 0.29 cm. The checks fail safe.

**Not settled.** How far the operational heading drifts — the one gate. Whether the detector
ever reports a robot where there is none. Whether the availability map learned with the true
pose survives being learned from the robot's own estimate. And everything downstream:
fusion on driven routes, and the planner.
