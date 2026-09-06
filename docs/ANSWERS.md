# Answers

Companion to [`open_questions.md`](open_questions.md). Every entry here is a question that
file asks, answered with a measurement, plus the artifact the measurement came from so it can
be re-run. **A question moves here only when a number settles it.** Partial answers stay in
`open_questions.md`.

Convention, as in `PLAN.md`: verdict first, one headline number per finding, centimetres and
degrees, and every quantity scored against the truth at the instant it describes.

Status legend: **SETTLED** (a measurement answers it) · **REVISED** (the previous answer was
wrong, and why) · **OPEN** (still in `open_questions.md`).

**How to regenerate every number here.** Nothing below is a scratchpad throwaway.

```
# Q6 range/camera/floor tables, Q2 shared-error tables, and the box-mismatch flag.
# No argument reads the 1 m/s campaign, which is what the Q6 tables were built from;
# pass a campaign directory for the same tables at another speed.
python3 experiments/estimator_consistency/tail_anatomy.py [campaign_dir ...]
    -> logs/studies/estimator_consistency/tail_anatomy.json

# The covariance re-fit, four arms against the pre-registered rule below.
python3 experiments/learned_measurement_covariance/refit_schema5.py [campaign_dir]
    -> logs/studies/learned_measurement_covariance/refit_schema5.json
```

Q1, Q8 and the F3 outlier are read straight from `experiment.csv` columns
(`odom_noisy_yaw`, `planner_belief_yaw`, `state_yaw`, `odom_yaw`, `pixel_corr_*`) and each
table below names the drives it came from.

---

## Q1 — How far does the operational heading actually drift? — **SETTLED**

**The wheels are fine. Peak encoder heading error across all 24 drives is 4.33 degrees, and
20 of 24 stay under 3.**

Measured 2026-08-31 on `drives_realistic_speed_n1_20260829` (schema 4, 1 m/s, the hardest
case on disk), `odom_noisy_yaw` against `odom_yaw` at every logged row.

| source | median | p95 | peak over 24 drives |
|---|---|---|---|
| **encoder odometry** | **0.45°** | **1.46°** | **4.33°** |
| planner belief | — | — | 115.4° |
| state estimator | — | — | 180.0° |

Over one full 40 s drive at 1 m/s the encoder integrates 596.0° of rotation against a true
596.3° — an effective angular scale error of **0.0%**, which is what the configured model
(mean 0%, std 3%, AR(1) alpha 0.80 averaging out over a multi-second turn) should give.

So heading drift is **not** a reason the method fails, and a gyro would supply information the
filter already has and is discarding. What destroys the belief heading is Q8.

---

## Q6 — What owns the tail? — **SETTLED**

**A range-independent ~1.0 cm error floor that the covariance model omits. The tail is a
NEAR-range effect: 9.3% of readings under 6 m, 0.1% beyond 16 m.**

Measured 2026-08-31 on 4,459 admitted readings from the 16 schema-4 hull drives (F1–F4), each
scored against truth at its own capture stamp via `experiments/fusion_on_fixed_routes/aligned.py`.

The claimed covariance is `Sigma_c = J^-1 R_pix J^-T` with a constant 0.76 px. That is **pure
pixel noise**, and it shrinks without limit as the camera gets closer. The real error does not.

| range | n | claimed sigma | real error | real/claim |
|---|---|---|---|---|
| 0–4 m | 407 | 0.69 cm | 1.18 cm | **1.70x** |
| 4–6 m | 791 | 0.81 cm | 1.45 cm | **1.79x** |
| 6–8 m | 509 | 1.09 cm | 1.35 cm | 1.23x |
| 8–10 m | 552 | 1.45 cm | 1.40 cm | 0.97x |
| 10–13 m | 856 | 2.24 cm | 1.76 cm | 0.78x |
| 13–16 m | 604 | 2.83 cm | 1.83 cm | 0.65x |
| 16–20 m | 692 | 4.37 cm | 4.96 cm | 1.14x |

A correct model would sit flat near 1.2x (the median of a 2-D Rayleigh). It does not: the
claim collapses with closeness while the real error bottoms out around 1.2 cm.

**The floor is model error, not timing.** Same route and arm at two speeds:

| | close-range (<6 m) real error | claimed sigma |
|---|---|---|
| 0.22 m/s | 1.02 cm | 0.79 cm |
| 1.00 m/s | 1.25 cm | 0.78 cm |

Speed rose 4.5x; the floor rose 1.23x. Solving `floor^2 = model^2 + (v*dt)^2` gives
**model ~1.01 cm and timing jitter ~7.4 ms**. Claimed sigma is identical at both speeds, as
pure geometry must be. *(One run per speed — being re-measured with more drives.)*

**The most accurate camera is the least honest.**

| camera | n | tail rate | median error |
|---|---|---|---|
| **B** | 1146 | **10.6%** | **1.46 cm** |
| C | 275 | 5.1% | 6.63 cm |
| D | 930 | 0.9% | 1.65 cm |
| E | 1483 | 0.6% | 1.90 cm |
| A | 625 | 0.2% | 1.30 cm |

B is the close camera. It is accurate *and* claims far more precision than it earns, so
precision-weighted fusion weights it hardest and the fused answer inherits its error while
claiming its ellipse. That is why two contributing cameras was the worst case in the
consistency study (NEES 31.6, x15 inflation).

**Bias is a FAR-range defect; inconsistency is a NEAR-range defect.** Two different problems at
opposite ends of the range axis — which is why chasing bias never moved the NEES.

**Confidence cannot flag the tail** (0.936 bulk vs 0.953 tail — the tail is marginally *more*
confident). **Range separates it 90:1**, is truth-free, and is known at planning time.

| floor added in quadrature | readings beyond 4 sigma |
|---|---|
| 0 (today) | 5.2% |
| **1.0 cm** | **1.9%** |
| 1.5 cm | 1.3% |

That is a one-parameter, physically motivated model term, not the x6 blanket inflation the
consistency study correctly rejected.

### IMPORTANT — the severity is speed-dependent; the mechanism is not

Everything above was measured at **1 m/s**, the speed `open_questions.md` #7 already ruled out
for the fusion comparison. Re-checked at 0.22 m/s on three independent sources:

| drives | n | NEES median | NEES mean | tail (>20) |
|---|---|---|---|---|
| 0.22 m/s, new schema-5 drive | 1610 | 0.47 | **1.06** | **0.1%** |
| 0.22 m/s speed probe | 711 | 0.89 | **5.69** | **1.7%** |
| 1.00 m/s speed probe | 317 | 1.47 | 11.85 | 4.7% |
| 1.00 m/s campaign, 16 hull drives | 4459 | 1.26 | 11.88 | 3.5% |

**At the speed the paper will use, the tail is 0.1-1.7%, not 3.5%, and mean NEES is 1.06-5.69,
not 11.88.** So *"the fused estimate is six times overconfident"* is a 1 m/s statement and must
not be quoted as the paper's number.

What does **not** change is the mechanism: the covariance is pure pixel noise, so it keeps
shrinking with closeness while the real error bottoms out. At 0.22 m/s the close-range
real/claim is still **1.30x** against 0.65-0.85x at mid range — the same wrong shape, milder.
The floor is still the right fix; it is a correctness fix to the model's form, not a rescue
from a crisis.

### RE-MEASURED on 23 drives at 0.22 m/s — Q6 revised again, and it moves

The diagnostic campaign landed (`diagnostic_schema5_20260831`, 23 valid runs, 24,172 admitted
hull readings, 6,776 multi-camera batches). **Everything above stands as a description of the
1 m/s drives and is superseded as a description of the method.**

| | 1 m/s (16 drives) | **0.22 m/s (23 drives)** |
|---|---|---|
| NEES median | 1.26 | **0.67** |
| NEES mean | 11.88 | **3.64** |
| tail (>20) | 3.45% | **1.36%** |
| median real/claim | 1.10x | **0.79x** |

**At the paper's speed the estimator is not broadly overconfident — it is mildly
conservative.** And the shape moves:

| range | n | claimed | real | real/claim | tail |
|---|---|---|---|---|---|
| 0-4 m | 1988 | 0.69 cm | 0.63 cm | **0.91x** | 1.4% |
| 4-6 m | 4414 | 0.82 cm | 0.98 cm | **1.20x** | **4.5%** |
| 6-8 m | 2843 | 1.12 cm | 0.88 cm | 0.78x | 0.6% |
| 8-10 m | 3181 | 1.47 cm | 1.05 cm | 0.72x | 0.9% |
| 10-13 m | 5147 | 2.28 cm | 1.48 cm | 0.65x | 0.9% |
| 13-16 m | 3367 | 2.86 cm | 1.31 cm | 0.46x | 0.3% |
| 16-20 m | 2923 | 4.41 cm | 5.66 cm | **1.28x** | 0.1% |
| 20+ m | 309 | 5.09 cm | 7.21 cm | **1.42x** | 0.0% |

The 1.70-1.79x near-range overconfidence at 1 m/s becomes **0.91x and 1.20x**. So most of what
looked like a near-range covariance defect was the belief being worse at 1 m/s, feeding a
worse prediction pose into the observation. **By the ratio, the worst range band is now the
far one (1.28-1.42x).** By the *tail*, near range still owns it (4.5% at 4-6 m against 0.1%
beyond 16 m) — so the two diagnostics disagree, and both are reported.

### The re-fit: no arm passes, and the reason is the finding

`refit_schema5.py`, held out on `fusion_network_traverse` as pre-registered, NSE 2.0 =
consistent:

| arm | A | B | C | D | E | tail | |
|---|---|---|---|---|---|---|---|
| as deployed | 0.67 | 1.90 | **2.89** | 1.55 | 0.50 | 2.73% | fail |
| pixel refit | **0.39** | 1.12 | 1.71 | 0.92 | **0.30** | 0.84% | fail |
| uniform floor | 0.50 | 0.84 | **3.74** | 1.04 | **0.45** | 1.29% | fail |
| per-camera floor | 1.04 | **0.46** | **2.74** | 2.12 | 0.79 | 1.66% | fail |

**The metric floor is real and now confirmed three independent ways.** The uniform fit returns
**1.00 cm**, against 1.25 cm from the earlier commissioning fit and ~1.0 cm from the range
asymptote in Q6 — three methods, three datasets, one number. The per-camera fit gives
A 0.00, B 1.55, **C 3.10**, D 0.30, E 0.00 cm, so it is genuinely not uniform.

**But every arm fails on camera C, and C is not a covariance problem.** Residual along its own
viewing ray:

| camera | n | bias along ray | spread along | median range |
|---|---|---|---|---|
| A | 3115 | +0.43 cm | 1.60 cm | 11.0 m |
| B | 6570 | +0.27 cm | 2.01 cm | 5.2 m |
| **C** | 966 | **+3.67 cm** | **8.06 cm** | 17.9 m |
| D | 6025 | +1.47 cm | 3.59 cm | 9.4 m |
| E | 7496 | +1.12 cm | 2.94 cm | 12.7 m |

C only ever sees the robot far away, and beyond 16 m it leans **+6.35 cm (16-18 m)** and
**+8.03 cm (20+ m)** along the ray while its spread grows to 8.06 cm against a claimed 4.4 cm.
Note what the uniform-floor arm does to it: **2.89 -> 3.74, worse**, because that fit also
scales the pixel term down to 0.65 to help the close cameras, and 1 cm of floor is nothing
against C's 4.4 cm ellipse.

> **So the covariance needs two different corrections at the two ends of the range axis** — a
> ~1 cm metric floor close in, and something range-growing beyond ~16 m where both the lean
> and the spread blow up. A two-parameter model cannot do both at once, which is exactly why
> all four arms fail. **That is the answer, not a failed experiment.** The pre-registered rule
> earned its keep: without it, "per-camera floor, C 3.10 cm" would have been reported as a win.

The remaining option the repo already names is the 16 m admission gate from
`unbiased_observation` — refusing what cannot be modelled rather than inflating it. It is
listed there as a proposal that nothing implements, and it is now the natural next arm.

### The observation model decides whether the robot survives the route

Outcome by arm, 23 valid drives, identical frozen routes and cameras:

| | F1 | F2 | F3 | F4 | O1 | O2 |
|---|---|---|---|---|---|---|
| network_traverse | goal | goal | goal | goal | **collision** | **collision** |
| overlap_rich | goal | goal | goal | goal | goal | **collision** |
| overlap_sparse | goal | goal | goal | goal | **collision** | **collision** |
| long_traverse | goal | goal | goal | *timeout* | **collision** | **collision** |

**Hull arms 15/16 goal, 0/16 collisions. Degraded observation models 1/8 goal, 7/8
collisions.** At 0.22 m/s the 5-8x error contrast becomes a difference in whether the robot
finishes — a far stronger statement of the same result. (`F4/long_traverse` is a wall-clock
timeout on the 63 m route, not a failure; 900 s was not enough at 0.22 m/s.)

Belief error, hull arms, network_traverse: **1.08-1.39 cm median** against 3.09-3.84 cm for
the same arms at 1 m/s.

### A second runtime flag, better than range

The new `pred_h_px` column gives the box mismatch `|detected/predicted - 1|`, and on the first
schema-5 drive it tracks inconsistency better than range does:

| \|ratio-1\| | n | median NEES | median error | median range |
|---|---|---|---|---|
| 0.00-0.01 | 778 | 0.30 | 0.72 cm | 8.9 m |
| 0.01-0.02 | 394 | 0.57 | 1.15 cm | 10.2 m |
| 0.02-0.04 | 239 | 0.80 | 1.89 cm | 12.2 m |
| 0.04-0.08 | 190 | 0.91 | 2.63 cm | 18.1 m |

Spearman against NEES: **+0.336 for box mismatch**, −0.174 for range. It is truth-free and
already computed inside the admission gate.

**Confirmed on the full campaign** (23 drives, 24,172 readings, 0.22 m/s): box mismatch
**+0.383** against NEES, range **−0.126**. So the box-agreement signal is three times the
rank-correlation of range, and unlike range it is not confounded with which camera is looking.
Together with the U-shaped error table in Q5, `|detected/predicted − 1|` is the best
truth-free quality signal found so far — better than detector confidence (which is flat, Q6)
and better than range.

---

## Q8 — Is heading what ends drives? — **REVISED**

**Yes, but it comes from the filter, not the wheels.** `RESULTS_why_drives_fail.md` says
*"it comes from the wheels, not the cameras"* and attributes ~12 degrees to encoder slip over
a 96-degree turn. That is wrong: the -16.0 degree figure it quotes matches `state_yaw` error,
not `odom_noisy_yaw`. See Q1 — the encoder never exceeds 4.33 degrees.

Heading error against truth **at the moment of contact**, all ten collision drives:

| route / arm | encoder | belief | state |
|---|---|---|---|
| long_traverse F2 | −1.17° | **+27.7°** | +33.7° |
| long_traverse F3 | +0.85° | +0.81° | +179.9° |
| long_traverse F4 | −0.12° | **+51.5°** | +57.6° |
| long_traverse O1 | −0.74° | **+108.8°** | +117.2° |
| long_traverse O2 | +0.55° | **+83.9°** | +83.0° |
| network_traverse O1 | −0.75° | **+33.4°** | −80.0° |
| network_traverse O2 | −2.38° | **+83.6°** | +88.6° |
| overlap_sparse F1 | +0.53° | +1.71° | +4.66° |
| overlap_sparse F3 | +0.30° | −3.15° | −1.39° |
| overlap_sparse O1 | −0.62° | **+29.5°** | −62.3° |

**The mechanism** is the second one that file already names, and it is correct. Under
`heading_update_mode: coupled` the prediction step builds a position–heading correlation, so
the gain's third row converts a position innovation into a heading correction. **The lever arm
is the distance driven since heading was last constrained.** The filter cannot tell an
innovation caused by heading error from one caused by a bad reading — and by Q6, 3.5% of
readings are wildly inconsistent. Those get levered straight into heading.

Caught in the act on `long_traverse/O1` — belief heading moving 4–6 degrees per **accepted**
correction while the encoder sits at 0.4:

| t | belief jump | belief−truth | encoder−truth | NIS | accepted |
|---|---|---|---|---|---|
| 6.00 | −5.87° | +6.99° | −0.10° | 6.5 | yes |
| 6.31 | −6.02° | +3.88° | +0.35° | 6.3 | yes |
| 6.70 | +6.10° | −6.40° | +0.39° | 6.2 | yes |
| 6.80 | +5.73° | −5.25° | +0.29° | 6.2 | yes |

**The NIS gate screens the xy innovation; nothing caps the heading share of an update.**

**Crashes are a speed artifact**, not a system defect: 10 of 24 collisions (42%) at 1 m/s
versus **4 of 108 (3.7%)** in the 0.22 m/s campaign, same `heading_update_mode: coupled`, same
routes, same detector.

The fix is not a gyro. `camera_xy_only` costs a config flag; the better experiment is coupled
*with a bound on the heading share per update*. Separately, `state_yaw`'s ~90° and ~180° peaks
are frame-convention signatures — a distinct defect, and not `atan2(dy,dx)`, since
`infer_yaw_from_motion` is False in these runs.

---

## Q7 — Does the method survive an operational driving speed? — **SETTLED**

**Yes, but the degradation is a sampling-rate effect, not a sensor effect, so the fusion
comparison must be run at 0.22 m/s.**

Already recorded in `open_questions.md` #7 and not re-derived here: between 0.22 and 1.00 m/s
the camera reading worsens 1.4x (1.28 → 1.82 cm) while the belief worsens 2.6x (2.92 →
7.49 cm), because corrections per metre fall from 22.2 to 5.3. The reading's 95th percentile
is unchanged at all three speeds. The arm difference is ~0.3 cm; the 1 m/s penalty is 4.6 cm.

---

## Q2 — How much of the cameras' error is shared? — **SETTLED**

**There is no common-mode offset, but the errors are not independent either: fusion delivers
about half the improvement independence predicts.**

Measured 2026-08-31 on the same 16 schema-4 hull drives — 2,638 detector batches, 1,199 of
them with two or more admitted cameras.

**(a) No shared offset, pooled.** Estimating the shared variance from the mean cross-product
of two cameras' residuals in the same batch (`E[r_i . r_j] = E|s|^2` for `i != j`):

| closest camera in the batch | pairs | rms error | shared component | share of squared error |
|---|---|---|---|---|
| all batches | 2577 | 4.83 cm | **0.00 cm** | **0.0%** |
| 0–6 m | 1962 | 3.96 cm | 1.41 cm | 12.7% |
| 6–10 m | 509 | 6.70 cm | 0.00 cm | 0.0% |
| 10–16 m | 104 | 4.29 cm | 2.08 cm | 23.5% |

Pooled it is exactly zero, confirming the earlier finding that per-pair correlations reach
0.92 but **flip sign between pairs**, so a scalar `fusion_common_mode_std_m` — which can only
represent a positive shared variance — is the wrong instrument. The 0–6 m band is the one
with a real shared component, and that is where the Q6 floor lives.

**(b) But fusion under-delivers, at every range.** Equal-weight fusion of the admitted
readings in a batch, against what independent errors would give:

| cameras | batches | best single | mean single | fused | independence predicts |
|---|---|---|---|---|---|
| 2 | 711 | 1.29 cm | 1.97 cm | **1.71 cm** | 1.39 cm |
| 3 | 354 | 0.92 cm | 2.02 cm | **1.65 cm** | 1.17 cm |
| 4 | 134 | 0.73 cm | 2.50 cm | **1.86 cm** | 1.25 cm |

| | mean single | fused | ratio | independence would give |
|---|---|---|---|---|
| close (<6 m) | 2.11 cm | 1.78 cm | **0.84** | 0.62 |
| far (>10 m) | 2.97 cm | 2.42 cm | **0.81** | 0.62 |

So the answer is not "shared" or "independent" but **structured**: the dependence is
geometric — each camera errs along its own viewing ray — so it cancels in a pooled offset
estimate while still robbing fusion of half its theoretical benefit. Any correction must be
a per-camera directional term, not a scalar added to both diagonal entries.

*Caveats: equal-weight fusion, not the runtime's precision weighting; admitted readings only,
which conditions on agreeing with the other cameras; n=4 has 134 batches and the 10–16 m band
104 pairs.*

**Re-measured at 0.22 m/s** (6,776 multi-camera batches, 23 drives) — the shortfall is real
but smaller, and it is not uniform in the number of cameras:

| cameras | batches | mean single | fused | independence predicts |
|---|---|---|---|---|
| 2 | 3852 | 1.34 cm | 1.10 cm | 0.94 cm |
| 3 | 1917 | 1.25 cm | **0.73 cm** | **0.72 cm** |
| 4 | 1007 | 3.17 cm | 2.11 cm | 1.59 cm |

At three cameras fusion **achieves the independence prediction exactly** (0.73 against 0.72).
At two and four it falls short. So the dependence is not a fixed tax on fusion; it is
geometric, and it depends on which cameras happen to overlap. That is consistent with the
sign-flipping correlations above and, again, rules out a scalar common-mode term.

---

## Q5 — Why are 21% of admitted boxes taller than the projected shape predicts? — **ANSWERED**

**Not answerable from the drives on disk: `fusion_observations.csv` logs the *detected* box
height (`bbox_h_px`) but never the *predicted* one.** The prediction is computed at
`silhouette_observation.py:197` and discarded.

What is known, from `logs/studies/learned_measurement_covariance/per_camera_error.json` — and
this is the source of the 21% figure (1,347 of 6,418 readings):

| height ratio | n | bias along the ray | spread | median range |
|---|---|---|---|---|
| 0.85–0.92 | 93 | 12.51 cm | 6.19 cm | 18.5 m |
| 0.92–0.96 | 206 | 7.30 cm | 4.69 cm | 17.2 m |
| 0.96–0.99 | 1460 | 2.00 cm | 2.66 cm | 9.7 m |
| **0.99–1.02** | **3312** | **0.17 cm** | 3.10 cm | 8.9 m |
| **1.02–3.00** | **1347 (21%)** | **1.20 cm** | 4.20 cm | 12.1 m |

The *short* boxes (ratio < 0.96) are the known far-range truncation story — 7–12 cm of lean at
17–18 m. The *tall* population is a different thing: mid-range, and carrying 1.2 cm.

**The leading hypothesis, untested:** the box is predicted from the *believed* pose, so a
belief that is too far away predicts too small a box, giving ratio > 1 **and** a lean. If that
is right, height ratio is a symptom of belief error rather than a cause, and it is directly
measurable once the predicted height is logged.

### Answered on 24,144 readings that carry a predicted box

`pred_h_px` was added (schema 5) and the diagnostic campaign run. **17.9% of admitted boxes
are taller than predicted** (>1.02) and 6.6% shorter, reproducing the historical 21%.

| height ratio | n | median error | median range |
|---|---|---|---|
| < 0.92 | 394 | **8.31 cm** | 13.3 m |
| 0.92-0.96 | 1195 | 4.20 cm | 12.4 m |
| 0.96-0.99 | 5692 | 1.42 cm | 10.5 m |
| **0.99-1.02** | **12546** | **0.83 cm** | 8.2 m |
| 1.02-1.06 | 3457 | 1.86 cm | 11.7 m |
| > 1.06 | 860 | **3.63 cm** | 12.5 m |

**The error is U-shaped with a sharp minimum where the boxes agree.** Disagreement costs in
*either* direction — 2.2x for mildly tall, 4.4x for very tall, 10x for very short. The
original question asked why boxes are tall; the useful answer is that tallness is one side of
a symmetric signal, and the signal is worth more than the explanation.

**It is not a range effect, and the leading hypothesis is wrong.** Within each camera the tall
readings sit at that camera's own median range, so they are not a distant subpopulation:

| camera | tall (>1.02) | median range of the tall | camera median range |
|---|---|---|---|
| A | 25.7% | 11.9 m | 11.0 m |
| B | 9.8% | 5.1 m | 5.2 m |
| C | **37.5%** | 17.8 m | 17.9 m |
| D | 11.7% | 10.5 m | 9.4 m |
| E | 24.2% | 12.5 m | 12.7 m |

What varies is the **camera** — 9.8% to 37.5% — which points at per-camera viewing geometry or
the hull's fit at that aspect, not at belief error (which would affect every camera at a given
instant alike). The mechanism is not settled; the measurement is.

---

## The F3 / long_traverse outlier — **SETTLED**

**A stale correction plus command replay produced a 3.2 m belief jump, after which the
replay-gap guard refused every subsequent correction and the belief never recovered.** It is
a latency and recovery-policy failure, not a fusion or heading failure.

| t | belief error vs map-frame truth | correction | note |
|---|---|---|---|
| 53.70 | 87.7 cm | accepted, NIS 0.031 | **correction age 4.50 s, 10 command replays** |
| 54.61 | 85.4 cm | accepted, NIS 0.031 | |
| **54.90** | **408.5 cm** | **refused** | `replay_gap_too_large` |
| 55.20 – 58.50 | 409–413 cm | refused, every row | `replay_gap_too_large` |

At 1 m/s a correction 4.5 s old describes a pose 4.5 m behind the robot. The replay machinery
tried to carry it forward across ten commands, the belief jumped 3.2 m in 0.3 s, and from
then on the replay gap exceeded its limit — so the filter locked itself out of the only thing
that could have fixed it, drove on dead reckoning, and hit `obs_A3b2e`. Peak belief error
413.4 cm.

This is the recovery policy (`state_max_predict_dt_s`, `state_reject_inflate_m2`) acting as a
trap: refusing corrections is the right response to a stale one, but there is no path back.
It reinforces Q7 — correction latency is a speed problem, and 1 m/s is where it bites.

---

## What new data is needed, and the rule written before it arrives

Three of the questions above stop at the edge of what the drives on disk can support. One
campaign serves all three: **24 runs, one seed per cell, 0.22 m/s, schema 5**
(`scripts/visibility_comparison/fusion_diagnostic_schema5_campaign.yaml`, launched
2026-08-31 into `logs/studies/fusion_on_fixed_routes/diagnostic_schema5_20260831`).

**LANDED 2026-08-31: 23 of 24 runs valid** (one wall-clock timeout on the 63 m route, where
900 s is not enough at 0.22 m/s — raise it before re-running that cell). Results are folded
into Q2, Q5, Q6 and the re-fit above.

It is a **diagnostic campaign, not the frozen fusion result.** The frozen 120-run campaign
must come after the covariance floor is commissioned, because the floor changes each camera's
weight and therefore changes what the fusion arms do. Running it first would measure a model
already known to be wrong.

### What the new drives carry that the old ones do not

`pred_h_px` / `pred_w_px` — the box the hull model predicted from the pose the correction was
made from. It was computed at `silhouette_observation.py:197` and discarded, so the height
ratio could not be reconstructed from any drive. Added to `camera_manager_node.py` and
`experiment_logger.py`; `logging_schema_version` 4 -> 5. Verified on a real drive: 1,610 of
1,612 rows populated, 13.5% of boxes above ratio 1.02. Cost measured, not assumed:
**56 microseconds per call, 0.14% of one core.**

### The floor re-fit — decision rule, written before the data exists

`experiments/learned_measurement_covariance/per_camera_error.py` already implements the model

```
R = (sigma_px / sigma_commissioned)^2 * C_stated  +  sigma_m^2 * I
```

and already fitted it once, on the pre-repair commissioning drives at the wrong speed. It
returned **`floor_cm` 1.25** — which independently matches the 1.0-1.25 cm measured by range
asymptote in Q6, by a different method on different drives. Its pre-registered rule then
**rejected** it:

| arm | A | B | C | D | E |
|---|---|---|---|---|---|
| as deployed | 0.76 | **2.00** | **1.70** | 1.33 | 0.71 |
| pixel refit | **0.32** | 0.83 | 0.71 | 0.55 | **0.29** |
| two-term, uniform floor | **0.38** | 0.67 | 1.58 | 0.78 | **0.36** |

Read this table carefully: `nse = median(NEES) / 2 ln 2 * 2`, so **2.0 is the consistent
value**, not the failure threshold. Above 2.0 is overconfident, below is conservative. As
deployed, only camera B is at the line (2.00) and every other camera is already
*conservative*. The uniform floor then pushes **every** camera further toward conservatism,
taking A to 0.38 and E to 0.36 — outside the rule's [0.5, 2.0] band.

**So a single floor shared by all cameras is already falsified, and re-fitting one on better
data is not the experiment.** Note also what that rule measures: NSE is built from the
*median* NEES, so it scores the bulk and is blind to the tail — which is the whole reason
this question exists. Hence the added tail criterion below.

Pre-registered for the re-fit, before the campaign lands:

1. **Arms.** (a) as deployed; (b) pixel-only refit; (c) uniform metric floor — the rejected
   one, re-run for continuity; (d) **per-camera metric floor**, one `sigma_m` per camera.
2. **Rule.** Same as before — every camera's NSE inside [0.5, 2.0] on a held-out drive — plus
   the tail test that motivated this at all: **readings beyond 4 sigma must fall below 2%**
   (they are 5.2% today). An arm that fixes the mean while leaving the tail is not a fix.
3. **Held out.** One whole route, never used for fitting, chosen before fitting begins.
4. **A null is a result.** If (d) does not beat (c) on the held-out route, the floor is
   uniform after all and the earlier rejection was a small-sample artifact. Report it as such.
5. **What must not happen.** The floor is not to be chosen by scanning it against NEES.
   It is measured as the asymptote of error versus range and then tested.

### Still needing a simulator beyond this campaign

- **Q3, false positives.** One empty-warehouse capture, five cameras. The commissioning
  capture kept a single empty frame per camera, so the rate is unmeasured — and it is half the
  definition of a usable sighting.
- **Q4, availability from the robot's own estimate.** Blocked on a truth-free operational
  dataset that does not exist. It serves downstream Stage 4 of the active paper and does not
  block the current characterization stage.

---

## Still open

- **Q3 — false positives.** Needs one empty-warehouse capture, five cameras. Not answerable
  offline: the commissioning capture kept only one empty frame per camera.
- **Q4 — availability learned from the robot's own estimate.** The fact is measured (usable
  rate 0.70 → 0.56 when re-judged from a pose 10 cm / 2° wrong), but the truth-free
  operational dataset it needs does not exist. This is downstream Stage 4, not evidence for
  the current characterization stage.
- **The gate.** No fusion result is frozen. `frozen_runs.json` points at the 1 m/s campaign,
  whose own `what_this_supports` says the four fusion rules are not separated by it. The
  paper's result needs a schema-4 campaign at 0.22 m/s — and, per Q6, it should be run only
  after the covariance floor is commissioned, since that changes the per-camera weights.
