# Experiment design: making the planning and fusion contributions separately visible

Two contributions are claimed. They must be **attributable**, which means neither may be
demonstrated by an experiment that also varies the other. The design below fixes one while
varying the other, then closes with a factorial corner test that shows the two are not the
same effect.

Companion to `00c_results_plan.md` (what gets reported) and `00b_methodology.md` (how things
are scored).

---

## The two contributions, stated so they can fail

**Planning.** *Routing on a spatially varying map of expected positional uncertainty changes
the route the robot drives and reduces the uncertainty it actually accumulates, beyond what
routing on availability alone achieves.*
Falsified if `P2` ties `P1` on the primary endpoint over the operating envelope.

**Fusion.** *Combining cameras by their expected uncertainty — not by availability, and not
by plain averaging — delivers a better-calibrated, sharper belief, by an amount the crossing
geometry predicts in advance.*
Falsified if `F2` ties `F3` once the observation function is fixed.

The second half of each — "beyond availability alone", "not by plain averaging" — is what
makes the contribution non-trivial, so each experiment carries the ablation that tests it.

---

## Gates. Nothing is allocated until these pass

| Gate | Requirement | Why | Status |
|---|---|---|---|
| **G1** Observation function | Two-camera fused ÷ single-camera delivered error **< 0.6** at crossing angles > 135° on real captures | On the deployed box-bottom reading the error is 40–48 mm of yaw-driven scatter *shared* between cameras; every fusion arm then measures the same shared error and ties by construction (measured: ×1.02 / 1.03 / 0.95 / 0.92 / 0.87) | **FAILS** — runtime is `homography_to_bev` / `uv`, no keypoint reading wired in |
| **G2** Heading observability | Planner yaw variance measured, not sitting at its non-informative prior (π²) | The multicam transport delivers position only; a belief-space planner that inflates clearance by predicted covariance is dominated by this term | **FAILS** at 4 cameras |
| **G3** Route separation | Candidate menu has ≥3 topologically distinct routes, max pairwise separation > robot width, **identical across arms** | Arms that could not have disagreed cannot evidence agreement | must be checked per task |
| **G4** Objective sensitivity | On the real menu, at the belief the robot actually carries, the objective must rank ≥2 candidates differently across arms | Do not spend closed-loop time on an objective that provably cannot discriminate | offline, cheap |

G1 and G2 are *work*, not checks. G1 is wiring the measured keypoint reading into the runtime
(it cuts the lean 7.78 → 1.51 cm and removes the heading dependence). G2 needs a correction
that observes heading — the natural fix is to keep the per-camera image-space measurement
alongside the fused position rather than replacing it, which is what the single-camera stack
did successfully.

---

## E1 — Do the two maps disagree enough to matter? *(offline, establishes the premise)*

**Question.** How much of the reachable floor is a decision point between the two maps, and
what does following the wrong one cost?

**Design.** No arms. A measurement on frozen maps over the reachable floor, stratified by
camera subset `{4, 3, 2-opposite, 2-same-wall}`.

**Primary endpoint.** Fraction of covered positions where `argmax p_use ≠ argmin tr(R_cond)`,
with the median error penalty for following availability and the median availability penalty
for following accuracy.

**Why it is here.** It sizes both contributions before either is run, and it is the number
that justifies carrying two maps rather than one. Already measured at four cameras (42.2 % of
353 positions, 1.3 cm penalty); this extends it across subsets and makes it a declared
premise rather than an incidental finding.

**Falsifies the whole pipeline if** the disagreement fraction is small at every subset — then
one map suffices and the architecture is over-built.

---

## E2 — The fusion contribution *(route held fixed)*

**Question.** With the route frozen, does combining cameras by expected uncertainty beat
availability-ranked selection and plain averaging?

**Held fixed.** The route. One frozen set of polylines, hash-bound before execution, taken
from E3's availability-aware arm so every fusion arm drives *exactly the same path*. Also
fixed: world, calibration, detector, weights, threshold, noise, seeds, controller, maps.

**Arms** — fusion policies over frozen maps, differing only in how corrections are combined:

| Arm | Policy | What it isolates |
|---|---|---|
| `F0` | nearest camera only | the no-fusion baseline |
| `F1` | select the single **max-availability** camera | fusing on the *wrong* map |
| `F2` | **R-weighted** fusion over all available cameras | the contribution |
| `F3` | plain average over all available cameras | *the ablation* — separates weighting from merely using more cameras |

`F2` vs `F3` is the fusion contribution proper. `F2` vs `F1` is the "expected uncertainty, not
availability" claim. `F3` vs `F0` absorbs the "more cameras is just better" objection.

**Primary endpoint.** Filter calibration on held-out real captures: median NEES and fraction
of truth outside the stated 95 % ellipse, with position RMSE reported alongside. Calibration
is primary because a sharper-but-overconfident belief is a failure, not a win.

**Secondary, closed-loop.** Time and distance driven with no accepted correction; minimum
clearance as a continuous statistic; goal attainment; physical contact.

**Mechanistic pre-registration — this is what makes it a contribution and not a horse race.**
The crossing-angle model predicts the effect size before the data: the repeatable lean
survives as `b·cos(θ/2)`, so `F2`'s advantage over `F3` should **grow with crossing angle**
and vanish for same-wall pairs. Stratify by pair crossing angle in bins
`{45–75, 75–105, 105–135, 135–165, 165–180}` and report the predicted-vs-measured curve. A
gain that appears with no angle dependence is *not* this mechanism and must be reported as
unexplained.

**Powering.** Offline arm runs on the existing balanced set-pose grid (1844 clear detections,
575 camera pairs on the same held pose), so no new capture is needed for the primary endpoint.

---

## E3 — The planning contribution *(fusion held fixed)*

**Question.** With the fusion policy frozen, does routing on both maps beat routing on
availability alone, and both beat routing on neither?

**Held fixed.** The fusion policy — `F2` if E2 selects it, otherwise the deployed policy,
declared before E3 runs. Same world, detector, threshold, lanes, seeds.

**Arms** — what the route decision is allowed to read:

| Arm | Reads | Rule |
|---|---|---|
| `P0` | nothing | shortest route over the lane graph |
| `P1` | availability only | minimise `∫(1 − p_any) dℓ` s.t. `len ≤ (1+ε)·len_min` |
| `P2` | availability **and** uncertainty | minimise `∫ E[σ(ℓ)] dℓ` s.t. `len ≤ (1+ε)·len_min` |

with the expected positional uncertainty along the route

```
E[σ(ℓ)] = (1 − p_any(ℓ))·σ_dr(ℓ)  +  p_any(ℓ)·sqrt(tr R_fused(ℓ))
```

`σ_dr` is the dead-reckoning growth the filter would accumulate unobserved; `R_fused` is the
uncertainty map composed under the frozen fusion policy.

**The property that makes P1 vs P2 a clean test.** `P2` **degenerates to `P1` when the
uncertainty map is spatially constant**. So `P1` is not a strawman — it is exactly `P2` with
the second map's spatial structure removed, and the comparison tests precisely whether that
structure carries routing-relevant information. One budget parameter `ε`, identical for all
arms, swept over the same declared grid.

**Primary endpoint.** Uncertainty actually accumulated along the driven route, scored against
**real detector outcomes** and ground truth: integral of realised position error, plus the
longest continuous stretch with no accepted correction. Not expected — realised.

**Secondary.** Expected blind distance (comparable to the existing 55 % result); minimum
clearance; goal attainment; collisions.

**Guardrail, stated in advance.** **Terminal** belief covariance is not an endpoint. It moves
0.6 mm across arms — below noise — and any result phrased as "ends up more certain" is
inadmissible.

**Route menu.** Lane-graph enumeration, identical for all arms, G3-gated per task. Report the
number of distinct candidates and their maximum separation for every task *before* costs.

---

## E4 — The corner test: are they two contributions or one? *(the joint claim)*

**Question.** Do planning and fusion contribute independently, or does one subsume the other?

**Design.** 2×2 at the corners, closed-loop, on the tasks where E3 found route divergence:

| | deployed fusion | `F2` R-weighted |
|---|---|---|
| `P0` shortest | baseline | fusion only |
| `P2` two-map routing | planning only | both |

**Primary endpoint.** The interaction term on realised accumulated uncertainty. Three
outcomes, all publishable, all pre-named:

- **additive** — the two contributions are independent, and the paper claims both;
- **sub-additive** — they overlap, and the honest claim is one mechanism reported two ways;
- **super-additive** — routing through well-observed ground is what *lets* uncertainty-weighted
  fusion pay, which is a stronger and more interesting claim than either alone.

This is the experiment that earns the word "and" in "planning and fusion".

---

## E5 — Placement as the unifying axis *(confirmatory)*

**Question.** Does the same camera geometry govern both contributions?

**Design.** Repeat E2 and E3 across `{4, 3, 2-opposite, 2-same-wall, 1}`, declared before any
outcome. Subsets are analysis-time for the offline endpoints, so this is nearly free there.

**Pre-registered prediction, from the mechanisms and not from the data.** Both contributions
should be **larger where crossing angles are large** — `2-opposite` (A–B 124°, C–D 122°,
diagonals 165–167°) should beat `2-same-wall` (A–C 68°, B–D 78°) on both endpoints, *despite
`2-same-wall` having the worse baseline*. Availability and uncertainty are estimated
independently and by different means, so if they agree on placement that is a real structural
result; if they disagree, the "placement not count" claim is withdrawn.

**Why this is the headline.** It is the only claim that ties the two contributions to a single
physical cause, and it is a network-design statement a practitioner can act on.

---

## Attribution summary

| Experiment | Varies | Holds fixed | Establishes |
|---|---|---|---|
| E1 | nothing | everything | the premise: two maps are two decisions |
| E2 | fusion policy | **route** | the fusion contribution, with its ablation `F2` vs `F3` |
| E3 | routing input | **fusion policy** | the planning contribution, with its ablation `P1` vs `P2` |
| E4 | both, at corners | — | independence, and the right to claim two contributions |
| E5 | camera subset | — | one physical cause behind both |

**Order of execution.** G1 → G2 → E1 → E2 → E3 → G3/G4 → E4 → E5. G1 first because every
fusion arm ties without it, and G2 before any four-camera closed-loop run.

**What is reusable.** E1, E2's primary endpoint, and E3's offline route selection all run on
existing real captures — the set-pose grid and the reconfiguration captures — so only E4
and the closed-loop halves of E2 and E3 need new simulator time.

---

## E6 — The changing world crossed with both contributions

The reconfiguration is not a robustness check bolted on the end. It separates the two maps,
because **they do not go stale the same way.**

### The asymmetry that makes this a result

| Quantity | Depends on | Stale after restocking? |
|---|---|---|
| Per-camera `R_cond = σ_px²·J Jᵀ` | camera pose and pixel location, through the back-projection Jacobian | **No.** Racks do not change the back-projection. |
| Per-camera `p_use` | sight-lines | **Yes.** This is the whole reconfiguration result. |
| **Fused** `R_fused` | per-camera `R` **and which cameras are available** | **Yes, but only through the availability channel.** |

So conditional uncertainty is structurally invariant to the building change, and every bit of
fused-uncertainty staleness is inherited from availability. That is a decomposition, not a
caveat, and it yields three predictions registered before the L1 numbers are looked at.

**H1 (planning robustness).** `P2` degrades **less** than `P1` from L0 to L1, because half of
`P2`'s input does not move. Endpoint: change in realised accumulated error, `P2` minus `P1`,
paired over tasks. This is the strongest available argument for carrying the second map: it is
the half that survives the warehouse.

**H2 (fusion geometry degrades faster than coverage).** Restocking does not merely remove
coverage; it removes *partners*. For every driveable cell, compute the best available crossing
angle in L0 and in L1. If restocking preferentially removes the diagonal partner (A–D 165°,
B–C 167°) and leaves a same-wall pair (68–78°), then fusion quality falls further than the
5.6 % coverage loss suggests. Endpoint: distribution of best-available crossing angle per
cell, L0 vs L1, and the fraction of cells whose best pair drops below 100° — the threshold
below which the offset is confidently wrong-signed.

**H3 (partial occlusion is a third channel, and it hits accuracy where availability is
unchanged).** A rack top that clips the robot turns a two-marker reading into a one-marker
reading. Measured: along-ray noise **0.88 → 3.21 cm** and bias **−0.30 → −1.79 cm**. So the
reconfiguration can degrade *accuracy* at cells whose *availability* is untouched — an effect
invisible to every availability metric in the study. Endpoint: fraction of detections with
both markers visible, L0 vs L1, and the induced change in realised error at cells with
unchanged `p_use`.

H3 is the one genuinely new thing the changing world contributes to the fusion story, and it
is recoverable **offline** — the L0/L1 captures keep their images (`image_path`), so the
keypoint reading can be run over stored frames. The current capture columns are box-bottom
only (`yolo_bottom_u/v`), so this needs an offline re-read, not a re-capture.

### Crossing table

| | L0 nominal | L1 restocked | What the pair tests |
|---|---|---|---|
| E1 map disagreement | measured | repeat | does the *decision* between maps move? |
| E2 fusion policy | primary | repeat | does `F2`'s advantage survive, and via H2 |
| E3 routing input | primary | repeat | **H1**, the headline |
| E5 placement | measured | repeat | does the placement rule survive? |

Every arm is fitted and frozen on L0 and scored on L1. No arm is refitted on L1 — an
out-of-date field is the object of study, not a nuisance.

---

## World requirements

The current world (`warehouse_full_4cam`) blocks parts of this design. These are requirements
for the v2 layout, in priority order.

**W1 — Put the cameras on the building.** They sit at **z = 6.10 m** while the walls top out
at **4.50 m**: they float 1.6 m above the warehouse with an unobstructed bird's-eye view no
wall-mounted camera has. Consequences of fixing it: physically defensible; sight-lines graze
rack tops instead of clearing them; and the reconfiguration gets much stronger than its
current 5.6 % of covered cells, because grazing sight-lines are the ones a restock breaks.

**W2 — Populate the 80–120° crossing-angle gap.** All four cameras are on the two long walls
at (±6, ±10), so pair angles cluster at 68/78°, 122/124° and 165/167° with **nothing near
90°**. That is exactly where the two fusion mechanisms disagree — random noise is *optimal* at
90° (×0.61) while the lean is only ×0.71 there and cancels at 180°. **The world as built
cannot discriminate the two mechanisms**, which is E2's whole mechanistic claim. Fix: cameras
on the east and west walls, which currently have none.

**W3 — Vary camera height and pitch.** All four are identical (6.10 m, pitch 0.92 rad), so the
camera-frame aspect `slant/height` is the same for all of them and R-weighting degenerates to
range-weighting. Heterogeneous mounting makes the uncertainty map genuinely camera-dependent,
which is what `F2` vs `F1` needs in order to mean anything.

**W4 — Design the map disagreement in.** One camera far and high (broad coverage, poor
precision), one near and low (precise, narrow). Then availability and uncertainty disagree in
a *known, designed* region rather than at an incidental 42.2 % of the floor, and E1 becomes a
prediction instead of an observation.

**W5 — Kill the zero-detection third.** Four cameras on two walls of a 24 × 20 m room means
each camera never sees the far half, so about a third of camera-by-block units contain no
detections at all and have undefined skill. That discards a third of the design.

**W6 — Aisle topology that offers real alternatives.** G3 needs at least three topologically
distinct corridors per task; the current world averages 1.85 m of route separation, i.e.
variations on one corridor rather than alternatives. A loop or grid aisle structure rather
than a comb, so a detour goes *around* something.

**The tension to resolve explicitly.** W1 and W5 pull against each other — lowering the
cameras reduces coverage and worsens the zero-detection problem. Only two consistent
resolutions exist: lower the cameras **and** shrink the footprint, or lower the cameras **and**
add side-wall cameras (which W2 wants anyway). The second also serves W2, W3 and W5 at once,
so it is the recommended one.
