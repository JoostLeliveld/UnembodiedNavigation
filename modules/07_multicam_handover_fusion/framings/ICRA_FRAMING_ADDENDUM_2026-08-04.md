# Framing addendum, 2026-08-04 — the evidence has converged back on the July framing

[Back to framings](README.md) · framing-of-record: [`ICRA_FRAMING_2026-07-22`](ICRA_FRAMING_2026-07-22.md)

**This is an addendum, not a replacement.** The 2026-07-22 framing-of-record and its
locked decisions (moderate runway, Gazebo-only/Toro-framed, headline = safe operating
envelope, novelty = systems + deployment-envelope characterization) stand. Two things
changed underneath it, and one of them is a course correction.

## 1. What changed

The **2026-07-30 ICRA workstream** narrowed the claim to:

> *Availability `p_use` and conditional accuracy `R_cond` are separate fields;
> propagating both through EFE as a hit/miss mixture changes route choice.*

Seven studies later, that narrowing is not supported and should be retired as a
headline. What the evidence actually says:

| the narrowing assumed | measured |
|---|---|
| `R_cond` is the unmeasured blocker | `R_cond` was never data-blocked — 1425/1426 detections associate within 0.15 s. It is **bias**-blocked. |
| the interesting field is conditional *accuracy* | the dominant term is per-camera **systematic bias**: correcting it takes belief NEES from 8.51 to **1.06** on a held-out capture (calibrated = 1.39) |
| per-camera fields beat pooled constants | per-camera `R_cond` went from losing to a pooled constant by 1.2 nats to winning by 0.04 — a **tie** |
| the payoff is route choice | untested, and the earlier C1-vs-C2 route-choice null in the 80 %-coverage world says it is the riskiest available verb |

Meanwhile the July framing's own nulls were *reinforced*: "more cameras is always
better" ✗ (uniform fusion still loses to the best single camera), "a nominal accuracy
win" ✗ (nominal remains a tie). Nothing in the last week contradicts the
framing-of-record. The July-30 narrowing was the drift, not the framing.

## 2. One new claim to add

**C5 — The dominant error in an infrastructure-camera network is per-camera
*systematic* error, and correcting it is a gated, GT-free, per-camera decision — not a
universal fitting exercise.**

Three components, all measured:

1. **Bias dominates noise.** The deployed along-bearing correction has one degree of
   freedom and leaves the lateral component untouched — camera C retains +78 mm. Every
   variance model on this data is limited by bias *transfer*, not by its variance form
   (fitting a bias in one region and applying it in another costs camera A 47 nats and
   drops empirical 90 % coverage to 13 %).
   → [`external_camera_bias_model/exp1`](../../../logs/studies/external_camera_bias_model/exp1_residual_characterization/),
   [`projection_amplification/exp1`](../../../logs/studies/projection_amplification/exp1_geometry_vs_detector/)
2. **A one-parameter gated fix restores belief calibration.** Adding one cross-bearing
   constant, gated on `|b_cross|/σ_cross ≥ 1.2`: NEES @detections **8.51 → 1.06** on the
   capture held out of the calibration fit, median point error 0.046 → 0.022 m, camera C
   oracle bias 0.077 → 0.002 m. Ungated, the same fix **harms** camera A by 27 mm.
   → [`external_camera_bias_model/exp2`](../../../logs/studies/external_camera_bias_model/exp2_two_dof_bias/),
   [`operational_residual_rcond/exp3`](../../../logs/studies/operational_residual_rcond/exp3_two_dof_rcond/)
3. **The gate is deployable — and its limits are measured.** It can be computed
   without ground truth (agrees with the oracle on 3 of 4 cameras; the textbook state
   correction is degenerate and must not be used). A large resolvable bias is decided
   from **20 detections**; every marginal camera is undecidable below a few hundred, and
   the small-sample failure mode is *false calibrate* — the harmful direction.
   → [`network_commissioning_realism/exp1`](../../../logs/studies/network_commissioning_realism/exp1_gate_without_truth/)

C5 sits under C1 rather than beside it: it supplies the error mechanism that dominates
the safe operating envelope, and it is exactly the kind of "deployable, GT-free policy"
C1 promises.

## 3. One new null to state up front

- ✗ **per-camera calibration/covariance can be fitted for every camera.** No. It is
  safely fittable only where the systematic is resolvable against that camera's own
  scatter *and* enough detections have accumulated. In a 50–200-camera warehouse most
  cameras satisfy neither, and fitting anyway is worse than not fitting. **The
  deployable policy is: correct the outliers, leave the rest raw.**

This null is a strength, not an apology: it is the same shape as the existing
"selection, not redundancy" null, and it converts a modelling paper into a deployment
paper. It also explains *why* selection beats redundancy — a network is heterogeneous
in ways per-camera fitting cannot fix at realistic sample sizes.

## 4. What the mixture result becomes

The hit/miss mixture keeps its place, demoted from headline verb to **belief-propagation
correctness** under C1's overconfidence story:

> Folding availability into a single `R` understates the posterior position trace by
> ~90 % at the runtime operating point, and the equivalent covariance varies 6.6–10.8×
> with the prior alone — so no cached `R_plan(s)` can be correct in principle.

Analytic, no data needed, bit-for-bit verified against the shipped CasADi expression.
→ [`planner_covariance_branching/exp1`](../../../logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/)

It is a *correctness* argument about the planner's uncertainty model, which is what C1
needs. Whether it changes route choice is a separate, riskier experiment that should be
run but must not carry the paper.

## 5. Scope — settled, see the register

The realism questions this addendum originally listed as "gaps" are now stated as
assumptions and deferrals in
[`ASSUMPTIONS_AND_DEFERRALS_2026-08-04`](ASSUMPTIONS_AND_DEFERRALS_2026-08-04.md), and are
**not open work**. The paper is Gazebo-based with a sim-trained detector by locked
decision; holding the calibration path to a stricter realism standard than the detector was
inconsistent and was costing runway.

The two that mattered, resolved:

- **Commissioning reference (A1).** A commissioning pass with a reference trajectory of
  known accuracy is *assumed* — that is how intrinsics and extrinsics are already
  commissioned in any real installation, so one more per-camera constant needs no new
  infrastructure. The GT-free gate becomes supporting evidence that the path exists
  (D1), not a prerequisite.
- **Sample efficiency (A2).** The gate is decided once per camera at commissioning, so
  the "marginal cameras are undecidable at small N" measurement is a
  **commissioning-duration requirement**, not a failure mode: an outlier is found within
  about one pass, a boundary camera needs a longer pass or stays raw by default. Same
  numbers, different meaning.

**Consequence: C5 is closed.** The mechanism is measured, the fix is implemented and
gated, its data requirement is quantified, and the belief-calibration gate passes on a
held-out capture. Nothing on C5 is blocked.

## 6. What to do and not do

The one genuinely open thing is the **headline**, C1, and it is open because *every result
in this workstream is offline*. C1 claims a policy stays breach-free where the state of
practice drives into breaches — a closed-loop claim with no closed-loop evidence.

**Do**, in order: (1) drift as controlled injection (A4) — cheap, assets exist, upgrades C5
into the lifecycle capability C3 already claims; (2) the closed-loop safety consequence —
corrected vs uncorrected belief, breach rate and NIS/NEES calibration. Keep the mixture as
a correctness argument.

**Do not:** promote per-camera `R_cond` (it ties a pooled constant); spend runway on a
route-choice demonstration; re-open anything in the register; point campaign configs at
`projection_calibration_v3` without deciding what happens to the v2-locked comparisons.
