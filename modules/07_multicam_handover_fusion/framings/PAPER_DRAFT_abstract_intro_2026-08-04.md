# Paper draft — abstract & introduction (2026-08-04)

[Back to framings](README.md) · scope: [`ASSUMPTIONS_AND_DEFERRALS`](ASSUMPTIONS_AND_DEFERRALS_2026-08-04.md)
· framing-of-record: [`ICRA_FRAMING_2026-07-22`](ICRA_FRAMING_2026-07-22.md)

Written against evidence that **exists and is measured**. One sentence in the abstract
is flagged as requiring the closed-loop campaign; everything else is backed today.

## Title candidates

1. **You Cannot Calibrate Your Way Out: Correlated Camera Bias and Belief Honesty in Infrastructure-Guided Navigation**
2. Honest Belief for Infrastructure-Camera Localization: Accounting for What Calibration Cannot Remove
3. What Repeated Looks Are Worth: Correlated Bias in Multi-Camera Robot Localization

Preference: **(1)**. It states a negative result the reader can disagree with, which is
what makes them read on, and the subtitle names the positive contribution.

## Abstract

Infrastructure cameras can localize a mobile robot that carries no localization sensor
of its own — but they do so through a measurement channel whose dominant error is not
noise. On a four-camera warehouse network we measure a persistent per-camera lateral
bias of up to 78 mm: an error that repeats identically on every frame rather than being
redrawn. We show that this *correlated* component, not the stochastic one, is what
breaks the robot's belief. A filter that treats repeated observations from one camera as
independent evidence contracts its covariance toward zero while the floor set by the bias
does not move, so the robot claims 2.8× more precision than it has and its true pose lies
outside its own 95 % confidence region 42 % of the time.

We show the standard remedies do not reach this error. A chi-square innovation gate
rejects 0.2 % of detections, because a 78 mm offset sits comfortably inside any sane
gate. A sharper per-camera noise model makes matters *worse*, tightening the belief
around a biased measurement. Judging a camera by its agreement with the current belief
inverts the diagnosis entirely, because a biased camera captures the belief and the
honest cameras then appear faulty. Calibration helps but cannot be the answer either: we
give a resolvability criterion showing that a per-camera correction is only safely
fittable where the systematic exceeds that camera's own scatter — two of our four
cameras — and that a fitted correction becomes actively harmful after 0.25° of mount
drift.

We therefore make the network account for what it cannot remove. Each camera is judged
against a belief constructed without it — the only reference a biased camera cannot
capture — and the belief is forbidden from claiming to be sharper than the residual bias
of the camera currently informing it. On real recorded multi-camera data this restores
calibration: a stated 5.1 cm against an actual 5.0 cm, with 3.3 % of updates outside the
stated 95 % region against a nominal 5 %, at unchanged positional accuracy. *[FLAGGED —
needs the closed-loop campaign: "and in closed-loop navigation this eliminates the
keep-out breaches an overconfident belief produces."]*

## Introduction

**The setting.** A warehouse robot with no localization sensor of its own is localized
by cameras mounted on the infrastructure. This is attractive because it moves cost and
maintenance off the fleet and onto a shared, serviceable network, and it is the setting
of a growing literature on infrastructure-assisted navigation. It also removes the
robot's fallback: if the network's estimate is wrong, nothing on board disagrees.

**The standing assumption.** Work in this area treats the camera channel the way a
sensor-fusion textbook does — calibrate the cameras once, model what remains as
zero-mean noise with covariance `R`, and let a Kalman filter combine it with odometry.
Everything downstream depends on that covariance being *honest*: a planner that reasons
about collision risk, a safety supervisor that decides when to stop, and any fusion rule
that weights one camera against another are all consuming the filter's stated confidence
rather than its true error.

**What we measure instead.** On a real four-camera network we find the residual after
commissioning is not zero-mean noise. One camera reports the robot 78 mm to one side,
consistently, on every frame it contributes. This is a *systematic* error with a
different statistical character from noise, and the difference is not academic:

- Noise is redrawn each frame, so repeated observations really are new evidence and a
  filter is right to become more confident.
- A bias is the *same* error each frame. Repeated observations carry no new information
  about it at all, yet a filter treats them as though they do — its covariance falls as
  `1/n` toward zero while the actual error floor never moves.

The result is a robot that becomes *more confident while staying wrong*, which is
precisely the failure mode a safety argument cannot tolerate. We measure it directly:
42 % of belief updates place the true pose outside the filter's own 95 % ellipse.

**Why the usual tools miss it.** We evaluate three standard responses and each fails for
a distinct and instructive reason. A chi-square innovation gate — the classical robust
answer — rejects 0.2 % of detections, because a bias small enough to be plausible is by
construction inside the gate. Fitting a better per-camera noise model makes the belief
*more* overconfident, because a sharper `R` around a biased measurement is a tighter
posterior around the wrong answer. And per-camera health monitoring against the current
belief gets the diagnosis backwards: the biased camera drags the belief onto its own
estimate, after which its innovations are small and the *honest* cameras look like the
outliers. We measure this self-confirmation independently at 4.2× on one camera.

**Why calibration is not the answer either.** The obvious reply is to remove the bias at
commissioning. We do exactly that — adding the one degree of freedom the deployed
correction structurally lacked — and it works where it applies, taking one camera's
residual bias from 77 mm to 4 mm. But it does not generalize into a policy:

1. It is only *safely fittable* where the systematic is resolvable against that camera's
   own scatter. Applied to a camera whose bias is below its scatter, the same correction
   costs 27 mm of held-out accuracy. Two of our four cameras qualify.
2. A large bias is identifiable from about 20 detections, but a marginal one is not
   decided reliably by any amount of data we have — and the small-sample failure mode is
   *false* correction, the harmful direction.
3. The correction expires. After 0.25° of mount yaw it is worse than doing nothing, and
   the exposure is proportional to the size of the correction — so the cameras a
   calibration policy helps most are exactly the ones it later endangers.

Calibration is therefore a useful but bounded operation: **correct the outliers, leave
the rest raw** — and accept that a residual bias always remains.

**Our approach.** If the systematic cannot be removed, the belief must account for it.
We make two changes, both of which require a network and neither of which is available
to a single camera:

1. **Judge each camera against a belief built without it.** This is the only reference a
   biased camera cannot capture, and it converts a per-camera lean into an observable
   quantity — the agreement between one camera and the consensus of the others.
2. **Floor the belief by the residual bias of the camera informing it.** The filter may
   not claim to be sharper than the systematic error of its current source. This is a
   statement about what repeated evidence from one camera is *worth*, not a better noise
   model, and it is the part no per-frame covariance can express.

Because the floor is per-camera, how much the robot may trust itself depends on *which*
camera is watching — a quantity that exists only in a network, and the point at which
the calibration policy and the filter become the same system: whatever bias
commissioning leaves behind is exactly the floor the filter must respect.

**Contributions.**

1. A measurement of the error structure of a real infrastructure-camera network showing
   the binding term is a *correlated* per-camera systematic, not noise, together with the
   quantified consequence for belief calibration.
2. Evidence that three standard responses — innovation gating, per-camera noise
   modelling, and belief-referenced health monitoring — do not address it, each for a
   different reason, including a measured self-confirmation effect of 4.2×.
3. A resolvability criterion for when a per-camera correction may be fitted at all, its
   data requirement, and its expiry under mount drift.
4. A network-referenced filter that restores belief calibration at unchanged accuracy by
   accounting for the residual systematic rather than attempting to remove it.
5. An analysis showing the planner inherits the same defect on the availability axis:
   folding "the robot might be seen" into a single inflated `R` understates the predicted
   posterior by ~90 % at the runtime operating point, and no position-indexed `R` can be
   correct because the equivalent covariance varies 6.6–10.8× with the prior alone.

**Scope.** Simulation-based, with a detector trained on simulated imagery; controlled
fault injection is a reason for that choice rather than an apology. Results are on real
recorded runs of a four-camera network and are stated as *regimes* — one large-resolvable
bias, two marginal, one unresolvable — not as a claim about camera count. The belief
calibration result is offline; the closed-loop consequence is the flagged sentence above.

## Full section outline, with fill status

Sections I–VII and IX are backed today. §VIII is the thin one, and it is the reason
there are two shippable versions of this paper.

| § | Section | Filled by | State |
|---|---|---|---|
| I | Introduction | above | **full** |
| II | Related work | infrastructure-camera localization (Toro-Diz et al. = direct baseline), robust filtering (gating / M-estimators / Student-t), camera-network extrinsic calibration. **Gap:** all three treat the post-calibration residual as zero-mean noise | outline only |
| III | Factorized measurement model | `reliability.observation_model` — availability, conditional accuracy, bias, timing, calibration terms, kept separate | **full** (code + 34 tests) |
| IV | What the residual actually is | 78 mm lateral bias, correlated not random; geometry amplifies a fixed pixel error 4.1× within one footprint; bias transfer dominates variance (47 nats, coverage 13 %) | **full** |
| V | Why the standard remedies fail | gate rejects 0.2 %; sharper per-camera `R` makes it worse (4.22→5.11); belief-referenced health monitoring inverts the diagnosis; self-confirmation measured at 4.2× | **full — strongest section** |
| VI | What calibration can and cannot do | one extra parameter takes 77→4 mm; resolvability criterion; 20 detections for an outlier, marginal cameras undecidable; expiry at 0.25°, detection at 0.1° | **full** |
| VII | Honest belief | leave-one-out cross-check + correlation floor; 5.1 cm stated vs 5.0 cm actual, 3.3 % vs 5 % nominal, unchanged RMSE; ablated (floor primary, cross-check ~2×, per-camera vs pooled 6×); generalization-tested leave-one-capture-out | **full** |
| VIII | Consequence for planning | analytic: folding availability into one `R` understates the predicted posterior ~90 %, and no position-indexed `R_plan(s)` can be correct (6.6–10.8× with the prior alone) — **plus** the achievable-precision field: on **15.7 %** of the reachable floor the most-available camera is not the most informative one, and camera C's territory collapses from 25 % to 14.8 % under a precision criterion | **half full** (map done; route-level prediction is the next step) |
| IX | Limitations & scope | the assumptions register (A1–A8, D1–D6) | **full** |
| — | Navigation outcomes | — | **empty** |

### Two shippable versions

**Version A — ship what exists.** Headline moves to the measurement channel: *the
dominant error is correlated, standard tools cannot see it, here is what does.* Complete
today. Exposure: a reviewer asks "what does it change?" and §VIII is a paragraph of
algebra.

**Version B — finish the planning half.** Fill §VIII with the achievable-precision map
and a route-level prediction (days, no machine time), then optionally the closed-loop
campaign. Restores the framing-of-record's navigation headline. Risk: ~6 weeks to
deadline, and the closed-loop step could return null as route-choice did once before.

**Recommendation: aim for B, built in the order that falls back to A.** The two offline
planning steps are days and no machine time. If routes differ in achievable precision,
§VIII becomes real and the closed-loop is worth requesting; if they do not, ship A with
§VIII stated honestly as an analytic limitation. Do not write the navigation claim until
the evidence exists.

## What each claim rests on

| claim | evidence |
|---|---|
| 78 mm lateral bias, correlated | `external_camera_bias_model/exp1` |
| overconfidence: 42 %, 2.8× | `bayesian_filter_showcase/exp1` |
| gate rejects 0.2 %; sharper `R` worse; self-confirmation | `bayesian_filter_showcase/exp1`, `operational_residual_rcond/exp2` |
| correction works (77→4 mm) but must be gated (−27 mm) | `external_camera_bias_model/exp2` |
| belief calibration restored (NEES 8.51→1.06) | `operational_residual_rcond/exp3` |
| 20 detections; marginal undecidable | `network_commissioning_realism/exp1` |
| expiry at 0.25°, detection at 0.1° | `calibration_drift_lifecycle/exp1` |
| honest filter: 5.1 stated / 5.0 actual, 3.3 % | `bayesian_filter_showcase/exp1` |
| planner ~90 %, `R_eff` 6.6–10.8× | `planner_covariance_branching/exp1` |
| coverage picks the wrong camera on 15.7 % of the floor | `achievable_precision_map/exp1` |
| honest belief generalizes (leave-one-capture-out) | `bayesian_filter_showcase/exp2` |
