# Assumptions & deferrals register — ICRA-2027

[Back to framings](README.md) · framing-of-record: [`ICRA_FRAMING_2026-07-22`](ICRA_FRAMING_2026-07-22.md) ·
addendum: [`ICRA_FRAMING_ADDENDUM_2026-08-04`](ICRA_FRAMING_ADDENDUM_2026-08-04.md)

**Purpose: end the realism spiral.** Every claim in this workstream is Gazebo-based and
the detector is trained on simulated imagery — that is a *locked decision*, not an
oversight. Demanding deployment-grade realism from the calibration path while accepting a
sim-trained detector is inconsistent, and it was costing more than it bought.

This register states what we **assume**, what we **defer**, and why each is defensible.
Anything on this list is settled: it does not get re-litigated as a blocker, and it goes
into the paper as explicit scope. Reviewers reward a stated assumption and punish a hidden
one.

## Assumptions (stated, defensible, not open work)

| # | Assumption | Why it is defensible | What it unblocks |
|---|---|---|---|
| **A1** | A **commissioning pass with a reference trajectory of known accuracy** is available. In sim that is Gazebo truth; in practice a total-station survey, a LiDAR-SLAM pass, or a fiducial route. | This is how camera intrinsics and extrinsics are *already* commissioned in every real installation. Fitting one more per-camera constant from the same pass adds no new infrastructure. | The per-camera bias constants in `projection_calibration_v3` are legitimate. Their GT provenance is the assumption, not a flaw. |
| **A2** | The **CALIBRATE / leave-raw gate is decided once per camera at commissioning**, using the whole commissioning dataset — not online from a handful of detections. | Commissioning is precisely when a controlled pass with adequate coverage exists. | Reframes the sample-efficiency result: it is a **commissioning-duration requirement**, not a failure mode. See below. |
| **A3** | Detector imperfection enters the model **only** through the measured `p_det` / `p_qual` fields. Real-image transfer is out of scope. | Locked decision (Gazebo-only, Toro-framed). The contribution is the observation model over a detector, not the detector. | No detector retraining is on the critical path. |
| **A4** | Network geometry is **static within a run**. Calibration drift is studied by **controlled injection** (E6), not by natural decay. | Controlled fault injection is a *reason* to use simulation, already argued in the framing-of-record. | The drift experiment is a fault-injection study, not a longitudinal one. |
| **A5** | **2-D position state**; heading is odometry-driven under the locked configuration. | Existing locked configuration; the camera measurement is a ground-plane point. | No heading-observability work. |
| **A6** | **One robot.** No multi-target association ambiguity. | Matches every capture and campaign in the repo. | The association factor `p_qual` stays a per-frame quality gate. |
| **A7** | Claims are about **regimes**, not camera counts: one large-resolvable bias, two marginal, one unresolvable. | Four cameras cannot license a statement about a hundred. The sample-efficiency curves carry whatever scaling argument exists. | No pressure to build a 50-camera world. |
| **A8** | Drift is injected **geometrically**: the camera pose moves and the recorded pixel is re-imaged through it, but the **image content is unchanged** — the detector is not re-run on a re-rendered view. | The quantity under study is the world-projection error a stale calibration produces, and that is fully determined by the geometry. Re-rendering would additionally perturb the detector's response to a ≤2° viewpoint change, which is far inside the viewpoint spread the detector was trained over (`split_mode: spatial_yaw_bucket`) and is second-order to a projection error that grows linearly with range. Strictly weaker than A3, which already excludes detector realism entirely. | The drift ladder is a pure-geometry sweep over real recorded detections: no re-render, no re-capture, no detector retraining. |

### A8: how much does it weaken the paper?

**Very little, and less than A3 already does.** The paper does not claim anything about how
YOLO responds to a moved camera; it claims that a *stale projection calibration* produces a
growing world-space bias and that the bias is detectable from operational signals. Both are
geometric statements. The honest residual risk is narrow: at large drift a real detector
might also degrade `p_det`, which would make drift *easier* to notice, not harder — so A8
is conservative with respect to the detection claim. State it in one sentence next to A3
and move on; a reviewer who accepts a sim-trained detector cannot consistently reject a
sim-geometry fault injection.

### A2 restated, because it changes the result's meaning

The measurement that looked like a scaling problem —

> camera C is decided from **20 detections**; marginal cameras stay under 60 % correct
> even with all available data, and the small-sample failure mode is *false calibrate*

— under A2 is a **specification for the commissioning pass**: a camera whose lateral bias
is large enough to matter is identified within roughly one pass, and a camera near the
boundary must either be given a longer pass or be left raw by default. Same numbers, and
now they are a design requirement rather than a negative result. The conservative default
(leave raw) is what makes that safe.

## Deferred to future work (named in the paper, not attempted)

| # | Deferred | Why deferring is safe |
|---|---|---|
| **D1** | **Fully GT-free commissioning end-to-end.** The gate is already GT-free-computable and the operational estimator recovers camera C's bias to 0.084 vs 0.077 m, but the constants themselves were fitted against truth (A1). | A1 makes this an enhancement, not a prerequisite. Report the GT-free gate as evidence the path exists. |
| **D2** | **Spatial `R_cond(x)` field.** 1426 detections over three captures cannot support it, and per-camera constants only tie a pooled matrix. | Already retired as a claim; attempting it would be fitting noise. |
| **D3** | **Longitudinal drift** over weeks/months of real operation. | Covered in the controlled-injection form by A4. |
| **D4** | **Real-image / real-hardware validation.** | Locked decision; stated up front in the framing-of-record's nulls. |
| **D5** | **Route-choice discrimination** from the hit/miss mixture. | Retired as a headline in the addendum; the mixture's correctness argument does not depend on it. |
| **D6** | **Larger networks** (50–200 cameras) and their commissioning scheduling. | A7. |

## Runway item #1 is DONE (2026-08-04)

`experiments/calibration_drift_lifecycle` →
`logs/studies/calibration_drift_lifecycle/exp1_stale_correction/`. C5 is now a **lifecycle**
claim: `CALIBRATE` is a decision with a detectable expiry condition.

Three results, all on the capture held out of the v3 fit:

1. **A stale correction becomes harmful at 0.25° of yaw** on camera C — the camera the
   correction helps most (0.043 → 0.100 m, against 0.094 m for doing nothing). Lifecycle
   risk is proportional to correction size, so it lands entirely on the cameras the
   "correct the outliers" policy acts on. Cameras left RAW cannot be harmed — the
   conservative default is also the lower-maintenance one.
2. **The commissioning gate cannot be reused as the in-service monitor.** Its absolute form
   fires at rest on A (10.2) and B (5.0) against a 1.2 threshold, and it can be *masked*:
   camera B's ratio *falls* to 0.31 at 0.25° yaw when the induced bias cancels the resident
   one. Non-monotone, so not merely late.
3. **The change form of the same statistic works.**
   `|b_cross(δ) − b_cross(0)| / σ_cross(0)` is monotone on all 4 cameras × both fault types
   and detects at 0.1° yaw / 0.025 m — **one rung before harm**. Commissioning asks "is
   there a bias worth correcting"; service asks "has this camera moved". Different
   questions, same residual, different statistic.

Adds no new deferral beyond **A8**. Follow-up that needs no new capture: a drift *ramp*
(this study injects a step, so it measures a detection *threshold*, not a detection
*latency in time*).

## What this leaves open — the actual critical path

With the register applied, **C5 is closed** and now carries its lifecycle half: the error
mechanism is measured, the fix is implemented and gated, the gate's data requirement is
quantified, the belief-calibration gate passes on a held-out capture, and the stale-fix
failure mode is bounded and detectable. Nothing on it is blocked.

What remains genuinely open is the **headline**, C1 — the safe operating envelope — and it
is open for one reason only: **every result in this workstream is offline.** C1 claims a
policy stays breach-free where the state of practice drives into breaches. That is a
closed-loop claim and it has no closed-loop evidence yet.

So the runway goes there, in this order:

1. ~~**Drift as controlled injection (A4)**~~ — **DONE 2026-08-04**, see above.
2. **Closed-loop safety consequence** — corrected vs uncorrected belief, breach rate and
   NIS/NEES calibration under the existing containment machinery. **This is the headline and
   it is now the only thing between the workstream and the paper.**
   Two decisions are needed before it can run, and both are the user's:
   - **Which calibration the campaign points at.** No campaign config points at v3 yet;
     switching invalidates the v2-locked comparisons. The natural design makes it the
     independent variable (v2-raw vs v3-corrected as the two arms), which needs no locked
     comparison to be reopened.
   - **Machine time.** Real Gazebo runs, per the no-synthetic-data rule.
3. Everything else is upside.
