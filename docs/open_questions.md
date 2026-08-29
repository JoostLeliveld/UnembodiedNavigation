# What is unresolved

Current as of 2026-08-29. Each entry says what is not known, why it matters, and what would
settle it. Everything here was checked against the code and the drives on disk on that date;
re-check before acting on any of it.

Companion to [`PLAN.md`](../PLAN.md) (what the paper has to earn),
[`HOW_IT_WORKS.md`](../HOW_IT_WORKS.md) (how the pipeline works) and
[`localization_metrics.md`](localization_metrics.md) (which numbers may be compared).

---

## The gate: no fusion result is currently frozen

The 108 drives in `logs/studies/fusion_on_fixed_routes/drives_full_20260829` were produced by
the pre-repair runtime. They carry `logging_schema_version: 3` and no
`correction_assimilations.csv`, so a filter update cannot be identified from the log — only
inferred from a timestamp, which is what the contract forbids. They are diagnostic evidence
and nothing more.

The repaired runtime writes schema 4 and logs one assimilation row per detector batch. Until
a schema-4 campaign exists and `logs/studies/fusion_on_fixed_routes/frozen_runs.json` names a
complete, provenance-homogeneous set of its runs, there is no fusion number to report.

---

## Open questions, in order of how much they could move a conclusion

### 1. How far does the operational heading actually drift?

The camera measures position only. In `coupled` mode a position correction may move heading
through the position–heading cross-covariance; the bottom-centre is only weakly
heading-sensitive, so it cannot correct a badly wrong heading, and the admission check cannot
act as a heading gate.

Half the gate is measured offline (`experiments/measurement_commissioning/heading_gate.py`):
heading error disturbs the predicted pixel less than the detector's own noise until about
14°. The other half — the drift a driving robot actually accumulates — needs schema-4 drives
that log the estimator heading and its joint covariance and score heading at its own stamp.

This matters beyond heading: heading error is shared by every camera, so if it is large, the
independence assumption fails in the one way fusion cannot survive (see question 2).

### 2. How much of the cameras' error is shared?

If several cameras err the same way — through the robot-shape model, the projection, common
timing, or a shared heading error — then adding precisions makes the fused covariance
confidently wrong while the mean barely improves. The runtime can represent this:
`fusion_common_mode_std_m` inflates the fused covariance by a stated shared standard
deviation, and `bias_floor_*_slope_m_per_m` gives each camera a floor that repeated looks
cannot shrink. **Both default to zero, and no commissioned value for either currently exists.**

A previous attempt commissioned 0.032 m for the shared term. It is withdrawn: the number came
from scoring the fused answer 0.25 s after the instant it described, which charged the robot's
own travel to the fusion rule. Any replacement must be measured with every quantity scored at
its own stamp.

This is the question `manager_fusion_rule: independent` (precisions add) versus `network`
(precisions add, divided by N) exists to answer, and it is currently unanswered on a valid
pipeline.

### 3. Does the detector ever report a robot where there is none?

The commissioning capture kept one empty frame per camera, so the false-positive rate is
unmeasured — and it is half the definition of a usable sighting. A small capture of the empty
warehouse closes it. Do this before the availability model, because it changes what that model
is counting.

### 4. Does availability learned from the robot's own estimate match availability learned
with the true pose?

Measured offline, it does not: re-judging the same detections from a pose 10 cm / 2° wrong
drops the usable rate from 0.70 to 0.56. So the availability map must be learned at the belief
quality the robot will actually have. The truth-free operational dataset that needs does not
exist yet.

### 5. Why are 21% of admitted boxes taller than the projected shape predicts?

They carry about 1.2 cm more lean than the rest, and occlusion does not explain it. Never
chased. It is a property of the observation model, so it would move every arm equally.

### 6. What owns the tail?

On the pre-repair drives, camera C read several times worse than any other camera at the same
range, and camera A was worst closer than 6 m. Both need re-measuring on schema-4 drives
before anything is concluded — the tail may simply be the alignment defect.

---

## Known limitations of the current implementation

**The covariance update is only valid at unit gain.** `belief_correction.compute_update` uses
`next_S = S_eff − gain_scale · Γ Σ⁻¹ Γᵀ`, which is the Kalman form and is correct only when
`gain_scale == 1`. The fused metric path — the one every fusion arm uses — always passes 1.0,
so the active experiment is unaffected. `gain_scale` is set to something else only by
`PixelMeasurementSource`, which scales the gain by the visibility GP's predicted visibility.
That path is the previous confidence-based method, retained as a planning baseline. If it is
driven again, the covariance it reports is inconsistent with its own mean and the update
should be switched to Joseph form first.

**`used == 1` is not a clean per-camera sample.** A reading is admitted partly by agreeing
with the other cameras, which is close to agreeing with the truth. Report admitted and
unconditional per-camera error side by side, never admitted alone.

**A 16 m admission range limit is a proposal, not a feature.** The `unbiased_observation`
study found that truncated boxes beyond about 16 m own the residual lean, and that refusing
them trades availability for bias. Nothing in the runtime implements such a limit.

---

## Facts worth not re-deriving

**The prize on these routes is small.** On the 30 m route the robot drives blind for about
3.1 m, longest single stretch 2.78 m, and dead-reckoning drift is 1.6–2.3 cm per metre — so
the whole outage costs 4–6 cm. Drift accumulates per **metre**, not per second, so driving
faster does not enlarge it. Expect fusion-rule arms to tie on median error; the discriminating
quantity is the stated uncertainty, and the long route is where an over-claim can compound.

**Replay is the right instrument for a fixed path, and only for a fixed path.** The route is
frozen, so what the cameras saw is an input: every rule can be run over one recorded
observation stream with no run-to-run variance. It cannot answer anything where the belief
steers the robot — which is most of what matters, because a better belief predicts a better
box and admits more sightings. That coupling is why the live drives exist.

**Two mistakes that cost real time, both easy to repeat.** Propagating the mean with
odometry's *world-frame* increment while the Jacobian assumes the increment depends on the
filter's own heading — heading then diverges to tens of degrees; read the increment in the
body frame and re-apply it with the filter's heading. And reusing one `CorrectionOutcome`
across two arms in a test, because committing a correction writes its result back into the
outcome object.
