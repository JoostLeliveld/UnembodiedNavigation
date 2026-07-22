# ICRA framing-of-record — fault-tolerant multi-camera warehouse localization

[Back to framings](README.md) · consolidates **Framing A** for an ICRA submission ·
results layer: [`RESULTS_SECTION_DRAFT.md`](RESULTS_SECTION_DRAFT.md)

> **Locked decisions (2026-07-22, with J. Leliveld).** These three set the scope of
> everything below and should not be relitigated without a reason.
>
> | Decision | Choice | Consequence |
> |---|---|---|
> | Runway | **Moderate (~3–4 months)** | Detector can gate; ~20–60 recorded runs + offline replay expansion + a reduced closed-loop campaign are all reachable. Not the full stretch campaign. |
> | Evidence base | **Gazebo-only, Toro-framed** | No physical-camera collection. Credibility rests on (a) an exact Toro reproduction and (b) projection/calibration validated against physically-grounded GT. Controlled fault injection is a *reason* to use sim, not an apology. |
> | Headline | **Fault-tolerance proof-of-concept** | Lead with "static calibration cannot detect/contain a bad camera; our health-aware system can." NOT "learned GP covariance beats calibration" (contradicted by our own nulls). |

---

## 1. The one-sentence thesis

> In a real warehouse, fixed-camera usefulness varies with robot position, availability,
> and calibration health; **static per-camera calibration (Toro et al.) is safe only while
> every camera stays healthy.** A localization service that combines a spatial reliability
> prior, calibrated frame-level detector evidence, and online innovation-health monitoring
> **detects and contains a faulty camera** — keeping localization and closed-loop navigation
> from degrading below simply dropping that camera — where static calibration silently
> trusts the stale measurement.

The proof-of-concept is the *deployable service*: everything the method consumes is
available operationally (no ground truth), it reproduces the real-world Toro baseline, and
it closes the loop with the navigation stack from the IWAI paper.

## 2. Contribution statement (calibrated to what we can defend)

We contribute a **multi-timescale observation-reliability model for fixed external cameras**
and show it delivers **fault tolerance** that static calibration cannot:

1. A per-camera **spatial reliability field** over the drivable region (availability ×
   conditional usability), learned from operational logs — used because *location matters*,
   not because a learned field beats geometry (see §4).
2. **Calibrated** frame-level detector confidence (isotonic/logistic), kept separate from
   spatial reliability and combined through a held-out trust stacker.
3. **Online camera-health monitoring** (innovation + bias EWMA, debounced) that flags a
   drifting/biased camera from operational evidence alone — the load-bearing mechanism.
4. A **trust→covariance map** feeding a robust sequential filter, and a **planning adapter**
   that consumes only spatially predictable reliability (never the current frame's
   confidence copied across the horizon).

Evaluated in high-fidelity Gazebo against an **exact Toro reproduction**, under camera-count
variation, dropout/latency, and **controlled calibration drift**, plus a reduced-scope
closed-loop confirmation.

## 3. Positioning vs Toro Diz et al. (honest extension table)

| Toro Diz et al. | This work | Demonstrated? |
|---|---|---|
| Static per-location error measurements | Operational detector + state-estimate logs | pipeline real; fits pending detector gate |
| Nearest calibration point | Continuous spatial reliability field | ranking win real (AUROC 0.78 vs 0.50) |
| Nominal, always-healthy cameras | **Camera outages, latency, calibration drift** | **the headline — detection real (WP5), containment pending** |
| Localization RMSE + smoothness | Localization consistency **and closed-loop safety** | closed-loop reduced-scope, pending |
| 2-cam vs 3-cam grouping | Every subset + timed failures + recovery | offline harness wired |
| 3 cams, 5.5×3 m, one person, straight lines | 4 cams, full warehouse, closed-loop robot | world real, campaign pending |

## 4. Claim discipline — what we do NOT claim (nulls as scoping)

Our own single-camera results bound the honest story. State these as design decisions, not
buried caveats:

- **We do not claim uncertain-input GP training helps.** At realistic belief uncertainty
  (~0.06 m) it is indistinguishable from the mean-only spatial GP (fold variance ≫ mode
  gaps). → We use the simpler mean-input spatial field.
  Source: [`expB_real_prediction/RESULTS.md`](../../../logs/studies/single_camera_uigp_reliability/expB_real_prediction/RESULTS.md).
- **We do not claim a learned service map beats first-principles geometry.** In a
  low-occlusion warehouse, 95% of misses are simply out-of-FOV and geometry predicts that
  for free. → Geometry is a legitimate baseline; our win is under *faults*, which geometry
  cannot see. Source: [`expB_falsesafe_baselines/RESULTS.md`](../../../logs/studies/single_camera_uigp_reliability/expB_falsesafe_baselines/RESULTS.md).
- **We do not claim more cameras is always better.** Overlap is only 7–13% of the floor;
  naive all-camera fusion can lose to the best single camera. → This is *why* the
  contribution is containment/selection, not redundancy.
- **We do not use ground truth operationally.** GT scores results only; a CI firewall
  rejects any operational import of evaluation-only fields.

Turning each null into a scoping decision is a strength in review, not a weakness.

## 5. Right-sized experiment slate (E0–E8 → ICRA)

| Exp | Role for ICRA | Why | Cost |
|---|---|---|---|
| **E0** component + Toro-calibration validation | **IN — foundational** | Gazebo-Toro credibility hinges on it; mostly real already (R0) | done + multi-session recapture |
| E1 spatial reliability prediction | **SUPPORTING (demoted)** | Method justification only ("location matters"); most exposed to the nulls — never a headline | offline |
| E2 conditional-covariance calibration | **SUPPORTING** | Coverage + sharpness so the fusion covariance is honest | offline |
| **E3** nominal multi-cam localization (B0–B8) | **IN — the "nominal tie"** | Narrative pivot: nominally everything is comparable → differences emerge only under faults | offline replay |
| **E4** camera count / geometry subsets | **IN** | Establishes the 7–13% overlap reality + naive-fusion-can-hurt | offline (cheap) |
| **E5** dropout / latency | **IN — fault mode** | Core fault tolerance; offline masking sweep | offline (cheap) |
| **E6** calibration drift | **IN — centrepiece** | Health monitor + Δ_fault containment; extends WP5 to multi-cam | offline + controlled |
| **E7** selection vs fusion | **IN — supporting mechanism** | The containment policy: when to fuse vs select/robust-subset | offline |
| **E8** closed-loop navigation | **IN — reduced scope** | The PoC payoff; extends IWAI C1/C2 to multi-cam fault conditions; ~1 Hz live-bound, so a small confirming matrix (not all 5×4×5) | live, limited |
| Stretch (spatial heteroscedastic R, uncertain-input GP, physical perturbation, multi-robot) | **OUT — future work** | Explicitly deferred; not in the central claim | — |

**Baselines for the paper (from B0–B9):** headline comparison B1 constant-cov · **B2 Toro** ·
B6 full health-aware · B9 oracle (eval-only upper bound). B3/B4/B5/B7/B8 appear as an
ablation row-set. **Pre-registered primary comparisons:** full-vs-Toro, full-vs-GP-only,
fusion-vs-selection — everything else is exploratory.

## 6. Critical path (next ~3–4 months)

> The load-bearing fault-containment block (E5/E6/E7) is pre-registered in
> [`PREREGISTRATION_fault_containment_2026-07-22`](../../../experiments/multicamera_fusion_extension/PREREGISTRATION_fault_containment_2026-07-22.md)
> — hypotheses, Δ_fault gate, critical-failure stop rule, and the executable-now
> vs. must-build harness split (top build item: a **health-aware fusion replay mode** —
> the B6 "full method" condition does not exist yet).


```
[NOW] detector retrain to GATE ──────────────►  gate: ≥0.90 @ ≤12 m, ≥0.75 @ 12–16 m
   (in flight; v2_640_diag ~0.40 mid-range = below gate — HARD BLOCKER #1)
        │
        ▼
record multi-session handover captures (operational recorder: odom+perception+GT)
   target: 4 routes × ~5 seeds × ~3 sessions (lighting/clutter/direction/startup)
        │
        ▼
offline replay expansion  ──►  E3 nominal · E4 subsets · E5 dropout · E6 drift · E7 select
   (one recording → all subsets × methods; validated harness run_containment_pilot.py)
        │
        ▼
E6 Δ_fault containment result (centrepiece)  +  E8 reduced closed-loop confirmation
        │
        ▼
evidence promotion (research_story ch.08/09 + registry) → write → submit
```

**Blocker ranking:** (1) detector below gate — everything downstream is detector-limited and
n=1 until fixed; (2) only one commissioning drive so far — need multi-session for run-level
CIs; (3) ~1 Hz live rate bounds closed-loop (offline replay unaffected — runs on sim time).

## 7. Paper section outline (ICRA, 6 pp + refs)

1. **Introduction** — warehouse fixed cameras give useful but *uneven and time-varying*
   localization; static calibration is safe only while healthy; contribution list (§2).
2. **Related work** — external-camera localization (Toro), detector calibration, robust
   filtering / fault detection, reliability-aware planning (cite IWAI).
3. **System & problem** — N fixed cameras, state model, three reliability quantities kept
   separate, the GT firewall.
4. **Method** — spatial reliability field; calibrated confidence + trust stacker; health
   monitor (EWMA/bias/debounce); trust→covariance; robust fusion; planning adapter.
5. **Experiments** — E0 validation + Toro reproduction; E3 nominal tie; E4/E5/E6 fault
   sweeps (E6 = Δ_fault centrepiece); E7 select-vs-fuse; E8 closed-loop confirmation.
6. **Results & discussion** — nominal comparable → fault divergence; containment; nulls as
   scoping; limitations (detector-limited, sim-only, narrow overlap, ~1 Hz).
7. **Conclusion / future work** — the stretch items.

**Working titles:** *Fault-Tolerant Multi-Camera Localization for Warehouse Robots via
Online Reliability and Health Monitoring* · *Containing Camera Faults: Reliability- and
Health-Aware Multi-Camera Fusion for Warehouse Navigation*.

## 8. Risks & mitigations

- **Detector never gates cleanly.** Mitigation: bound all claims to the observed operational
  band and report detector-limitation as a first-class limitation (draft already does). But
  detector-limited n=1 is *not* an ICRA result — getting to gate is the top priority.
- **Reviewer: "Gazebo-only."** Mitigation: exact Toro reproduction; projection validated vs
  physically-grounded GT; controlled measured fault injection is infeasible on real hardware
  — a genuine reason for sim, stated up front.
- **Nominal result is a tie.** Mitigation: this is the intended narrative (tie → fault
  divergence), not a failure — but pre-register it so it doesn't read as a negative surprise.
- **Δ_fault comes out ≈0 for the health-aware method AND for naive fusion.** Then the
  containment claim is empty. Mitigation: the pilot already shows naive fusion is
  non-monotone in drift (moderate gate-evading bias pollutes worst) — that gap is the claim;
  confirm on real capture before writing the number.
