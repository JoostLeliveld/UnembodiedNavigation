# ICRA framing-of-record — safe navigation on a realistic infrastructure-camera network

[Back to framings](README.md) · evidence layer: [`RESULTS_SECTION_DRAFT.md`](RESULTS_SECTION_DRAFT.md) ·
two-tier experiment contract:
[`PREREGISTRATION_realistic_network_2026-07-22`](../../../experiments/multicamera_fusion_extension/PREREGISTRATION_realistic_network_2026-07-22.md)

> **Locked decisions (2026-07-22, with J. Leliveld) — still in force.**
>
> | Decision | Choice | Consequence |
> |---|---|---|
> | Runway | **Moderate (~3–4 months)** | detector can gate; ~20–60 recorded runs + offline replay + a reduced closed-loop campaign are reachable. |
> | Evidence base | **Gazebo-only, Toro-framed** | no physical collection; credibility rests on an exact Toro reproduction + projection validated vs physically-grounded GT. Controlled fault injection is a *reason* to use sim, not an apology. |
> | Headline | **NOT "learned GP covariance beats calibration"** | contradicted by our own nulls (§3). |
>
> **Refinement (2026-07-22, reason-backed).** The headline is **broadened** from "fault-tolerance
> proof-of-concept" to **"safe navigation on a realistic infrastructure-camera network."** Reason:
> the reviewer red-team showed the fault-containment-*alone* framing is exposed on (a) novelty — a
> health monitor is innovation-gating + covariance inflation, known robust estimation — and (b) the
> containment mechanism is inert where overlap is 7–13% of the floor. The realism framing keeps every
> asset and every null but makes novelty a **systems + deployment-envelope characterization** claim,
> which is defensible where "we contain a fault" is not. Fault-tolerance is now **one pillar (C3)**,
> not the whole paper. The three locked decisions above are unchanged.

---

## 1. Problem statement

A warehouse robot with **no onboard localization** is guided by a network of **fixed infrastructure
cameras** it does not own or control. In any real deployment that network is **partial** (most of the
floor is single-camera; overlap is 7–13%; some zones see nothing), **heterogeneous** (rate,
resolution, viewpoint, hardware — e.g. a CPU camera at ~1 Hz beside GPU cameras), **latent**
(detector/network-bound, ~1 Hz, variable), and **non-stationary** (calibration drifts, shelves
occlude, lighting shifts). Commissioning and upkeep must be **ground-truth-free**.

Prior work treats such networks as *observers* (multi-camera tracking / MOT / re-ID — a database or
human consumes the track) or assumes *onboard* localization (AMR/SLAM). The closed-loop problem —
**navigating a robot safely on a realistic, partial, degrading infrastructure-camera network with no
fallback** — is uncharacterized. We ask: *under what network conditions is infrastructure-guided
navigation safe, and what must the robot know and do to stay safe as coverage runs out and cameras
degrade?* This is the IWAI single-camera reliability-aware planner, generalized to the network **as it
really is**.

## 2. Contribution — the claims we license

**C1 — Headline (a research finding, not a demo): the safe operating envelope.** Safety of
infrastructure-guided navigation degrades **predictably** as a function of *coverage geometry* and
*camera health*; a coverage- and health-aware policy consuming only operational evidence stays
**breach-free across a region of that envelope where the state of practice** (static-calibration
fusion + network-unaware planning) **drives into breaches**. *The contribution is mapping the envelope
and showing a deployable, GT-free policy that widens the safe region.*

**C2 — Coverage as a first-class navigation quantity.** A learned per-camera reliability/coverage
field lets the robot reason about *where it can be localized* and plan routes **and speeds** that keep
it observable — including bounded-risk traversal of blind / single-camera zones. (What tracking never
needs and AMR assumes away.)

**C3 — Online, GT-free health-aware safe degradation (the fault-tolerance pillar).** The system
detects camera degradation (drift/occlusion/dropout) from operational evidence alone (WP5: proven,
0 false alarms / 3700 healthy frames) and degrades safely — down-weight in fusion where overlap
exists, re-route / slow / safe-stop where it doesn't — the lifecycle capability static calibration
structurally lacks.

**C4 — Heterogeneity + latency handled honestly (systems/supporting).** Per-camera trust + age-gating
so an uneven, latent network feeds a real-time control loop without acting on a stale pose.

## 3. What the nulls forbid (state as scoping, up front — a strength in review)

- ✗ **learned reliability beats first-principles geometry.** No — geometry predicts ~95% of coverage
  for free at low occlusion. Geometry is a **legitimate baseline**; the learned field earns its place
  only because it *also* carries detector + health signals geometry cannot see.
  Source: [`expB_falsesafe_baselines/RESULTS.md`](../../../logs/studies/single_camera_uigp_reliability/expB_falsesafe_baselines/RESULTS.md).
- ✗ **uncertain-input GP training helps.** No — indistinguishable at ~0.06 m belief uncertainty. Use
  the mean-input field. Source: [`expA_alpha_sweep/RESULTS.md`](../../../logs/studies/single_camera_uigp_reliability/expA_alpha_sweep/RESULTS.md).
- ✗ **more cameras is always better.** No — 7–13% overlap; naive fusion can lose to the best single
  camera. This is *why* the contribution is coverage/health-aware **selection, not redundancy**.
- ✗ **a nominal accuracy win over static calibration.** No — **nominal is a tie** (pre-registered:
  tie → divergence under realistic/degraded conditions).
- ✗ **formal safety guarantees.** No — an empirical safe-degradation *characterization*, not
  certification.
- ✗ **hardware-validated.** Sim-only, controlled injected faults — a reason for sim, stated up front.
- ✗ **ground truth used operationally.** GT scores results only; a CI firewall rejects any operational
  import of evaluation-only fields.

## 4. Two-tier structure (headline lives in tier 1; tier 2 is upside)

- **Tier 1 — single-camera core (primary; carries C1/C3).** One external camera, **no fallback**, so
  the robot *must move to stay localizable*. Cleanly isolates the health→planning mechanism with **no
  fusion/overlap crutch** — which removes the "you only caught it because a healthy camera anchored
  the belief" and "just drop the camera" objections. Runs on `warehouse_aws` with the **good v1
  detector** — off the 4-cam detector-gate critical path.
- **Tier 2 — camera-network generalization (supporting; C2 coverage geometry + C3 fusion
  containment).** The realistic network: coverage-aware planning + health-aware fusion/selection +
  the Δ_fault containment result. **This tier is the existing fault-containment pre-registration**
  ([`PREREGISTRATION_fault_containment_2026-07-22`](../../../experiments/multicamera_fusion_extension/PREREGISTRATION_fault_containment_2026-07-22.md),
  E5/E6/E7, B6, Δ_fault, stop rules — code-complete). **Cuttable to future work with zero damage to
  the headline** if the detector does not gate in time.

Single-cam *legitimizes* multi-cam: proving the mechanism with one camera and no fallback lets the
network containment claim inherit that credibility instead of defending it from scratch.

## 5. Positioning (the gap is between two literatures)

| Field | What it does | What it assumes away |
|---|---|---|
| Multi-camera tracking / MOT / re-ID | network *observes* an agent; DB/human consumes the track | no control loop → coverage gaps and degradation are never *safety* events |
| Warehouse AMR / SLAM | robot localizes with **onboard** sensors | the no-fallback infrastructure-only robot is outside the frame |
| Infrastructure-aided localization (incl. IWAI) | shows it *works* — nominal, well-covered, healthy | the realistic envelope: partial coverage, per-camera degradation, latency, heterogeneity |
| **This work** | **closed-loop navigation on the network as it really is** | — (characterizes the envelope) |

Honest Toro extension table (unchanged): static per-location covariance → continuous spatial field
(ranking win real, AUROC 0.78 vs 0.50); nominal-always-healthy → **outages/latency/drift**; RMSE+
smoothness → **consistency + closed-loop safety**; 2-vs-3-cam grouping → every subset + timed
failures; 3 cams/5.5×3 m/straight lines → 4 cams/full warehouse/closed-loop robot.

## 6. Experiment slate → the two tiers

| Exp | Tier | Role | Cost |
|---|---|---|---|
| **E0** component + Toro/projection validation | both | Gazebo-Toro credibility anchor | done + multi-session recapture |
| E1 spatial reliability prediction | supporting | "location matters" (most exposed to the nulls — never a headline) | offline |
| E2 conditional-covariance calibration | supporting | fusion covariance honesty (coverage+sharpness) | offline |
| **Envelope characterization** (NEW headline) | 1 | safety vs **co-observation coverage × health severity** — the C1 surface | offline + closed-loop |
| E3 nominal multi-cam localization | 2 | the "nominal tie" narrative pivot | offline replay |
| E4 camera count / geometry subsets | 2 | 7–13% overlap reality + naive-fusion-can-hurt (C2 geometry) | offline (cheap) |
| E5 dropout / latency | 2 | fault mode (C3) | offline (cheap) |
| E6 calibration drift → Δ_fault | 2 | multi-cam containment centrepiece (C3) | offline + controlled |
| E7 selection vs fusion | 2 | the containment policy | offline |
| **E8 closed-loop navigation under degradation** | **1 (primary) + 2** | **the PoC payoff — single-cam core is the headline run; extends IWAI C1/C2** | live, limited |
| Stretch (heteroscedastic R, uncertain-input GP, physical perturbation, multi-robot) | — | future work | — |

**Baselines:** headline comparison B1 constant-cov · **B2 Toro** · B6 full health-aware · B9 oracle
(eval-only upper bound); B3/B4/B5/B7/B8 as an ablation row-set. **Confirmatory comparisons
(pre-registered):** (1) health-aware-vs-Toro under fault; (2) coverage-aware-vs-network-unaware
planning; (3) fusion-vs-selection. Everything else exploratory.

## 7. Critical path (next ~3–4 months)

```
TIER 1 (headline, off the detector-gate path):
  wire online health h_t -> planner R_plan (the one new integration) ─┐
  single-cam warehouse_aws EFE nav + mid-mission fault injection ─────┤
        │                                                             │
        ▼                                                             ▼
  E8 single-cam closed-loop: health-aware vs IWAI-fixed-map vs const-cov
  + the safety-vs-(coverage×health) envelope characterization  ← C1 headline
        │
        ▼  (paper stands alone here)

TIER 2 (generalization, on the detector-gate path — cuttable):
  detector to gate ─► multi-session handover captures ─► offline replay
  E3/E4/E5/E6/E7 (pre-reg fault_containment: B6, Δ_fault, stop rules)
```

**Blocker ranking:** (1) tier-2 detector below gate (`v2_640_diag` ~0.40 mid-range) — bounds all
multi-cam numbers; (2) only one commissioning drive → need multi-session for CIs; (3) ~1 Hz live rate
bounds closed-loop (offline replay unaffected). **Tier 1 is not blocked by (1).**

## 8. Paper section outline (ICRA, 6 pp + refs)

1. **Introduction** — infrastructure cameras give useful but partial/heterogeneous/latent/degrading
   localization; the no-fallback robot must navigate safely within that envelope; contribution (§2).
2. **Related work** — infrastructure-aided localization (Toro/IWAI), multi-camera tracking (why
   open-loop differs), detector calibration, robust filtering / fault detection, perception-aware
   planning.
3. **System & problem** — N fixed cameras, no onboard fallback, three reliability quantities kept
   separate, the GT firewall, the realistic-network model.
4. **Method** — coverage/reliability field; calibrated confidence + trust stacker; online health
   monitor (EWMA/bias/debounce); trust→covariance; robust fusion; **planning adapter consuming
   online health** (the tier-1 integration).
5. **Experiments** — E0 + Toro reproduction; **the operating-envelope characterization (headline)**;
   single-cam closed-loop under degradation (E8, tier 1); network generalization E3/E4/E5/E6/E7
   (tier 2).
6. **Results & discussion** — nominal tie → degraded divergence; the envelope; nulls as scoping;
   limitations (detector-limited tier-2, sim-only, narrow overlap, ~1 Hz).
7. **Conclusion / future work** — the stretch items + physical perturbation.

**Working titles:** *Navigating on a Faulty Camera Network: Coverage- and Health-Aware Localization
for Infrastructure-Guided Warehouse Robots* · *When Can a Robot Trust the Cameras? A Safe Operating
Envelope for Infrastructure-Camera Navigation*.

## 9. Risks & mitigations

- **Kitchen-sink dilution (biggest risk).** Coverage+health+heterogeneity+latency+fusion+planning is
  too many headlines. → Lead with **one finding** (the envelope / safe-vs-unsafe under realistic
  conditions); demote the rest to mechanisms.
- **"Just engineering integration."** → Lead with the **characterization** (a surface, a research
  finding), not the system.
- **Detector never gates.** → Tier 1 doesn't need it; tier 2 bounds claims to the operational band +
  reports detector-limitation first-class.
- **Reviewer: "Gazebo-only / injected fault."** → exact Toro reproduction; projection vs
  physically-grounded GT; controlled measured fault infeasible on hardware — stated up front; lead
  with the **healthy-data false-alarm budget** (anti-circularity).
- **Nominal is a tie.** → intended narrative (tie → divergence), pre-registered so it doesn't read as
  a negative.
- **Δ_fault ≈ 0 for both health-aware and naive.** → the pilot already shows naive fusion is
  non-monotone in drift; that gap is the claim; confirm on real capture before writing the number.
