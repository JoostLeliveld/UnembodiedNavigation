# Pre-registration — safe navigation on a realistic infrastructure-camera network (two-tier)

Experiment contract for the ICRA framing-of-record
([`research/papers/correlated_error_icra.md`](../../research/papers/correlated_error_icra.md)).
**Two tiers, one story:**

- **TIER 1 — single-camera core (PRIMARY, headline).** Closed-loop navigation under mid-mission
  camera degradation with **no onboard fallback**, plus the **safe-operating-envelope
  characterization** (C1). Detailed in full below. Off the 4-cam detector-gate critical path.
- **TIER 2 — camera-network generalization (SUPPORTING, cuttable).** Coverage-aware planning + health-
  aware fusion/selection + Δ_fault containment. **Registered separately and unchanged** in
  [`PREREGISTRATION_fault_containment_2026-07-22`](PREREGISTRATION_fault_containment_2026-07-22.md)
  (E5/E6/E7, B6, Δ_fault, stop rules — code-complete). That document is **subsumed here as Tier 2**;
  its hypotheses/metrics/stat design are not restated, only referenced.

> **Registration status: DRAFT → freeze on commit.** Freezing = commit this file + fill §10.
> After freeze: no primary-metric swap, no gate moved, no exploratory→confirmatory promotion;
> amendments are dated addenda. Any result before freeze is pilot/plumbing, never evidence
> ([[no synthetic data / real Gazebo only]]). The headline (Tier 1) uses the **gate-passing v1
> detector on `warehouse_aws`** and is therefore *not* blocked by the 4-cam detector gate.

---

## 1. Hypotheses (Tier 1, directional, pre-declared)

- **H1-nav (closed-loop safety under degradation — the headline).** When the external camera
  degrades mid-mission (drift / partial occlusion / dropout), a planner that consumes **online camera
  health** (N3, health→`R_plan`) achieves **higher breach-free clean-goal completion and lower
  post-fault max belief error** than a planner using a **fixed offline reliability map** (N1 = IWAI)
  and a **constant-covariance** planner (N0), which keep trusting the degraded camera. The claim is
  the **paired gap**: breach-free(N3) − breach-free(best of N0/N1) > 0, Wilson/paired CI excluding 0.
- **H1-envelope (the C1 finding).** Breach-free rate is a **characterizable function of
  (time-unseen / single-coverage exposure) × (fault severity)**; N3 **shifts the safe boundary
  outward** (tolerates longer blind exposure and larger drift before breaching) vs N0/N1. Reported as
  a 2-D surface + the boundary shift.
- **H1-nominal-tie (pre-registered narrative).** With **no fault**, N0/N1/N2/N3 are statistically
  indistinguishable on clean-goal rate and localization error. The story is *tie under nominal →
  divergence under degradation*; declaring it up front prevents it reading as a negative.
- **H1-detect (single-camera detection, extends WP5).** The health monitor flags degradation at
  **sub-degree** drift and hard-rejects by **~1°**, with **nominal FAR ≤ 0.02** (WP5 achieved
  0.000 / 3700 healthy frames). *Mechanism note:* with one camera the reference for the innovation is
  **odometry / the motion model over a short horizon** (bounded odom drift), not a healthy camera —
  this is why single-camera detection is valid and is pre-registered as detection-over-a-bounded-
  horizon, not steady-state.

## 2. Independent variables (Tier 1)

- **Fault type:** (a) calibration drift (perturb the estimator's calibration copy — "controlled
  calibration-ablation"; primary fault model = [`calibration_perturbation.py`](../../src/reliability/reliability/calibration_perturbation.py),
  faithful re-projection so world bias is range/viewing-angle dependent); (b) partial occlusion
  (mask-region / bbox truncation); (c) dropout / stale stream. Plus one **sim-physical** variant:
  move the camera model in Gazebo by a measured amount, leave runtime calibration stale.
- **Onset:** fault triggered at `{25, 50, 75}%` route progress (so the robot must re-plan mid-route,
  not just start degraded).
- **Severity ladder (drift):** yaw/pitch `{0,.25,.5,1,2,5}°`, translation `{0,1,2,5,10} cm`,
  focal/pp per the §15-E6 ladder. Occlusion `{0,25,50,75}%` of the bbox. Dropout `p_drop ∈
  {0,.1,.25,.5,.75}`, burst `{0.5,1,2,5} s`.
- **Coverage/exposure axis (for the envelope):** routes chosen to span a range of **single-camera
  blind-exposure duration** (how long the robot is out of reliable view) — the single-camera analogue
  of co-observation coverage.

## 3. Conditions / methods (Tier 1 — identical detections, world, Q, no-go geometry; only the planner-facing model differs)

| ID | Planner-facing reliability | Online health? | Status |
|---|---|---|---|
| **N0** | constant covariance | no | exists (IWAI C1) |
| **N1** | **fixed offline reliability map** (IWAI) | no | exists (IWAI C2 — the primary rival) |
| **N2** | coverage-aware (reliability field → `R_plan`, plan to stay observable) | no | exists (map→`R_plan` path) |
| **N3** | **coverage + online health → `R_plan`** (the full method) | **yes** | **BUILD: wire `h_t` into live `R_plan`** (§7) |
| (opt) | N3 + explicit safe-stop policy on `DEGRADED`/blind | yes | small add |

The only difference between N1 and N3 is that N3's predicted future covariance is driven by an
**online** health estimate, not a frozen map. This isolates the contribution to *acting on online
sensor health*.

## 4. Primary outcomes + exact definitions (Tier 1)

- **Breach-free clean-goal rate** (primary, binary) — reached goal region with **0 GT geometry
  breaches and 0 physics contacts**. Harness: campaign `run_summary.json` GT-outcome fields (the
  §16 "no raw position columns" path; GT eval-only).
- **Post-fault max belief error** (continuous) — max `‖μ − gt‖` after fault onset (eval-only).
- **Safe-stop-vs-breach** — did the robot detect-and-stop/reroute vs drive into a breach.
- **Min GT clearance**, **path-length / travel-time overhead** (must stay ≤ pre-declared tolerance —
  no buying safety with huge detours), **time-above-uncertainty-threshold**.
- **Detection delay / FAR** — from the health-state timeline (`evaluate_fault_detection` in
  [`experiment_evaluators.py`](tools/experiment_evaluators.py)); FAR from a nominal (no-fault) arm.
- **Envelope surface** — breach-free rate binned over (blind-exposure duration) × (severity), per
  condition; the C1 deliverable is the N3-vs-N1 **safe-boundary shift**.

## 5. Pre-declared decision rules / gates (Tier 1)

- **Headline gate (H1-nav):** breach-free(N3) − breach-free(best of N0/N1) > 0 under fault, Wilson
  paired CI excluding 0, **without** path-length/time overhead exceeding the tolerance and **without**
  increasing physics contacts.
- **Nominal-tie gate (H1-nominal-tie):** N0–N3 CIs overlap under no-fault (else the "tie" premise is
  wrong and must be reported).
- **Detection gate (H1-detect):** smallest reliably-flagged drift ≤ pre-declared (WP5 basis: inflate
  ~0.36°, hard reject ~0.72–1°); **nominal FAR ≤ 0.02**.
- **CRITICAL-FAILURE STOP RULE (non-negotiable):** if **FAR ≥ true-detection rate** (the monitor
  degrades the healthy camera at least as often as it catches the faulty one), the health monitor is
  **not used downstream** — the paper reports detection-only and drops the health-aware-planning
  claim. Honest kill-switch, declared before data.

## 6. Statistical analysis (Tier 1)

- **Unit of analysis = one run** (route × seed × fault-type × onset). **Never frames.**
- **Paired design:** the same route × seed × fault replayed/re-run through every condition → paired
  differences. **Wilson intervals** for breach-free (binary); **hierarchical bootstrap**
  route→seed→run for continuous (belief error, clearance). Backbone:
  `reliability.campaign_statistics`.
- **Confirmatory comparisons (only these, pre-registered):** (1) N3 vs N1 (IWAI) under fault;
  (2) N3 vs N0 under fault; (3) coverage-aware (N2) vs network-unaware (N0) under fault. Multiple-
  comparison control across the three. All else exploratory.

## 7. Harness readiness — executable-now vs must-build (Tier 1)

**Exists (reused):** EFE planner + belief EKF (IWAI stack, `warehouse_aws`); fixed reliability map →
`R_plan` (`covariance_mapping` / `planning_covariance`, reconciled); the `honest_campaign` nav +
scoring harness; `campaign_statistics`; `experiment_evaluators` (detection metrics + stop rule);
`calibration_perturbation` (faithful fault); WP5 health monitor (`health_ewma`, validated on real
single-cam innovations).

**Must build (small, on the tier-1 critical path):**
1. **Live `h_t` → `R_plan` adapter** — the one new integration: run `health_ewma` in the loop off the
   planner's `/pixel_correction_diagnostics` (NIS + innovation) and feed the online health into
   `R_plan` in `unicycle_planner_node` / `efe_agent_node`, replacing the frozen-map value for N3.
   (The library pieces exist; wiring them into the live node is the build.)
2. **Single-camera live fault injector** — inject calibration drift / occlusion / stale-stream into
   the single-cam runtime at a route-progress trigger (ROS node; drift via the perturbed calibration
   copy, occlusion via an image republisher). No offline-only path — the headline is a live run.
3. **Envelope binning** — a thin offline analysis binning breach-free over (blind-exposure × severity)
   from the recorded runs (reuses `campaign_statistics`).

**Tier 2** harness is code-complete (see the fault_containment pre-reg §7); its only gap is data
(detector gate + multi-session 4-cam capture).

## 8. Data requirements

- **Tier 1 (primary):** `warehouse_aws`, gate-passing **v1** detector. Reuse the honest-campaign
  route/seed structure; **≥4 routes × ≥5 seeds × the fault matrix**, spanning blind-exposure durations
  and severities; multi-session (lighting/clutter/direction) for run-level CIs. The existing
  single-cam commissioning drive supplies the fixed map (N1/N2); the nav campaign is new.
- **Tier 2 (generalization):** per the fault_containment pre-reg §8 (4 routes × 5 seeds × 3 sessions =
  60 runs, held-out split), gated on the 4-cam detector reaching gate.

## 9. Hard preconditions

- **Tier 1:** v1 detector validated on `warehouse_aws` (already the accepted-paper detector); the
  live `h_t`→`R_plan` adapter unit-tested against a firewall (no `gt_*` in the operational path);
  positive control (a nominal run with sane NIS/NEES + released corrections).
- **Tier 2:** 4-cam detector at gate (`≥0.90 @ ≤12 m, ≥0.75 @ 12–16 m`; `v2_640_diag` ~0.40 is below
  gate) + E0 passed. **No Tier-2 confirmatory episode until the detector gates.**

## 10. Registration integrity (fill on freeze)

```
freeze_commit:            <git sha>
tier1_detector_hash:      <v1 warehouse_aws detector>          gate: accepted-paper detector
tier2_detector_hash:      <gate-passing 4-cam detector>        gate_passed: <yes/no + numbers>
frozen_planner_config:    {efe weights, nogo weight, Q, nis_gate 9.21, R_plan map}
health_config:            {eta/m0/rho/debounce — frozen from WP5}
overhead_tolerance:       <max path-length / travel-time increase for N3>
envelope_bins:            <blind-exposure edges × severity edges, pre-declared>
ground_truth_access:      evaluation_only   (firewall: tests/reliability/test_leakage_firewall.py)
```

## 11. What would falsify the claim (stated up front)

- N3 does **not** beat N1/N0 on breach-free under fault (CI includes 0) → the health-aware-planning
  claim fails; report the null (the paper falls back to the envelope characterization + detection).
- N1 (fixed map) is **already** robust to the fault (breach-free ≈ N3) → online health adds nothing
  in closed loop; report honestly.
- Health monitor trips the **critical-failure stop rule** (FAR ≥ detection rate) → detection dropped;
  paper is coverage-aware-planning + envelope only.
- The **nominal tie fails** (a condition wins/loses without a fault) → the narrative premise is wrong;
  re-examine before any fault claim.
- Overhead blows the tolerance (N3 only "wins" by huge detours / near-stopping everywhere) → not a
  usable policy; report the safety/efficiency trade honestly.
