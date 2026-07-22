# Pre-registration — fault-containment experiment family (E5 · E6 · E7)

> **This is TIER 2 (camera-network generalization) of the two-tier realistic-network
> pre-registration** ([`PREREGISTRATION_realistic_network_2026-07-22`](PREREGISTRATION_realistic_network_2026-07-22.md)).
> Tier 1 — the single-camera closed-loop core + the safe-operating-envelope characterization — is the
> **headline** and lives there; this document holds the multi-camera fusion/containment detail
> unchanged. Tier 2 is **on the 4-cam detector-gate critical path and is cuttable to future work**
> without touching the headline. Content below is not restated in Tier 1, only referenced.

Camera-network fault generalization for the ICRA paper (framing-of-record:
[`ICRA_FRAMING_2026-07-22`](../../modules/07_multicam_handover_fusion/framings/ICRA_FRAMING_2026-07-22.md)).
**E6 (calibration drift → containment) is the Tier-2 centrepiece.** E5 (dropout/latency) and
E7 (selection-vs-fusion) share its metrics, harness, and statistical design and are
pre-registered here together so the whole multi-camera fault-tolerance claim is frozen at once.

> **Registration status: DRAFT → freeze on commit.** Freezing = commit this file, then
> fill the integrity block (§10) with the git commit, detector artifact hash (must be a
> **gate-passing** detector — §9), calibration hashes, and frozen `ReplayConfig`. After
> freeze: no metric may be swapped for the primary, no gate threshold moved, no
> comparison promoted from exploratory to confirmatory. Amendments are dated addenda, not
> edits. Any result computed before freeze is pilot/plumbing, never paper evidence
> ([[no synthetic data / real Gazebo only]] hard rule applies).

---

## 1. Hypotheses (directional, pre-declared)

- **H-E6 (containment).** Under single-camera calibration drift, **health-aware robust
  fusion (B6)** keeps `Δ_fault ≤ ε_c` (contained) across the *gate-evading moderate-drift
  band* (§4), while no-health methods (B2 Toro, B4 GP-only, B5 GP+conf, M5 naive fusion)
  show `Δ_fault > 0` there. **The claim is the gap:** `Δ_fault(best no-health) −
  Δ_fault(B6) > 0` with a 95% hierarchical-bootstrap CI excluding 0.
- **H-E6-detect (detection, extends WP5/R2).** The health monitor flags drift (health drops
  below the inflate threshold) at **sub-degree** magnitude and hard-rejects by **~1°**, with
  a nominal false-alarm rate at or below the §5 bound (WP5 achieved 0.000 / 3700 frames).
- **H-E5 (dropout/latency).** Under camera dropout/outage/delay, reliability-/health-aware
  fusion has **smaller max error, shorter outage duration, lower error-severity AUC, and
  faster recovery** than constant-cov (B1) and Toro (B2).
- **H-E7 (select vs fuse).** Covariance-weighted **fusion** dominates when cameras are
  unbiased and mutually consistent; **selection / robust-subset** dominates when one camera
  is persistently biased. Tested, not assumed — either direction is publishable.

## 2. Independent variables

**Faulted camera:** one at a time, each of `{A, B, C, D}`. Primary analysis on (a) a
camera that dominates a **single-camera** region (only innovation-health can catch it) and
(b) a camera with **≥2-camera overlap** (cross-camera disagreement also available). Overlap
is 7–13% of the floor, so isolation is reported only in ≥3-camera co-visible regions;
elsewhere the claim is **detection**, not isolation.

**E6 calibration-drift fault model** (perturb the *estimator's calibration copy*; images
unchanged = "controlled calibration-ablation evidence"), one-factor-at-a-time ladder
(§15-E6): yaw `{0,.25,.5,1,2,5}°`, pitch same, roll `{0,.25,.5,1,2}°`, translation
`{0,1,2,5,10} cm`, principal-point `{0,1,2,4,8} px`, focal/scale `{0,.25,.5,1,2}%`. A small
set of compound faults is **exploratory**. Plus one **sim-physical** variant per camera:
move the camera model in Gazebo by a measured amount and leave runtime calibration stale
(deployment-like evidence, Gazebo-compatible).

**E5 profiles:** permanent outage; sudden outage at `{25,50,75}%` route; intermittent
`p_drop ∈ {0,.1,.25,.5,.75}`; burst `{0.5,1,2,5} s`; delay `{0,50,100,200,500,1000} ms`.

**E7 conditions:** all-accurate · one-noisy-unbiased · one-biased · correlated-viewpoints ·
complementary-viewpoints · intermittent-false-high-confidence.

## 3. Conditions / methods (identical detections, timestamps, calibration, CV model, one Q)

| ID | Method | Replay mode | Status |
|---|---|---|---|
| B1 | Constant covariance | `SINGLE_FIXED_R` (R1) | exists |
| B2 | **Toro nearest-point** (primary rival) | — | **BUILD: no Toro `ReplayMode`** (`toro_baseline.py` not wired) |
| B4 | GP-only | `CURRENT_GP_R` (R4) | exists |
| B5 | GP + calibrated confidence | — | **BUILD: no GP+conf mode** |
| **B6** | **Full health-aware robust fusion** | — | **BUILD — the containment condition does not exist** (see §7) |
| M5 | Naive sequential fusion | `SEQUENTIAL_FUSION` | exists (the "no-health" foil) |
| M6/M7/M8 | Selection / handover-aware | `CONSERVATIVE/HANDOVER/HYSTERETIC_*` | exist (E7 selectors) |
| hard-NIS | Immediate NIS rejection | `..._NIS` variants / config | partial |
| B9 | Oracle bad-camera removal | — | **BUILD** (eval-only, firewall; upper bound, never headline) |

## 4. Primary outcome + exact definitions

- **Localization error `E` = time-aligned 2D ATE p95** (harness `ReplayMetrics.p95_error_m`;
  matches §17 nominal criterion). Secondary: max error, RMSE, median, p99.
- **`Δ_fault(method, c, b) = E(method | c drifted at severity b) − E(method | c dropped from
  the healthy subset)`.** Harness: [`run_containment_pilot.delta_fault_for_camera`](tools/run_containment_pilot.py).
  **Convention (matches code): `Δ_fault ≤ 0` = contained** (keeping-but-down-weighting the
  bad camera is no worse than dropping it); `Δ_fault ≫ 0` = the bad camera pollutes.
- **Error-severity AUC** = `error_severity_auc(severities, p95s)` (in `replay_sweeps`). Lower
  = more robust across the severity axis. Primary degradation metric for E5/E6.
- **Detection delay** = time from fault onset (sudden faults) to health-state `DEGRADED`.
  **BUILD: requires health-state-timeseries instrumentation** (§7).
- **Isolation accuracy** = fraction of episodes flagging the correct camera (≥3-cam regions
  only). **Build-dependent.**
- **False-isolation / false-alarm rate (FAR)** = fraction of *healthy* cameras/episodes
  flagged `DEGRADED`. **Build-dependent.** Feeds the critical-failure stop rule (§6).

**Gate-evading moderate-drift band (crux of E6):** the per-camera severity window where the
fraction of frames with NIS `> 9.21` lies strictly in `(0, 1)` — the drift passes the NIS
gate yet biases the estimate. Determined per camera from the full severity sweep; the
band is the **pre-declared primary analysis window** for H-E6 (the pilot showed the
`Δ_fault` peak there, ~0.5 m in the position-bias proxy).

## 5. Pre-declared decision rules / gates

- **Containment gate (primary, H-E6):** B6 median `Δ_fault ≤ ε_c` across the moderate band
  AND `Δ_fault(best no-health) − Δ_fault(B6) > 0` (95% CI excludes 0). `ε_c` **finalized
  after pilot** from sensor-resolution/warehouse tolerance; provisional `ε_c = 0.02 m`.
  Provisional §17 target: **≥20% reduction in error-severity AUC** vs Toro under fault.
- **Detection gate (H-E6-detect):** smallest reliably-flagged drift ≤ pre-declared (WP5
  basis: inflate by ~0.36°, hard reject ~0.72–1°); **nominal FAR ≤ 0.02**.
- **CRITICAL-FAILURE STOP RULE (non-negotiable, §21):** if `FAR ≥ isolation-TPR` (the monitor
  removes healthy cameras at least as often as it catches faulty ones), the health monitor
  **must not be used downstream** — the paper reports detection-only and drops the
  containment/isolation claim. This is the honest kill-switch, declared before data.
- **Navigation (E8 hand-off):** containment must not increase physics contacts and must keep
  path-length/time overhead below the (separately pre-registered) E8 tolerance.

## 6. Statistical analysis (§16)

- **Unit of analysis = one fault episode** (one held-out run × one faulted camera × one
  severity). **Never frames.**
- **Paired design:** identical recorded detections + identical injected fault replayed
  through every method → paired `Δ`.
- **Hierarchical bootstrap** route→seed→episode for 95% CIs; report mean & median paired
  difference, CI, and proportion-of-episodes-improved. Wilson intervals for binary outcomes
  (isolation success, breach-free completion). Backbone: `reliability.campaign_statistics`.
- **Confirmatory comparisons (only these three, from the ROADMAP):** (1) B6 full vs B2 Toro;
  (2) B6 full vs B4 GP-only; (3) fusion vs selection. Multiple-comparison control across the
  three. **All other comparisons are exploratory** and labelled as such.

## 7. Harness readiness — executable-now vs must-build

**Executable now (real primitives verified in `replay_sweeps.py`):**
- E4 all-subset sweep (`run_camera_subset_sweep`, 15 subsets of A–D) — the R3/coverage setup.
- E5 dropout (`drop_camera_permanent` / `drop_camera_after` / `run_dropout_sweep`) + latency
  (`run_latency_sweep`).
- E6 `Δ_fault` via the **position-bias proxy** through M5/M6/M7/M8 + R0–R4
  (`run_containment_pilot`), and `error_severity_auc`.

**Built since (2026-07-22) — committed:**
- **B2 Toro baseline — DONE** as a standalone constant-velocity Kalman filter
  ([`toro_filter.py`](../../src/reliability/reliability/toro_filter.py), `run_toro_filter`),
  NOT a `ReplayMode`: Toro uses a CV/no-odometry temporal model, so it is a separate
  estimator (nearest-point covariance + validated-FOV gate + B2a-simultaneous / B2b-sequential
  variants), scored through the SAME `compute_replay_metrics` for paired comparability. Tests:
  `tests/reliability/test_toro_filter.py` (9, synthetic plumbing only — not evidence).
- **B6 health-aware fusion — DONE**: `ReplayMode.HEALTH_AWARE_FUSION` (per-camera
  innovation-health → inflate SUSPECT / reject DEGRADED) + per-step health-state logging;
  `run_containment_pilot --fusion-mode M5|B6`. Tests: `test_health_aware_fusion.py`,
  `test_containment_pilot.py`.
- **Detection-metric scoring — DONE**: `evaluate_fault_detection` / `summarize_fault_detection`
  in [`experiment_evaluators.py`](tools/experiment_evaluators.py) compute detection delay,
  escalation delay, isolation accuracy and false-alarm rate from the health-state timeline,
  with Wilson intervals and the §5 **critical-failure stop rule** as a `Verdict`. Firewalled
  study code (it compares against the injected fault). Tests: 10 in `test_experiment_evaluators.py`.
- **Calibration-parameter perturbation — DONE** (the faithful E6 fault model):
  [`calibration_perturbation.py`](../../src/reliability/reliability/calibration_perturbation.py)
  — a self-contained pinhole ground-plane camera + `CalibrationPerturbation`
  (yaw/pitch/roll/translation/principal-point/focal). `perturb_camera_calibration` re-projects
  a camera's observations through the drifted calibration (recovering the pixel from the world
  point), so the world bias is range- and viewing-angle-dependent — strictly stronger than the
  constant position-bias proxy (a 1° yaw barely moves a near point, badly moves a far one). Tests:
  10 in `test_calibration_perturbation.py`. This is now the **primary** E6 fault model; the
  position-bias proxy in `run_containment_pilot` is retained only as a coarse secondary.
- **Pilot integration — DONE**: `run_containment_pilot` now takes `--fault-model
  position|calibration` (calibration builds a `PinholeGroundCamera` per camera from the
  world SDF and injects the faithful yaw drift) and, when run with
  `--fusion-mode HEALTH_AWARE_FUSION`, emits the **detection table** (delay/isolation/FAR +
  §5 stop rule) alongside `Δ_fault` in a single pass — including one nominal (no-fault)
  episode for the false-alarm rate. Tests: 5 fixture-free plumbing tests in
  `test_containment_pilot.py` (incl. B6 catching a gross persistent drift end-to-end).

**Must build before the confirmatory campaign (blocking, ranked):**
1. **B5 GP+confidence** and **B9 oracle-removal** conditions (the remaining §3 methods).
2. *(optional)* mirror the pluggable fault model into `replay_sweeps.run_calibration_drift_sweep`
   so the standalone severity sweeps also use the faithful calibration fault.

Everything else the load-bearing block needs is now code-complete; the binding constraint
is **data** — the detector must reach gate (§9) and a multi-session handover capture must
exist before any confirmatory Δ_fault / detection number is produced.

## 8. Data requirements (§18)

Reuse the frozen `paper_protocol.yaml` route/seed structure: **4 routes × 5 seeds × 3
sessions = 60 runs** (sessions vary lighting/clutter/startup-timing/direction) → 30 train /
10 calibration / **20 held-out (spatially novel)**. Faults injected offline on held-out runs;
one recording expands to subsets × methods × severities without recollection. Minimum for
each confirmatory cell (moderate band, per faulted camera): **≥20 paired episodes**;
finalize N by a post-pilot power check. Inferential target is **effect size + CI**, not a
lone p-value.

## 9. Hard preconditions (must hold before confirmatory capture)

- **Detector at gate.** `≥0.90 @ ≤12 m, ≥0.75 @ 12–16 m`. Current `v2_640_diag` (~0.40
  mid-range) is **below gate** — all current multi-cam data is detector-limited pilot data.
  No confirmatory episode is recorded until the detector passes.
- **E0 gate passed** (projection residual, timestamp skew, coverage) — the freeze record.
- **Positive control:** ≥1 regime with released corrections and sane NIS/NEES in the capture.

## 10. Registration integrity (fill on freeze)

```
freeze_commit:            <git sha>
detector_artifact_hash:   <gate-passing detector>       gate_passed: <yes/no + numbers>
calibration_hashes:       {A:…, B:…, C:…, D:…}
frozen_replay_config:     {mode set, nis_gate: 9.21, Q: …}
epsilon_c (containment):  <m, finalized post-pilot>
moderate_band_per_cam:    <finalized post severity sweep>
ground_truth_access:      evaluation_only   (firewall: tests/reliability/test_leakage_firewall.py)
```

## 11. What would falsify the claim (stated up front)

- B6 does **not** reduce `Δ_fault` vs the best no-health method in the moderate band (CI
  includes 0) → containment claim fails; report the null.
- Naive fusion is **already** contained (`Δ_fault ≈ 0` everywhere) → no fault mode to
  contain; the paper becomes a nominal-localization + detection paper only.
- Health monitor trips the **critical-failure stop rule** (FAR ≥ isolation-TPR) → detection
  reported, containment/isolation dropped.
- Toro is **not** meaningfully worse under fault → the whole premise ("static calibration is
  insufficient") fails; report honestly.
