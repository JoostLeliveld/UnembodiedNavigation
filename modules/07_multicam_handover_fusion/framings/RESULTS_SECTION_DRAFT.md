# Paper 2 — Results (working draft)

[Back to module 07](../README.md) · framing: **fault tolerance / containment**, not "beats Toro".

> **Honesty status (2026-07-22).** This is a results section *in progress*. Sections
> **R0–R2 are REAL, measured** results (live headless Gazebo on the P2000, no synthetic
> data). Sections **R3–R6 are PRE-REGISTERED and PENDING** — their tables and gates are
> fixed here, but the numbers await a full operational-recorder handover capture
> (`odom + perception + GT`) → `load_commissioning_run` → replay/fusion sweep. The
> centrepiece containment number (Δ_fault, R4) is **not yet measured** and is not
> written as if it were. Provenance for every real number: see
> [`REAL_RUN_FINDINGS_2026-07-21`](../../../experiments/multicamera_fusion_extension/REAL_RUN_FINDINGS_2026-07-21.md),
> the WP5 probe, and the 4-cam detector audit.

The claim under test: *static per-camera calibration is insufficient because camera
usefulness changes with position, availability and calibration health; a spatial
reliability prior + calibrated frame evidence + online health monitoring, mapped to
camera-specific covariance, does better — and above all **contains** a faulty camera.*
Because ≥2-camera overlap is geometrically scarce here (R1), the defensible headline is
**fault containment**, and the load-bearing mechanism is **single-camera innovation
health monitoring** (R2), not cross-camera disagreement.

---

## R0 — Component & calibration validation  [REAL]

**Projection calibration, live vs ground truth, all four cameras** (one full
south→north traverse, 295 GT-joined detections; GT is evaluation-only, timestamped by
`/clock` on receipt):

| camera | n | raw mean err (m) | + v2 along-bearing calib (m) |
|---|---:|---:|---:|
| A | 50 | 0.128 | **0.087** |
| B | 59 | 0.058 | **0.026** |
| C | 91 | 0.184 | **0.086** |
| D | 95 | 0.107 | **0.036** |

The v2 along-bearing calibration more than halves world-projection error on **every**
camera (camera_C, the worst, 0.184→0.086 m; camera_B confirmed 0.058→0.026 m). This
clears the E0 coordinate-frame gate. *Caveat: the live-pipeline wiring of
`manager_projection_calibration` sits in the parallel commissioning workstream's files;
offline replay applies the same calibration by re-projecting from the recorded pixel.*

**Detector (retrained for the 4-cam world, `warehouse_yolo_detector_4cam_v2_640_diag`).**
Fine-tuned from the single-camera `v1` on a merged 4-camera capture (871 train / 186 val,
imgsz 640). Per-viewing-camera detection roughly **doubled** in the operational mid-range
band (≈0.26→0.40 @ 8–12 m, ≈0.21→0.36 @ 12–16 m) but **remains below the acceptance gate**
(≥0.90 @ ≤12 m, ≥0.75 @ 12–16 m), and 0 beyond 16 m (robot subtends too few pixels).
**All downstream results are therefore detector-limited** — the robot is observed only
~40% of the time mid-range. This is reported as a limitation, and it *reinforces* the
containment framing (missing detections are themselves a fault mode).
Figure: `logs/studies/fourcam_detector_audit/reliability_maps_v2diag.png`.

**Throughput.** Batched single-model 4-camera GPU inference at imgsz 640 yields all four
cameras fresh at **0.97 Hz** (age 0.11 s; sim RTF 0.748; GPU 2.56/4 GiB). imgsz 416
doubles the rate but collapses detection to one camera — unusable. **Verdict:** collect at
640/~1 Hz; the ~1 Hz wall rate bounds only *live* closed-loop (R6), since offline
record→replay runs on sim time (5 Hz).

**GT firewall (apparatus).** The real-data bridge `load_commissioning_run.py` is
validated (`tests/reliability/test_load_commissioning_run.py`, 4 tests): ground truth
enters only `EvaluationFrame`s, never an operational frame/observation (odom ≠ truth,
median 5.7 cm apart), operational contracts reject evaluation-only keys structurally, and
the output is consumable by `run_replay`.

---

## R1 — Multi-camera coverage geometry  [REAL]

Cameras are corner-mounted (~6.1 m). The region where **≥2 cameras** simultaneously
cover the robot is only **7–13%** of the drivable area; most of the space is single-camera.
Figures: `.../four_camera_showcase/overlap_handover_corridor.png`,
`dayzero_reliability_atlas.png`, `best_camera_and_reliability.png`.

**Consequence for the paper's claim:** broad redundancy is not available, so a
"fusion beats best-single" headline would be thin. The defensible contribution is
**fault containment** (R4), and fault *detection* must rest primarily on single-camera
innovation consistency (R2) — cross-camera disagreement is available only in the narrow
overlap corridors and, with only pairwise overlap, cannot isolate which camera is at
fault (isolation needs a ≥3-camera co-visible region; reported as a limitation).

---

## R2 — Fault detection: innovation health monitor  [REAL — controlled ablation]

Does the online health monitor (`reliability.health_ewma`) detect a calibration fault
from operational evidence alone (no GT)? Tested by **controlled ablation** on the clean
single-camera commissioning drive (real logged innovations; a persistent innovation bias
`ν' = ν + β·d` models a stale-calibration / camera-shift fault, images unchanged). NIS is
reconstructed from the logged non-circular `pixel_corr_nis` via a constant effective
covariance, validated against the logged NIS (median matched, max 6.0 vs 5.1, per-frame
corr 0.98).

| injected drift | ~yaw | continuous health h | fraction NIS > 9.21 | hard DEGRADED? |
|---|---|---:|---:|---|
| 0 px (healthy) | 0.0° | 0.94 | 0.00 | no |
| 2 px | 0.18° | 0.73 | 0.00 | no |
| 4 px | 0.36° | 0.18 | 0.00 | no |
| 8 px | 0.72° | ~0 | 0.44 | **yes** (slow) |
| 12 px | 1.07° | ~0 | 1.00 | **yes** (1.5 s) |

Figure: `logs/studies/single_camera_uigp_reliability/wp5_self_monitoring/fig_wp5_drift_detection.png`.

**Result.** The continuous health + bias-EWMA flag **sub-degree** drift (health 0.94→0.18
by 0.36°) — an early *inflate* signal — while the debounced hard *reject* engages
conservatively at ~0.7–1°. Crucially, **zero false alarms over 3,700 healthy frames**
(the monitor never leaves HEALTHY on nominal data), satisfying the G5 stop-rule.

**Honest scope.** This is fault *detection*, not *isolation* (single camera); a controlled
software ablation (physical camera-movement evidence would be stronger); and it is the
*detection* half only — whether health-aware fusion then *contains* the fault (R4) needs
real multi-camera data.

---

## R3 — Nominal multi-camera localization  [HARNESS WIRED — awaiting evidence capture]

Pre-registered. Offline replay of identical recorded detections through B0–B8 (best
single / constant-cov / Toro nearest-point / GP / GP+confidence / full / selection /
information-selection). Metrics per method: ATE RMSE, p95, max; RPE(1 s, 5 s); mean NIS;
NEES (eval-only). **Gate:** fusion no worse than the best constituent camera at nominal;
report per-region error.

The E4 harness is **wired and validated end-to-end on a real capture**
([`run_containment_pilot.py`](../../../experiments/multicamera_fusion_extension/tools/run_containment_pilot.py)
→ `run_camera_subset_sweep`; regression test `tests/reliability/test_containment_pilot.py`):
it loads a capture via the validated bridge, replays every camera subset through one
fusion pipeline, and reports `fusion_gain_p95 = best-single − full-set`. The table below
fills from the A/C→B/D handover capture with the operational recorder; a plumbing run on
the sparse v1-detector pilot returned a finite, sane gain (best single camera_B p95
0.130 m → full-set 0.119 m, gain **+0.011 m** — marginal, as expected for detector-limited
sparse data with narrow overlap; not evidence).

| method | ATE RMSE ↓ | ATE p95 ↓ | max err ↓ | mean NIS | — |
|---|---|---|---|---|---|
| B1 constant-cov | _pending_ | | | | |
| B2 Toro nearest-point | _pending_ | | | | |
| B4 GP | _pending_ | | | | |
| B6 full (health-aware) | _pending_ | | | | |

---

## R4 — Fault containment  [HARNESS WIRED — evidence run pending]

Pre-registered centrepiece. Inject one faulty camera (calibration drift, R2's regime, and
dropout/outage) on identical detections; measure whether health-aware fusion *contains* it:

- **Δ_fault = E(system with the bad camera) − E(healthy subset without it)** — target ≈ 0
  (a bad camera must not hurt more than simply dropping it);
- error-severity AUC vs fault magnitude; max error before isolation; recovery time;
- false-isolation rate (must not remove healthy cameras more than it catches faults — the
  critical-failure stop-rule).

The Δ_fault harness is **wired and validated end-to-end** (`run_containment_pilot.py`:
`bias_camera_position` vs `drop_camera_permanent` through one fusion pipeline). A plumbing
run on the sparse v1-detector pilot (**not evidence** — detector-limited, single capture,
point estimates) already reveals the mechanism the evidence run will quantify: under naive
`SEQUENTIAL_FUSION`, Δ_fault is **non-monotone in drift magnitude** — ~0 at small bias
(within NIS tolerance), a **peak at moderate bias** (0.5 m: camera_B +0.33, D +0.22, C +0.14 m
— the drift passes the NIS gate yet pollutes the estimate), and back to **0 at large bias**
(1.0 m trips the gate, so the gross fault is rejected outright). **The gate-evading
moderate-drift regime is exactly what the WP5 bias-EWMA health monitor (R2) targets.**

The containment claim is the mode comparison **naive `SEQUENTIAL_FUSION` (M5) vs health-aware
`HEALTH_AWARE_FUSION` (B6)** — and B6, the pre-registration's top build item ("the full
method does not exist yet"), **now exists as a replay mode** (`reliability.replay`, wiring
`health_ewma` per-camera into the update: innovation/NIS → debounced health → inflate on
SUSPECT, reject on DEGRADED, with a per-camera health-state timeseries). It is
**unit-validated** (`tests/reliability/test_health_aware_fusion.py`): on a controlled
gate-evading persistent bias it drives the bad camera to DEGRADED, never flags the healthy
camera, and lowers final + RMSE error vs naive fusion. So the containment comparison is one
`--fusion-mode` flag once the capture lands. The headline Δ_fault *number* is **deliberately
left blank until the v2 handover capture exists**; the detection mechanism (R2), the B6 full
method, the fusion/health machinery, the bridge, and this harness are all validated.

| drifted camera | bias (m) | p95 with bad (m) | p95 dropped (m) | Δ_fault (m) |
|---|---|---|---|---|
| _(v2 capture)_ | | | | _pending_ |

---

## R5 — Selection vs fusion  [PENDING]

E7 pre-registered: when overlap exists, fuse; when one camera is persistently biased,
select/robust-subset. Conditions and metrics fixed; awaiting data.

## R6 — Closed-loop navigation  [PENDING, reduced scope]

E8 pre-registered: breach-free clean-goal rate, geometry breaches, belief growth, under
reliability-aware vs constant-covariance planning. Bounded by the ~1 Hz live wall rate
(R0), so this is a reduced-scope live confirmation of the offline result, not the primary
evidence.

---

## Limitations (carried into the discussion)

1. **Detector-limited** — below the acceptance gate; the robot is unobserved ~60% of the
   time mid-range and entirely beyond 16 m. Every localization/fusion number inherits this.
2. **Single commissioning drive** so far — spatial cross-validation only; multi-session
   captures (lighting/clutter/direction) are needed for run-level CIs.
3. **Controlled ablations** (R2) pending physical-fault confirmation.
4. **Narrow overlap (7–13%)** — no broad redundancy; isolation needs a ≥3-camera region.
5. **Throughput** (~1 Hz) bounds live closed-loop; offline replay is unaffected.

*Draft generated 2026-07-22. Real numbers trace to REAL_RUN_FINDINGS_2026-07-21, the WP5
probe (`logs/studies/single_camera_uigp_reliability/wp5_self_monitoring/`), and the
`fourcam_detector_audit`. R3–R6 fill in from `load_commissioning_run` → the tools in
`experiments/multicamera_fusion_extension/tools/` once a real handover capture lands.*
