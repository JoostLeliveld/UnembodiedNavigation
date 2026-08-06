# Results — single-camera health-aware localization (ICRA Tier-1)

*Draft results section. All evidence is real Gazebo (warehouse_aws, single infrastructure
camera, no onboard localization fallback), honest_v1 config, isolated per run. Figures:
`logs/studies/single_camera_uigp_reliability/tier1_consolidated/figures/`.*

## Setup
A differential-drive robot navigates a warehouse under an EFE planner whose belief EKF is
corrected only by a single overhead camera (YOLO detection → oblique-ground projection). We
inject a **calibration fault** the way a real bumped/sagging/drifting mount produces it: the
detected pixel is re-imaged through **perturbed camera extrinsics** (the pre-registered E6
re-projection model), so the world error is pose-dependent and grows with range — not a
synthetic output offset. An online **innovation-health monitor** (integrated-innovation EWMA
+ debounced state machine) consumes the planner's per-correction diagnostics and emits a
continuous health `h` and a HEALTHY/SUSPECT/DEGRADED state.

## R1 — the health loop closes on real perception (Fig. 1)
On a real drive with a mid-route drift (0.5 m aim-slide, onset t+18 s), health holds `h≈0.95`
through an 84-frame healthy baseline with **zero false alarms**, then collapses to 0.01 as the
drift ramps, latching **DEGRADED ~5 s after onset**; raw NIS spikes past the 9.21 gate. The
detection half of the loop is validated end-to-end on real perception.

## R2 — detectability envelope (C1) (Fig. 2)
Sweeping severity × onset (12 real runs) yields a clean detectability surface: **0 false alarms
in every baseline**; a **detection floor ~0.1 m** (below which the drift stays undetectable, `h`
dips but never crosses); a **coverage-dependent boundary at 0.2 m** (detected early-route, missed
late-route); reliable detection ≥0.3 m at **4–12 s latency**. Detectability depends on both fault
severity and where on the route (camera-coverage geometry) it strikes.

## R3 — the gate interaction: moderate drift is the dangerous regime
A fixed per-frame NIS gate (9.21) reliably rejects **gross** faults (each frame's NIS is far above
threshold) but **silently admits moderate/slow drift** whose per-frame NIS is marginal: at 0.5 m
the fixed gate leaves belief error at **~0.68 m in 4/5 seeds** and the robot stuck (0/5 goals). A
*larger* 0.8 m drift is *less* damaging — it trips the gate and is rejected. So the danger regime
is the moderate drift that sneaks under a per-frame test, and it manifests as **loss of service
(stuck), essentially never collision** (0 collisions across the reproducibility runs; the earlier
single-seed "collisions" did not replicate).

## R4 — health-gated rejection recovers accuracy where the fixed gate fails (Fig. 3)
Holding everything constant but the rejection strategy (n=5 seeds): using the **integrated-innovation
health** signal to reject the camera once DEGRADED (drive on odom) cuts belief error at the moderate
drift from the fixed gate's **0.58 m to 0.20 m** and restores availability (**2–3/5 vs 0/5 goals**).
At the gross drift the fixed gate already suffices (0.12 m, 4/5 goals) and health-gating adds nothing.
This is the concrete value of integrated-innovation health: it catches the accumulating bias a
per-frame gate cannot.

## R5 — two honest nulls that sharpen the claim
- **Blunt safe-STOP of the robot on DEGRADED is the wrong response** (16 runs): it over-triggers on
  benign drift, converts completed missions into non-completions, and cannot be cleanly scored. The
  right response is *reject the bad sensor and keep moving*, not stop the robot.
- **Rejecting too early is also wrong** (Fig. 3, B2p): triggering on `h<0.5` before the debounced
  DEGRADED gives no accuracy gain at moderate drift and *hurts* at gross drift (0.27 m, 1/5 goals) by
  making the robot navigate blind on odom too long. **Reject-on-DEGRADED is the right operating point.**

## Claim & scope
*A fixed innovation gate contains gross camera faults but silently admits moderate/slow calibration
drift; an integrated-innovation health monitor detects that regime, and health-gated sensor rejection
recovers localization accuracy (0.20 vs 0.58 m) and availability exactly where the fixed gate fails —
with no benefit, and a cost if over-eager, in the gross-fault regime the gate already handles.*

**Honest limits.** Collisions are low-rate and sit in the detection-latency transient — the benefit
axis is accuracy/availability, not collision prevention; safety in this regime is bounded by
detection latency vs time-to-unsafe (R2). Single route, single fault type (calibration drift),
single camera. Multi-camera generalization (Tier-2) is left to future work, gated on the 4-cam detector.

## Figures
1. `fig1_health_trace.png` — R1, HEALTHY→DEGRADED on real perception.
2. `fig2_detection_envelope.png` — R2/C1 detectability surface.
3. `fig3_regime_belief.png` — R3/R4, regime-specific accuracy recovery (+ R5 early-trigger cost).
