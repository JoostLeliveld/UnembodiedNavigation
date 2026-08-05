# 07 — Multi-camera handover & fusion (paper extension)

[Back to modules index](../README.md)

| | |
|---|---|
| **Claim** | Static per-camera error calibration is insufficient because camera usefulness changes with position, individual detections, availability and calibration health; we combine a spatial GP prior, calibrated frame-level detector evidence and online camera-health monitoring, mapped into camera-specific covariance for fusion and planning. |
| **Status** | Data-independent library implemented + unit-tested. First REAL Gazebo pilot done (single pass, not paper evidence): both camera fixes verified live — camera_A works in batched GPU mode; camera_C's v2 projection calibration cuts its error 0.156→0.077 m vs GT. See [REAL_RUN_FINDINGS](../../experiments/multicamera_fusion_extension/REAL_RUN_FINDINGS_2026-07-21.md). Full multi-camera campaign still pending (needs a full traverse for camera_B + repeated runs). |
| **Chapter** | [08 — large-warehouse scaling](../../research_story/08_large_warehouse_scaling/) (ACTIVE) / [09 — multicamera handover & fusion](../../research_story/09_multicamera_handover_fusion/) (PLUMBING+PILOT) |

This is the home of the **paper extension** over Toro-Diz et al. The full plan,
roadmap, and per-module implementation plans live in the owning study:
[`../../experiments/multicamera_fusion_extension/`](../../experiments/multicamera_fusion_extension/)
([ROADMAP](../../experiments/multicamera_fusion_extension/plans/ROADMAP.md)).

## What it computes (three reliability quantities kept separate)
1. Spatial availability `a_i(s)` — GP classifier.
2. Conditional usability `q_i(s)` — GP classifier.
3. Anisotropic conditional covariance `R_cond_i(s)`.
Plus instantaneous evidence: calibrated confidence, EWMA camera health, stacked
trust — fused via robust/Joseph sequential updates and information-aware
selection. Planning uses only spatially predictable reliability, never the
current frame's confidence copied across the horizon.

## Where it lives (runtime library — all unit-tested)
- [`observation_model.py`](../../src/reliability/reliability/observation_model.py) — the factorized measurement model in one place: `p_use = p_det·p_qual`, the additive innovation decomposition (`H P H' + R_pixel + R_cal + R_time + R_model`), the hit/miss branch posterior (stdlib twin of the CasADi runtime, parity-tested bit-for-bit), the `R/p` baseline, the numerical `R_eff` equivalence, and robust (Student-`t` / contaminated-Gaussian) conditional likelihoods
- [`toro_baseline.py`](../../src/reliability/reliability/toro_baseline.py) · [`conditional_covariance.py`](../../src/reliability/reliability/conditional_covariance.py) · [`confidence_calibration.py`](../../src/reliability/reliability/confidence_calibration.py) · [`trust_stacker.py`](../../src/reliability/reliability/trust_stacker.py) · [`health_ewma.py`](../../src/reliability/reliability/health_ewma.py) · [`fusion.py`](../../src/reliability/reliability/fusion.py) (v2 primitives) · [`planning_covariance.py`](../../src/reliability/reliability/planning_covariance.py)
- Statistics backbone: [`campaign_statistics.py`](../../src/reliability/reliability/campaign_statistics.py)
- Evaluators / replay drivers: [`../../experiments/multicamera_fusion_extension/tools/`](../../experiments/multicamera_fusion_extension/tools/)

## Real-run findings (2026-07-21)
- [REAL_RUN_FINDINGS](../../experiments/multicamera_fusion_extension/REAL_RUN_FINDINGS_2026-07-21.md) — brick-by-brick live Gazebo: sim RTF, detector rate/accuracy tradeoff, and both camera fixes verified vs ground truth.
- [THROUGHPUT_DIAGNOSIS](../../experiments/multicamera_fusion_extension/THROUGHPUT_DIAGNOSIS_2026-07-21.md) — why 3 Hz is inference-bound on the P2000 (corrected by the real runs).

## ICRA-2027 observation-model workstream (started 2026-07-30)

> **Course correction, 2026-08-04.** The original narrowing — *"availability `p_use` and
> conditional accuracy `R_cond` are separate fields, and propagating both through EFE as a
> hit/miss mixture is what changes route choice"* — is **retired as a headline**. `R_cond`
> was never data-blocked (1425/1426 detections associate) and per-camera `R_cond` only ties
> a pooled constant; the dominant term is per-camera **systematic bias**. The
> [2026-07-22 framing-of-record](framings/ICRA_FRAMING_2026-07-22.md) stands, with a new
> claim C5 and a new null — see
> [`framings/ICRA_FRAMING_ADDENDUM_2026-08-04.md`](framings/ICRA_FRAMING_ADDENDUM_2026-08-04.md).
> The mixture result survives as a belief-propagation *correctness* argument, not a
> route-choice demonstration.

Tracks below, each self-contained with its own figures and `RESULTS.md`.

| track | code | evidence + figures | headline |
|---|---|---|---|
| **Residual audit** | [`experiments/external_camera_bias_model/`](../../experiments/external_camera_bias_model/) | [`logs/studies/external_camera_bias_model/exp1_residual_characterization/`](../../logs/studies/external_camera_bias_model/exp1_residual_characterization/) — `fig_b1`–`fig_b6` | Deployed correction kills the along-bearing bias but leaves cross-bearing untouched; camera C keeps **+0.078 m** lateral. Fusion **loses to the best single camera** (beats it in only 12.6% of clusters). |
| **2-DOF bias fix** | [`experiments/external_camera_bias_model/`](../../experiments/external_camera_bias_model/) (exp2) | [`logs/studies/external_camera_bias_model/exp2_two_dof_bias/`](../../logs/studies/external_camera_bias_model/exp2_two_dof_bias/) — `fig_x1`–`fig_x3` | Adding **one** cross-bearing parameter to the frozen deployed correction takes camera C's remaining bias 77→4 mm (−46% RMS, P=1.00) and D's 33→2 mm, and buys 2.4 nats on C's conditional covariance — but it must be gated on `\|b_cross\|/σ_cross ≳ 1.2` or it costs camera A 61%. Candidate calibration emitted, **not deployed**. |
| **EFE mixture** | [`experiments/efe_hit_miss_mixture/`](../../experiments/efe_hit_miss_mixture/) | [`logs/studies/efe_hit_miss_mixture/exp1_mixture_vs_blend/`](../../logs/studies/efe_hit_miss_mixture/exp1_mixture_vs_blend/) — `fig_e1`, `fig_e2` | Precision-blending understates posterior uncertainty by up to **37×**, peaking at `p_use`=0.5. Flag `use_hit_miss_mixture`, **default off**. |
| **Residual logging** | `scripts/visibility_comparison/build_belief_gp_events.py`, `experiments/multicamera_commissioning_bigwarehouse/tools/build_actual_commissioning_inputs.py` | [`logs/studies/residual_logging_schema/exp1_schema_demo/`](../../logs/studies/residual_logging_schema/exp1_schema_demo/) | Signed `eval_res_x/y` now in both event builders, firewalled by `^eval_`. Unblocks fitting `b_c(x)` / `R_cond,c(x)`. |
| **Availability** | [`experiments/fused_observation_model/`](../../experiments/fused_observation_model/) | [`logs/studies/fused_observation_model/exp1_availability_fusion/`](../../logs/studies/fused_observation_model/exp1_availability_fusion/) — `fig_a1`–`fig_a4` | Spatially-blocked held-out redo of the fusion-rule and recalibration comparison. |
| **Planner mapping** | [`experiments/planner_covariance_branching/`](../../experiments/planner_covariance_branching/) | [`logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/`](../../logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/) — `fig_p1`–`fig_p3` | Any single-`R` availability mapping (`R/p` **or** the deployed blend) understates the posterior trace by ~90 % at the runtime operating point, independent of the miss endpoint. The equivalent `R_eff` varies 6.6–10.8× with the prior alone, so `R_plan(s)` cannot be correct in principle. |
| **Projection geometry** | [`experiments/projection_amplification/`](../../experiments/projection_amplification/) | [`logs/studies/projection_amplification/exp1_geometry_vs_detector/`](../../logs/studies/projection_amplification/exp1_geometry_vs_detector/) — `fig_g1`–`fig_g3` | Prices the geometric null model before any `R_cond(x)` fit: a fixed pixel error is amplified several-fold across one camera's footprint, and a one-parameter Jacobian-propagated covariance is compared against constant-`R` on leave-region-out folds. |
| **Deployment realism** | [`experiments/network_commissioning_realism/`](../../experiments/network_commissioning_realism/) | [`logs/studies/network_commissioning_realism/exp1_gate_without_truth/`](../../logs/studies/network_commissioning_realism/exp1_gate_without_truth/) — `fig_n1`–`fig_n3` | Can the calibration gate run in a real warehouse? **GT-free: yes** (agrees with the oracle on 3/4 cameras; the textbook state correction is degenerate — do not use it). **At scale: only for outliers.** A large resolvable bias is decided from **20 detections**; every marginal camera stays under 60 % correct even with all data, and the small-sample failure mode is *false calibrate* — the harmful direction. |
| **Drift lifecycle** | [`experiments/calibration_drift_lifecycle/`](../../experiments/calibration_drift_lifecycle/) | [`logs/studies/calibration_drift_lifecycle/exp1_stale_correction/`](../../logs/studies/calibration_drift_lifecycle/exp1_stale_correction/) | Does the commissioning decision expire? **Yes, and fast.** A *stale* correction turns **harmful at 0.25° of yaw** on camera C — the camera it helps most (0.043 → 0.100 m, vs 0.094 m for doing nothing) — so lifecycle risk scales with correction size and lands only on `CALIBRATE` cameras. The commissioning gate **cannot** double as the in-service monitor: it fires at rest on A (10.2) and B (5.0) against a 1.2 threshold, and is *masked* on B at 0.25° (ratio falls to 0.31) when the induced bias cancels the resident one. The **change** form of the same statistic is monotone on all 4 cameras × both fault types and detects at **0.1° / 0.025 m — one rung before harm**. |

**Two premises were checked and did not survive.** (1) The 4-camera detector is *not* built
on a contaminated lineage — `warehouse_yolo_detector_v1` is the clean 2026-06-17 seg-capture
retrain from a generic `yolo11n-seg` base; the real gaps are zero negatives and no
bottom-point localization eval. (2) Camera detection events *are* conditionally independent:
marginal hit correlations of −0.25…+0.17 collapse to 0.015–0.066 once conditioned on
position, so the noisy-OR union field needs no repair. The deployed
`projection_calibration_v2` along-bearing bias model is likewise **pre-existing** work these
studies audit rather than claim.

**Blocker status (updated 2026-08-04): `R_cond` is not blocked on data — it is blocked on
per-camera bias.** The recorded blocker ("no capture carries a signed residual *and* an
operational state stream; the spawn-grid re-capture is the unblocking step") was
mis-attributed: **1425 of 1426 detections associate to odometry within 0.15 s**, worst
per-camera median offset 10 ms
([`operational_residual_rcond/exp1`](../../logs/studies/operational_residual_rcond/exp1_timing_and_coverage/)).
The 0-events failure lives in a gate inside the commissioning event builder, not in the
data or the clocks. **No re-capture is needed.**

What does block it is the uncorrected lateral bias: belief NEES is 8.5–10.8 at detection
instants against 1.39 calibrated, because an update contracts `P` toward a measurement
carrying 3–8 cm of systematic offset
([`exp2`](../../logs/studies/operational_residual_rcond/exp2_operational_rcond/)). Two
further studies reach the same verdict independently — from held-out likelihood
([`projection_amplification`](../../logs/studies/projection_amplification/exp1_geometry_vs_detector/):
bias transfer costs camera A 47 nats and drops 90 % coverage to 13 %) and from the bias fit
itself ([`exp2_two_dof_bias`](../../logs/studies/external_camera_bias_model/exp2_two_dof_bias/):
removing camera C's lateral bias buys 2.4 nats on its conditional covariance). The mixture
still falls back to `R_visible`.

**That step is now done and the gate passes**
([`exp3_two_dof_rcond`](../../logs/studies/operational_residual_rcond/exp3_two_dof_rcond/)):
`reliability.projection` gained a gated cross-bearing degree of freedom, and with it belief
NEES at detection instants falls **8.51 → 1.06** on the capture held out of the calibration
fit (calibrated = 1.39), median point error **0.046 → 0.022 m**, camera C's oracle bias
**0.077 → 0.002 m**. Per-camera `R_cond` went from losing to a pooled constant by 1.2 nats to
winning by 0.04 — a tie, so per-camera `R_cond` is *unblocked*, not yet *justified*.
Calibration `projection_calibration_v3/` is fitted but **no campaign config points at it
yet**; switching invalidates comparisons against v2-locked artifacts.

## baselines/
External methods reimplemented as comparisons for this contribution. The
Toro-Diz et al. baseline (static nearest-point covariance + CV Kalman) is coded
in [`toro_baseline.py`](../../src/reliability/reliability/toro_baseline.py); see
[`baselines/`](baselines/) for the paper notes and the B0–B9 comparison table.

## framings/
Candidate framings of our own paper centered on this contribution live in
[`framings/`](framings/).
