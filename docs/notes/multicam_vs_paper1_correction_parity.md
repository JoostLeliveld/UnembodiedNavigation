# Parity audit: multicam correction path vs the paper-1 pixel path

Date: 2026-07-29. Triggered by a **1.87 m belief jump in one ~0.1 s sample** in the robustness
campaign (`rob_hardA`), which the existing NIS gate did not stop.

Two correction paths exist in `src/planning/planning/nodes/unicycle_planner_node.py`:
- **Paper-1 (single-camera):** `_apply_pixel_correction` (~line 1320), consuming `/state/bev` from
  `pixel_to_bev_state_node`.
- **Multicam (this extension):** `_apply_state_correction`, consuming the fused `/state/bev` from
  `camera_manager_node`. Written for this extension; **NOT the same gate chain.**

The campaign configs are identical on all relevant params (`pixel_max_correction_jump_m: 0.5`,
`pixel_correction_nis_threshold: 9.21`, `pixel_timeout_s: 1.25`,
`skip_stale_pixel_correction: true`) — so this is **not a config difference. The multicam code path
silently ignores some of them.**

## Gate-by-gate comparison

| Protection | Paper-1 | Multicam | Status |
|---|---|---|---|
| Stale-age rejection | yes (`stale_age`) | yes (`_state_msg_is_fresh`) | parity |
| Implausible dt | yes (`dt_implausible`) | yes (re-anchor if dt > 1.5 s) | parity |
| **Jump limiter** (`pixel_max_correction_jump_m` = 0.5 m) | **yes** — rejects if belief update > 0.5 m (`jump_too_large`) | **NO — not implemented.** Code comment says *"No jump limit: the fused correction, already NIS+disagreement gated"*. Param is set in config and never read. | **MISSING** |
| NIS gate (9.21) | yes, rejects outright (`nis_too_large`) | present, BUT on reject it inflates `S += 1.0 m²`, so the NEXT correction sees `S_y ~ 1.0` -> NIS ~3.4 for a 1.87 m innovation -> **accepted with gain ~0.97** (effectively un-gated snap) | **WEAKENED** |
| Physical plausibility of the PREDICTION | implicitly bounded (jump limiter catches the mismatch) | **none.** `_replay_cmd_log_interval` falls back to `planner.predict(..., fallback_cmd=last_cmd, dt=fallback_dt)` when odometry samples are absent for the interval -> integrates the LAST COMMAND over up to 1.5 s => up to ~0.9 m of invented motion | **MISSING** |
| Rejection diagnostics | yes: `_publish_pixel_correction_rejection`, 7 reason codes (`stale_age`, `dt_implausible`, `jump_too_large`, `nis_too_large`, `update_failed`, `missing_snapshot`, `callback`) | none (throttled text warning only) | **MISSING** — why this was invisible until CSV forensics |

## What actually happened in rob_hardA (evidence)

The belief moved **1.87 m in one 0.1 s sample** (physically impossible: <= 0.06 m at v_max 0.6 m/s),
and moved **away from both ground truth and the fresh correction available at that instant** (that
correction was only 0.30 m from GT). Sitting ~1.65 m wrong, subsequent good corrections looked like
outliers; the belief stayed wrong ~2 s while the robot **rotated in place on a phantom pose**
(cmd_v = 0, cmd_w = 1.0) and then clipped a rack.

So the failure is NOT dead-reckoning across the aisle's ~1 m blind gap (that costs centimetres, as
expected) and NOT a bad camera measurement. It is an **unguarded belief discontinuity**.

The jump size also fell in a **dead band**: too large for the (inflated) NIS gate to stop, but below
the `state_reanchor_m = 2.0 m` divergence guard, so the re-anchor never fired either.

## Fixes to apply (restoring paper-1-equivalent protection only; NO method/visibility tuning)

1. **Jump limiter on the update** — reject a correction whose belief update exceeds
   `pixel_max_correction_jump_m`, mirroring paper 1; but still advance the belief stamp and mildly
   inflate covariance so a rejection can never *freeze* the belief (the runaway this path was
   originally written to avoid).
2. **Physical-plausibility guard on the prediction** — cap the predicted step at what the kinematics
   allow (`v_max * dt` + margin), catching invented motion from the `fallback_cmd` extrapolation.
3. **Reason-coded rejection diagnostics** on the multicam path.

## Still-open parity items (flagged, not fixed)

- `latency_compensate_plan_handoff` is `false` in BOTH configs (parity, but both disabled).
- Multicam reports a **fixed** measurement covariance (`manager_fusion_report_std_m`), so the
  published 2-sigma never grows; paper-1 figures showed a growing filter covariance. Affects
  fig27-style uncertainty plots, not the estimate.
- The smoothing that fixed NEES 16.8 -> 2.8 in the Option-A commissioning suite is NOT in this live path.

---

# PART 2 — Wider parity audit (beyond the correction path), 2026-07-29

Full config diff (locked `warehouse_visibility_campaign_honest_v1.yaml` vs
`warehouse_full_4cam_tours.yaml`) + world-profile comparison. Ordered by significance.

## D1. Driveable-lane annotation / `keep_in` / lane-graph seeds — ABSENT (most consequential)

| | paper-1 world (`warehouse_aws`) | 4-cam world (`warehouse_full_4cam`) |
|---|---|---|
| `known_2d_regions` | **19** (10 `traversable` marked driveable lanes + 9 staging) | **5** (only **1** `traversable`, 2 obstacle, 2 staging) |
| `driveable_geometry_json` | **set explicitly** (13 lane/aisle/connector prisms) | **never set in ANY 4cam config**; profile fallback yields no corridor union |
| `nogo_mode` | **`keep_in`** (must stay inside the driveable union) | **`keep_out`** (only penalises being near obstacles) |
| `nogo_weight` | **2000.0** | 1200.0 |
| lane-graph route seeds | available | **0 seeds** (the recurring runtime warning "lane-graph generated 0 seeds") |

The 4-cam world profile states its design intent: *"The floor is intentionally open; green lines mark
obstacle footprints rather than driveable corridors."* So the extension world was **never given the
marked-lane annotation** paper 1 relies on, and therefore paper-1's entire lane machinery is
inactive here:

- **No `keep_in` constraint** → the planner has no notion of "stay in the lane". `keep_out` is
  precisely the mode that produced the **corner-cut artifact** documented earlier (it once
  invalidated a C2>C1 result; `keep_in` was the fix). Every collision seen in this session has the
  corner-cut signature: clipping a rack corner at y=6.9, drifting 0.66 m into rack E3 mid-aisle,
  clipping rack W3 while turning.
- **No lane-graph seeds** → long single-goal global plans fall back to cold/warm start, which is why
  the ~20 m diagonal produced a curly plan that stalled halfway.
- **Weaker obstacle penalty** (1200 vs 2000).

**Consequence for the conclusions drawn this session:** the repeated diagnosis "tight-aisle
*execution* is the bottleneck / local tracker drifts" is **substantially confounded** by this. A
large part of it is the missing lane constraint, not the tracker. The `ff_fb` tracker swap helped
(0.66 m → 0.04 m lateral) but it was compensating for an absent planner constraint.

## D2. Correction-path gaps — see PART 1 (jump limiter dead, NIS weakened by inflate-on-reject,
no prediction plausibility guard, no rejection diagnostics).

## D3. Belief covariance is a fixed constant, not a filter covariance

The multicam path reports `manager_fusion_report_std_m` (2σ ≈ 0.36 m) and never grows it. Paper-1
figures (e.g. `05_representative_run`) show a *growing* 2σ during blind stretches, which is what
makes over-confidence visible. Consequence: fig27's uncertainty band is flat; NEES/consistency
claims cannot be made from this path as-is.

## D4. No smoothing in the live path

The Option-A commissioning result (NEES 16.8 → 2.8 via smoothing) is **offline/replay only**. The
live multicam path explicitly documents "no smoothing, no latency compensation"
(`unicycle_planner_node.py:185`). `latency_compensate_plan_handoff` is `false` in BOTH configs
(parity, but both disabled).

## D5. Trivial

`debug_log_period_s` (0.25) present in paper-1 config, absent in multicam.

## Recommended order of work

1. **Annotate driveable lanes for the 4-cam world** (aisle centrelines x = 0, ±3.57, ±5.67, ±7.77 +
   the cross-aisles) and run `nogo_mode: keep_in`, `nogo_weight: 2000` — this is world/task
   annotation + restoring paper-1 planner config, NOT method tuning, and it should be done **before**
   any further navigation-failure conclusions are trusted.
2. Restore the correction-path guards (PART 1 fixes 1–3).
3. Then re-run the robustness campaign; expect a materially different failure profile.

---

# CORRECTION to D1 (2026-07-29, after user clarification + verification)

**D1 as written above is WRONG and is retracted.** The user clarified the 4-cam world's design:
*everything is driveable EXCEPT the green-marked non-driveable parts.* Verification confirms it:
the planner receives `collision_geometry_json` with **32 prisms — 4 walls + all 28 rack segments**
(`model=warehouse_walls,warehouse_rack_occluders`). So:

- The obstacle map is **complete**, not missing.
- `nogo_mode: keep_out` is the **correct** semantics for an open-floor world with obstacle
  footprints. `keep_in` belongs to paper-1's marked-lane world and does NOT apply here.
- The "0 lane-graph seeds" warning is a **consequence of the open-floor design** (there are no lane
  corridors to build a lane graph from), not a missing config. The optimizer legitimately falls back
  to explicit seeds / warm-cold start.

**What remains genuinely different after the correction:**
- `nogo_weight` **1200** (multicam) vs **2000** (paper 1) — a weaker obstacle penalty, plausible
  contributor to rack clipping in the 1.55 m aisles.
- The correction-path gaps in PART 1 (**these stand and are confirmed**): jump limiter configured but
  never read, NIS gate weakened by inflate-on-reject, no plausibility guard on the prediction, no
  rejection diagnostics.
- D3 (fixed vs growing reported covariance) and D4 (no smoothing / no latency compensation) stand.

---

# CONSOLIDATION STEP 1 — DONE (2026-07-29): gate chain extracted, paper-1 behaviour pinned

The shared chain now lives in `src/planning/planning/core/belief_correction.py` (ROS-free numpy).
`unicycle_planner_node._apply_pixel_correction` was rewired onto it as a **pure move** — no
behaviour change, paper-1 semantics as the reference. Removed from the node:
`_compute_pixel_uv_update`, `_apply_yaw_anchor_after_pixel_update`, `_pixel_correction_reject_code`,
`_pixel_correction_age_is_invalid`, `project_to_psd`, the covariance regularizer body (node
−132 lines).

The seam is `MeasurementSource.linearize()`, the ONLY thing that differs between the stacks:
- `PixelMeasurementSource` — z = (u,v), R = `R_plan` from the visibility GP (used now);
- `FusedMapMeasurementSource` — z = (x,y) m, H = [I2|0], R from the manager (**built and unit-tested,
  not yet wired** — that is step 2).

Gate order is one function, `evaluate_gates`: stale_age → dt_implausible → missing_snapshot →
update_failed → jump_too_large → nis_too_large. Wire codes 1–6 unchanged (campaign CSVs decode them).

**Regression evidence** — `tests/planning/` (29 tests, all green; 649-test suite green apart from 2
pre-existing detector failures unrelated to planning):
- `test_belief_correction.py::test_golden_trace_verdicts_are_unchanged` replays **6365 unique
  corrections** from the locked `honest_campaign_v1` (43 runs, deduped by apply-stamp) through
  `evaluate_gates`: **zero verdict changes**. 5 pre-update rejections skipped — their inputs (age,
  dt) are not in the CSV.
- Coverage gap made explicit by `test_golden_trace_exercises_the_nis_gate`: the locked trace contains
  a real `nis_too_large` rejection but **never trips the 0.5 m jump limiter** (max observed update
  0.205 m), so the jump gate is unit-tested only.
- `compute_update` is proven equal to BOTH legacy formulas — the pixel `Gamma Σ_y⁻¹` form and the
  multicam closed form `K = S[:, :2] S_y⁻¹`, `S − K S[:2, :]`. That is what makes step 2 a rewiring
  rather than a numerics change.
- `test_planner_node_correction_wiring.py` drives the real node method on a stubbed clock/planner
  (no rclpy.init): accept commits + publishes code 0, jump/NIS reject leaves the belief untouched,
  stale rejects before any planner work, throttle publishes nothing.

**Latent issue found, NOT changed** (pre-existing, out of scope for a behaviour-preserving move):
`_stamp_age_s` catches `TypeError` and returns age `0.0`. rclpy raises exactly that on a
clock-type mismatch (`SYSTEM_TIME` vs `ROS_TIME`), so such a mismatch would **silently disable the
staleness gate** rather than fail. Same silent-failure class as the gaps in PART 1.

---

# CONSOLIDATION STEPS 2–3 — DONE (2026-07-29): PART 1 gaps closed, `nogo_weight` restored

`_apply_state_correction` now runs the **same chain** as the pixel path, via
`FusedMapMeasurementSource`. The node keeps only what is genuinely multicam: bootstrap, the
`state_max_predict_dt_s` re-anchor, the `camera_xy_only` heading override, and the posterior
variance floor at `0.5·R`.

| PART 1 gap | status |
|---|---|
| Jump limiter configured but never read | **CLOSED, then REVISED** — a gate now runs, but it is NIS, not a jump limit. See "Jump limiter removed from the metric path" below. |
| NIS gate weakened by `S += 1.0 m²` on reject | **CLOSED** — `state_reject_inflate_m2` (default **0.05**) |
| No plausibility guard on the prediction | **CLOSED** — `max_predict_speed_mps` clamp (see caveat) |
| No reason-coded rejection diagnostics | **CLOSED** — the full 7-code vector, on `/planner/pixel_correction_diagnostics` |
| `nogo_weight` 1200 vs 2000 | **CLOSED** — 30 4-cam configs set to 2000.0 |

**The measured failure, re-derived numerically** (`belief_cov = R = 1e-3`, innovation 1.87 m, inside
the 2.0 m re-anchor dead band):

| | 1st correction | 2nd correction |
|---|---|---|
| old `+1.0 m²` inflate | rejected | NIS **3.49**, gain **0.999** → **accepted, un-gated snap** |
| new `+0.05 m²` inflate | rejected (update 0.935 m > 0.5 m) | NIS **67.2** → **still rejected** |

The trade-off is pinned from **both** sides, because over-tightening would resurrect the belief-freeze
runaway the separate chain was written to avoid. A *modest* persistent disagreement (0.35 m) is still
accepted on the second correction — as a bounded 0.343 m update under the jump limit, not a snap —
because when corrections keep agreeing that the belief is off by a little, the belief is the thing
that is wrong. Both directions are tests:
`test_the_measured_1_87m_dead_band_jump_stays_rejected_after_a_rejection` and
`test_a_modest_persistent_disagreement_is_eventually_accepted`.

**New reason code 7 = `diverged`.** Divergence is not a plain rejection: the *belief* is wrong, so the
caller re-anchors. It is evaluated **before** the jump/NIS gates, since a diverged belief trips both
and rejecting on either would block the recovery. Codes 1–6 are unchanged and a test now asserts they
can never be renumbered, that no live code decodes as `unknown`, and that the logger's header and data
row stay the same width (they are two list literals ~1700 lines apart; a mismatch silently shifts every
later column).

**CSV note:** the `pixel_corr_*` columns are now written by **both** stacks. New column
`pixel_corr_measurement_space` — **0 = pixels, 1 = metres** — says which. `pixel_corr_predict_clipped_m`
reports the prediction clamp. Both appended at diag indices 44/45, so old readers are unaffected.

**Honest caveat on the prediction clamp.** It bounds a step against `v_max·dt + margin`, so it catches
bad stamps, duplicated replay entries, and a stale high-velocity `last_cmd`. It does **not** by itself
solve the "invented motion" concern from PART 1: if the robot genuinely could have driven 0.9 m in
1.5 s, the cap is 0.95 m and permits it. What actually catches that case is the now-live jump limiter
plus NIS. A principled follow-up is to inflate covariance in proportion to the extrapolated distance
whenever `cmd_replay_used_fallback` is set — not done.

Enabled at `v_max` (0.6) in the 4-cam configs, left **0.0 (off)** for paper 1, because
`honest_campaign_v1` is locked and ran without it.

**Config drift guard.** `run_visibility_campaign.py` is a bare `yaml.safe_load` — no `extends` — so all
30+ configs are flat copies, which is exactly how `nogo_weight` drifted. Pending a base+overlay,
`tests/experiments/test_campaign_config_parity.py` asserts the method-critical keys agree across
worlds, with an explicit allowlist (`nogo_mode`, correctly world-dependent) that is itself checked for
staleness. A parameter still has to be declared in **four** places (node, launch file,
`visibility_launch_common`, campaign runner) — that sprawl is a live drift risk worth removing.

**Still open:** D3/D4 (fixed reported covariance, no smoothing), the per-camera sequential source, the
config base+overlay, and the robustness re-run.

---

## Why the per-camera sequential source is the right next step (analysis, 2026-07-29)

Two findings from reading the fusion path, both verified in code:

1. **Per-camera pixel covariance is computed, published, validated as SPD — and discarded.**
   `CameraObservation.conditional_cov_uv` reaches the manager and goes into `_contract_quality`, but
   the covariance actually used is `fixed_measurement_cov_m2` = constant `diag(0.08², 0.08²)`
   (`camera_manager_node.py:280`, `replay.py:71`). In GP mode it is still isotropic — `((var,0),(0,var))`
   with a scalar interpolated by availability (`replay.py:585-604`).
2. **`project_observation_to_world` returns `(x, y)` only** — no Jacobian, no covariance propagation.

So every camera is modelled as an isotropic 8 cm circle at every range. For an oblique camera the
projected pixel noise is a long thin ellipse — small across-bearing, large along-bearing, growing with
distance. That is the same axis the `along_bearing_slope_per_m` calibration already corrects in the
*mean*; the *variance* along it was dropped.

**This plausibly explains two existing results.** The 7–12 m fusion spikes (fig24/fig26): two bearings
should intersect two elongated ellipses into a tight fix, but modelled as circles a far-range
observation is weighted like a near one. And *selection ≈ fusion*: with isotropic equal-weight noise
there is little for fusion to exploit, so of course it does not beat picking one.

There is also a **double filter** — the manager runs its own Kalman update from a median seed with an
**identity prior** (`camera_manager_node.py:338`), no motion model and no knowledge of the robot's
belief, and the planner then treats that output as an independent measurement.

⇒ **The fig24/fig26 "combine rule barely matters" conclusion is provisional on a misspecified noise
model** and should be tagged as such now, not after the re-run.

Costs of moving to per-camera sequential updates, to state rather than discover: fig24/fig26 stop being
answerable as posed (there is no combine rule left to compare); common-mode error (contact height,
detector bbox-bottom bias) gets double-counted by N "independent" updates → overconfidence; out-of-order
arrival becomes the filter's problem; and the planner *objective* stays single-camera
(`base_planner.py:209/941` bake one `camera.H` into the CasADi function) while the filter becomes
4-camera.

**Sequencing:** do the robustness re-run on the guarded fused path FIRST. Otherwise the run conflates
"added the guards" with "changed the estimator" — the same confound this document exists to fix — and
there is no baseline to measure the per-camera source against. Both are `MeasurementSource`
implementations, so the comparison is one config flag.

---

# STEP 2b — DONE (2026-07-29): per-camera sequential updating, as the second ablation arm

`state_correction_mode: fused | per_camera` on the planner. Same gate chain both ways, so it is a
clean A/B.

- **`fused`** (arm A, unchanged default) — `camera_manager` fuses in map space, planner does ONE update.
- **`per_camera`** (arm B) — the per-camera map observations are folded in **sequentially**: each
  predicted to its own stamp, each carrying its own covariance, each gated and reason-coded on its own.

**Split of work.** The seam is a new JSON wire type, `MAP_OBSERVATION_BATCH_SCHEMA` =
`map_observation_batch/1` (`reliability/fusion.py`), published by `camera_manager_node` on
`/reliability/camera_manager/map_observations` (additive; fused/selected outputs untouched). Belief
updating consumes whatever covariance each observation carries. **Improving that covariance — Jacobian
propagation from `conditional_cov_uv`, anisotropic R — is a separate track and touches different files
(`projection.py`, `replay.py`, `camera_manager_node._map_observations`). No overlap: this change never
reads or writes the covariance model, and benefits automatically when it improves.**

**What this removes:** the double filter. `camera_manager` seeds a Kalman update from a median with an
**identity prior** and no motion model, and the planner then treats that output as an independent
measurement. Sequential updating into the one belief that actually has a prior and a motion model is
the textbook form. It also gives **per-camera gating**: camera B rejected while A/C/D are accepted,
each with its own reason code and camera index in the CSV.

## Design change forced by a failing test

The first version reused the single-observation divergence guard per camera, and a test immediately
showed why that is wrong: **one camera reporting 4 m away tripped the re-anchor and the belief snapped
to the worst observation in the batch.** That guard was earned by the *fused* pose — already through
the manager's NIS + disagreement gates — and a single unvetted camera has not earned it.

Divergence is a claim about the **belief**, so it now requires **corroboration**:

- per-observation re-anchor is **off** in `per_camera` mode; a lone far camera is simply gated out;
- a **quorum** may re-anchor before the updates run — at least 2 and at least half the batch beyond
  `state_reanchor_m`, and mutually agreeing to better than `state_reanchor_m` (else it is scattered
  noise, not a lost belief). Re-anchors to their median.

**Second correction from the same review:** inflate-on-reject fires **at most once per batch, and only
if no camera was accepted**. Per-camera inflation would let one persistently bad camera balloon the
covariance at N×rate until it is itself waved through — reintroducing the neutering bug by a different
route. If some camera is anchoring the belief there is no freeze risk, so no inflation is needed.

## Plumbing and CSV

New column `pixel_corr_camera_index` (diag index 46) — which camera produced a row in `per_camera`
mode, NaN otherwise. Read it together with `pixel_corr_measurement_space`. `planning` now
`exec_depend`s on `reliability` for the wire type (`reliability.fusion` is ROS-free; no cycle).

Ablation configs: `_rob_rob_{easy,hardA,hardB}.yaml` (arm A, now explicitly `fused`) and
`_rob_pc_{easy,hardA,hardB}.yaml` (arm B, `per_camera`), otherwise identical.

---

# REVISION: jump limiter removed from the metric path (2026-07-29)

Restoring paper-1's jump limiter on the metric path was **wrong**, and it introduced a deadlock. New
parameter `state_max_correction_jump_m`, **default 0.0 (off)**, separate from
`pixel_max_correction_jump_m` (which stays 0.5 on the locked pixel path).

**Why NIS is the right gate here and a jump limit is not.** They fire in opposite regimes:

| belief | innovation | NIS | gain | update | jump limit |
|---|---|---|---|---|---|
| confident (small S) | large | huge → **rejects** | small | small | does not fire |
| uncertain (large S) | large | moderate → passes | ≈1 | large | **fires** |

So NIS already catches the real outlier — a confident belief contradicted by a measurement — while the
jump limit only ever fires when the filter is uncertain and a large correction is *warranted*. On this
path `H = [I2 | 0]` is linear, so NIS is exactly the right statistic. (The pixel path keeps its limit:
nonlinear observation model, moment-matched update, and it is locked method.)

**And it deadlocks.** Belief genuinely 1.5 m off, every camera agreeing, R = 1e-3:

```
jump limiter ON (0.5 m)                    jump limiter OFF
 1: NIS 1125.0  update 0.750 m  reject      1: NIS 1125.0  update 0.750 m  reject
 2: NIS   43.3  update 1.471 m  reject      2: NIS   43.3  update 1.471 m  reject
 ...                                        ...
 8: NIS    6.4  update 1.496 m  reject      6: NIS    8.9  update 1.494 m  ACCEPT
 never recovers                             recovers in ~1.2 s at 5 Hz
```

Rejecting inflates S, which **raises** the gain, which **raises** the update, so the limit keeps
firing. The belief only escapes once drift pushes the innovation past `state_reanchor_m` — it has to
get *worse* first. That is the same dead band the original audit found, just with the sign flipped from
"wrongly accepted" to "wrongly rejected".

**The 1.87 m regression is still caught**, by NIS: rejected at NIS 1748 on the first correction and 67
on the second. It is accepted only after ~7 consecutive consistent corrections (~1.4 s). That is the
correct distinction — the original bug was a snap in ONE sample with no corroboration, not convergence
under sustained agreement. Both directions are tested
(`test_the_measured_1_87m_dead_band_jump_stays_rejected_after_a_rejection` and
`test_a_genuinely_lost_belief_recovers_monotonically`).

Plumbed through all four declaration sites, so it cannot become another "set in config, silently
ignored" parameter — the exact bug this whole audit started from.

---

# REVIEW of the paper1_historical covariance profile (2026-07-29)

Implementation reviewed and it is sound: `J R Jᵀ` with a symmetrized `R_uv`, central differences with a
one-sided image-edge fallback, SPD floor, cross-covariance carried to the wire. 709 tests green
(2 pre-existing detector failures unrelated). Two things found.

## 1. BUG FIXED (mine, was blocking the rerun)

`camera_manager` publishes map-frame cross-covariance at `cov[1]`/`cov[6]`, but the planner's fused
path read only `cov[0]`/`cov[7]` and returned `np.diag(...)` — **discarding the anisotropy at the last
step**. The four `_mc_2x2_*` arms default to `fused`, so the reproduction would have measured a
*diagonalised* paper-1 profile. Same "published and discarded" pattern as `conditional_cov_uv` itself.

Fixed via one SPD-safe reader, `_xy_covariance_from_pose`, now used by the fused measurement
covariance, the belief bootstrap and the re-anchor. Falls back to the diagonal with a warning if the
symmetrized block is not positive definite. Five tests, including one that asserts a correlated R
steers the update differently from its diagonal — so if the cross term is ever dropped again, it fails.
`state_correction_mode: fused` is now explicit in the four arms rather than defaulted.

## 2. MODELLING GAP — read before launching the 12 runs

Propagated map-frame 1σ from the frozen blend, at real `camera_A` geometry:

| trust p | px 1σ | range 1.7 m | range 4.2 m | range 7.7 m | anisotropy |
|---|---|---|---|---|---|
| 1.0 | 2.50 | 0.022 m | 0.033 m | 0.051 m | 1.07× → 1.70× |
| 0.5 | 3.53 | 0.031 m | 0.047 m | 0.073 m | " |
| 0.2 | 5.55 | 0.049 m | 0.074 m | 0.114 m | " |

Legacy constant profile: 0.080 m isotropic. **Measured fused accuracy vs GT: ~0.19 m.**

So the propagated covariance is **2–9× overconfident**. `conditional_cov_uv` is the *detector pixel
jitter* only; it excludes calibration error, contact-height error, the along-bearing correction's own
residual, detector bbox-bottom systematic bias, and the robot's physical extent — and those dominate
the 0.19 m. The `r_miss` endpoint models *availability*, not these.

Predicted consequence (arithmetic, NOT a measurement): with R ≈ 0.03 m and true error ≈ 0.19 m, the
posterior floor `0.5·R` pins the belief near 0.02 m σ, so a typical 0.19 m innovation scores
NIS ≈ 27 » 9.21. A 5 Hz steady-state sketch gives **~32 % rejection vs ~8 % on the legacy profile**.
The new guards hold it (hold-and-inflate, quorum re-anchor) so it should not run away, but it will
*look* like the profile failed when the real cause is an incomplete error budget.

**Recommended before the full 12 runs:** one smoke run, then read `pixel_corr_reject_reason_code` /
`pixel_corr_nis` from the CSV to measure the actual rejection rate. That is now cheap and visible
precisely because the reason codes exist. If it confirms, the fix is an additive commissioned
unmodelled-error term in the profile — which is what `MissEndpointPolicy.require_reconciled()`'s
"residual-tail measurement on real data" is for.

## 3. Correction to my earlier claim

I previously suggested the elongated projected ellipse plausibly explained the 7–12 m fusion spikes in
fig24/fig26. Measured anisotropy is **1.07× near-field rising to 1.70× at 7.7 m** — real, but modest.
That is not enough to carry the explanation. The overconfidence in *scale* (above) is the larger
effect. The fig24/fig26 "combine rule barely matters" result remains provisional on the noise model,
but the specific anisotropy mechanism I proposed is **weakly supported and should not be quoted**.

---

## Read this before interpreting arm B

Arm B fixes **which filter** the measurements go into. It does **not** fix the **noise model**. Every
camera still carries a constant isotropic `diag(0.08², 0.08²)`. So arm B is expected to help via the
removed double filter and per-camera gating, **not** via better weighting — the weighting is still
wrong, identically in both arms. Do not read an arm-B win as evidence that per-camera covariance
works; that claim needs the anisotropic-R track landed first.

---

**Root architectural cause (user's point):** the multicam correction path
(`_apply_state_correction`) was written as a **separate implementation** rather than reusing/extending
paper-1's `_apply_pixel_correction`. Every confirmed parity gap above is a direct consequence of that
fork. The durable fix is to consolidate the two paths onto one gate chain (paper-1's), with the fused
observation as just another input source, instead of maintaining a second chain.

---

# WHY 2.5 px WORKED IN PAPER 1 BUT NOT IN MULTICAM (measured, 2026-07-29)

Question raised: if paper 1 ran on the same 2.5 px blend, why did it not suffer the overconfidence
predicted for the multicam path? Answer from the locked `honest_campaign_v1` (43 runs, 6370
corrections) -- measurements, not simulation:

| quantity | paper 1 |
|---|---|
| pixel innovation \|z - h(x)\| | median **1.53 px**, p95 4.01 px, max 13.52 px |
| R applied | 2.5 px |
| NIS (gate 9.21) | median **0.29**, p95 1.83, max 10.82 |
| NIS rejections | **1 / 6370 = 0.016 %** |

A well-specified 2-DOF filter sits near NIS 1.4-2. Paper 1's median is 0.29, so R = 2.5 px was if
anything slightly LOOSE. It was not overconfident and there was nothing to reject.

**The mechanism is where the systematic error goes.** In single-camera pixel-space updating the
innovation is `measured_pixel - h(belief)`, both in the same camera's frame through the same
calibration. A calibration error biases `h()` in the same direction, so the belief converges to the
pose that drives the pixel residual to ~0. The error becomes a BIAS IN THE ESTIMATE, never a large
innovation, and NIS cannot see it. The locked data shows exactly this: belief error vs GT p95
**0.127 m** while pixel residuals were **1.5 px** -- internally consistent, quietly ~13 cm biased.

With four cameras in a shared metric frame, camera A reports X + δ_A and camera C reports X + δ_C with
different per-camera calibration / contact-height / bbox-bottom errors. No single pose satisfies both,
so the innovations are irreducibly ~|δ_A - δ_C|. That is the ~0.19 m.

⇒ **The 0.19 m is not "the cameras are worse than paper 1 thought". It is inter-camera disagreement, a
quantity that does not exist in a single-camera system.** Propagating `conditional_cov_uv` is correct
for one camera's jitter and structurally silent about a term that only appears with more than one
camera. This is why the frozen 2.5/40 blend transfers correctly in SHAPE but not in SCALE.

**Consequence for the noise-model track:** `Σ_unmodelled` is not a fudge factor to be tuned -- it is
the between-camera disagreement, and it is directly measurable. Where two cameras see the robot
simultaneously, the spread of their projected positions IS the quantity (`overlap.py`,
`fusion.camera_disagreement_m`). Commission it, add it to `J R_uv Jᵀ`: correct shape from the
Jacobian, correct scale from data, no tuning. It also explains the legacy `fusion_report_std_m: 0.18`
floor -- that empirical number was mostly inter-camera disagreement all along.

**Prediction worth testing rather than assuming:** the per-camera sequential path should need LESS
inflation than the fused path, because a sequential filter can partly absorb per-camera bias into the
belief between updates the way paper 1 did, instead of being handed one pre-averaged disagreement.
