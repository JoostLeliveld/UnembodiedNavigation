# Plan 07 — Trust→covariance mapping, single reconciled implementation (§10, ch.05)

## Purpose
One frozen, bounded, monotone interface from trust to update covariance. This
plan closes the KNOWN ch.05 blocker: three divergent implementations exist
(planner runtime `r_miss_uv=120px`, `single_camera_adapter`, geometry_visibility
offline `40px`). No new numbers get quoted until reconciled.

## Mapping (bounded interpolation preferred over ratio inflation)
```
R_update_i(t) = R_good_i + (1 − τ_i,t)^γ · (R_bad_i − R_good_i)
```
- `R_good_i` = plan-06 conditional covariance (estimated, not asserted).
- `R_bad_i` = miss-regime endpoint — must be estimated/justified (e.g. from
  near-miss residual tails), NOT tuned to make navigation win.
- γ pre-registered. The α(τ,h)-ratio variant of §10 is a documented ablation.

## Required properties (validation-gate enforced, property tests)
monotone (lower τ never shrinks R), PSD, fixed 2×2, correct units, finite
endpoints, logged decomposition per factor (spatial / confidence / health
contributions in the `UpdateCovariance.reason`-style fields).

## Work items
1. Add `gate_decision: accept|inflate|reject` + `reason` + factor decomposition
   to the trust message path (`contracts.UpdateCovariance` — additive fields).
2. Make `single_camera_adapter` the single source of truth for the mapping;
   `geometry_visibility.trust_to_r_plan` delegates to it; planner runtime reads
   the same constants from one config (`world_profiles.yaml` dangling default
   also gets fixed here).
3. Resolve `r_miss_uv` 40 vs 120 px: decide from data (miss-regime residual
   tails at 4-cam ranges), pre-register, and grep-kill the loser.
4. Property tests + golden-curve figure (Fig 05B bounded monotone curve) →
   FREEZE per ch.05 gate.

## Gates
- ch.05 registry gate flips to frozen: one implementation, Fig 05B/05C, both
  runtime and offline import the same function (test imports both paths and
  asserts identical outputs on a grid).
- No downstream module (09/10) may take a mapping constant from anywhere else.
