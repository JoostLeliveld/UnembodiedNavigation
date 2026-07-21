# Study 1 plans — index & sequencing

Single-camera uncertain-input GP reliability study (`warehouse_aws`, good
detector). Read `NO_SHORTCUTS.md` first, then `DATA_SOURCE_commissioning_drive.md`
(reliability data comes from a dedicated coverage drive, NOT the navigation
campaign), then `REUSE_MAP.md`. Experiments run in order; each freezes before the next.

**This study IS the roadmap's Paper 1.** For the work-package mapping (P1-WP0…WP6,
gates G0…G6, checkpoints C1…C4), the service-vs-UIGP framing overlay, and the
model-grid reconciliation, read `PAPER1_WP_CROSSWALK.md`. Three front/health work
packages are new plan files here — `WP0_baseline_freeze.md`,
`WP1_camera_measurement_validation.md`, `WP5_self_monitoring.md` — and precede A/B/C/D
in the pipeline. Programme context:
`../../../research_story/PROGRAMME_ROADMAP_2026-07-21.md`.

## The baseline grid (from `research_story/03_uncertain_input_gp`)

| id | model | source |
|---|---|---|
| U0 | global constant reliability | trivial |
| U1 | point-input GP | `fit_belief_aware_gp.py mode=naive` |
| U2 | point-input GP, larger length-scale | U1 config |
| U3 | Gaussian spatial smoothing | small add / `belief_spread` |
| U4 | covariance-weighted point-input GP | `mode=uncertainty_weighted` |
| **U5** | **uncertain-input expected-kernel GP** | `mode=expected_kernel` — champion, implemented |
| U6 | GT-position GP | **evaluation-only** ceiling |

**Central claim (ch.03):** on route-disjoint splits, U5 beats ALL of U0–U4 on
held-out NLL/Brier, with U6 as the ceiling. A null (U5 ≈ U1/U3) is an acceptable,
publishable finding: *"at this world's pose uncertainty the expected-kernel
treatment is unnecessary."*

## Experiment sequence

| plan | experiment | world/data | serves | gate |
|---|---|---|---|---|
| `A_controlled_covariance_sweep.md` | A — controlled α covariance-fidelity sweep | real `warehouse_aws` trajectory + injected `Σ=α·Σ₀`, α∈{0,0.5,1,2,4,8} | ch.03 | U5 degrades most gracefully; report the α where U5 separates from U1/U3 (**controlled ablation, not operational evidence**) |
| `B_real_reliability_prediction.md` | B — real logged reliability prediction | NEW `warehouse_aws` **commissioning coverage drive** | ch.03 (+ ch.04) | U5 beats U0–U4 on region-disjoint NLL/Brier; σ high in unexplored regions; factorised a(s)/q(s) beats scalar |
| `C_localization_effect.md` | C — localization effect (L0–L4) | replay of B's detections | ch.05 | reliability-aware R_update lowers NIS inconsistency / ATE tail, Q frozen once |
| `D_closed_loop_navigation.md` | D — closed-loop navigation (P0–P4) | NEW `warehouse_aws` nav runs | ch.06 | U5-based R_plan improves clean-goal/GT-breach at bounded path/time overhead; visibility reward stays 0 |

## Cross-study

Study 2 (multi-camera) = `experiments/multicamera_fusion_extension/`, gated on
this study freezing. Do not start Study-2 fitting until B/C freeze here.
