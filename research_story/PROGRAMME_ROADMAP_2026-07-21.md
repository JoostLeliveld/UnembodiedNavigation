# Programme roadmap — two papers, mapped onto this repo (2026-07-21)

This is the reconciliation of the external *"Detailed Research Roadmap Through
Paper 2"* onto the existing repo layout. The external roadmap proposed a fresh
`programme/paper1|paper2|shared/` tree; **we do not create that tree** — CLAUDE.md
forbids new top-level directories, and the work already lives in the five
organizing layers. This file is the front door that maps the roadmap's
programme / paper / work-package / gate / checkpoint vocabulary onto what is
already here. It points at plans and code; it never copies them.

Supersedes nothing. Layers on top of: [`README.md`](README.md) (the claim-layer
index and contribution structure), [`registry.yaml`](registry.yaml),
[`THESIS_PLAN_2026-07-15.md`](THESIS_PLAN_2026-07-15.md).

## Programme boundary

| Paper | Working title | Scope |
|---|---|---|
| **Paper 1** | A Self-Monitoring External-Camera Localization Service for Warehouse Robot Navigation | ONE fixed external camera: validated measurements → operational service map → calibrated covariance interface → self-monitoring/health → service-aware navigation. World `warehouse_aws`. |
| **Paper 2** | Health-Aware Multi-Camera Localization for Fault-Tolerant Warehouse Robot Navigation | N fixed cameras: per-camera service models, selection/fusion, fault isolation, safe degradation, fault-tolerant navigation. World `warehouse_full_4cam` (frozen methods only). |
| **Paper 3** | camera placement / fleet routing / infrastructure optimization | **PARKED.** Nothing in Paper 1 or 2 depends on it. |

Sequencing rule (non-negotiable, from the roadmap and from
`research_story/README.md`): **Paper 1 must prove the single-camera localization
service is meaningful before Paper 2 combines several such services.** Study 2
fitting does not start until Paper 1's WP3/WP4 freeze.

## The Paper-1 headline is decided at checkpoint C1 — not now

The external roadmap *demotes* uncertain-input GP (UIGP) to "an optional
ablation" and makes "self-monitoring localization service" the Paper 1
contribution. The existing `research_story/README.md` makes UIGP the
**Contribution-1 core** (ch.03). Per the 2026-07-21 scoping decision we **keep
both framings alive** and let the WP3 pilot pick:

- **Framing S (service):** contribution = the localization *service* (service-map
  prediction M0/M1/M2 + covariance interface + self-monitoring + service-aware
  navigation). UIGP is one candidate service model (M3), reported as an ablation.
- **Framing U (uncertain-input GP):** contribution = the UIGP passive-commissioning
  method (the O'Callaghan mapping onto camera reliability); health + service
  framing are supporting.

**Checkpoint C1** (= ch.03 gate, after WP3) resolves it on route-disjoint held-out
data: *does the learned model (M2) beat geometry-only (M1), and does the
uncertain-input GP (U5) beat point-input/smoothing (U1/U3)?* If UIGP wins →
Framing U. If UIGP ties at real σ (already the leading finding — the honest null)
but the learned service still beats geometry → Framing S. The plans are written
so **either outcome is publishable**; neither headline is asserted before C1.

## Progression spine

```
validated camera measurements  (WP1 / ch.02)
  ⇓
operational localization-service model  (WP2 dataset / WP3 model / ch.01,03,04)
  ⇓
calibrated uncertainty interface  (WP4 / ch.05)
  ⇓
self-monitoring single-camera localization  (WP5 — new)
  ⇓
service-aware navigation  (WP6 / ch.06)      ── PAPER 1 freezes here ──
  ⇓
multi-camera commissioning  (P2-WP1 / ch.08)
  ⇓
selection and fusion  (P2-WP3/WP4 / ch.09)
  ⇓
fault isolation and safe degradation  (P2-WP5 / ch.09)
  ⇓
fault-tolerant multi-camera navigation  (P2-WP8 / ch.11)
```

## Paper ↔ chapter ↔ study crosswalk

| Paper | research_story chapters | investigation study | contribution mapping |
|---|---|---|---|
| Paper 1 | 00 (locked anchor), 01, 02, 03, 04, 05, 06, (07 supporting) | `experiments/single_camera_uigp_reliability/` + `experiments/geometric_baseline/` | Contribution 1 (+ option-A ch.04) |
| Paper 2 | 07, 08, 09, 10 (future), 11 | `experiments/multicamera_fusion_extension/` + `experiments/multicamera_commissioning_bigwarehouse/` | Contribution 2 option B (08+09) |

## Paper-1 work packages → repo (roadmap Part II)

Plans live in [`experiments/single_camera_uigp_reliability/plans/`](../experiments/single_camera_uigp_reliability/plans/).
Crosswalk detail: `plans/PAPER1_WP_CROSSWALK.md`.

| WP | Roadmap content | Repo home | Gate | Status |
|---|---|---|---|---|
| **P1-WP0** | Baseline freeze + evidence audit (reproduce C1/C2, provenance, historical separation) | `plans/WP0_baseline_freeze.md` (**new**); anchor = ch.00 `honest_campaign_v1` (LOCKED) | G0 | plan written; runs pending (human-driven sim) |
| **P1-WP1** | Camera measurement pipeline validation (association, pixel source, projection, timestamps, error map, failure gallery) | `plans/WP1_camera_measurement_validation.md` (**new**); serves ch.02 | G1 | plan written; run pending |
| **P1-WP2** | Operational service dataset (opportunity logic, label hierarchy, grouped splits, coverage audit) | `plans/DATA_SOURCE_commissioning_drive.md` (= WP2) + opportunity/LOO builders in Study-2 tools | G2 | plan frozen; capture pending |
| **P1-WP3** | Single-camera service model (M0 const / M1 geometry / M2 learned / M3 UIGP-ablation) | `plans/B_real_reliability_prediction.md` (= WP3, U0–U6 grid) + `plans/A_controlled_covariance_sweep.md` (α-sweep feeds C1) | G3 = **C1** | plan frozen; fit pending on WP2 data |
| **P1-WP4** | Observation-covariance product (R_cond, τ, R_update, R_plan; calibration/coverage/sharpness) | `plans/C_localization_effect.md` (= WP4, L0–L4) + `reliability.covariance_mapping` (frozen interface) | G4 = **C2** | plan frozen; replay pending |
| **P1-WP5** | Self-monitoring localization (stream/detection/consistency health, EWMA bias, state machine, fault injections) | `plans/WP5_self_monitoring.md` (**new**); reuses `reliability.health_ewma` | G5 = **C3** | plan written; runs pending |
| **P1-WP6** | Service-aware navigation (N0–N4, conflict scenarios) | `plans/D_closed_loop_navigation.md` (= WP6, P0–P4) + `experiments/geometric_baseline/` (C0/N1) | G6 = **C4** | plan frozen; campaign pending |

## Paper-2 work packages → repo (roadmap Part III)

Plans live in [`experiments/multicamera_fusion_extension/plans/`](../experiments/multicamera_fusion_extension/plans/).
Crosswalk detail: `plans/PAPER2_WP_CROSSWALK.md`. Most Paper-2 *library code* is
already implemented and unit-tested (see that study's `ROADMAP.md` §"Structural
implementation status"); what remains is real multi-camera data + replay drivers.

| WP | Roadmap content | Repo home | Status |
|---|---|---|---|
| **P2-WP0** | Multi-camera requirements + camera-agnostic interfaces | `reliability.contracts` (Observation/Quality/Prediction/UpdateCov/PlanningCov) + `reliability.camera_manager` | interfaces implemented |
| **P2-WP1** | Multi-camera commissioning (intrinsics/extrinsics/sync/overlap/static-calib set) | plan 02 + `experiments/multicamera_commissioning_bigwarehouse/` (parallel workstream) | detector retrain M1/M2 blocking |
| **P2-WP2** | Per-camera service models | plan 03 (factorised availability/quality GP wrappers) | structural; fit blocked on WP1 |
| **P2-WP3** | Reproduce fusion + selection baselines (Toro B2a/B2b, A0–A5, B0–B4) | plan 01 `reliability.toro_baseline` + `reliability.fusion` selectors | implemented + tested |
| **P2-WP4** | Proposed selection/fusion manager (Joseph, Student-t, info-selection, fuse-or-select) | plan 09 `reliability.fusion` v2 primitives | implemented + tested → **C5** |
| **P2-WP5** | Multi-camera health + fault isolation (EWMA, cross-cam disagreement, debounce, odd-one-out) | plan 08 `reliability.health_ewma` | implemented + tested → **C6** |
| **P2-WP6** | Camera subset + redundancy study (all 7 subsets, N_installed/supporting/healthy/observing) | plan 11 campaign E4 + `reliability.overlap` | drivers pending |
| **P2-WP7** | Selection vs fusion study | plan 11 campaign E7 | drivers pending → **C7** |
| **P2-WP8** | Multi-camera service-aware planning (expected information, health-aware horizon) | plan 10 `reliability.planning_covariance` | implemented + tested |

**Reframe note:** the external roadmap recenters Paper 2 on *fault tolerance /
fault containment* (hypothesis P2-H3 and the Δ_fault = E(system-with-bad-camera) −
E(healthy-subset) metric), where the earlier `multicamera_fusion_extension`
framing led with "reliability-aware fusion beats Toro." Both are compatible: Toro
reproduction (WP3) and beats-Toro criteria stay as the nominal comparison; the
headline claim shifts to redundancy value + fault containment. See the Paper-2
crosswalk.

## Gate + checkpoint index

Gates are module acceptance gates (defined per WP; see
`_shared/module_gates_and_stop_rules.md`). Checkpoints are go/no-go story
decisions (roadmap §Decision checkpoints).

| Gate | After | Repo gate it equals | | Checkpoint | Decision |
|---|---|---|---|---|---|
| P1-G0 | WP0 | ch.00 reproduces + firewall clean | | — | — |
| P1-G1 | WP1 | ch.02 measurement validated | | — | — |
| P1-G2 | WP2 | ch.01 dataset gate | | — | — |
| P1-G3 | WP3 | ch.03 acceptance | | **C1** | learned beats geometry? UIGP beats point? → Framing S/U |
| P1-G4 | WP4 | ch.05 freeze | | **C2** | service covariance improves consistency? |
| P1-G5 | WP5 | new (health) | | **C3** | faults detected without excess false alarms? |
| P1-G6 | WP6 | ch.06 nav | | **C4** | service-aware planning matters in conflicts? |
| P2-G4 | P2-WP4 | plan 12 fusion gate | | **C5** | learned fusion beats static calibration? |
| P2-G5 | P2-WP5 | plan 12 health gate | | **C6** | bad camera isolated reliably? |
| P2-G7 | P2-WP7 | — | | **C7** | hybrid select/fuse justified? |

Each checkpoint has an explicit fallback in the roadmap (e.g. C1-no → use geometry
model, shift novelty to self-monitoring; C6-no → claim fault *detection* only, not
identity isolation). The fallbacks are recorded in the relevant WP plan.

## Governance layer (roadmap Part I)

Cross-cutting rules, now written as authoritative docs under `_shared/`:

- **Evidence classes** — [`_shared/evidence_classes.md`](_shared/evidence_classes.md):
  the 8 evidence classes (CURRENT / HISTORICAL / REAL EXPERIMENT / EVALUATION ONLY /
  CONTROLLED ABLATION / MODEL ONLY / DIAGNOSTIC / HYPOTHETICAL), reconciled with the
  4 honesty tags in [`_shared/honesty_tags.md`](_shared/honesty_tags.md).
- **Ground-truth firewall** — [`_shared/ground_truth_firewall.md`](_shared/ground_truth_firewall.md):
  the two-channel (operational vs evaluation) spec + the CODE that enforces it
  (`reliability.contracts`, `reliability.firewall`, `tests/reliability/test_leakage_firewall.py`).
- **Module gates + stop rules** — [`_shared/module_gates_and_stop_rules.md`](_shared/module_gates_and_stop_rules.md):
  the 10-point module-acceptance README rule + the 8 stop rules.

## Hard rules carried from repo conventions (do not relax)

- Two-world rule: method development in `warehouse_aws` only; `warehouse_full_4cam`
  evaluates frozen methods. `honest_campaign_v1` is LOCKED.
- No shortcuts: reuse code, run NEW data on GOOD data; synthetic only where
  labelled CONTROLLED ABLATION (WP3 Experiment A). See the study's `NO_SHORTCUTS.md`.
- Metrics via `scripts/shared/metrics.py` only; run-level (not frame-level) unit of
  analysis; grouped/hierarchical bootstrap.
- GT / oracle / CAD geometry are EVALUATION-ONLY; every new export reader gets a
  firewall test.
- `r_miss_uv` 40-vs-120 px and the 0.45 release threshold stay pre-registered /
  blocked (`MissEndpointPolicy.require_reconciled`) until measured on real residuals.
