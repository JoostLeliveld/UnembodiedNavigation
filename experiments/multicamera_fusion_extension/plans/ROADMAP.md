# Multicamera reliability-fusion paper — implementation roadmap (2026-07-17)

Paper: *Spatial and Instantaneous Reliability-Aware Multi-Camera Fusion and
Planning for Warehouse Robots* — extension over Toro Diz et al. (static
per-camera calibrated covariance + nearest-point lookup + CV Kalman).

Central claim (defensible form): static per-camera error calibration is
insufficient because camera usefulness changes with robot position, individual
detections, camera availability and calibration health; we combine a continuous
spatial reliability prior (GP), calibrated frame-level detector evidence, and
online camera-health monitoring, mapped into camera-specific observation
covariance for fusion and planning.

## Three reliability quantities (keep separate — §5 of the draft)

1. `a_i(s)` — spatial availability P(detection | state) — GP classifier.
2. `q_i(s)` — conditional usability P(usable | detection, state) — GP classifier.
3. `R_cond_i(s)` — anisotropic conditional measurement covariance (px², image frame).

Plus instantaneous evidence: calibrated confidence `p^C`, health `h`, stacked
trust `τ`. Planning uses only spatially predictable parts (`ã·q̃·h̄`), never the
current confidence copied across the horizon.

## Dependency spine

```
commissioning M1 (detector retrain) → M2 (projection v3)      [other plan, other workstream]
        │
        ▼
02 opportunity dataset + LOO labels ──► 03 availability/quality GPs (ch.04)
        │                                    │
        ├──► 04 confidence calibration ──► 05 trust stacker
        │                                    │
        └──► 06 conditional covariance ──► 07 trust→covariance mapping (ch.05 reconcile)
                                             │
01 Toro baseline (independent, start now)    │
        │                                    ▼
        └────────────► 09 fusion v2 (robust/Joseph/info-selection) ◄── 08 health monitor v2
                                             │
                                             ▼
                          10 planning adapter (R_plan, expected information)
                                             │
                                             ▼
                          11 experiment campaign E0–E8 (B0–B9 baselines)
                                             │
                                             ▼
                          12 evidence bundle, gates, disclosure register
```

Library modules 01, 04, 05, 06, 08, 09, 10 are **data-independent** (pure code
+ unit tests + synthetic fixtures) and can be implemented immediately. Modules
02/03 and every experiment need commissioning data (detector retrain first —
current 4-cam detection rates 0.10–0.70 are OOD; anything fitted now is
throwaway).

## Structural implementation status (2026-07-17)

- **Implemented and unit-tested:** 01 Toro baseline, 04 confidence calibration,
  05 trust stacker, 06 conditional covariance, 08 calibration-health EWMA,
  09 fusion-v2 primitives, and 10 planning-covariance primitives. Their public
  APIs are exported by `reliability/__init__.py`. This is the pure-library
  layer; replay conditions, ROS/planner call sites, and fitted artifacts are
  still downstream work.
- **Implemented structurally, intentionally not fitted:** 02 opportunity/LOO
  data builders now write operational-only rows and reject evaluation-only
  inputs. 03 supplies per-camera availability/quality wrappers around the
  canonical belief-aware GP fitter, with preserved run-level holdouts and
  model cards; fitting still waits for commissioning M1/M2 so discarded OOD
  detector outputs do not become paper artifacts.
- **Blocked on an explicit empirical/configuration decision:** 07 must reconcile
  the 40 px versus 120 px miss-covariance endpoints from data before runtime and
  offline mappings are unified.
- **Not yet executed:** 11 campaign drivers and 12 evidence promotion. No
  multi-camera paper result is claimed by this structural implementation.

## Plan files

| Plan | Module | New code | Serves |
|------|--------|----------|--------|
| 01 | Toro-style baseline (B2a/B2b) | `reliability/toro_baseline.py` | E1–E5 baseline, §14 |
| 02 | Opportunity dataset + LOO labels | tools + `contracts` fields | §6, E1 data |
| 03 | Factorised availability/quality GPs | wrapper scripts | ch.04, RQ1, H1, E1 |
| 04 | Confidence calibration | `reliability/confidence_calibration.py` | §8, H1, E1 |
| 05 | Trust stacker | `reliability/trust_stacker.py` | §8.3, E1 |
| 06 | Conditional covariance | `reliability/conditional_covariance.py` | §9, E2, H2 |
| 07 | Trust→covariance mapping reconcile | adapter + property tests | §10, ch.05 gate |
| 08 | Health monitor v2 (EWMA/cross-cam/debounce) | `reliability/health_ewma.py` | §11, RQ4, H4, E6 |
| 09 | Fusion v2 (Joseph, Student-t, info-selection) | extend `reliability/fusion.py` | §12, RQ2, E3/E7 |
| 10 | Planning adapter (R_plan, expected info) | `reliability/planning_covariance.py` | §13, RQ5, H5, E8 |
| 11 | Experiment campaign E0–E8 + statistics | sweep tools | §14–18 |
| 12 | Evidence bundle + acceptance gates | docs only | §20–21 |

## Baseline table (§14) → existing replay condition IDs

B0 best-single ≈ R2/S-single; B1 constant-R = R0; B2a/B2b Toro = NEW;
B3 confidence-only = R1-style score policy + calibrator; B4 GP-only = R3/R4;
B5 GP+confidence = NEW stack; B6 full = NEW; B7 GP-selection ≈ M8-style
conservative-best; B8 info-selection = NEW; B9 oracle = `reliability.oracle`
(evaluation-only, firewall-enforced).

## Hard constraints carried from repo conventions

- Two-world rule; `honest_campaign_v1` frozen; no gate/weight retuning from 4-cam data.
- GT/oracle = evaluation-only (firewall tests must cover every new export reader).
- Metrics via `scripts/shared/metrics.py` only; run-level (not frame-level) units of analysis, grouped/hierarchical bootstrap (§16).
- Pre-registered primary comparisons: full-vs-Toro, full-vs-GP-only, fusion-vs-selection. Everything else exploratory.
- The 0.45 release threshold and `r_miss_uv` 40 vs 120 px mismatch must be resolved/pre-registered before quoting numbers (plans 07, 11).
- Parallel workstream owns `experiments/multicamera_commissioning_bigwarehouse/{README,TODO,paper_protocol,paper_campaign,...}` and `research_story/{08,09}/evidence.yaml` — new files only until their commit lands.

## Definition of "beats Toro" (pre-registered, §17)

Nominal: lower run-level median p95 ATE AND lower localization NLL AND coverage
closer to nominal AND no meaningful max-error increase AND real-time. Degraded:
lower error-severity AUC, lower max error, faster isolation, lower nav-failure
rate. Navigation: better breach-free completion or fewer geometry breaches at
≤ pre-declared path-length/time overhead. Margins finalized after pilot; a
single secondary metric win is NOT a claim.
