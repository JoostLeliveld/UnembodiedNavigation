# Paper 1 — work-package crosswalk (roadmap ↔ this study)

Maps the external roadmap's Paper-1 work packages (P1-WP0…WP6), gates (G0…G6) and
checkpoints (C1…C4) onto this study's plan files + `experiments/geometric_baseline/`.
Programme context: [`research_story/PROGRAMME_ROADMAP_2026-07-21.md`](../../../research_story/PROGRAMME_ROADMAP_2026-07-21.md).

## The framing overlay (read first)

Paper 1's **headline is not fixed yet.** Two framings are kept alive and resolved
at **checkpoint C1** (the WP3 = ch.03 gate) on route-disjoint held-out data:

- **Framing S — localization *service*:** contribution = service-map (M0/M1/M2) +
  covariance interface + self-monitoring + service-aware navigation. UIGP (U5) is
  reported as one service-model ablation (M3).
- **Framing U — uncertain-input GP:** contribution = the UIGP passive-commissioning
  method (this study's current README); service framing is the umbrella.

C1 rule: *learned (M2) beats geometry-only (M1)?* and *UIGP (U5) beats
point-input/smoothing (U1/U3)?* → both yes = Framing U; UIGP ties at real σ (the
current leading finding) but learned service still beats geometry = Framing S;
neither beats geometry = fall back to the geometry service model and move novelty
to self-monitoring (WP5). **Every plan below is written to survive either outcome.**

## Crosswalk

| Roadmap WP | This study | Gate / Checkpoint | Status |
|---|---|---|---|
| **P1-WP0** baseline freeze + evidence audit | `WP0_baseline_freeze.md` (**new**) | G0 | plan written |
| **P1-WP1** camera measurement validation | `WP1_camera_measurement_validation.md` (**new**) | G1 | plan written |
| **P1-WP2** operational service dataset | `DATA_SOURCE_commissioning_drive.md` + opportunity/label logic in `B_real_reliability_prediction.md` | G2 | frozen |
| **P1-WP3** single-camera service model | `B_real_reliability_prediction.md` (U0–U6 = M0/M1/M2/M3) + `A_controlled_covariance_sweep.md` (α-sweep feeds C1) | **G3 = C1** | frozen |
| **P1-WP4** observation-covariance product | `C_localization_effect.md` (L0–L4) + `reliability.covariance_mapping` | **G4 = C2** | frozen |
| **P1-WP5** self-monitoring localization | `WP5_self_monitoring.md` (**new**) | **G5 = C3** | plan written |
| **P1-WP6** service-aware navigation | `D_closed_loop_navigation.md` (P0–P4) + `experiments/geometric_baseline/` (C0/N1) | **G6 = C4** | frozen |

## Model-grid reconciliation (WP3)

The roadmap's M-grid and this study's U-grid are the same models:

| Roadmap | This study (U-grid) | Fitter mode |
|---|---|---|
| M0 constant | U0 | arithmetic |
| M1 geometry-only (range/FOV/edge/scale/angle) | (new logistic — see `07_weak_priors_and_geometry`) | small model |
| M2 learned spatial | U1/U2 (`naive`), U4 (`uncertainty_weighted`) | `fit_belief_aware_gp.py` |
| M3 uncertain-input GP (ablation) | **U5** (`expected_kernel`), U3 smoothing | `fit_belief_aware_gp.py` |
| — GT ceiling | U6 | eval-only, firewalled |

Full reuse map: `REUSE_MAP.md`. Data source: `DATA_SOURCE_commissioning_drive.md`.
Integrity rule: `NO_SHORTCUTS.md`.

## Not-yet-planned (roadmap items with no plan file)

- WP0/WP1/WP5 plans are new here (above).
- The roadmap's service-class map (A/B/C/D reliability tiers) and false-safe-rate
  primary metric are additions to WP3's metric set — folded into `B`'s gate; not a
  new file.
