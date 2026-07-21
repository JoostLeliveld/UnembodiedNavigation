# Paper 2 — work-package crosswalk (roadmap ↔ this study)

Maps the external roadmap's Paper-2 work packages (P2-WP0…WP8), gates and
checkpoints (C5/C6/C7) onto this study's existing plans 00–12 and the
`reliability` library modules. The dependency spine, baseline table and structural
implementation status already live in [`ROADMAP.md`](ROADMAP.md) — this file adds
only the roadmap→repo mapping and the fault-tolerance reframe. Programme context:
[`research_story/PROGRAMME_ROADMAP_2026-07-21.md`](../../../research_story/PROGRAMME_ROADMAP_2026-07-21.md).

## Reframe (important)
The roadmap recenters Paper 2 on **fault tolerance / fault containment**, not
"reliability-aware fusion beats Toro." Headline hypotheses become:

- **P2-H1 redundancy value** — N cameras improve service coverage, availability,
  tail error vs best single camera.
- **P2-H3 fault containment** — one bad camera does not drag the system below the
  healthy-camera subset. Key metric **Δ_fault = E(system-with-bad-camera) −
  E(healthy-subset)**; target small.
- **P2-H4 selection vs fusion**; **P2-H5 fault-tolerant navigation**.

Toro reproduction and the pre-registered "beats-Toro" criteria (ROADMAP §17) stay
as the *nominal* comparison — they are no longer the sole headline.

## Crosswalk

| Roadmap WP | Repo home | Gate / Checkpoint | Status |
|---|---|---|---|
| **P2-WP0** requirements + camera-agnostic interfaces | `reliability.contracts`, `reliability.camera_manager` | G0 | interfaces implemented + tested |
| **P2-WP1** multi-camera commissioning (intrinsics/extrinsics/sync/overlap/static-calib) | plan 02 + `experiments/multicamera_commissioning_bigwarehouse/` | G1 | **blocked on detector retrain M1/M2** (4-cam det rate 0.10–0.70 = OOD) |
| **P2-WP2** per-camera service models | plan 03 (`train_availability_gp` / `train_quality_gp`) | G2 | structural; fit blocked on WP1 |
| **P2-WP3** reproduce fusion + selection baselines (Toro B2a/B2b; A0–A5; B0–B4) | plan 01 `reliability.toro_baseline`; `reliability.fusion` selectors | G3 | implemented + tested |
| **P2-WP4** proposed selection/fusion manager | plan 09 `reliability.fusion` v2 (Joseph, Student-t, `expected_information_gain`, `select_information_best`, `fuse_or_select`) | **G4 = C5** | implemented + tested |
| **P2-WP5** health + fault isolation | plan 08 `reliability.health_ewma` (`InnovationHealthMonitor`, `HealthDebouncer`, `isolate_suspect_camera`) | **G5 = C6** | implemented + tested |
| **P2-WP6** camera subset + redundancy study | plan 11 campaign E4; `reliability.overlap` | G6 | replay drivers pending |
| **P2-WP7** selection vs fusion study | plan 11 campaign E7 | **G7 = C7** | drivers pending |
| **P2-WP8** multi-camera service-aware planning | plan 10 `reliability.planning_covariance` (expected-info + sequential forms) | G8 | implemented + tested |

## Camera-count vocabulary (roadmap E4/E6 — do not collapse)
Report four distinct counts, never one "number of cameras":
`N_installed`, `N_supporting(s)` (geometric coverage at state s), `N_healthy(t)`,
`N_observing(t)`.

## Checkpoints
- **C5** (after WP4): learned fusion beats static calibration? If no → focus Paper 2
  on fault containment, not nominal accuracy.
- **C6** (after WP5): bad camera isolated reliably? If no → claim fault *detection*
  only, not identity isolation (2-camera ambiguity is a documented caveat).
- **C7** (after WP7): hybrid select/fuse policy justified? If no → deploy the simpler
  winning policy.

## Hard blocker
Every Paper-2 *result* (not the library code) waits on real multi-camera data,
which needs the 4-cam detector retrain (M1) + projection recalibration (M2). No
multi-camera paper claim is made by the current structural implementation.
