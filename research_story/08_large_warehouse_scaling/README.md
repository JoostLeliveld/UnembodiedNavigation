# 08 — Scaling: the frozen pipeline meets warehouse_full_4cam

**Question.** Does the single-camera pipeline — target, kernel, calibration, split rules,
R_plan endpoints, ALL frozen in ch.02/03/05 — transfer per-camera to a larger four-camera
facility with **zero retuning**?

**Status: ACTIVE.** This chapter is an *evaluation environment*, not a
contribution. **Canonical world: `warehouse_full_4cam.world.sdf`** — 24.5 × 20.5 m, two rack
blocks around a 4.5 m central aisle, four wall-mounted cameras A–D at (±6, ±10, 6.1) looking
perpendicular to their walls, deliberately heterogeneous occlusion (tall inner racks give
the north/south pairs different blind regions). Layout of record:
`docs/warehouse_full_4cam_layout.md` (+ map PNG/SVG in `docs/assets/`). The top-down camera
at (0,0,26) is media-only, never a localization source.

`warehouse_full_4cam` is the ONLY large world in the repo (consolidation 2026-07-15).
Prior development testbeds and their tooling were deleted and archived. Their data is
historical development material — cite it as method history, never as thesis evidence.

## What "success" looks like (no contribution claimed)

> The frozen single-camera learning pipeline produces four distinct, calibrated,
> camera-specific trust fields τ_A..τ_D(s) at facility scale — without touching a single
> hyperparameter.

If a frozen choice fails at scale, that is a **finding to report**, not a licence to retune
here. Fusion language is forbidden in this chapter (that's ch.09).

## The results we're aiming for

- **Fig 08A** — facility overview: cameras, nominal supports, overlap regions
  (day-zero says 42.2% multi-camera overlap, 99.2% union coverage), gaps, commissioning
  routes. PARTIAL today via the layout map + day-zero maps.
- **Fig 08B** — per-camera data coverage small-multiples (observations + covariance ellipses).
- **Fig 08C (headline)** — four trust fields, one colour scale, zero retuning.
- **Fig 08D** — heterogeneity table: per-camera detection rate, calibration, conditional
  error, map NLL, range/edge distributions. Aim: visibly different cameras, same pipeline.
- **Fig 08E** — runtime/memory scaling vs observations, area, cameras.
- **V08** — coverage-overview animation.

## Implemented now

| Item | Tag | Note |
|---|---|---|
| `warehouse_full_4cam.world.sdf` + layout doc + map assets | established (asset) | the clean world |
| Day-zero artifact `paper_artifacts/gp/warehouse_full_4cam_dayzero_v1` | model_plumbing | per-camera + union/best/coverage maps from calibration only; `training_data_used: false`, `ground_truth_used: false`; union 99.2%, overlap 42.2% |
| 4-cam campaign config (`scripts/visibility_comparison/warehouse_full_4cam_campaign.yaml`) + world/prior generators (`make_warehouse_full.py`, `build_full4cam_planner_prior.py`) | model_plumbing | |
| Commissioning collection stack (launch `warehouse_full4cam_commissioning.launch.py`, operational recorder, route driver, camera contracts, leakage firewall, asset tests — all retargeted to cameras A–D) | established (infra) | routes for full_4cam geometry **not yet designed** (`config/study.yaml` routes: []); per-camera GP fits blocked on data (never fabricate) |
| Four-camera showcase deck (`../presentations/2026-07_four_camera_showcase/`) | presentation | pitch built from the day-zero artifact |
| Retired-world exploration | historical | worlds+tooling archived 2026-07-15; development history only |

## Gap → next experiments

1. Design the full_4cam collection routes (single-camera regions for A–D, adjacent-overlap
   corridors, one camera-gap crossing) and fill `config/study.yaml` routes.
2. Run per-camera D0/D1 commissioning passes with the retargeted stack
   (`warehouse_full4cam_commissioning.launch.py` + recorder + route driver).
3. Fit τ_A..τ_D with the frozen pipeline → Figs 08B–08D.

## Gate

No method development. No fitted-map claim for a camera until that camera has a real
detector evidence chain. Deck plots from the day-zero artifact are `MODEL/PRIOR`, never
"learned".
