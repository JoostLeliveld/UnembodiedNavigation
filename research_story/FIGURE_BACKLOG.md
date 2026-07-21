# Figure backlog — highest-value plots first (plan §7)

Status: `DONE` (exists, linked) · `DERIVABLE` (data exists, script needed) · `NEEDS-DATA`
(logger/capture missing) · `PLANNED` (later chapter). Every figure gets a provenance JSON
next to it (pattern: `paper_artifacts/figures/*.provenance.json`) and data-source labels
(`BELIEF` / `PIXEL` / `MODEL` / `GT — evaluation only`).

Output convention: figure scripts live in the owning study (`experiments/<study>/`),
rendered figures in `logs/studies/<study>/...`, and the chapter's `evidence.yaml` points at
them. Promote to `paper_artifacts/figures/` only when locked.

## Tier 1 — immediate, from existing original-warehouse logs (decision-enabling)

| # | Figure | Chapter | Data source | Status |
|---|---|---|---|---|
| 1 | Route + belief covariance ellipses + detector hits/misses (01A) | 01 | `honest_campaign_v1` runs via `campaign_metrics.load_run` (column traps — see campaign-metrics skill) | DERIVABLE |
| 2 | Covariance trace + actual position error vs time (01B) + coverage calibration (01C) | 01 | same; GT eval-only; note exp5 NEES 16.8→2.8 smoothing result | DERIVABLE |
| 3 | YOLO confidence vs projected bottom-centre BEV error (02A) | 02 | `warehouse_visibility_capture_v1` + targets; exp0 partially covers (beware Simpson confound) | DERIVABLE |
| 4 | Confidence reliability diagram (02B) | 02 | exp0_confidence_audit | PARTIAL-DONE (`logs/studies/optionA_commissioning/exp0_confidence_audit`) |
| 5 | Spatial hit-rate / miss-rate map (02C) | 02 | capture + targets v1 | DERIVABLE |
| 6 | Current trust map vs distance/FOV baseline (07A vs GP) | 07a | `paper_artifacts/gp/warehouse_visibility_gp_v1` + `calibrated_prior_v1` | PARTIAL-DONE (geometry study figs) |
| 7 | Existing C1/C2 paired route + outcome table (00B/00C) | 00 | `paper_artifacts/figures/paired_mechanism_*`, `docs/paper_vs_current/current/` | DONE |

## Tier 2 — small offline prototypes (method decision)

| # | Figure | Chapter | Status |
|---|---|---|---|
| 8 | 1-D uncertain-input GP demonstration (03A) | 03 | NEEDS-SCRIPT (synthetic) |
| 9 | 2-D anisotropic: point GP vs smoothing vs uncertain-input (03B) | 03 | PARTIAL (exp1_synthetic_gp exists — extend to U3/U5 panel) |
| 10 | Preliminary AWS point-GP vs uncertain-input-GP maps (03D) | 03 | DERIVABLE (`fit_belief_aware_gp` + `belief_gp_events`) |
| 11 | Trust / GP-uncertainty / R_plan triptych (05C) | 05 | DERIVABLE (blocked on r_miss_uv reconciliation for quoted numbers) |
| 12 | Preliminary availability map vs conditional-error map (04A) | 04 | DERIVABLE |

## Tier 3 — warehouse_full_4cam planning visuals (scope/feasibility, label PLANNED/MODEL in decks)

| # | Figure | Chapter | Status |
|---|---|---|---|
| 13 | Four-camera nominal coverage map (08A) | 08 | PARTIAL — layout map (`docs/assets/warehouse_full_4cam_map.png`) + day-zero coverage/best-camera maps (`warehouse_full_4cam_dayzero_v1`: union 99.2%, overlap 42.2%) + showcase renders |
| 14 | Planned per-camera data-collection routes | 08 | PLANNED (design for the current four-camera geometry) |
| 15 | Overlap and handover regions | 08/09 | PARTIAL — `overlap_handover_corridor.png` from the day-zero artifact (MODEL) |
| 16 | Dataset-size and runtime scaling table (08E) | 08 | PLANNED |

All Tier-3 material is `warehouse_full_4cam.world.sdf`; retired-testbed media
does not appear in decks.

## Standing figure IDs per chapter

Full specs are in [THESIS_PLAN_2026-07-15.md](THESIS_PLAN_2026-07-15.md) §5:
00A–00C, V00 · 01A–01D, V01 · 02A–02E, V02 · 03A–03F, V03 · 04A–04D, V04 · 05A–05D ·
06A–06D, V06 · 07A–07E · 08A–08E, V08 · 09A–09F, V09A–C · 10A–10E, V10.
