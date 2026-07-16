# Supervisor decision deck — outline (plan §6)

Goal: NOT eleven research directions — show what can be measured now and what each option
would prove, then ask for a scope decision. ~10 slides. Build on the TU/e template
(reuse `midterm_presentation/generators/` tooling).

Figure prerequisites = FIGURE_BACKLOG Tier 1 (items 1–7) + Tier 2 (8–12); Tier 3 (13–16)
shown as `PLANNED`. Concrete files for the READY slides: see `../../MEDIA_INDEX.md`.

| # | Slide | Content | Evidence source | Status |
|---|---|---|---|---|
| 1 | Warehouse problem | One camera, spatially uneven; existing C1/C2 route evidence | ch.00 (`paired_mechanism_*`, outcome table) | READY |
| 2 | Existing pipeline + realism gap | camera obs → trust field → R_plan → planner; missing: learn trust from ordinary driving without exact positions | ch.00 + `docs/contribution_map.md` | READY |
| 3 | Two-world strategy | mechanisms (AWS 1-cam) vs scale (full 4-cam) table | research_story/README table | READY |
| 4 | Data-collection reality | one AWS route: covariance ellipses, hits/misses, unobserved sections | Fig 01A | NEEDS FIG (backlog #1) |
| 5 | Is belief covariance meaningful? **GO/NO-GO** | tr(P) vs error, 90% coverage, error vs camera-update age | Figs 01B/01C (+ exp5 NEES caveat) | NEEDS FIG (#2) |
| 6 | Is confidence a useful target? | conf vs BEV error, calibration curve, miss-rate map | Figs 02A/02B/02C (Simpson caveat) | NEEDS FIG (#3–5) |
| 7 | Uncertain-input GP option | synthetic point-GP vs uncertain-input fig; U0–U6 grid | Figs 03A/03B | NEEDS FIG (#8–9) |
| 8 | Three contribution packages | A: lowest risk (03+02+05+06) · B: +factorised (04) · C: +multicam (08/09) | research_story/README | READY |
| 9 | warehouse_full_4cam role | clean 4-cam world: layout map + day-zero coverage/best-camera/overlap maps (union 99.2%, overlap 42.2%) — **all labelled MODEL/PRIOR**; learned per-camera maps / handover / dropout labelled PLANNED; mention the historical pilot's honest D2 fail only as “what we still must measure” | ch.08 (`warehouse_full_4cam_dayzero_v1`, showcase renders, layout doc) + ch.09 | READY (label!) |
| 10 | Scope recommendation | Contribution 1 on AWS first; then choose factorisation OR fusion; active commissioning = future work | plan §6/§10 | READY |

Decision requested: approve two-world scoping + choose the second-contribution track
(or defer the choice until slides 5–7's gates resolve).
