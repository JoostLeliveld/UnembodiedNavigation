# Study 1 — Uncertain-input GP reliability mapping (single camera)

**Question it answers:** *Does accounting for uncertainty in the robot
locations used to train a camera-reliability GP produce a better-calibrated and
more useful reliability map than treating estimated robot positions as exact?*

This is the **single-camera** study. It isolates the thesis's central scientific
mechanism — representing each detector outcome at a *distribution* over robot
position `(μ_t, P_t)` rather than a point — before any multi-camera fusion,
association, or selection complexity is added. The multi-camera work
(`experiments/multicamera_fusion_extension/`) is **Study 2** and is gated on this
study freezing first (see that study's ROADMAP; sequencing per the proposal §29).

## Paper framing (2026-07-21) — headline decided at checkpoint C1

This study is the roadmap's **Paper 1**. The roadmap reframes Paper 1 around a
*self-monitoring localization service* and demotes uncertain-input GP to an
optional ablation, whereas this study's original framing makes UIGP the central
mechanism. Per the 2026-07-21 scoping decision **both framings are kept alive** and
resolved at **checkpoint C1** (the ch.03 / Experiment-B gate): if UIGP (U5) beats
point-input/smoothing on route-disjoint held-out data → keep the UIGP headline; if
UIGP ties at real σ (the current leading finding) but the learned service still
beats geometry-only → adopt the service headline; if neither beats geometry → fall
back to the geometry service model and move novelty to self-monitoring (WP5). The
plans are written to survive either outcome. See `plans/PAPER1_WP_CROSSWALK.md` and
`../../research_story/PROGRAMME_ROADMAP_2026-07-21.md`.

## Intellectual anchor

O'Callaghan et al., *uncertain-input Gaussian-process contextual mapping*: a
standard GP silently treats every logged observation as occurring at one exact
point, which lets a noisy location distort the wrong part of the map. Their
extension represents each training input as a Gaussian and integrates the kernel
over it, so an uncertain observation's influence is spread and weakened by its
variance. We map that idea from **occupancy** onto **camera reliability**:

| O'Callaghan et al. | This study |
|---|---|
| occupancy probability `p(O\|x)` | usable-observation probability `p(G\|x)` |
| laser-return location | robot location of a detector outcome |
| robot localization uncertainty | belief covariance `P_t` at the logged location |
| standard GP = exact inputs | `naive` mode (mean-only) |
| uncertain-input GP = distribution inputs | `expected_kernel` / `belief_spread` modes |
| occupancy map → planning | reliability → **separate** `R_plan` mapping |

The GP predicts **reliability**, never covariance directly; reliability is
converted through a separate observation model into `R_update` / `R_plan`.

## Serves these research_story chapters

| Experiment | research_story chapter | What it establishes |
|---|---|---|
| **A** controlled covariance-fidelity (γ/α) sweep | `03_uncertain_input_gp` | mechanism correctness — U5 vs U1–U4 as reported input covariance is scaled; *controlled ablation, not operational evidence* |
| **B** real logged warehouse reliability prediction | `03_uncertain_input_gp` (+ `04_factorised_observation_model`) | held-out route-disjoint NLL/Brier; availability `a(s)` vs quality `q(s)` |
| **C** localization effect (L0–L4) | `05_trust_to_rplan` | does the map change the measurement update — NIS/NEES/ATE tail |
| **D** closed-loop navigation (P0–P4) | `06_original_warehouse_navigation` | clean-goal rate / GT breaches under reliability-aware `R_plan` |

## Two hard rules for this study

1. **Reuse code, run NEW data.** Every algorithm this study needs already
   exists (see `plans/REUSE_MAP.md`). Do NOT reimplement the GP, the calibrators,
   or the covariance mapping. But do NOT re-fit the stale
   `belief_aware_gp_score_v1` / `belief_gp_events` captures and call it the new
   result — collect a **new** `warehouse_aws` capture. See `plans/NO_SHORTCUTS.md`.
2. **Two-world rule holds.** All method development and Study-1 evidence come
   from the ORIGINAL `warehouse_aws` world (good detector, real data available).
   `honest_campaign_v1` stays frozen; the 4-cam world is Study 2's concern only.

## Layout

- `plans/` — one file per experiment + `REUSE_MAP.md` + `NO_SHORTCUTS.md`.
- Outputs → `logs/studies/single_camera_uigp_reliability/<expN_name>/` with a
  `RESULTS.md` each (per CLAUDE.md; nothing lands in the repo tree).
- Register promoted results in `research_story/03_uncertain_input_gp/evidence.yaml`
  and `research_story/registry.yaml`.
