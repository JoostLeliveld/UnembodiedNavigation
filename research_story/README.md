# Research story — the investigation, chapter by chapter

One numbered folder per storyline chapter. Each chapter is an **investigation**: a research
question, what a contribution would look like if the answer is yes, the concrete result we
are aiming for, what is implemented today, and the gate that decides whether we proceed.
Chapters hold claims and `evidence.yaml` manifests pointing at real artifacts — never copies.
Master plan: [THESIS_PLAN_2026-07-15.md](THESIS_PLAN_2026-07-15.md). Statuses:
[registry.yaml](registry.yaml). Figures: [FIGURE_BACKLOG.md](FIGURE_BACKLOG.md) +
[MEDIA_INDEX.md](MEDIA_INDEX.md). Chapter media views: `<chapter>/{figures,videos}/`
(rebuild: `python3 _tools/link_media.py`).

Two-paper programme view (Paper 1 = single-camera self-monitoring localization
service, chapters 00–06; Paper 2 = multi-camera fault tolerance, chapters 07–11;
Paper 3 = placement/fleet, PARKED) mapped onto these chapters + the investigation
studies: [PROGRAMME_ROADMAP_2026-07-21.md](PROGRAMME_ROADMAP_2026-07-21.md).
Governance layer: [`_shared/evidence_classes.md`](_shared/evidence_classes.md),
[`_shared/ground_truth_firewall.md`](_shared/ground_truth_firewall.md),
[`_shared/module_gates_and_stop_rules.md`](_shared/module_gates_and_stop_rules.md).

## The two worlds (hard rule)

| | **Original warehouse** | **Full four-camera warehouse** |
|---|---|---|
| World file | `warehouse_aws.world.sdf` | `warehouse_full_4cam.world.sdf` (24.5×20.5 m, cameras A–D wall-mounted at (±6, ±10, 6.1), layout: `docs/warehouse_full_4cam_layout.md`) |
| Role | isolate mechanisms, compare models, causal evidence | scale, coverage, heterogeneity, handover, fusion |
| Method development | **yes** | **no — frozen methods only** |
| Chapters | 00–07 | 08–10 |

**These are the ONLY two worlds in the repo** (consolidated 2026-07-15). Other development
testbeds and their tooling were deleted and archived (`../../_archive/`). The multicamera
commissioning study was retargeted to `warehouse_full_4cam` (cameras A–D; routes pending
redesign); all data from retired worlds is **historical** and must be re-collected on
`warehouse_full_4cam` before appearing as thesis evidence. `honest_campaign_v1` is the locked AWS reference
campaign; never rerun or modify it.

## The storyline in one causal chain

```text
00  A fixed external camera helps unevenly — and reliability-aware planning already wins   [LOCKED]
01  Ordinary driving yields trust-training records at UNCERTAIN robot positions            [PARTIAL]
02  What signal should "trust" even be?                                                    [PARTIAL]
03  Learn the trust field AT those uncertain positions (uncertain-input GP)  ← CONTRIBUTION 1 CORE
04  Availability ≠ conditional accuracy (factorised model)                   ← option A, 2nd contribution
05  Trust → R_plan through one frozen, bounded interface                                   [BLOCKED on freeze]
06  Close the loop: does the operational trust map still win at navigation?               [PLANNED]
07  How much is free before driving? (FOV/range prior; geometry only if it earns it)       [ACTIVE]
08  Freeze everything, transfer per-camera to warehouse_full_4cam                          [PRECURSOR ACTIVE]
09  Select, hand over, fuse — without overconfidence                         ← option B, 2nd contribution
10  Drive less, learn the same map (active commissioning)                    ← future paper
11  Final frozen evidence package                                                          [LAST]
```

## Investigation board — question / implemented / aiming for

| Ch | Question | Implemented now | Result we're aiming for |
|---|---|---|---|
| [00](00_problem_and_existing_baseline/) | Does spatial camera trust matter downstream? | **Answered**: C1 15/20 (4 GT breaches) vs C2 20/20 (0), 40 locked runs | — (frozen anchor) |
| [01](01_operational_belief_and_logging/) | Do real drives produce usable (belief-uncertain) training records? | belief-stamped events; NEES honesty study (16.8→2.8 after smoothing) | Fig 01B/01C: covariance tracks error → **GO** for ch.03 |
| [02](02_trust_target_and_calibration/) | What should the GP learn? | exp0 confidence audit (Simpson-confounded); detection-rate switch precedent | Fig 02D target table → scalar vs factorised decision |
| [03](03_uncertain_input_gp/) | Does modelling input uncertainty beat smoothing/weighting? | expected-kernel GP code + first fits; synthetic harness; "tie at real σ" finding | Fig 03E: U5 degrades most gracefully over α-sweep, route-disjoint |
| [04](04_factorised_observation_model/) | Are availability and conditional noise different fields? | nothing (opens on 02's gate) | Fig 04A two-maps + NIS calibration win over scalar trust |
| [05](05_trust_to_rplan/) | Can the interface be frozen, bounded, monotone? | 3 implementations (divergent 40 vs 120 px); mapping study | one reconciled implementation + Fig 05C triptych → FREEZE |
| [06](06_original_warehouse_navigation/) | Does the operational map still win closed-loop? | C1/C2 harness, replay tooling | Table 06C: N4 ≥ N1/N3 on goals + breaches + calibration |
| [07](07_weak_priors_and_geometry/) | What's free from calibration/geometry before driving? | geometry module, S1 PASSED (ρ=0.73); calibrated prior; day-zero 4cam prior | Fig 07B data-efficiency curves; 07E geometry-vs-FOV verdict |
| [08](08_large_warehouse_scaling/) | Does the frozen pipeline transfer per-camera at scale? | 4-cam world + day-zero artifact (99.2% union / 42.2% overlap); collection stack retargeted to A–D (routes pending) | Fig 08C: four distinct calibrated trust fields, zero retuning |
| [09](09_multicamera_handover_fusion/) | Can we select/hand over/fuse without overconfidence? | M8 hysteretic manager, S0–S4 baselines, inflation; historical 2-cam pilot's D2 fail (retired world) defines what to measure | Fig 09D: calibrated fusion, no overconfident overlap updates |
| [10](10_active_commissioning/) | Can informative routes cut commissioning driving? | nothing (by design) | Fig 10B: A5 reaches target map quality with least driving |
| [11](11_final_thesis_campaign/) | — | checklist only | frozen thesis evidence package |

## What the contributions look like

**Contribution 1 (primary, fixed)** = 02 + 03 + 05 + 06:
> *A passive commissioning method for learning spatial external-camera trust from ordinary
> robot driving while accounting for uncertainty in the robot locations assigned to camera
> observations.*
Earned when: the ch.01 covariance gate passes; ch.02 fixes the target; ch.03 shows the
uncertain-input GP beats point/smoothing/weighting baselines route-disjoint (or documents
the honest null); ch.05 freezes the interface; ch.06 shows N4 beats realistic baselines
closed-loop.

**Contribution 2 — choose ONE after ch.03/06:**
- **Option A (statistical)** = 04: *a factorised observation model separating availability
  from conditional localisation noise.* Earned only if both components are learned,
  independently validated, beat scalar trust, and change R_plan/navigation measurably.
- **Option B (systems)** = 08 + 09: *reliability-aware selection and conservative fusion
  across a network of fixed warehouse cameras.* Earned only when every camera has a real
  detector evidence chain, overlap disagreement is calibrated, and a closed-loop handover
  campaign exists (the current D2 fail shows exactly what's missing).

**Future paper** = 10: *safe informative route selection for commissioning reliability fields.*

**Supporting only** (never headline): confidence calibration, weak FOV/range prior,
trust→R_plan mapping, onboard sensors, the 4-cam world itself. Geometry/occlusion (07b) is
promoted only with realistically available geometry AND a clear win over range/FOV.

## Honesty rules (every chapter)

- Evidence tags: `established` / `measured_in_sim` / `model_plumbing` / `open`
  (defined in [`_shared/honesty_tags.md`](_shared/honesty_tags.md)).
- GT (`gt_*`, oracle labels) and CAD are **evaluation-only**; figures/videos carry
  `BELIEF` / `PIXEL` / `MODEL` / `GT — evaluation only` labels.
- Unrun plots in decks are `PLANNED` / `HYPOTHETICAL`.
- Negative results are results: the D2 overlap-gate failure and the "methods tie at real σ"
  finding stay in the story.

## Where things live

Code: `experiments/<study>/` · outputs: `logs/studies/<study>/` · locked artifacts + media:
`paper_artifacts/` · storyline + presentations: here (`presentations/`). Conventions:
`../CLAUDE.md`. Mapping of the plan's ideal `thesis/{src,data,research_story}` tree onto this
repo (kept because the ROS workspace + 4.5 GB append-only logs make literal renames wrong
mid-thesis):

| Plan location | Actual location |
|---|---|
| `thesis/src/reliability/…` | `src/reliability/` (runtime) + `scripts/visibility_comparison/fit_belief_aware_gp.py` (canonical GP) + `scripts/shared/` (metrics) |
| `thesis/data/<world>/{raw,processed,manifests}` | `logs/visibility_comparison/` + `logs/studies/` + `paper_artifacts/` manifests |
| `thesis/research_story/NN_…` | this directory |
| `experiment_registry.yaml` | [registry.yaml](registry.yaml) + `docs/experiment_registry.md` (artifact chain) |
| `current_runtime_contract.yaml` | `docs/current_runtime_contract.yaml` |
