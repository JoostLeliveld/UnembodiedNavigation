# Demo-layer restructure plan — contributions demonstrated separately (2026-07-16)

> **Correction (2026-07-20), after building the first instance (D4):**
> `experiments/demos/` is NOT free — it already exists for an unrelated
> purpose (single-camera-campaign README media: `outcome_counts_by_condition.png`,
> `campaign_result_table.png`, referenced from `experiments/README.md`). Do not
> repurpose it as the demo-layer root; the "Repo structure" section below is
> aspirational for that one path and should read `experiments/<owning_study>/`
> instead of a shared `experiments/demos/` — each demo lives in the study that
> owns its data (code in that study's `tools/`, outputs in that study's
> `logs/studies/<study>/demos/<dN_name>/`), with its own short markdown doc
> (see `experiments/multicamera_commissioning_bigwarehouse/DEMOS_D4_CAMERA_METHOD.md`
> for the pattern). A shared cross-study index can still exist later as a
> pure links-page once more than one study has a demo.
>
> **Also noted while building D4:** `research_story/presentations/2026-07_four_camera_showcase/full_story_walkthrough/`
> is being actively extended by a concurrent workstream (new camera-C story
> renderer, readiness-audit figures, TALKING_POINTS edits across stages
> 03–07, as of 2026-07-16). That folder is the natural eventual home for the
> "retarget the walkthrough to consume demo outputs" step in the Execution
> Order below — do not touch its files while that workstream is live; land
> new demos as standalone, run-anywhere scripts first (as D4 did) and wire
> the walkthrough retarget as a separate, later, coordinated step.

Problem being solved: the repo proves things (manifests, gates, CSVs) but does
not SHOW things. Current showcase renders fail a new reader in five specific
ways:

1. **Internals without a question.** Trust heatmaps, GP posteriors, and
   pipeline box-diagrams appear without the failure they exist to prevent. A
   heatmap with no route, no events, no consequence overlaid is wallpaper.
2. **No with/without.** Almost no figure pairs the method against its absence
   on the same seed. A reader cannot see what any contribution *changes*.
3. **No motion.** Every phenomenon that matters here is temporal — belief
   drift, staleness, trust collapse, hysteretic switching, fallback. Static
   figures make the mechanisms invisible. (The strongest assets in the whole
   thesis are the midterm C1-vs-C2 gifs — real footage, paired, motion.)
4. **Numbers without anchors.** "0.247 m disagreement" means nothing until it
   is drawn next to the robot footprint and the 0.30 m gate line inside the
   actual aisle.
5. **Text boxes as figures.** `05_actual_algorithm_execution.png` is four
   rounded rectangles containing sentences.

## The fixed demo grammar (every contribution, same four panels + one animation)

| panel | question it answers |
|---|---|
| **P1 The problem, happening** | "Why does this need to exist?" — the failure/ambiguity visible in the world (not described) |
| **P2 The mechanism, overlaid** | "What does the method compute?" — its quantity drawn ON the warehouse map / camera frame, with events |
| **P3 With vs without, paired** | "What changes?" — same seed, method on/off, side by side |
| **P4 The verdict, anchored** | "Did it work?" — the metric with the gate drawn as a line and the physical anchor (robot footprint / aisle width) |
| **A One animation** | the temporal mechanism as a gif/mp4 (≤20 s, captioned, colorbar fixed) |

A demo that cannot fill P1 and P3 is not a contribution demo yet — it is a
status report; the registry should say so.

## Repo structure (respects existing layout rules; no new top-level dirs)

```
experiments/demos/                       # already exists — becomes the demo layer
  README.md                              # DEMO INDEX: table of claim | hero thumb | command | runtime
  d1_initialization/                     # one folder per contribution (below)
    run.sh                               # ONE command, re-renders everything from cached data
    README.md                            # claim (one sentence), what you will see, runtime, data provenance
    render_*.py                          # figure/animation scripts (import scripts/shared + owning study code)
  d2_data_collection/ ...
logs/studies/demos/<dN_name>/            # rendered outputs (figures, gifs, provenance JSONs)
paper_artifacts/figures/demos/           # promoted, locked hero assets only
research_story/<chapter>/evidence.yaml   # each chapter gains a `showcase:` entry pointing at its demo
research_story/registry.yaml             # per-chapter demo status: NONE | STATUS-REPORT | DEMONSTRATED
```

Standards (all demos): one command; re-render from cached/logged data with NO
Gazebo dependency (capture happens once, rendering forever); provenance JSON
per asset; data-source labels (BELIEF / PIXEL / MODEL / GT-eval-only) stamped
on the figure; fixed camera colors (A blue #2f80ed, B green #21a366, C purple
#8d53c7, D orange #ed8a25); the same warehouse base-map underlay everywhere;
robot footprint + 1 m scale bar on every spatial figure; every compared number
gets its gate drawn.

## The demos (mapped to chapters and to the 2026-07-16 research-plan modules)

### D1 — Initialization: day-zero geometry prior (ch 07/08)
- P1: robot enters a *never-seen* facility; blank trust vs the question "where
  can each camera be believed on day zero?"
- P2: raycast visibility fans from each of the four mounts → the day-zero
  prior map building up camera by camera (animation: fans sweep, map fills).
- P3: day-zero prior vs the later *learned* map on identical color scale —
  "what geometry alone predicted vs what data proved" (Spearman 0.73 becomes
  a picture, not a number).
- P4: coverage/overlap stats (99.2% union, 42.2% overlap) drawn on the map,
  not in a table.
- Data: exists today (`warehouse_full_4cam_dayzero_v1`, geometry module). **Derivable now.**

### D2 — Data collection: operational recording + leakage firewall (ch 01/10)
- P1: the trap — a logger that reads simulator truth "works" and lies
  (odom-as-truth incident as one panel: wheel-odom vs GT diverging in a turn).
- P2: split-screen "what the SYSTEM sees" (contracts, noisy odom, ages) vs
  "what only EVALUATION sees" (GT stream, red-bordered) while the robot
  drives; events accumulating on the map per camera.
- P3: the same run scored against odom-as-truth vs against GT — the bias
  audit that flipped the C↔D attribution is the perfect exhibit.
- P4: firewall diagram generated FROM `leakage_firewall.yaml` (never drawn by
  hand) + match-rate/coverage stats.
- Data: exists today (gt_validation runs, firewall config). **Derivable now.**

### D3 — Updating method: uncertain-input / expected-kernel GP (ch 03)
- P1: the 1-D didactic — same observations, but input uncertainty grows;
  naive GP confidently wrong vs expected-kernel honestly wide (animation:
  belief ellipse inflates, posteriors react).
- P2: real 4-cam events with their belief ellipses feeding the fit — show the
  INPUT uncertainty on the map, which no current figure does.
- P3: naive vs uncertainty-weighted vs belief-spread vs expected-kernel
  posterior maps, same events, same color scale (the protocol's four modes).
- P4: held-out NLL/ECE ladder with the baseline (constant/geometry) drawn as
  reference lines.
- Data: partially derivable now (synthetic + pilot events); final panels
  blocked on research-plan **M3** (route-disjoint fits, post detector retrain).

### D4 — Camera method: per-camera trust + projection calibration (ch 02/04 + new)
- P1: one frame per camera of the SAME robot pose with the four projected
  points scattered ~0.27 m apart — disagreement made visible at robot scale.
- P2: bias arrows (pred → GT) per camera on the map, before calibration:
  every camera pulls toward its own wall; the distance-dependence as
  arrow-length growing along the aisle.
- P3: same arrows after `correction = intercept + slope·d` — arrows collapse;
  C's residual cross-bearing arrows near the pillar left visible and labeled
  (honest remainder).
- P4: C↔D synchronized disagreement before/after (0.247 → 0.078 m) as paired
  dots under the 0.30 gate line, with the robot footprint for scale.
- Data: exists today (both GT runs + v2 calibration). **Derivable now — highest value/effort ratio in the backlog.**

### D5 — Handover method: hysteretic reliability-aware selection (ch 09)
- P1: why not switch greedily — direct-switching replay oscillating between
  cameras in the overlap corridor (animation: selection strip flickers).
- P2: the mechanism timeline — per-camera operational score traces with the
  0.45 threshold band, the 0.08 margin band, the 3-frame streak counter, and
  the selection strip underneath; rejection reasons as glyphs (stale /
  low-trust / inconsistent).
- P3: M8 vs direct-switch vs fixed-preferred on the same replay: switches,
  gaps, NIS around handovers — three selection strips stacked.
- P4: the release-cliff curve (threshold sweep) with the frozen 0.45 marked —
  turning today's "0 corrections" from a dead end into an explained,
  disclosed operating point.
- Data: partially derivable now (pilot replay + sweep); full P3 blocked on **M5**.

### D6 — Closed loop & safety (ch 09/11)
- The midterm-gif format, upgraded: paired same-seed runs, passive vs active
  handover, external-camera footage + map inset with live trust/selection
  overlay; stop-safe vs collision outcomes as the verdict.
- Blocked on **M7/M8** (shadow agreement, gated activation). Design now,
  record once.

### D7 — Robustness: fault injection (protocol D5 conditions)
- One animation per fault: camera blackout at t=X → selection strip falls
  back within N frames, covariance visibly inflates, no false-confident
  updates; the monotonicity check ("degradation never raises confidence") as
  a per-condition panel.
- Blocked on **M6**.

## Execution order (value ÷ effort, respecting data availability)

1. **D4** (all data exists; the calibration story is the strongest new result)
2. **D2** (data exists; establishes the credibility frame every other demo leans on)
3. **D1** (data exists; cheap, opens the story)
4. **D5** P1/P2/P4 from pilot replay now; P3 after M5
5. **D3** didactic panels now; real panels after M3
6. **D7**, then **D6** as the campaign lands (design their scripts early so
   collection records what rendering needs — e.g., camera frames WITH overlay
   inputs, decision streams at full rate)

Retarget `research_story/presentations/.../full_story_walkthrough` stages to
CONSUME demo outputs instead of maintaining bespoke renders (delete
`render_actual_commissioning_showcase.py`-style one-offs after migration).
Registry rule: a chapter may not claim DEMONSTRATED until its demo fills all
four panels + animation from real (non-synthetic) data.

## Definition of done for the layer

A new reader opens `experiments/demos/README.md`, sees seven rows each with a
hero thumbnail and a one-sentence claim, runs any `run.sh` in under 5 minutes
without Gazebo, and can answer for each contribution: what failed without it,
what it computes, what changed with it, and whether the gate passed.
