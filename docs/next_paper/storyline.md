# Storyline: from a learned map to self-commissioning observability

## Recommended research question

> **Can a robot safely initialize and continually improve the external-camera
> observation-reliability model used for navigation in a previously unseen or
> changed warehouse, without an offline detector survey?**

The phrase *observation reliability* is deliberate.  The field predicts how
trustworthy a camera-derived position measurement will be; it is not an
obstacle map, a traversability map, or a direct visibility reward.

## Why this is the right sequel

The completed paper answers a downstream question:

```text
offline learned reliability map → state-dependent R_plan → safer route choice
```

That result leaves a practical deployment gap.  A new camera pose, moved rack,
or new facility requires a costly pre-capture before the planner has its map.
The post-paper work supplies the missing loop:

```text
calibration + sensed structure
        ↓
conservative day-zero reliability prior + local prior strength
        ↓
normal driving: detected / missed camera observations + belief covariance
        ↓
posterior reliability map + explicit map-confidence
        ↓
updated R_plan → visibility-aware navigation and map-health monitoring
```

This turns the fixed external camera from a one-off, hand-surveyed sensor into
a **self-commissioning localization infrastructure**.  It is a clean research
advance because it changes the source and lifecycle of the existing planner
input, while retaining the validated downstream planning mechanism.

## Narrative arc for a paper, meeting, or pitch

### 1. Begin with the completed result — establish credibility

“We already know why reliability matters for planning: an offline learned map
changes the observation covariance and the robot selects routes where it can
stay localized.”  Use the archived/current paired-route result only as context;
do not spend the talk re-proving the first paper.

### 2. Reveal the deployment failure

“But that map is not available on day zero.”  A detector survey is
layout-specific, expensive, and stale after a layout change.  A uniform or
overconfident map makes unexplored shadow regions look safe exactly when the
robot lacks evidence.

### 3. Introduce the new insight

“The missing map is not blank knowledge.”  Known camera calibration tells us
FOV, range, image-edge and projection-conditioning effects.  A one-time sensed
height map can add viewpoint-specific occlusion.  Those signals form a prior
mean.  Just as importantly, their confidence becomes a spatial **prior
strength**: confident cells resist a few noisy observations; ambiguous and
unobserved cells yield quickly to operational evidence.

### 4. Close the loop with driving

“Every ordinary drive is an experiment.”  A detection and a miss are both
evidence.  An observation is deposited at the estimated robot position; when
the position belief is broad, the update is spatially spread rather than simply
discarded.  This preserves hard failure evidence while avoiding a falsely
precise map.

### 5. Make safety the differentiator

“The objective is not merely a higher map AUROC.”  A cold-start reliability
model must be conservative enough to keep the robot in an observable core,
then improve sample-efficiency and route availability as it gathers evidence.
The key failure is **false-safe confidence in an unvisited region**, not an
ordinary classification error alone.

### 6. End with scale — but do not overclaim it

“Multi-camera infrastructure is the natural scale-out.”  It reduces geometric
blind regions, but introduces a new uncertainty at camera handover.  Present it
as a controlled scalability ablation: each camera keeps its own reliability
model and source switches inflate covariance until overlap agreement confirms
them.  It becomes a separate paper only after the handover evidence chain is
closed.

## Paper-level claims to earn

| Claim | What must support it |
| --- | --- |
| A calibrated/sensed prior is useful before any detector survey. | Held-out spatial and cross-layout reliability labels; compare with uniform and calibration-only priors. |
| The prior is safe when wrong. | Measure false-safe area/exposure and calibration on unvisited cells, not only average error. |
| Driving improves the map efficiently. | Frozen operational trajectories or prospective survey runs; learning curves on disjoint, unvisited evaluation cells. |
| Prior strength + covariance-aware deposition give a better update policy. | Ablate low/high spatial prior strength, naive deposition, covariance spread, and hard gating. |
| Better maps improve operation. | Matched seeded closed-loop runs using only the permitted deployment inputs; report success, collision/stop/retreat, exposure, and calibration. |

The existing paper already supports the *last link* — better `R_plan` can affect
navigation.  The sequel must establish the first four links without silently
using CAD, ground truth, or an offline oracle as a deployment input.

## Testable hypotheses

**H1 — useful day-zero structure.**  A calibration-plus-sensed-structure prior
has lower unvisited-cell error and lower false-safe exposure than a uniform
reliability field.  Calibration-only is an important low-cost baseline; it
separates FOV/range value from the value of sensed occlusion.

**H2 — conservative uncertainty accelerates safe adaptation.**  Spatial prior
strength lets ambiguous cells change quickly while preventing confident,
well-supported cells from flipping after isolated noise.  This should improve
the safety–sample-efficiency frontier, not just final map score.

**H3 — normal driving is enough to close the residual.**  With the same
mission/survey budget, a day-zero prior plus updates achieves a given map
quality with less dedicated detector sampling than data-only learning.

**H4 — navigation benefits survive the cold start.**  With no pre-fitted map,
the updated field leads to fewer localization-risk excursions and better
closed-loop outcomes than uniform- `R` and a stale/static-prior control.

**H5 — optional scale-out.**  Independent per-camera priors plus
handover-aware covariance inflation reduce blind coverage without producing
overconfident source-switch corrections.

## Minimal, defensible experiment matrix

### Worlds and split

Use at least three static warehouse layouts: a source layout, an unseen
rearrangement, and a deliberately hard changed-layout case with a tall pallet
or rack shadow.  Split **by spatial block and by layout**, never by adjacent
frames.  Detector labels and ground-truth position are evaluation-only.

### Conditions

| ID | Deployment-time reliability source | Why it is needed |
| --- | --- | --- |
| B0 | Uniform `R_plan` / no reliability map | Existing planner baseline. |
| B1 | Calibration-only prior: FOV, resolution, image boundary | Tests how much is free from calibration. |
| B2 | Sensed-structure prior with low/variable prior strength | Proposed day-zero method. |
| B3 | B2 + online hit/miss updates | Primary method. |
| B4 | Data-only online learner from an uninformative prior | Separates prior value from learning value. |
| A1 | B3 with hard confidence gating | Negative control; tests whether discarded uncertain samples hurt. |
| A2 | B3 with covariance-spread deposition | Robustness variant; do not assume it wins over naive deposition without a new split. |
| U | Offline dense detector map | Evaluation-only upper reference, never a deployment baseline. |

The previous paper's constant-`R` and offline-GP conditions remain useful
anchors, but they do not answer the commissioning question by themselves.

### Metrics — report them in this order

1. **Day zero:** Brier/log loss, AUROC/rank correlation, reliability diagram,
   and false-safe-cell fraction on unvisited spatial blocks/new layouts.
2. **Learning:** metric-versus-observations curves on untouched evaluation
   cells; map-change magnitude; time/distance to a calibration target.
3. **Safety:** distance/time in cells predicted safe but measured unreliable,
   stop/retreat triggers, collision and timeout rates, and confidence coverage.
4. **Navigation:** clean goal rate, localization error conditional on outcome,
   route observability/exposure, and path/time only as secondary outcomes.
5. **Diagnostics:** NIS/NEES or coverage calibration, camera update age, belief
   covariance, and a layout-change residual map.

## What makes this publishable rather than an engineering demo

- It asks a general question about **observation-model commissioning under
  spatially structured, asymmetric uncertainty**.
- It couples a prior **mean** with a spatial **epistemic strength**, which is
  the mechanism that makes online corrections safe rather than merely smooth.
- It evaluates the question where it matters: unvisited and changed regions,
  rather than random frame splits from the same survey.
- It preserves strict input discipline: calibration and sensed structure are
  deployment inputs; CAD and ground truth score the system only.
- It creates an operational output — reliability, evidence, and
  prior-vs-observation residual — useful to facilities staff as well as the
  planner.

## The multi-camera branch: keep it coherent

The two/four-camera layouts, camera-specific fields, fusion interface, and
handover model are worth retaining because they pressure-test the central
principle: **a camera source is only useful when its uncertainty is trustworthy**.
For this paper, use two cameras only as a pre-registered scalability study:

```text
per-camera cold-start prior → per-camera online update → quality-aware selection
→ handover covariance inflation → fused localization/reliability map
```

Do not make “multi-camera fusion” the title until these are available: detector
evidence for camera B, overlap-disagreement calibration, replay baselines, and
a closed-loop handover campaign.  Otherwise it obscures the already strong
self-commissioning contribution.

## A 10-slide supervisory-meeting version

1. The paper result: reliability-aware planning works once the map exists.
2. The practical gap: every new/changed warehouse starts with no map.
3. The proposed question and what changes relative to the paper.
4. Day-zero prior: calibration → sensed structure → mean + strength.
5. Why strength matters: conservative unknowns and fast correction at boundaries.
6. Learning by driving: hits/misses, belief covariance, and the update loop.
7. Evidence so far: clearly tagged measured versus model/simulation results.
8. The decisive experiment matrix and anti-leakage rules.
9. Scale-out: camera coverage/handover as a bounded ablation.
10. Decision requested: approve the commissioning-first paper scope and run the
    frozen changed-layout campaign.
