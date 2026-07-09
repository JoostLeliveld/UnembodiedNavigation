# Deploying the camera-observability GP in a real warehouse

How a real facility would set up, bootstrap, and safely maintain the
observation-reliability map that the visibility-aware planner consumes — and why
the robot can drive safely while it collects the data to refine it.

The offline module and all figures referenced here live in
[`scripts/geometry_visibility/`](../scripts/geometry_visibility/); outputs (incl.
the operator dashboard) in `logs/geometry_visibility_prior/`.

---

## The problem this solves

The planner needs to know **where a fixed camera can reliably localize the robot**
(the `P(reliable detection)` field → measurement covariance `R_plan`). Today that
field is *learned* from a dedicated YOLO teleport-capture per world — data-hungry,
world-specific, and stale the moment a rack moves.

The fix: derive an **informed prior** for that field from geometry the facility
already has (or can sense), so the system works on day one and *refines itself
during normal operation*. Validated this session: geometry explains the learned
field at Spearman **0.73**; a **sensed** height map from infrastructure stereo
recovers that prior at **0.97** (no CAD needed); the prior predicts a *fresh*
independent YOLO capture at **0.68**; and driving cuts residual map error **3×**.

## Phased setup

**Phase 0 — Commission (one-time, no new data model needed)**
- Calibrate each fixed camera's intrinsics + pose in the warehouse frame (surveyed
  fiducials). This is the only genuinely new commissioning step.
- Reuse existing assets: the driveable-region map (from the AMR fleet's SLAM map or
  the WMS rack layout) gives free-space **and** obstacle footprints for free.
- If cameras overlap, run multi-view stereo for a metric height map. For
  infrastructure cameras this is the robust depth source — no monocular
  out-of-distribution risk (a top-down oblique view breaks monocular depth nets).

**Phase 1 — Cold-start prior (offline, before any driving)**
- Run the geometry chain per camera → FOV + raycast occlusion + range/obliquity →
  per-camera reliability → `R_plan`.
- Combine cameras as a **precision union** (a cell is observable if *any* camera
  sees it): 1→2 cameras took driveable coverage **39% → 78%** in our world.
- Result: a dense reliability map over the whole facility with zero collected data.

**Phase 2 — Refine during normal operation (online)**
- The robot runs real jobs; every detection (hit **or** miss) plus its belief
  covariance updates the GP. No dedicated mapping phase.
- Update rule (validated): apply each datapoint **at the belief position, spread by
  the belief covariance** — a confident fix deposits sharp, concentrated evidence;
  an uncertain fix spreads the *same* evidence over its belief ellipse. Do **not**
  hard-gate uncertain points — that discards the informative failure samples and
  underperforms (RMSE 0.083 gated vs 0.064 spread/naive).

**Phase 3 — Lifelong health**
- The prior-vs-observed **residual** flags change (a rack moved, a new pallet
  stack) → re-derive the geometry prior for that zone from fresh depth. The map
  maintains itself; no periodic full recapture.

## Can it collect data safely? Yes — safe by construction

The chicken-and-egg (need the map to drive safely; need to drive to build the map)
resolves on four legs:

1. **The prior makes the unknown look *dangerous*, not safe.** Occluded / far /
   uncovered cells get *low* assumed reliability, so the EFE planner routes around
   them. A conservative prior steers toward observable regions from the first move.
2. **Drive the confident core, grow outward.** Enter a region only where the prior's
   lower confidence bound is acceptable; expand as accumulated data raises
   confidence. Safe exploration using the GP variance you already have.
3. **Rising self-uncertainty pulls the robot back.** The same belief covariance that
   spreads a datapoint gates motion: if it grows past threshold (entering a blind
   spot), slow / stop / retreat — banking a low-observability boundary sample
   safely instead of getting lost. (This is the staleness/uncertainty limiter the
   campaign analysis found was missing.)
4. **The GP is not the safety system.** Onboard sensing (LiDAR/depth, wheel odom,
   IMU) is the collision fallback and runs regardless. The camera GP governs
   *global localization quality*, not last-resort obstacle avoidance.

**Hard prerequisite — a calibrated uncertainty estimate.** Our logs showed the EKF
`state_cov` was overconfident (σ≈0.05 m, flat) while the *planner belief*
covariance told the truth (up to 0.45 m, tracking the real 0.30 m drift). Motion
gating and learning-weight **must** use the honest (planner-belief) covariance — an
overconfident filter would let the robot think it is fine while drifting into
exactly the collision the campaign documented.

## Visualizations — match the view to the audience

The operator dashboard ([`operator_dashboard.py`](../scripts/geometry_visibility/operator_dashboard.py)
→ `logs/geometry_visibility_prior/demo/operator_dashboard.png`) is the operational
picture, built from real runs:

| View | Message | Audience |
|---|---|---|
| **Reliability** (green→red) | where the robot is trackable vs blind | operator |
| **Confidence** (evidence) | how much *real data* backs the map vs prior-only | operator |
| **Routes** (colored by reliability) | visibility-aware routing + data provenance | operator / engineer |
| **Day-one prior** | the map before any driving | commissioning |
| **Change vs prior** (diverging) | where reality ≠ geometry → investigate | facilities / health |
| **Multi-camera coverage** | "add a camera here to kill this blind spot" | commissioning |

Color means the same thing across panels (green = trackable everywhere; a
single-hue ramp for confidence; diverging-at-zero for change). Engineering/thesis
figures (init comparison, depth-source ranking, online-update video, the
confident-sharp/uncertain-spread mechanism) stay separate from the operator view.
Drop raw per-component maps (`f_occ`, `f_range`, Jacobian) from operator surfaces —
diagnostic, not communicative. **Productionization note:** the green→red
reliability ramp is the operator convention but is not colorblind-safe; a shipped
product should pair it with a secondary encoding or a CVD-safe ramp.

## What is validated vs open

**Validated this session:** geometry explains the learned GP (0.73); a **sensed**
height map (infrastructure stereo point cloud, no CAD) recovers the prior at 0.97
with ~18% honestly-unknown mutual blind spots; depth-source ranking (RGBD recovers
the prior only if range covers the scene; stereo robust; monocular OOD-risky);
multi-camera union (39%→78%); online refinement from real runs (3× error cut) with
the spread-not-gate update; the overconfident-EKF finding; and a live Stage-3
capture where the prior predicts fresh independent YOLO detection at 0.68.

**Deployment ladder (Level 2 footprint-only dropped — weak lift, fragile heuristic):**
Level 1 camera-only cold start (FOV+range+edge, needs no map) → Level 3 sensed
height map (infra stereo / robot LiDAR / RGB-D, via `height_map_from_points`) →
Level 4 empirical YOLO hit/miss refinement online. Geometry is the safe cold start
that makes online collection possible and the fallback where data is sparse.

**Open (needs a live loop, not done here):** closing Phase-2/3 in the running stack
(planner consumes the online-refined GP); uncertainty-gated motion; the *layout-change*
transfer claim specifically (needs a targeted capture in the aisle shadow — the
teleport grid did not sample it); a calibrated planner-belief covariance end-to-end.
