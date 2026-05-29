# Supervisor Progress Update — Visibility-Aware EFE Navigation
### Slide-by-slide storyline (AWS warehouse, task B1: apron → upper A3)

Figures live next to this file (`F1`–`F5`). Each slide below lists the talking
points, which figure to show, and the one-line takeaway ("so what").

---

## Slide 1 — Status since last meeting

- Three things since we last spoke: **(1)** simplified the AWS world,
  **(2)** wired and tested a fix for the "half-circle" planner behaviour, and
  **(3)** ran a full timing + closed-loop feasibility study.
- Headline: the visibility-aware **route choice now works** — but the runtime
  **cannot execute it yet**. I know exactly why, and what to fix next.
- *(no figure — title slide)*

**So what:** real progress on the method, one concrete engineering blocker left.

---

## Slide 2 — World simplified + GP refit  · [F1_world_and_task.png]

- Removed the tall R4 stack; R4 is now a single low crate like R1/R2/R3/R5, so
  the rack row is uniform and the visibility claim isn't confounded by one odd
  occluder.
- Re-fit the camera-visibility GP for the new geometry (`aws_gp_v5`) by
  **partial recapture** — only the ~30 samples whose camera ray crossed the old
  stack were re-shot; the other ~880 were reused.
- New diagnostic task **B1**: start in the A4 apron (3,−1), goal in upper A3
  (1,2). The only westward route must clear the R4 rack column.

**So what:** clean, uniform world; the R4 corner (red) is the geometric pinch point.

---

## Slide 3 — The mystery we chased  · [F4_closedloop_fail.png, right panel]

- Long-horizon plans kept driving in **half-circles / spin-in-place**, with the
  steering command pinned to its limit.
- Open question: was this the **horizon** (too short to see a way out), the
  **discretisation** (high ω_max, zero control penalty), or the **cost
  landscape** (a bad local optimum)?

**So what:** the planner "knew" where the goal was but kept taking a weird route —
we needed to know why before trusting any result.

---

## Slide 4 — Diagnosis + fix: multistart

- Diagnosis (confirmed offline): it's a **non-convex / local-optimum** problem.
  A single cold start falls into a bad basin; it is not a horizon limit.
- Fix: **multistart** — the optimiser is seeded with several initial guesses
  (straight-to-goal + two lateral "detour" seeds); the EFE objective scores them
  and keeps the lowest-cost one.
- Important for fairness: the **same seeds are offered to every condition**
  (C1/C2/C3). The route difference still emerges from the objective, not from
  hand-scripted waypoints.

**So what:** a principled, condition-neutral fix — not a route hack.

---

## Slide 5 — It works (offline)  · [F2_multistart_fix.png]

- Controlled offline test, terminal distance to goal vs horizon:
  - single-start stays stuck (≈ **1.6–2.3 m** from goal — the bad basin),
  - multistart **reaches the goal** at H≥80 (≈ **0.6 m** at H80/120,
    **0.26 m** at H200).
- The route it finds is the **visible detour**: up aisle A4, across the top
  cross-aisle, down into A3 (right panel).

**So what:** the half-circle problem is solved — the planner now finds the
sensible, camera-visible detour.

---

## Slide 6 — But it's too slow  · [F3_solve_scaling.png]

- The detour needs a **long horizon** (H≥80) to be found. But solve time grows
  steeply with horizon, and multistart multiplies it by the number of seeds
  (solves run sequentially).
- Single-start crosses the **1 s plan budget around H≈75**; multistart at
  H=200 takes **~20 s per plan** (≈20× over budget; ~28 s on a slower run).

**So what:** the very thing that makes the route findable (long horizon +
multiple seeds) makes it too slow to run in real time.

---

## Slide 7 — And closed-loop breaks  · [F4_closedloop_fail.png]

- In Gazebo, the robot **sits idle 6–10 s** while the first plan solves, then
  open-loop-follows that one stale plan (no time to replan).
- That plan's opening move is a hard pivot (steering saturated, zero control
  penalty); with no margin it **clips the south-east corner of the R4 rack**.
- **All five runs collide there** after ~0.8 m, regardless of horizon or
  multistart (left panel = all paths, X = collision).

**So what:** the route is correct on paper, but solve latency + a tight corner
mean it never actually gets driven.

---

## Slide 8 — Pipeline timing health  · [F5_pipeline_timing.png]

- The state estimate is **fresh** (belief age ~0.10 s, updated 10 Hz).
- Perception is the slow *sensing* link: YOLO runs ~**5–6 Hz** with ~**0.6 s**
  detection staleness (CPU, 1280×720) — but for a 0.22 m/s robot that's minor.
- The **solver is the real bottleneck** (seconds-scale vs a 1 s budget).

**So what:** sensing and estimation are fine; the planner solve time is what
breaks the loop.

---

## Slide 9 — Blockers / decisions needed

1. **Solve latency.** Either (a) "plan-once" mode (accept one long solve, then
   execute), or (b) cut candidates + optimiser iterations + horizon to get under
   the 1 s budget. Trade-off: the detour needs H≥80, so (b) has a floor.
2. **Corner clearance.** Raise the no-go safe distance and/or add a small
   control penalty / lower ω_max so the plan keeps margin from the R4 corner and
   doesn't pivot-and-clip.
3. **Task geometry.** B1's start is boxed into the apron with a tight corner —
   may want more lateral room, or accept the south detour as success.
4. *(Optional)* GPU YOLO to cut the 0.6 s perception staleness.

**So what:** one focused engineering pass unblocks a clean executed result.

---

## Slide 10 — Proposed next step

- One experiment: **plan-once, long-horizon, multistart** + raised no-go margin
  + small control penalty, on B1.
- Expected outcome: the robot actually **drives the visible detour** end-to-end
  (the offline result, executed) — giving the first trustworthy C1-vs-C2
  comparison on this task.

**So what:** clear, single next milestone to turn "the plan is right" into
"the robot does it".

---

## Slide 11 — Driveability should be a constraint-like barrier · [F19, F20]

- F19 shows the current known 2D driveable-region landscape: green regions are
  the floor the robot may use; red staging pads/racks/walls are not a visibility
  tradeoff.
- The no-go layer has now been tightened into a **2-sigma driveable-region log
  barrier**: the predicted belief tube must remain inside the known driveable
  floor.
- F20 shows the first offline diagnostic after this change. The H80 A3-detour
  seed remains feasible and reaches near the goal, while invalid/cold basins
  safe-stop rather than cutting through forbidden floor.

**So what:** route choice is now constrained to physically valid floor; the
visibility-aware effect must happen through covariance inside that floor.

---

## Slide 12 — What does initial planning actually choose? · [F21]

- F21 shows the optimizer's converged initial rollouts for the current shelf-pick
  task, after the 2-sigma driveable-region barrier.
- C2 at H80 has a valid A3-detour basin that reaches near the goal, with the
  extra ambiguity term visible in the cost decomposition.
- C2 at H120 currently collapses to safe stop under the current risk/barrier
  schedule, so it is not yet a clean paper result.

**So what:** the desirable moving behavior exists, but the current objective
still needs tuning so the visibility-aware route is selected from neutral
optimizer initializations.

---

## Slide 13 — Fairer multistart candidates · [F22]

- F22 replaces diagnostic labels ("A3 seed", "A4 seed") with realistic,
  condition-neutral initial guesses: hold, local left/right escape, shortest
  driveable route, and alternate driveable route.
- Both C1 and C2 receive the same candidate set. The selected path is simply
  the lowest-cost valid optimized rollout under that condition's objective.
- Current result: C2 selects a floor-route initialization and converges to the
  lower visible route; C1 selects a local escape. This is more honest than F21,
  but still not yet the clean C1-direct vs C2-visible comparison.

**So what:** route choice is now being tested with a fair candidate set; the
remaining work is to tune/choose a task where the objective difference produces
the intended qualitative contrast.

---

### Backup / data provenance
- World: `warehouse_aws.world.sdf`; detector `aws_yolo_simseg_v2`; GP `aws_gp_v5`.
- Offline solve-time benchmark: `runs/offline_solve_scaling.csv`.
- Five Gazebo runs (H40/H80/H200 × multistart off/on): `runs/gazebo/`.
- All figures regenerated by `scripts/make_presentation_figures.py`.
