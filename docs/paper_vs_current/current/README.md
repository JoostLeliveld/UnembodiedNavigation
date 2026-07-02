# Current snapshot (the "after")

Regenerated artifacts from the honest re-run pipeline, to compare against
`../paper/`. Runtime: retrained detector `warehouse_yolo_detector_v1` (imgsz 960),
refit GP `warehouse_visibility_gp_v1` (detection-rate), `keep_in` actually
honoured, χ² NIS gate 9.21 (self-heal off), `global_horizon 75 × dt 0.4` (30 s),
ground-truth metrics + functional physics-contact channel.

```
current/
├── warehouse_visibility_campaign.yaml   the current campaign config
├── figures/
│   ├── paired_mechanism_taskA_current.pdf          regenerated task-A paired mechanism
│   └── paired_mechanism_taskA_current.provenance.json
└── data/
    └── paired_mechanism_taskA/   seed-0 C1 & C2 run data (from the fresh run)
```

## Task-A paired mechanism — paper vs current (seed 0)

Source run: `logs/visibility_comparison/paired_mechanism_current_taskA`
(config `_paired_current_taskA.yaml`, task A only, C1+C2, seed 0). Campaign
result: **2/2 goal_reached, 0 collisions**.

| metric | paper C1 | current C1 | paper C2 | current C2 |
|---|---|---|---|---|
| outcome | goal_reached | goal_reached | goal_reached | goal_reached |
| collision_contact | none | none | none | none |
| collision_geom | none | none | none | none |
| final_goal_distance (m) | 0.124 | 0.155 | 0.143 | 0.090 |
| peak belief error (m) | **0.57** | **0.23** | ~0.29 | ~0.20 |
| mean belief error (m) | 0.172 | 0.108 | 0.109 | 0.130 |

**Read:** Task A (`route_apron_to_a3_mid`) was already a *success* in the paper —
both C1 and C2 reach goal with no collision; it is the route where the
visibility mechanism is *demonstrated*, not where the paper failed. What
improved:

- **C1 (short/blind route):** the paper's runaway belief spike to **0.57 m** at
  t≈7.5 s is gone. The current C1 peaks at ~0.23 m and a camera correction snaps
  it back to ~0.07 m at t≈10 s — the retrained detector + χ² NIS gate now recover
  mid-run instead of letting the belief run away.
- **C2 (long/visible route):** stays tight (~0.13 m) as before; route choice
  preserved (C1 short through the shadow, C2 the visible detour).
- **ρ_plan map:** now near-binary (visible aisles ≈0.9, shadow ≈0) — the GP refit
  (confidence → detection-rate) sharpens the "stronger camera support" contrast
  vs. the paper's smooth gradient.

### Caveat (metric parity)
The error panel (c) in both figures uses `truth_belief_error_m`, i.e.
**odom-as-truth** — the same metric the paper used. This keeps the paper-vs-
current comparison apples-to-apples, but it is NOT the honest GT metric. A fully
honest version would plot `belief_error_gt_m` (ground truth); the plotter
(`scripts/paper_figures/make_paired_mechanism.py`) reads the odom column today.
Worth switching before these go in the thesis.

Task A is the *easy* case; the improvements matter more on the harder tasks
(a2_mid, west, control) where the paper showed the failures — those are now
covered by the **completed 40-run honest campaign**
(`logs/visibility_comparison/honest_campaign_v1`, 4 routes × 2 conditions ×
5 seeds, 2026-07-01). Per-route representative figures (seed 0, except west
seed 1 to show the modal C1 collision) are regenerated in `figures/` as
`paired_mechanism_{taskA,a2mid,west,control}_lowlat.pdf` straight from that
campaign.

### 40-run honest campaign headline (`honest_campaign_v1`, imgsz-640 / low-latency arm)

| route | C1 goal | C1 collisions (geom / phys) | C2 goal | C2 collisions |
|---|---|---|---|---|
| route_apron_to_a3_mid (taskA) | 4/5 (1 interrupted) | 0 / 0 | 5/5 | 0 |
| route_apron_to_a2_mid (a2mid) | 5/5 | 0 / 0 | 5/5 | 0 |
| route_west_to_a1_upper (west) | **1/5** | **4 / 0** | 5/5 | 0 |
| control_west_to_a1_low (control) | 5/5 | 0 / 0 | 5/5 | 0 |
| **total** | **15/20** | **4 / 0** | **20/20** | **0 / 0** |

**Read:** C2 (visibility-aware) reaches goal on **20/20** runs with **zero**
collisions on either channel. C1 (visibility-blind) fails only where the short
route runs through the ρ≈0 blind west lane — 4/5 west runs end in a geometric
wall-graze (all `collision_contact` False: the ~0.105 m robot mesh stayed off
the wall, so these are near-wall safety breaches of the 0.125 m envelope, not
hard contacts). The lone non-west C1 miss is one taskA seed that was
`interrupted`, not a collision.

## Hardest failing task — `route_west_to_a1_upper` (paper "b2"), seed 0

This is where the paper failed hardest: in the paper robustness campaign,
**C1 got 0/5 goal with 5 collisions**, while **C2 got 4/5 with 0 collisions** —
the cleanest "C1 blindly fails, C2 survives" case. Regenerated with the current
pipeline (`figures/paired_mechanism_west_current.pdf`, source
`logs/visibility_comparison/paired_mechanism_current_west`):

| | current C1 | current C2 |
|---|---|---|
| outcome | **collision (fails)** | **goal_reached** |
| collision_geom | True (`geometry:wall_penetration`) | none |
| collision_contact (physics) | False | none |
| min_wall_distance (m) | **−0.009** (grazes west wall) | ok |
| min_obstacle_distance (m) | 0.44 | 0.48 |
| final_goal_distance (m) | 1.78 (fails short) | 0.17 |
| mean belief error (m) | 0.138 | 0.112 |

**The mechanism, honestly reproduced:**
- **C1 (constant-R, visibility-blind)** routes up the narrow west service lane,
  which the detection-rate GP marks as **ρ≈0 (no camera reliability)**. With no
  corrections there, the belief **2σ uncertainty explodes past 0.5 m by t≈6 s**;
  the true pose grazes the west wall (−9 mm into the 0.125 m safety radius) and
  the run ends at t≈13.4 s, 1.78 m short of goal.
- **C2 (visibility-aware)** takes the long detour through the **ρ≈0.9** bright
  regions (down to the lower aisle, across, up the visible side), keeps camera
  support, and reaches goal with 0.48 m clearance.

**Honesty nuance (dual-channel cross-check):** C1's failure is a `collision_geom`
(GT position breaches the 0.125 m safety envelope of the wall by ~9 mm), but
`collision_contact` (real physics) is False — the actual robot mesh (~0.105 m)
stayed ~2 cm off the wall. The two channels agree it is a near-wall safety
failure and disagree only on whether it was a *hard* contact. Either way C1 does
not reach goal; C2 does.

### On the residual C2 belief-error spikes (both tasks)
The C2 error panels still show occasional spikes to ~0.2–0.33 m. Using the GT
columns now logged, these are **real belief lag, not odom artifacts**
(`belief_error_gt_m` ≈ `truth_belief_error_m`, `odom_truth_drift_gt_m` ≈ 0.003 m
at the peak). They occur **in turns** (|cmd_w| elevated) while camera corrections
lag **~0.42 s** (the delivered low-latency imgsz-640 arm — these figures run
`yolo_imgsz: 640`; the ~1.2 s figure in the archived analysis below is the
pre-low-latency imgsz-960 baseline) and are **accepted** (not gate-rejected): the belief coasts on
odom prediction through the turn and lags the true pose until the next detection
lands. This is the residual **correction-latency-in-turns** floor (see
`docs/archive/why_c2_not_100_definitive_2026-06-30.md`); the detector retrain cut the
peak (0.57→0.33 m on task A) but cannot remove the pipeline staleness limit. A
lower-latency camera/detector is the lever, which vindicates the compute-deck
argument.
