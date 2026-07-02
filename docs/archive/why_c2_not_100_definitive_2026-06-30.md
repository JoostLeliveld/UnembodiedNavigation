# Why C2 is not 100% — the definitive proof (2026-06-30)

**One sentence:** every collision is a *stale-belief-in-turns* failure — the camera
correction is ~1.2 s old (hardware-bound), a fast turn opens a belief-vs-truth gap, and
once that gap is large the innovation gate **rejects the very correction that would heal
it**, so the belief runs away and the robot steers into a rack. C2 is hit harder only
because its (correct) visibility-aware detours execute **more turning**, so it enters that
runaway regime ~8× more often. The planner/method is not choosing bad paths.

**Source:** the campaign's own per-timestep logs, all 40 runs of
`logs/visibility_comparison/robustness_keepin_clean_20260619/` (4 routes × {C1,C2} × 5
seeds), post-crash samples trimmed. Reproduce with
`scripts/visibility_comparison/diag/prove_c2_limit.py`.
Figure: `.../figures/why_c2_not_100.png` · stats: `.../why_c2_not_100_stats.json`.
This upgrades the anecdotal Leg-3 of `results_storyline_2026-06-19.md` (5 seeds, one route)
to a campaign-wide statistical chain, and adds the **gate-runaway** link, which is new.

---

## The chain (each link is measured across 40 runs)

### L1 — the staleness is a hardware floor, identical for both conditions
Median correction age used by the planner = **1.21 s (C1) / 1.20 s (C2)** — identical.
Per `docs/correction_age_timing_findings.md` the dominant ~0.16 s term lives in the
Gazebo-Fortress `ros_gz_bridge` and is **invariant to image resolution, camera rate, CPU
load and executor threads** → it cannot be tuned away. GPU offload (banked) takes the
end-to-end age 0.92 → 0.27–0.31 s but does not reach the 0.2 s needed for fast turns
without a bridge-bypass / Harmonic migration / new hardware.

### L2 — a fast turn during that 1.2 s window opens a belief-vs-truth gap
Pooled over all runs, belief error vs commanded turn rate:

| \|cmd_w\| (rad/s) | belief err median | belief err p95 |
|---|---|---|
| 0.0–0.2 (straight) | **0.049 m** | 0.23 m |
| 0.2–0.5 | 0.162 m | 1.40 m |
| 0.7–0.9 | 0.128 m | 1.12 m |
| 0.9–1.0 (hard turn) | 0.132 m | **1.84 m** |

Localization is **excellent at rest/straight (~0.05 m)** and degrades monotonically with
turn rate. At 1.2 s staleness a ~1 rad/s turn rotates the robot far between corrections, so
the planner steers on a lagged pose.

### L3 — once the belief diverges, the gate REJECTS the recovery (runaway) — *new*
Acceptance rate of the camera correction by belief-error regime:

| belief regime | C1 accept | C2 accept |
|---|---|---|
| good (<0.15 m) | **0.96** | **0.95** |
| diverged (>0.5 m) | 0.35 | **0.08** |

When tracking well, both conditions accept ~95 % of corrections — the estimator is sound.
But once a turn has opened a >0.5 m gap, the correction that would pull the belief back
implies a large jump from the (wrongly dead-reckoned) pose, so the innovation gate rejects
it as **`jump_too_large` (1199 samples) / `nis_too_large` (212)**. The belief then keeps
dead-reckoning and diverges further — a self-reinforcing runaway. The NIS self-heal
(commit `6a8be89`) does not catch it fast enough mid-turn.

### L4 — the runaway is what causes the collisions
All **18 collisions** (both conditions) are immediately preceded by a belief excursion:
run-median belief error **0.050 m → 0.67 m in the 3 s before the crash (median 13.8×, tail
to 3.2 m)**. The robot steers on a 0.5–3 m-wrong pose and clips the rack (penetration
~0.02 m). Collisions are a **perception/estimator failure in turns, not a planning
failure** — the chosen path is feasible; the executed pose tracking is what breaks.

---

## Why C2 specifically is not 100% (and why it is *not* a clean method verdict)

C1 and C2 are matched on every driver of the mechanism:

| | C1 | C2 |
|---|---|---|
| correction age (staleness) | 1.21 s | 1.20 s |
| turn exposure (frac \|w\|≥0.7) | 0.123 | 0.122 |
| obstacle clearance in turns | 0.49 m | 0.62 m *(C2 has **more**)* |
| longest sustained turn | 2.67 s | 2.43 s |
| per-turn belief-err median | 0.113 m | 0.152 m |

So C2 does **not** fail by turning *more often*, in *tighter* spots, or with a *worse*
per-turn estimator. What differs is **absolute exposure**: C2's correct visibility-aware
detours are longer and execute **more total turning (9.8 s vs 7.7 s)**, so it accumulates
**8× more diverged-in-turn samples (426 vs 51)** → enters the L3 runaway regime far more →
**11 vs 7 collisions**. C2 is punished for doing the right thing on a machine whose
perception cannot keep up in turns.

**Caveat (keeps us honest):** the 11-vs-7 split is *limiter-dominated*. The 2026-06-19
load experiment showed the **same** C2 run flips collision→goal when re-run near-idle
(camera 1.60→1.94 Hz, belief 0.094→0.076 m). So this campaign is not a clean C1-vs-C2 merit
comparison — it measures how the shared hardware limiter taxes C2's longer routes.

---

## What this says is fixable (two independent levers)

1. **Hardware / transport (the staleness floor, L1):** GPU render offload (banked),
   C++ gz-transport bridge-bypass, Fortress→Harmonic migration, or a workstation. Lowers
   the gap that L2 can open.
2. **Estimator / method (the runaway, L3) — machine-independent:**
   - **Latency-aware fusion:** fuse each measurement at its *capture* timestamp with odom
     forward-prediction to "now", so a 1.2 s-old fix is not applied as if current. Hook
     `latency_compensate_plan_handoff` exists but is `false`; the EKF fuses at arrival time.
   - **Turn/staleness-aware gating:** relax `jump_too_large` / NIS when correction age is
     high and \|cmd_w\| is large, so the gate stops rejecting the legitimate recovery during
     exactly the turns that need it.

Only after one of each can C2's advantage be evaluated on the merits.

---

## Artifacts
- `scripts/visibility_comparison/diag/prove_c2_limit.py` — regenerates everything.
- `figures/why_c2_not_100.png` — 4-panel proof (L1+L2 / L4 / L3 / why-C2).
- `why_c2_not_100_stats.json` — the numbers above.
- Supersedes nothing; complements `results_storyline_2026-06-19.md` (§4) and
  `correction_age_timing_findings.md` (the L1 floor).
