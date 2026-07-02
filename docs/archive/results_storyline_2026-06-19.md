# Corrected results & the current limiter — 2026-06-19

**Campaign:** `logs/visibility_comparison/robustness_keepin_clean_20260619/`
**Config:** `scripts/visibility_comparison/warehouse_visibility_campaign.yaml` (the locked "best settings")
**Matrix:** 4 routes × {C1 constant-cov, C2 visibility-aware} × 5 seeds = **40 runs, 0 infra-invalid.**

---

## 1. TL;DR (read this first)

Running the **best current settings** with the **correctly-activated mechanism**, the
visibility-aware planner (C2) **does not beat the baseline (C1)** — it is currently *worse*:

| | goal | collision | stuck |
|---|---|---|---|
| **C1** constant-cov baseline | **12 / 20** | 7 | 1 |
| **C2** visibility-aware (full method) | **6 / 20** | 11 | 3 |

This is the **opposite** of the submitted paper (C2 16/20, 2 collisions vs C1 12/20, 8 collisions).
Two independent reasons, both now proven:

1. **The paper's advantage was an artifact.** The paper numbers were produced with the no-go
   cost in `keep_out` mode (a runner bug — see §3.1), not the `keep_in` belief-tube the paper
   text describes. Under the correct `keep_in` mechanism the contrast collapses/inverts.
2. **A runtime limiter is currently sabotaging C2.** Localization is accurate in steady state
   (~0.05–0.10 m), but the simulator camera delivers only **~1.6 Hz** on this laptop, so the
   belief correction the planner uses is **~1.3 s stale**. In the hard turns that C2's longer
   visibility-aware routes require, the belief diverges **0.3–1.2 m from truth right before the
   robot clips a rack**. C1's shorter/straighter routes are far less exposed.

**Most important caveat: the campaign is not a clean method comparison — it is limiter-dominated.**
The *same* C2 run that collides under full 40-run CPU load **reaches the goal when re-run nearly
idle** (camera 1.60→1.94 Hz, belief error 0.094→0.076 m flips collision→goal; §4 Leg 4). So
"C2 6/20" is not a verdict that C2 is worse on the merits — it largely reflects the load-dependent
camera-staleness limiter, which hits C2's tighter routes hardest. **The method's advantage can only
be evaluated once the camera is fast enough (GPU render offload / workstation / latency-aware
fusion).**

So the headline number did **not** improve — but the *methodology and infrastructure* did, and we
now have a clean, provable account of **why the method cannot show its advantage yet**: it is the
**Gazebo camera throughput on this machine**, not the planner.

---

## 2. The new results (per route)

`[goal / collision / stuck]` out of 5 seeds each:

| Route | C1 (baseline) | C2 (visibility-aware) |
|---|---|---|
| `control_west_to_a1_low` (easy, all-visible control) | **5 / 0 / 0** | 3 / 1 / 1 |
| `route_apron_to_a3_mid` | 4 / 1 / 0 | 2 / 2 / 1 |
| `route_apron_to_a2_mid` | 2 / 3 / 0 | 1 / 3 / 1 |
| `route_west_to_a1_upper` (hardest) | 1 / 3 / 1 | **0 / 5 / 0** |

Key reads:
- C2 is worse or equal on **every** route — including the easy visible control task where it
  should match C1. That a *visible* task degrades C2 is the tell that this is a **runtime**
  problem, not a route-discrimination problem.
- When C2 *does* succeed it behaves exactly as designed: it routes around shadow
  (`f_shadow = 0.00`, detection rate `1.00`) on **longer paths** (e.g. 7.3 m vs C1's 5.5 m).
  The mechanism is sound; the execution is what fails.

Plot: `logs/visibility_comparison/robustness_keepin_clean_20260619/figures/robustness_spread_keepin_clean.png`
Metrics: `.../paper_metrics.csv`, `.../paper_summary.txt`

---

## 3. Why these results differ from the paper (explicit, commit-grounded)

Paper finalized at commit `ba713ee` (2026-06-13). Everything below landed after.

### 3.1 The mechanism the paper describes was never actually running (the big one)
The paper config already said `nogo_mode: keep_in`, **but the campaign runner never passed
`nogo_mode` to Gazebo**, so the runtime fell back to its default `keep_out`. In `keep_out` the
no-go cost/validity uses the **rack (occluder) geometry**, not the **driveable lanes**, so the
"belief-tube keep-in on driveable lanes" the paper claims **was inactive in every paper run**.
With `keep_out`, C2 was allowed to **corner-cut across the open floor below the racks** — which
happened to be the *safe* shortcut — and that is what produced its low collision count.

Fix: `nogo_mode` is now passed in the per-task launch args (`run_visibility_campaign.py:341,500`);
manifests confirm `nogo_mode:=keep_in`. Under the correct mechanism, C2 is forced onto the
in-lane routes, which thread tight rack-adjacent transitions — and (combined with §4) it collides.
→ **The paper's collision contrast was a `keep_out` corner-cutting artifact.**

### 3.2 Clean detector retrain
Paper used `aws_yolo_simseg_v2` (contaminated dataset, imgsz 640, conf 0.10). Now
`warehouse_yolo_detector_v1` (occlusion-gated clean dataset, imgsz 960, conf 0.05, Box mAP50
0.995). Commits `72d8404`, `ab0c932`, `3610367`. Fixes the box-bottom periphery localization
error. → localization is now ~0.05–0.10 m (see §4).

### 3.3 Localization + estimator fixes
- Affine BEV calibration replacing a constant y-offset that never reached the projection node
  (median BEV err 0.099 → 0.058 m); commit `e28e9d6`.
- NIS innovation gate **+ self-heal** so the gate stops rejecting good camera updates after drift
  (commit `6a8be89`; C2 published-belief err 0.28 → 0.05 m).
- Heading via `camera_xy_only`.

### 3.4 GP retrain
`aws_gp_v7b` → `warehouse_visibility_gp_v1` (detection-rate methodology).

### 3.5 Runtime fixes (now built into `install/`)
Camera 30 → 5 Hz, detector inference moved out of the GIL-contended daemon worker into the
callback (single-thread), `/clock` throttled to 50 Hz, segmentation camera disabled. Paper-era
runtime was 30 Hz + a GIL-bound worker.

### 3.6 Task redesign
Paper tasks `b2/b5/F31_b1/b6` → 4 named routes. The discriminating regime was rebuilt; this is
the first full campaign on it under the correct mechanism.

---

## 4. What is currently limiting it — the clearest possible proof

The limiter is the **Gazebo synthetic-camera render + ros_gz bridge throughput on this laptop**,
not the method. Three independent, mutually-reinforcing pieces of evidence:

### Leg 1 — The strong GPU is not rendering the camera
- `glxinfo -B` → default OpenGL renderer is **Mesa Intel UHD Graphics 630** (integrated), *not*
  the NVIDIA Quadro P2000.
- During a run, `nvidia-smi` shows **only Xorg + the YOLO python process (CUDA, ~1.6 GB) on the
  P2000 — `ign gazebo` is absent from the GPU.**
- `libEGL warning: failed to create dri2 screen` (×2) at every launch → EGL cannot get a hardware
  screen, so the camera sensor renders on the iGPU/software path.
- ⇒ The discrete GPU sits idle while the camera renders on a weak path.

### Leg 2 — The detector is *not* the bottleneck; the camera rate is
From the campaign's own `perception.csv` (40 runs, `analyze_runtime_limiter.py`):

| metric | median |
|---|---|
| camera arrival rate | **1.6 Hz** (configured 5 Hz) |
| YOLO inference | 122 ms → ~8 Hz capable |
| detector callback | 141 ms |
| frame age at publish | 0.1 s |
| **belief-correction age used by planner** | **1.3 s** |

The detector can do ~8 Hz; the camera only delivers ~1.6 Hz, so the bottleneck is **upstream of
the detector** (render + bridge), and the correction the planner applies is **~1.3 s old**.

### Leg 3 — The staleness causes the collisions (collision-level proof)
On the C2 collisions of the hardest route (`route_west_to_a1_upper`, all 5 seeds collide):

| seed | run-median belief error | **belief error in the 3 s before the crash** |
|---|---|---|
| 0 | 0.054 m | **0.63 m** (max 0.74) |
| 1 | 0.058 m | **0.47 m** (max 0.62) |
| 2 | 0.053 m | **0.35 m** (max 0.64) |
| 3 | 0.053 m | **1.07 m** (max 1.22) |
| 4 | 0.054 m | **0.32 m** (max 0.71) |

Steady-state localization is excellent (~0.05 m); the belief error **explodes 10–20× to
0.3–1.2 m in the seconds before each crash**, and those crashes occur **mid-turn**
(`|cmd_w| ≈ 1.0 rad/s`, near max). At 1.6 Hz / 1.3 s staleness, a ~1 rad/s turn rotates the robot
far between corrections, so the planner steers on a pose lagging ~0.7 m behind truth → it clips
the rack (penetration ~0.02 m).

**Why C2 and not C1:** C1's belief is equally accurate (~0.07 m) but its routes are
shorter/straighter with fewer hard rack-adjacent turns, so it is far less exposed to the staleness.
C2's *correct* visibility-aware behaviour (longer detours around shadow) puts it through exactly
the turns the stale belief cannot handle. **The method's strength is, on this machine, what gets
it killed.**

### Leg 4 — The outcome flips with CPU load (natural experiment, clincher)
Same route/seed/config (`route_west_to_a1_upper` C2 seed0):

| condition | camera Hz | correction age | belief err | **outcome** |
|---|---|---|---|---|
| campaign (full 40-run load) | 1.60 | 1.27 s | 0.094 m | **collision** |
| re-run (near-idle) | 1.94 | 1.14 s | 0.076 m | **goal reached** |

A small improvement in camera freshness flips the outcome. This is direct evidence that the
collisions are a **load-dependent staleness** effect, not a fixed route/method failure — and it is
why the campaign C1-vs-C2 comparison cannot be trusted as a method verdict on this machine.
(Attempting to *force* a collision by adding 8 CPU burners failed for the opposite reason: it
starved the single-threaded global solve so hard the robot never started moving — i.e. heavy load
breaks the planner before it breaks the camera. The campaign's milder, realistic contention is the
regime that produces driving-then-colliding.)

See the side-by-side videos (§6): the belief marker (orange) lags the truth marker (black) through
the hard turns; the bottom error panel spikes in exactly those turns and recovers in the straights.

---

## 5. What would fix it (so C2 can be evaluated fairly)

The limiter is machine/config-dependent. Options, cheapest first:
1. **Make Gazebo render the camera on the P2000** (PRIME/EGL device offload). The discrete GPU is
   idle; this is the most direct fix and needs no method change.
2. **A workstation** with verified GPU camera rendering (the case in
   `docs/gazebo_compute_request_presentation.md`).
3. **Latency-aware belief fusion** (machine-independent): fuse each camera measurement at its
   *capture* timestamp + odom forward-prediction to "now", so a 1.3 s-old fix is not applied as if
   current. Hook exists (`latency_compensate_plan_handoff`, currently false) but the EKF fuses at
   arrival time today.

Only after one of these can we say whether C2 beats C1 on the merits.

---

## 6. Artifacts
- Campaign: `logs/visibility_comparison/robustness_keepin_clean_20260619/`
- Metrics: `paper_metrics.csv`, `paper_summary.txt`
- Plot: `figures/robustness_spread_keepin_clean.png`
- Runtime-limiter table: `analyze_runtime_limiter.py` → `/tmp/runtime_limiter_per_run.csv`
- Side-by-side videos (moving plot | camera) under `_video_runs/`:
  - `videoA_west_C2_success.mp4` — C2 on the hard `route_west_to_a1_upper`, run near-idle: belief
    tracks well enough through the turns → reaches goal (the same route that collides 5/5 under
    campaign load). Watch the error panel spike in the turns.
  - `videoB_control_C2_success.mp4` — C2 on the easy all-visible control route → clean success,
    belief glued to truth.
  - Frames captured via `yolo_debug_frame_dir`; composited by `diag/diag_side_by_side.py`.
  - NOTE: a load-induced *collision* video was not capturable cleanly (heavy artificial load
    starves the solve so the robot never drives); the collision evidence is the campaign data
    (Leg 3) + the load natural experiment (Leg 4).
- Archived diagnostics: `logs/visibility_comparison/_archive_diagnostic_20260619/` (+ README),
  superseded campaigns under `archive/superseded_campaigns_20260619/`.
