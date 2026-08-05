# operational_residual_rcond — conditional camera covariance without ground truth

**Question.** Can the per-camera conditional measurement covariance `R_cond,c`
be estimated from **operational signals only** — wheel odometry plus the camera
network itself — instead of from a ground-truth-referenced residual?

**Why it matters.** The repo's only residual is
`eval_res_x/y = pred_world − ground_truth`. That column is evaluation-only and
firewalled (`^eval_` in `src/reliability/config/leakage_firewall.yaml`), so it can
characterise the camera network but can never *train* a deployable covariance.
`R_cond` being unmeasured is the recorded open blocker on the ICRA-2027
observation-model claim: the hit/miss mixture currently falls back to a constant
`R_visible`.

**The move.** Reference the residual to a smoothed operational belief instead of
truth — `r_t = z_t − h(μ_t^s)` — and subtract the state contribution explicitly:
`C_t = H_t P_t^s H_tᵀ + R_cond(s_t)`.

**Chapters served.** `research_story/01_operational_belief_and_logging` (PARTIAL;
supplies the missing prior-belief/covariance-calibration evidence) and
`research_story/04_factorised_observation_model` (PLANNED, `implemented_now: []`;
supplies the conditional-accuracy half of the `p_use` × `R_cond` factorisation).

## Status — 2026-08-04

Executed on real recorded captures. Full write-up:
`logs/studies/operational_residual_rcond/exp2_operational_rcond/RESULTS.md`.

| gate | outcome |
|---|---|
| R0 timing + coverage | **PASS** — 1425/1426 detections associate; the recorded "join yields no in-window pairs / needs re-capture" blocker was **mis-attributed** |
| R1 belief calibration | **FAIL** — belief overconfident (median NEES 8.5–10.8 at detections vs 1.39); cause = uncorrected per-camera bias, not the smoother |
| R2 state-corrected estimator | **BUILT** |
| R3 circularity | **PASS and load-bearing** — the anchored estimate understates camera A by **4.2×** |
| R4 operational vs oracle | **PARTIAL** — within ~10% for A/C/D; recovers camera C's +0.078 m lateral bias without truth; camera B unresolvable |
| R5 deliver into frozen `R_good` | **WITHHELD** — per-camera `R_cond` loses to a constant pooled matrix on held-out MNLL |

**Headline:** `R_cond` is not blocked on data, it is blocked on **per-camera bias**. The
identified next step needs no new capture: fit and remove an operational per-camera 2-D
bias `b_c` (the residual *mean* this study already recovers without truth), then re-run.

## Plan

`PLAN.md` — the adapted specification. It is an adaptation of an incoming
greenfield spec (M0–M9); §2 records which of its milestones were **cut as already
built or already answered**, with the evidence, and §3–§4 specify what remains
and which of its gates were wrong for this repo.

Headline adaptations:

1. No new top-level tree — library into `src/state` / `src/reliability`, study
   here, outputs to `logs/studies/operational_residual_rcond/`.
2. The spec's M0/M1/M4/M6 are already built or already answered; M4 in
   particular has a **documented null** (GP loses to a two-parameter FOV/range
   logistic on held-out routes) that the spec's weaker gate would have reversed.
3. Two additions the spec lacks: **leave-one-camera-out** estimation (its
   residual is otherwise anchored by the camera it is measuring) and an
   **operational-vs-oracle** comparison.
4. The spec's Gate M2 is restated on *covariance calibration* rather than point
   error, because exp5 already measured that smoothing does not improve the mean.

## What is NOT ours

- The along-bearing projection correction is pre-existing and **deployed**
  (`logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v2/`).
  Applied exactly as the runtime applies it; never refitted here.
- The trust→covariance mapping is frozen (`reliability.covariance_mapping`). This
  study supplies its `R_good` endpoint; it does not add a mapping form.
- The KF+RTS smoother behaviour originates in
  `experiments/optionA_commissioning/exp5_trajectory_smoothing.py`; this study
  promotes it to a library, it does not invent it.
- `p_use` is already selected (B2 FOV/range, per Gate 4). Not revisited.

## Data

Real recorded 4-camera captures, operational streams only; ground truth is read
from `evaluation_only/` for scoring after inference and never joined before it.

| capture | odom | per-camera detections |
|---|---|---|
| `logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716/raw/` | 6750 @ 50 Hz | B 69 · C 320 · D 348 |
| `.../gt_validation_smoke2_20260716/raw/` | 3000 @ 50 Hz | A 99 · B 211 · C 65 · D 84 |
| `logs/studies/multicamera_fusion_extension/fusion_handover_real_20260721/data/raw/` | 5296 | A 27 · B 15 · C 90 · D 98 |

~1200 usable detections total — thin for a spatial field, adequate for per-camera
constants and a range-dependent term. Coverage is published before any fit, per
Gate R0.

## Reproduce

```bash
python3 -m pytest tests/state/test_trajectory_smoother.py tests/reliability/test_operational_residual.py
python3 experiments/operational_residual_rcond/timing_audit.py        # Gate R0
python3 experiments/operational_residual_rcond/estimate_rcond.py      # R1-R4
```

Outputs land in `logs/studies/operational_residual_rcond/`.

Pure offline analysis — no Gazebo, no ROS launch, no new capture required
(see `PLAN.md` §2: the recorded "0 events / needs re-capture" blocker was
mis-attributed; the detection↔odometry join is sound to one odometry tick).

## Reuse map

See `PLAN.md` §7.
