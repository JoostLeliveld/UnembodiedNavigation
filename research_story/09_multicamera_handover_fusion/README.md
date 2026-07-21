# 09 — Selection, handover, fusion — without overconfidence (candidate contribution B)

**Question.** Given per-camera trust fields (ch.08), how should the robot *choose* among
cameras, *hand over* between them, and possibly *fuse* them — without ever producing an
overconfident belief in overlap regions?

**Status: PLUMBING + historical pilot.** World: `warehouse_full_4cam` — the only large
world in the repo. The pilot from the retired auxiliary testbed is development history that
defines *what to measure*, and every measurement must be re-collected on
`warehouse_full_4cam`.

## What the contribution looks like

> *Reliability-aware selection and conservative fusion across a network of fixed warehouse
> cameras: camera-specific trust maps support stable source selection and prevent
> overconfident fusion in overlap regions.*

Earned only when ALL exist (evidence-ledger rule): a detector evidence chain per camera;
**calibrated** overlap disagreement (a D2 pass); replay baselines; and a closed-loop handover
campaign. Today none of the four are complete — and we know precisely where we stand,
because the historical pilot *failed its gate honestly* (below).

## Conditions

Plan IDs M0–M7 map onto the implemented benchmark: S0–S4 (selection/fusion baselines:
best-fixed, nearest, highest-confidence, highest-trust, naive/conservative fusion) plus
**M8 hysteretic handover selection** — the stateful operational manager (trust, freshness,
association, overlap-consistency gates; multi-frame hysteresis; explicit failure fallback).
M7-style oracle selection stays evaluation-only.

## The results we're aiming for

- **Fig 09B** — handover timeline: selected camera, per-camera trust, availability, handover
  events, belief covariance. **Aim: few, well-timed switches — M8's pilot already cut
  switching 3→1; hold that while not paying accuracy.**
- **Fig 09C** — overlap disagreement ‖z_i − z_j‖ heatmap on `warehouse_full_4cam`'s
  adjacent-camera corridors. **Aim: a D2 PASS — ≥30 synchronized overlap pairs, ≤10%
  spatial outliers** (the historical pilot managed 5 pairs with 20% outliers, 4/5 within
  0.30 m — underpowered, not disproven).
- **Fig 09D (decision figure)** — fusion calibration: RMSE, innovation NLL, NIS coverage,
  covariance-overconfidence rate for naive vs calibrated vs conservative fusion. **Aim:
  conservative fusion never overconfident; note calibrated fusion may show up in trajectory
  variance/smoothness (Fig 09E) more than absolute accuracy.**
- **Fig 09A** selection-region map · **Fig 09F** camera-dropout robustness ·
  **V09A/B/C** handover / naive-vs-conservative / camera-failure videos.

## Implemented now

| Item | Tag | Note |
|---|---|---|
| M8 hysteretic CameraManager (operational-only fields) | established (code) | no evaluation-field access by construction |
| S0–S4 baselines + offline overlap validation (D2 machinery) | model_plumbing | |
| Handover covariance inflation + estimator NIS gating | established (code) | contract tests in `tests/reliability/` |
| Synchronized static pilot (`pilot_synchronized/`) — **HISTORICAL: ran on a retired auxiliary testbed** | historical (measured_in_sim) | **D2 gate FAILED, deliberately reported**: 5/5 pairs exactly synchronized, 4/5 within 0.30 m, but 5 « 30 required and 20% > 10% outlier limit. M8: switching 3→1, map error 0.130→0.136 m — a stability trade-off, not an accuracy win. SHA-256 manifest pins world/detector/config/figures. Re-collect on full_4cam |
| Camera contracts + operational/evaluation export split | established (infra) | the leakage firewall every later claim depends on |

## Gap → next experiments

1. Once ch.08's routes exist: collect synchronized overlap passes in full_4cam's
   adjacent-camera corridors until D2 is powered (≥30 pairs) — the single blocking item for
   any overlap-calibration statement.
2. Detector evidence chains for cameras B/C/D; per-camera trust fits (ch.08).
3. Replay benchmark S0–S4+M8 on commissioned maps → Figs 09A–09D; only then design the
   closed-loop handover campaign.

## Gate

No fusion claims before: per-camera detector chains + D2 pass + replay baselines +
closed-loop campaign. Active planner handover stays disabled until then (current state).
