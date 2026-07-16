# 06 — Closing the loop: does the operational map still win?

**Question.** Chapter 00 proved a *survey-fitted* trust map improves navigation. Does a map
learned the realistic way — from ordinary driving at uncertain positions (ch.03), through
the frozen interface (ch.05) — still deliver that downstream win?

**Status: PLANNED** (the anchor is locked; the N-campaign runs after ch.02/03/05 freeze).
World: original warehouse. This is Contribution 1's closing evidence.

## What the contribution-grade result looks like

**The bar:** N4 must beat the *realistic operational* baselines N1 and N3. It does NOT need
to match N2 — the survey map is an optimistic reference whose data-collection assumption the
thesis rejects. Beating N1/N3 while approaching N2 is the ideal outcome; N4 ≈ N3 would say
input-uncertainty modelling doesn't matter downstream (report it — ch.03's map-quality claim
can still stand on calibration alone).

| ID | Meaning | Source |
|---|---|---|
| N0 | No external camera | new |
| N1 | Constant covariance | ≈ locked C1 config |
| N2 | Survey-fitted GP (reference, optimistic) | ≈ locked C2 |
| N3 | Point-input GP on operational driving data (U1) | new |
| N4 | **Uncertain-input GP on the same data (U5)** | new |
| N5 | Factorised model | only if ch.04 promoted |

## The results we're aiming for

- **Table 06C (headline)** — clean-goal rate, GT geometry breaches, contacts, fallbacks,
  travel time, path length, planning latency per condition. **Aim: N4 ≥ N1/N3 on goals AND
  breaches, with sane time/length cost.**
- **Fig 06A** — one route's paired mechanism: map+path, trust along path, R_plan along path,
  belief covariance, hit/miss timeline. The "how it works" figure.
- **Fig 06B** — all seeds/conditions trajectories, failures marked.
- **Fig 06D** — estimation calibration: GT-eval RMSE, NIS, coverage, rejection rate. Aim:
  N4's filter is *better calibrated*, not just luckier.
- **V06** — three-way video (N1 / N3 / N4) with live detector, ellipse, R_plan, route, trust.
  The main defense video.

## Routes

The four locked routes + one **designed uncertainty route**: long odometry-only segment →
camera reacquisition → a genuine choice between short-camera-poor and longer-camera-good
corridors; plus the existing control route where all conditions should tie (sanity anchor).

## Implemented now

| Item | Tag | Note |
|---|---|---|
| Campaign harness: 4 routes × 5 seeds, launch + config + outcome audit | established | `honest_campaign_v1` machinery, reused as-is |
| exp7 planner replay | measured_in_sim | cheap offline pre-check of N3/N4 route behaviour before Gazebo |
| exp6 stress test (inflation curves, stale-map) | measured_in_sim | informs failure-mode expectations |
| Replay compositing + camera recording (`generate_run_replay.py`, `record_camera_stream.py`) | established | V06 tooling |

## Gap → next experiment

Pre-register the N-campaign (conditions, routes incl. the new uncertainty route, seeds,
outcome definitions, contract) exactly as honest_campaign_v1 was; dry-run N3/N4 in exp7
replay first; then run.

## Gate

Runs only after: ch.01 gate passed, ch.02 target frozen, ch.03 U-grid done, ch.05 interface
frozen. Metrics via `campaign_metrics.load_run` + `scripts/shared/metrics.py` only. Keep
this campaign's numbers separate from ch.00's.
