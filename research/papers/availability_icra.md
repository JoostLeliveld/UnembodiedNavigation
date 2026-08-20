# Availability-aware external-camera navigation — ICRA paper plan

## Purpose

The supported paper claim is now narrower: the probability that a fixed camera
network delivers a usable observation can be estimated from the cameras' own images,
without a surveyed 3-D model of the building. The attempted closed-loop extension
found that the frozen planner consumed the field but selected the same global route
under every observation-model condition.

Study code `experiments/availability_paper/`. Evidence `logs/studies/availability_paper/`.
Registry `EXP-AVAIL-SOURCE` (complete) and `EXP-AVAIL-CL` (blocked after the
incomplete campaign exposed a structural planner limitation).
Analysis plan frozen in `experiments/availability_paper/PREREGISTRATION.md`.

## Where the novelty is, and where it is not

**Not** in separating "did an observation arrive" from "how noisy was it". That is
standard in partially-observable formulations, and a reviewer will say so in one
line. Leading with it also walks into our own result: the explicit Bernoulli planner
and the folded-covariance shortcut chose identical routes on all four tasks, and the
registry's own falsifier for `C3` is that the two agree within 10 %.

**Yes** in the estimator: the chain from a fixed camera's RGB → monocular metric
depth → floor-anchored occlusion raycast → calibrated per-camera detection
probability → expected-free-energy planner, measured end to end against 8,808 real
detector outcomes with spatially blocked held-out ground. Nobody has that chain.

The explicit Bernoulli formulation stays, as one paragraph, for the one thing it
genuinely earns: its miss branch takes no update, so it needs no miss endpoint at
all, which dissolves the unreconciled 40 px versus 120 px `r_miss` constant that
`reliability.covariance_mapping.MissEndpointPolicy` still refuses to bless.

## The sentence the paper must earn

> Availability-aware planning reduces [closed-loop endpoint] by X % for Y % extra
> path over N matched runs, and an availability field inferred from the cameras' own
> RGB recovers most of the benefit of a surveyed 3-D model.

Second half: **supported** (§2 below). First half: **not earned**. The readiness gate
passed, but E4 was stopped after 12/45 plan-bearing runs because all persisted C1–C3
global plans were coordinate-identical. E5 shows the availability-sensitive term is
present but dominated by the frozen risk and obstacle terms.

## Section plan, with the numbers and figures bound

| § | Content | Figure | Key numbers |
|---|---|---|---|
| I | External cameras as an availability problem, not a noise problem | — | — |
| II | Related work: belief-space planning with sensing regions; occlusion-aware active perception; monocular metric depth | — | — |
| III | Model: `P_D,c(x)` and `R_cond,c(x)` separately; noisy-OR over cameras; hit/miss expected posterior | — | miss branch removes the `r_miss` endpoint |
| IV | Estimating `P_D` from the camera's own image: floor-anchored monocular depth → height map → raycast → link | — | inputs are RGB + calibration + 2-D drivable map |
| V | Held-out prediction | `01`, `02` | mono-depth Brier 0.068 vs CAD 0.062, ties at p = 0.15; distance-only 0.128; geometry-free GP 0.218 vs constant 0.239 |
| VI | Route consequence and where it applies | `04`, `05` | on `mc_blind_L` the 5.6 s unobserved stretch goes to **0.0 s for 16 cm** of detour, hit rate 64 % → 84 %; benefit largest for 2 opposing cameras (3.4 s) and smallest for 2 same-wall (1.5 s) |
| VII | Planner limitation | `06`, `11`, `12` | 12 persisted plans, one coordinate array; maximum pointwise difference 0.0 m |
| VIII | Limits | — | sim only, one world, one detector, one clear frame per camera |

`03` (availability ≠ accuracy) is compressed to a single panel plus two sentences in
§III. It argues for the representation, not for the navigation claim, and does not
earn a section at 6 pages.

## Claims, in the order a reviewer will test them

1. Availability is predictable from deployment-legal inputs at parity with a
   surveyed model. **Supported**, paired sign test over 24 camera-folds.
2. The geometry carries the prediction, not the learning. **Supported**: a GP with
   no geometric mean function reaches Brier 0.218 against a constant's 0.239, with
   identical calibration error, while still ranking (AUROC 0.738).
3. Distance alone is not enough. **Supported**: AUROC 0.765 but Brier 0.128, worse
   than the CAD raycast on 23 of 24 folds. Answers `RQ01`.
4. A weak availability model buys nothing. **Supported, and weaker than first
   claimed.** Distance-only and FOV/range leave the unobserved stretch at the
   availability-blind 2.40 s. An earlier version of this experiment reported them as
   actively *worse*; that was an artifact of boundary-hugging routes and is withdrawn.
   They are only worse on the negative-control task (2.80 s → 4.00 s).
5. The benefit depends on camera placement, not just camera count. **Supported.** Two
   cameras on opposite walls give the largest saving of any configuration (3.40 s);
   two on the same wall give the smallest (1.50 s) despite a worse blind baseline.
6. Prediction parity is not decision parity. **Supported, and it is a limitation.**
   Monocular depth ties CAD on held-out Brier yet, on `mc_blind_L`, spends an
   equal-length detour in the wrong place and stays exposed for 4.80 s against CAD's
   0.00 s.
7. It improves closed-loop navigation. **Not supported by this campaign.** The
   preregistered campaign is incomplete and no arm-level navigation comparison is
   admissible. The supported result is instead that the frozen runtime objective did
   not make route choice sensitive to the available spatial signal.

## Reviewer defences already in place

| Likely attack | Defence |
|---|---|
| "The GP result looks too good / too bad" | The cached GP fields are fitted on every event; scoring them held-out gave Brier 0.021. `gp_refit.py` refits per fold and fits the link on inner out-of-sample predictions. Both numbers are reported. |
| "Random k-fold on a dense grid is meaningless" | Leave-one-spatial-block-out over six contiguous blocks, verified to reproduce the frozen capture's groups. |
| "You gave your method a calibration the baselines didn't get" | Identical two-parameter link on every arm, training folds only. A calibrated arm recovers `a ≈ 1, b ≈ 0`. |
| "Terminal accuracy doesn't move, so who cares" | Stated outright: terminal sigma moves 0.6 mm and *hides* the effect. The endpoint is time unobserved, and it is preregistered as such for the closed loop. |
| "Why not distance only?" | Arm 3 above. |
| "One frame of monocular depth?" | All four monocular models run through the same held-out pipeline: Brier 0.0675-0.0742 across depth MAE 0.247-0.420 m. The weakest depth model stays within 0.012 Brier of the surveyed reference, so the result is not one lucky model. The pallet frame is reported as a separate dynamic regime. |
| "p = 0.15 is not equivalence" | The paired difference carries a bootstrap interval: +0.0060 with 95 % [-0.0017, +0.0146], so any surveyed-model advantage is bounded at 0.015 Brier. |
| "Are those routes even drivable?" | A clearance pre-flight against the controller's own keep-in contract caught routes at 0.062 m and 0.005 m clearance; route search now runs on the mask eroded by the 0.25 m contract. |
| "Effect is one task out of four" | Said in the paper, and it is why the negative-control task is preregistered into the campaign. |

## Two framing corrections this paper forces

**The CAD raycast is not an oracle.** `research/08_figures.md` F07 says "oracle
remains an upper bound", and `reliability_source_comparison.md` calls complete CAD
the "oracle/reference". Monocular depth ties it at p = 0.15 here, and beat it on the
earlier AUROC comparison. CAD is a baseline that happens to require a survey. Both
documents need the claim line revised.

**Monocular depth is no longer an exploratory challenger.**
`reliability_source_comparison.md` retains it as "an exploratory
zero-additional-hardware challenger". On this evidence it is the primary operational
depth arm.

## Remaining work

The 2026-08-18 successor audit completed the first two requested checks in
`experiments/factorized_observation_successor/`:

1. **Planner formulation:** DS-Route uses a fixed 5% path budget and directly
   minimizes exact expected longest missed-update duration.  The A+B development
   gate passed (28.6% median reduction); a previously untested B+C pairing also
   passed (12.6% median reduction), with median CAD-reference decision regret
   falling from 1.579 s to 0.012 s.
2. **Independent `R_cond`:** the range-conditioned pixel-covariance candidate
   collapsed exactly to the constant baseline on 1844 current detections and
   covered only 87.0% at nominal 95%.  The spatial-`R_cond` gate therefore failed.
3. **Closed loop:** deliberately not launched.  The combined decision is
   `STOP_FAIL_CLOSED` because the covariance gate failed.

What remains is (a) a geometry-diverse, genuinely held-out conditional-error
capture that can establish or reject spatial `R_cond` without reusing these rows,
(b) matched closed-loop evaluation only if that gate passes. The two-world governance
item is closed: that rule was retired 2026-08-20.

## Scope, stated once

Simulation only. One warehouse, one frozen four-camera YOLO detector, four identical
simulated cameras, one clear RGB frame per camera for the depth fields. No hardware,
detector, optical, vendor or lighting generality is claimed. "Without a surveyed 3-D
obstacle model" — not "map-free": camera calibration and the 2-D drivable map are
still required, the latter to anchor floor depth.

Unresolved: `EXP-AVAIL-SOURCE` is a source comparison run in `warehouse_full_4cam`,
which `06_world_camera_design` §2 reserves for frozen-method evaluation. Needs a
registered exception or a `warehouse_aws` re-run of the geometric arms.
