# Prospective confirmation on a second warehouse reconfiguration

Frozen 2026-08-19 after seeing all L1 results and before requesting the public
randomness used to choose L2. No L2 world, image, detector outcome, depth field, GP
prediction, or route result existed at freeze time. L1 is the discovery study; L2 is
the prospective confirmation and will not be replaced if its result is inconvenient.

This document resolves five avoidable researcher degrees of freedom before L2:
which layout is tested, which samples constitute the comparison, which prediction
contrast is primary, which route cell is primary, and how multiplicity is controlled.

## 1. Externally randomised layout

L2 is generated independently from nominal L0, not cumulatively from L1. It applies
the same physical intervention as L1: one 0.40 m layer of stock on 12 structural
rack segments, entirely above z=2.09 m, so the driveable floor and every lane remain
unchanged.

The eligible set is frozen before randomness is requested:

- split the 27 structural segments into six strata: west/east crossed with
  south/middle/north;
- choose exactly two segments from every stratum, hence 12 total;
- order segment names lexicographically within each stratum; order strata
  `(W,south), (E,south), (W,mid), (E,mid), (W,north), (E,north)`;
- enumerate the Cartesian product of the six within-stratum two-combinations in
  that order. This gives exactly 216,000 eligible layouts;
- after this file is frozen, obtain one NIST Randomness Beacon 2.0 pulse and preserve
  the official response plus its `https://beacon.nist.gov/` URL. Canonicalise its
  hexadecimal `outputValue` by stripping surrounding whitespace and lowercasing;
- compute
  `SHA256("reconfiguration-holdout-L2-layout-v1" + NUL + outputValue)`, interpret the
  digest as a big-endian integer, and choose its value modulo 216,000 as the
  zero-based index into the frozen enumeration.

The pulse is requested only after the SHA-256 of this document is recorded. A local
seed may exercise the implementation in a temporary output, but the canonical
`L2_layout.json` refuses any source other than a persisted NIST beacon record.

The complete factored enumeration is frozen in
`layouts/L2_eligibility.json` (SHA-256
`2aeb3647dbeac8289e5f9ca0e2784c5ef8cca1936ff831a32fb04f89505368d3`); it commits to
the candidate coordinates, every within-stratum combination, the full enumeration
digest, and the index rule without selecting a layout. The generator
`choose_second_layout.py` has SHA-256
`9c644815c235c4ddcfbe98d7c861dfb12eec6edf0cb5ac60c4f0d390b12bb69b` at this freeze.

Detector outcomes, captured images, learned fields, route outcomes, and
visibility-impact scores are forbidden selection inputs. The geometry oracle is run
only after the public draw fixes the index. Its fused-cell and camera-cell losses are
reported as realised intervention strength; there is no minimum-strength rejection,
rerandomisation, seed retry, or layout substitution.

## 2. Apparatus and exact sample membership

L2 uses the same camera poses, intrinsics, nominal lighting, robot, declared lanes,
capture grid, detector weights, and detector settings as L0 and L1. The only world
edit is the externally selected rack-top stock. The capture is:

- 942 L0 grid positions;
- headings exactly `{0, pi/2, pi, 3pi/2}`;
- cameras exactly `external_camera`, `external_camera_b`, `external_camera_c`, and
  `external_camera_d`;
- 942 x 4 x 4 = **15,072 unique camera-position-heading observations**;
- YOLO `warehouse_yolo_detector_4cam_v3_960/model.pt`, image size 640, confidence
  0.01 at extraction, IoU 0.45, class `robot`, no masks;
- primary labels rethreshold `yolo_raw_best_score` at the already frozen 0.25.

Before capture, a deterministic contract hashes this preregistration, the eligibility
and selected-layout artifacts, the base and generated worlds, the generator and
capture scripts, the study world profile, detector weights, L0 capture manifests,
and the exact canonical L0 comparison membership. The expected membership key is
`(camera_frame, x rounded to 8 decimals, y rounded to 8 decimals, theta rounded to
8 decimals)` after selecting the four registered headings.

The L2 capture fails closed before detector inference unless its manifest declares
the registered world, grid, four headings, four cameras, zero stale views, and
15,072 samples, and unless `samples.csv` contains the expected key set exactly once
with every referenced image present and nonempty. It fails closed again after
detector inference unless `perception_targets.csv` has the identical key set. Missing
or extra rows are not silently intersected with L0, reconstructed, imputed, or
dropped. An incomplete capture is repeated from the beginning using the same L2
layout and contract.

## 3. Frozen estimators and leakage-free transfer scoring

All learned parameters use L0 only. L2 outcomes never fit a field, link, threshold,
hyperparameter, or route weight.

The analysis uses six fixed contiguous spatial blocks (three x-thirds by two
y-halves) and four cameras. For every camera and outer block:

1. fit the GP spatial field only on L0 detector events outside the outer block;
2. construct calibration-link training predictions by inner spatial folds within
   those remaining five blocks, so the link never receives an in-sample GP score;
3. freeze that GP and link, score L0 in the held outer block, and score L2 in the
   same geographic block;
4. for the hybrid, fit the residual once against the **L0** monocular-prior latent
   field on L0 training data. At L2, freeze that residual and add it to the newly
   computed L2 monocular-prior latent field. It is forbidden to refit an L0 residual
   against the L2 prior;
5. the monocular-depth floor-affine anchor is fitted in L0 and frozen. Only raw depth
   inference and raycasting are recomputed from L2 camera RGB.

This replaces the discovery analysis's leaked full-L0 GP score and its incorrectly
recomputed hybrid residual. Full-L0 deployment fields may be shown for route choice,
but they are not used as held-out prediction evidence.

## 4. Confirmatory hypotheses and multiplicity

There are two confirmatory hypotheses. Their two-sided exact sign-test p-values form
one Holm family of size two at family-wise alpha 0.05. A hypothesis supports the
claim only if its effect has the registered positive sign and its Holm-adjusted
p-value is at most 0.05. Effect sizes and deterministic 10,000-resample paired
percentile bootstrap 95% intervals (seed 20260819) are reported regardless of sign.

### H1: prediction transfer (mechanistic primary)

The experimental unit is camera x outer spatial block, `n=4 x 6=24`. For unit `u`,
let `B(arm, environment, u)` be raw Brier score against `det_hit` under the
leakage-free procedure above. The registered contrast is

```
D_u = [B(GP, L2, u) - B(GP, L0, u)]
    - [B(mono-depth, L2, u) - B(mono-depth, L0, u)].
```

Positive `D` means the historical GP loses more accuracy under reconfiguration than
the field recomputed from current imagery. The exact sign test drops exact-zero ties
at tolerance 1e-12 and tests equal probability of positive and negative signs. The
paper reports all 24 unit values, their mean, median, positive/negative/tie counts,
raw p-value, Holm-2 p-value, and paired bootstrap interval.

### H2: offline route consequence (one frozen operational cell)

The one route cell is **four cameras and a 20% maximum length budget**. This cell is
chosen because it is the deployed camera count and a concrete operational detour
budget, not because it was the smallest L1 p-value. The task set is the 89 unordered
start-goal pairs already declared in E3: the 16 frozen waypoints separated by at
least 8 m. No task is added or removed after L2.

For task `t`, the route solver chooses the least predicted expected blind distance
among routes no longer than 1.20 times the shortest route. Detector outcomes,
aggregated over the four headings as already specified in E3, score realised blind
distance and never enter route selection. The contrast is

```
R_t = [(blind(GP route, L2, t) - blind(GP route, L0, t))]
    - [(blind(mono-depth route, L2, t) - blind(mono-depth route, L0, t))].
```

Positive `R` means the frozen field's route degrades more than the route chosen from
current imagery. The same tie rule, exact two-sided sign test, and paired bootstrap
reporting apply. Tasks are a fixed benchmark set with shared waypoints, so inference
is explicitly limited to this declared task set and not presented as 89 independent
warehouses.

## 5. Secondary and exploratory families

The remaining 24 camera-subset x detour-budget route cells are secondary and receive
Holm correction as one family of 24. The direct L2 GP-minus-mono route gaps over all
25 cells form a separately labelled exploratory Holm-25 family. No unadjusted cell
is called significant.

Prediction log loss, AUROC, ECE, false-visible rate, Brier skill, CAD references,
hybrid, distance, FOV/range, and constant arms are secondary effect-size analyses.
Thresholds 0.05 and 0.50 are sensitivity analyses. They do not replace H1, and no
post-hoc subgroup or metric becomes primary. Per-camera plots are diagnostic, not
four additional confirmatory hypotheses.

L1 and L2 are both shown. L1 remains explicitly discovery evidence; it is not pooled
with L2 to rescue a failed confirmation. A descriptive across-layout mean and both
layout-specific intervals may be reported, but the central replication statement is
decided by L2 H1. The stronger joint statement that stale fields harm the registered
route decision requires L2 H2 as well.

## 6. Falsifiers and stopping rules

- If L2 H1 is non-positive or fails Holm-2, the prospective evidence does not confirm
  that current imagery transfers better than historical detections.
- If H1 confirms but H2 does not, the paper may claim prediction transfer but not an
  operational routing consequence at four cameras and 20% detour.
- If the external draw produces a geometrically weak change, that is the realised
  randomised replication. It is reported; there is no second pulse or redraw.
- If any capture-contract check fails, no L2 result is computed from that capture.
- If a bug is found after L2 outcomes are exposed, the fix, discovery time, affected
  hashes, and before/after results are disclosed. The preregistration is not edited;
  an amendment is appended as a separate dated file.

## 7. Scope

L2 removes the single-reconfiguration objection within the controlled simulation
testbed. It does not turn two simulated layouts into a real-deployment claim, does
not add algorithmic novelty, and does not rehabilitate the deployed EFE objective
whose frozen visibility term failed to change a route. H2 remains an explicitly
offline decision-rule result. Any real-image depth sanity check or closed-loop route
execution is reported under its own protocol and evidence boundary.
