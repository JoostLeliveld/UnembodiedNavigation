# Reviewer attack register

This register separates an available scientific answer from a publication-ready answer.
`ANSWERED` means the scoped concern has direct evidence; it does not imply that the mapped
figure is publication-locked. Unanswered items remain `OPEN` or `LIMITATION`.

| ID | Concern | Evidence of record | Figure(s) | Status | Defensible answer now | Release condition / if unanswered |
|---|---|---|---|---|---|---|
| RQ01 | Why not use distance only? | No source-benchmark package exists at the paths quoted by `experiments/usable_observation/README.md`. The deterministic precision map answers a different question. | F05, F07 | OPEN | None for source prediction. Distance-only must be a named baseline, not an omitted straw man. | Recover or rebuild route-block evidence and compare constant, distance-only and FOV/range on the same held-out routes; otherwise limit novelty to taxonomy. |
| RQ02 | Why use a GP? | The README narrates a GP null (0.058 versus 0.055), but `logs/studies/usable_observation/gp_v1/` is absent. Older GP artifacts under `paper_artifacts/` belong to a prior surface and do not establish this benchmark. | F07, F09 | OPEN | No verified advantage. A GP is a challenger whose commissioning cost must be earned. | Verify the missing package or rerun frozen splits. Preserve a GP tie/loss as a permanent null. |
| RQ03 | Is depth realistic? | No common source-benchmark run exists. | F07, F09 | OPEN | Only sensed depth, commissioned static geometry, and complete-map oracle may be compared, with provenance exposed. Oracle geometry is an upper bound, not an operational arm. | Report map/sensor origin, update age and forbidden truth inputs; otherwise restrict the result to commissioned static maps. |
| RQ04 | Does geometry become stale? | `drift_lifecycle.json` demonstrates calibration staleness under controlled pose faults, not map staleness. | F04, F10 | OPEN | A calibration change statistic can be scoped; layout-change robustness remains untested. | Add prespecified layout-shift blocks for geometry sources or state static-layout scope. |
| RQ05 | Does the GP transfer? | The claimed held-out-route evidence directory is absent. Existing single-camera GP surfaces do not bind the future split. | F07, F10 | OPEN | No transfer claim. | Hold out complete routes/captures and at least one layout/site; otherwise require recommissioning. |
| RQ06 | Why is hybrid worth commissioning? | No verified common-input hybrid comparison exists. | F07, F09 | OPEN | It is not yet worth commissioning on evidence. | Promote only if held-out calibration or route discrimination beats the simpler winning arm with measured cost. |
| RQ07 | What if calibration drifts? | Historical only: `logs/studies/calibration_drift_lifecycle/exp1_stale_correction/drift_lifecycle.json`; one held-out capture, controlled v3 yaw/translation faults. It predates the fresh epoch. | F04 | OPEN | The old v3 study suggests a usable change-before-harm construction, but its 0.1/0.25 degree values are not admissible evidence for the new paper package. | After identifiability and policy selection, preregister a new fault ladder/threshold, execute it on fresh session-2 blocks and render F04 deterministically from the frozen new JSON; otherwise omit F04. |
| RQ08 | Does this generalize beyond YOLO? | No alternate detector evidence. | None | LIMITATION | The contract is detector-agnostic in form but the evidence is for one frozen simulated YOLO detector. | Claim a frozen-detector contract only. |
| RQ09 | Are cameras genuinely diverse? | Four cameras have geometric diversity and nominally identical optics; E6 shows that part of the apparent residual-bias diversity was route/yaw-confounded. | F01, F03 | LIMITATION | The study probes four installed viewpoints, not optical/hardware diversity or four independent camera archetypes. | State this limitation in methods and captions; do not use camera count as a generalization unit. |
| RQ10 | Are the worlds representative? | No benchmark world-property table or site holdout exists. | F09 | OPEN | Gazebo warehouse evidence only. | Report measured layout, occlusion, range and support properties by unique site/world; otherwise narrow scope. |
| RQ11 | Is fusion overconfident? | Historical only: `bayesian_filter_showcase/exp1` and `exp2`; three capture blocks and a leave-one-capture-out check, all before the fresh epoch. | F02 | OPEN | The historical mechanism is credible enough to preregister, not to populate the new paper estimate. | Re-run the frozen ablations on fresh route/capture blocks with a site-grouped holdout and newly fitted floors; keep the result offline and preserve a null. |
| RQ12 | How expensive is commissioning? | Historical `network_commissioning_realism/exp1` gives contiguous-window sample curves for four cameras; no fresh comparable time/runtime accounting exists across source families. | F03, F09 | OPEN | No current-epoch cost or sample-efficiency estimate. | Report fresh route/capture blocks, unique sites, elapsed setup time, storage and runtime for every source. |
| RQ13 | Does improved prediction actually change navigation? | F05 is a steady-state map and generic F08 mechanics are analytic; F06 has no campaign data. | F05, F06, F08 | OPEN | Prediction and route consequences remain hypotheses. | Complete a valid matched campaign or retain a documented null. A calibration-only F06 cannot validate the whole correlation-floor/LOO package. |
| RQ14 | What makes each method fail? | Historical failures suggest bias-transfer, marginal-camera and stale-correction strata; no fresh common split exists. | F01, F03, F04, F10 | OPEN | The old failures may define preregistered strata but cannot populate current figures or estimates. | F10 must use fresh shared route/capture blocks and prespecified failure strata, with a fallback for every arm. |
| RQ15 | Are camera-specific residuals identifiable from robot appearance, route and yaw? | Historical E6 reduces recorded-yaw mean error 143.9 to 34.7 mm with the CAD model and removes the C/D cross-bearing gate signal; A/B then become confounded outliers. | F01, F03 | OPEN | The existing C/D constants are not identifiable as camera calibration, and E6 is rationale rather than fresh evidence. | Collect the preregistered 6-site x 8-yaw x 2-session matrix with simultaneous A-D observations. Pass the site-clustered 10 mm benefit, region-stability and 20 mm yaw-harm gates on untouched session 2, or preserve the no-camera-calibration null and keep F06 blocked. |

## Cross-cutting reviewer disclosures

- The fresh epochs are `correlated_error_icra_20260810_v1` and
  `reliability_source_20260810_v1`, frozen at `2026-08-10T12:02:26+02:00`.
  All earlier captures, runs, fitted artifacts, summaries and plots are historical/demo
  inputs only. Re-analysis or re-rendering does not reset their evidence date.
- No F01-F10 result is currently fresh or publication-locked. Historical `LOCKED` study
  status means preserved source evidence only, not admissibility in the new epochs.
- Frames and detections are within-run observations, never independent replicates.
  Inferential units are matched task-seed runs, route/capture blocks, or unique
  commissioning sites.
- Runtime NIS `9.21` is the chi-square 2-DOF 99% rejection policy. Offline F02 uses
  NIS `5.991`, the 95% diagnostic gate. They are different policies and must remain
  separately labelled in text, tables and figures.
- The usable-observation README quotes `dataset_v1`, `baselines_v1`, `gp_v1`,
  `confidence_v1` and `planner_conditions_v1`, but all five evidence directories are
  absent under `logs/studies/usable_observation/`. Narrative numbers are not evidence
  of record.
- Existing single-camera GP figures, paired-mechanism variants, demos and setup images
  may explain a method or diagnose a run; they cannot answer RQ01-RQ06 or RQ10 merely
  because they are tracked under `paper_artifacts/`.
- E6's 1,195 recorded-yaw rows are the primary stratum. The additional 226 tangent-derived
  rows are supporting only and cannot be treated as equivalent heading evidence.
