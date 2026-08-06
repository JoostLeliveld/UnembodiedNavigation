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
| RQ07 | What if calibration drifts? | `logs/studies/calibration_drift_lifecycle/exp1_stale_correction/drift_lifecycle.json`; one held-out capture, controlled v3 yaw/translation faults. | F04 | ANSWERED | For the tested v3 correction, a change statistic detects camera-C yaw drift at 0.1 degrees before the stale correction becomes harmful at 0.25 degrees. | Keep the controlled-injection, single-capture and v3-specific boundary. A v4 paper claim requires a v4-aligned lifecycle decision/evaluation. |
| RQ08 | Does this generalize beyond YOLO? | No alternate detector evidence. | None | LIMITATION | The contract is detector-agnostic in form but the evidence is for one frozen simulated YOLO detector. | Claim a frozen-detector contract only. |
| RQ09 | Are cameras genuinely diverse? | Four cameras have geometric diversity and nominally identical optics; E6 shows that part of the apparent residual-bias diversity was route/yaw-confounded. | F01, F03 | LIMITATION | The study probes four installed viewpoints, not optical/hardware diversity or four independent camera archetypes. | State this limitation in methods and captions; do not use camera count as a generalization unit. |
| RQ10 | Are the worlds representative? | No benchmark world-property table or site holdout exists. | F09 | OPEN | Gazebo warehouse evidence only. | Report measured layout, occlusion, range and support properties by unique site/world; otherwise narrow scope. |
| RQ11 | Is fusion overconfident? | `bayesian_filter_showcase/exp1` and `exp2`; three capture blocks, leave-one-capture-out check. | F02 | ANSWERED | On these recordings, repeated biased-camera updates produce overconfidence; per-camera residual flooring restores offline coverage without material RMSE loss. | Keep this offline and correlation-floor-specific. Do not claim independent fusion or closed-loop benefit. |
| RQ12 | How expensive is commissioning? | `network_commissioning_realism/exp1` gives contiguous-window sample curves for four cameras; no comparable time/runtime accounting exists across source families. | F03, F09 | OPEN | A large outlier was identifiable from 20 detections; marginal cameras were not reliably decidable with available evidence. | Report route/capture blocks, unique sites, elapsed setup time, storage and runtime for every source. |
| RQ13 | Does improved prediction actually change navigation? | F05 is a steady-state map and generic F08 mechanics are analytic; F06 has no campaign data. | F05, F06, F08 | OPEN | Prediction and route consequences remain hypotheses. | Complete a valid matched campaign or retain a documented null. A calibration-only F06 cannot validate the whole correlation-floor/LOO package. |
| RQ14 | What makes each method fail? | Current-paper failures are documented (bias transfer, marginal-camera gate, stale v3 correction); source-family failures are not on a shared split. | F01, F03, F04, F10 | OPEN | Method-specific current-paper mechanisms may be claimed separately. No source family may be promoted yet. | F10 must use shared route/capture blocks and prespecified failure strata, with a fallback for every arm. |
| RQ15 | Are camera-specific residuals identifiable from robot appearance, route and yaw? | E6 tracked evidence reduces recorded-yaw mean error 143.9 to 34.7 mm with the CAD model and removes the C/D cross-bearing gate signal; A/B then become confounded outliers. | F01, F03 | OPEN | The existing C/D constants are not identifiable as camera calibration on the current logs. Structured correlated error remains real, but its source decomposition is open. | Run the yaw-diverse, route/region-disjoint WS05 identifiability protocol. If it fails, remove the camera-calibration-benefit claim and preserve the null. |

## Cross-cutting reviewer disclosures

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
