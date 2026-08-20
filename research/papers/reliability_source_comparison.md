# Reliability-source comparison thesis chapter

## Purpose

Compare operational sources for predicting future usable observations and expose their
deployment trade-offs. This is a methodological investigation for the full thesis: model
design, interface definition, prototype comparisons and failure analysis may proceed while
the correlated-error paper is being completed. The frozen confirmatory benchmark and any
headline source-ranking claim start only after that paper package is closed.

The scientific value is not merely to select a winner. The arms add different information
in a controlled progression—prevalence, distance, calibrated field of view, sensed 3-D
structure, operational experience, and combinations of structure and experience—so the
comparison can identify what occlusion information buys, where it fails, and what it costs
to obtain and maintain.

## Arms

1. fixed conservative constant;
2. distance-only baseline;
3. calibrated FOV/range;
4. operational sensed-depth or raycast geometry with explicit provenance and an unknown-cell
   fallback;
5. Gaussian process learned from operational observations;
6. hybrid geometric prior plus operational updates;
7. a DL challenger only after offline legality and calibration gates;
8. complete-map/CAD raycast as an evaluation-only reference, never as the operational depth
   arm.

All arms predict the same `p_use` target and use only information available at future
candidate poses. Evaluation truth and current detector outcome are forbidden operational
features.

The depth provenance ladder is a sensitivity structure, not a request to make every depth
variant a headline method. The final comparison selects one primary operational depth rung,
keeps complete CAD as the oracle/reference, and may retain monocular depth as an exploratory
zero-additional-hardware challenger. Perfect, degraded, stale and missing-depth variants test
the chosen method's assumptions and fallback rather than inflating the main arm count.

## Subquestions

- Which source predicts held-out usable observations most accurately and honestly?
- Which failures are explained by occlusion, unsupported space, stale geometry, or layout
  shift?
- Which sources change expected belief and discriminate meaningful route alternatives?
- What commissioning, runtime, transfer, and update costs purchase those gains?
- After fields are frozen, which camera-selection policy best uses them without becoming
  overconfident?

## Gate sequence

Method investigation now: common interface → prototype implementation → controlled
sensitivity and failure probes.

Confirmatory comparison later: feature legality → held-out calibration → failure audit →
offline route discrimination → closed-loop navigation → deployment decision matrix. No
method skips a confirmatory gate, and exploratory results are not silently promoted into the
frozen comparison.
