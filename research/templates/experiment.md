---
experiment_id:
status: PLANNED
claim_ids: []
assumption_ids: []
reviewer_question_ids: []
figure_ids: []
dependencies: []
operational_inputs: []
operational_interface_inputs: []
evaluation_only_inputs: []
primary_metric:
promotion_gate:
evidence_paths: []
archive_rule:
next_action:
---

# Experiment title

## Question and causal comparison

State the one scientific decision this experiment resolves and the independent variable.

## Frozen controls

Record data split, world, cameras, detector hash and threshold, calibration, robot,
planner/controller, seeds, noise interface, and sample budget.

## Procedure

Keep operational inputs separate from evaluation-only inputs.

## Outputs

Write raw outputs under ignored `logs/`, then promote `RESULTS.md`, provenance, summary
JSON/CSV, and decisive figures. Null results are permanent.

## Cleanup

Remove scratch configs and failed generated output; move irreplaceable retired raw data to
verified cold storage; update the registry and regenerate `research/STATUS.md`.
