# Bayesian filter showcase

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-BELIEF
status: LOCKED
claim_ids:
- C1
- C3
assumption_ids:
- A10
- A11
- A13
- A14
- A15
reviewer_question_ids:
- RQ11
- RQ13
- RQ14
figure_ids:
- F02
dependencies:
- ASSET-RUNTIME
- ASSET-PLANNER
operational_inputs:
- camera_measurement
- camera_id
- odometry_belief
- residual_floor
evaluation_only_inputs:
- ground_truth_pose
- nees
primary_metric: NEES and truth outside stated 95 percent ellipse
promotion_gate: Honest belief on each leave-one-capture-out fold without material
  RMSE loss.
evidence_paths:
- logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting/summary.json
- logs/studies/bayesian_filter_showcase/exp2_does_it_generalize/summary.json
archive_rule: Preserve all summaries and provenance as headline evidence.
next_action: Carry the frozen belief fields into the matched closed-loop campaign.
```

<!-- RESEARCH-METADATA:END -->


The generated registry metadata below makes this locked belief-honesty evidence traceable.
