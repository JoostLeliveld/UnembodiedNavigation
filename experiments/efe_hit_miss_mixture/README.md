# Expected hit/miss belief correction

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-HIT-MISS
status: LOCKED
claim_ids:
- C1
- C3
assumption_ids:
- A11
- A15
reviewer_question_ids:
- RQ13
- RQ14
figure_ids:
- F05
dependencies:
- ASSET-PLANNER
operational_inputs:
- prior_covariance
- usable_probability
- conditional_covariance
evaluation_only_inputs: []
primary_metric: posterior covariance error relative to explicit mixture
promotion_gate: Analytic and runtime parity; route claim requires separate discrimination.
evidence_paths:
- experiments/efe_hit_miss_mixture/make_figures.py
archive_rule: Preserve correctness implementation and golden tests.
next_action: Require offline route discrimination before campaign allocation.
```

<!-- RESEARCH-METADATA:END -->


The generated registry metadata below makes this locked analytic result traceable.
