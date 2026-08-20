# Factorized observation planning successor

This is a new, fail-closed study.  It does not reopen or overwrite the retired
availability-paper C1--C3 campaign.

The study keeps two quantities separate:

\[
p_{\mathrm{use}}(x,c)=P(\text{camera }c\text{ delivers a usable update at }x),
\qquad
R_{\mathrm{cond}}(x,c)=\operatorname{Cov}(z-h(x)\mid\text{usable update}).
\]

The frozen availability input is the operational monocular-depth prior plus GP
residual.  The conditional covariance is commissioned independently from the
registered current pixel-to-ground evidence.  Ground truth is permitted only in
commissioning and evaluation; neither artifact may read it at runtime.

The planner studied here is deliberately named **DS-Route** (decision-sensitive
route selector).  It selects from a common set of clearance-certified route
candidates under a fixed 5% path-length budget.  Its primary endpoint is the
exact expected longest run of missed updates under independent, non-identical
Bernoulli delivery probabilities.  Conditional covariance contributes a separate
information-quality tie-break.  This is a route-level decision layer, not a claim
that the existing continuous EFE optimizer implements the method.

Run in order:

```bash
python3 experiments/factorized_observation_successor/freeze_inputs.py
python3 experiments/factorized_observation_successor/commission_rcond.py
python3 experiments/factorized_observation_successor/offline_gate.py
python3 experiments/factorized_observation_successor/heldout_configuration.py
```

`heldout_configuration.py` refuses to score B+C unless the development gate in
`offline_gate.py` passes.  A closed-loop campaign is permitted only when the
combined gate says `PASS`; otherwise the correct result is to stop.

