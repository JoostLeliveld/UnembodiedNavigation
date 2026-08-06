# planner_covariance_branching — is folding availability into R defensible?

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-PLANNER-BRANCH
status: LOCKED
claim_ids:
- C3
assumption_ids:
- A11
- A15
reviewer_question_ids:
- RQ13
- RQ14
figure_ids:
- F08
dependencies:
- ASSET-PLANNER
operational_inputs:
- prior_covariance
- candidate_route
- observation_model
evaluation_only_inputs: []
primary_metric: branch choice and predicted terminal covariance
promotion_gate: Preserve as offline evidence; navigation claim still requires closed
  loop.
evidence_paths:
- logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/summary.json
archive_rule: Preserve summary grid and null route cases.
next_action: Reuse its route-discrimination pattern for EXP-USABLE.
```

<!-- RESEARCH-METADATA:END -->


**Chapter served:** [09 — multicamera handover & fusion](../../research_story/09_multicamera_handover_fusion/)
(ICRA-2027 observation-model workstream; index in
[`modules/07_multicam_handover_fusion`](../../modules/07_multicam_handover_fusion/README.md)).

## Question

The planner needs a covariance for a *future* camera observation that may or may
not arrive. Two families exist:

| model | form | needs a miss endpoint? |
|---|---|---|
| single-`R` shortcut, deployed | `1/R = p/R_visible + (1-p)/R_miss`, then ONE Kalman update | yes (unreconciled 40 vs 120 px) |
| single-`R` shortcut, endpoint-free | `R_det / p`, then ONE Kalman update | no |
| honest branch | `E[P+] = p·P_hit(R_det) + (1-p)·P-` | no |

`R_det/p` is not a straw man: it is exactly the `R_miss -> inf` limit of the
deployed blend (asserted in `tests/reliability/test_observation_model.py`). So the
two shortcut rows are the same family, and the question is how far that family is
from the branch model, and **where**.

## Why this needs no data

Everything is the algebra the planner already runs, evaluated over a grid of
(prior covariance, availability, conditional noise). There is no fitted model, no
capture and no ground truth, so there is nothing to be wrong about except the
arithmetic — which is checked against the CasADi runtime bit-for-bit by
`tests/reliability/test_observation_model.py::test_branch_posterior_matches_the_casadi_runtime_expression`.
That is the point of running it: it settles a modelling question that no amount of
Gazebo time would settle.

This is *not* evidence that the branch model navigates better. It bounds the
approximation error of the mapping. The closed-loop claim needs the real campaign
with `use_hit_miss_mixture=True`, which is still gated on a measured `R_cond`.

## Run

```bash
python3 experiments/planner_covariance_branching/exp1_scaled_vs_branch.py
```

Outputs → [`logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/`](../../logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/)
(`grid.csv`, `summary.json`, `fig_p1`–`fig_p3`, `RESULTS.md`).

## Reuse map

| need | reused from |
|---|---|
| branch posterior `E[P+]` | `reliability.observation_model.expected_posterior_branch` (stdlib twin of `planning.core.casadi_efe.hit_miss_posterior_ca`) |
| deployed precision blend | `reliability.single_camera_adapter.precision_blend_covariance` (mirror of `_blend_observation_covariance_ca`) |
| `R/p` baseline | `reliability.observation_model.scaled_covariance_baseline` |
| equivalent single `R` | `reliability.observation_model.equivalent_isotropic_covariance` |
| repo paths | `scripts/shared/paths.repo_root` |

Sibling study [`experiments/efe_hit_miss_mixture/`](../efe_hit_miss_mixture/) asks
the *same* question one level up (what the EFE cost sees); this one isolates the
covariance mapping itself and adds the equivalent-`R` inversion.
