# achievable_precision_map — what the robot can KNOW, not just where it is SEEN

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-PRECISION
status: LOCKED
claim_ids:
- C1
- C3
- C5
assumption_ids:
- A03
- A11
- A13
- A16
reviewer_question_ids:
- RQ01
- RQ09
- RQ13
figure_ids:
- F05
dependencies:
- ASSET-RUNTIME
- ASSET-PLANNER
operational_inputs:
- availability
- odometry_growth
- camera_residual_floor
evaluation_only_inputs: []
primary_metric: floor fraction where availability and precision select different cameras
promotion_gate: Demonstrate nontrivial decision disagreement.
evidence_paths:
- logs/studies/achievable_precision_map/exp1_precision_vs_coverage/summary.json
archive_rule: Preserve summary map and generator.
next_action: Use the locked fields in camera-management experiments only after source
  comparison.
```

<!-- RESEARCH-METADATA:END -->


**Chapter served:** [09 — multicamera handover & fusion](../../research_story/09_multicamera_handover_fusion/)
· fills §VIII of [`PAPER_DRAFT_abstract_intro_2026-08-04`](../../modules/07_multicam_handover_fusion/framings/PAPER_DRAFT_abstract_intro_2026-08-04.md)

## Question

Coverage-aware planning asks *"will a camera see me here?"*. The correlation-floor
result ([`bayesian_filter_showcase/exp1`](../bayesian_filter_showcase/)) says the belief
can never be sharper than the residual systematic of the camera supplying it — so a
well-covered spot watched only by a leaning camera is a spot the robot cannot know
precisely, no matter how reliably it is seen.

Is that distinction big enough to change which camera a planner should use?

## Answer (exp1)

**Yes, on 15.7 % of the reachable floor.** Achievable precision is 2.6 cm median when you
select for it, 3.5 cm if you follow coverage, and 3.6 cm worse in the region where the
two criteria disagree. Camera C is most *available* on 25 % of the floor but most
*informative* on only 14.8 % — coverage-only planning over-trusts it across roughly a
tenth of the warehouse.

## Run

```bash
python3 experiments/achievable_precision_map/exp1_precision_vs_coverage.py
```

Offline, seconds, no Gazebo and no new capture — it composes measurements that already
exist. Outputs → [`logs/studies/achievable_precision_map/exp1_precision_vs_coverage/`](../../logs/studies/achievable_precision_map/exp1_precision_vs_coverage/).

## Reuse map

| need | reused from |
|---|---|
| per-camera availability field `p_c(x,y)` | `paper_artifacts/gp/warehouse_full_4cam_fused_v1/fused_planner_four_camera.npz` (frozen) |
| per-camera residual bias floor | `logs/studies/operational_residual_rcond/exp2_operational_rcond` |
| detection rate, odometry drift constant | recorded runtime; same constants as `bayesian_filter_showcase` |
| repo paths | `scripts/shared/paths.repo_root` |

## Next

Route-level prediction: do two routes between the same endpoints differ in achievable
precision even where coverage is equal? That is the gate on whether the closed-loop
campaign is worth machine time — see the framing addendum's runway list.
