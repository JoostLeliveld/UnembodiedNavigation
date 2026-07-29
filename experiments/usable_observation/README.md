# usable_observation — learning p_use,c(s) for planning

**Question:** where is the warehouse external-camera system likely to provide a *usable* robot
localization observation, and can that probability improve navigation planning?

Decomposition: `p_use,c(s) = p_det,c(s) · p_qual,c(s)` where `p_det = P(detection | s)` and
`p_qual = P(usable | detection, s)`. Planner input is `p_use`, **not** raw YOLO confidence.
Serves the observability refocus (see `docs/usable_observation/`); relates to
`research_story` ch.04 (reliability GP) and ch.08 (planning) — registration in those manifests
is a pending follow-up.

## Layout (this study points at, does not copy)

| piece | where |
|---|---|
| Contract + gate (P1) | `src/reliability/reliability/observation_opportunity.py`, `observation_gates.py` |
| Frozen gate configs | `config/usable_observation_gate.yaml` (template), `config/usable_observation_gate_warehouse_aws.yaml` (single-cam corpus) |
| JSON schema | `schemas/observation_opportunity.schema.json` |
| Exporter (P2) | `src/reliability/reliability/observation_exporter.py` + `scripts/reliability/export_observation_dataset.py` |
| Dataset + maps | `logs/studies/usable_observation/dataset_v1/` (parquet, manifest, maps, `RESULTS.md`) |
| Tests | `tests/observability/` |
| Docs | `docs/usable_observation/{audit,data_contract}.md` (method/confidence/final reports pending) |

## Status

- P1 contract — **DONE**, Gate 1 PASS (23 tests).
- P2 exporter/dataset — **DONE**, Gate 2 PASS. Headline: on single-cam, `p_use` is driven by
  `p_det` (0.920); `p_qual` is near-saturated (0.997). See `dataset_v1/RESULTS.md`.
- P3 baselines (B0 constant / B1 distance / B2 FOV-range / B3 grid) — **DONE**, Gate 3 PASS.
  Headline: B3 grid wins pooled (Brier 0.043) by **memorizing** training routes but collapses on
  the novel route (0.133); smooth B1/B2 generalize (0.055). See `baselines_v1/RESULTS.md`. P4
  gate = beat ~0.055 on the held-out route, not pooled Brier.
- P4 learned GP (direct + two-stage), Gate 4 — **DONE, RESOLVED: simpler model selected.** GP
  does not beat B1 distance / B2 FOV-range on held-out-route generalization (0.058 vs 0.055);
  selected planner input = **B2 FOV/range** (best-calibrated, transfers from calibration alone).
  Two-stage ≈ direct (p_qual inert). See `gp_v1/RESULTS.md`. Contribution stands: spatial
  observability cuts p_det Brier 0.135→0.055 on a held-out route.
- Confidence critique (§10, deliverable D) — **DONE.** EVAL-ONLY: YOLO confidence is
  *positively* associated with localization error (partial Spearman +0.59 after geometry
  controls, U-shaped) and adds no out-of-route value beyond geometry → must NOT be inverse
  covariance. Refutes the old assumption on real data. See `docs/usable_observation/confidence_analysis.md`
  and `logs/studies/usable_observation/confidence_v1/`.
- P6 planner conditions — **DONE, Gate 5 PASS.** Four matched p_use field artifacts
  (uniform/geometry/gp/oracle) all consumed by the identical frozen adapter
  (`GPVisibilityMapModel` + `expected_visibility_ca` + precision blend); sigma-point / limit /
  monotonicity tests against the real planner code. Only the field differs. See
  `planner_conditions_v1/RESULTS.md`.
- P7 closed-loop navigation eval — NEXT, needs real Gazebo runs (swap only
  `visibility_artifact_path` per condition; measure predicted-vs-realized observability +
  nav outcomes). P5 multicam deferred (needs 4-cam detector gate; single-cam has one camera).

## Reproduce

```
python3 -m pytest tests/observability/ -q
python3 scripts/reliability/export_observation_dataset.py \
    --gate-config config/usable_observation_gate_warehouse_aws.yaml \
    --output logs/studies/usable_observation/dataset_v1 \
    --holdout-routes route_apron_to_a3_mid
```

## Two-world discipline

Method development here uses `warehouse_aws` (single camera). The 4-camera world evaluates the
frozen method later (P5); multicam demonstrations stay MODEL-ONLY/DIAGNOSTIC until the 4-cam
detector gates. No Gazebo GT enters any model/dataset input (firewall enforced + audited).
