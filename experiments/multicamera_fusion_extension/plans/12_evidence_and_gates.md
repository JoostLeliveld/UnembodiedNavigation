# Plan 12 — Evidence bundle, acceptance gates, disclosure register (§20–22)

## Where evidence lives (repo conventions override the draft's §20 layout)
- Study outputs: `logs/studies/multicamera_fusion_extension/<expN>/` (figures,
  metrics CSV, RESULTS.md each).
- Locked artifacts (frozen GPs, calibrators, stackers, covariances, detector,
  calibration JSONs): `paper_artifacts/` with model cards.
- Manifests: `research_story/04|08|09/evidence.yaml` + `registry.yaml`
  (update AFTER the parallel workstream's in-flight edits are committed).
- Claims/figure list: chapter claim files + `FIGURE_BACKLOG.md` additions
  (§20's 15-figure list mapped to 09A–09F plus new 04x/08x entries).

## Per-result provenance row (extend `docs/experiment_registry.md` pattern)
experiment_id, git_commit, world_hash, camera_calibration_hashes,
detector_hash, gp_artifact_hashes, calibrator_hashes, trust_model_hash,
filter_config_hash, planner_config_hash, route, seed, camera_subset,
fault_profile, ground_truth_access=evaluation_only.

## Module acceptance gates (§21 — each blocks its downstream)
perception → GP → confidence → covariance → health → fusion → planning; the
concrete pass conditions live in plans 01–10. CI-style enforcement:
`tests/reliability/test_leakage_firewall.py` grows one case per new module
(no gt_/eval_ imports or columns on the operational side).

## Disclosure register (extend GATE_PROVENANCE.md)
Standing entries: asserted-vs-derived constants (0.45 threshold, r_miss_uv
decision, γ, λ_a/λ_q); in-sample vs held-out validations; camera-C residual;
CPU-camera-A constraint; superseded artifacts after detector retrain;
single-camera evidence remains the locked core until multi-camera chains
exist (claim discipline).

## Minimum viable paper (§22) — what must be DONE vs stretch
MVP: Toro baseline, per-camera availability/usability GPs, calibrated
confidence, trust stack, anisotropic per-camera R_cond (global, not spatial),
reliability-aware fusion, subsets study, one dropout study, one drift study,
selected closed-loop comparison.
Stretch (only after MVP passes): spatial heteroscedastic R(s), uncertain-input
GP as a *feature* (it is a core thesis contribution in ch.03 — coordinate,
don't duplicate), online GP updates, physical camera moves, joint
selection/fusion policy, active commissioning (ch.10).
