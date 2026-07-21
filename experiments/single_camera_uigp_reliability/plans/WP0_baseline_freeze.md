# P1-WP0 — baseline freeze and evidence audit

**Claim.** There is a reproducible, fully-provenanced single-camera reference
system before any perception / GP / covariance / planning change is made.

**Anchor:** `research_story/00_problem_and_existing_baseline` — `honest_campaign_v1`
(C1 15/20 with 4 GT breaches vs C2 20/20, 0 contacts, 40 locked runs) is the
FROZEN headline. **WP0 never reruns or modifies `honest_campaign_v1`.** WP0
freezes provenance and *confirms the current runtime still reproduces the locked
result* in a fresh log-root.

## Assumptions / non-assumptions
- Assumes: current detector, GP artifact, C1/C2 configs, routes, seeds are the ones
  Paper 1 will build on. Non-assumes: no GT online; the reproduction run does not
  overwrite any locked run dir.

## Tasks
1. **Freeze software** → `baseline_runtime_contract.yaml`: git commit, ROS 2 +
   package versions, Python env, Gazebo/sim version, detector dep versions,
   hardware (P2000 GPU, CPU). Extend `docs/current_runtime_contract.yaml`, don't fork it.
2. **Freeze artifacts** → `baseline_artifact_manifest.csv`: sha256 of world SDF,
   camera calibration, detector checkpoint (`warehouse_yolo_detector_v1/model.pt`),
   GP artifact, route files, planner config, filter config, no-go geometry.
3. **Reproduce C1/C2** in a NEW log-root (`logs/visibility_comparison/wp0_repro_v1`,
   append-only) via `run_visibility_campaign.py` (RUNBOOK in `REUSE_MAP.md`;
   4 tasks × 5 seeds × 2 conditions = 40 runs). Collect goal outcome, final
   distance, GT breach, contact, path length, time, belief trace, update count, NIS
   rejections. Compare against frozen `honest_campaign_v1` — same headline direction,
   within run-level CI.
4. **Historical separation** → the CURRENT-vs-HISTORICAL-vs-PLANNED table (roadmap
   P1-WP0.4): submitted-paper (old detector, raw-confidence GP) = HISTORICAL;
   current C1/C2 = CURRENT; new Paper 1 service model = PLANNED. Uses evidence
   classes (`research_story/_shared/evidence_classes.md`).

## Deliverables
`baseline_runtime_contract.yaml`, `baseline_artifact_manifest.csv`,
`baseline_reproduction_report.md`, baseline plots, current-vs-historical table.
Outputs → `logs/studies/single_camera_uigp_reliability/wp0_baseline_freeze/RESULTS.md`.

## Gate G0
Pass only when: every reproduction run has complete provenance; the current
runtime reproduces the locked headline direction; CURRENT and HISTORICAL artifacts
cannot be accidentally mixed; operational nodes pass the GT-firewall check
(`tests/reliability/test_leakage_firewall.py`).

## Reuse (no new runtime code)
`run_visibility_campaign.py`, `campaign_metrics.load_run/load_detections`,
`scripts/shared/metrics.py`. Hashing/manifest = a small study tool only.
