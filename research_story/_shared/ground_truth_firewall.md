# Ground-truth firewall — the two-channel architecture

Every recorded run has two physically separated data paths. Operational/
reliability-learning code may touch only the **operational channel**; ground
truth lives in the **evaluation channel** and may *score* a result but can never
train, tune, or feed an online model. This is the single most important integrity
rule of the programme. See
[`../PROGRAMME_ROADMAP_2026-07-21.md`](../PROGRAMME_ROADMAP_2026-07-21.md) and the
per-study `NO_SHORTCUTS.md`.

## Channels

| | **Operational channel** (allowed to the method) | **Evaluation channel** (scoring only) |
|---|---|---|
| Perception | camera frames, detector boxes, selected bottom-centre pixel, raw score | — |
| State | odometry, belief mean `μ`, belief covariance `P`, innovations, health stats | Gazebo world pose, true projection error |
| Calibration | camera intrinsics/extrinsics/homography, calibration hash | true calibration perturbation |
| Geometry | drivable-region polygon (`driveable_geometry_json`) | CAD/SDF **shelf** geometry, exact clearance, geometry breach |
| Outcome | timestamps, camera status | physics contacts, collision, goal success, oracle best-camera identity |

Drivable-lane geometry is operational (the planner already uses it); CAD *shelf*
geometry and all `gt_*` / `eval_*` / oracle fields are evaluation-only.

## Code enforcement (do not reinvent — extend)

The firewall is enforced in code and CI, not by convention alone:

- **`reliability.contracts`** — `EVALUATION_ONLY_FIELD_NAMES` + `EVALUATION_ONLY_TOKENS`;
  `reject_evaluation_only_keys(...)` runs in every operational dataclass
  `__post_init__` (`OperationalReliabilitySample`, `CameraObservation`,
  `CameraQuality`, `ReliabilityPrediction`, `UpdateCovariance`, `PlanningCovariance`).
  A `LeakageError` is raised if a GT field or token appears.
- **`reliability.firewall`** — `validate_feature_columns`, `validate_training_loader_sources`,
  `validate_config_sources`, `validate_planner_facing_imports` / `_import_paths`,
  `scan_import_targets`, `assert_operational_feature_table`. Use these in every new
  dataset builder / loader / config reader.
- **`reliability.evaluation`** and `reliability.oracle` are the **quarantined**
  evaluation-only surface: study code (`tools/…`) may import them; the operational
  `reliability` package must not. `validate_planner_facing_imports` proves this.

## The CI obligation

Every new operational export reader, dataset builder, or planner-facing module
must be covered by a leakage test. Existing coverage in
`tests/reliability/test_leakage_firewall.py`:

- `test_feature_columns_reject_ground_truth_and_outcome_fields`
- `test_training_loader_sources_reject_gt_topics_and_paths`
- `test_evaluation_contexts_may_name_oracle_sources` (evaluation code *may* name oracle sources)
- `test_operational_feature_table_combines_column_and_source_checks`
- `test_planner_facing_imports_reject_evaluation_modules`
- `test_new_reliability_package_has_no_planner_facing_eval_imports`
- `test_normal_runtime_config_rejects_truth_state_or_reliability_sources`
- `test_active_warehouse_visibility_campaign_is_gt_free_for_runtime_sources`

**Rule:** a new operational component without a corresponding firewall test does
not pass its module gate.

## Label-source caveat (WP2/WP3)

Because the method has no operational ground truth, training labels come from
leave-one-camera-out / short-horizon-odometry / surveyed-commissioning-point
references (roadmap §6.3). These are *operational* labels with documented
uncertainty — not GT. GT may **audit** whether those labels are biased, but the
audit output is evaluation-only and never re-enters training.
