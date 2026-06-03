# Publication Checklist

Use this as the release gate before putting the repository next to the paper.

## Must Be True For Main Results

- [ ] P0 paper issues in `docs/paper_alignment.md` are resolved in the paper:
  odometry source, projection geometry, calibration limitation, and
  traversability/visibility separation.
- [ ] Every reported figure/table maps to a run directory, campaign log, GP
  artifact, and generation script.
- [ ] For compact benchmark claims, `paper_campaign_config.yaml` is the config
  used for reported compact runs.
- [ ] For AWS claims, `scripts/visibility_comparison/aws_paper_final_config.yaml`
  is cleaned, locked, and matches the run manifests.
- [ ] Run manifests record the YOLO model, GP artifact, planner condition,
  world, task, seed, heading source, noise settings, and no-go settings.
- [ ] `compute_paper_metrics.py` is run against the same GP artifact used by
  the visibility-aware run manifests.
- [ ] Figures are generated from real run traces unless explicitly labeled as
  schematics.
- [ ] The paper states that heading comes from odometry in the current campaign.
- [ ] The paper states that homography calibration uncertainty is not propagated.
- [ ] Traversability/no-go geometry is described separately from visibility.
- [ ] Figures distinguish known forbidden zones from learned observation
  reliability / camera occlusion.
- [ ] Main result figures include planner-facing ambiguity or EFE cost
  decomposition, not only the GP reliability field.
- [ ] GP coverage / uncertainty is shown or explicitly discussed as
  extrapolation outside sampled regions.
- [ ] Route choice is not scripted by mission waypoints.

## Must Be True For Experiment B Claims

- [ ] Final AWS world geometry, camera pose, route geometry, and learned
  reliability artifact are fixed.
- [ ] `warehouse_aws.world.sdf` has its own validated camera image.
- [ ] AWS detector dataset exists and is documented.
- [ ] AWS YOLO model path is explicit in the AWS configs.
- [ ] AWS visibility samples are captured in the AWS world.
- [ ] AWS GP artifact contains `P_conservative_plan_map`.
- [ ] Main AWS route-choice campaign has complete matched seeds for C1, C2, and
  C3 risk-only ablation.
- [ ] Uniform-visible sanity campaign shows C1/C2 behave similarly when learned
  visibility should not change the route.
- [ ] AWS figures/metrics are generated from final logs, including route class,
  localization error, p95 error, yaw error, clearance, and cost decomposition.
- [ ] Paper wording states the current AWS claim as localization-safety /
  stability unless a completed stress task shows a fair failure-rate contrast.

## Release Hygiene

- [ ] No build/install/log/cache directories are included in the source release.
- [ ] No local model weights are committed unless the license and size are
  intentional.
- [ ] No hidden `.claude`, `.codex`, `.agents`, or chat-context dumps are
  committed. Root `CLAUDE.md` and `AGENTS.md` are intentional maintained
  guidance.
- [ ] Local absolute paths in docs are either removed or clearly marked as local
  reproduction examples.
- [ ] Third-party assets keep their license and notice files.
- [ ] A repository license is chosen by the author.
- [ ] `CITATION.cff` or equivalent citation metadata is added.
- [ ] Data/model availability is stated in the README or paper.
