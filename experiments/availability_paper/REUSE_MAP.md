# Reuse map

What this study imports rather than reimplements, and the one thing it deliberately
does not reuse.

| Need | Reused from | Why not local |
|---|---|---|
| Brier, logloss, AUROC, ECE, Spearman | `scripts/shared/metrics.py` | An audit found 15 divergent copies and three different Spearman formulas. Numbers from divergent copies are not comparable. |
| Repo root | `scripts/shared/paths.py::repo_root` | `parents[N]` silently resolves wrong if a file moves depth. |
| GP fitting | `scripts/visibility_comparison/fit_belief_aware_gp.py` | The canonical GP. `gp_refit.py` calls its `_load_events`, `_aggregate_events`, `_predict_mode_at_events` with the hyperparameters frozen in the 2026-07-27 validation manifest, so a refit differs from the published artifact only in what it trained on. |
| Candidate-pose grid, driveable mask, warehouse prisms, camera models, and the FOV / CAD / depth / GP / hybrid fields | `experiments/usable_observation/supervisor_comparison/render_all.py::build_context` | These fields are the frozen apparatus of the supervisor comparison. Rebuilding them here would fork them. |
| Monocular-depth visibility | `experiments/usable_observation/supervisor_comparison/10_monocular_depth_results/four_camera/maps/` | Produced by `four_camera_study.py`; `common.load_mono_depth_fields` only resamples onto the working grid and folds in the unknown-cell fallback. |
| Conditional covariance `R_cond` | `logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json` | The current zero-parameter floor IPM. Read at runtime, never hardcoded, so a change to that study propagates. |
| Belief / truth from campaign logs | `scripts/geometry_visibility/campaign_metrics.py::load_run` | `state_x/state_y` and `truth_x/truth_y` are stale in these logs — up to 4.5 m from truth. The loader asserts the canonical columns. |
| Camera poses, colours, world extent, figure canvas | `render_all` + `supervisor_comparison/FIGURE_CONTRACT.md` | One canvas across the whole supervisor package. |
| Closed-loop campaign schema | `experiments/usable_observation/supervisor_comparison/11_static_probability_planning/closed_loop_gazebo/campaign.yaml` | E4 copies it key-for-key except tasks, route seeds and `ros_domain_id_base`, so the three arms differ only in the observation model. |

## Deliberately not reused

**`render_all.dijkstra_route`.** It returns a path simplified to direction-change
corners, which is right for drawing and wrong for integrating a belief along a route:
the corner points are metres apart, so the covariance would be propagated in a
handful of giant steps. `e3_route_discrimination.dense_route` is the same search
returning the dense cell path, and `simplify()` does Douglas-Peucker only at the end,
for waypoint export.

**`render_all`'s cached `gp` and `hybrid` fields, for scoring.** They are fitted on
all events including E1's held-out blocks. They are still used for *route* planning in
E3, where a full-fit field is the right object (the deployed planner would use a field
fitted on everything it has), but never for a held-out prediction number.
