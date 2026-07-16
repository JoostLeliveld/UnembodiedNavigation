# 07b — Geometry/occlusion model (MODEL-ONLY until gated)

**Status:** S1 explanation gate PASSED (geometry explains the empirical GP at Spearman 0.730 /
R² 0.51 over 22,917 driveable in-FOV cells; Jacobian range term co-dominant). Next step: S3
zero-shot transfer via the existing teleport-capture scripts.

**Hard rules:** predictions stay `MODEL ONLY` until checked against detector evidence
(Fig 07D); becomes a contribution only with realistically available geometry AND a clear win
over range/FOV baselines (Fig 07E); otherwise it is reported as an ablation with the honest
"range and obliquity dominate" conclusion.

Code: `scripts/geometry_visibility/` · Outputs: `logs/studies/geometry_visibility_prior/` ·
Sensed-structure alternative: `logs/visibility_comparison/depth_sensed_initial_gp_v1/`.
