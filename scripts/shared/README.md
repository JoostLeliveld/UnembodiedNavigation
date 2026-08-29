# scripts/shared — the one shared analysis library

| module | status | use for |
|---|---|---|
| `metrics.py` | **canonical** (2026-07-15) | ALL scoring in new analysis code: brier, logloss, auroc, ece, fhtr, spearman, probit_prob, gaussian_nll_logit, coverage_logit, binned. Never re-implement these inline — an audit found 15 divergent copies (3 different Spearman formulas, inconsistent Brier epsilons). |
| `paths.py` | **canonical** | `repo_root()` — locating the checkout. Never re-derive a repo root by counting `..`. |

Canonical modules that live elsewhere (don't duplicate them here):
- fusion-study run LOADING: `experiments/fusion_on_fixed_routes/aligned.py`. This is the
  only sanctioned reader of fusion run CSVs: it scores each quantity at the instant that
  quantity describes and counts each detector batch once. See `docs/localization_metrics.md`.
- per-timestep campaign-CSV column safety: `scripts/geometry_visibility/campaign_metrics.py`.
  Diagnostic scope only — it does not time-align and does not deduplicate.
- GP fitting (point / uncertainty-weighted / belief-spread / expected-kernel):
  `scripts/visibility_comparison/fit_belief_aware_gp.py`.
- camera model: `src/unav_common/unav_common/camera_model.py` (`ObliqueCameraModel`).
- trust→R_plan precision blend: `reliability.covariance_mapping` (the single source of truth)
  (`trust_to_r_plan`) — most-tested equivalent: `reliability.single_camera_adapter.precision_blend_covariance`.

Usage:
```python
import sys; sys.path.insert(0, "scripts/shared")
import metrics as M
M.brier(y, p); M.spearman(a, b)
```
