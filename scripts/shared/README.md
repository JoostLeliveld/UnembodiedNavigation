# scripts/shared — the one shared analysis library

| module | status | use for |
|---|---|---|
| `metrics.py` | **canonical** (2026-07-15) | ALL scoring in new analysis code: brier, logloss, auroc, ece, fhtr, spearman, probit_prob, gaussian_nll_logit, coverage_logit, binned. Never re-implement these inline — an audit found 15 divergent copies (3 different Spearman formulas, inconsistent Brier epsilons). |
| `common.py` | **stale twin — do not edit** | Near-identical duplicate of `scripts/visibility_comparison/common.py` (the one every consumer actually imports via sys.path). Until consolidated, treat the `visibility_comparison` copy as canonical and keep this one untouched. |

Canonical modules that live elsewhere (don't duplicate them here):
- campaign-log LOADING: `scripts/geometry_visibility/campaign_metrics.py` (column-safety
  asserts; see the `campaign-metrics` skill).
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
