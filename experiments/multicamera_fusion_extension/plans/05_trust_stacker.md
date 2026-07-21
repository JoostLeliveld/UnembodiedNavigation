# Plan 05 — Trust stacker τ (§8.3)

## Purpose
Combine spatial GP quality prior, calibrated per-frame confidence, GP
uncertainty, and camera health into one trust probability for the CURRENT
observation:

```
τ = σ( β0 + βs·logit(q̂(s)) + βc·logit(p^C) − βu·σ_q(s) + βh·logit(h) )
```

Not a naive product (GP and confidence are correlated); fitted as a stacking
model on a held-out split (never the split that trained the GP or calibrator).

## New code
`src/reliability/reliability/trust_stacker.py` + tests.

```python
@dataclass(frozen=True)
class TrustFeatures:
    spatial_quality: float       # q̂_i(s_t) mean
    spatial_uncertainty: float   # σ_q,i(s_t)
    calibrated_confidence: float # p^C
    camera_health: float         # h from plan 08 (1.0 until health lands)

class TrustStacker:
    fit(features, labels, groups)   # logistic IRLS, grouped input
    predict(features) -> tau
    coefficients / to_json / from_json (provenance + hashes)
```

Grouped cross-validation by route/run for hyper-choices; clipped logits
(ε-bounded) so degenerate 0/1 inputs stay finite.

## Gates (= E1 combined-model acceptance)
- Stack beats BOTH GP-only and confidence-only on grouped held-out Brier/NLL —
  otherwise the combination is unjustified complexity and the paper reports the
  simpler winner (H1 acceptance gate).
- False-trust rate at τ ∈ {0.8, 0.9, 0.95} reported.
- Ablation table: each feature dropped once (feeds the §20 trust-ablation figure).
- Unit tests: IRLS convergence on separable/noisy synthetic data, monotone
  response to each feature, serialization round-trip, group-leakage guard
  (fit refuses identical group ids across train/validate when asked to split).
