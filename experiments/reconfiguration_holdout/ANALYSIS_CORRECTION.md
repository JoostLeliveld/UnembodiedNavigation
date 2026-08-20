# Analysis correction — 2026-08-19

This document is an immutable audit note, not a rewrite of
`PREREGISTRATION.md`. It records two implementation errors found after the first
L0→L1 analysis and the result after correcting them.

## Errors in analysis v1

1. The paper and preregistration described leave-one-spatial-block-out scoring for
   L0. The v1 code fitted the GP spatial field on all L0 detector outcomes and held
   out only the two-parameter calibration link. Consequently the nominal GP and
   hybrid scores were partly in-sample while their L1 scores were out of environment.
2. The v1 hybrid fitter supplied each target environment's monocular prior both when
   subtracting a mean from the L0 training labels and when adding a mean at query
   points. This produced a different learned residual for every target environment;
   it was not the frozen L0 residual described by the method.

The affected v1 E1 tables, figures, and manuscript claims are withdrawn. They must
not be quoted.

## Corrected protocol (v2)

For each camera and each of six outer spatial blocks:

- fit the GP on L0 outcomes outside the outer block;
- define the hybrid residual only as
  `logit(L0 outcome) - logit(L0 monocular prior)`;
- predict the outer block in L0 and the same spatial block in L1 with that one model;
- for the hybrid, add the unchanged residual prediction to the L0 or L1 query prior;
- fit the calibration link on nested out-of-sample L0 predictions: each inner block
  is predicted by a GP that saw neither the outer block nor that inner block; and
- score L0 and L1 with the identical outer-fold model and link.

The implementation is in `gp_transfer_refit.py`; membership and exclusion invariants
are tested by `tests/experiments/test_reconfiguration_transfer_protocol.py`.
Full-L0 fields used as deployment artifacts for routing are separate. Hybrid `.npz`
artifacts now persist the residual latent grid and assert the L0 training prior
separately from the environment-specific query prior.

## Corrected L1 result

The preregistered raw-Brier interaction is null:

| comparison | mean interaction | paired bootstrap 95% CI | exact sign p | Holm (2) |
|---|---:|---:|---:|---:|
| ΔBrier(GP) − ΔBrier(mono) | +0.00424 | [−0.00307, +0.01310] | .167 | .334 |
| ΔBrier(hybrid) − ΔBrier(mono) | +0.00362 | [−0.00381, +0.01140] | .167 | .334 |

Here positive would mean the historical component degraded more. Neither interval
excludes zero. The earlier skill-score headline was also a post-outcome substitution:
the frozen preregistration names raw Brier as primary. Skill is therefore retained
only as descriptive sensitivity analysis.

The machine-readable result is
`logs/studies/reconfiguration_holdout/e1_reconfiguration_holdout/e1_inference_L1.csv`
with its hashed manifest. This correction narrows the paper's prediction claim: the
present L1 data do not establish differential raw-Brier degradation under the
registered block-holdout protocol.
