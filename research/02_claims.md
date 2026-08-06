# Claims

## C1 — Observation-quality representation

Availability alone is insufficient to characterize observation quality. A usable contract
contains:

- usable-observation probability, `p_use,c(x, y)`;
- conditional localization covariance;
- a persistent camera-specific bias or correlation floor;
- epistemic support;
- runtime freshness and health.

The historical precision blend remains only as a named legacy baseline.

## C2 — Estimator taxonomy

Reliability methods form three operational families with different information requirements
and failure modes: geometric, learned, and hybrid.

## C3 — Belief and routes

Explicit observation-quality modelling can change predicted belief and route choice. The
canonical expected correction is

```text
E[P+] = p_use P_hit + (1 - p_use) P-
```

It is compared with constant covariance, the legacy precision blend, and `R/p`.

## C4 — Navigation consequence

Better observation models improve navigation only on routes where observation quality
changes achievable belief. This remains an open hypothesis until clean-goal, breach, and
belief-calibration evidence is complete.

## C5 — Camera management

Camera management must be evaluated separately from quality estimation. Reliability fields
are frozen before comparing nearest camera, maximum availability, achievable precision,
hysteretic selection, and conservative fusion.

## C6 — Deployment regimes

Different operational regimes favour different estimator families. Relevant regimes include
layout change, stale or rescanned geometry, frozen or updated learning, calibration drift,
dropout, and latency.
