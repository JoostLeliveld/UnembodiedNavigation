# Research control plane

This directory is the scientific source of truth for the thesis. The current paper is the
focused correlated-error and belief-honesty study. The broader reliability-source
comparison is the next thesis chapter and is not a prerequisite for writing that paper.

## Current focus

**Research:** `EXP-CL-CAL`, the matched 30-run closed-loop calibration campaign.

**Gate:** complete the preregistered matrix, or preserve a fully documented null result.

**Next action:** rebuild and launch the frozen campaign from a clean checkout, then finish
the matrix and promote its table and figure to `paper_artifacts/`.

**Repository maintenance:** `MNT-CONSOLIDATION`, migration to this control plane and a
verified external cold archive.

## What is authoritative

- [`registry.yaml`](registry.yaml) is the only machine-readable authority for status,
  dependencies, next actions, and evidence.
- [`STATUS.md`](STATUS.md) is generated; never edit it manually.
- The numbered documents explain the science and do not carry competing progress state.
- [`papers/`](papers/) selects thesis claims for publications without creating new claim
  systems.

Regenerate and validate with:

```bash
python3 scripts/research/validate_registry.py
python3 scripts/research/build_status.py
python3 scripts/research/hygiene_check.py
```

## Locked evidence already available

- Camera-specific residual bias and projection amplification.
- Conventional-filter overconfidence under persistent correlated error.
- Per-camera residual flooring and leave-one-out belief checks.
- Gated cross-bearing calibration and its drift-expiry monitor.
- The difference between availability and achievable precision.
- Null results for sharper conditional covariance, naive fusion, and GP superiority.

The live progress table is in [`STATUS.md`](STATUS.md). Decisions, archive manifests, and
recovery instructions belong in [`09_decisions_and_risks.md`](09_decisions_and_risks.md).
