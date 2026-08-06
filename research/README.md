# Research control plane

This directory is the scientific source of truth for the thesis. The current paper is the
focused correlated-error and belief-honesty study. The broader reliability-source
comparison is the next thesis chapter and is not a prerequisite for writing that paper.

## Current focus

**Research:** `EXP-CL-CAL`, the closed-loop protocol and matched calibration campaign.

**Gate:** freeze scientifically valid calibration arms and analysis, pass readiness, then
complete the matched matrix or preserve a fully documented null.

**Next action:** resolve the v2-v3-v4 arm conflict, primary endpoint, seed matrix, and
world-field compatibility. Do not generate confirmatory configs or launch runs before those
decisions pass the workstream gates.

**Repository maintenance:** `MNT-CONSOLIDATION` is complete: the control plane, retirement
wave, verified cold archive, clean rebuild, and bounded campaign launch all passed.

## What is authoritative

- [`registry.yaml`](registry.yaml) is the only machine-readable authority for status,
  dependencies, next actions, and evidence.
- [`STATUS.md`](STATUS.md) is generated; never edit it manually.
- The numbered documents explain the science and do not carry competing progress state.
- [`papers/`](papers/) selects thesis claims for publications without creating new claim
  systems.
- [`workstreams/`](workstreams/) contains bounded handoffs for separate chats. It defines
  ownership and dependencies but never overrides the registry.

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
