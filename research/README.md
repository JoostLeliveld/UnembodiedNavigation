# Research control plane

This directory is the scientific source of truth for the thesis. The current paper is the
focused correlated-error and belief-honesty study. The broader reliability-source
comparison is the next thesis chapter and is not a prerequisite for writing that paper.

## Current focus

**Research:** `EXP-CL-CAL`, currently blocked at calibration identifiability before any
closed-loop protocol is frozen.

**Gate:** separate silhouette/yaw, camera and region effects on held-out groups. Only then
freeze a causal arm pair, pass readiness and run a matched matrix or preserve a null.

**Next action:** design the WS05 yaw-diverse, route/region-disjoint identifiability protocol.
Do not select v2/v3/v4 confirmatory arms, generate configs or launch runs first.

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

- Structured residual error and projection amplification, with camera-bias attribution open.
- Conventional-filter overconfidence under persistent correlated error.
- Per-camera residual flooring and leave-one-out belief checks.
- Historical gated cross-bearing calibration and its v3-specific drift-expiry monitor.
- The difference between availability and achievable precision.
- Null results for sharper conditional covariance and naive fusion. The narrated GP null is
  not evidence of record until its missing package is recovered or rebuilt.

The live progress table is in [`STATUS.md`](STATUS.md). Decisions, archive manifests, and
recovery instructions belong in [`09_decisions_and_risks.md`](09_decisions_and_risks.md).
