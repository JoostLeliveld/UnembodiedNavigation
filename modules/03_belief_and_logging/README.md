# 03 — Operational belief & logging

[Back to modules index](../README.md)

| | |
|---|---|
| **Claim** | Ordinary robot driving yields trust-training records at *uncertain* robot positions, logged behind a ground-truth firewall so evaluation signals can never train the operational model. |
| **Status** | Status report — no four-panel demo yet. |
| **Chapter** | [01 — operational belief and logging](../../research_story/01_operational_belief_and_logging/) (PARTIAL) |

## What it computes
Belief-stamped detector events (mean + covariance) from real drives, plus the
opportunity/leave-one-camera-out labelling that turns "a miss" into a
meaningful, opportunity-gated label. The NEES honesty study (16.8 → 2.8 after
smoothing) lives in this chapter.

## Where it lives
- Contracts + firewall: [`../../src/reliability/reliability/contracts.py`](../../src/reliability/reliability/contracts.py), [`firewall.py`](../../src/reliability/reliability/firewall.py)
- Reliability-contract schema: [`../../docs/reliability_contracts/`](../../docs/reliability_contracts/)
- Opportunity / LOO builders: [`../../experiments/multicamera_fusion_extension/tools/build_opportunity_dataset.py`](../../experiments/multicamera_fusion_extension/tools/build_opportunity_dataset.py), [`build_loo_labels.py`](../../experiments/multicamera_fusion_extension/tools/build_loo_labels.py)

## Demo owed (D2)
Split-screen "what the SYSTEM sees" vs "what only EVALUATION sees" while the
robot drives, plus the odom-as-truth incident and the firewall diagram drawn
from config. Design in [`../../research_story/DEMO_LAYER_PLAN_2026-07-16.md`](../../research_story/DEMO_LAYER_PLAN_2026-07-16.md).
