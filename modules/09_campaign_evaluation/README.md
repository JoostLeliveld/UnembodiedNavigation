# 09 — Campaign & evaluation

[Back to modules index](../README.md)

| | |
|---|---|
| **Claim** | A single representative route is not evidence; matched C1/C2 conditions across four routes and five seeds, scored at the run level, are. |
| **Status** | Demonstrated via the experiments surface (static counts + figures). |
| **Chapter** | [00](../../research_story/00_problem_and_existing_baseline/) / [06](../../research_story/06_original_warehouse_navigation/) / [11 — final thesis campaign](../../research_story/11_final_thesis_campaign/) |

## What it computes
The benchmark surface — tasks, conditions, seeds, metrics, reproduction commands
— and the run-level inference backbone (Wilson intervals, paired differences,
clustered route→seed→episode bootstrap, pre-registered beats-Toro decision rules).
`honest_campaign_v1` is the locked AWS reference: C1 15/20 vs C2 20/20 clean goals.

## Where it lives
- Benchmark surface + demo media: [`../../experiments/`](../../experiments/) (`experiments/demos/` holds the current outcome-count figures)
- Campaign runner: [`../../scripts/visibility_comparison/run_visibility_campaign.py`](../../scripts/visibility_comparison/run_visibility_campaign.py)
- Metrics (canonical, never hand-rolled): [`../../scripts/shared/metrics.py`](../../scripts/shared/metrics.py)
- Run-level statistics: [`../../src/reliability/reliability/campaign_statistics.py`](../../src/reliability/reliability/campaign_statistics.py)
- Locked media: [`../../paper_artifacts/`](../../paper_artifacts/)

> This module intentionally does not move `experiments/` — investigation code and
> data stay in their study folders; this page is the pointer + evaluation contract.
