# 11 — Final thesis campaign and evidence package

**Purpose.** The frozen, end-to-end package the defense stands on: ch.06's N-campaign
(original warehouse), ch.08's per-camera transfer (`warehouse_full_4cam`), whichever second
contribution was chosen (ch.04 or ch.09), and the complete figure/video set.

**Status: PLANNED — populated last.** Nothing lands here until upstream chapters freeze.

## Rules (inherited from honest_campaign_v1 practice)

- Pre-register conditions, routes, seeds, outcome definitions, and the runtime contract
  before the first run.
- Every figure regenerated from logs with a `*.provenance.json`; no hand-drawn behaviour
  claims. Final artifacts promote to `paper_artifacts/`.
- GT firewall: `GT — evaluation only` labels; data-source tags (`BELIEF`/`PIXEL`/`MODEL`)
  on every figure and video.
- Video package (plan §8): V00 motivation · V01 data collection · V03 GP explainer ·
  V06 three-condition navigation (main) · V08 4-cam coverage · V09 handover · V09C dropout ·
  V10 active commissioning (as scoped).

## Freeze checklist

- [ ] ch.01 covariance gate PASSED and documented
- [ ] ch.02 trust target FROZEN (choice + Fig 02D archived)
- [ ] ch.03 U0–U6 comparison complete, route-disjoint (claim or honest null stated)
- [ ] ch.05 R_plan interface FROZEN (r_miss_uv reconciled, one implementation)
- [ ] ch.06 N-campaign run + Table 06C
- [ ] ch.08 per-camera transfer on `warehouse_full_4cam` with zero retuning
- [ ] second contribution chosen (A: ch.04 / B: ch.09) and its own gates passed
      (for B: D2 pass + closed-loop handover campaign)
- [ ] every chapter's `evidence.yaml` points at final artifacts
- [ ] thesis document updated from this package (`../../../thesis-report/` remains the
      historical submitted paper)
