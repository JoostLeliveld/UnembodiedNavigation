# 00 — The problem, and the result we already have

**Question.** Does spatially varying external-camera trust matter for navigation at all —
or is a constant camera covariance good enough?

**Status: LOCKED — this question is ANSWERED.** World: original warehouse
(`warehouse_aws.world.sdf`, one camera at (0, −5.5, 4.8)). Nothing here is re-run; the
thesis cites it.

## The answer we hold (honest_campaign_v1, 2026-07-01)

4 routes × 5 seeds × {C1 constant R, C2 GP-scaled R_plan}, GT bridge active, GT
evaluation-only:

| Condition | Clean goals | GT geometry breaches | Physics contacts |
|---|---:|---:|---:|
| C1 constant R | 15/20 | 4/20 (all on the camera-poor west route) | 0 |
| C2 reliability-aware R_plan | 20/20 | 0/20 | 0 |

Robustness companion `whitenoise_campaign_v1` (C2 20/20, C1 19/20, the one C1 failure SAFE):
the gap is not a process-noise artifact. Mechanism: C1 takes the short blind lane and
breaches the safety envelope; C2 pays a detour to stay observable.

## Why this opens the thesis rather than closing it

The GP behind C2 was fitted from a **dedicated survey with near-exact robot poses** — an
assumption no deployed warehouse satisfies. So the established result proves *downstream
value* while leaving the *upstream question* open:

> How can that trust map be learned from ordinary driving, where the robot's position is
> itself uncertain?

That question is chapters 01–06 and Contribution 1.

## Implemented now (all `established`)

- Runs: `logs/visibility_comparison/{honest_campaign_v1,whitenoise_campaign_v1}` (append-only).
- Figures: `paper_artifacts/figures/paired_mechanism_*` (+ provenance, `_data/` bundles);
  route GIFs in `paper_artifacts/figures/current_surface/` (real external-camera footage).
- Result pages: `docs/paper_vs_current/` (submitted-paper numbers stay quarantined in
  `paper/` — different contract, never mix).
- Frozen config + contract: `warehouse_visibility_campaign.yaml`,
  `docs/current_runtime_contract.yaml`. Stack description: `docs/contribution_map.md`.

## Remaining deliverables (presentation only)

Fig 00A problem map (camera FOV, good/poor regions, two candidate paths) — DERIVABLE ·
Fig 00B paired routes — DONE · Fig 00C outcome table — DONE (above) · V00 split-screen
constant-vs-learned video — PARTIAL (replay composites exist via `generate_run_replay.py`).

## Gate

Frozen. Any new numbers on these routes belong to chapter 06's N-campaign, not here.
