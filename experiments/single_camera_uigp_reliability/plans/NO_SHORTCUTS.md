# NO SHORTCUTS — the evidence-integrity rule for this study

Written 2026-07-21 on explicit user instruction: *"stop taking the easy way out
on old data or synthetic data."* This is the standing rule for every experiment
in this study and its outputs.

## The rule

**Reuse code. Run NEW experiments on GOOD data.**

- ✅ Reuse `fit_belief_aware_gp.py`, `reliability.*` modules, `scripts/shared/metrics.py`,
  `campaign_metrics.py`, `ObliqueCameraModel`, `reliability.covariance_mapping`.
  Reimplementing any of these is a bug, not progress.
- ❌ Do NOT re-fit the existing `logs/visibility_comparison/belief_aware_gp_score_v1`
  or `belief_gp_events` captures and present the numbers as this study's result.
  Those are the OLD single-external-camera evidence. A NEW experiment means a
  NEW `warehouse_aws` capture driven for this study's route/seed design.
- ❌ Do NOT substitute a synthetic fixture for a run that has not happened.

## Synthetic is allowed in exactly one place

Experiment A (controlled covariance-fidelity sweep) MAY inject controlled noise
`Σ_reported = γ·Σ_true` to prove the uncertain-input mechanism behaves correctly
when the true input uncertainty is known. It must be:

- labelled **"controlled ablation, not operational evidence"** in its RESULTS.md;
- built on top of a REAL `warehouse_aws` trajectory + detector run wherever
  possible (the proposal permits fully synthetic, but the real-trajectory
  version is preferred and is what we run);
- never the paper headline and never a stand-in for Experiment B.

The discriminating, reportable result is Experiment B on real operational data
with route-disjoint held-out splits. `research_story` ch.03 already records why:
*"methods tie at real σ; alpha-sweep is the discriminating regime"* and
*"single-route training = false-confidence trap; route-disjoint splits mandatory."*

## Ground-truth firewall (unchanged)

GT / `gt_*` / oracle labels and CAD shelf geometry are **evaluation-only**. They
may score a result; they may never be a GP training input or an online trust
signal. Every new export reader gets a firewall test.

## When something fails

Say so immediately and report the negative/real result. A negative real result
beats a positive fake one. Do not paper over a dead camera, a failed gate, or a
capture that did not run.
