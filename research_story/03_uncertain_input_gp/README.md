# 03 — Learning trust at uncertain robot locations (Contribution 1 core)

**Question.** When trust observations are deposited at *estimated* robot positions with
covariance P⁻, does explicitly modelling that input uncertainty (expected-kernel GP) produce
better-calibrated trust maps than the cheap alternatives — a point GP, a longer length
scale, Gaussian smoothing, or covariance-weighting?

**Status: ACTIVE** (blocked on the ch.01 gate). Order: synthetic diagnostic → original
warehouse proof → `warehouse_full_4cam` only as frozen-method generalisation (ch.08).

## What the contribution looks like

> *Modelling the uncertain spatial locations of camera-reliability observations improves
> held-out trust calibration when odometry uncertainty is significant.*

Defensible when shown on **route-disjoint** splits against ALL of U0–U4, with U6 (GT
positions) as an evaluation-only ceiling. The honest null is equally publishable and must be
stated up front:

> *Under this warehouse's uncertainty regime, a simple smoother performs equivalently — the
> expected-kernel treatment is unnecessary.*

(We already have a hint: methods **tie at real σ**. The claim therefore lives or dies in the
uncertainty-scaling regime — and in how honestly we report where the crossover sits.)

## Baselines (frozen IDs — use everywhere)

| ID | Method | Exists? |
|---|---|---|
| U0 | Global constant reliability | trivial |
| U1 | Point-input GP | `fit_belief_aware_gp.py` supports |
| U2 | Point-input GP, larger learned length scale | config of U1 |
| U3 | Gaussian spatial smoothing | to add (small) |
| U4 | Covariance-weighted point-input GP | supported (weighting path) |
| U5 | Uncertain-input expected-kernel GP | **implemented** — the canonical module |
| U6 | GT-position GP | evaluation-only ceiling |

## The results we're aiming for

- **Fig 03A** (1-D explainer) + **Fig 03B** (2-D anisotropic synthetic: true field, uncertain
  points with ellipses, U1 vs U3 vs U5 + posterior uncertainty). The visual question: does
  U5 differ *meaningfully* from blur? exp1's harness extends to this panel.
- **Fig 03C/03D** — AWS training observations and the four trust maps side by side.
- **Fig 03E (the decision figure)** — held-out NLL/Brier vs α for P^(α) = αP, α ∈
  {0, 0.5, 1, 2, 4, 8}. **Aim: U5's curve degrades most gracefully; the α where U5 separates
  from U1/U3 is reported as the operating-regime statement.**
- **Fig 03F** — performance conditioned on time-since-camera-update / tr(P) / odometry-only
  distance: shows *where* in a run the method earns its keep.
- **V03** — construction animation (same data, three GP assumptions).

## Implemented now

| Item | Tag | Note |
|---|---|---|
| Expected-kernel / belief-aware GP (`fit_belief_aware_gp.py`) | established (code) | THE canonical module — import, never reimplement |
| First belief-aware fits (`belief_aware_gp_score_v1`) | measured_in_sim | pre-gate; treat as pipeline proof |
| Synthetic harness (exp1: setup/prediction-maps/metrics figs) | model_plumbing | extend to the U1/U3/U5 panel |
| Training events (`belief_gp_events`) | measured_in_sim | usable only after ch.01 prior-pairing audit |
| Finding: methods tie at real σ; single-route training = false-confidence trap | measured_in_sim | dictates the α-sweep design + route-disjoint rule |

## Gap → next experiment

New study `experiments/uncertain_input_gp/` (outputs `logs/studies/uncertain_input_gp/`):
U0–U6 grid on synthetic (03A/03B), then on audited AWS events (03C–03F), metrics from
`scripts/shared/metrics.py` only.

## Gate

Route-disjoint held-out NLL/Brier across the α-sweep decides between the contribution claim
and the honest null. Trust target is frozen by ch.02 — no target tuning here.
