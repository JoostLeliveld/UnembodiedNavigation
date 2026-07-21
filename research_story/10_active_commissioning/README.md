# 10 — Active commissioning: drive less, learn the same map (future paper)

**Question.** Can the robot learn the four camera trust fields of
`warehouse_full_4cam` with less driving by choosing informative-but-safe commissioning
routes — especially when route value is discounted by the *pose uncertainty the robot will
have while driving it*?

**Status: FUTURE — starts only after ch.03's model and ch.08's per-camera maps exist.**
World: `warehouse_full_4cam` (its 4-camera coverage-allocation problem is what makes the
question non-trivial; an AWS-world run is an implementation sanity test only).

## What the contribution looks like

> *Safe informative route selection for efficient commissioning of camera-reliability
> fields: pose-uncertainty-aware planning reaches a specified map quality with less driving
> than ordinary or uniform coverage.*

This is a different claim from *using* the trust map (Contribution 1) — hence a separate
paper, not thesis scope unless everything earlier finishes early.

## Conditions (frozen IDs)

A0 ordinary task routes · A1 uniform raster · A2 random safe · A3 max GP posterior variance ·
A4 information gain (pose-certain) · **A5 information gain under predicted pose uncertainty**.
The A4-vs-A5 gap is the thesis-specific hypothesis: sampling value depends on how well the
robot will know *where* it sampled.

## The results we're aiming for

- **Fig 10B (decision figure)** — held-out NLL/Brier vs driven metres / minutes / samples.
  **Aim: A5 reaches the target map quality with the least driving; A4 overrates
  camera-poor-approach routes whose samples land at uncertain positions.**
- **Fig 10A** equal-budget route portraits · **Fig 10C** per-camera integrated posterior
  variance over time · **Fig 10D** sample redundancy · **Fig 10E** safety & cost (clearance,
  duration, compute, fallbacks) · **V10** side-by-side commissioning video.

## Implemented now

Nothing, by design. Adjacent reusables when it opens: ch.03's GP posterior (variance/info
gain), ch.01's belief prediction for A5, the ch.08 collection stack and routes, and
`experiments/optionA_commissioning/`'s metrics/event tooling (its offline data-value and
init-budget studies are cousins — reuse tooling, not conclusions).

## Gate

Blocked until ch.03 + ch.08 deliver. Route policies must respect the same safety envelope as
task driving; a commissioning route that needs a safety exemption is disqualified.
