# Next-paper package: self-commissioning observability

## Decision

The strongest next paper is **not** “a bigger version of the previous
benchmark” or “more cameras improve coverage.”  It is:

> **How can external-camera navigation self-commission a conservative
> observability model for an unseen or changed warehouse, then refine it safely
> from normal driving?**

The submitted paper established the downstream half of this chain: a
planner-facing reliability field can improve robust route choice.  The new work
has uncovered the missing upstream half: the old field is expensive to collect,
is tied to one layout, and becomes stale when the scene changes.  Camera
calibration and sensed structure provide a day-zero prior; detector hits and
misses collected while driving correct that prior over time.

This directory is the canonical, presentation-ready boundary between the
completed paper and exploratory extensions.  It makes no claim that the
extensions are already a completed paper result.

## Start here

| Item | Use |
| --- | --- |
| [Storyline](storyline.md) | Recommended research question, claim arc, hypotheses, and experiment design. |
| [Evidence ledger](evidence_ledger.md) | What is measured, what is a simulator/model demonstration, and what must still be validated. |
| [Media plan](media_plan.md) | Slide-by-slide figures, existing videos, spoken message, and regeneration source. |
| [Visual board](index.html) | A local, widescreen narrative board that embeds the curated plots and videos. Open it in a browser. |

## The one-sentence story

**A robot should not need an offline detection survey every time a warehouse
changes: start conservative from what the cameras can see, learn from driving,
and use the resulting confidence-aware map to keep localization dependable.**

## Scope discipline

- **Primary contribution:** single-camera day-zero prior plus online refinement.
- **Primary evaluation:** unseen-layout / layout-change generalisation,
  data-efficiency, calibration, and closed-loop navigation safety.
- **Scalability study:** multi-camera coverage and handover uncertainty.  It is
  a valuable testbed and optional ablation, not the paper headline until a
  real camera-B evidence chain and closed-loop handover campaign exist.
- **Not a contribution:** CAD/ground truth as a deployment input, synthetic
  depth as a real sensing result, or an offline map update presented as a live
  planner capability.

## Boundaries and provenance

The package only organizes documentation and presentation material.  It does
not alter the submitted TeX paper, the locked paper-facing campaign, or the
runtime planner.  Every quantitative statement is scoped in the
[evidence ledger](evidence_ledger.md) and linked back to its local artifact.
