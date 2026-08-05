# Paper framings — multi-camera handover & fusion

[Back to module 07](../README.md)

Candidate framings of our own paper centered on this contribution. Each framing
is a way of telling the same evidence; we keep more than one until the pilot
data say which lands best.

## Framing A — *Spatial and Instantaneous Reliability-Aware Multi-Camera Fusion and Planning for Warehouse Robots*
The current working framing: static per-camera calibration is insufficient
because camera usefulness varies with position, individual detections,
availability and calibration health; we combine a spatial GP prior, calibrated
frame-level evidence and online health monitoring into camera-specific
covariance for fusion **and** closed-loop planning.

Plan of record (research questions, hypotheses, contribution claim, experiment
campaign E0–E8, statistics, beats-Toro criteria) lives in the owning study:
- [ROADMAP](../../../experiments/multicamera_fusion_extension/plans/ROADMAP.md)
- [Per-module plans 01–12](../../../experiments/multicamera_fusion_extension/plans/)

**Results section (working draft):** [`RESULTS_SECTION_DRAFT.md`](RESULTS_SECTION_DRAFT.md)
— R0–R2 real/measured (E0 calibration validated on all four cameras; throughput;
WP5 fault-detection with zero false alarms), R3–R6 pre-registered and pending the real
handover capture. Reframed around **fault containment** (overlap is only 7–13%).

**ICRA positioning (framing-of-record):** [`ICRA_FRAMING_2026-07-22.md`](ICRA_FRAMING_2026-07-22.md)
— the single source of truth for the paper-level story. Under the three locked decisions (moderate
runway · Gazebo-only + Toro-framed · NOT "GP beats calibration"), the headline was **broadened
(2026-07-22, reason-backed) from a fault-tolerance PoC to "safe navigation on a realistic
infrastructure-camera network"** — coverage- and health-aware localization for a no-onboard-fallback
robot, generalizing IWAI; fault-tolerance is now one pillar (C3). It holds the problem statement, the
C1–C4 claim block, the nulls-as-scoping discipline, and a **two-tier structure**:
- **Tier 1 (headline)** — single-camera closed-loop core + the safe-operating-envelope
  characterization. Contract: [`PREREGISTRATION_realistic_network_2026-07-22`](../../../experiments/multicamera_fusion_extension/PREREGISTRATION_realistic_network_2026-07-22.md).
- **Tier 2 (generalization, cuttable)** — multi-camera fusion/containment. Contract:
  [`PREREGISTRATION_fault_containment_2026-07-22`](../../../experiments/multicamera_fusion_extension/PREREGISTRATION_fault_containment_2026-07-22.md)
  (subsumed as Tier 2). Start here for the story; the results draft above is its evidence layer.

**Addendum (2026-08-04):** [`ICRA_FRAMING_ADDENDUM_2026-08-04.md`](ICRA_FRAMING_ADDENDUM_2026-08-04.md)
— the framing-of-record **stands**; the 2026-07-30 narrowing to *"the hit/miss mixture changes
route choice"* is **retired as a headline** (`R_cond` was never data-blocked, it was bias-blocked;
per-camera `R_cond` only ties a pooled constant). Adds **C5 — the dominant error is per-camera
*systematic* bias, and correcting it is a gated, GT-free, per-camera decision** (NEES 8.51 → 1.06
on a held-out capture; ungated the same fix harms camera A by 27 mm), and one new null: **per-camera
calibration cannot be fitted for every camera — correct the outliers, leave the rest raw.** The
mixture is demoted to a belief-propagation *correctness* argument under C1. Ranks the three
realistic-warehouse exposures, with **calibration drift** as the recommended next experiment.

**New-world role (2026-08-04):**
[`MEERHOVEN_PAPER_ROLE_2026-08-04.md`](MEERHOVEN_PAPER_ROLE_2026-08-04.md) — Meerhoven is a
second, exploratory external-validity test of the assistive localization service contract,
not a replacement for the compact AWS/B1 paper core and not a “twelve beats four” claim.
Promotion requires its complete detector → calibration → GP → seeded-run provenance chain.

> Add a sibling section here when a second framing is drafted (e.g. a
> localization-only framing that drops the planning claim, or a
> calibration-health-centred framing). Keep each framing's evidence pointers
> honest about what is demonstrated vs. blocked on data.
