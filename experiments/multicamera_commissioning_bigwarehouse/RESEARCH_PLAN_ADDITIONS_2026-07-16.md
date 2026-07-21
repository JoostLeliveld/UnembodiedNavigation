# Modularized research plan — four-camera additions (2026-07-16)

Scope: everything between today's state (GT tooling, projection calibration v2,
shadow manager node, per-camera device assignment, frozen `paper_protocol.yaml`)
and defensible paper claims for research_story ch.08/09/10.

**Dependency spine (order is load-bearing):**
detector retrain → projection re-calibration → trust maps (D0/D1) → overlap
graph (D2) → replay policies (D3/D4) → robustness (D5) → shadow → gated active.
A detector change alters box-bottom statistics, which alters the projection
calibration, which alters every projected coordinate feeding the trust maps and
overlap edges. Nothing fitted downstream survives a change upstream.

**Two-world rule position (state this in the thesis):** the frozen METHODS
(planner, trust pipeline, gates, hysteresis) were developed in `warehouse_aws`
and stay frozen. Per-site perception commissioning — detector fine-tuning for
the new camera viewpoints and per-camera projection calibration — is part of
the pre-registered deployment procedure (ch.09/10), exactly like the BEV affine
calibration in the single-camera work. It is NOT method tuning; no gate,
weight, or policy parameter may be adjusted from 4-cam data.

---

## Module 1 — Detector adaptation to the four-camera world (YOLO retrain)

**Why (measured, 2026-07-16):** with `warehouse_yolo_detector_v1` (trained in
warehouse_aws at camera z≈4.5–4.8 m, imgsz 960) the 4-cam world yields
detection rates 0.10–0.70 and mean raw scores 0.32–0.81 depending on camera —
at z=6.10 m, 8–16 m range, and launch-default **imgsz 640**. Detector scores
feed `availability_probability` → trust targets → the manager score; the
pilot's 0.41 trust ceiling below the 0.45 release threshold is plausibly
detector-OOD at root. Camera C's ~0.11 m cross-bearing projection residual
(occlusion-clipped boxes near the central pillar) is also detector territory.

1a. **Config audit before any training** (cheap, do first)
   - Rerun one collection pass at `yolo_imgsz:=960` (launch currently defaults
     640; the aws campaign standard is 960). Record per-camera detection
     rate/score/latency deltas. If 960 alone fixes rates, retraining shrinks
     to fine-tuning for score calibration.
   - Audit `camera_observation_r_visible_uv/r_miss_uv` (launch 2.5/40 px) vs
     the 120 px runtime default in `unicycle_planner_node.py` — reconcile
     BEFORE any R_plan numbers are quoted (known documented mismatch).
   - Gate: a one-page table (camera × imgsz × device → rate, score, ms/frame).

1b. **Training data capture (per-camera, occlusion-gated)**
   - Reuse the proven clean-retrain recipe (occlusion-gated segmentation
     capture + analytic labels — the method that fixed the contaminated-dataset
     saga), but capture from ALL FOUR 4-cam viewpoints: teleport sweeps over
     the drivable area, episodic crate randomization, both robot headings,
     range-stratified (5–16 m) so long-range examples aren't underrepresented.
   - Include occlusion-truncated silhouettes near the pillar/racks explicitly
     (labels from analytic geometry, not hand annotation) — this targets the
     camera-C clipped-box failure.
   - Leakage rules: no evaluation-route poses in training captures; dedup by
     pose grid (the 70%-duplicate contamination lesson); persist a capture
     manifest with pose provenance.
   - Gate: dataset card (counts per camera × range bin × occlusion state,
     dedup rate) committed before training.

1c. **Retrain + evaluate**
   - Fine-tune v1 → the locked successor `warehouse_yolo_detector_4cam_v2_640`
     at imgsz 640 (P2000 4 GB; batch 4 + AMP). The successor is frozen before
     D1/D2 evidence and supersedes the unresolved 960 candidate.
   - Acceptance gates (per camera, held-out capture): detection rate ≥0.9 at
     ≤12 m and ≥0.75 at 12–16 m on unoccluded poses; box-bottom localization
     audited vs analytic truth (flat over image periphery — the 0.027 m
     standard); score distribution documented (feeds Module 3 targets).
   - The deployment-size decision is already locked in the versioned successor;
     if scores remain poorly calibrated, keep the detection-RATE-based trust
     methodology (calibration-invariant, already the campaign standard) and
     record the disclosure.

> **Addendum (2026-07-16, later):** the parallel workstream measured that one
> four-image batched GPU call (single shared model) runs 38.9 ms vs 79.5 ms
> for four sequential calls — this supersedes the camera_A-on-CPU mitigation
> in 1d AND the OOM constraint (one model load instead of four). It also
> added a geometry-certified **negative-frame contract** (deployment usually
> has no robot in three of four views; positives-only training would let a
> rack false-alarm read as availability) plus per-camera capture gates
> (reserved-route/collision/range/split/occlusion, localhost-only transport).
> Module 1b/1d below should be read through that lens; do not duplicate.

1d. **Latency vs the age gate (CPU camera reality check)**
   - camera_A on CPU currently delivers ~1 Hz images; the manager's
     `max_measurement_age_s = 0.15` will mark a 1 Hz stream stale almost
     always → A can never be selected live even with perfect trust. Resolve
     BEFORE shadow claims: measure per-device end-to-end contract age at
     imgsz {640, 960}; then either (i) a smaller/quantized model for the CPU
     camera meeting 0.15 s, or (ii) pre-register a per-camera age gate with
     justification, or (iii) accept and document A as replay-only. No silent
     gate loosening.
   - **Invalidates on completion:** day-zero priors, all fitted 4-cam GPs,
     pilot showcase numbers, projection calibration v2. Mark those artifacts
     superseded in their manifests.

## Module 2 — Projection calibration, final version

- Re-fit `fit_projection_calibration.py` on the NEW detector's outputs:
  dedicated calibration passes (2–3 routes spanning 5–16 m per camera,
  including camera_A which currently has only a 1 m distance span → constant
  fallback), then QUALIFY on held-out routes — v2 was validated in-sample and
  says so.
- Investigate replacing box-bottom-centre with mask-derived bottom point
  (detector already supports masks) for the occlusion-clipping lateral error;
  keep whichever wins on held-out cross-bearing residual.
- Gates: held-out per-camera |bias| ≤ 0.05 m along AND cross bearing; C↔D,
  A↔C, B↔D synchronized disagreement mean ≤ 0.10 m. Freeze as
  `projection_calibration_v3` with fit provenance in the JSON.

## Module 3 — Per-camera trust maps (D0/D1, route-disjoint)

- Execute `paper_protocol.yaml` route-disjoint mapping: train routes = four
  single-camera passes, held-out = both handover traverses; 5 repeats × 3
  lateral offsets × 2 speeds, paired seeds 0–19.
- Fit all four GP modes (naive / uncertainty-weighted / belief-spread /
  expected-kernel) per camera via the canonical `fit_belief_aware_gp.py`;
  compare against constant, geometry-only prior, and pooled/shared
  alternatives.
- Metrics via `scripts/shared/metrics.py` ONLY (Brier, logloss, AUROC, ECE,
  false-high-trust rate) on held-out routes.
- Gate (pre-registered, TODO step 2): "no per-camera map is a result until
  this evidence exists." Also record where the trust ceiling lands relative
  to 0.45 — with the retrained detector this is the test of the
  "OOD-suppressed trust" hypothesis.

## Module 4 — Overlap graph qualification (D2)

- All three declared edges (A↔C south, B↔D north, C↔D central), 10 repeats ×
  offsets × speeds per protocol; ≥30 HELD-OUT synchronized pairs per edge,
  ≤10% outliers, ≤0.30 m disagreement (frozen gates).
- Persist per edge: count, median/p90 disagreement, residual bias vector,
  time offset, spatial coverage, validation date — the bias term now has a
  measured meaning (post-calibration residual).
- Run GT-attached (`record_evaluation_truth.py` alongside every pass) so any
  gate failure is attributable same-day.

## Module 5 — Replay policy evaluation (D3/D4) + threshold provenance

- Replay matrix from the protocol: R0–R4 baselines, M5–M8 selection/handover
  policies, on matched exports and paired seeds; report the frozen metric
  list (RMSE, p95, NIS/NEES, coverage, false-high-trust, handover counts,
  unsafe handovers).
- Threshold: the 0.45 constant remains asserted. Either (i) derive a release
  threshold from warehouse_aws-era evidence (e.g., trust level bounding
  false-high-trust ≤ target on aws data) and pre-register the derivation, or
  (ii) keep 0.45 explicitly as a convention and ALWAYS publish the
  sensitivity curve next to release/no-release claims (sweep tooling exists).
  Decide before running, not after seeing results.
- Positive control REQUIRED: at least one regime where the winning policy
  releases corrections and NIS/NEES stay sane — a 0-corrections-everywhere
  result cannot support a handover claim.

## Module 6 — Robustness suite (D5)

- One factor at a time per protocol: low_light, camera_latency,
  camera_dropout, odometry_stress, then combined_shift; paired seeds.
- Assertion to verify explicitly: degraded input never INCREASES manager
  confidence (monotonicity check per corruption).
- The camera-A CPU path is itself a latency condition — fold Module 1d's
  measurements in rather than simulating latency for that camera.

## Module 7 — Shadow mode on held-out runs

- `camera_manager_node` (authority=shadow) already reproduces offline replay
  on one run (292/292 decisions). Scale the claim: shadow vs offline replay
  decision agreement ≥99% across ALL held-out D3/D4 runs, plus decision-rate
  and age-distribution stats. This is the ch.09 step-6 evidence.
- Zero method changes permitted in this module; it is measurement only.

## Module 8 — Gated active handover (closed loop)

- Entry condition = ALL protocol release gates green (mapping, overlap,
  covariance, replay). Then `manager_authority:=active` on tasks T1–T5 with
  the frozen policy; paired against the passive baseline with identical seeds.
- Report safety-framed (thesis claim axis): stop-safe vs collision outcomes,
  NEES around handovers, covariance spikes, recovery time. The planner stack
  must run the frozen aws configuration (goal/no-go/weights untouched).
- Any gate failure ⇒ report the failure, do not iterate the method.

## Module 9 — Paper assembly & disclosure register

- Claims map: ch.08 (scaling) ← Modules 3–4; ch.09 (handover/fusion) ←
  Modules 5–8; ch.10 (active commissioning) ← the commissioning procedure
  itself (GT tools, calibration fitting, gate provenance) as a contribution.
- Disclosure register (already started in GATE_PROVENANCE.md): asserted vs
  derived constants, in-sample vs held-out validations, camera-C residual,
  CPU-camera constraint, threshold sensitivity curve, superseded artifacts.
- Figures from FIGURE_BACKLOG 09A–09F; every figure with a provenance JSON.

## Cross-cutting: operations hygiene (bit us twice today)

- Before every sim session: `pgrep -f drive_study_route` and kill stale
  drivers (yesterday's pilot drivers were still steering /cmd_vel a day
  later); one run-manifest per capture; recorders and GT recorder started
  together; commit configs before collecting.
- GPU budget: 3 GPU detectors max on the P2000; camera_A CPU (or the Module
  1d resolution); no training jobs concurrent with collection.
- Coordination: the paper-protocol/replay workstream is active in parallel —
  new files or disjoint edits only, commit early, never commit their
  in-flight files.

**Rough effort (sim-hours dominate):** M1 ≈ 2–3 days (capture+train+eval),
M2 ≈ ½ day, M3 ≈ 2 days of collection + fits, M4 ≈ 1 day, M5 ≈ 1 day
(mostly replay compute), M6 ≈ 1–2 days, M7 ≈ ½ day, M8 ≈ 1 day, M9 ≈ 2 days.
Critical path M1→M2→M3→M4→M5; M6 can interleave after M3.
