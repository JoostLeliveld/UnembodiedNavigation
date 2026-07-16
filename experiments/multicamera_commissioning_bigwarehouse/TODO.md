# Implementation Plan — Full-Warehouse Four-Camera Commissioning

## Scope lock

- **World:** `warehouse_full_4cam.world.sdf` (24.5 × 20.5 m).
- **Cameras:** A–D on the four wall dock columns (`external_camera`, `_b`,
  `_c`, `_d`). The presentation-only overview camera is excluded from this
  study.
- **Core result:** reliability-aware selection and hysteretic handover. Fusion
  is optional and only follows a successful selection/handover result.
- **Planner boundary:** output one selected map observation and its covariance;
  do not change obstacle handling, no-go geometry, control, or the planner
  objective.
- **Data boundary:** manager inputs are operational detector/projection/age/
  trust/association/consistency fields. Ground truth is evaluation-only.

## Foundation already present

| Item | Status | Use in this study |
| --- | --- | --- |
| `warehouse_full_4cam.world.sdf` + A–D detector launch | Implemented | Canonical scale environment and isolated camera streams. |
| Camera contracts, health state, export/replay split, leakage firewall | Implemented | Keeps operational decisions separate from evaluation labels. |
| Per-camera geometry prior and independent reliability learner | Implemented | Day-zero maps followed by one fitted field per camera. |
| Overlap validation and conservative selection baselines | Implemented | D2 and S0–S4 comparison interface. |
| Handover covariance inflation and NIS gate | Implemented | Downstream uncertainty treatment after source change. |
| Stateful reliability-aware manager | Implemented | M8 gates and hysteresis; no evaluation-field access. |
| Per-camera fitted GP | Pilot only | Four operational posteriors exist; route-disjoint validation is still required. |
| Frozen paper protocol + campaign audit | Implemented | `config/paper_protocol.yaml` and `tools/paper_campaign.py` materialise the matrix and show unmet gates. |
| Propagated noisy-odometry covariance | Implemented | New recordings log a rotated process covariance; legacy data use an explicit fallback. |

## Ordered work

### 1. Design full-warehouse collection routes — complete; freeze them

- [x] Add routes for A, B, C, and D single-source regions.
- [x] Add adjacent-camera overlap corridors A/C, B/D, and C/D.
- [x] Add both central-aisle travel directions.
- [x] Set documented spawn poses, offsets, speeds, and repeat counts.
- [ ] Run `paper_campaign.py` and collect every missing frozen plan row; do
  not retrospectively edit the protocol to fit pilot results.

**Acceptance gate:** every frozen route is valid in the current world and
appears in `config/study.yaml`; no route is copied from a retired testbed.

### 2. Commission every camera independently (D0/D1) — data-dependent

- [ ] Collect baseline and repeated routes at two speeds and lateral offsets.
- [ ] Export A–D CSVs to a shared replay timeline with operational/evaluation
  separation intact.
- [ ] Audit confidence, detection rate, NIS, spatial error, and stale frames by
  camera.
- [ ] Fit and hold out one spatial trust model per camera by complete route;
  use `fit_belief_aware_gp.py --holdout-run-id ...` rather than random frames.

**Primary evidence:** held-out NLL/MAE/calibration per camera and false-high-
trust rate. No per-camera map is a result until this evidence exists.

### 3. Commission the overlap graph (D2) — data-dependent

- [ ] Record synchronised, valid projected observations in each adjacent-camera
  corridor.
- [ ] Separate geometric FOV overlap, detector overlap, and reliable overlap.
- [ ] Persist an edge with count, median/p90 disagreement, bias, time offset,
  coverage, and validation date.
- [ ] Reject handover/fusion claims when pair count or consistency gates fail.

**Acceptance gate:** at least 30 held-out synchronised pairs per claimed edge,
at most 10% outliers, and the configured disagreement threshold.

### 4. Evaluate selection and handover (D3/D4) — blocked on D0–D2 data

- [ ] Compare fixed preferred, detector-score, fixed-zone, static-trust,
  spatial-trust, uncertainty-aware, and evaluation-only oracle policies.
- [ ] Compare direct switching, trust-only switching, hysteresis, and trust +
  consistency + hysteresis.
- [ ] Report source switches, oscillations, observation gaps, NIS/NEES around
  handovers, covariance spikes, and recovery time from held-out data.

**Decision rule:** retain the smallest policy that improves both selection
quality and handover stability without worsening calibration.

### 5. Failure robustness (D5) — blocked on commissioned maps

- [ ] Replay controlled camera drop-out, staleness, quality degradation,
  temporary occlusion, and injected cross-camera bias.
- [ ] Compare fixed preferred, score-only, manager, and manager + fallback.
- [ ] Verify degraded input never makes the manager more confident.

### 6. Shadow mode, then active handover — blocked on D2–D5

- [ ] Log manager decisions alongside the existing estimator without authority.
- [ ] Validate replay and shadow decisions on held-out runs.
- [ ] Enable an opt-in estimator correction only after the gates pass.

### 7. Conservative overlap fusion — optional, last

- [ ] Compare best-camera selection, sequential updates, individual NIS gates,
  pairwise consistency, and covariance intersection only if justified.
- [ ] Never average pixel covariances or double-count correlated evidence.

## Historical context

An archived two-camera pilot failed its overlap gate honestly. It is useful only
as a protocol lesson: enough synchronous pairs and calibrated disagreement are
preconditions for a handover claim. Re-collect all measurements in the current
four-camera world.
