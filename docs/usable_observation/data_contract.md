# Usable-observation data contract (P1)

**Status:** CURRENT · **Schema:** `obs_opportunity_v1` · **Frozen gate:** `usable_observation_gate_v1`
**Code:** `src/reliability/reliability/observation_opportunity.py`, `observation_gates.py`
**Config:** `config/usable_observation_gate.yaml` · **Schema file:** `schemas/observation_opportunity.schema.json`
**Tests:** `tests/observability/test_observation_gates.py` (23 cases) · **Gate:** Gate 1 (see below)

This contract defines one record per camera per synchronized observation opportunity —
**including misses** — and the frozen labels the observability models are trained on. It
replaces, for the p_use refocus, the miss-dropping `opportunity.build_opportunity_row`
(preserved unchanged as the historical/diagnostic path).

## 1. Claim

A single deterministic function `evaluate_observation_opportunity(raw_record, gate_config)`
turns an operational (never-GT) raw opportunity into an `ObservationOpportunity` carrying:

```
detection_label = 1  iff the detector produced a robot-class detection candidate (a box)
                      for that opportunity — independent of the confidence threshold.
quality_label   = 1  iff detection_label == 1 AND all enabled frozen quality checks pass.
usable_label    = detection_label AND quality_label.
failure_reason  = the single earliest failing gate (controlled enum), or USABLE.
```

Therefore, per camera `c` and state `s`:
`p_det,c(s)=E[detection_label]`, `p_qual,c(s)=E[quality_label | detection_label=1]`,
`p_use,c(s)=E[usable_label]=p_det·p_qual`.

## 2. Realistic assumptions

- Inputs are operational only: PIXEL (detector/bbox/image), STATE/BELIEF/ODOM (pose),
  camera health/frame liveness, projection/association/track/localizer validity.
- The confidence threshold is a **quality** check, not part of `detection_label`. This makes
  the A5/A6 ablation ("quality gate without vs with confidence") a config flip.
- Frame liveness (`NO_FRAME`/`STALE_FRAME`) is treated as a detection-availability
  precondition (health axis); the exporter may optionally exclude these from a *spatial*
  `p_det` fit and account for them on the health axis instead (documented per dataset).

## 3. Non-assumptions

- No claim that confidence is measurement covariance (see `confidence_analysis.md`, pending).
- No Gazebo ground truth as a feature, label, pose, or gate input. `state_source == GT` and any
  `gt_*`/`eval_*`/oracle key raise `LeakageError` at gate entry and at contract construction.
- `quality_label` is conditional on `detection_label`; a miss is never a quality negative — it
  is a `detection_label = 0` record with a detection-stage `failure_reason`.

## 4. The frozen gate (order = priority; earliest failure wins)

| # | Stage | Check (enabled by) | Threshold | `failure_reason` | Justification |
|---|---|---|---|---|---|
| 1 | detection | frame expected & received | — | `NO_FRAME` | no frame ⇒ no possible detection |
| 2 | detection | frame fresh (`check_frame_freshness`) | `max_frame_age_ms=500` | `STALE_FRAME` | matches `OpportunityConfig.max_measurement_age_s=0.5` |
| 3 | detection | detection candidate exists | — | `NO_DETECTION` | `detection_label` denominator event |
| 4 | quality | confidence ≥ threshold (`check_confidence`) | `0.25` | `LOW_CONFIDENCE` | node default (`yolo_robot_detector_node.py:61`); frozen per §Freeze |
| 5 | quality | class/id match (`check_class`) | `expected_class` | `WRONG_CLASS_OR_ID` | off today (single-class detector) |
| 6 | quality | valid projection (`check_projection`) | — | `INVALID_PROJECTION` | `OperationalReliabilitySample.projection_valid` |
| 7 | quality | valid association (`check_association`) | — | `INVALID_ASSOCIATION` | association-window validity |
| 8 | quality | valid track (`require_track`) | — | `INVALID_TRACK` | off today (no runtime tracker) |
| 9 | quality | not edge-clipped (`check_edge_clip`) | `min_edge_distance_px=4` | `CLIPPED_OR_EDGE` | selected pixel must sit ≥4 px inside the image |
| 10 | quality | localizer accepts (`check_localizer_accept`) | — | `REJECTED_BY_LOCALIZER` | final `covariance_mapping.gate_decision` accept |
| — | — | all enabled checks pass | — | `USABLE` | `usable_label = 1` |
| — | — | an enabled check lacks its input | — | `UNKNOWN` | never coerced to a negative; exporter counts/drops |

Selected pixel defaults to the **bottom-centre** of the bbox (localizer convention) when not
supplied; `edge_distance_px = min(u, v, W−u, H−v)`.

## 5. `ObservationOpportunity` fields

Identity/context: `timestamp, run_id, route_id, camera_id`. State (never GT):
`state_x, state_y, state_yaw, state_source∈{ODOM,BELIEF,STATE}, state_covariance=(xx,xy,yy)|null`.
Frame: `frame_expected, frame_received, frame_age_ms`. Detection:
`detection_received, detector_class, detector_confidence, confidence_threshold`. BBox/image:
`bbox_{xmin,ymin,xmax,ymax,width,height,area,center_u,center_v}, selected_pixel_{u,v},
image_width, image_height, edge_distance_px`. Checks:
`projection_attempted/valid, association_attempted/valid, tracking_available/valid,
accepted_by_localizer`. Labels: `detection_label, quality_label, usable_label`.
Bookkeeping: `failure_reason, source_labels, schema_version`. Miss records carry `null` for
detector-conditional fields; labels and `failure_reason` are always populated.

Enforced invariants (contract + JSON schema): `usable_label == detection_label & quality_label`;
`quality_label==1 ⇒ detection_label==1`; `failure_reason==USABLE ⇔ usable_label==1`.

## 6. The frozen confidence threshold

`confidence_threshold = 0.25` (node default). Datasets **re-gate detections offline** from the
logged `raw_score`; this is lossless only where the capture threshold was ≤ 0.25. The 4-cam
commissioning launches captured at 0.05, so those logs *can* be re-gated to 0.25; single-cam
capture at 0.25 cannot recover sub-0.25 boxes and is used as-is. The chosen value is frozen via
`gate_id`; changing it requires a new `gate_id` and re-export (`config_hash` changes).

## 7. Gate 1 (validation gate — PASSED)

- Misses are represented (one record per opportunity, `detection_label` may be 0). ✅
- Every camera opportunity yields a record with a controlled `failure_reason`. ✅
- Labels are deterministic from the frozen config (`config_hash=45b81578…` for the shipped YAML). ✅
- No GT leakage: `state_source=GT` and eval-only keys raise `LeakageError`. ✅
- Schema tests pass: gate output validates against `observation_opportunity.schema.json`;
  inconsistent labels are rejected by both the contract and the schema. ✅
- 23 unit tests cover every `failure_reason`, priority ordering, the A5/A6 ablation, `UNKNOWN`
  handling, firewall, roundtrip, and config hashing. ✅

## 8. Baselines / caveats

- This is the *contract*, not a model. Baselines and models are P3/P4.
- `UNKNOWN` records depend on which checks a given log can populate — the exporter (P2) must
  report the `UNKNOWN` count per camera/route and justify any disabled check.
- Frame-liveness rows are health-axis; their treatment in the spatial fit is a P2 decision.

## 9. Reproduction

```
python3 -m pytest tests/observability/ -q
python3 -c "import json,jsonschema; jsonschema.Draft202012Validator.check_schema(json.load(open('schemas/observation_opportunity.schema.json')))"
```

## 10. Evidence status

CURRENT (contract + tests). No experimental claim is made here; dataset and model evidence
follow in P2–P4.
