# 04 repair follow-up: frame identity and durable outcomes

2026-09-07. This implements the remaining **detector-side identity and outcome-transport** assignment following the user's “Fix the rest?” authorization. The original ranked findings and earlier repairs remain in [04_camera_acquisition_and_batching.md](04_camera_acquisition_and_batching.md). This is software evidence, not a navigation or camera-accuracy result.

**Implemented:** each received image, logical cycle and physical model call has an explicit identity; CameraObservation carries the member mapping; outcome records are journaled before transport. The detector model, thresholds, selected pixel, camera registry, all-camera schedule and existing receipt limits are unchanged. No experiment, mixed commit or push was started. The manager, contracts, logger and launch integrations were coordinated with owners07,10 and13; this work does not replace their acceptance evidence.

## Defect, reproduction and repair

The previous outcome publisher emitted JSON only to ROS and the process log. It had no independently durable record before publication, no immutable event sequence, and no retained QoS for late subscribers. Physical chunk IDs existed in diagnostic events, but CameraObservation carried only the whole-cycle source ID. A camera frame was distinguished solely by its capture timestamp.

The retained failing reproduction executes the real `_publish_batch_outcome` method with a failing publisher. It fails the invariant “journal append precedes transport”: [04_remaining_before.txt](04_remaining_before.txt). The repaired test verifies the actual JSONL file remains readable when publication raises.

Repairs:

1. `PendingFrame` adds `source_frame_id` and `content_sha256`. Receipt hashes image encoding, dimensions, stride and bytes. Equal camera/stamp/content remains a duplicate; unequal bytes at the same camera/stamp produce an explicit integrity failure. The hash does not alter or resize the image. Strict and asynchronous paths retain their existing scheduling rules; asynchronous replacement/expiry/duplicate outcomes are now recorded too, including the previously silent all-expired drain.
2. `_process_frames` retains `source_batch_id` as the logical cycle, creates an explicit member-to-invocation map, and journals selection and terminal outcomes. `_predict_batch` records intent before each model call, validates each returned chunk independently, and records completion or error. The compiled diagnostic path has one corresponding call identity. Graceful interruption records an error/abort with a nonempty reason, including an empty-message KeyboardInterrupt.
3. CameraObservation receives the four additive fields agreed with07: `producer_epoch`, `source_frame_id`, `capture_stamp_ns`, and `detector_invocation_id`. All are attached together. Legacy records remain readable by the updated contract parser; deploy the updated producer and consumers together. This work does not add fields to the separate strict MapObservation wire payload.
4. `OutcomeJournal` creates an exclusive JSONL file, serializes writes under a lock, calls `fsync` before returning, and assigns a stable event ID and sequence. The containing directory and newly created directory entries are synced. It refuses overwriting an existing journal. A partial write poisons the writer; a byte-cap failure stops processing instead of deleting old evidence. Earlier durable records remain intact.
5. Outcome publication uses reliable, transient-local, keep-last QoS with depth4096. The journal precedes publication; it remains the recovery source when DDS history expires, a subscriber starts too late, or publication itself fails. After executor quiescence, shutdown journals pending unselected images and a final `session_stopped` record without relying on live DDS.

## Identity contract

| Field | Current meaning | Compatibility boundary |
|---|---|---|
| `source_batch_id` | Whole logical detector cycle, unchanged current format `strict:<producer_epoch>:<cycle_sequence>:<camera@exact_ns,...>` | Remains the existing manager/correction join key. Do not reinterpret historical strings as chunk IDs. |
| `detector_invocation_id` | One actual model API call: `<source_batch_id>/chunk/<index>` | Native chunks2/2/1 have three IDs; compiled diagnostic call has one. The planned mapping appears at selection; completed/error records establish the observed call outcome. |
| `source_frame_id` | `frame:<producer_epoch>:<camera_id>:<capture_stamp_ns>:<content_sha256>` | This identifies received source content within a detector process epoch. It is not a hardware frame counter. |
| `capture_stamp_ns` | Exact integer from ROS/Gazebo header fields | Float observation seconds remain for timing compatibility. Equality of those floats is not event identity. |
| `producer_epoch` | Unique detector writer/process namespace | A manager outcome writer must use its own epoch, retaining the detector epoch in source/member metadata. |
| `event_id` | `<writer_producer_epoch>:<event_seq>` | Repeated transport of the same stored record keeps the same ID. A new append has a new event ID, even at an equal ROS timestamp. |

The image digest is SHA-256 over canonical JSON `{encoding,height,width,step}`, a newline, then raw image bytes. Encoding is lowercased; row padding remains part of the received byte identity. Camera and capture stamp are separate components of the frame ID. There is no inferred producer round sequence. Distinct physical captures with identical stamps and identical bytes cannot be distinguished by sensor_msgs/Image; the unique-monotonic-stamp requirement and explicit reset-stop policy remain. Timestamp-window grouping still depends on the declared capture cadence.

An inference `started` record is durable **intent immediately before the call**, not proof that execution occurred after a hard kill. An open intent is indeterminate. It must not be turned into a miss or a fabricated successful invocation.

## Outcome schema and transport

The schema is **`camera_batch_outcome.v2`**. Version1 records remain historical input; this producer emits version2. Authoritative journal fields are:

- `schema_version`, `producer_epoch`, `event_seq`, `event_id`;
- absolute `journal_path`, `previous_event_sha256`, `event_sha256`;
- caller `stage` (`detector` or `manager`), `status`, and contextual payload.

Detector cycle events retain `source_batch_id`. Their `members` contain camera ID, exact capture stamp, frame ID, content hash and receipt clocks. Once selected, members also include detector producer epoch and physical invocation ID. Invocation events use top-level `invocation_id` matching the member's `detector_invocation_id`. Failure records contain a nonempty reason. Member publication events distinguish hit/miss through `detection_valid`; errors never manufacture a false detector miss.

Main statuses:

| Stage | Records |
|---|---|
| Session/input | `session_started`, `image_received`, `image_duplicate`, `image_out_of_order`, `conflicting_duplicate_image`, `producer_error` |
| Buffer | `frame_replaced`, `stamp_skew`, `incomplete_timeout`, `incomplete_capacity`, `superseded_by_complete_batch`, `incomplete_shutdown` |
| Cycle | `selected`, `published`, `aborted`, `dropped_clock_wait` |
| Physical invocation | `inference_started`, `inference_completed`, `inference_error` |
| Camera publication | `member_publication_started`, `member_published`, `member_publication_error` |
| Quiescent close | `session_stopped` with `transport="journal_only"` |

A start intent includes `inference_scheduled_stamp_s`. Completion/error records include actual host-call start time; successful completion includes host return time and wall duration. `timing_semantics="host_model_call_start_return"` explicitly excludes journal/transport delay and does not claim CUDA kernel timing. Existing full-cycle inference timing remains full-cycle timing and now includes required per-call accounting overhead. `publish_stamp_s` is the outcome-envelope preparation time, not a subscriber acknowledgment. Receipt wall times are monotonic operational clocks, not capture timestamps.

`read_journal(path)` verifies every complete record's hash chain, sequence and identity. Missing/changed records and torn tails fail explicitly. This is a consistency check, not a cryptographic signature or an automatic repair of incomplete evidence.

The shared API is in **`unav_common.camera_outcomes`**, approved by the coordinator to avoid a reliability→perception dependency cycle:

```python
OutcomeJournal(path, producer_epoch, max_bytes=256 * 1024 * 1024)
journal.append(event)  # returns only after fsync
journal.close()
read_journal(path)     # verified iterator; raises on inconsistent/torn data
journal_path(configured_path, producer_epoch, environ=None)
canonical_json(payload)
```

`perception.core.detector_outcomes` re-exports this API for compatibility and owns only image hashing/frame-member construction. Caller stages are preserved; the shared helper does not relabel manager events as detector events.

## Run paths and recovery

Detector parameters: `outcome_journal_path` and `outcome_journal_max_bytes` (default256MiB). A configured path may contain `{producer_epoch}`. Otherwise the required fallback is `$ROS_LOG_DIR/camera_outcomes/<producer_epoch>.jsonl`; there is no inferred home-directory fallback. Missing/unwritable durability storage fails startup. Exhausting the declared journal cap is a failure, not silent rotation.

Owner13 wired primary and full4cam commissioning launch arguments and the campaign's explicit `<attempt>/detector_outcomes.jsonl` path. That named file is recoverable even if the logger receives **zero** outcome messages and never learns the journal path from DDS. A literal configured path is exclusive; a retry must use its own attempt path or an epoch template.

Owner10 receives the version2 schema, matching QoS and journal reader contract. It must retain every delivery, deduplicate canonical event interpretation by event ID, and recover/verify referenced or known attempt journals **after producer quiescence**. The4096-record retained DDS history is bounded replay, not complete-run archival. The final shutdown-only records require journal recovery. Logger copy/finalization and manager durable-outcome acceptance remain their owners' implementation gates.

## Verification and remaining limits

Current detector/contract verification: **97 tests passed**, including **21 dedicated transport/identity tests**. The transcript is [04_identity_transport_tests.txt](04_identity_transport_tests.txt).

The new tests exercise actual callback/batcher/decoder/inference/selection/CameraObservation serialization methods with five distinguishable images, reversed arrival order, fake inference and transport. One logical cycle produces three native calls and five correctly labelled observations; repeated delivery adds no inference. Geometry/ROS transport are stubbed at declared boundaries; this is not a live ROS end-to-end run. Other cases cover publisher failure, fsync ordering, partial disk writes, journal capacity, retained-record corruption/gaps, concurrent writers, shutdown without DDS, same-stamp conflicting bytes, per-call timing excluding journal delay, and model exceptions/interruption with explicit terminal errors.

Reproduction for the detector-owned and unchanged adapter/contract checks:

```bash
python3 -m pytest -q tests/perception/test_detector_outcome_transport_04.py tests/perception/test_batched_four_camera_yolo.py tests/perception/test_scheduled_camera_registry.py tests/reliability/test_single_camera_adapter.py tests/reliability/test_fusion_contracts.py tests/reliability/test_boolean_contracts.py
```

The older combined `test_camera_acquisition_audit_04.py` includes manager AST fixtures. Owner07 is updating those to its concurrently changed locked/snapshotted manager API; missing mock-lock errors during that transition are retained as an integration boundary, not a detector result. Run that file with07's final source/test packet before combined acceptance.

Still required: real ROS/RMW late-subscriber/process-exit validation, producer-quiescence journal collection, and a newly frozen integration run. Hard kill or storage failure may leave an explicit open intent or torn tail; no finite journal guarantees an outcome for work whose result was never durably recorded. The standalone legacy scheduled/single-camera nodes retain their separately audited weaknesses and were not switched into the active path. Hot clock reset, true hardware frame/round counters, and proof of physical round membership beyond the timestamp window remain outside this bounded repair.

## Changed files and ownership

04 production edits:

- `src/perception/perception/core/four_camera_batch.py`
- `src/perception/perception/core/detector_outcomes.py` (new)
- `src/perception/perception/nodes/batched_four_camera_yolo_node.py`
- `src/unav_common/unav_common/camera_outcomes.py` (new; coordinator-approved common helper)

04 regression work: new `tests/perception/test_detector_outcome_transport_04.py` plus detector fixture updates in `tests/perception/test_camera_acquisition_audit_04.py`. The latter's manager fixtures are now owned by07. Source hashes for this repair are in [04_identity_transport_sha256.txt](04_identity_transport_sha256.txt). Contracts/manager changes belong to07; logger changes to10; launch/campaign path wiring to13. All frozen logs and source snapshots were preserved.
