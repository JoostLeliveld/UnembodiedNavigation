# 04 — Camera acquisition, scheduling and batch identity

Reviewed 2026-09-06 against the shared working tree. **The active strict path prevents ordinary repeated delivery from becoming a second manager decision, but does not provide a complete end-to-end outcome ledger. Incomplete manager batches are unbounded, reset recovery is absent, and malformed chunk result counts can silently shift camera identity.** These are software findings, not diagnoses of the running pilot or measurements of camera accuracy.

Read first: AGENTS.md, PLAN.md, localization metrics contract and registry, open questions and runtime integrity audit. Also checked the module investigation map, state-estimation module review and existing perception tests. Source identity is retained in [04_source_sha256.txt](04_source_sha256.txt). No runtime source, weights, selection rules, configurations, running processes or experimental outputs were changed. Coordinated with “Advance ICRA Localization Paper”; that thread confirmed runtime sources remain frozen during the three-arm pilot. Geometry investigation 05 owns geometry/model verification; investigation 07 owns fusion mathematics.

## Reachability and intended scheduling

The registered anchor is `experiments/icra_commissioning/network_navigation_runtime_pilot.yaml`, with frozen experiment provenance at `logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/protocol.json` and `source_snapshot/`. This report checks current source against that configured path, not whether each fault occurred in a drive.

`warehouse_primary_comparison.launch.py` delegates to `visibility_launch_common.py`. `multicam_belief=true` selects `_multicam_belief_nodes` (1700–1812); `multicam_scheduled` defaults false (primary launch:388). Detector: native YOLO, CPU, 960 image size, configured frozen detection checkpoint, masks false, strict ROS input, cameras A–E, stamp skew 0.05 s, pending wall age 0.50 s. Inference chunk defaults to two (batched node:180,236). Manager requires batch IDs, subscribes to A–E and decides at 5 Hz. Trace defaults off (launch common:1769; primary launch:261). World profile order must equal the detector contract (launch common:1716–1724); manager subsets must belong to the profile (1782–1794).

**Policy:** strict inference is arrival-triggered when all five cameras complete a sufficiently close timestamp bucket. It is not a timer promising inference on every physical capture. ROS depth-one image queues discard intermediate traffic while synchronous inference blocks callbacks. Open buckets expire on later offers. Completing a newer bucket discards earlier buckets. A manager timer selects its latest complete batch; multiple completions before a tick can supersede one another. A missing camera intentionally prevents strict subset inference. These sampling choices require observable drops; changing to partial-camera inference changes the experiment and is not a correctness-only repair.

Optional scheduled mode ranks coverage and tries cameras until a usable world correction, or chooses one round-robin camera. Contrary to its “one inference/cycle” prose, coverage fallback may invoke every eligible camera in a tick (scheduled node:190–255). Ordinary single-camera mode uses inline inference or a latest-slot worker. Neither optional node is the selected pilot detector.

## Ranked findings

### F01 — P1, confirmed current: incomplete manager transactions grow without limit and never close

**Active:** yes. Camera manager:1177–1210, especially `setdefault` at1191. Each new source ID creates a dictionary; cleanup happens only when a complete newer batch arrives. There is no receipt deadline, capacity limit, missing-camera outcome or timeout sweep. Identity mismatch/invalid JSON returns before completing that slot (1180–1188).

**Trigger:** camera E publications absent, rejected or lost while A–D continue; or detector failure after publishing a prefix. **Expected:** bounded pending state and explicit incomplete/aborted outcome naming missing members. **Observed:** 1,000 synthetic A–D batches retain 1,000 dictionaries, zero ready batches. The manager can continue ticking forever without new evidence. A later complete batch clears old records without recording their individual terminal outcomes.

**Consequence:** memory growth and indistinguishable starvation/publication failure; camera-opportunity logs cannot reconstruct all scheduled inference outcomes. **Smallest repair:** record first monotonic receipt per batch, cap pending count, expire on a wall timer, emit terminal reason with expected/received/missing camera IDs, and reject closed IDs before allocation. Retain all-camera admission until a separate scheduling policy is approved.

### F02 — P1, confirmed current: active launch does not implement the advertised fatal shutdown policy

Runtime contract declares `fatal_process_exit_and_launch_shutdown_no_synthetic_miss` (four_camera_runtime_contract.py:64). `_fatal` raises (batched node:730–737). Publication is sequential: `_process_frames`:1026–1029, diagnostics then observation at1224–1228, then pixel pose at1098–1101. Preparation validates selections before publishing, but does not preconstruct/validate all outgoing observations or make publication atomic.

**Active:** `_multicam_belief_nodes` constructs detector at launch common:1728–1774 with no `on_exit` handler. The separate commissioning launch does attach `Shutdown`; its source regression does not test the active primary launch. **Trigger:** inference/decoding/serialization/publisher failure. **Expected:** declared launch shutdown and identifiable batch abort. **Observed:** source wiring lacks that handler; a fake publisher fails at B after A is emitted, leaving a prefix. No abort message is emitted. Exceptions are explicit fatal logs, not detector misses, but those logs lack a structured transaction outcome. Full ROS launch death propagation was not exercised; no claim that all other processes necessarily survive every failure.

**Smallest repair:** attach detector process-exit shutdown to the active launch, prebuild all camera messages, and publish a batch terminal event. Receiver timeout from F01 is still necessary because reliable independent topics are not an atomic transaction. Do not synthesize misses on exceptions.

### F03 — P1, confirmed current under malformed backend output: chunk counts can cancel and mislabel cameras

Batched node `_predict_batch`:636–650 extends results from each chunk, then checks only total length. `_process_frames`:1008–1016 checks total result interface and maps by index.

**Active:** default chunk=2 yields calls [A,B], [C,D], [E]. **Trigger:** first call returns one result, second returns three, final returns one. **Expected:** reject first malformed chunk before assembling a camera mapping. **Observed:** deterministic actual-method fake yields [A,C,D,extra,E], length five, accepted; B receives C's result. Correct native backend output is expected to preserve input order; this test establishes a missing integrity guard, not a demonstrated Ultralytics failure.

**Consequence:** plausible boxes assigned to wrong calibration/camera. **Smallest repair:** apply `validate_batch_results(part, len(group))` inside each chunk before extending. Also retain chunk ID/member mapping if “physical invocation” means each actual `model.predict` call. The existing source batch ID identifies a whole inference cycle, which currently comprises three physical calls, not one.

### F04 — P1, confirmed current on reset: timestamp identity has no epoch

Batcher:155–156,244–248,271; source ID construction batched node:944–949; manager:1194–1204. **Active:** yes, conditional on restart/reset; selected config has `reset_world=false` and routine runs start fresh processes.

**Trigger:** surviving detector sees simulation clock move from 100 s to zero. **Expected:** explicit reset transition or declared fatal reset requiring coordinated restart. **Observed:** all five reset frames are `out_of_order` until old high-water timestamps are exceeded. Fresh detector instances given the same camera stamps produce identical `strict:camera@stamp,...` IDs. A surviving manager also rejects complete new-epoch batches below its old maximum. Two distinct source IDs at the same maximum timestamp are rejected as old.

**Consequence:** long lockout, identity collisions across sessions; a distinct physical same-stamp image is indistinguishable from retransmission. This does not repeat the repaired direct-camera equal-time bug: different cameras within one normal batch are preserved.

**Smallest repair:** define producer epoch/session plus capture sequence and inference-cycle sequence, propagate them across the manager boundary, and explicitly handle coordinated clock jumps. Use exact capture time for timing, not as the sole identity. Keep duplicate tombstones by identity; separately classify lateness. If simulator guarantees unique monotonic stamps within an epoch, document and validate that producer assumption.

### F05 — P2, confirmed current: missing-camera and overload observability remains incomplete

Batcher:183–188 `bucket_report` returns **present**, not missing cameras despite its docstring. Batched trace:914–924 formats these without an explicit present/missing label. Trace is disabled in the active config. `_expire_locked`:204–220 reports dropped camera IDs without round/image identity. `_image_callback`:740–778 warns about expired healthy cameras, not the camera preventing completion. No strict expiry timer calls `expire`; with no further offers, pending frames stay resident indefinitely. Completing a newer bucket deletes all earlier buckets silently (batcher:311–315). Startup clock guard consumes a selected batch and returns with a bounded warning (node:958–964).

**Expected:** distinguish absent input, expired/replaced input, selected inference, miss and failed publication, with counts and identities. **Observed:** absent-E test reports A–D as expired; 1,000 bursts at one wall instant create 1,000 open buckets with no count cap; completing the last deletes 999 older rounds with empty drop metadata. Wall expiry bounds residence under continuing callbacks, not absolute capacity. Actual sustained camera rate limits normal occupancy; this is not evidence of ordinary 5 Hz detector memory runaway. DDS depth-one drops occur before callback receipt and are uncounted.

**Smallest repair:** explicit present/missing sets and periodic wall-time liveness; bounded bucket count with reasoned per-round eviction counters; record superseded/expired/startup-clock drops. Add source sequence gaps or middleware loss instrumentation if physical acquisition loss must be measured. Fix outdated “latest-only/one pending frame” module/node prose. Avoid turning absence or unscheduled input into a detection miss.

### F06 — P2, confirmed conditional limit: tolerance grouping does not prove physical round identity

Batcher `_bucket_key_locked`:197–202 picks the first existing bucket within tolerance; offer:271–280 can replace the same camera inside that bucket. Final max–min check:283–306 prevents span above tolerance, but cannot identify physical rounds inside it.

**Trigger/reproduction:** A at1.000 s from round1 and B–E at1.040 s from round2 form a valid mixed batch. Equal stamps with different image bytes are classified duplicate. **Expected:** if strict means physical round membership, require producer round identity; **observed:** strict currently means timestamp-window membership. **Active reachability:** selected 5 Hz cadence separates ordinary adjacent rounds by0.20 s, exceeding0.05 s, and existing regression verifies their separation. Mixing inside0.05 s is a constructed changed-cadence/jitter/clock scenario, not a demonstrated active-cadence bug.

**Smallest repair:** state the timestamp-window assumption and validate cadence/skew, or carry an explicit capture round. Do not claim tolerance is an exact physical-round identifier. Nanosecond parsing itself is exact; tolerance conversion rounds once to integer nanoseconds, with no millisecond rounding in batch identity.

### F07 — P2, confirmed optional-path defects: scheduling outcome and timing contracts are weaker

Scheduled detector:126–129 caches any arrival;206–219 claims float stamp before decoding/inference, catches exceptions with warning and returns;175 sets receipt equal to capture;176 uses publication clock as inference finish;246 stamps `/state/bev` at publication rather than capture. There is no source batch ID, max image age or explicit error outcome. `res[0]` at220 assumes a nonempty result list outside the exception handler. Coverage-skipped/missing inputs are intentionally unscheduled, but not recorded as such.

Actual-method exception probe runs inference once, logs a warning, publishes zero terminal observations, and refuses to retry the claimed stamp. Cached unprocessed images can be arbitrarily old when eventually selected. Out-of-order arrivals can overwrite a newer unprocessed slot. Reset high-water lockout also applies.

Single-camera detector:439–451 and510–534 has no duplicate guard: actual-method repeated identical delivery invokes inference twice. Its latest slot is cleared correctly (487–493), so it does not autonomously reuse an empty slot; repeated transport delivery is the trigger. `_inference_worker`:495–509 has no exception recovery/final status, so a processing exception can end the worker while ROS remains spinning. JSON publication failures are warned and swallowed (411–436). No cycle ID is added. The manager's compatibility path generates a new `unidentified:N` for every observation (1215–1220), so repeated delivery can become new decisions there; active `require_source_batch_id=true` rejects these unidentified messages.

**Repair:** snapshot image plus exact identity and real receipt clocks, enforce age/epoch policy, use one explicit hit/miss/error outcome per attempted inference, propagate IDs, stamp the correction at capture, and retain per-camera dedup. Keep selection/weights fixed. These optional modes require separate validation before replacing strict mode.

## Physical image trace and timing ownership

| Stage | Identity and clocks | Owner / terminal behavior |
|---|---|---|
| Capture and transport | Simulator image header sec/nanosec; no capture sequence/epoch. Subscription topic fixes camera, not header frame. | Producer/bridge/DDS; detector cannot observe images dropped before receipt. |
| Receipt | `_image_callback`:740–755 validates exact ns; records ROS `_clock_s()` and monotonic `perf_counter()` at callback entry. This is application callback receipt, not wire arrival. | Depth-one ROS queue; strict callbacks/inference serialized by `rclpy.spin` (1274–1284). |
| Buffer | `PendingFrame(camera_id,stamp_ns,receive_stamp_s,receive_wall_s,payload)`; integer-key buckets. | Batcher owns pending sets, high-water stamps and lock. Duplicate/out-of-order explicit decisions; replacement/expiry/supersession incompletely identified. |
| Selection | Contract A–E order independent of arrival; latest complete selected bucket spans≤0.05 s. | Batch removed/claimed before decode and startup clock check. All five required. |
| Decode | `ros_image.py`:9–60 respects row stride, strips padding, converts rgb/bgra/rgba/mono to contiguous BGR without resizing. | Invalid geometry/encoding/short bytes raises fatal on strict path. 8-bit endian irrelevant; trailing bytes ignored. Test verifies padding, channel order and row coordinates. |
| Inference | Same ordered images split into consecutive chunks. ROS start/finish and total wall milliseconds recorded once for the cycle (982–1005). | Synchronous model owner; no post-inference max-age rejection. Fake100 s inference still reaches publication. Manager may later age-refuse; detector age is measured, not bounded by pending timeout. No per-chunk timestamps/IDs. |
| Selection | `yolo_selection.py`:44–95 filters malformed/non-target boxes and stable score-sorts;108–213 applies frozen confidence threshold; selected pixel remains bbox bottom-centre. | Original input-coordinate bbox assumed from backend; strict native path adds no manual resize transform. Malformed candidates deliberately filtered like no candidates, without distinct invalid-output reason. Masks are diagnostic and disabled in active config. |
| Publication | source ID is cycle member camera@exact-ns list; diagnostics float stamps include capture, callback receipt, start, finish, publication and wall duration. `single_camera_adapter.py`:209–227 transfers times to CameraObservation; replace adds source ID (node:1238–1244). | Every successful camera inference produces hit or miss observation; pixel pose only on hit. Publication clock is read before actual publish, not acknowledgment time. Diagnostic array/pixel pose carry no batch ID; JSON does. Multi-topic publication is not atomic. |
| Manager receipt | JSON parsed, camera checked against subscribed camera (1177–1188), source ID used as key; original capture/timing fields retained. No local manager receipt timestamp stored here. | Pending dict requires configured camera membership. No timeout/cap, no verification source-ID member list matches observation time, and no conflicting-duplicate validation. |
| Manager decision | Latest complete max capture stamp advances watermark. `_decide`:1523–1533 processes source ID once; faster ticks/repeated complete delivery create no new decision. | Latest complete batch can supersede earlier complete batch before timer; no per-batch superseded outcome. Mathematical gates and covariance are investigation07 scope. |

Timing caveats: float observation timestamps can collapse adjacent nanoseconds at epoch-sized seconds even though source ID retains integer ns. The manager's `<=` watermark would then discard a distinct ID. At ordinary short simulation times this is not a demonstrated resolution problem. Callback age excludes time spent in middleware; total publication age includes it through the capture clock but depends on valid clock alignment. A clock jump during inference may fatal at publication; a jump before inference is treated as startup waiting without checking that startup is actually in progress.

## Regression evidence and limits

Reproduce from repository root:

```bash
python3 -m pytest -q tests/perception/test_camera_acquisition_audit_04.py tests/perception/test_batched_four_camera_yolo.py tests/perception/test_scheduled_camera_registry.py
```

**28 passed**, retained in [04_test_results.txt](04_test_results.txt):14 new audit probes plus14 existing checks. New tests intentionally assert observed defects; green does **not** mean the proposed repairs are implemented. After repair, convert defect assertions to desired invariants. AST extraction executes unchanged node methods with fake ROS boundaries rather than duplicating their algorithms. Synthetic images contain camera-distinguishing byte values, with controlled stamps and fake inference outputs. No GPU, simulator or running experiment is used.

Covered: staggered order, 0.20 s adjacent-round regression, same-stamp duplicate and out-of-order delivery, missing camera, explicit expiry, burst capacity, silent supersession, restarted instances and clock rollback, same-time distinct manager IDs, repeated deliveries and60 rapid decision ticks, malformed chunk counts, slow inference, partial publication, scheduled exception, duplicate single-camera inference, RGB padding/coordinates, malformed/empty result contract. Existing registry test checks aligned configurable scheduled camera lists. Strict all-empty result objects are valid misses; missing result objects are fatal.

Not exercised: actual DDS loss/backpressure, ROS launch process teardown, physical camera sequence guarantees, inference hangs, live reset, actual backend resize/letterbox coordinate restoration, every ROS encoding and compiled diagnostic backend. These remain integration checks/hypotheses, not confirmed runtime faults. No accuracy/run comparison was performed, so no historical results were promoted and no scoring loader was invoked.

Documented repairs independently reverified here: A–E deterministic ordering; adjacent0.20 s rounds not merged; exact ROS stamp validation; malformed total result rejection; no manager reuse on faster ticks. Documented direct-camera simultaneous-update repair belongs to estimation/07 and was not rerun. Incomplete-round trace infrastructure exists, but its active default and present/missing ambiguity mean the original liveness requirement is only partially met.

## Handoffs and smallest repair order

1. Investigations13/10: match active detector-exit policy to contract; introduce structured cycle completion/abort and manager timeout/cap with missing-member reporting.
2. Perception: validate each inference chunk before concatenation; preserve all selection rules and artifacts.
3. Investigations02/13: agree epoch/reset semantics and exact capture versus invocation identities across producer, manager and estimator.
4. Investigations10/12: expose buffer eviction, unscheduled input and source gaps separately from detector misses; current logger explicitly scopes opportunities to received detector outputs (`experiment_logger.py`:631), not all physical captures.
5. Investigation05: decoder probe preserves coordinates, but original-dimension/calibration validation and backend coordinate contracts remain geometry work. No active bbox-coordinate mismatch established here.
6. Investigation07: keep manager evidence scheduling, equal-time distinct-ID rejection and latest-complete supersession visible when testing fusion; no fusion mathematics changed.
