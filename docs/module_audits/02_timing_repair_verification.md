# 02 — Independent timing repair verification

2026-09-07. **Acceptance pending the production-owner handoff.** The coordinator authorized the remaining correctness repairs and assigned audit 02 independent timing tests and review. Audit 01 owns unicycle/state/motion production changes; audit 03 owns EFE changes; audit 07 owns the manager consumer. This report does not claim that an intermediate working tree is repaired.

## Test boundary

[test_timing_publication_invariants.py](../../tests/planning/test_timing_publication_invariants.py) contains 23 deterministic cases. It invokes actual node callbacks and consumers with controlled ROS clocks, event barriers, observed lock contention and fake publishers. It creates no ROS graph or simulator and reads no experiment data. Synthetic linear prediction checks snapshot ownership; it does not test or alter the production filter mathematics.

The first 14 cases ran before the repairs: **13 failed, one passed**. All failures were the expected invariant assertions; there were no import, fixture or synchronization failures. The passing case verified full planar covariance, state, frame and target-time coherence in the ordinary path. [Output](../../experiments/runtime_integrity/timing_callbacks_repair_20260907/before_repairs.txt), [execution identity](../../experiments/runtime_integrity/timing_callbacks_repair_20260907/before_repairs.json) and [JUnit results](../../experiments/runtime_integrity/timing_callbacks_repair_20260907/before_repairs.xml) are retained. Later schema and review cases extend that packet; they were not part of the 14-case baseline.

[The snapshot manifest](../../experiments/runtime_integrity/timing_callbacks_repair_20260907/before_snapshot_manifest.json) distinguishes execution from later file copying. EFE changed between the baseline execution and snapshot copy; its later copy is not claimed as the exact baseline file. The packet exercises base-node timing and manager consumption rather than EFE execution.

## Required invariants

| Boundary | Regression |
|---|---|
| Overlapping public predictions | A target 10.0 delayed past target 10.2 cannot become the latest publication. |
| Same-anchor correction | A real correction changes revision and suppresses in-flight work from the earlier posterior even when anchor time is unchanged. |
| Reset during prediction | Pre-reset prediction cannot publish after the clock rewinds. |
| Reset during correction | Commit rechecks the clock; pre-reset numerical work cannot replace the posterior afterward. |
| Invalid/future age | An invalid time conversion or future anchor cannot become fresh age-zero state. |
| Command ingestion | Receipt time, history and held command have one commit order; NaN and either infinity cannot mutate them. |
| Goal ingestion | Goal coordinates, signature and progress invalidation belong to one callback event. |
| Published state | Full 3×3 planar covariance, position, heading, frame and exact projection target agree; prediction leaves the recursive anchor unchanged. |
| Frame ownership | A compatibility pose cannot relabel committed map coordinates without a transform. |
| Planning metadata | A correction during prediction cannot change only the returned anchor metadata. |
| Diagnostic failure | A committed source ID and immutable terminal outcome survive a later diagnostic exception; redelivery cannot reclassify it as never assimilated. |
| Slow diagnostics | The terminal record retains the actual commit time and posterior when later reporting takes time. |
| Simultaneous duplicate | Exactly one accepted terminal commit; the duplicate receives the existing integrity disposition. |
| Equal public target | Higher posterior revision remains distinct at unchanged state time. |
| Explicit state event | Epoch, revision, anchor, target, mean, covariance and frame agree with the compatibility pose. |
| Invalid status | Rewind revokes the retained valid status. A delayed uninitialized-status calculation cannot revoke a newer bootstrap. |
| Manager integration | Higher revision replaces the old prior; delayed canonical/compatibility deliveries cannot revive an older or invalid prior. |

Test fixtures install their initialized posterior through `_commit_belief` when the new API is available. They do not overwrite authoritative-state mirrors. Motion and heading fixtures contain preceding samples so the timing checks do not depend on an unsupported-input recovery assumption. Invalid-motion consumption is exercised separately by advancing the clock beyond recorded support.

## Review handoff

The production owner received the initial failures and the following additional review constraints:

* Capture goal reference and goal revision together. A later belief-metadata read must not replace the request's originating goal token.
* Check clock/epoch at state commit as well as prediction publication. Capture terminal commit time before fallible diagnostics.
* Pass record/epoch provenance into invalid-status publication; a reason computed for an earlier record must not describe its replacement.
* Route optional pixel and legacy recursive writers through the same central commit as the active metric path.
* Treat initialization separately from revision numbering in the public schema; valid adopted revision-zero records need an explicit initialized flag if supported.

The manager owner confirmed that an invalid canonical event removes pose readiness while retaining the fact that an anchor existed. It cannot silently reopen first-bootstrap rules. A producer epoch change requires coordinated runtime restart. Compatibility poses remain supported before canonical-stream binding and cannot override the canonical stream afterward.

Audit 02 has edited only its dedicated test file and evidence/reports. It has not edited production or shared test files, committed/pushed mixed changes, altered Q/R or camera/turn/recovery settings, or started/stopped/reset a running experiment.

## Acceptance

Pending: run the completed packet against the handed-off source; run the relevant A/B regressions; inspect the final recursive writer, callback-group and publication boundaries; record source hashes, failures and remaining limits. Passing method tests will not certify live executor latency, DDS delivery, physical actuation or full-stack reset recovery.
