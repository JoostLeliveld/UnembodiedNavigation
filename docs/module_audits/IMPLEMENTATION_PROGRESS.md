# Remaining audit repairs — implementation progress

The user authorized implementation with “Fix the rest?” after all 15 audits completed.
This record supersedes the earlier report-only scope. Runtime acceptance remains open.
The [consolidated queue](COORDINATION.md) defines the required invariant packages;
the [audit-15 matrix](15_end_to_end_acceptance.md) is the historical starting verdict.

## Current ownership

| Owner | Production scope | Work / verification |
|---|---|---|
| Coordinator | `unav_common/correction_ledger.py` | Implemented raw publication/outcome validator; 79 dedicated tests pass, 105 combined with offline repair tests. Logger/offline/campaign integration underway. |
| 01 | Unicycle node, belief core, motion history, encoder validity | Central immutable belief record, epoch/revision, atomic inputs, support and correction accounting implementation underway. |
| 02 | Dedicated independent timing tests and review | Stable source packet passes 113 tests: 23 independent timing cases and 90 A/B/runtime/state/encoder regressions. Evidence under `timing_callbacks_repair_20260907/review_checkpoint_02`. |
| 03 | EFE route/request/command installation | Handoff in `03_remaining_repairs_20260907.md`; 79 focused passes reported. Expanded set has three shared runtime transaction failures still under 01 integration. |
| 08 | Pure planner/candidate/numerical validation | Shared result contract and planner fixes underway; EFE integration through 03. |
| 04 | Detector acquisition and invocation/member identity | Component complete: `04_identity_and_outcome_transport.md`, 97 focused passes and source hashes. Manager/logger/live transport verification remains separate. |
| 05 | Geometry/calibration validation helpers | Pure validation API and independent cases; manager integration through 07. |
| 06 | Independent admission/recovery review and tests | Preserve observations during turns and declared gate/noise policies; report production fixes to 01/07. |
| 07 | Manager transactions and common-time motion | Coordinates 01 motion/revision API, 04 identity, 05 semantics and 10 ledger. |
| 09 | Mission node and route helpers | Zero-centred seeds, atomic mission-goal helper and ROS stamps implemented; belief consumer integration pending. |
| 10 | Logger and opportunity log | Component handoff complete with 241 cross-owner tests reported; raw/canonical schema8, exact terminal evidence and bounded drain/close implemented. Post-shutdown producer journal reconciliation belongs to 13 integration. |
| 11 | Offline alignment/scoring/replay consumers | Shared raw-ledger validation and explicit reference support, exact posterior evidence and other audited fixes underway. |
| 12 | Capture, commissioning/export and historical config loading | 31 focused passes reported, including exact historical profile recovery and transactional resume/conversion. Export integrity closure underway. |
| 13 | Campaign, manifest/provenance and common experiment launch | Attempt/config/evidence transactions underway; simulation bringup reserved for 14 after handoff. |
| 14 | Native final actuator guard, simulator/reset and physical geometry closure | Component handoff complete: native guard, reset/readiness, collision closure and scoped private timeout/contact probe. Final source-bound raw-event and sustained-rest verification remains separate. |
| 15 | Independent successor acceptance harness | Owns executable cross-chain assertions and the successor matrix; execution waits for source-bound owner handoffs. |
| Refactor loose pivot code | Machine-local-path regression scan | Classification repair complete, 9 focused passes. New source and active config remain scanned; frozen evidence remains untouched. |

Counts from other owners are reported component results, overlap, and must not be summed.
New red tests intentionally reproduce unclosed bugs; no full green suite is claimed.
All production owners preserve the pre-existing dirty tree and packages A/B.

## Shared validator contract

`validate_correction_ledger(publications, outcomes, require_timestamps=True,
allow_repeated_publications=False, time_tolerance_s=1e-9)` returns `LedgerValidation`.
The result exposes structured errors, valid status, terminal/publication maps, update IDs,
refusal counts and a JSON-safe summary. `require_valid()` raises `CorrectionLedgerError`.

Input is raw canonical evidence, before reference filtering. Exactly five terminal statuses
are permitted. Both rejected and dropped events require reasons. Explicit accepted flags
must agree with status. Distinct equal-time events survive; missing, extra, duplicate or
contradictory identities fail. Per-camera publication representations collapse only when
explicitly enabled and their common numerical payload agrees. Terminal duplicates never
silently collapse. Schema wrappers still enforce file presence, required fields/shapes,
model/covariance validity, epoch capability and deployment provenance.

Supplied nanosecond fields are strict nonnegative integers. Exact publication/outcome
correction times must agree, even when float seconds would round them to the same value.
Accepted application cannot precede exact correction time; a reasoned future-input
refusal remains valid evidence. Canonical adapters must retain supplied exact fields.

Ledger validity is separate from reference scoreability. No default reference interpolation
gap is invented: an explicit protocol/caller bound enables interpolation; exact timestamp
references remain usable without it. Unsupported intervals are counted and labelled.

## Remaining integration gates

1. Land and agree 01 snapshot/publication/terminal schema with 03/07/09/10.
2. Convert independent red timing/admission cases to passing desired invariants.
3. Integrate ledger consumers and producer identity, then test failure/drain paths.
4. Resolve frozen-profile loading without changing original captured hashes.
5. Verify physical final command timeout/reset/zero behavior in a reserved private fixture.
6. Freeze repaired source identities and run the combined component suite and integration
   matrix; no broad campaign or runtime acceptance claim before prerequisites pass.

The coordinator owns final cross-chain integration and I14 engineering documentation
truthfulness after runtime source freeze. Owner 15 independently verifies the join from
invocation/capture through outcome, correction/terminal, belief revision, plan/result,
command receipt/application, odometry/contact, mission terminal, logger drain and offline
decision. Component passes alone do not close that gate. Owner 14 supplies the executable
private physical fixture after native guard wiring; the coordinator does not substitute a
synthetic component test for physical verification.

The landed state API is documented in [01_state_repair_api.md](01_state_repair_api.md):
`/planner/belief_state` schema 1 is the coherent operational publication; correction
assimilation schema 2 carries exact prior/posterior, revision and motion-support evidence.
Owner 09 owns its shared pure consumer validator. Integration must use these actual APIs,
not the previously proposed `/planner/belief_metadata` topic.

The machine-local-path scan now distinguishes recorded provenance from executable input
dependencies, with negative controls for new source, home paths and active configuration.
Frozen audit evidence was not rewritten to make a test pass. Performance configurations
with different physics/contact rates retain separate identities.
