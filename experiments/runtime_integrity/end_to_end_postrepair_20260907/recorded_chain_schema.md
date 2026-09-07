# Recorded cross-chain acceptance fixture

The final, isolated simulator check must produce one JSON fixture and name its
repository-relative path as `cross_chain_fixture` in `READY.json`. The fixture is an
audit adapter over immutable raw logs; it is not a replacement log format. Every adapted
row must retain an `evidence_id` that resolves to the raw file and row/event identity.

The fixture contains `scenarios` S1–S10 with `exercised` and `evidence_ids`; a `manifest`
with exact source, resolved config, model, calibration, world, robot, route and attempt
identities; and these event arrays:

- `detector_invocations`: `source_batch_id`, `invocation_id`, capture, arrival, detector
  start/end timestamps and a clock-domain field for every timestamp. Ordering is asserted
  only for adjacent events in the same domain.
- `corrections` and `assimilation_outcomes`: the exact correction identity and source epoch,
  one terminal status, reason, independent belief epoch/revisions, frame, state timestamp
  and posterior for accepted work. Source and belief epochs are never joined by field-name
  coincidence.
- `beliefs`: epoch, revision, anchor/prediction timestamps, frame, anchor mean/covariance,
  validity and motion support.
- `state_commits`, `authorized_state_writer` and `authorized_state_transactions`: unique
  serialized commit IDs, before/after revisions and explicit restart-only epoch changes.
- `plans`: complete request and immutable tokens, belief identity, commit tokens, installed
  flag and reason. A supported newer-belief install carries a complete revalidation record;
  it is not mistaken for a blanket stale-result cancellation.
- `commands`: unique ID, receipt/forward timestamps and domains, requested/forwarded values,
  stop provenance, `application_observed=false` and no invented application timestamp. The
  native guard establishes forwarding; subsequent odometry establishes physical rest.
- `command_ownership` and `resolved_limits`: the sole adapter/guard publishers, explicit
  stop priority, one canonical limit record with units, identical resolved values at
  planner/tracker/adapter/guard, and bounded forwarded commands.
- `odometry`, `contacts`, `mission_terminal`, `terminal_events`, `run_summary` and
  `offline_alignment` as asserted by `test_recorded_cross_chain_evidence.py`.

Scenario closure additionally requires `availability_intervals` for S2/S3;
`observation_deliveries` with source identity, arrival sequence and disposition;
reasoned `liveness_events` and ordered `admission_events`; `restart_events` covering stop,
drain, process restart and all fresh inputs under an orchestration `run_epoch` (without
equating independent producer/belief epochs); ordered `mission_events`; and
`shutdown_accounting` receipt sequences proving buffered terminal delivery precedes drain,
independent file closure and atomic summary commit.

The fixture also declares `logger_write_failures` and a `rest_evidence` object supplied by
owner 14: clock domain, observed interval, separate linear tolerance in m/s and angular
tolerance in rad/s. Ordered finite odometry samples must cover the whole declared interval
and remain within both tolerances. Timestamp ordering is asserted only within an explicit
common clock domain. Capture, arrival, commit, belief,
receipt and forwarding timestamps remain separate fields. Physical application remains
unobserved unless a future source-bound acknowledgement is added; it cannot be inferred
from forwarding. Odometry and contact evidence remain separate observations.

The current producer convention uses a logical cycle `source_batch_id` plus a unique
physical `invocation_id`. With `source_batch_identity_contract` set to
`logical_cycle_plus_invocation_v2`, the fixture must also contain one
`detector_batch_outcomes` row per cycle with an exact `member_invocation_ids` list. Every
physical invocation must occur exactly once in those lists, each fused correction must
retain its member invocation IDs, and misses, silence and rejection require explicit
terminal reasons. This is a visible versioned correction to the historical wording that
called the per-invocation value `source_batch_id`; it cannot be inferred from a distinct
identifier alone. A runtime that keeps the literal historical naming instead declares
`per_physical_invocation` and must make `source_batch_id` unique.
