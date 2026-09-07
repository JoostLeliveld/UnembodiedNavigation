## Component report inventory and current boundary

Audit 15 read every Markdown report and addendum in `docs/module_audits/`, including the coordinator queue and the later implementation notes. The matrix preserves the original findings and records later repair evidence separately.

| Audit | Completed report and scope | Reconciled current status |
|---|---|---|
| 01 | [`01_state_estimation.md`](01_state_estimation.md), package [plan](01_repair_package_a_plan.md) and [implementation](01_repair_package_a_implemented.md) | Snapshot race, envelope bypass and input chronology repaired and revalidated. Missing-motion validity, capture-time heading, full transaction/revision and epoch remain open. M08–M12, M20–M25, M27, M66. |
| 02 | [`02_timing_and_callbacks.md`](02_timing_and_callbacks.md) | Publication/revision/frame coherence, planner snapshot provenance, epoch and worker scheduling remain open; package 01 closes only its named overlap. M06, M08–M14, M16, M27–M28. |
| 03 | [`03_command_execution.md`](03_command_execution.md), [repair plan](03_command_execution_repair_package.md) and [implementation](03_command_execution_repair_implemented.md) | Ordinary-stop cancellation and real age gate repaired and revalidated. Belief/goal/epoch revision, stamped commands, final actuator watchdog, physical stop and fixed-time disturbance remain open. M15–M19, M52, M62, M66. |
| 04 | [`04_camera_acquisition_and_batching.md`](04_camera_acquisition_and_batching.md) | Bounded queues, chunk cardinality/IDs, conflict/timeout outcomes and fail-fast exits pass 101 scoped tests. `source_batch_id` still names a logical cycle, durable outcomes and hard-loss reconciliation remain open. M01–M04, M24, M27, M65. |
| 05 | [`05_observation_geometry_and_calibration.md`](05_observation_geometry_and_calibration.md) | Active frozen NN, offset and full R reproduce exactly. Input/calibration semantic binding and optional geometry/covariance paths remain open. M05, M25, M48, M64. |
| 06 | [`06_admission_bootstrap_recovery.md`](06_admission_bootstrap_recovery.md) | Reasoned refusal and some valid recovery pass. Point-prior suppression, repeated-inflation admission, motion support and optional paths remain policy or correctness gates. M08, M20, M22–M26. |
| 07 | [`07_multicamera_fusion.md`](07_multicamera_fusion.md) | Active robust formula and registered B/C event reproduce. Later fusion contract hardening passes scoped tests; common-time motion and state/outcome transaction remain open, model-floor choices stay isolated. M06–M08, M20–M26, M65. |
| 08 | [`08_planner_and_future_camera_model.md`](08_planner_and_future_camera_model.md) | Active score-precision proxy is documented; it is not a recursive camera posterior forecast. Complete-route gate, immutable solve provenance, numerical/geometry validation and forecast support remain open; later cache tests are scoped. M16, M18, M30–M35, M65. |
| 09 | [`09_tracking_routes_termination.md`](09_tracking_routes_termination.md) | Clearance accumulation, belief/goal validity, route handoff/corners, stuck/clock handling and physical terminal-stop proof remain open. Historical controller changes stay separate. M50–M53. |
| 10 | [`10_logging_and_event_accounting.md`](10_logging_and_event_accounting.md) | Current logs cannot guarantee one durable terminal outcome per consumed correction, reconstruct exact posteriors/applied commands, or drain before final summary. M20–M21, M36–M39, M63. |
| 11 | [`11_alignment_replay_and_reporting.md`](11_alignment_replay_and_reporting.md) | `aligned.py` parses valid selected ledgers but does not provide one strict run certificate; posterior association, epoch/reference support, selection, aggregation and metric validation remain open. M21, M36, M39–M45. |
| 12 | [`12_capture_commissioning_and_export.md`](12_capture_commissioning_and_export.md) | Frozen 7,138 image / 15,440 view corpus and active artifacts verify within scope. Capture resume, exact inference-image identity, acquisition failures and calibration/lineage sufficiency remain open. M46–M49, M64. |
| 13 | [`13_configuration_provenance_campaigns.md`](13_configuration_provenance_campaigns.md) | Selected file hashes verify, but attempt identity, strict resolved config, execution attestation, transitive asset closure, leases and durable campaign ledger remain open. M54–M58. |
| 14 | [`14_simulator_world_and_physical_outcomes.md`](14_simulator_world_and_physical_outcomes.md) | Exact direct shell/occluder boxes and private contact transport verify. Five physical props are omitted, reset is not a fresh epoch, final drive target survives failures, TF/startup/contact/provenance contracts fail. M59–M64. |

The coordinator report [`COORDINATION.md`](COORDINATION.md) is a moving ownership/status queue. It is not runtime evidence and still says audits 14/15 are pending; the documentation table assigns its final update.

## Scenario-by-invariant verification projection

Codes: **P** = verified within the exercised scope, **R** = a scoped repair/local property passes but the complete invariant remains open, **F** = reproduced failure, **N** = the scenario did not exercise that channel. Every row below accounts for I1–I14 from the obligation list above; the cited matrix rows contain commands, evidence, owners and repairs.

| Scenario | P | R | F | N | Primary matrix evidence |
|---|---|---|---|---|---|
| S1 normal observations | I7 | I1,I3,I4,I6,I8 | I2,I5,I10,I11,I12,I13,I14 | I9 | M01,M05–M08,M12–M13,M18,M20–M21,M36,M54–M60,M64–M66 |
| S2 long camera outage, complete odometry and turn | I3,I4,I5,I9 | I2,I8 | I11,I12,I13,I14 | I1,I6,I7,I10 | M09–M12,M20–M23,M39–M45,M54–M61,M66 |
| S3 outage with missing/gapped odometry | — | I5,I8 | I3,I4,I9,I11,I12,I13,I14 | I1,I2,I6,I7,I10 | M09–M11,M23,M39–M45,M54–M61,M66 |
| S4 delayed/duplicate/reordered/simultaneous camera | — | I1,I5,I8 | I2,I3,I4,I11,I12,I13,I14 | I6,I7,I9,I10 | M01,M03,M06–M08,M13,M20–M26,M36,M39–M49,M54–M58,M65 |
| S5 correction during planning/execution | I6,I7 | I5 | I3,I4,I8,I11,I12,I13,I14 | I1,I2,I9,I10 | M13–M18,M30–M35,M50–M53,M54–M58,M66 |
| S6 detector/command silence | — | I1,I9 | I6,I8,I10,I11,I12,I13,I14 | I2,I3,I4,I5,I7 | M03–M04,M14,M17,M20,M23,M36–M39,M54–M58,M62–M63,M65 |
| S7 rejection then valid recovery | — | I2,I3,I4,I8,I9 | I5,I11,I12,I13,I14 | I1,I6,I7,I10 | M05–M08,M20–M25,M33,M39,M43,M48–M49,M54–M58 |
| S8 final approach, stop and contact probe | I7 | I5 | I6,I10,I11,I12,I13,I14 | I1,I2,I3,I4,I8,I9 | M15–M18,M29–M32,M37,M50–M53,M59,M62–M63,M66 |
| S9 controlled restart/reset | — | I1 | I2,I3,I4,I5,I6,I8,I9,I10,I11,I12,I13,I14 | I7 | M01,M11,M13,M17,M27–M28,M51,M54–M63 |
| S10 shutdown with buffered terminal events | — | I8 | I2,I10,I11,I12,I13,I14 | I1,I3,I4,I5,I6,I7,I9 | M20–M21,M29,M36–M49,M54–M58,M63 |

`N` means absent evidence, never an implicit pass. I14 fails until the exact documentation corrections below are made. I11 fails system-wide because the selected protocol lacks complete imported-source and transitive physical-asset closure even where narrower hashes match.
