# State repair API — implementation contract

Owner 01: `planning/core/belief_state.py`, `motion_history.py`, `belief_correction.py`,
`nodes/unicycle_planner_node.py`, encoder validity. Owner 09:
`unav_common/operational_belief.py` shared pure consumer validator and mission integration.
Owner 07: manager consumer plus measured-odom common-time compensation. Owner 03: EFE
request/install consumer. Owner 02: independent tests/review. Owner 10: durable logging.

## Canonical operational belief

Topic **`/planner/belief_state`**, `std_msgs/String`, publisher attribute `belief_state_pub`.
There is no proposed `/planner/belief_metadata` topic. Existing `/planner_belief` remains
Pose compatibility and carries only valid, supported predictions.

JSON `schema_version=1`: `epoch` (opaque string), `revision` (integer anchor revision),
`frame_id`, `anchor_stamp_ns`, `state_stamp_ns` (prediction target, integer ROS time),
`mean` (3), `covariance` (3x3), `valid`, `invalid_reason`, `motion_supported`,
`motion_support`. Support contains `start_stamp_ns`, `end_stamp_ns`, `source`, `supported`,
and `gaps` with integer endpoints and reason. An absent belief publishes an invalid event
with null mean/covariance/stamps/support; consumers must invalidate readiness first.

Each `(epoch, state_stamp_ns, revision)` is published at most once. Higher state time
supersedes older; at equal state time a higher anchor revision supersedes older. A changed
motion buffer cannot republish conflicting values with the same key. A backward clock
invalidates the runtime epoch/readiness until coordinated restart. No partial hot reset.
Invalid records do not make their diagnostic mean current merely because they were received.

## In-process planner API

`_ensure_belief_runtime_locked()` is called under `_data_lock` and initializes fixture
metadata safely. It adopts existing m/P/stamp only on initial setup. `_belief_record` is
an immutable `BeliefRecord`; `_commit_belief(m, P, stamp_msg, *, motion_support=None)`
validates then atomically installs it. Record fields are `mean`, `covariance`, `stamp_ns`,
`frame_id`, `epoch`, `revision`, `motion_support`. `.arrays()` returns independent arrays.

`_resolve_belief_for_planning()` returns the existing `(mean, covariance, meta)` tuple.
Unavailable/unsupported/invalid-epoch state returns `None, None, meta`. Metadata names:
`belief_epoch`, `belief_revision`, `belief_stamp_ns`, `prediction_stamp_ns`,
`belief_frame_id`, `belief_valid`, `motion_supported`, `motion_support`, `goal_revision`,
`invalid_reason`, plus compatibility `belief_stamp`, `belief_age_s`, `measurement_available`.
Mean/P/time/frame/revision in that result refer to one immutable anchor and motion snapshot.
`_current_belief_context()` provides current identity/validity for consumer installation
checks. `_belief_context_is_current(meta, require_revision=True)` checks epoch, frame,
goal revision, operational validity and, when requested, anchor revision. A consumer
chooses explicit revalidation on a changed revision; this helper does not change policy.

`_predict_belief_to_now` retains `(m,P)` return compatibility and optional `motion_snapshot`.
`MotionHistorySnapshot` freezes both motion sources and timestamped odometry headings.
`plan_replay` yields immutable segments and support. No corrected-belief displacement is
used as physical motion. Missing-motion assumptions remain explicit; unchanged numerical
recovery does not promote unsupported history to a measured current state.

## Correction assimilation

Topic remains `/planner/correction_assimilation`; JSON `schema_version=2` retains existing
`source_batch_id`, `correction_stamp`, `apply_stamp`, `belief_stamp_after`, `status`,
`reason`, `accepted`, `nis`. Add:

- `epoch`, `revision_before`, `revision_after`, `frame_id`;
- `correction_stamp_ns`, `apply_stamp_ns`, `belief_stamp_before_ns`, `belief_stamp_after_ns`;
- `prior_kind` (`prediction` or `anchor`), `prior_stamp_ns`, `prior_mean`, `prior_covariance`;
- `posterior_mean`, `posterior_covariance`, `state_stamp_ns` (actual committed after-time);
- `valid`, `motion_supported`, `motion_support`.

Bootstrap prior fields are null. Accepted/bootstrap/reanchored commits increase revision.
The posterior is the exact committed state, including held prediction/recovery; accepted
statuses alone select correction events for offline scoring. Full covariance retains all
cross terms. `frame_id` must be matched to the actual offline reference frame, not assumed.

`_correction_outcomes[source_batch_id]` retains an immutable mapping with these fields.
State/identity/outcome are retained before fallible diagnostic publication. Redelivery
does not produce a second dropped/not-newer terminal classification for a committed event.
Durable delivery/CSV persistence remains the logger's responsibility.

Manager correction-envelope schema 2 will retain schema-1 measurement fields and add
physical member/event identities and exact times. The state receiver will preserve those
identity fields in its terminal outcome; manager capture/common-time transformations remain
owned by 07 and are not inferred from planner-belief corrections.
