"""Independent ROS-free assertions at the repaired cross-module joins.

These tests intentionally use public owner APIs.  They are not replacements for
the owner suites and must run only after the handoff preflight freezes a source.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for package in ("planning", "reliability", "unav_common"):
    sys.path.insert(0, str(ROOT / "src" / package))

from planning.core.belief_state import BeliefRecord, MotionSupport, PredictionSnapshot  # noqa: E402
from reliability.contracts import ContractValidationError  # noqa: E402
from reliability.manager_state import AdmissionBeliefHistory  # noqa: E402
from unav_common.correction_ledger import validate_correction_ledger  # noqa: E402
from unav_common.operational_belief import OperationalBeliefReceiver  # noqa: E402


EPOCH = "run-epoch-0001"
SOURCE_EPOCH = "detector-epoch-0001"
FRAME = "map_bev"
P = np.diag([0.04, 0.05, 0.01])


def snapshot(*, revision=3, anchor_ns=1_000_000_000, state_ns=1_300_000_000,
             supported=True, goal_revision=7):
    gaps = () if supported else ((1_100_000_000, 1_200_000_000, "missing_odom"),)
    support = MotionSupport(anchor_ns, state_ns, "wheel_odom", gaps)
    anchor = BeliefRecord.create([1.0, 2.0, 0.2], P, anchor_ns, FRAME, EPOCH,
                                 revision, support)
    return PredictionSnapshot.create(anchor, [1.2, 2.1, 0.3], P, state_ns, support,
                                     valid=supported,
                                     invalid_reason="" if supported else "unsupported_motion",
                                     goal_revision=goal_revision)


def invalid_payload(*, revision=4, anchor_ns=1_300_000_000, state_ns=1_500_000_000):
    return {
        "schema_version": 1,
        "initialized": True,
        "epoch": EPOCH,
        "revision": revision,
        "frame_id": FRAME,
        "anchor_stamp_ns": anchor_ns,
        "state_stamp_ns": state_ns,
        "mean": None,
        "covariance": None,
        "valid": False,
        "invalid_reason": "unsupported_motion",
        "motion_supported": False,
        "motion_support": {
            "start_stamp_ns": anchor_ns,
            "end_stamp_ns": state_ns,
            "source": "wheel_odom",
            "supported": False,
            "gaps": [{
                "start_stamp_ns": anchor_ns + 1,
                "end_stamp_ns": state_ns - 1,
                "reason": "missing_odom",
            }],
        },
    }


def canonical_terminal(raw_runtime_terminal):
    """Apply the documented runtime-schema-2 to common-ledger epoch mapping."""
    result = dict(raw_runtime_terminal)
    result["belief_epoch"] = result["epoch"]
    result["epoch"] = result.pop("source_epoch")
    return result


def test_one_prediction_snapshot_is_accepted_identically_by_both_consumers():
    prediction = snapshot()
    payload = prediction.to_dict()
    manager = AdmissionBeliefHistory(FRAME)
    receiver = OperationalBeliefReceiver(expected_frame=FRAME, max_age_s=1.0)

    assert manager.offer(payload)
    assert receiver.receive_json(json.dumps(payload), now_ns=1_350_000_000)
    operational = receiver.usable(now_ns=1_350_000_000)

    assert operational is not None
    assert manager.latest is not None
    assert (manager.latest.epoch, manager.latest.state_stamp_ns, manager.latest.revision) == operational.identity
    assert manager.latest.frame_id == operational.frame_id == prediction.anchor.frame_id
    assert manager.latest.mean == operational.mean == prediction.mean
    assert manager.latest.covariance == operational.covariance == prediction.covariance
    assert prediction.planner_meta()["belief_revision"] == operational.revision
    assert prediction.planner_meta()["prediction_stamp_ns"] == operational.state_stamp_ns

    # Both consumers own validated values rather than the producer's mutable JSON tree.
    payload["mean"][0] = 99.0
    payload["motion_support"]["source"] = "tampered"
    assert manager.latest.mean[0] == 1.2
    assert operational.mean[0] == 1.2
    assert operational.motion_support["source"] == "wheel_odom"


def test_motion_gap_invalidates_both_consumers_and_old_delivery_cannot_revive_it():
    valid = snapshot(revision=3).to_dict()
    invalid = invalid_payload()
    manager = AdmissionBeliefHistory(FRAME)
    receiver = OperationalBeliefReceiver(expected_frame=FRAME, max_age_s=1.0)

    assert manager.offer(valid)
    assert receiver.receive_json(json.dumps(valid), now_ns=1_350_000_000)
    assert manager.offer(invalid)
    assert receiver.receive_json(json.dumps(invalid), now_ns=1_500_000_000)
    assert manager.latest is not None and not manager.latest.valid
    assert receiver.usable(now_ns=1_500_000_000) is None
    assert receiver.reason == "unsupported_motion"

    assert manager.offer(valid) is False
    assert receiver.receive_json(json.dumps(valid), now_ns=1_510_000_000) is False
    assert manager.latest.revision == 4
    assert receiver.usable(now_ns=1_510_000_000) is None


def test_equal_anchor_revision_can_advance_prediction_but_cannot_conflict():
    old = snapshot(state_ns=1_300_000_000).to_dict()
    newer = snapshot(state_ns=1_400_000_000).to_dict()
    conflicting = json.loads(json.dumps(newer))
    conflicting["mean"][0] += 1.0
    manager = AdmissionBeliefHistory(FRAME)
    receiver = OperationalBeliefReceiver(expected_frame=FRAME, max_age_s=1.0)

    assert manager.offer(old)
    assert manager.offer(newer)
    assert manager.offer(old) is False
    with pytest.raises(ContractValidationError, match="conflicting"):
        manager.offer(conflicting)

    assert receiver.receive_json(json.dumps(old), now_ns=1_300_000_000)
    assert receiver.receive_json(json.dumps(newer), now_ns=1_400_000_000)
    assert receiver.receive_json(json.dumps(old), now_ns=1_410_000_000) is False
    assert receiver.receive_json(json.dumps(conflicting), now_ns=1_420_000_000) is False
    assert receiver.usable(now_ns=1_420_000_000) is None
    assert receiver.reason == "conflicting_belief_revision"


def test_terminal_assimilation_and_published_belief_join_on_exact_committed_identity():
    source_batch_id = "cycle-44"
    correction_ns = 1_000_000_000
    apply_ns = 1_200_000_000
    posterior = BeliefRecord.create([1.1, 2.0, 0.2], P, correction_ns, FRAME,
                                    EPOCH, 4, MotionSupport(correction_ns, correction_ns, "wheel_odom"))
    publication = {
        "source_batch_id": source_batch_id,
        "correction_stamp": correction_ns / 1e9,
        "correction_stamp_ns": correction_ns,
        "frame_id": FRAME,
        "epoch": SOURCE_EPOCH,
        "member_ids": ["camera-a:44", "camera-b:44"],
        "payload_sha256": "a" * 64,
    }
    runtime_terminal = {
        **publication,
        "schema_version": 2,
        "epoch": EPOCH,
        "source_epoch": SOURCE_EPOCH,
        "apply_stamp": apply_ns / 1e9,
        "apply_stamp_ns": apply_ns,
        "status": "accepted",
        "accepted": True,
        "reason": "",
        "revision_before": 3,
        "revision_after": posterior.revision,
        "state_stamp_ns": posterior.stamp_ns,
        "posterior_mean": list(posterior.mean),
        "posterior_covariance": [list(row) for row in posterior.covariance],
    }

    ledger = validate_correction_ledger(
        [publication], [canonical_terminal(runtime_terminal)],
    ).require_valid()
    stored = ledger.by_batch[source_batch_id]
    assert ledger.accepted_update_ids == (source_batch_id,)
    assert stored["epoch"] == SOURCE_EPOCH
    assert stored["belief_epoch"] == posterior.epoch
    assert stored["frame_id"] == posterior.frame_id
    assert stored["revision_after"] == posterior.revision
    assert stored["state_stamp_ns"] == posterior.stamp_ns == correction_ns
    assert tuple(stored["posterior_mean"]) == posterior.mean
    assert tuple(tuple(row) for row in stored["posterior_covariance"]) == posterior.covariance
    assert correction_ns < apply_ns  # input time and serialized commit time remain distinct


def test_reasoned_refusal_is_terminal_but_does_not_claim_a_recursive_update():
    publication = {
        "source_batch_id": "cycle-gap",
        "correction_stamp": 2.0,
        "correction_stamp_ns": 2_000_000_000,
        "frame_id": FRAME,
        "epoch": SOURCE_EPOCH,
    }
    runtime_terminal = {
        **publication,
        "schema_version": 2,
        "epoch": EPOCH,
        "source_epoch": SOURCE_EPOCH,
        "apply_stamp": 2.2,
        "apply_stamp_ns": 2_200_000_000,
        "status": "dropped",
        "accepted": False,
        "reason": "missing_motion_support",
        "revision_before": 4,
        "revision_after": 4,
    }
    ledger = validate_correction_ledger(
        [publication], [canonical_terminal(runtime_terminal)],
    ).require_valid()
    assert ledger.accepted_update_ids == ()
    assert ledger.refusal_counts["dropped"] == 1
    assert ledger.by_batch["cycle-gap"]["reason"] == "missing_motion_support"
    assert ledger.by_batch["cycle-gap"]["revision_before"] == ledger.by_batch["cycle-gap"]["revision_after"]
