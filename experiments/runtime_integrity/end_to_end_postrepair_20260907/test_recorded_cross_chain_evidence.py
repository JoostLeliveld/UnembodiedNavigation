"""Assertions over the final source-bound recorded chain fixture.

The preflight supplies the fixture through READY.json.  Absence outside the final
stage is a skip so broad developer test runs do not masquerade as acceptance.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load_fixture():
    ready_path = HERE / "READY.json"
    if not ready_path.exists():
        pytest.skip("final source-bound READY.json has not been issued")
    ready = json.loads(ready_path.read_text())
    relative = ready.get("cross_chain_fixture")
    if not relative:
        pytest.fail("READY.json does not name cross_chain_fixture")
    path = ROOT / relative
    if not path.is_file():
        pytest.fail(f"recorded cross-chain fixture is absent: {relative}")
    return ready, json.loads(path.read_text())


@pytest.fixture(scope="module")
def evidence():
    return _load_fixture()


def test_fixture_exercises_all_ten_bounded_scenarios(evidence):
    _, data = evidence
    observed = {item["id"] for item in data["scenarios"] if item.get("exercised") is True}
    assert observed == {f"S{i}" for i in range(1, 11)}
    for item in data["scenarios"]:
        assert item.get("evidence_ids"), f'{item["id"]} has no source-bound evidence IDs'


def test_camera_outage_and_motion_support_are_independent(evidence):
    _, data = evidence
    by_scenario = {row["scenario_id"]: row for row in data["availability_intervals"]}
    complete_motion = by_scenario["S2"]
    assert complete_motion["camera_available"] is False
    assert complete_motion["motion_supported"] is True
    assert complete_motion["contains_turn"] is True
    assert complete_motion["motion_gap_ids"] == []
    gapped_motion = by_scenario["S3"]
    assert gapped_motion["camera_available"] is False
    assert gapped_motion["motion_supported"] is False
    assert gapped_motion["motion_gap_ids"]
    assert gapped_motion["invalid_belief_event_id"]
    assert gapped_motion["plan_refusal_event_id"]


def test_observation_delay_duplicate_reorder_and_simultaneity_stay_explicit(evidence):
    _, data = evidence
    rows = data["observation_deliveries"]
    assert rows
    assert [row["delivery_sequence"] for row in rows] == sorted(row["delivery_sequence"] for row in rows)
    by_observation = {}
    for row in rows:
        assert row["source_frame_id"] and row["detector_invocation_id"]
        by_observation.setdefault(row["observation_id"], []).append(row)
    duplicated = [group for group in by_observation.values() if len(group) > 1]
    assert duplicated
    assert all(sum(row["disposition"] == "accepted_for_fusion" for row in group) <= 1
               and all(row["disposition"] in {"accepted_for_fusion", "duplicate", "rejected"}
                       for row in group) for group in duplicated)
    capture_order = [row["capture_stamp_ns"] for row in rows]
    assert capture_order != sorted(capture_order), "no reordered delivery is present"
    assert any(row["delayed"] is True for row in rows)
    simultaneous = {}
    for row in rows:
        simultaneous.setdefault(row["capture_stamp_ns"], set()).add(row["observation_id"])
    assert any(len(observation_ids) > 1 for observation_ids in simultaneous.values())


def test_silence_rejection_and_recovery_have_reasoned_terminal_events(evidence):
    _, data = evidence
    liveness = data["liveness_events"]
    for channel in ("detector", "planner", "command"):
        matching = [row for row in liveness if row["channel"] == channel]
        assert matching and any(row["status"] == "silence_detected" for row in matching)
        assert all(row["reason"] and row["terminal_action"] for row in matching)
    admission = sorted(data["admission_events"], key=lambda row: row["event_sequence"])
    rejected_index = next(i for i, row in enumerate(admission)
                          if row["status"] == "rejected" and row["reason"])
    assert any(row["status"] == "accepted" and row["recovery_evidence_ids"]
               for row in admission[rejected_index + 1:])


def test_restart_closes_old_epoch_before_fresh_inputs_and_commands(evidence):
    _, data = evidence
    events = data["restart_events"]
    assert [row["sequence"] for row in events] == sorted(row["sequence"] for row in events)
    stages = [row["stage"] for row in events]
    required = [
        "old_epoch_stop", "old_epoch_drain", "process_restart", "fresh_clock",
        "fresh_odometry", "fresh_camera", "fresh_belief", "fresh_command",
    ]
    assert all(stage in stages for stage in required)
    assert [stages.index(stage) for stage in required] == sorted(stages.index(stage) for stage in required)
    old_epoch = next(row["run_epoch"] for row in events if row["stage"] == "old_epoch_stop")
    new_epoch = next(row["run_epoch"] for row in events if row["stage"] == "fresh_belief")
    assert old_epoch != new_epoch
    restart_sequence = next(row["sequence"] for row in events if row["stage"] == "process_restart")
    assert all(row["run_epoch"] == new_epoch for row in events
               if row["sequence"] > restart_sequence and row["stage"].startswith("fresh_"))


def test_manifest_binds_every_required_identity(evidence):
    ready, data = evidence
    manifest = data["manifest"]
    required = {
        "source_identity", "resolved_config_identity", "model_identity",
        "calibration_identity", "world_identity", "robot_identity",
        "route_identity", "attempt_identity",
    }
    assert required <= manifest.keys()
    for name in required:
        assert isinstance(manifest[name], str) and manifest[name].strip()
    assert manifest["source_identity"] == ready["source_identity"]
    assert manifest["resolved_config_identity"] == ready["resolved_config_identity"]


def test_each_physical_detector_invocation_has_unique_lossless_identity(evidence):
    ready, data = evidence
    invocations = data["detector_invocations"]
    source_ids = [row["source_batch_id"] for row in invocations]
    invocation_ids = [row["invocation_id"] for row in invocations]
    assert invocations
    assert all(isinstance(value, str) and value.strip() for value in source_ids + invocation_ids)
    assert len(invocation_ids) == len(set(invocation_ids))
    contract = ready["source_batch_identity_contract"]
    if contract == "per_physical_invocation":
        assert len(source_ids) == len(set(source_ids))
    else:
        assert contract == "logical_cycle_plus_invocation_v2"
        batch_outcomes = data["detector_batch_outcomes"]
        assert len({row["source_batch_id"] for row in batch_outcomes}) == len(batch_outcomes)
        membership = {}
        for row in batch_outcomes:
            assert row["status"] in {"observed", "miss", "silence", "rejected"}
            if row["status"] != "observed":
                assert isinstance(row["reason"], str) and row["reason"].strip()
            for invocation_id in row["member_invocation_ids"]:
                assert invocation_id not in membership
                membership[invocation_id] = row["source_batch_id"]
        assert set(membership) == set(invocation_ids)
        assert all(membership[row["invocation_id"]] == row["source_batch_id"] for row in invocations)


def test_capture_arrival_and_detector_times_remain_separate(evidence):
    _, data = evidence
    for row in data["detector_invocations"]:
        fields = ("capture", "arrival", "detector_start", "detector_end")
        assert all(isinstance(row[f"{name}_stamp_ns"], int) and row[f"{name}_stamp_ns"] >= 0
                   for name in fields)
        assert all(isinstance(row[f"{name}_clock_domain"], str)
                   and row[f"{name}_clock_domain"] for name in fields)
        for before, after in zip(fields, fields[1:]):
            if row[f"{before}_clock_domain"] == row[f"{after}_clock_domain"]:
                assert row[f"{before}_stamp_ns"] <= row[f"{after}_stamp_ns"]


def test_every_fused_correction_has_exactly_one_terminal_outcome(evidence):
    _, data = evidence
    corrections = data["corrections"]
    terminals = data["assimilation_outcomes"]
    counts = Counter(row["source_batch_id"] for row in terminals)
    expected = {row["source_batch_id"] for row in corrections}
    assert expected
    assert set(counts) == expected
    assert all(counts[key] == 1 for key in expected)
    correction_by_id = {row["source_batch_id"]: row for row in corrections}
    invocation_ids = {row["invocation_id"] for row in data["detector_invocations"]}
    for row in corrections:
        assert row["member_invocation_ids"]
        assert set(row["member_invocation_ids"]) <= invocation_ids
    for row in terminals:
        assert row["source_epoch"] == correction_by_id[row["source_batch_id"]]["source_epoch"]
        assert isinstance(row["epoch"], str) and row["epoch"]
        assert row["status"] in {"accepted", "accepted_bootstrap", "reanchored", "rejected", "dropped"}
        assert row["accepted"] is (row["status"] in {"accepted", "accepted_bootstrap", "reanchored"})
        if not row["accepted"]:
            assert isinstance(row["reason"], str) and row["reason"].strip()


def test_accepted_terminal_outcomes_match_exact_belief_revisions(evidence):
    _, data = evidence
    beliefs = {
        (row["epoch"], row["revision"], row["anchor_stamp_ns"]): row
        for row in data["beliefs"]
    }
    for row in data["assimilation_outcomes"]:
        if not row["accepted"]:
            assert row["measurement_update_committed"] is False
            if row["revision_after"] != row["revision_before"]:
                assert row["revision_cause"] in {"motion_prediction", "recovery"}
                key = (row["epoch"], row["revision_after"], row["state_stamp_ns"])
                assert key in beliefs
            continue
        assert row["measurement_update_committed"] is True
        key = (row["epoch"], row["revision_after"], row["state_stamp_ns"])
        assert key in beliefs
        belief = beliefs[key]
        assert row["frame_id"] == belief["frame_id"]
        assert row["posterior_mean"] == belief["anchor_mean"]
        assert row["posterior_covariance"] == belief["anchor_covariance"]


def test_recursive_state_changes_only_through_authorized_transactions(evidence):
    _, data = evidence
    commits = data["state_commits"]
    assert commits
    assert len({row["transaction_id"] for row in commits}) == len(commits)
    authorized = set(data["authorized_state_transactions"])
    for row in commits:
        assert row["transaction_kind"] in authorized
        assert row["writer"] == data["authorized_state_writer"]
        if row["transaction_kind"] == "coordinated_restart":
            assert row["epoch_before"] != row["epoch_after"]
            assert row["revision_after"] >= 0
        else:
            assert row["epoch_before"] == row["epoch_after"]
            assert row["revision_after"] > row["revision_before"]


def test_beliefs_and_plans_use_coherent_snapshots_and_stale_results_do_not_install(evidence):
    _, data = evidence
    identities = {
        (row["epoch"], row["revision"], row["anchor_stamp_ns"], row["state_stamp_ns"], row["frame_id"])
        for row in data["beliefs"]
    }
    assert identities
    for row in data["plans"]:
        key = (row["belief_epoch"], row["belief_revision"], row["belief_stamp_ns"],
               row["prediction_stamp_ns"], row["belief_frame_id"])
        assert key in identities
        if row["installed"]:
            assert row["request_immutable_token"] == row["commit_immutable_token"]
            if row["request_token"] != row["commit_token"]:
                revalidation = row["newer_belief_revalidation"]
                assert revalidation["accepted"] is True
                assert revalidation["evidence_id"]
                revalidated_key = (
                    revalidation["belief_epoch"], revalidation["belief_revision"],
                    revalidation["belief_stamp_ns"], revalidation["prediction_stamp_ns"],
                    revalidation["belief_frame_id"],
                )
                assert revalidated_key in identities
        else:
            assert row["rejection_reason"] in {
                "stale_belief", "stale_goal", "stale_config", "stale_epoch",
                "expired", "cancelled", "infeasible", "invalid_result",
            }


def test_command_application_stop_contact_and_terminal_status_are_distinct(evidence):
    _, data = evidence
    commands = data["commands"]
    assert commands
    assert len({row["command_id"] for row in commands}) == len(commands)
    for row in commands:
        assert row["receipt_stamp_ns"] <= row["forward_stamp_ns"]
        assert row["receipt_clock_domain"] == row["forward_clock_domain"]
        assert row["application_observed"] is False
        assert row.get("application_stamp_ns") is None
    forwarded_stops = [row for row in commands if row["forwarded_linear"] == 0 and row["forwarded_angular"] == 0]
    assert forwarded_stops
    rest_contract = data["rest_evidence"]
    stop_by_id = {row["command_id"]: row for row in forwarded_stops}
    assert rest_contract["trigger_command_id"] in stop_by_id
    trigger = stop_by_id[rest_contract["trigger_command_id"]]
    if trigger["forward_clock_domain"] == rest_contract["clock_domain"]:
        assert rest_contract["interval_start_ns"] >= trigger["forward_stamp_ns"]
    assert 0 <= rest_contract["interval_start_ns"] < rest_contract["interval_end_ns"]
    linear_limit = rest_contract["linear_tolerance_mps"]
    angular_limit = rest_contract["angular_tolerance_radps"]
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and value >= 0 for value in (linear_limit, angular_limit))
    rest = [row for row in data["odometry"]
            if row["clock_domain"] == rest_contract["clock_domain"]
            and rest_contract["interval_start_ns"] <= row["stamp_ns"] <= rest_contract["interval_end_ns"]]
    assert len(rest) >= 2
    assert [row["stamp_ns"] for row in rest] == sorted(row["stamp_ns"] for row in rest)
    assert rest[0]["stamp_ns"] == rest_contract["interval_start_ns"]
    assert rest[-1]["stamp_ns"] == rest_contract["interval_end_ns"]
    assert all(math.isfinite(row["linear_speed"]) and math.isfinite(row["angular_speed"])
               and abs(row["linear_speed"]) <= linear_limit
               and abs(row["angular_speed"]) <= angular_limit for row in rest)
    assert data["contacts"]["health_events"]
    assert data["contacts"]["physical_contact_events"]
    assert data["mission_terminal"]["status"] in {"succeeded", "failed", "reasoned_refusal", "shutdown"}
    assert data["mission_terminal"]["event_id"] not in {row["event_id"] for row in data["contacts"]["physical_contact_events"]}
    mission_stages = [row["stage"] for row in sorted(data["mission_events"], key=lambda row: row["sequence"])]
    for stage in ("goal_approach", "stop_requested", "physical_rest_verified", "contact_probe", "terminal_status"):
        assert stage in mission_stages


def test_command_owner_stop_precedence_and_limits_match_forwarded_commands(evidence):
    _, data = evidence
    ownership = data["command_ownership"]
    assert ownership["guard_input_publishers"] == [ownership["authorized_adapter"]]
    assert ownership["diffdrive_input_publishers"] == [ownership["authorized_guard"]]
    assert ownership["stop_precedence"] == ["fatal", "restart_or_rewind", "watchdog", "mission", "planner"]
    limits = data["resolved_limits"]
    canonical = limits["canonical"]
    assert canonical["linear_unit"] == "m/s" and canonical["angular_unit"] == "rad/s"
    for stage in ("planner", "tracker", "adapter", "guard"):
        assert limits[stage] == canonical
    for row in data["commands"]:
        assert abs(row["forwarded_linear"]) <= canonical["max_abs_linear"]
        assert abs(row["forwarded_angular"]) <= canonical["max_abs_angular"]


def test_logger_drain_and_offline_decision_make_terminal_accounting_reconstructible(evidence):
    _, data = evidence
    summary = data["run_summary"]
    terminal_times = [row["stamp_ns"] for row in data["terminal_events"]]
    assert summary["producer_cutoff_stamp_ns"] >= max(terminal_times)
    assert summary["queue_drained"] is True
    assert summary["atomic_summary_committed"] is True
    assert summary["write_failures"] == data["logger_write_failures"]
    shutdown = data["shutdown_accounting"]
    assert shutdown["producer_cutoff_sequence"] < shutdown["drain_complete_sequence"]
    assert shutdown["drain_complete_sequence"] < shutdown["files_closed_sequence"]
    assert shutdown["files_closed_sequence"] < shutdown["summary_committed_sequence"]
    assert max(row["logger_receipt_sequence"] for row in data["terminal_events"]) <= shutdown["drain_complete_sequence"]
    assert set(shutdown["buffered_terminal_event_ids"]) <= {row["event_id"] for row in data["terminal_events"]}
    offline = data["offline_alignment"]
    assert offline["decision"] in {"accepted", "rejected"}
    assert offline["reason_codes"]
    assert offline["manifest_identity"] == data["manifest"]["attempt_identity"]
    assert offline["ground_truth_used_online"] is False
