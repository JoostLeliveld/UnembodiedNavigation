"""Create a successor acceptance matrix without rewriting historical audit evidence."""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "end_to_end_20260906" / "matrix.tsv"
OUT = HERE / "successor_matrix.tsv"

GROUPS = {
    "sensor_manager_fusion": {1, 2, 3, 4, 5, 6, 7, 20, 23, 24, 25, 26, 46, 47, 48, 49, 64},
    "state_timing_transactions": {8, 9, 10, 11, 12, 13, 14, 21, 22, 27, 28, 60, 61},
    "planner_route_command_mission": {15, 16, 17, 18, 30, 31, 32, 33, 34, 35, 50, 51, 52, 53},
    "ledger_logger_offline": {29, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 58, 63},
    "configuration_capture_provenance": {54, 55, 56, 57, 59},
    "simulator_pure_and_private": {62},
    "isolated_policy_or_claim_gate": {19},
    "historical_scoped_repair_only": {65, 66},
}

OWNER_STATE = {
    "sensor_manager_fusion": "WAIT: 04 source-bound handoff ready; 05/07/10 integration handoffs pending",
    "state_timing_transactions": "WAIT: 01 belief/schema implementation and 02 independent desired tests pending",
    "planner_route_command_mission": "WAIT: 03 source-bound handoff ready; 01/02 integration and 08/09 final handoffs pending",
    "ledger_logger_offline": "WAIT: shared validator source-bound handoff ready; 10/11 producer, drain and consumer integration pending",
    "configuration_capture_provenance": "WAIT: 12/13 attempt/config/artifact integration pending",
    "simulator_pure_and_private": "WAIT: 14 guard wiring and private physical verification pending",
    "isolated_policy_or_claim_gate": "ISOLATED: preserve declared policy/model; no correctness bundle",
    "historical_scoped_repair_only": "BASELINE: retained scope evidence; must be rerun under successor identity",
}

with OLD.open(newline="") as stream:
    reader = csv.reader(stream, delimiter="\t")
    header, *rows = list(reader)

assigned: dict[int, str] = {}
for group, ids in GROUPS.items():
    for number in ids:
        if number in assigned:
            raise RuntimeError(f"M{number:02} assigned twice")
        assigned[number] = group

expected = {int(row[0][1:]) for row in rows}
if set(assigned) != expected:
    raise RuntimeError(f"matrix assignment mismatch missing={sorted(expected-set(assigned))} extra={sorted(set(assigned)-expected)}")

new_header = header + ["Successor evidence group", "Current successor state", "Acceptance disposition"]
new_rows = []
for row in rows:
    number = int(row[0][1:])
    group = assigned[number]
    disposition = "WAIT"
    if group == "isolated_policy_or_claim_gate":
        disposition = "ACCEPT ONLY AS DECLARED LIMIT"
    elif group == "historical_scoped_repair_only":
        disposition = "RERUN; DO NOT CARRY PASS FORWARD"
    new_rows.append(row + [group, OWNER_STATE[group], disposition])

with OUT.open("w", newline="") as stream:
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(new_header)
    writer.writerows(new_rows)

print(f"wrote {len(new_rows)} successor rows to {OUT}")
