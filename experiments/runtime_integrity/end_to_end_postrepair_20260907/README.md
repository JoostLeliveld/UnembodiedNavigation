# Successor end-to-end acceptance workspace

This directory prepares the independent post-repair acceptance pass requested after the
historical audit-15 verdict. It does not revise
[`docs/module_audits/15_end_to_end_acceptance.md`](../../../docs/module_audits/15_end_to_end_acceptance.md)
or any evidence in `end_to_end_20260906/`.

The working tree is under active module-owner repair. Files here may inspect and test the
result after handoff, but they do not edit production code, frozen runs, scientific model
parameters, campaign selections or historical source snapshots. Defect-asserting audit
probes remain historical reproductions and are not included in a green acceptance suite.

## Execution gate

`run_acceptance.py` refuses to run until every required owner in `owner_handoffs.json` is
marked `ready` with a machine-readable source-bound handoff. Preflight checks its passing
command, stable-source flag and current file hashes. The coordinator's `READY.json` names
the intended source/config/schema identity and pins the SHA-256 of `owner_handoffs.json`.
This is a mechanical guard, not permission to launch a campaign.

The acceptance order is:

1. Freeze the coherent owner-handoff source and resolved configuration identity.
2. Run desired-invariant, ROS-free component tests.
3. Run the ROS-free cross-module transaction assertions for identity, belief, planning,
   correction accounting and motion-support invalidation.
4. If all deterministic checks pass, run one private bounded simulator scenario with the
   final actuator/contact/reset guards. No campaign is part of this workspace.
5. Run the recorded cross-chain assertions over the simulator fixture, then update the
   successor matrix from evidence. Preserve failures and reasoned refusals.

## Files

- `owner_handoffs.json`: current handoff/readiness snapshot.
- `scenario_contracts.json`: ten bounded scenarios and fourteen required invariants.
- `acceptance_commands.json`: component/cross-chain/physical commands, separated by gate.
- `matrix_update.py`: copies the historical 66-row matrix and adds successor state/evidence
  columns without modifying the old matrix.
- `preflight.py`: validates handoff completeness, desired test paths, and `READY.json`.
- `run_acceptance.py`: records exact source hashes and executes deterministic commands only
  after preflight. The isolated simulator and recorded-final stages remain separately gated.
- `test_cross_chain_contracts.py`: independent ROS-free assertions at public owner API joins.
- `test_recorded_cross_chain_evidence.py`: final assertions over the source-bound simulator
  adapter fixture named by `READY.json`.
- `recorded_chain_schema.md`: required evidence fields and the versioned physical-invocation
  identity interpretation.
- `missing_gates.md`: acceptance responsibilities not yet closed by component work.

The documentation-truthfulness invariant and the final cross-chain event join are explicit
acceptance work. Component pass counts do not satisfy either one.
