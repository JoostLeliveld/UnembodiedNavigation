# Thesis support scripts

Only scripts needed by the current thesis pipeline remain in the working tree.

- `perception/`: capture, label, and train the frozen YOLO detector.
- `visibility_comparison/`: run and monitor the final navigation campaign.
- `shared/`: canonical path and metric helpers.
- `sim/`: fetch simulator assets.

The current commissioning and model-selection programs live under
`experiments/reference_controlled_commissioning_v1/`. Obsolete correction families,
availability prototypes, parameter searches, and conclusion-specific analyses are in the
cold archive outside this repository.

The method contract is `docs/COMMISSIONED_SENSOR_MODEL_CONTRACT.md`; the execution order is
`PLAN.md`.
