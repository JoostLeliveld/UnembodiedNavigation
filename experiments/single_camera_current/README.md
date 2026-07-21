# Single-Camera Current Result

This namespace freezes the current thesis evidence chain. It is not a new
experiment and it must not drift while multi-camera extension code is developed.

## Locked Evidence

- Active config: `scripts/visibility_comparison/warehouse_visibility_campaign.yaml`
- Runtime contract: `docs/current_runtime_contract.yaml`
- Result surface: `docs/paper_vs_current/current/`
- Detector checkpoint path: `logs/perception_models/warehouse_yolo_detector_v1/model.pt`
- GP artifact: `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz`
- Conditions: `C1` constant-R EFE and `C2` visibility-aware EFE
- Tasks: four locked warehouse routes
- Seeds: `0, 1, 2, 3, 4`

## Current Headline

The current packaged campaign is:

| Condition | Clean goals | Safety notes |
| --- | ---: | --- |
| C1 | 15/20 | 4/20 GT geometry breaches, 0 physics contacts |
| C2 | 20/20 | 0/20 GT geometry breaches |

## Freeze Rule

Extension work must not modify the active C1/C2 campaign, detector artifact, GP
artifact, covariance mapping, planner settings, current figures, or current metric
definitions. New code may add adapters, replay tools, or extension configs, but
the current result remains reproducible through the existing campaign command in
`experiments/README.md`.

## Reproduction Gate

After extension code is added, this test selection must still pass:

```bash
python3 -m pytest tests/visibility_comparison/test_current_runtime_contract.py
```

Full campaign reruns remain a separate long-running check.
