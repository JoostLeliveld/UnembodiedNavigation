# Frozen runtime inputs

The large detector weights and local study logs are intentionally not committed to ordinary
Git. A machine is eligible to produce evidence only when these exact bytes are present; the
run manifest hashes them again for every drive.

| input | repository-relative runtime path | SHA-256 |
|---|---|---|
| detector weights | `logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt` | `efff1949c1b8cdeeb11438b36de80f6cf8daeef5f3a4682cfce8ae7dfe314f34` |
| commissioned calibration | `logs/studies/measurement_commissioning/calibration.json` | `de578957683905763c5890e686345dee56cd583f5a5ade05b5050adc593ecc30` |

The campaign configuration itself is tracked at
`scripts/visibility_comparison/fusion_on_fixed_routes_campaign.yaml`; its hash changes with
an intentional repair and is therefore recorded dynamically in each run manifest rather
than copied here.

No paper result may depend on a local artifact merely existing. The explicit frozen-run
selection, per-run provenance hashes, schema-4 assimilation evidence, and model/calibration
hashes must all agree before scoring.
