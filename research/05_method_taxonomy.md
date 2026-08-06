# Observation-quality method taxonomy

| Family / method | Operational information | Cold start | Commissioning | Adaptation | Expected failure | Fallback | Evidence |
|---|---|---:|---:|---|---|---|---|
| Geometric: constant | Camera identity at most | Yes | None | None | Ignores all spatial structure | Conservative constant | Baseline ready |
| Geometric: range/FOV | Candidate pose, calibrated pose and optics | Yes | Calibration only | Geometry update | Predicts visible through occluders | Constant or unavailable | Held-out null/competitive result exists |
| Geometric: depth/raycast | Candidate pose, camera model, depth/occupancy with provenance | Yes if map exists | Map/sensor setup | Rescan or live sensing | Stale, missing, or misregistered geometry | FOV or conservative unknown | Infrastructure exists; benchmark pending |
| Learned: GP | Candidate pose and fitted operational field | No | Labelled commissioning route | Refit/update | Unsupported regions and distribution shift | Geometric prior | Current GP ties/loses to FOV-range |
| Learned: DL challenger | Features available at future candidate poses | No | Dataset and calibration | Retrain/update | OOD geometry and miscalibrated probability | Geometric prior | Gated; not yet admitted |
| Hybrid | Geometric prior plus operational updates | Partial | Calibration plus samples | Online/post-run update | Wrong prior not overridden by evidence | Revert to prior with epistemic flag | Planned |

Instantaneous detector confidence, current detection validity, and recent observations are
camera-management signals. They are not legal planning-time predictors for future candidate
poses unless a causal forecast of them is explicitly defined.

Every method is evaluated on prediction, route discrimination, navigation consequence, and
deployment cost in that order. A method that fails an earlier gate consumes no campaign time.
