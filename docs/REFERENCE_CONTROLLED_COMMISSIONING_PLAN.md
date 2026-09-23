# Reference-controlled camera commissioning plan

Current as of 2026-09-10. This is the prospective sensor-model plan. It does not change
existing run selections or promote development results in
`localization_metrics_registry.json`.

## Objective

Turn a validated YOLO detector into a commissioned camera-position observation model.
The model consists of:

```text
frozen YOLO
-> fixed belief-independent observation-support checks
-> bottom-centre ground-plane projection
-> selected belief-independent position correction
-> correction-matched conditional measurement covariance R_hit
```

NIS is part of estimator assimilation. It is not part of detector validation or camera
observation support.

## Study boundary

The work has three experimental stages.

```text
separate YOLO poses
-> detector training and held-out detector validation

reference-controlled commissioning drives
-> correction/R-pair selection and held-out camera-reading evaluation

navigation drives
-> frozen commissioned sensor model, estimator and planner evaluation
```

The stages answer different questions.

| Stage | Quantity evaluated | Permitted conclusion |
|---|---|---|
| YOLO development | detector output per attempted image | The detector returns the robot box with stated recall and false-positive rate. |
| Commissioning drives | individual camera reading at its capture timestamp | The selected correction changes the error of admitted camera readings, and its matched `R_hit` describes their residual distribution to the reported degree. |
| Navigation drives | recursive belief and physical outcome | The frozen sensor model changes belief or navigation outcomes under the tested conditions. |

Camera-reading error, fused-correction error and belief error remain separate quantities.

## Terminology

- **Detector:** frozen YOLO model that returns a box, confidence and class, or an explicit
  miss.
- **Observation support:** deterministic checks that decide whether a returned box can
  support the declared ground-plane measurement.
- **Raw camera reading:** ground-plane point obtained from the admitted box bottom-centre.
- **Corrected reading:** raw camera reading plus the correction predicted from
  runtime-available inputs.
- **`R_hit`:** residual covariance of a camera reading conditional on the frozen observation
  support rule admitting it.
- **NIS rejection:** estimator refusal because a measurement disagrees with the predicted
  belief relative to `P` and `R_hit`.
- **Commissioning drive:** a reference-controlled acquisition run. It is not a navigation
  trial.

Do not call an NIS rejection a bad bounding box. Do not call a detector miss a high-noise
measurement.

## Frozen detector stage

Use a detector dataset separate from the commissioning drives. Bind the following items in
one detector manifest:

- YOLO fit pose IDs;
- detector-validation pose IDs;
- spatial split and exclusion buffers;
- headings and camera IDs;
- image and label hashes;
- detector configuration and weight hash;
- box selection and confidence rules;
- every negative image and failed capture.

Keep every camera and heading from one physical pose in the same split. Report attempted
images, supported positives, negatives, recall, precision, false positives and performance
by camera and projected target size. State the exact IoU and confidence criteria.

Freeze YOLO before collecting or processing the commissioning drives. Retrain it only when
held-out detector evidence shows a detector failure that prevents the camera-position study.

## Deterministic observation support

The support rule uses no robot belief, estimator covariance, native pose or ground-truth
error. It applies four checks:

1. The fixed detector returns the target class with confidence of at least 0.25.
2. The box coordinates are finite and its width and height are at least 16 source-image pixels.
3. The complete box is at least five source-image pixels from every image boundary.
4. The box bottom-centre has a finite calibrated floor intersection.

Each refusal receives one or more explicit reason codes. A miss and a refusal are separate
outcomes. Record both in the denominator of camera opportunities.

The rule excludes expected-size checks and comparisons against a box projected from the
current robot belief. A belief-dependent check may remain as a diagnostic ablation. It may
not define the correction-fit population or the primary commissioned observation model.

The completed drives preserve `config/commissioning_sensor_gate_v1.yaml` as collection
provenance. `config/commissioning_sensor_gate_v2.yaml` is the analysis and deployment rule;
replay it from the recorded detector and projection fields before fitting correction,
matched covariance and planning-information fields. Its validator fails if association, tracking or
localizer acceptance is enabled. Freeze the complete sensor model before opening the held-out
commissioning audit.

## Reference-controlled acquisition

A simulator-native pose controller follows predeclared warehouse trajectories. Use smooth
physics-based motion rather than continuous teleportation. Collect stationary heading
anchors separately when needed.

For every attempted camera frame, record:

- source-frame identity and image hash;
- camera ID and calibration identity;
- integer capture timestamp;
- native pose and yaw at that timestamp;
- odometry and executed motion evidence;
- robot speed and commanded motion;
- YOLO miss or selected box and confidence;
- deterministic support outcome and reason;
- raw projected position when available;
- candidate-corrected position when available;
- deployed feature values required to replay the correction.

Native pose may control the acquisition path and provide offline fit targets and evaluation
references. It may not enter YOLO, observation support, correction inputs, `R_hit` features, NIS,
the online estimator or the planner.

Retain misses, truncated boxes, partial views, refusals and failed processing outcomes.
Do not select locations after observing detector or correction performance.

## Commissioning-drive partition

Assign complete drives, passes or buffered spatial panels before model fitting.

### Fit partition

Use the fit drives to:

- fit the candidate position corrections;
- select the simplest adequate correction configuration using only declared development
  folds;
- produce whole-drive out-of-fold residuals and fit one matched covariance per correction;
- retain all camera opportunities for the direct expected-information target; misses and
  refusals contribute zero information rather than being discarded.

Train the correction only on detector outputs positively associated with the robot and
admitted by the fixed support rule. Detector false positives have no valid correction target.

### Held-out commissioning partition

Use the held-out drives once the correction and support rule are frozen. Evaluate:

- raw and candidate-corrected readings on the same admitted detections;
- support-rule retention and refusal by reason;
- post-correction residual bias, spread and tail behaviour;
- candidate `R_hit` calibration and sharpness;
- camera, range, heading, speed, truncation and occlusion strata with adequate support.

Do not randomly split video frames. Adjacent frames are correlated. Keep synchronized views
from all cameras in the same partition. Aggregate or bootstrap by complete drive, pass,
transect or buffered spatial panel.

## Correction candidates

The selected model corrects the position reading. It does not decide whether the current
robot belief is correct, and it does not read native pose online. Candidate families may use
raw-box and camera geometry, a geometry-first representation, or RGB context; the raw reading
remains the no-correction baseline.

For each admitted reading at capture timestamp `t`, define:

```text
e_raw(t) = z_raw(t) - p_GT(t)
e_corr(t) = z_corr(t) - p_GT(t)
```

Fit the correction target from `p_GT(t) - z_raw(t)` using only fit drives. Runtime inputs may
include the current image crop, box geometry, detector confidence, raw projection, camera
identity and fixed camera geometry. List the exact inputs and excluded fields in the frozen
artifact.

All model selection and out-of-fold prediction use complete drives as the split unit. The
primary comparison is paired on the same held-out readings:

- raw median, RMSE and 95th-percentile position error;
- corrected median, RMSE and 95th-percentile position error;
- signed along-camera and across-camera bias;
- fraction improved and fraction worsened;
- correction of difficult but admitted observations;
- per-camera and declared operational-stratum results.

Report sample counts and opportunity coverage beside conditional accuracy.

## Conditional covariance

Estimate a fresh `R_hit` after fitting each correction candidate:

```text
R_hit,i,c = Cov[e_corr,c | camera i, support rule passed]
```

Start with the simplest model and stop when the next rung does not improve held-out proper
score and calibration at comparable sharpness:

```text
R0  one global isotropic covariance
R1  one isotropic covariance per camera
R2  one full 2 x 2 covariance per camera
R3  declared dependence on runtime-observable range or box strata
```

Report 50%, 90% and 95% containment with ellipse size. Inspect signed residual means and
tail shape. A covariance does not repair a wrong conditional mean, and an `R` learned for
one correction may not be used to score another. Fit `R` last, after temporal and
simultaneous cross-camera dependence has been measured.

`R_hit` is conditional on a measurement arriving and passing the support rule. A miss does
not generate an update with a large covariance.

## NIS and estimator boundary

After commissioning, the estimator receives `z_corr` and the matched `R_hit`. It computes:

```text
nu  = z_corr - H m_minus
S   = H P_minus H^T + R_hit
NIS = nu^T S^-1 nu
```

NIS may reject an ordinary update. Its decision depends on the predicted belief and its
covariance. It does not retroactively classify the image or box as unsupported.

Log measurement identity, prior belief, `R_hit`, innovation, innovation covariance, NIS,
threshold and terminal decision. Persistent NIS rejection is a recovery condition. Any
multi-camera relocalization rule must be specified and evaluated separately from camera
commissioning.

## Commissioned sensor-model freeze

The freeze bundle contains:

- detector architecture, configuration and weight hash;
- camera registry and calibration hashes;
- box selection and confidence rules;
- deterministic support-rule implementation, thresholds and reason-code registry;
- projection implementation and robot/camera geometry identities;
- selected correction architecture, input contract, weights and hash;
- `R_hit` model, parameters, fit selection and hash;
- exact software source identity;
- fit and held-out commissioning manifests.

The deployed camera model returns one terminal outcome for every camera opportunity:

```text
miss
refused:<reason>
measurement:<z_corr,R_hit>
```

## Navigation stage

Navigation runs use the frozen commissioned sensor model:

```text
camera image
-> frozen YOLO
-> deterministic support
-> frozen selected correction
-> frozen matched R_hit
-> NIS and estimator
-> planner
```

Native pose is an offline reference only. It does not control the navigation run or enter
the camera model, estimator, planner, goal decision or stuck decision.

Score individual readings at their capture timestamps and beliefs at their belief
timestamps. Use the manifest and loader required by `localization_metrics.md` and
`localization_metrics_registry.json`. Report correction refusal fraction and longest
correction gap beside belief accuracy.

All navigation conditions use identical route candidates and identical geometric and
rollout feasibility. The planner sums the expected-information fields of its configured
active cameras. These fields change belief prediction and route cost but do not remove
candidates. The primary estimator is one robot filter that consumes each camera frame once.
If commissioned residuals justify persistent camera-error states, use an augmented joint
filter rather than cascading two filters over the same evidence.

## Minimal ablations

Avoid a full factorial study. Retain only comparisons needed to identify the sensor model:

1. Raw projection versus each candidate correction on identical held-out admitted readings.
2. Minimal deterministic support versus the final deterministic support rule when the final
   rule contains additional geometry checks.
3. The prior belief-dependent admission rule as a diagnostic ablation when replay support is
   available.
4. One estimator arm with the frozen sensor model and declared NIS policy.

An offline ground-truth refusal is an oracle diagnostic only. It may not be presented as a
deployable gate.

## Completion criteria

The sensor model is ready for navigation only when:

- detector fit and validation identities are frozen;
- all commissioning opportunities have terminal outcomes;
- fit and held-out drives are manifest-bound and disjoint;
- the selected correction/`R` pair improves the declared held-out camera-reading and
  calibration criteria without an unacceptable tail or retention loss;
- the selected `R_hit` passes the declared held-out calibration criterion at useful
  sharpness;
- support decisions use no belief or evaluation-only evidence;
- all runtime artifacts and software identities are frozen;
- the full online path reproduces the offline camera reading for the same source frame.

Failure of one criterion stops the navigation claim. It does not delete misses, refusals or
failed runs from the report.

## Claims excluded by commissioning

Commissioning drives alone do not establish:

- lower recursive belief error;
- calibrated fused uncertainty;
- improved route choice;
- faster navigation;
- lower collision risk;
- a benefit from the planner's future-observation model.

Those claims require separately selected navigation evidence.
