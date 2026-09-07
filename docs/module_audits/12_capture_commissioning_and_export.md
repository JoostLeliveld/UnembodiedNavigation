# 12 — Capture integrity, commissioning and runtime export

Reviewed 2026-09-06. **The frozen detector/NN/reference artifacts and saved pixels agree with their recorded identities, and the tested offline/runtime numerical mappings agree. The main confirmed defects are upstream: capture freshness is based on callback arrival, the capture writer bypasses the tested resume audit, and some exporters can certify incomplete or changed inputs.** These findings do not justify replacing the NN, fitting a richer covariance, or claiming a navigation benefit.

This audit changed only uniquely named `12_*` audit files. No detector or NN was trained, replaced or modified; no capture, campaign, simulator or ROS runtime process was started, stopped or reset. Temporary synthetic files exercise real functions and an extracted, unchanged resume branch. Passing defect probes means reproduction, not repair. No recorded-run accuracy, RMSE, NEES or coverage was calculated.

Read first: repository AGENTS.md and PLAN.md, localization metrics contract/registry, open questions, runtime integrity audit, and completed audits 02–05. Audits 01/06/07 became available during the investigation and their relevant ownership/results were checked. The September 6 plan update governs; old characterization and development data do not become confirmatory through successful software checks.

## Ranked findings

P1 means fix the integrity boundary before relying on another capture/export through it. P2 means a conditional, auxiliary-path or narrower defect. Ranking describes consequences and reachability, not observed incidence in a drive.

| Rank | ID / status | Finding | Reachability |
|---|---|---|---|
| 1 | C12-01 / P1 confirmed | Writer resume accepts prefixes the regression-tested preflight refuses; even the preflight does not verify common commanded pose | Current `capture_bbox_grid.py --resume`; not the already frozen completed capture |
| 2 | C12-02 / P1 confirmed | Delayed/repeated old frames satisfy “fresh” capture after a new command | Current static capture acquisition, including its repeat mode |
| 3 | C12-03 / P1 confirmed | Detector exporter trusts declared image hashes; interpretation import does not enforce its upstream file digests | Current offline capture→detector→interpretation commands, on changed inputs |
| 4 | C12-04 / P2 confirmed | Capture failures disappear from downstream attempted-opportunity counts | Captures with failed batches; the active frozen field has none |
| 5 | C12-05 / P2 confirmed | Lossless storage conversion is not interruption-safe | Explicit emergency conversion of a stopped `status=running` capture |
| 6 | C12-06 / P2 confirmed | Sparse out-of-fold path silently returns in-sample residuals | Older covariance-ladder helper on small datasets; not the selected current R fit |
| 7 | C12-07 / P2 confirmed, overlaps 05/13 | Runtime artifact validation does not enforce the complete camera/geometry/data provenance contract | Loader boundaries; exact active checkpoint is protected by the reference-calibration hash |
| 8 | C12-08 / P2 confirmed | Legacy operational exporter can use a later belief, duplicate a detection, and assume reception | Historical single-camera `warehouse_aws` adapter; not current network commissioning |
| 9 | C12-09 / P2 confirmed | Operational residual helper can count repeated evidence and misinterpret a projection Jacobian as the observation function | Auxiliary residual estimator; absent from the selected active mean/R path |
| 10 | C12-10 / P2 confirmed | Provisional residual dataset builder silently truncates short backend output and writes `.complete` | Alternative pixel-residual pipeline; not the active box-feature NN |

Split proximity, one-sample covariance identifiability, false-positive coverage, static versus operational availability, and previously inspected holdouts are **dataset/model limitations**, discussed separately below. No exact successful-image leakage across the selected outer roles was found.

## Active artifact and source boundary

The registry names `network_navigation_runtime_pilot` and the separate `network_navigation_recovery_pilot`. Their YAMLs retain native YOLO, five cameras A–E, 1280×720 originals, inference size 960, score threshold 0.25, IoU 0.45, masks off, strict 0.05 s batches, `learned_nn`, `commissioned_reference_r`, zero fixed offset, and robust `joint_network` fusion. The recovery follow-up changes controller behavior, not these artifacts. No newest-directory selection was used.

| Exact artifact, relative to repository | SHA-256 prefix | Role / verification |
|---|---|---|
| `logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt` | `efff1949c1b8cdee` | Frozen detector; matches the capture detector manifest and the historical commissioning declaration |
| `logs/perception_models/box_feature_bias_correction_20260831/models.joblib` | `5da33c63a2bb46b3` | Active metric correction; schema `box_feature_bias_correction.joblib.v1` |
| `logs/studies/icra_commissioning_20260905/models.joblib` | `84454e296a9a5c83` | Frozen per-camera residual offset and covariance models |
| `logs/studies/icra_commissioning_20260905/network_planner/reference_calibration.json` | `0a462258197b4253` | Active post-NN offset/full world R; binds the exact NN hash |
| `logs/studies/icra_commissioning_20260905/field_study/field.joblib` | `00cd08656e2e8afc` | Frozen commissioning field, before gridded planner export |
| `logs/studies/icra_commissioning_20260905/network_planner/uniform.npz` | `9a312f1970c1689c` | P0 score/q/R arrays |
| `logs/studies/icra_commissioning_20260905/network_planner/geometry.npz` | `7a3f1c535c913731` | P1 score/q/R arrays |
| `logs/studies/icra_commissioning_20260905/network_planner/gp.npz` | `2564739e96dc30d4` | P2 score/q/R arrays |
| `logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/capture_manifest.json` | `d1b4d9ee3eefe6ce` | Frozen **v1** capture, not today's v2 protocol |
| same capture, `capture_index.csv` | `444eacca6ddcb210` | 15,440 camera opportunities |
| `logs/studies/measurement_commissioning/calibration.json` | `de57895768390576` | Older pixel sigma/geometry study; configured auxiliary input, not the active final reference R |

Full hashes, imported source paths and 41 matching manifest/sidecar references are retained in [12_probe_results.json](12_probe_results.json). The model summary's three `source_hashes` keys are symbolic names; they resolve to the files used by `package_bias_model.py:292`, not root-level files with those names. The transcript records the audit-harness correction of that lookup.

All 7,138 unique indexed saved image paths decoded successfully and matched their capture-time shape/dtype/BGR-array SHA-1. The indexed capture, detector output, world and profile file SHA-256 checks match. The older v1 capture's recorded capture-script hash differs from today's source; the recorded capture-helper hash still matches. That is a historical source boundary, not evidence the archived images changed. Its detector manifest lacks detector/selection source hashes. Neither model packaging nor this audit can retroactively prove the original settle condition or physical execution identity.

Audit 04 began a separately authorized scheduling repair after the recovery freeze was released. `camera_manager_node.py` changed during this audit; baseline manager line references and 04/05 findings refer to their retained source snapshots. The capture scripts, learned wrapper, calibration loader and frozen artifact bytes used by the principal probes remained unchanged. The detector parity probe records the exact later `_predict_batch` method it exercised; the node source changed again during inference, so its `unchanged_source=false` is an explicit concurrent-source boundary. This report does not certify a subsequently edited whole ROS runtime. Final hashes and retained source identities are in [12_final_provenance.json](12_final_provenance.json).

## C12-01 — Resume validates a row count instead of a capture transaction

**Location:** [writer resume, lines 667–706](../../experiments/camera_observation_characterization/capture_bbox_grid.py#L667); [separate audit, lines 35–126](../../experiments/camera_observation_characterization/audit_capture_resume.py#L35); [existing regression](../../tests/perception/test_capture_resume_audit.py#L69).

**Trigger:** an interrupted prefix has five rows but duplicate A/missing E, a gap before the largest successful pose ID, inconsistent stamps, corrupt images, mixed successful/failed camera rows, or changed helper/source bytes. **Expected:** refuse before modifying the manifest/index; require complete, coherent, verified camera×pose×repetition transactions with the frozen acquisition contract. **Observed:** the unchanged writer branch accepts every listed case except a literal four-row batch. It computes `max(successful_pose)+1`, so a mixed-success pose is skipped permanently. A complete two-repeat prefix is incorrectly refused because the count is checked per pose against five rather than per `(pose,repetition)`.

The six prior regression examples pass only for `audit_capture_resume.audit`; the CLI never calls it. `expected_manifest` is constructed and then not compared on resume beyond `plan`. Changed noise/seed, semantic mode, settling settings or geometry/configuration can therefore be used while the original manifest persists. Some acquisition settings, including settle/min-new-frame policy and robot z, are not fully frozen in that manifest at all.

**Additional reproduction:** a five-camera batch with camera E's `robot_x=10` while A–D have `robot_x=0` passes both writer and preflight. The preflight verifies camera membership/stamps, not consistency with the commanded pose plan. The complete-capture audit similarly relies on supplied spans rather than independently validating every pose/time relation ([audit_capture_integrity.py:77](../../experiments/camera_observation_characterization/audit_capture_integrity.py#L77)).

**Consequence:** resumed acquisition can skip data, combine incompatible populations, and retain false provenance. Index rewriting occurs before ROS readiness and is not atomic. The tail-retry logic also removes all-failed tail rows, so original failed attempts are not an append-only ledger.

**Evidence:** `capture_resume` in [probe results](12_probe_results.json), 13 compact variants, executing the actual AST branch before any ROS initialization. The changed-configuration variants vary the candidate `expected_manifest`; they demonstrate the missing comparison, not a changed live mount.

**Smallest repair:** make one reusable validator mandatory inside the writer before writes. Check unique expected cameras per pose/repeat; complete contiguous successful prefix; common command/heading/reference fields against the frozen plan; finite recomputed timing; decoded hashes; full acquisition/source/geometry/perturbation identity. Persist attempt IDs and explicit supersession instead of deleting failed history. Update index/manifest atomically under a writer lock. Add CLI-level regressions, not only tests of the optional preflight.

## C12-02 — Receipt after settling does not prove capture after settling

**Location:** [callbacks and candidates, lines 179–263](../../experiments/camera_observation_characterization/capture_bbox_grid.py#L179); [capture, lines 279–300](../../experiments/camera_observation_characterization/capture_bbox_grid.py#L279); [timestamp chooser](../../experiments/camera_observation_characterization/capture_bbox_grid.py#L97).

**Trigger:** transport delivers queued pre-command frames after the wall-clock settle interval; or repeats a message three times. **Expected:** one newly acquired image per camera after the declared command/settle barrier, with a new physical identity. **Observed:** `_rgb_cb` increments on every delivery; `_pair_candidates` only checks callback counts. The actual `capture` method returns timestamp 99 s images for commands declared at simulated time 100 s. A second command to a different pose returns that same timestamp/image again. Three duplicate callbacks satisfy `min_new_rgb=3`.

The timeout/settle clocks are monotonic wall time, whereas image stamps are simulated time. There is no recorded set-pose issue/ack stamp, settled simulation-time barrier, measured stationary pose confirmation, or last-consumed capture stamp. A successful `SetEntityPose` response proves service success, not the subsequent image's pose. Selecting the minimum-span combination can favor an old perfectly synchronized set over a newer slightly skewed set.

**Separate membership limit:** a camera-A image at 100.000 s from one labelled pose and B–E at 100.040 s from another fit the 0.05 s window. All 120 camera dictionary permutations preserve identity correctly, but a timestamp window alone cannot prove shared physical pose. This is constructed evidence, not a diagnosed cross-pose batch in the frozen capture. Ordinary 0.20 s adjacent-round separation is the documented chooser repair and its two existing tests pass.

**Smallest repair:** freeze and log a command/ack plus simulation-time settle barrier; require each accepted producer stamp/sequence to advance past it and the camera's last consumed identity. Preserve receipt time separately. Verify stationarity/reference pose if the protocol requires an actual settled pose, and explicitly reject or restart on epoch/clock changes. Coordinate sequence/epoch design with 04 and physical reference verification with 05. A pixel hash alone is insufficient: an actually new static image may legitimately have identical pixels.

## C12-03 — Declared image identity is not checked at the detector boundary

**Location:** [run_bbox_detector.py:62](../../experiments/camera_observation_characterization/run_bbox_detector.py#L62), especially `unique_by_hash` at 74–78 and the copied hash at 146; [derive_interpretations.py:128](../../experiments/camera_observation_characterization/derive_interpretations.py#L128).

**Trigger:** a saved image is re-encoded lossily, cropped, resized, replaced or corrupted after capture while its CSV hash remains unchanged. **Expected:** decode/shape/hash verification before inference, followed by exact upstream file-digest/key validation before interpretation. **Observed:** the detector passes the current file to YOLO and labels the result with the old hash without checking pixels. The synthetic probe replaces a declared zero image with a white image; its fake detector sees the white pixels and the output still claims the zero-image hash. Deduplication uses that unverified hash, including across cameras.

The interpretation importer verifies duplicate detector keys, key-set equality and equality of **recorded hash strings**, and checks the world-profile digest. It does not check the detector manifest's declared capture/output SHA-256 digests before using the tables. A changed box with an unchanged image hash can consequently be interpreted and certified under newly generated output hashes. Downstream `study.load` properly rejects changes to its already-frozen input files, but freezing an inconsistent upstream set is still possible.

**Consequence:** the claimed image/detector/interpretation chain need not describe the actual inputs. **Smallest repair:** decode and hash the exact array passed to inference; reject shape/hash mismatch and unknown calibration geometry, verify expected upstream file hashes and cardinality, and refuse overwrite of frozen outputs. Hash weights before loading and verify the loaded-file boundary/after-load identity as appropriate. Carry selection/version/preprocessing metadata with the output. Do not change normalization, thresholds or detector weights to resolve this defect.

**Observed active-data boundary:** all 7,138 frozen images pass decoded-hash verification; this is an unguarded mutation path, not discovered lossy corruption of the selected corpus. PNG and the verified lossless WebP path require no crop or resize before native YOLO's own preprocessing. Original-coordinate box conventions remain half-open; audit 05 owns the optional compiled-backend endpoint mismatch.

## C12-04 — A failed capture is silently outside the downstream opportunity population

**Location:** [detector input/output populations](../../experiments/camera_observation_characterization/run_bbox_detector.py#L65), [interpretation population](../../experiments/camera_observation_characterization/derive_interpretations.py#L143), [field frame assembly](../../experiments/icra_commissioning/field_study.py#L61).

**Trigger:** `complete_with_failed_batches`. **Expected:** distinguish planned exposure, acquisition failure/outage, attempted inference, detector miss, interpretation rejection and admitted observation. **Observed:** a synthetic hit + genuine detector miss + capture failure produces two detector rows and `attempt_rows=2`; the capture failure is absent. The ordinary detector miss survives with `detected=0`. Interpretation follows the same successful-capture subset. Five-camera membership checks do not detect an entirely absent failed pose.

**Consequence:** a detector-return probability conditional on successful acquisition can be mistaken for end-to-end availability. **Smallest repair:** retain failed acquisitions in an opportunity ledger with separate status/reason and no fabricated detector outcome. Either fail dataset commissioning when all-camera acquisition is required, or export both acquisition availability and conditional detector availability with explicit denominators. Keep failed/rejected opportunities when changing gates.

The selected frozen field has zero failed capture batches and retains **15,440 views, 6,412 detections and 9,028 detector misses**. Its field assembly initializes five false hit flags per frame, then sets `raw_valid` hits. This regression example is therefore prevented for ordinary detector misses in the current selected corpus, but not for future capture failures.

## C12-05 — Lossless conversion can orphan indexed images on interruption

**Location:** [reencode_capture_lossless.py:74](../../experiments/camera_observation_characterization/reencode_capture_lossless.py#L74), index commit at 89–97.

**Trigger:** failure between converting/renaming one image and the final CSV replacement. **Expected:** the old index remains readable or a durable conversion journal supports recovery. **Observed:** injected failure on the second codec operation leaves `0.webp` with verified original pixels, while the index still names missing `0.png`. Retrying the utility follows that missing path. In-place overwrite before verification also exposes the original inode to a partial-write failure.

**Consequence:** a storage operation that preserves image values on success can break capture integrity on interruption. **Smallest repair:** record a durable per-file conversion journal and recover incomplete transactions; prefer encode/verify a new file, atomically switch the indexed reference, then remove the original. If free space requires in-place recovery, explicitly implement that weaker journaled protocol instead of claiming atomic conversion. No frozen capture was converted by this audit.

## C12-06 — The sparse out-of-fold fallback reinstates the earlier procedure error

**Location:** [learn_measurement_covariance.py:242](../../experiments/camera_observation_characterization/learn_measurement_covariance.py#L242), particularly 270–272; manifest claim at 777.

**Trigger:** an NN fold leaves fewer than 50 fitting rows. **Expected:** refuse unsupported covariance fitting, reduce the fold design under a newly declared protocol, or explicitly mark the residual unavailable. **Observed:** four synthetic training rows return their stored in-sample NN residuals unchanged, without any held-out model fit, while the enclosing output claims variance was fitted on out-of-fold residuals. No NN fit was called in the reproduction.

**Consequence:** a smaller commissioning/repeat subset can silently reproduce the old optimistic covariance-target error. **Smallest repair:** fail closed for fitted mean models; record each residual's mean-training exclusion and model identity, with a postcondition that no covariance target came from a model trained on it. Grouping only by position also does not create a spatial buffer between neighboring positions.

**Current repair boundary:** the selected ICRA R uses a different procedure: [study.py:109](../../experiments/icra_commissioning/study.py#L109) fits on `covariance_fit` rows unseen by the frozen NN. The outer roles contain 3,249 NN-training hits, 1,172 covariance-fit hits, 819 selection hits, and 1,172 evaluation hits. The covariance-ladder fallback is not on that active artifact's fitting path. These are configuration samples, not independent repeat-noise observations.

## C12-07 — Version/units checks are stronger than lineage and camera-support checks

**Locations:** [learned wrapper:57](../../src/reliability/reliability/learned_box_correction.py#L57), [reference calibration:12](../../src/reliability/reliability/reference_calibration.py#L12), [network loader:47](../../src/planning/planning/core/camera_network.py#L47), [reference export:17](../../experiments/icra_commissioning/export_reference_calibration.py#L17).

**Trigger / expectation / observation:** temporary artifact mutations establish the following boundaries:

| Mutation or query | Observed behavior |
|---|---|
| NN unknown version or missing feature order | Refused |
| Ordinary unknown camera, nonfinite raw XY/score, absent bbox | NN returns `None`, as intended |
| NN missing `target` | Loads and predicts; physical target semantics are not enforced |
| NN missing camera-B geometry | Loads; every B correction subsequently returns `None` |
| Extra camera-Z geometry, absent from trained `camera_ids` | Loads and predicts using all-zero camera one-hot |
| Reference schema/units/NN hash/missing camera/nonfinite R | Refused |
| Reference nonfinite query or unknown requested camera | Refused |
| Reference missing or forged `source_hashes` | Loads and applies |
| Network schema/units/unknown camera/nonfinite query | Refused |
| Network missing or forged `source_hashes` | Loads and queries |
| Consistent NPZ camera-axis permutation | Exact same ID-keyed answers |
| Degenerate constant GP queried at NaN XY | Returns 0.9999; outer commissioned-field/network query validation correctly rejects nonfinite poses |

**Consequence:** “artifact loads” proves neither a compatible complete camera registry nor detector/projection/data provenance. The active reference calibration binds the exact NN bytes, so merely replacing the NN while retaining that calibration is rejected. The standalone NN mutation tests must not be misreported as bypassing that combined guard. The NN geometry contains XY/yaw/image dimensions, omitting full K/extrinsics/robot-reference/detector identity. Audit 05 G02 reproduces the missing projection binding; 13 owns expected campaign artifact identities and hash/load races.

**Smallest repair:** validate exact supported camera sets and finite positive geometry, require target/transform semantics, bind artifacts to a canonical detector/calibration/preprocessing/reference contract and expected model-data identities, and read/hash/parse the same bytes. Keep archived source hashes as lineage, not as a demand that current source always equal historical source. Any compatible software migration needs frozen-example equivalence and a new provenance record. No refit is needed to add trustworthy metadata.

The current exported R also needs an accurate fitting description: `study.py:125–138` applies selection-fitted global scale/isotropic shrinkage after covariance-fit centering. `export_reference_calibration.py:46` names only the covariance-fit population. The numeric round trip is correct; the minimal metadata repair is to record both covariance-fit and selection roles and the transformation, not to call R an untouched empirical covariance.

## C12-08 — Legacy operational exporter has no causal physical-event join

**Location:** [observation_exporter.py:122](../../src/reliability/reliability/observation_exporter.py#L122), particularly 139–180 and 188–210. The requested `tests/reliability/test_observation_exporter.py` is absent; its actual tests are [tests/observability/test_observation_exporter.py](../../tests/observability/test_observation_exporter.py).

**Trigger:** duplicate perception rows, or a nearer belief sample after the perception event. **Expected:** once-per-camera physical acquisition identity; capture-aligned operational features that could exist at the stated decision time; explicit missing reception/association states. **Observed:** two identical rows both survive; the 1.000 s receipt row chooses belief at 1.050 s instead of 0.900 s. The output stores receipt `log_stamp` as timestamp, has no image/capture/batch identity, sets `frame_expected=frame_received=True`, and substitutes yaw zero when unavailable. The joined belief may already include the same observation.

No truth field enters the raw output schema. However, `_load_belief` calls a GT-checking diagnostic loader, and the field-name whitelist test alone does not prove causality. Input run files are selected by glob, not a frozen selection; exporter metadata contains configuration hashes/HEAD but no complete input-file/model identity. Missing belief/drop counts are partly recorded, while rows without valid detected/stamp fields disappear before `rows_seen`.

**Consequence:** the resulting spatial model may learn post-event belief or receive pseudoreplicated samples; it cannot establish acquisition-outage probability or true association quality. **Smallest repair:** implement a current manifest-bound event adapter using capture/producer IDs and a declared pre-event belief snapshot or checked causal replay; retain all scheduled opportunities and stage statuses. Keep unknown heading unknown. Separate an optional offline truth validation from the operational export input contract. The existing tests check the whitelist and gate on constructed records, not this iterator.

## C12-09 — Operational residuals require stronger identity and observation semantics

**Location:** [build_operational_residuals:113](../../src/reliability/reliability/operational_residual.py#L113), [summarize_residuals:183](../../src/reliability/reliability/operational_residual.py#L183), [shrink_summary:237](../../src/reliability/reliability/operational_residual.py#L237).

**Trigger 1:** duplicate the same measurement objects. **Expected:** count independent physical evidence once, or explicitly model repeated/correlated observations. **Observed:** duplicating two records ten times changes `sample_count` from 2 to 20 and shrinkage weight from 0.8333 to 0.3333. There is no physical-event key beyond a trajectory index, no duplicate guard, and `held_out=True` is inferred from the caller-supplied anchor list, whose default is empty. That is a provenance assertion, not a verification of the reference trajectory's construction.

**Trigger 2:** use the advertised pixel path with a nonlinear/affine projection. **Expected:** residual `z-h(mu)` and state projection `H P Hᵀ`. **Observed:** code uses `predicted=H@mu`. Even the simple affine camera `h(x)=diag(2,3)x+[640,360]` with an exact observation returns residual `[640,360]` instead of zero. A Jacobian cannot replace the function value. This pixel helper is not used by the current metric reference observations.

**Smallest repair:** carry acquisition and reference-trajectory identities; reject duplicate physical evidence or use a declared effective-sample model; require verified smoother provenance. Accept an observation-function value/callable and a per-state Jacobian separately, or explicitly restrict this API to linear zero-offset XY. Retain mean residual, state-covariance subtraction and PSD-projection flags. Existing tests verify XY algebra, bias separation, covariance subtraction and held-out smoother examples; the pixel test checks only covariance propagation.

A smoothed belief can be built offline from operational inputs without GT, but future samples used by smoothing are not available to an instantaneous online estimator. Held-out-camera status alone also does not establish independence from common odometry/calibration/camera errors. These are identification assumptions to validate, not reasons to absorb bias into R.

## C12-10 — Alternative residual dataset completion does not guarantee row completeness

**Location:** [build_residual_bias_dataset.py:83](../../scripts/perception/build_residual_bias_dataset.py#L83), `zip(chunk,predictions)` at 98, completion at 165–188; [train_residual_bias.py:46](../../scripts/perception/train_residual_bias.py#L46); [residual_bias_model.py:24](../../scripts/perception/residual_bias_model.py#L24).

**Trigger:** backend returns one result for two source positives. **Expected:** refuse wrong cardinality before assigning results or sealing a dataset. **Observed:** real `build` with a fake short backend writes one record and `.complete`, status `complete_provisional_residual_dataset`. Extra results would also be silently ignored by `zip`. **Smallest repair:** exact per-call length/interface validation and strict zip; verify source/record/manifest digests before sealing and before training. The trainer checks `.complete` existence but does not verify its manifest hash or `records_sha256`.

This is a different model family from the active NN. It excludes background/clipped-centre records for a geometry-correction task, labels the target using commanded ground pose minus semantic-mask bottom centre, and marks itself provisional. It cannot supply unconditional availability or false-positive rates. Its `detected=0` can mean either no box or invalid baseline/target projection, losing stage-specific reasons.

Its heading feature defaults to `row['robot_yaw']`, which is commanded truth in the provisional source; an explicit online heading argument works without that field. Even `disable_heading=True` still reads the missing truth-yaw key first. Nonfinite confidence produces nonfinite features instead of explicit refusal. **Repair before any operational use:** require an explicit timestamped operational heading when enabled, avoid reading it when disabled, validate features and artifact normalization/version, and name the semantic-mask labels evaluation/commissioning-only. There is no current runtime import of this pixel model, so this is not a GT leak in the active box-feature correction.

## Frozen sample through the full chain

This is the first camera-B sample in `covariance_fit`, chosen by role/identity, not error. Complete values are in `sample_trace` in [12_probe_results.json](12_probe_results.json). It is deliberately **held out of NN training** and enters the offset/R fit and future-field fit. A single row cannot belong to both the NN-training and honest NN-held-out covariance roles.

| Stage | Identity, frame/reference, time and transform | Label source / online availability |
|---|---|---|
| Commanded pose | Capture root fixed above; pose 0, position 0, heading 0, repetition 0; commanded XY=(-8.025,-8.700) m, yaw=0 rad; model/base-footprint ground reference | Commanded simulator reference, offline. Command issue/ack/settle stamps and actual settled pose are absent |
| Fresh RGB claim | `camera_B/images/pose_000000_r00.png`; decoded SHA-1 `c4b089a9f19824814360785a9b2fe76e9ca3d5b8`; 1280×720 BGR decoded from saved PNG; image stamp 40.400 s; batch span 0.200 s | Image/camera/stamp can exist online. File/hash is archived evidence. v1 cannot prove freshness from its logs |
| Detector | Same detector hash above; selected xyxy=[0.6729227901,366.1938476563,120.6364593506,454.7211914063] px; score=0.9600066543; bottom centre=(60.6546910703,454.7211914063) px; image identity retained | Detector output, online. No mask association label; `detected=1` is a returned robot-class box, not independently established correctness |
| Manifest/index join | Key `(pose_id=0,repetition_id=0,camera_B)` and decoded hash; original capture has no `source_batch_id`; tables lose the timestamp and commissioning joins it back from capture | Numeric command fields remain offline references. No epoch/capture sequence or detector invocation ID can be recovered from v1 |
| Interpretation | B mount XY=(-1.5,-9.72) m, height 5 m, pitch .8378, yaw 2.0071 rad; frozen profile/world; floor z=0; raw ray landing=(-7.7734594602,-8.9827215942) m | Calibration plus observed bottom pixel, online. Raw landing is not robot centre. The separate hull column uses commanded truth/yaw and is explicitly an offline oracle |
| Mean feature vector | Raw-derived range=6.3166346298 m, inverse range, box fractions/aspect, normalized bottom pixel, relative bearing cos/sin, score, camera one-hot; scaler then MLP; camera-ray along/left correction added to raw XY | All active NN features exist online without robot truth/heading. True range 6.6042429543 m is diagnostic and **not** the NN feature |
| Split and fit | Outer checkerboard=`test`; tile=(-5,-5), role=`covariance_fit`. Frozen NN predicts (-7.9915362765,-8.6371366884) m. This sample contributes to post-NN residual fit, not NN fitting | Commanded XY supplies the offline residual label. Offset and R are estimated across a population; no local covariance can be identified from this row |
| Packaged reference | b_B=(-.0206736841,-.0399250047) m; active z=NN−b=(-7.9708625924,-8.5972116837) m. R_B=[[.0333131036,.0126174749],[.0126174749,.2790908185]] m², map_bev | Calibration hash/NN binding; full metric covariance, separate from confidence and q. Selection-role scale/shrinkage is part of R's fitting history |
| Future field/export | This opportunity contributes to its position's score mean (miss=0) and raw-valid hit labels. Pose/heading is the surveyed/commanded commissioning coordinate; NPZ order `[camera,y,x]`, covariance `[camera,2,2]` | Future query uses predicted XY/heading, fixed calibration and stored field. It never reads this sample's future score, image or truth at runtime |
| Runtime import | `LearnedBoxCorrection` followed by `ReferenceCalibration.apply`, then manager map observation with capture stamp; future `CameraNetworkModel` uses the separate NPZ | Active reference computation verified numerically. Manager admission/fusion and robot belief are separate downstream quantities owned by 06/07/08 |

## Dataset/model limitations and documented repairs

1. **The old static capture is not a validated new capture protocol.** All 3,088 pose groups contain exactly five camera IDs. Recomputed spans match the CSV, but **978 groups exceed 0.05 s**, maximum 0.20 s. At a truly stationary pose, asynchronous views can still describe that same pose; those spans alone do not prove cross-pose contamination. Missing settle/pose verification prevents ruling it out. Keep the frozen study diagnostic; do not relabel it v2 after repairing capture.

2. **One image per camera-position-heading does not estimate both conditional mean and repeat covariance.** There are 386 positions ×8 headings ×5 cameras, one repetition. Three successful-image hash groups each contain two views from the same position/camera at opposite headings. They stay within their outer split/commissioning role. Reusing deterministic image bytes is legitimate for counting declared physical opportunities or a heading distribution, but is not independent detector-noise evidence. Added post-transport read noise is only the declared read-noise experiment; without a separate raw hash, its per-repeat noise can conceal duplicate underlying captures.

3. **Outer successful-image leakage was checked and not found.** Five image-hash groups cross train/test, all misses; none of the 6,412 successful rows crosses the outer split or commissioning roles by hash. `study.py:61–70` excludes cross-role hit hashes, `field_study.py:82` refuses them. These are useful repairs. The original NN fitting/checkerboard helper itself does not enforce image grouping. A synthetic identical image on positions 1.999 and 2.001 m crosses its split only 0.002 m apart. For the actual field, nearest train distance from a held-out position is minimum/median **64.4 cm**, maximum **133.8 cm**: same-installation interpolation, not separated-region or new-installation generalization. All headings/cameras per actual position remain together. Future confirmatory splits need a declared spatial buffer or route/installation holdout and image-identity groups, not a post-hoc removal selected by errors.

4. **Inner validation and endpoint behavior are weaker than the outer claim.** The packaged MLP uses row-random early stopping within TRAIN; its scaler is fitted before that internal split. This is not independent position-level internal validation, although the external TEST tiles remain outside both. `dataset_split_utils.assign_splits:77–105` also forces at least one group on each side for fractions 0 and 1; the synthetic endpoint probes reproduce this P3 helper bug. `spatial_yaw_bucket`/`cyclic` modes can split one position's headings by design; do not use them to claim position-held-out evidence. The active metric NN uses `tile_split`, not these optional modes.

5. **Covariance semantics remain conditional on a declared population.** `CameraModel.covariance` centers residuals; `fit` stores mean/second moment separately; geometry/spatial/confidence cells save remaining means rather than silently correcting them. Current R fitting excludes mean-training rows, repairing the in-sample mean-target procedure. It nevertheless describes scatter across static configurations and includes unmodeled mean structure; it is not established repeated-sampling covariance at a fixed pose, temporal independence, Gaussian-tail calibration or cross-camera independence. The selection-fitted scale is a working-Gaussian calibration operation, not proof all residual bias disappeared. `CommissionedField` stores predictions of that fitted R at training covariates; it does not re-estimate R from mean-training residuals.

6. **Availability, confidence and conditional covariance remain different outputs.** Current `hits` means returned detection with finite ground projection **before manager admission**. False positives also satisfy that definition. The synthetic empty-image case injects a confident robot-class box and the real detector exporter retains `detected=1`; this tests label semantics, not the frozen detector's actual false-positive rate. The field capture has no semantic masks or independent association labels, so it cannot identify true-positive availability. The older measurement commissioning study explicitly reports false-positive rate unmeasured; `capture.write_availability` retains unsaved/no-image diagnostic rows as unusable, which must not be interpreted as observed detector misses. Its admission labels use the commanded pose, and its offset-position selection consults semantic visibility ([capture.py:97](../../experiments/measurement_commissioning/capture.py#L97)); that is a simulator/survey oracle design input, not operational detector evidence.

7. **Static q is not belief-conditioned operational usability.** Active runtime silhouette gates use a nearest belief pose and discard its covariance; 06 documents that changing belief changes the usable set. Static field q does not model uncertainty, freshness, missing cameras, quorum, association, common-time support or NIS recovery. The current NPZ correctly labels its narrower pre-gate target. `ObservabilityGP` can be fitted on belief coordinates by the legacy adapter or truth/survey coordinates by commissioning; the class name/docstring is not provenance. The field's future API uses only predicted poses and frozen geometry/quality distributions, so no future image/score/truth leak was found in that API. This establishes input availability, not calibration under uncertain operational belief.

8. **Previously inspected holdouts are development evidence.** NN summary and `study.py`/`field_study.py` explicitly acknowledge earlier inspection. New roles prevent mechanical reuse in the current fit/selection code but cannot erase earlier model/threshold choices. Older covariance-ladder `best_by_heldout_nll` is model selection on that set, not an independent final test. Current field hyperparameters use selection tiles and evaluation predictions follow model freezing; those evaluations remain development diagnostics. Freeze future validation routes/thresholds before inspecting their outcomes. No detector/NN metric-driven retraining was performed here.

9. **Geometry/reference defects must be repaired before learning explanations.** The active correction target is commanded ground-reference XY minus raw floor landing, in the raw camera-ray basis. Half-open box interpretation and correction order agree on the frozen inputs. Audit 05 verifies actual projection/NN/manager equality and documents missing geometric semantic validation. The offline truth-centered hull column is not runtime-capable as written; do not compare it to a belief-centered online implementation as if inputs matched. Fix a reference, projection or convention mismatch explicitly instead of training the NN to compensate for it.

## Numerical verification, evidence and repair handoffs

The main [probe](12_probe.py) verifies 6,412 saved hit rows with the frozen MLP/scaler, actual offline feature/application helpers, deployed NN wrapper and frozen `CameraModel` versus `ReferenceCalibration`. Maximum feature/NN/reference-array discrepancy is **3.56×10⁻¹⁵** in the relevant numeric units; no model is refitted. It also verifies all saved pixels, counts/roles/hash groups and the listed synthetic defects. Audit 05 independently extends the same comparison through actual floor projection and manager construction; this report does not present its live trace as a new capture.

The [detector parity probe](12_detector_parity.py) runs the frozen native checkpoint on one preselected complete pose's five camera views. Offline saved-file inference (batch five, confidence floor .001) and the captured runtime method on identical decoded BGR arrays (chunks of two, confidence floor .05) give identical selected boxes/scores and hit/miss outcomes at the common .25 selection threshold. Candidate counts below that threshold differ, as expected. Maximum difference from historical saved boxes is **1.02×10⁻⁵ px**, and from historical scores **1.20×10⁻⁷**. This is a CPU preprocessing/chunking check on five examples, not a detector accuracy, GPU equivalence or live scheduling test. The [results](12_detector_parity_results.json) retain the exact executed method and library versions. An initial harness attempt omitted the method's newly added `time` dependency; its failure transcript is retained separately, and the corrected run completed.

[Boundary probes](12_boundary_probe.py) run the actual reference exporter to a temporary output: JSON **and bytes** equal the active reference calibration. Frozen field joblib save/load gives zero q difference. Frozen field q agrees exactly with all three NPZ availability grids at supported grid nodes. All 120 reference-camera request permutations agree; a consistent NPZ camera-axis permutation gives zero score/q difference. These checks do not establish between-grid approximation accuracy, support-boundary calibration or planner/fusion forecast equivalence; 08 owns those semantics.

The requested and adjacent existing tests initially report **73 passed, 1 failed**. The failure is `tests/test_commissioned_field.py::test_forecast_uses_all_odometry_changes_including_a_turn`: a generic `replay` import resolves to `fusion_on_fixed_routes/replay.py` after the other suite imports it. Running `tests/test_commissioned_field.py` alone gives **5 passed**. This is retained test/import-order fragility, not a failed covariance numerical check and not a claimed clean combined-suite pass. Transcripts: [combined](12_existing_tests.txt), [isolated](12_isolated_tests.txt). No production import was renamed in this audit.

Reproduce from repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/12_probe.py --all-pixels
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/12_boundary_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/12_detector_parity.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/perception/test_capture_resume_audit.py tests/perception/test_bbox_characterization_capture.py tests/perception/test_dataset_split_utils.py tests/observability/test_observation_exporter.py tests/reliability/test_operational_residual.py tests/test_learned_box_correction.py tests/reliability/test_reference_calibration.py tests/test_icra_commissioning.py tests/test_commissioned_field.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_commissioned_field.py
```

The reports/results are written only under `docs/module_audits/12_*`; synthetic data and re-exported models are temporary. Pixel verification is read-only and can be slow. The principal transcript retains an explained alias-resolution/label correction in the audit harness; it did not alter images, models or inference results.

**Coordination:** 04 received the receipt-freshness and capture-membership evidence and owns scheduling epochs/invocation outcomes. 05 supplied geometry/reference binding constraints and its independent projection equality. 06 received the denominator/causal-feature findings and owns operational gate semantics. 08 confirmed active score versus q/R meanings and owns interpolation/forecast/fusion consequences. 13 confirmed the registered campaign's detector/field/reference hashes and NN transitive hash binding; it owns campaign expected-identity enforcement and the separately reproduced hash-then-parse race. Their repairs must retain distinct source/protocol boundaries.

**Small repair order:** enforce the capture transaction validator and capture-time barrier; make detector/interpretation imports verify their exact inputs; preserve failed opportunities and make storage conversion recoverable; refuse unsupported covariance/sparse/residual paths; enforce complete artifact semantics and metadata. Then collect a separately frozen dataset with verified settled reference, explicit negative/association trials, declared repeat nuisances and independent validation. Decisions about richer mean, covariance, availability, fusion or planner models follow that evidence; the current numerical equality checks cannot decide them.
