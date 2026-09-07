# 11 — Offline alignment, replay, statistical aggregation and reporting

2026-09-06. **The evidence chain does not enforce its contract end to end.** The most serious confirmed defects are an event scorer that labels a pre-application prediction as a post-correction belief, inconsistent event-ledger validation, identity selection that depends on timestamps or row order, and reporting paths that accept mismatched or stale evidence. Ordinary time alignment and several recent repairs work in their tested scope. Passing those tests does not establish complete event accounting or valid replication.

This report delivers investigation and proposed repairs only. No runtime, production analysis, configuration, model, frozen selection, recorded drive or existing figure was changed. Synthetic logs and audit outputs are confined to `experiments/runtime_integrity/alignment_reporting_20260906/`; the two audit scripts and this report are under `docs/module_audits/`. No simulator, campaign, process cleanup or live failure injection was run.

Read before tracing code: repository/root AGENTS, PLAN, the localization contract and registry, open questions, runtime integrity audit, and completed audits 02–05. The investigation map and current ICRA status were also consulted. Audits 01, 06 and 07 became available during this investigation; their related findings were checked before attribution. The post-correction association problem was already identified by the estimation investigation and is independently reproduced here, not claimed as a new discovery.

## Ranked findings

Priority reflects the consequence of taking the affected output literally and its reachability, not observed frequency in drives. “Confirmed” means a deterministic reproduction or direct source proof. Current diagnostic consumers are distinguished from fixed-route paths blocked by the absent paper selection.

| Rank | ID | Classification | Finding and reachability |
|---|---|---|---|
| 1 | A11-01 | P1 confirmed; prior finding independently reproduced | “Immediately after correction” can select a prediction before application, or reuse one belief for several events. Shared helper; fixed-route comparison panels call it. |
| 2 | A11-02 | P1 confirmed | Ledger checks disagree; unknown status can pass fixed-route selection. Truth availability can turn a correctly accounted shutdown event into an “extra assimilation.” |
| 3 | A11-03 | P1 confirmed | Camera/held-belief deduplication is not based on a validated identity ledger; conflicting/reordered events change selected samples. Shared active loaders. |
| 4 | A11-04 | P1 conditional integrity defects; interpolation policy limit | Truth sorting merges clock epochs, accepts conflicting ties and interpolates arbitrary gaps. Advertised buffered-reference preference is unimplemented. All aligned consumers. |
| 5 | A11-05 | P1 confirmed | Frozen selection does not consistently validate exact keys, required hashes and controlled provenance across arms. Current navigation/network consumers and fixed-route selector. |
| 6 | A11-06 | P1 confirmed | An existing per-run result bypasses all new input/model checks. Current thesis/field analysis cache. |
| 7 | A11-07 | P2 confirmed | Held beliefs still alter fixed-route scores; corrections are sampled through logger ticks; initial/final blind periods and acceptance populations differ between consumers. |
| 8 | A11-08 | P2 confirmed | Aggregate output retains first-run duration/outcome/gaps; captions invent fixed seed counts and omit five-camera bins. Fixed-route reporting, presently without a paper selection. |
| 9 | A11-09 | P2 confirmed legacy defects and replay-policy limits | Legacy replay bypasses aligned selection/time rules; some commissioning comparisons omit the common prediction grid. Registered exhaustive network replay has the grid repair. |
| 10 | A11-10 | P2 confirmed conditional figure defect | A storyline scores/draws fused output against camera-capture truth, bypassing the repaired fused-answer loader. Fixed-route figure dependency. |
| 11 | A11-11 | P2 confirmed | A misspelled score option silently selects all routes and exits successfully on selection failures; custom navigation config produces hard-coded protocol task/seed metadata. |
| 12 | A11-12 | P2 confirmed input-validation defect | Negative-definite covariance produces negative NEES and apparent perfect coverage. Shared planar and commissioning scores; current navigation full-pose guard is stronger when fields exist. |
| 13 | A11-13 | P2 confirmed | Monitor merges task identities, counts retries as seeds, omits summary-less attempts, and calls dropped fraction “refused.” Ground-truth field substitution itself is repaired. |
| 14 | A11-14 | P3 confirmed conditional utility defects | AP can fail to terminate on NaN scores; ECE silently returns zero for entirely invalid predictions. No current AP caller was found. |

## Reproduction, source identity and evidence boundary

[Synthetic probe](11_alignment_probe.py) executes the actual loader/scorer/replay/CLI functions. [Results](../../experiments/runtime_integrity/alignment_reporting_20260906/results.json) contain **59 cases**, independent expected values, observed selections/outputs, resolved import paths and full source hashes. [Transcript](../../experiments/runtime_integrity/alignment_reporting_20260906/probe_output.txt) retains execution output. Assertions establish both correct controls and reproduced defects; successful completion is not a passing production acceptance test.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/audit11_mpl python3 docs/module_audits/11_alignment_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/11_registered_evidence_inventory.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/audit11_mpl python3 -m pytest -q tests/experiments/test_fusion_study_alignment.py tests/test_network_replay.py
```

The two requested existing test files returned **17 passed**, with three optional installed-dependency warnings, no failures/skips. The synthetic probe completed with **zero changes to the 21 watched source/contract files during execution**. Selected hash prefixes identify the line references below; full digests are in results.json:

| Source | SHA-256 prefix |
|---|---|
| `fusion_on_fixed_routes/aligned.py` | `dabd8b9167af3f11` |
| `fusion_on_fixed_routes/score.py` | `3cf40042216d0d58` |
| `fusion_on_fixed_routes/compare.py` | `4113e47c611b0dd8` |
| `fusion_on_fixed_routes/replay.py` | `94df68d35368a454` |
| `icra_commissioning/replay.py` | `2910ab2e45b3ecbe` |
| `icra_commissioning/field_driving.py` | `cbb3ff81030bfcc4` |
| `icra_commissioning/network_replay.py` | `cc9624da33838064` |
| `icra_commissioning/network_navigation_analysis.py` | `504c08f1ca4d6e3f` |
| `visibility_comparison/monitor_campaign.py` | `efc02d43031511ec` |

At final review, concurrent work changed navigation figure titles (`cbbe86d6b80312e1` → `504c08f1ca4d6e3f`) and the separate recovery-pilot registry status (`8259f4138e5998f4` → `1b622751291cfb12`). The affected numeric/selection functions and both inspected selection boundaries remained unchanged. The 59-case probe and exact inventory were rerun against that checkout; earlier outputs are preserved as `*_before_presentation_followup.*`. The final probe retains content-addressed copies of all 21 watched files under its `source_snapshot/`, with paths in results.json. These presentation/status edits are not repairs to the findings below. Navigation protocol references after the title edit are four lines later than the initial inspection.

**Existing-drive boundary, before any counts.** No existing accuracy, NEES, coverage, bias, collision rate or improvement estimate is reissued here. The contract has no paper-facing fixed-route fusion selection; schema-3 evidence and the invalidated campaign remain diagnostic. Parsing an older file does not restore its missing assimilation evidence. Because the loader/prose disagreements below are real, affected outputs must be repaired or explicitly narrowed before new numerical claims are made.

The read-only [inventory script](11_registered_evidence_inventory.py) follows exactly two registry selections, checks every selected file hash and opens rows/observations/assimilations through aligned.py. [Inventory](../../experiments/runtime_integrity/alignment_reporting_20260906/registered_inventory.json) records identities and sample counts, not independent replicates or performance estimates:

| Frozen set | Schema, design and permitted use | Timing/source boundary |
|---|---|---|
| [Six-run thesis selection](../../logs/studies/icra_commissioning_20260905/thesis_evidence/selection.json), SHA prefix `d4ae8716e6b446dc` | Schema 7; two routes, seeds 110/111/112 per route. Paired **within-run** replay and individual pilot diagnostics only; no significance, physical validation, complete-route ranking or commissioned navigation gain. | Capture-time idealization with sampled noisy odometry; processing latency/live refusals are not replayed. The recordings came from two config digests and multiple dirty-tree digests. This is not a homogeneous cross-runtime navigation comparison. |
| [Corrected-runtime selection](../../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json), SHA prefix `66b928fbaf8f4703` | Schema 7; P0/P1/P2, seed 210 once per arm. Separate correctness-repair pilot; per-run diagnostics only. | Logged live beliefs, not a counterfactual replay. Config `72d7318718c9510`; [protocol](../../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/protocol.json) `5c115b021a563614`. All 13 snapshot files match the protocol ([check](../../experiments/runtime_integrity/alignment_reporting_20260906/snapshot_verification.json)). This verifies stored bytes, not which import every process actually executed. |

Both use the named YOLO checkpoint (`efff1949c1b8cdee`) and existing bbox NN (`5da33c63a2bb46b3`). Named commissioning covariance models hash to `84454e296a9a5c83`, the fitted field to `00cd08656e2e8afc`, and runtime residual calibration to `0a462258197b4253`. Full hashes and per-run manifest provenance are retained. Within a replay, constant/confidence alternatives use the same loaded NN/offset and recorded path; the `recorded` alternative explicitly uses a different original mean/R and is not a covariance-only control. A common repository commit alone is insufficient source identity in this dirty checkout. Differing manifest dirty hashes do not themselves prove different executed filter code.

All nine summaries declare **`gt_stamp_source=receipt_sim_clock`**. aligned.py calls this series `gt_stamp`, which names a column rather than establishing a source capture clock. Its reference is simulator pose timed at logger receipt. Inter-arrival spacing does not bound unknown source/transport lag. Audit 14 independently confirmed this interpretation. Contact silence means no recorded contact; it does not establish healthy collision telemetry or verified collision-free physical motion.

The following are **accounting counts only**. “Beliefs” means unique finite state stamps between first command and stop, before an accuracy/covariance support filter. “Candidates” means unique `(source_batch_id,camera)` manager rows across the recorded file, after manager admission and before fusion selection; it is not all detector opportunities. Assimilations are terminal rows over the whole file, not just accepted updates.

| Exact selected key | Belief stamps | Manager candidate identities | Assimilation rows |
|---|---:|---:|---:|
| `fusion_overlap_rich__N1__seed110` | 1500 | 1387 | 648 |
| `fusion_overlap_rich__N1__seed111` | 1511 | 1470 | 678 |
| `fusion_overlap_rich__N1__seed112` | 1489 | 1379 | 642 |
| `fusion_network_traverse__N1__seed110` | 787 | 796 | 290 |
| `fusion_network_traverse__N1__seed111` | 792 | 704 | 272 |
| `fusion_network_traverse__N1__seed112` | 782 | 759 | 283 |
| `fusion_network_traverse__P0__seed210` | 1891 | 1112 | 813 |
| `fusion_network_traverse__P1__seed210` | 1818 | 1065 | 779 |
| `fusion_network_traverse__P2__seed210` | 1895 | 862 | 598 |

All stored hashes matched; raw fused/assimilation ID sets matched; accepted flags agreed with statuses; no decreasing logger/belief timestamps were found in these nine inventories. These checks do not prove complete acquisition, truth timing, exact post-update reconstruction, installed-source equivalence or independent noise replicates. The synthetic clock/identity failures below are not asserted to have occurred in those drives.

## Chain and quantity ownership

| Stage | Identity, time and population | Consumer and boundary |
|---|---|---|
| Raw detector delivery | Batch + camera, capture stamp, callback receipt, hit/miss/invalid/duplicate metadata | `camera_opportunities.jsonl`; audits 04/10 own losses before logging and terminal flush. It is received detector output, not an exhaustive physical-capture ledger. |
| Manager candidate | Original observation at `obs_stamp`, XY in map/world metres and full R in m²; separate common-time aligned fields | `aligned.observations/readings`; `admitted_only=False` means all logged manager candidates, not all detector misses or admission refusals. |
| Fused correction | Batch, `fused_stamp`, fused XY/full R; repeated per-camera representation in CSV | `fused_answers`; each field must agree across rows of a batch before selection. Fused measurement error is separate from camera-reading error. |
| Terminal filter outcome | Same batch, status/reason, correction/apply/belief-after stamps and accepted flag | `assimilations`; only accepted/accepted_bootstrap/reanchored are measurement-update events. A reasoned rejection/drop remains valid evidence. |
| Periodic belief | Published mean and P at `planner_belief_stamp`; repeated by logger | Own-time error from `aligned_error_cm`; this is a public prediction, not the unlogged immediate event posterior. State time is not a revision identifier. |
| Reference | `gt_*` at declared reference time; wheel `truth_*`/`odom_*` separately diagnostic | `TruthSeries`; receipt-clock provenance and interpolation support must accompany the pairing. |
| Replay | Prescribed measured noisy controls, fixed initialization/Q and selected camera events | Commissioning `run_filter`; capture-time updates, no live NIS/delay/refusal reproduction. Fixed path is reused experimental input, not independent navigation evidence. |
| Statistics | Within-run distribution first, then unique declared run/seed units | Median, mean, RMSE, p95, signed bias, planar/full-pose NEES and coverage require distinct names and denominators. |
| Display | Selected source set + metric definition + units + valid/invalid/missing counts | Plots, tables and monitor must inherit these definitions, not infer them from a filename or caption template. |

The synthetic control uses `x_GT(t)=t` m: six camera readings have 2 cm error at their own captures, three fused corrections have 1 cm at their own fused stamps, and 21 beliefs have zero own-time error. An unrelated wheel-odometry reference shifted by 10 m changes none of those scores. The heading control interpolates 179° to −179° through 180°. For two simultaneous camera updates with `P=.0025 I`, `R_A=R_B=.01 I`, `z_A=(.2,0)`, `z_B=(.4,0)`, independent information arithmetic gives posterior `x=.1 m`, `P_xy=I/600 m²`; actual commissioning replay agrees in either camera order and refuses a duplicate camera/time update.

## A11-01 — An accepted batch does not identify its posterior

**References/reachability:** [aligned.py:406](../../experiments/fusion_on_fixed_routes/aligned.py:406), especially selection at 434–449; [compare.py:83](../../experiments/fusion_on_fixed_routes/compare.py:83). This is the fixed-route event-belief panel, not the separate periodic-belief evaluation used by current navigation analysis. [Estimation audit R04](01_state_estimation.md) already identified the causal defect.

**Trigger and expected behavior:** a capture is processed after later public predictions, or two updates occur between logger ticks. “Immediately after” requires the actual committed posterior, its state stamp, covariance and revision. A later public sample cannot be assigned by proximity alone.

**Observed:** `before_apply_is_post_correction` has capture 1.0 s, apply 1.3 s, a public belief at 1.1 s with 10 cm synthetic error, and the first public sample after apply at 1.4 s with 40 cm. The helper returns the **1.1 s / 10 cm** prediction as the accepted event. It ignores `apply_stamp` and `belief_stamp_after`; the latter is parsed but never used in this join. `multiple_updates_between_ticks` maps captures 1.0/1.05 to the same 1.1 belief, yielding two copies of its 30 cm error. The 0.30 s cutoff silently discards longer-lag/unflushed events. It also rejects an equal-time exact public sample through strict `>`.

**Consequence:** both error and covariance are attributed to the wrong population, despite nominal own-time truth pairing. Camera-count bins can inherit another correction's belief. Filtering by apply time would remove the first example but still would not recover two posteriors from one sample.

**Smallest sound repair:** with investigation 10/01, log terminal posterior mean, full covariance, authoritative frame, state time and monotonically identified revision; score that record directly. Until then, refuse the “immediate posterior” quantity and expose periodic-belief diagnostics separately, with event association marked unavailable. Do not reconstruct missing historical posteriors from timestamps.

## A11-02 — Validity checks are distributed and can depend on truth coverage

**References/reachability:** [aligned.py:270](../../experiments/fusion_on_fixed_routes/aligned.py:270), [score.py:162](../../experiments/fusion_on_fixed_routes/score.py:162), [field_driving.py:68](../../experiments/icra_commissioning/field_driving.py:68), [navigation analysis:267](../../experiments/icra_commissioning/network_navigation_analysis.py:267), [runtime report:31](../../experiments/icra_commissioning/network_runtime_report.py:31).

`assimilations` rejects blank/duplicate IDs but does not validate status, reason, accepted/status agreement, finite/consistent times or the complete correction-ID set. `belief_at_fusion_events` ignores missing/extra/unclassifiable records after constructing its accepted subset. Fixed-route selection adds set and reason checks but **omits the status whitelist**. The commissioning loaders implement other checks independently; field_driving uses Python assertions, which disappear under `python -O`. The runtime-report comment claiming that simply calling `assimilations` enforces the contract is false.

**Reproductions:** missing/extra outcomes and reasonless rejection still return from the aligned event helper. Fixed-route selection correctly rejects those examples but accepts `teleported`, contradictory `accepted=0,status=accepted`, and a terminal record whose correction/state stamps disagree with its fused event. The accepted subset changes silently. Duplicate/blank assimilation IDs, an entirely missing schema-4 ledger and reasoned refusals are verified controls: duplicate/blank/missing-ledger cases refuse, while a reasoned rejected/dropped event does not invalidate the drive.

**Shutdown trigger:** final raw fused ID and terminal ID are both `final` at 1.1 s; experiment truth ends at 1.0 s and stop is 1.2 s. The event is correctly accounted but unscoreable. `fused_answers` filters it out for missing truth, then `_selected_runs` compares this truth-filtered set with raw terminal IDs and reports **extra assimilation**. That is a false accounting diagnosis, not evidence the tail is numerically scoreable.

**Smallest repair:** one explicit validator over raw publication/terminal identities, before truth pairing or numerical filtering. Require valid schema, nonempty ID, exactly one terminal outcome, whitelist, refusal reasons, accepted/status consistency, finite times and batch field agreement. Define timestamp consistency tolerances and clock epochs; processing order need not equal capture order. Keep ledger validity separate from reference/posterior scoreability. Every consumer must use the validator rather than reimplementing subsets. Export counts of unscoreable events and reasons.

Schema-3 compatibility remains explicitly labelled `legacy_timestamp_inference` by the helper and is refused by fixed-route selection. That label must not disappear in a higher-level plot. The opportunity consumer also accepts a synthetic schema-3 entry with modern-looking sidecars and an empty hash map; its upstream thesis selector supplies a schema-7 guard, but the consumer itself does not. Old evidence without terminal records cannot support post-update populations.

## A11-03 — Deduplication confuses state time with event identity

**References:** [readings:321](../../experiments/fusion_on_fixed_routes/aligned.py:321), [fused grouping:360](../../experiments/fusion_on_fixed_routes/aligned.py:360), [landed_mask:158](../../experiments/fusion_on_fixed_routes/aligned.py:158), commissioning raw lookup [replay.py:63](../../experiments/icra_commissioning/replay.py:63).

**Trigger/expected:** duplicate, changed, simultaneous or reordered payloads. Exact repeated delivery should be harmless; conflicting copies must be refused. Distinct declared identities must not silently collapse. Current schema must not borrow legacy fallback identities without an explicit compatibility mode.

**Observed synthetic cases:**

- Reversing two different payloads with the same `(batch,camera,capture)` changes selected camera error from **2 to 100 cm**, and fused error from **1 to 100 cm**. `readings` uses first `(camera,round(capture,6))`; `fused_answers` uses the first batch row for fused data and last camera assignment for per-camera data. Neither validates agreement.
- Two batch IDs at the same camera/time become one camera reading but two fused events. One batch/camera with changed capture time becomes two readings but one fused event. Either a supported identity rule or explicit refusal is needed; timestamp equality alone does not settle physical duplication.
- Schema-7 observations with blank source IDs still become timestamp-keyed fused events. Microsecond rounding can additionally merge nearby captures.
- `landed_mask([1,2,1,2])` returns `[T,T,F,T]`, counting time 2 twice. `[NaN,1,1,2]` returns `[F,F,F,T]`, losing the first valid time entirely. It tests adjacent positive differences, not first occurrence. Equal-time revised beliefs are also indistinguishable without revision metadata.

**Consequence:** selection and weighting change with logging/delivery order, and different helpers no longer describe the same population. This affects current periodic-belief consumers when their input stream triggers the pattern; the nine inventory runs had no decreasing belief stamps.

**Smallest repair:** validate `(run/epoch,source_batch_id,camera)` and payload agreement before admission filtering; keep physical capture time as data. For periodic beliefs, use publication/revision identity where available. Under an explicitly declared legacy mode, globally deduplicate identical timestamped values and refuse conflicting ties/reset epochs. Do not silently sort away a reset or reinterpret a revised belief as retransmission.

## A11-04 — Interpolation hides missing temporal evidence

**References:** [TruthSeries:45](../../experiments/fusion_on_fixed_routes/aligned.py:45), [truth_series:100](../../experiments/fusion_on_fixed_routes/aligned.py:100), [aligned_error_cm:127](../../experiments/fusion_on_fixed_routes/aligned.py:127); claimed preference for buffered columns in the module docstring at 25–28.

**Observed:** conflicting truth at time 1 with x=1 and x=9 is accepted; querying time 1 returns 9. A reset sequence `(t,x)=[(0,0),(1,1),(0,100),(1,101)]` is sorted across epochs and returns x=**50.5 m** at .5 s, a bridge between unrelated runs of the clock. Two truth samples ten seconds apart produce an unqualified interpolated midpoint. One NaN heading poisons all later `np.unwrap` output, even where later headings are measured and finite.

`buffered_truth_columns_ignored` defines a triangular path: x(0)=0, x(.5)=1, x(1)=0. A belief at .5 has x=1 and correctly logged high-rate buffered reference x=1/error=0. The exported 10 Hz-style endpoint series alone is x=0 at both endpoints. aligned.py ignores the own-stamp buffered columns and returns **100 cm** error. The current docstring's assertion that it prefers those columns is not implemented. This is a constructed interpolation example, not an estimate of the effect on a drive.

**Expected/consequence:** preserve epoch/order evidence, collapse only identical held truth, refuse conflicting ties, and carry bracket times/gap/reference provenance to each paired sample. Finite interpolation must not imply supported timing. Out-of-range queries correctly return NaN; downstream silent sample removal still needs denominator reporting. A truth gap bound is **not currently specified numerically in the repository contract**: selecting one is an explicit policy decision, not an already-implemented threshold.

**Smallest repair:** validate raw order and epoch before interpolation; define finite-segment yaw handling, maximum supported bracket width and endpoint rules. Prefer a retained full-rate truth ledger or explicitly validated own-stamp reference fields, documenting the source and checking consistency. Do not accept the logger buffer merely because it is finite: its own support/timing policy needs investigation 10/14's validation. Report `gt_stamp_source`, not just the string `gt_stamp`.

## A11-05 — A frozen filename is not a validated experimental design

**References:** [score selection:113](../../experiments/fusion_on_fixed_routes/score.py:113), provenance at 194–220; [navigation freeze:137](../../experiments/icra_commissioning/network_navigation_analysis.py:137), [analyze_run:240](../../experiments/icra_commissioning/network_navigation_analysis.py:240); [network_replay:34](../../experiments/icra_commissioning/network_replay.py:34); [field loader:60](../../experiments/icra_commissioning/field_driving.py:60); [thesis selection:28](../../experiments/icra_commissioning/thesis_evidence.py:28).

**Confirmed triggers and results:**

- A fixed-route manifest with all six provenance fields absent is accepted: a tuple of `None`s counts as homogeneous. Each arm is checked separately, so F1 with `git_sha=source_B` and F4 with no source identity both pass their selectors. There is no cross-arm controlled-provenance check. The chosen fields omit the learned mean and commissioned-world-R identity. Fixed-route run selections name paths without raw-file content hashes.
- A navigation ledger with one expected key plus an unexpected task/arm/seed returns only the expected entry. That entry may point to a manifest with a different task/seed. The real `analyze_run` also accepts the mismatch after its config/field/R checks; the synthetic probe stubs camera-model computation only, not identity validation or numeric scoring.
- Six copies of the same entry satisfy network_replay's list-length check. With the registry path redirected solely to the synthetic selection and model loading stubbed, the consumer constructs a protocol and reaches `load_run` rather than rejecting the duplicated design. It does not validate the unique two-route × three-seed product or uniqueness of run paths/file identities.
- `field_driving.load_run` accepts an empty `files` map, mismatched entry task/seed/model metadata and schema 3 with plausible opportunity sidecars. The probe stubs camera projection/NN loading because all opportunities are misses. This demonstrates consumer validation, not old schema-3 logs acquiring data they never recorded.

**Reachability/limits:** the thesis selector does enforce its hard-coded task/seed set and schema >=7 before ordinary calls, and network_replay verifies the registered selection path, status/count and existing protocol equality. Those guards are useful. They do not validate every consumer input or cross-arm comparison. No duplicate registered entry or mismatched selected manifest was found in the two inspected real selections. Investigation 13 owns the independent upstream runner/resume weaknesses.

**Smallest repair:** centralize a manifest-bound run-set validator: required file digests, unique keys/resolved paths, exact declared task/condition/seed product, manifest/event/config agreement, schema and terminal completeness. Fail on unexplained missing/extra entries; retain failed attempts as attempts. Define a controlled-provenance signature across arms with only the intended treatment fields allowed to differ. Validate the actual named models/geometry/source, not merely nonempty strings or a common Git commit. Do not require unrelated documentation dirty hashes to match as a substitute for executed-source provenance.

## A11-06 — The cache can certify old results using new source metadata

**References:** [field_driving.analyze:159](../../experiments/icra_commissioning/field_driving.py:159), [thesis_evidence:87](../../experiments/icra_commissioning/thesis_evidence.py:87), analysis-source write at 90–92; model loading in [commissioning replay:164](../../experiments/icra_commissioning/replay.py:164).

**Trigger/observed:** place `{run: old_run, field_sha256: old}` in `same_key/results.json`, then request analysis of a different/nonexistent run with changed file hashes. `analyze` returns the old result immediately, before `load_run`, model loading or digest comparison. Thesis evidence subsequently writes hashes of the **current** analysis sources even when every result was returned from cache. Stored entry `field_sha256`/`requirements_sha256` are not checked by the loader. Other replay entry points hash current models into newly written output without universally requiring the previously frozen model bytes.

**Consequence:** a stale result can appear associated with current source/models, or a failed/changed input can be concealed by a cache hit. This is active code; no existing cached score was promoted or diagnosed as stale in this audit.

**Smallest repair:** verify a result fingerprint before any cache return: selected input hashes, model/geometry/Q/initialization/rate/timing policy, analysis dependency hashes and run identity. Record the source that actually produced a cached result. Refuse mismatches or compute into a new versioned output directory; do not overwrite historical evidence or stamp it with new producer metadata. Thesis selection should likewise preserve prior invalid/pending/terminal attempt records rather than silently replacing the overall selection account.

## A11-07 — Different within-run populations share accuracy/gap labels

**References:** [score._score_one:280](../../experiments/fusion_on_fixed_routes/score.py:280), [correction sampling:301](../../experiments/fusion_on_fixed_routes/score.py:301), [aligned.corrections:171](../../experiments/fusion_on_fixed_routes/aligned.py:171), [field_driving:119](../../experiments/icra_commissioning/field_driving.py:119), [navigation analysis:259](../../experiments/icra_commissioning/network_navigation_analysis.py:259), [navigation gap logic:274](../../experiments/icra_commissioning/network_navigation_analysis.py:274), [network gap logic:71](../../experiments/icra_commissioning/network_replay.py:71).

**Observed:** fixed-route `_score_one` takes every finite belief row. Two unique errors `[0,100]` cm give median 50 cm; holding the second belief over five rows changes the reported median to **100 cm**. It also lacks the first-command/stop crop used by current navigation/field scoring. Its correction accuracy comes from sampled `state_stamp` changes rather than every raw fused ID, so multiple publications between ticks disappear even with a complete terminal ledger.

`blind_boundaries` has a drive [0,10] s with updates at 4 and 5. Independent gaps are **[4,1,5]**, longest 5 s. `aligned.corrections` and legacy replay report **1 s** because they only take differences between observed corrections. Zero/one-update behavior also differs. `aligned.corrections` measures fresh published state stamps, including refused corrections, not accepted measurement updates; fixed-route scoring omits the required dropped fraction.

Current navigation analysis improves this: it crops unique beliefs to the mission interval and includes start/stop in gaps between **apply times** of accepted events. Field driving includes endpoints but uses **belief_stamp_after** and calculates dropped fraction over **all** terminal rows, including pre-command events. These are different questions: processing-time availability versus spacing of measurement-state times. Network subset “no reading” gaps use pre-innovation detector/projection hits, not accepted live corrections. The distinction is documented in some outputs but lost in shared generic gap names. Field driving also removes reference-unscoreable batches before constructing readings/availability support; it reports their count, but those batches should not disappear from the sensor-event schedule merely because truth is absent.

**Smallest repair:** define and emit explicit populations and interval bounds for each statistic. Score every fused event once at fused_stamp; score unique supported public beliefs at their own stamp; retain raw opportunity schedules independent of offline reference support. Use accepted terminal events for update gaps, include initial/final windows, and name capture/state-time gaps separately from apply-time gaps. Report drop/reject counts/fractions over the same declared mission population and expose unscoreable samples. Time samples remain correlated within-run descriptive observations, never seed replicates.

## A11-08 — Aggregation and captions overstate what was summarized

**References:** [score.score:409](../../experiments/fusion_on_fixed_routes/score.py:409), [aggregate_section:422](../../experiments/fusion_on_fixed_routes/score.py:422), [compare.per_arm:44](../../experiments/fusion_on_fixed_routes/compare.py:44), [bins:139](../../experiments/fusion_on_fixed_routes/compare.py:139), captions at 180–186, 218–225, 252 and 306.

**Observed:** `score` copies the first run's report, aggregates selected accuracy/consistency fields, and leaves duration, completion, driving/path, odometry and correction-gap metadata from that first run. In a two-run synthetic example with durations 2/100 s, outcomes stuck/collision and different gaps, aggregate `duration_s` is **2**, `completion` is **stuck**, and gaps describe only the first run, although its aggregation sentence says **each displayed point estimate** summarizes both seeds. `completion_counts` correctly retains both outcomes, which makes the surviving single `completion` field particularly easy to misuse.

Missing per-run metric values are dropped from the numeric median without a `n_valid_runs` per metric. The per-run list retains `None`, but the headline still says the point is across all declared seeds. `compare.per_arm` catches every selection `SystemExit` as an absent arm, hiding the reason for invalid evidence; `--partial` can render remaining arms. The [bin loop](../../experiments/fusion_on_fixed_routes/compare.py:153) also omits runs with fewer than five samples, then needs only three surviving run medians, while claiming five paired seeds. It uses `[1,2,3,4]`, omitting the fifth-camera bin entirely. Showcase captions hard-code seed 0 even though the selector may name a different seed. `n_candidates` means manager candidates after prior admission, not all camera acquisition opportunities.

**Correct distinction retained:** the second compare figure deliberately plots p95 **across run median errors**, and its full caption says so. That is not a within-run p95 tail, a confidence interval or a significance test. The console's short `p95` label should retain this qualification. Per-run median aggregation itself is the right hierarchy; the metadata/caption/eligibility problems do not justify pooling frames.

**Smallest repair:** construct a new aggregate schema rather than copying a run. Preserve per-run task/seed/outcome, report declared/valid/unscoreable n for each metric/bin, and either aggregate operational fields explicitly or leave them only under their run IDs. Generate captions, bins and seed labels from validated inputs. Compare paired eligible seeds explicitly; report exclusions and all failed attempts. Invalid evidence must remain an explained invalid entry, not become “no drive yet.”

## A11-09 — Legacy replay and current capture-time replay have different guarantees

**References:** legacy [load:52](../../experiments/fusion_on_fixed_routes/replay.py:52), [bind:94](../../experiments/fusion_on_fixed_routes/replay.py:94), [initialization:162](../../experiments/fusion_on_fixed_routes/replay.py:162), [summary:196](../../experiments/fusion_on_fixed_routes/replay.py:196), [CLI selection:233](../../experiments/fusion_on_fixed_routes/replay.py:233); commissioning [run_filter:82](../../experiments/icra_commissioning/replay.py:82), callers at 191–213; [field_driving:167](../../experiments/icra_commissioning/field_driving.py:167); repaired [network_replay:66](../../experiments/icra_commissioning/network_replay.py:66).

**Legacy confirmed defects:** its CLI discovers run directories by glob/recursive glob, ignores schema/assimilation/provenance, chooses the last path as “held out,” and writes a common output. It reads held GT/odometry using logger time and initializes the estimator from GT. Captures are bound to the nearest logger tick within .15 s and one camera is retained per tick. Synthetic captures .96 and 1.04 both bind to 1.0 and only .96 survives. Thus its prose “applied at the instant the camera saw it” is false even before delay policy is considered. It is a separate 2D world-odometry-increment filter, not the deployed coupled 3-state recursion. Its `distance_angle` rule is inverse-square range only, omitting the runtime angle factor. Its reported `nees` rescales a median by `2/(2 ln 2)`; synthetic raw median/mean 2.5 becomes **3.6067**, neither raw mean nor raw median NEES. These are legacy diagnostics, not current paper evidence; no legacy drive was discovered or rescored here.

**Current commissioning behavior:** it uses measured `/odom_noisy`, declared task initialization, the actual dynamics/Joseph helper, exact capture event times and a common odometry scoring grid. No replay NIS gate or logged live refusals are applied; all eligible camera readings become updates. This is intentionally capture-time idealized. Live image admission and the path were produced by the collection controller; replay cannot establish what another controller would have seen or how navigation would improve.

**Remaining comparison control:** `run_filter` allows a common `prediction_times` grid, and registered exhaustive network replay supplies **all** opportunity times and verifies equal scoring stamps/update counts. Older commissioning `main` and `field_driving.analyze` calls do not. Camera masking/subsampling therefore changes the Euler/Jacobian/Q integration partition as well as camera evidence. The synthetic no-camera turning control `(v,w)=(1,1)` over one second changes both final mean and P solely when prediction-only .25/.75 nodes are inserted. Independently, one Euler step gives XY=(1,0); three steps give `(.25+.5 cos(.25)+.25 cos(.75), .5 sin(.25)+.25 sin(.75))`. This is a numerical-control difference, not a camera effect. Existing common-grid regressions pass; extend the repaired call pattern to every compared arm/rate, without silently replacing frozen outputs.

Neither loader proves dense supported control history: a ten-second odometry gap is silently held at the previous velocity, and sorted timestamp dictionaries conceal original order/conflicting same-time samples. The fake ten-second case propagates x=10 m with no refusal. That demonstrates an **unbounded ZOH assumption**, not a known wrong physical displacement; the replay's supported-gap policy must be declared and missing controls reported. Do not identify live arrival-time performance, calibrated uncertainty, or navigation gain from these computations.

**Smallest repair:** quarantine the old CLI as explicitly historical or route it through validated selection/alignment with its distinct model named. Use one prescribed propagation grid and initial time/control support across current comparisons. Preserve capture-time policy labels in every table/figure/result, and require an event-complete arrival/commit/odometry replay for any future live-policy-equivalence claim.

## A11-10 — A figure dependency reintroduces wrong-time fused truth

**References/reachability:** [story/fusion_examples.load:34](../../experiments/fusion_on_fixed_routes/story/fusion_examples.py:34), grouping/reference at 54–82; imported by [compare.py:32](../../experiments/fusion_on_fixed_routes/compare.py:32) and called at 277.

**Trigger/observed:** a camera capture is .4 s, fused output describes .6 s, and x_GT(t)=t. Fused x=.6 is exact. aligned.fused_answers returns **0 cm**; the storyline uses truth at .4 and reports/draws **20 cm** fused error. Its grouping key is rounded individual camera capture, ignoring batch identity and fused_stamp, so skewed cameras in one physical batch can also become separate moments.

**Reachability limit:** the configured corrected runtime has propagation-to-now disabled and its selected opportunity batches are simultaneous; this particular .2 s mismatch is not established there. It is reachable for supported staggered-capture/propagation variants and historical data, and remains a dependency of the repaired comparison CLI. It is not cured by fixing compare's main score arrays.

**Smallest repair:** obtain moments from validated fused events. Plot original camera readings against their own references, or plot explicitly common-time aligned camera values and covariance against fused-time truth. A single truth marker is valid only after making all displayed quantities common-time. Include batch, state time and named showcase seed in each illustration.

## A11-11 — CLI success and output names do not consistently certify requested work

**References:** [score.main:469](../../experiments/fusion_on_fixed_routes/score.py:469), repaired [compare.main:105](../../experiments/fusion_on_fixed_routes/compare.py:105), [navigation protocol:385](../../experiments/icra_commissioning/network_navigation_analysis.py:385), CLI at 410; legacy replay label/output at [271](../../experiments/fusion_on_fixed_routes/replay.py:271) and 298.

**Observed:** actual score argument parsing with `--taks=fusion_overlap_rich F4` calls all four default tasks. Unknown `--...` options are discarded; selection `SystemExit`s are caught, printed and the command returns **0**. In the synthetic test computation is replaced with a refusing scorer, so no real result is written. `--task NAME` is not handled like `--task=NAME` there. The compare CLI **does** accept both documented spellings and rejects unknown arguments, verified independently.

For a navigation `--config` declaring `fusion_overlap_rich`, seed 999, P0, actual protocol construction still writes task **`fusion_network_traverse`**, seeds **[210]**. A synthetic writer captures the generated payload; source hashing and config parsing are real. The navigation plotting code also hard-codes the traverse goal and P0 preflight background. Unsupported tasks should be refused or handled completely.

**Output identity:** `story_dir(task,arm)` and compare's task-specific output directory correctly separate valid route arguments. Legacy replay still writes `replay/results.json` for any pattern; its per-run key uses `run.parents[3].name/run.parents[2].name`, omitting arm/seed/run ID in the normal campaign/task/arm/seed/run hierarchy, so different runs can overwrite dictionary entries. Current commissioning replay writes fixed `replay_results.json` and run-ID filenames under the chosen output without an existing-protocol equality guard. Re-running with another selection/model can overwrite them. These output paths were inspected, not exercised against study outputs.

**Smallest repair:** one argparse task/arm interface with unknown-option refusal and nonzero exit if any requested analysis fails. Freeze resolved task/seed/condition/source/model/timing policy before output; derive plot goal/background from it. Use complete run identity and an output fingerprint, refusing incompatible reuse. Preserve compare's repaired aliases and route directory isolation.

## A11-12 — Algebraic invertibility is insufficient for NEES/coverage

**References:** [aligned.nees:464](../../experiments/fusion_on_fixed_routes/aligned.py:464), [score covariance mask:289](../../experiments/fusion_on_fixed_routes/score.py:289), [study.score:93](../../experiments/icra_commissioning/study.py:93), [navigation planar scoring:261](../../experiments/icra_commissioning/network_navigation_analysis.py:261), full-pose guard at 287–291.

**Trigger/observed:** residual `(1,0)` and `P=-I` pass the planar positive-determinant test. aligned.nees returns **−1**, which is classified inside the 95% ellipse. Commissioning `study.score` also returns negative Mahalanobis distance and **100% coverage** with a finite Gaussian NLL, because `slogdet`'s determinant sign is ignored. `rms_sigma` becomes NaN. An asymmetric lower triangle is ignored by aligned's closed form. These are malformed-input findings; runtime envelope/reference SPD repairs and the inspected selected inputs do not demonstrate such a covariance occurred online.

**Expected/repair:** verify shape, finiteness, symmetry and Cholesky-positive-definiteness in the declared frame/units before every covariance score. Reject invalid samples/runs according to an explicit policy and disclose denominator changes; do not clip negative NEES into plausible coverage. Preserve full XY cross terms and align P to the same state time/revision as its residual. Navigation's full-pose SPD check is a useful stronger control, but absent/nonfinite pose fields skip it while planar scoring still uses finite matrices alone.

For valid SPD data, independent quadratic-form tests pass. The correct planar targets remain mean 2, median `2 ln 2`, and 95% threshold 5.991; full `(x,y,theta)` coverage uses dimension 3 and wrapped heading. Signed bias, Euclidean error median/mean/RMSE/p95, per-axis RMS sigma, sqrt(trace P) and containment are distinct quantities. `study.score` bootstrapping a single `one_drive` group gives a degenerate NLL interval; it is not a replicate-based uncertainty interval and should be unavailable when independent group count is insufficient.

## A11-13 — Monitoring still loses task, attempt and refusal semantics

**References:** [monitor.collect:49](../../scripts/visibility_comparison/monitor_campaign.py:49), task truncation at 61, summary fields at 72–74, header at 94, progress at 128; [logger aligned means:3088](../../src/experiments/experiments/nodes/experiment_logger.py:3088), [summary dropped fraction:3487](../../src/experiments/experiments/nodes/experiment_logger.py:3487).

**Observed synthetic campaign:** two `fusion_overlap_rich/N1/seed110` attempts, one `fusion_network_traverse/N1/seed110` attempt, and a summary-less seed111. Monitor returns three rows all named **`fusion`**, all seed110; the two routes merge in progress counts, a retry becomes another “seed,” and the summary-less attempt vanishes. `--expect` is a count lower bound, not an exact expected seed set; extra attempts can satisfy it. Per-condition totals pool tasks. It discovers only summary files, so it cannot represent running/failed/no-summary ledger entries. A truthfully malformed summary can also crash collection rather than be retained as an invalid attempt.

**Verified repair:** the synthetic summary distinguishes GT mean .032 m from wheel disagreement .297 m. Monitor correctly selects **.032 m**. Logger currently accumulates buffered own-time GT errors for that mean, so the prior wheel-as-truth substitution is repaired. This remains a logger-tick-weighted mean, not the median/RMSE or unique-belief population used by another evaluator; the display omits sample/timing provenance.

**Remaining label bug:** header “refused” displays `correction_dropped_fraction`, which excludes NIS `rejected` outcomes. One accepted, one rejected, one dropped means total refused 2/3, dropped 1/3. Its `blind_s` is the logger summary gap, whose endpoint/flush semantics differ from offline mission-window gap recomputation. “No plan” also does not establish that computation failed, and many such rows do not prove machine contention; the final restart advice overstates the available diagnostic evidence.

**Smallest repair, coordinated with 13/10:** enumerate exact configured task/condition/seed attempts through the ledger, retain full keys/run IDs and explicit pending/no-summary/invalid states, distinguish retries from seeds, validate hashes before post-hoc use, and disclose missing/extra cells. Label the summary value “mean aligned belief-vs-receipt-timed-GT, logger ticks,” dropped fraction as dropped, and report rejected separately. Keep monitor triage outside evidence selection. Physical contact absence and final goal-reference semantics remain explicit, coordinated with 14.

## A11-14 — Invalid probability inputs can hang or look calibrated

**References:** [metrics.auprc:76](../../scripts/shared/metrics.py:76), tied-score loop at 103–112; [ece:115](../../scripts/shared/metrics.py:115).

**Observed:** `auprc([1,0],[.5,NaN])` reaches its NaN after processing the finite score. The inner equality test is false even against itself, so `j==i` and the outer loop never advances. An isolated process prints “ready,” exceeds four seconds and is terminated by the probe. `ece([1,0],[NaN,2])` returns **0.0** because neither value enters a probability bin. Finite tied inputs give the correct ROC AUC/AP=.5 control.

**Reachability/consequence:** AP currently has no call site among searched experiment/scripts consumers; this is a shared utility boundary defect, not a measured pilot hang. ECE is called by field-study/GP reports; malformed prediction exposure was not established. Other utility functions also rely on callers for shapes/ranges, allowing broadcasting or inconsistent missingness semantics.

**Smallest repair:** validate equal intended shapes, finite labels/probabilities and [0,1] range before scoring, or declare one common exclusion policy with counts. Keep tie groups together; guarantee loop progress. Retain named mean/marginal/group aggregation semantics rather than interpreting every array cell as a replicate.

## Verified repairs, limitations and coordinated next work

The starting regressions and positive synthetic controls verify own-stamp camera/fused/belief errors; no extrapolation outside reference support; ordinary held-camera deduplication; explicit schema-4 ledger requirement for event-belief inference; duplicate/blank terminal ID refusal; reasoned refusal validity in fixed-route/campaign validation; correct valid-SPD NEES targets/formula; compare argument spellings; the monitor's GT field; simultaneous camera updates and the registered network replay's common propagation/scoring-grid controls. They do **not** verify all conditions in the prose contract. Logger row pairing/shutdown repairs and filter Joseph mathematics are owned by other investigations, not claimed as work performed here.

Coordination was sent directly to **Audit runtime logging pipeline** (10), the configuration/campaign investigation (13), and **Audit simulator integrity** (14). Shared findings and repairs to coordinate:

1. **10 with 01/02:** introduce the immutable terminal posterior/revision/frame and complete raw event ledger. Investigation 10 independently reproduced same-clock fusion loss, malformed/duplicate raw-row loss, summary-before-final-callback divergence and summary-write failure. Its exact runtime selections did not show an observed terminal count mismatch. This report's synthetic shutdown example demonstrates the consumer's separate false “extra assimilation” diagnosis. Close/flush/hash in a declared order before freezing a completed selection.
2. **13 with 11:** one validated task/condition/seed/source/config/model selection and cache fingerprint across runner, freeze, replay, analysis and monitor. Investigation 13 separately reproduced runner/resume selection weaknesses; those are upstream of A11-05/A11-13 and should not be silently papered over in a consumer. Publish a new frozen output identity for any changed replay policy or analysis definition.
3. **14 with 10/11:** retain original reference timestamp/clock provenance and contact-channel liveness/entity evidence. Receipt-time truth does not establish bounded capture-time alignment; no-contact messages do not prove healthy collision sensing. Keep simulator poses as offline references and never feed them into online admission, estimation, planning or goal decisions.
4. **07/08 with 11:** preserve the declared robust runtime fusion versus independent sequential replay/planner model distinction, admission population and update cadence. Agreement on a fixed logged path is not an independently driven navigation arm. Equal seeds also do not guarantee identical time-indexed actuation disturbances when noise advances per message.

Smallest repair order: close the shared raw-ledger/quantity contract; refuse unavailable exact posteriors; repair identity/time-support selection; enforce frozen-set/model/cache provenance; then fix aggregation/captions/monitor and CLI output identity. Add desired-invariant regressions from these fixtures when those repairs land. Recompute only under a newly recorded analysis/protocol identity, retaining historical results and all failures.

**Repair-ready interface for 01/10/13:** use one ROS-free `validate_correction_ledger(publications, outcomes)` core, with a file/schema wrapper in aligned.py. Its inputs are raw records, never truth-filtered score arrays. Return a validated batch map, accepted-update IDs and refusal counts; raise a structured integrity error for missing/extra/duplicate/blank IDs, unknown status, reasonless refusal, contradictory accepted flag, or inconsistent required event fields. Keep schema/capability checks explicit rather than falling back on parsing failure. Correctly repeated per-camera representations of one publication are collapsed only after agreement checks. A separate `score_public_beliefs` path may use supported periodic samples; `score_committed_posteriors` must fail with an unavailable-evidence reason unless committed state/P/frame/time/revision exist. Neither function may manufacture posterior records. Convert A11-01/02 fixtures to desired-invariant tests first, then require campaign, field, navigation, fixed-route and report consumers to share those decisions. This interface proposal requests coordinated ownership, not an edit to shared runtime files.

No live DDS ordering/loss, full process reset, physical command/contact response, source-to-installed equivalence for every process, continuous ground-truth transport delay, or navigation counterfactual was measured. No runtime repair or replay result in this audit supports a navigation-improvement claim.
