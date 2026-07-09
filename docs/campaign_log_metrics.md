# Campaign log metrics — canonical columns (read before computing any metric)

The visibility-comparison per-run logs (`logs/visibility_comparison/<campaign>/<route>/<C>/<seed>/experiment_*/`)
have **~216 columns**, including **six overlapping position fields**. Picking the wrong
one silently produces garbage. Several fields are **stale**. This is the single source
of truth for which column means what.

**Always load metrics via `scripts/geometry_visibility/campaign_metrics.py`** — it exposes
only the canonical fields and asserts them on load (`assert_canonical`). Do not hand-pick
position columns in analysis code.

## Canonical fields — use these

| quantity | column(s) | file | notes |
|---|---|---|---|
| **belief (pose estimate)** | `planner_belief_x`, `planner_belief_y` | experiment.csv | == `est_x/est_y`; reproduces the GT-error column exactly |
| **ground truth** | `gt_x`, `gt_y` | experiment.csv | GT-bridge truth. **EVALUATION-ONLY** — score/compare against it, never feed it (or exact CAD shelf geometry/heights) to the model as a deployment input |
| **belief error** | `belief_error_gt_m` | experiment.csv | == \|\|belief − truth\|\|, verified across all 43 runs |
| **reported uncertainty** | `state_sigma_major_m` | experiment.csv | 1σ major axis; **OVERCONFIDENT** (median 0.017 m vs actual error median 0.051 m) |
| **detection** | `detected` {0,1}, `yolo_score_raw` | perception.csv | join to experiment by nearest `log_stamp`↔`stamp` |

## Stale / deprecated — DO NOT use as belief or truth

| column | why not |
|---|---|
| `state_x`, `state_y` (experiment.csv) | **STALE** — updates rarely, frozen for long spans. \|\|state − gt\|\| reaches **4.5 m** while the real belief error is **< 0.35 m**. |
| `state_x`, `state_y` (perception.csv) | also stale (a snapshot that lags). |
| `truth_x`, `truth_y` | wheel-odometry "truth" from before the GT bridge (contaminated). Use `gt_x/gt_y`. |

## Verified relationships (checked over all 43 honest_campaign_v1 runs)

- `||planner_belief − gt||` and `||est − gt||` each equal `belief_error_gt_m` to within 1e-3 m (max diff 0.000).
- `||state − gt||` deviates from the column by up to **4.494 m** → `state_*` is not the operational belief.
- Real belief-vs-GT error: **p50 0.051 m, p95 0.127 m, max 0.350 m** — the belief stays close to truth; it does **not** explode.

## Mistakes this prevents (all actually made before this doc existed)

1. Using `state_x/state_y` as "belief" → fake heavy-tailed error (p95 1.65 m) → a bogus
   "spreading the update over the belief helps" result. With the canonical belief the
   error is < 0.35 m, well under the 0.6 m kernel, so position handling is irrelevant
   (naive ≈ oracle) — the finding was retracted.
2. Cross-file stamp joins that look clean on the median but hide stale rows — always run
   `assert_canonical` after loading.
3. Treating `state_sigma_major_m` as calibrated — it is not (≈3–4× too small); any
   certainty-weighted scheme must account for that or inflate it.

## Recipe

```python
import campaign_metrics as cm
ev = cm.load_detections("logs/visibility_comparison/honest_campaign_v1")  # self-checks on load
# ev[i]: belief=(x,y), truth=(x,y), belief_error_m, reported_sigma_m, detected, yolo_score
```
