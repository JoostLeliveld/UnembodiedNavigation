# Ground anchoring and visibility inference from monocular depth

**Question this study answers:** can a fixed warehouse camera work out, from its own
RGB frame alone, where it would and would not be able to see the robot — including
after someone leaves a pallet in the aisle?

No new hardware, no depth sensor, no CAD model of the obstacle. The inputs are the
things a deployed camera already has: its calibration, the drivable-floor map the
planner already uses, and one monocular depth prediction of its own image.

This folder is **agent 3** of a three-part effort:

| agent | owns | this folder's relationship |
|---|---|---|
| 1 | dynamic Gazebo world, controllable obstacles, oracle depth / poses / visibility | supplies frame records; its oracle outputs are **evaluation-only and never read here** |
| 2 | monocular depth adapter (Metric3Dv2, UniDepthV2, Depth Anything V2) | supplies one `DepthPrediction` per frame |
| 3 | **this** — ground anchoring, metric recovery, visibility inference | consumes both, emits the visibility field |

## The idea in one paragraph

A camera bolted to a warehouse wall already knows the depth of every pixel whose ray
lands on the floor: intersect the ray with the floor plane and read off the distance.
That is a free ruler, available at deployment, needing no ground truth. A monocular
model produces depth that is correct in shape but wrong in scale — so fit the model's
output onto the ruler, robustly, and the rest of the image becomes metric too,
*including the pallet the floor plane cannot explain*. Back-project it, and ask for
each candidate robot position whether anything now stands between it and the camera.

```text
RGB ──► [agent 2] monocular depth (some convention, maybe not metric)
                        │
   camera calibration ──┤
   drivable-floor map ──┤
                        ▼
        floor anchors: pixels whose true depth is known analytically
                        ▼
        robust affine fit, in the model's own depth convention  ─► refuse the frame if it does not hold up
                        ▼
        metric depth + per-pixel sigma
                        ▼
        ├─► 2.5-D obstacle-height map + observed mask   (the planner-facing map)
        └─► sightline test per robot position           (the visibility answer)
                        ▼
              p_visible / p_occluded / p_unknown
```

## What it emits

`estimate_visibility()` returns a `VisibilityResult` carrying the full agreed contract:

- **metric depth estimate** and **depth uncertainty** — `result.metric_depth.depth_m`, `.sigma_m`, `.valid`
- **ground-fit parameters** and **residual / inlier count** — `result.ground_fit`
- **frame validity status** — `result.status`
- **p_LOS grid** and **unknown mask** — `result.visibility`
- provenance: model name, checkpoint, declared convention, config fingerprint, and
  which frame-record keys were deliberately withheld

`p_visible + p_occluded + p_unknown == 1` in every cell. `p_los` is the ratio
`p_visible / (p_visible + p_occluded)` and is **NaN** where the cell is wholly
unknown, so nothing downstream can mistake ignorance for a coin flip.

## Five decisions worth knowing about

**The affine fit lives in the model's own convention.** `metric_z`, `euclidean_range` and
`relative_depth` fit `z = a·p + b`; `inverse_depth` fits `1/z = a·p + b`. Fitting a
disparity map in depth space is not a worse fit, it is the wrong model, and it fails
hardest exactly where obstacles are. The declared convention is also *checked* against
the data: in the correct space the slope must be positive, so a mislabelled prediction
raises `DepthConventionError` rather than being quietly fitted around.

**The method is allowed to refuse.** Too few floor anchors, anchors bunched at one
range, an ill-conditioned design, too many outliers, residuals that stay large — each
returns a named `FrameStatus` and an all-unknown field. The depth-span gate is the
interesting one: anchors at a single range pin down the shift but barely constrain the
scale, the same identifiability problem as two cameras on nearly the same bearing being
unable to separate their offsets. A frame like that is refused, not fitted.

**Robustness is one-sided by physics.** A box standing in the aisle occludes floor the
drivable map calls clear, so a minority of anchors report the box rather than the
floor. RANSAC on a range-scaled tolerance rejects them, and `n_shorter_than_floor` is
reported because those rejects are evidence something is standing there.
`n_beyond_floor` should stay at zero — anchors *behind* the floor plane are physically
impossible and point at bad calibration, not at occlusion.

**The sightline test is a depth-buffer comparison, not a march over the grid.** For a
body point, project it and ask whether the measured depth reaches at least that far.
That is the exact ray cast at image resolution; marching the rasterised height map
answers the same question with the grid's rounding added — a 0.8 m box becomes a
1.05 m box on a 0.25 m grid, and its shadow grows to match. The comparison is
probabilistic, `P(measured ≥ body-point depth)` under the fitted depth sigma, so a
sightline the camera cannot resolve lands near 0.5 instead of being forced to a
confident answer. Both depths lie on the same pixel ray, so their errors are correlated
and the *difference* is much better determined than either absolute depth.

**Hidden by something we can see is occlusion, not unknown.** The ground behind the box
has no depth return of its own, but a robot there would be invisible *because of the
box* — a reason we can name. `unknown` is reserved for genuine absence of evidence:
pixels the adapter marked invalid, positions outside the image, and frames whose ground
fit was refused.

## Optional online anchoring (the two post-study improvements)

The frozen study path remains the default. Two opt-in additions are available for an
online map that is refreshed repeatedly:

1. `AnchorConfig(quality_filter=True)` erodes an externally supplied floor mask,
   removes high-gradient depth boundaries and their neighbourhood, and uses native
   model confidence or a genuine uncertainty map to rank and weight the remaining
   anchors. Confidence is treated only by rank because its raw units are
   model-specific. A larger-is-better confidence score is never propagated as a
   metric depth standard deviation.
2. `TemporalGroundAnchorFilter` maintains a separate Gaussian posterior over
   `[scale, shift]` for every camera/model/depth-convention tuple. It forgets old
   information with a configurable half-life, robustly downweights plausible jumps,
   rejects gross innovations, and can reuse a recent affine when one frame's floor
   fit is refused. The stale limit is 30 s by default.

Only the two affine parameters are remembered. Current-frame depth and obstacles are
always recomputed, so a removed pallet is not left behind by temporal map memory.
Neither option reads a depth oracle or object pose. `floor_segmentation` is an optional
method input to `estimate_visibility()`; if no segmentation system is installed, the
enhanced mode still uses geometric floor candidates, depth edges, and any confidence
provided by the depth adapter.

## Talking to the other two agents

Both peer contracts are read in their own on-disk form, no translation step and no
re-export asked of them:

- **Convention names match the adapter's** (`metric_z`, `euclidean_range`,
  `relative_depth`, `inverse_depth`), so a prediction crosses the boundary unchanged.
  `metric_range` and `disparity` are accepted as synonyms; anything else raises.
- **Predictions** load from the adapter's `<image_id>__<model_name>.npz` +
  `.json` sidecar pair — pass either path. `prediction_index()` maps image ids to
  sidecars so frames and predictions join by the adapter's own id, and `--model`
  disambiguates a directory holding several models for one image.
- **Calibration** accepts both spellings in circulation: the scenario runner's
  `img_width`/`img_height` + `cam_pos`/`look_at`, and plain `width`/`height` +
  `R_world_to_cam`/`T_cam_world`. Anything unrecognised raises rather than being
  guessed at.

## The boundary, enforced in code

Oracle depth, simulator obstacle poses and oracle visibility grids are evaluation-only.
`io_contract.method_visible_record()` whitelists the six method-visible keys and
returns everything else under `_withheld`, so a run log shows the oracle fields were
present and deliberately not consumed. `assert_no_oracle_access()` raises on request,
and a test asserts that no module in the package other than the one that refuses them
even mentions their names.

## Robot target volume

Sightlines are cast to a small upright cylinder, not a point: 0.14 m radius,
z ∈ [0.05, 0.20] m, which is the TurtleBot3 Burger. Parts of this repo use a 0.35 m
marker height, which is *taller than the robot* and therefore anti-conservative for
occlusion — a shorter body is easier to hide. `TargetVolume` is an explicit input here
so that choice is visible rather than inherited.

## Layout

```text
ground_anchoring/
  contracts.py     typed interface shared with agents 1 and 2; configs; outputs
  conventions.py   depth-convention handling and the loud failures
  floor_anchors.py which pixels are trustworthy — floor segmentation lives here
  ground_fit.py    robust scale/shift fit, validity gates, uncertainty propagation
  temporal.py      optional per-camera Bayesian filter over affine scale/shift
  heightmap.py     back-projection, 2.5-D height map, observed mask
  raycast.py       sightline test to the robot volume
  pipeline.py      end to end; single-frame by default, temporal affine opt-in
  io_contract.py   agent-1/agent-2 I/O and the oracle boundary
run_frames.py      CLI: scenario manifest + predictions -> results + index.csv
```

Floor segmentation is deliberately inside this package: choosing anchors *is* the
segmentation problem for this method, and it is not a separate contribution.

## Running it

```bash
python3 experiments/mono_depth_visibility/run_frames.py \
    --manifest    logs/studies/<scenario>/frames.json \
    --predictions logs/studies/<scenario>/depth/<model> \
    --out         logs/studies/mono_depth_visibility/<run>
```

For the online version, add `--enhanced-anchors --temporal-anchoring`. These flags are
absent by default so reproducing an earlier result does not silently change methods.

Outputs one `.npz` + `.json` per frame plus an `index.csv` timeline, so a visibility
change can be lined up against the scenario's event timestamps.

## Status and what counts as evidence

The method and its acceptance suite are complete and green
(`tests/experiments/test_ground_anchoring.py`, 60 tests, run 2026-08-19): exact plane →
exact recovery; scale/shift recovered in all four conventions; corrupted floor masks
handled and majority corruption refused; insufficient evidence returns unknown; a known
obstacle casts the shadow the trigonometry predicts and removing it restores the field;
convention mismatches raise. The added tests cover segmentation erosion, confidence
semantics, weighted fitting, temporal fusion, innovation rejection, stale expiry, and
independent camera states.

Those are unit tests of algebra and geometry against hand-checkable answers — they are
**not** experimental evidence, and no number from them belongs in a result. Operational
evidence requires agent 1's dynamic Gazebo scenarios run through agent 2's real
monocular predictions. Until that exists, this folder has apparatus, not findings.

Still deliberately absent: cross-camera fusion (that is the belief filter's job), and
oracle access inside the method package. Scoring lives in separate evaluators allowed
to open both sides. The temporal option now has a 21-update × four-camera longitudinal
Gazebo replay and a predeclared anchor-dropout test in
`experiments/usable_observation/supervisor_comparison/10_monocular_depth_results/temporal_anchor_sequence/`.
It supports recent-affine fallback through isolated anchor failures, but the deterministic
untouched sequence does not establish a natural drift/stability gain; that requires a
real-camera duration and lighting campaign.

**Not registered in `research/registry.yaml`.** That file records claims, and this study
has none yet — apparatus does not earn an entry. The entry goes in when the first real
scenario run produces a finding, with `study_path` pointing here.

**Reused rather than rebuilt:** `unav_common.camera_model.ObliqueCameraModel` supplies
the camera convention (`CameraCalibration.from_oblique` adopts one unchanged), and the
height rasteriser is checked for agreement against
`geometry_visibility.height_map_from_points` — it carries a per-cell sigma that helper
cannot return, so it is a superset with a parity test rather than a fork.

**Known limit:** the published height map's `observed` mask depends on the grid being
coarser than the ground sample spacing at the far end of the image, or distant cells
look unobserved for a sampling reason rather than a physical one. The forward
depth-buffer test removes most of it, and the pipeline records
`unobserved_in_fov_fraction` so what remains is visible rather than silent. The
sightline answer does not depend on the grid at all.
