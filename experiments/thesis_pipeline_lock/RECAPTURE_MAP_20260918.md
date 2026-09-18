# Recapture map — the 2026-09-18 world edits

What changed in the world, which measurements it invalidates, and what has to be
re-run. Written before any recapture so the scope is a decision rather than a
discovery.

## 0. The framing this recapture serves

**This is a COMMISSIONING problem, not a generalisation problem.** The warehouse,
the five camera poses and the detector are a fixed installation. The correction,
`R` and `q` are fitted TO that installation on purpose. Learning every part of
the workspace — including the exact ground the robot will drive — is the intended
outcome, not leakage.

Consequences for how this data is split and reported:

- There is no "unseen ground" to generalise to. Wide spatial holdouts are not
  required to license the method, and a spatial model that interpolates the
  lattice closely is doing its job.
- A close holdout (~10 cm) is sufficient, and its purpose is narrow: show the
  model did not memorise the exact IMAGES. It is a reproducibility check, and
  that is all it needs to be.
- The acceptance test is NAVIGATION. Offline NLL and containment rank candidate
  models; they do not decide whether the method works. Closed-loop performance
  does.

One requirement survives this framing and must not be dropped: the mean and the
covariance may not be fitted to the SAME residuals. `R` describes the error of a
FROZEN correction, so it needs observations that correction did not train on.
That is about valid uncertainty, not about generalisation.

## 1. What changed

Two edits to `src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf`:

| # | edit | detail |
|---|---|---|
| 1 | camera C pitch | 38° → 44° → **48°** (`0.6632` → `0.8378` rad) |
| 2 | Cm barrels de-stacked | 30 visual drum links removed (tiers 1–2); the four `obs_Cm_p*` collision prisms capped 2.64/1.76 m → **0.88 m** |

`Cs` and `Cn` stacks are untouched. No other model, camera, light or declared
region changed. Camera heights stay at 5.00 m for all five.

World identity:

    captured against   49655f69…   (recorded in camera_capture_map_manifest.json)
    on disk now        c46f4afb…

Note these are **three** different hashes across the session — the capture
manifest's `49655f69…` matched neither the pre-edit file nor the current one, so
the world had already drifted from its manifest before these edits. That is a
separate defect and is NOT resolved by this recapture.

## 2. Which regions changed

- **Camera C's entire image.** A pitch change re-renders every pixel, so every
  view from that camera is different regardless of where the robot stood.
- **The Cm volume**, x ∈ [2.25, 9.40], y ∈ [−0.02, 1.18], z ∈ [0.88, 2.64]. Two
  tiers of drums no longer exist there. This changes (a) sightlines that passed
  through that volume and (b) the appearance of any frame containing it.

Only cameras **D** and **E** have a Cm stack inside their frustum; A and B are
affected only where a stack sat between camera and robot.

## 3. Which rows are invalidated

A row is invalidated only when **the robot's own appearance or the sightline to
it** changed. A barrel elsewhere in the frame is background: the stage 04 label
contract is a function of the ROBOT's semantic mask and bbox only —
`nonzero_semantic_robot_mask`, `min_semantic_area_px` 256, `min_bbox_*_px` 16,
`min_visible_to_projected_*_ratio` 0.55, `forbidden_border_contact_px` 2,
`max_bottom_edge_gap` — and stages 06–08 read confidence, box size, border
contact, hull agreement and the robot's residual. Background content is an input
to none of them.

    total capture rows        22130
    affected                   6402   28.9%
    unaffected                15728   71.1%
    positions touched       695/695   100%

By reason:

    camera C pitch                 3934
    barrel occluded the robot      1976
    both                            492

By dataset role:

    role                    rows  affected      %   positions
    detector_fit            3200       984   30.8%      80/80
    detector_validation     1600       560   35.0%      40/40
    commissioning_fit       9600      2560   26.7%    240/240
    final_audit             1600       368   23.0%      40/40
    extension_edge          5650      1746   30.9%    283/283
    extension_aisle          480       184   38.3%      12/12

**Positions are still 100% touched in every role**, because camera C photographs
every pose. The row percentage says how much EVIDENCE changes; the position
count says how many poses must be revisited, and that is all of them.

One exception to "background does not matter": the runbook allows copy-paste
augmentation using `detector_fit` **backgrounds**. Any background that contained
a three-tier stack no longer matches the world. That bears on whether to
RETRAIN, not on what to recapture.

## 4. Why a partial recapture is not possible

`BATCHED_CAMERA_ORDER` is `(camera_A … camera_E)` and the batched detector emits
a batch only when every camera in the contract has contributed a frame. A pose
is captured as one five-camera batch, so "re-capture only camera C and D/E"
cannot produce a valid row. Since every position is touched, the recapture is
the full 695-position map.

## 5. Stages that must re-run

From `pipeline_lock.json`. Stages 01–05 are marked `locked`; this breaks that
seal on 03, 04 and 05.

| stage | status now | after recapture |
|---|---|---|
| 01 warehouse | locked | **world hash changed — re-freeze** |
| 02 robot_target | locked | unaffected (robot unchanged) |
| 03 camera_capture_map | locked | **re-capture all 695 positions** |
| 04 labels_and_dataset | locked | **re-label**: 31% of detector_fit and 35% of detector_validation rows changed |
| 05 detector_training | locked | **re-train**: its training set changed |
| 06 detector_gate | pending | re-run on the new commissioning_fit |
| 07 correction_and_covariance | pending | refit correction + R ladder |
| 08 availability_model | pending | refit q GP |
| 09–11 | pending | downstream |

**Stage 05 is the expensive and contentious one.** `AGENTS.md` calls the
detector frozen, and `OPERATING_LOCK.json` records that re-freezing after images
exist "would misrepresent the evidence chain". Re-training is a deliberate
decision to move a locked artifact, not a mechanical consequence.

An alternative worth stating explicitly: keep the current detector weights and
re-run only stages 06–08 on fresh images. The detector would then be applied to
a world it was not trained on. That is defensible (a fixed detector is the
thesis premise) but the training/deployment mismatch must be declared.

## 6. Order of operations

1. Re-freeze the world (stage 01) and record the new hash everywhere the old one
   appears: `camera_capture_map_manifest.json`, `OPERATING_LOCK.json`,
   `world_freeze_manifest.json`.
2. Re-capture all 695 positions × 8 headings × 5 cameras. The v3 lattice
   (`camera_capture_poses_v3.json`) and the v4 extension both still apply — the
   position map did not change, only what the cameras see.
3. Audit the capture per `POST_CAPTURE_RUNBOOK.md` stage 04 step 1–2
   (`audit_master_capture.py --verify-pixels`).
4. Re-label detector_fit + detector_validation; re-train or explicitly decline to
   re-train (see §5).
5. Re-run the gate (06), refit correction and R (07) and q (08), keeping every
   heading of a position in the same fold.
6. Open `final_audit` once, after 06–08 are immutable. No tuning after.
7. Re-run the navigation campaign (09) and regenerate the figures.

## 7. Artifacts that are stale until §6 completes

- the blind-region map: DELETED rather than kept stale. Regenerate with
  `experiments/warehouse_v2_sketches/plot_blind_regions.py` once `q` is refitted.
- `experiments/warehouse_v2_sketches/system_maps.png` (panels 3 and 4)
- `experiments/warehouse_v2_sketches/visible_corridors.png`
- every `*_planner_field.npz` under
  `logs/studies/reference_controlled_commissioning_v1/selected_correction_r_planner_fields_20260915_v4/`
- every route-choice number produced this session, which additionally used
  `ambiguity_weight` 3.0 before it was reset to 1.0

The camera-C tilt was justified on **frustum geometry only** — it puts 48 of the
61 patch cells in frame, up from 38, but it has never been shown to raise `q`
above 0.10 there. The 38 cells that were already in frame at 44° stayed blind,
which is evidence that being in frame is not the binding constraint. Re-check
that claim against the refitted `q` before repeating it.
