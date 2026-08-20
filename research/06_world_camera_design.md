# World and camera design

Worlds and cameras are described by **measurable properties**, never by anonymous
"Warehouse A/B/C" or optical-archetype labels. This document defines the property schema, the
route archetypes, the grouped splits and the noise contract. The assumption register and the
two frozen-control contracts are in [`03_assumptions.md`](03_assumptions.md); unresolved
decisions are numbered there (§8) and referenced here as `U#`.

A value marked **UNMEASURED** has a definition and a recipe but no number yet. It may not be
quoted, and a split that depends on it may not be frozen.

---

## 1. Measurable world properties

Every world and every benchmark split must report this schema, and each split must state
which property changes and which are held fixed.

| Property | Definition | Recipe |
|---|---|---|
| Reachable floor area | Area of the planner-driveable union, excluding conservative no-go envelopes | Sum of declared `traversable` lanes in `world_profiles.yaml` minus `non_driveable_obstacle` envelopes |
| Occlusion density | Occluder footprint area above the target height, per unit reachable floor | Parse occluders from the world SDF (evaluation-only, D0); count those with height > target height |
| Aisle openness | Mapped lane width, per lane class, and its ratio to the robot's collision diameter (0.25 m) | Lane extents in `world_profiles.yaml` |
| Camera-overlap fraction | Fraction of reachable cells with ≥ 2 cameras at `p_use ≥ τ`, for a declared τ | Threshold the per-camera fields on the reachable mask; `coverage_count` in the fused artifact is the untresholded precursor |
| Camera-poor floor fraction | Fraction of reachable cells with `max_c p_use,c < τ` | Same mask and τ |
| Symmetry | Fraction of reachable cells whose mirror-image counterpart differs in `max_c p_use,c` by more than δ | Reflect the field about the world's design symmetry axes |
| Range distribution | Distribution of camera-to-candidate-pose range over reachable cells, per camera | Calibrated camera pose against the reachable mask |
| Route alternatives | Number of distinct lane-graph routes within X % of the shortest, and the length ratio of the best well-observed alternative to the shortest | Lane-graph enumeration plus the frozen field |
| Layout-change frequency | Number and extent of declared layout states | Declared per world; currently 1 (nominal) everywhere |

Two properties are already measured on the four-camera world and may be quoted:

- 43,758 reachable cells; achievable localization σ median 0.0256 m, p90 0.0781 m, max
  0.0936 m.
- On 15.7 % of reachable cells the best-coverage camera is *not* the best-precision camera,
  at a median penalty of 0.0357 m (p90 0.0546 m).

Both come from `logs/studies/achievable_precision_map/exp1_precision_vs_coverage/summary.json`.
Overlap fraction, camera-poor fraction and symmetry are **UNMEASURED** under a declared
`(τ, δ, reachable mask)` and must be computed before any split that names them is frozen.

---

## 2. The worlds, as measured

**The two-world hard rule was RETIRED on 2026-08-20.** It confined method development to
`warehouse_aws` and reserved `warehouse_full_4cam` for frozen-method evaluation. It predated
`warehouse_v2`, which is now the development world: five cameras mounted below the roofline,
crossing angles spread uniformly over 0-180 degrees, six independent cycles in its lane
network, and a restock that moves 13,554 camera-cell sight-line pairs against the old world's
575. Nothing is reserved any more. Choose a world for what it measures, and record the reason
in the study README.

| Property | `warehouse_aws` | `warehouse_full_4cam` |
|---|---|---|
| Role | Method development, single camera | Frozen-method evaluation, four cameras |
| Ground plane | 11.0 × 10.0 m | 24.5 × 20.5 m; walls at x = ±12, y = ±10 |
| Site boundary | — | x ∈ [−11.20, 11.50], y ∈ [−8.60, 8.60] |
| Declared lanes | 10 traversable regions plus 9 non-driveable staging pads | 16 traversable lanes plus 2 conservative no-go envelopes |
| Rack-aisle width (mapped) | 0.90 m (A1–A3), 1.10 m (A4) | 0.91 m |
| Cross-aisle width | 0.65 m (upper), 0.78 m (mid) | 0.96 m mapped between green envelopes for the two interior cross aisles (1.60 m physical); 1.13–1.53 m for the perimeter aisles |
| Main corridor | Lower main aisle, 0.82 m | Central aisle 4.50 m physical, 3.86 m mapped, narrowing to two 1.36 m bypasses around the pillar |
| Service lanes | West lane 0.55 m; apron lane 0.60 m | West 1.53 m, east 1.83 m |
| Tall occluders | R4 high stack deliberately removed; low crates and rack geometry only, with `r4_occluder_variants` at 1.20 / 1.90 / 2.60 m available but not active | Four tall handover segments at 2.61 m: `W2_north`, `W3_north` (west block, north), `E2_south`, `E3_south` (east block, south) |
| Deliberate asymmetries | A4 low-reliability band versus the A3 detour | Tall segments placed on opposite quadrants for the north and south camera pairs; one west wall-backed shelf reachable only from its east side; a fixed 0.50 × 0.50 m building pillar inside the central aisle |
| Cameras | 1, at (0.00, −5.50, 4.80), pitch 0.92 rad | 4, at (±6.00, ±10.00, 6.10), pitch 0.92 rad |
| Field grid | 220 × 200 over x ∈ [−5.5, 5.5], y ∈ [−5.0, 5.0] | 240 × 184 (fitted fields) and 470 × 360 (planner profile) over x ∈ [−11.7, 11.7], y ∈ [−9.0, 9.0] |
| Target height | 0.35 m | 0.35 m — see `03_assumptions.md` A17 and U11 |
| Layout states | 1 (nominal) | 1 (nominal) |
| Dynamic occluders | None | None |

Note the mounts are **not** the same across worlds: 4.80 m in `warehouse_aws` versus 6.10 m in
`warehouse_full_4cam`, at the same pitch. Constants derived for one mount height do not
transfer to the other. The two worlds consequently do not even share a detector —
`warehouse_yolo_detector_v1` versus `warehouse_yolo_detector_4cam_v3_960` — so a
cross-world comparison changes the camera geometry *and* the perception process at once and
is not an admissible OFAT arm under A03/A08.

---

## 3. World splits required by benchmark B

The principal comparison needs three world states:

1. a relatively symmetric, low-to-moderate-occlusion world for mechanism checks;
2. an asymmetric world with unequal route alternatives, uneven overlap and camera-poor
   regions, for route discrimination;
3. a **changed-layout variant** of (2), for staleness and transfer.

States (1) and (2) map onto the older pair. State (3) now **exists**: `warehouse_v2` ships
with a matched restocked variant (`warehouse_v2_shipout`), generated from one layout so the
lane network is identical between them. The rule that once blocked this is retired. `whatif_layout_change.py` predicts the effect of dropping a 2.6 m pallet into an
aisle but never measures it, and rungs D3–D5 of the geometry ladder are unimplemented. This is
`U3`, and it blocks the B4 transfer gate, the D4/D5 arms and route archetype R5 below.

Each split must state which property in §1 changes and which are held fixed. A split that
changes two properties at once is not admissible as an OFAT arm.

---

## 4. Measurable camera properties

| Property | Definition |
|---|---|
| Mount pose | (x, y, z) and (roll, pitch, yaw) in world frame, from the world SDF |
| Optics | Horizontal FOV or focal length; derived vertical FOV |
| Sensor | Resolution and pixel pitch |
| Rate | Render rate and the *effective* measurement→belief update rate, which may be much lower |
| Floor footprint | Near and far intersection of the image extent with the floor plane along the optical axis |
| Range distribution | Measured camera-to-target range over the reachable mask and over the actual captures |
| Overlap type and role | Which cameras share coverage, and whether the pair is adjacent-on-a-wall or facing |
| Occlusion exposure | Which tall structures shadow this camera, and the resulting blind regions |
| Calibration bias | Measured persistent along- and cross-bearing bias, held out from the fit |
| Conditional covariance | `R_cond` given a detection, per camera |
| Residual correlation floor | The persistent component that does not average away across views |

---

## 5. The four cameras, as measured

| Property | camera_A | camera_B | camera_C | camera_D |
|---|---|---|---|---|
| Mount | (−6.00, −10.00, 6.10) | (−6.00, 10.00, 6.10) | (6.00, −10.00, 6.10) | (6.00, 10.00, 6.10) |
| Wall / role | South, west column | North, west column | South, east column | North, east column |
| Yaw / pitch | 1.5708 / 0.92 rad | −1.5708 / 0.92 rad | 1.5708 / 0.92 rad | −1.5708 / 0.92 rad |
| Optics | 90° HFOV, 58.7° derived VFOV | identical | identical | identical |
| Sensor / rate | 1280 × 720, 5 Hz render, ≈2.2 Hz effective | identical | identical | identical |
| Floor footprint along axis | ≈0.85 m to ≈14.1 m from the mount, axis crossing at 4.64 m | identical | identical | identical |
| **Current IPM mean measurement error** (balanced set-pose data) | 0.0646 m | 0.0681 m | 0.0666 m | 0.0671 m |
| **Current IPM signed lateral bias** (same data) | −0.0068 m | +0.0144 m | +0.0188 m | −0.0160 m |
| Historical v2 signed lateral bias (confounded driving data; not current accuracy) | −0.0071 m | +0.0123 m | +0.0769 m | −0.0323 m |
| Conditional σ (`R_cond`, anchor 0.05 m) | 0.0267 m | 0.0127 m | 0.0250 m | 0.0224 m |
| Historical-v2 best-precision share (sensitivity only) | 31.0 % | 28.0 % | 14.8 % | 26.2 % |
| Best-coverage share of reachable floor | 25.0 % | 25.0 % | 25.0 % | 25.0 % |

Range is a shared row: capture medians run 5.4–12.8 m across the three commissioning
captures, and the pixel-ground dataset spans 1.7–16.6 m. Occlusion exposure is also shared
in kind but not in effect — the 2.61 m tall segments sit on the north side of the west block
and the south side of the east block, so the north and south pairs have structurally
different blind regions, but the per-camera blind areas are **UNMEASURED**.

The current IPM rows are the only fair A–D accuracy comparison: all four cameras are scored
on the same detector dataset, four-yaw set-pose protocol, projection, and truth reference.
The historical v2 row is retained only to identify the input to the locked belief-mechanism
study. Its camera, route, region, and yaw are confounded; the +76.9 mm Camera C component
collapses to +8.1 mm after accounting for the robot silhouette. It must not be interpreted as
current Camera C accuracy or hardware quality. See `docs/localization_metrics.md`.

Sources: `docs/warehouse_full_4cam_layout.md`; `src/sim/models/external_camera*/model.sdf`;
`logs/studies/achievable_precision_map/exp1_precision_vs_coverage/summary.json`;
`logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json`;
`logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting/summary.json` (historical v2 mechanism only);
`logs/studies/operational_residual_rcond/exp1_timing_and_coverage/timing_and_coverage.json`.
Floor-footprint figures are derived from the documented mount geometry and should be
recomputed with the camera model rather than quoted from here.

---

## 6. What four optically identical cameras can and cannot support

All four are the same model at the same height, pitch, FOV, resolution and frame rate. The
supported differences are position, occlusion exposure, handover role and installed-view
residual structure. E6 prevents attributing the last item uniquely to camera calibration.

**Supported.**

- Viewpoint-geometry effects: range, obliquity and projection amplification as a function of
  where the camera is.
- Occlusion-geometry effects: different blind regions for the north and south pairs arising
  from the deliberately asymmetric tall segments.
- Handover and overlap behaviour: source switching, cross-camera disagreement in overlap
  bands, and fallback on active-camera loss.
- Installed-view residual structure and its consequences: retired-v2 driving data show an 11×
  spread in signed residual magnitude (7.1 to 76.9 mm), correlated-error overconfidence, and different
  availability/precision selections on 15.7 % of the reachable floor. Camera-specific bias
  is a hypothesis until RQ15's grouped identifiability design separates it from route,
  region, yaw and robot silhouette. The current balanced-IPM A–D comparison instead spans
  only 64.6–68.1 mm in mean camera-measurement error.
- Mount-role transfer: whether a field fitted on three mounts predicts the fourth (split S4).

**Not supported.**

- Any optical archetype claim — wide versus narrow angle, long versus short focal length.
- Resolution or frame-rate heterogeneity. The repository contains 640 × 360 @ 5 Hz and
  640 × 360 @ 3 Hz variants of all four cameras, but they are uniform simulator-load
  downgrades, were never used as a designed heterogeneity axis, and changing resolution
  changes the detector's input scale, so per-camera use would confound A03 *and* A08.
- Vendor, sensor-technology or hardware-transfer claims.
- Any claim that the *number* four is sufficient, representative or optimal.
- Lighting or image-domain diversity (A07).

Anything in the second list requires new evidence, not reinterpretation of the current
captures. RQ09 stays a standing LIMITATION.

---

## 7. Route archetypes

Route archetypes are defined by what they discriminate, then bound to concrete tasks in
`src/experiments/config/tasks.yaml`. **Each binding must be certified against the frozen
fields before use** — a route is only an instance of its archetype if the measurement says so.

| ID | Archetype | Discriminates | Candidate binding | Certification measurement |
|---|---|---|---|---|
| R1 | Short poor-observation route versus a modest well-observed detour | Whether a source changes route choice at all | `mc_blind_L`, `mc_blind_R` (direct route up an aisle at fused reliability ≈ 0 over ~9 m; observed detour documented as ~27 % longer); `route_apron_to_a3_mid` in the single-camera world | Exposure ∫(1 − `p_use`) along both branches and the exact length ratio, on the frozen field |
| R2 | Equal-length routes differing only in occlusion | Occlusion sensitivity, with path length controlled | `mc_south_we` versus `mc_north_we` — a mirror pair, x −7.77 → +7.77 at y = ∓7.5 — exploiting the north/south tall-segment asymmetry | Verify lengths match by construction *and* that exposure genuinely differs; otherwise R2 is unfilled |
| R3 | Overlap and handover | Source switching, cross-camera disagreement, fallback | `mc_central_ns`; `south_to_north_handover`, `north_to_south_handover`, `central_overlap_sweep` from the commissioning study | Number of source switches, overlap-band disagreement against `max_cross_camera_disagreement_m`, and time-delta within `max_overlap_time_delta_s` |
| R4 | Unavoidable camera-poor negative control | Whether a method degrades gracefully when no better route exists | **Candidate, uncertified**: the west service lane against the wall-backed one-sided shelf, reachable only from its east side | Prove no alternative route to the same goal has materially better exposure — this is `U13` |
| R5 | Changed-layout route through the changed region | Staleness and transfer | **None exists** | Blocked by `U3` |
| R6 | Uniformly well-observed route | No-spurious-detour control: no method should change behaviour | `rob_easy` / the central corridor; `control_west_to_a1_low` is the single-camera analogue | min over the route of max_c `p_use,c` ≥ τ, for the declared τ |
| R7 | Long-horizon multi-leg (optional) | Accumulated drift, repeated handover, replanning | `mc_grand_tour`, `mc_aisle_3_to_6`, `mc_tour_L/C/S`, `rob_hardA`, `rob_hardB` | Report legs, handovers and total exposure separately from the outcome |

**A control that is easy to lose.** Tasks carrying `waypoints` prescribe the route and
therefore test *tracking*, not route choice. Only waypoint-free tasks — `rob_hardA_free`,
`rob_hardB_free`, `mc_blind_L/R`, the single-goal `mc_*` traverses — test the one-shot global
route decision. Any route-discrimination result must be reported from waypoint-free tasks;
waypointed variants are diagnostics.

---

## 8. Grouped splits

Random cell-level or frame-level splits leak (`03_assumptions.md` A19). Every held-out
evaluation uses grouped folds, and every fold reports its unique-site count, its camera
composition and its in-FOV / out-of-FOV balance.

| ID | Split | Group unit | What it supports | Status |
|---|---|---|---|---|
| S1 | Leave-one-route-out | Route / mission id | Generalization to an unvisited route | Required; currently **absent** — all four four-camera GP manifests record an empty `route_disjoint_validation_csv` |
| S2 | Spatial block | Contiguous floor blocks of at least 2 × the GP length scale (≥ 2.4 m), ideally rack-band × aisle-column | Generalization to unvisited floor | Required; the 6-fold validation that exists is not spatially blocked in a documented way |
| S3 | Changed-layout holdout | Layout state | Staleness and transfer | Blocked by `U3` |
| S4 | Leave-one-camera-out | Camera id | **Mount-role transfer only** — whether a field fitted on three mounts predicts the fourth. Never optical transfer | Required, and mandatory for covariance work: the anchored leave-one-out reference understates camera A by ~4.2× |
| S5 | Leave-one-yaw-out | Robot heading band | Guards against fitting at one heading and testing at another | Required. Two of the three existing bias captures are single straight lines at two fixed yaws, and the third repeats the first route (median nearest neighbour 0.006 m) |
| S6 | Second world | World | External validity | **Confounded for the older pair**: `warehouse_aws` and `warehouse_full_4cam` differ in mount height *and* detector, so a holdout across them would not isolate the world. `warehouse_v2` + `warehouse_v2_shipout` are matched by construction and do isolate it. |

A fold containing no out-of-FOV opportunities is invalid: it measures conditional detection
quality, not usable-observation probability. This is the failure mode of the inherited
single-camera dataset, which the README describes as in-FOV-only and quality-saturated. Its
quoted component rate is not evidence of record until the missing package is recovered.

---

## 9. Noise contract: primary versus sensitivity

**Rules, in force for both contracts.**

1. The primary comparison adds **no** synthetic noise. Nominal is the zero level of every
   factor below.
2. Sensitivities are one-factor-at-a-time. The Cartesian product is prohibited.
3. Noise is applied at the **shared measurement interface**, never inside one method's
   implementation, and never with per-method parameters.
4. Paired seeds and identical sites across arms; identical total sample budget.
5. Everything else in §2, §5 and the contract tables is held fixed: camera geometry,
   detector and threshold, robot, controller, planner weights, route start and goal,
   evaluation labels, seeds, budget.
6. Pilot data may refine the ladder **once**, before preregistration, on the basis of a
   non-degenerate pilot — never per method and never after seeing an arm's result.
7. The `R_plan` / EFE visibility cost is frozen method. It is never reweighted to change a
   navigation outcome; visibility-driven failures are reported as findings.

**Candidate OFAT ladder.** `FROZEN` levels are anchored to a measurement; `PILOT` levels need
a non-degenerate pilot or a supervisor decision before preregistration.

| Factor | Levels | Injection point | Status |
|---|---|---|---|
| Pixel-output jitter | 0, 0.5, 1.0, 2.0 px SD | Selected pixel, before projection | **PILOT** — the detector's measured edge error is ±0.34 px with sd 0.46–0.97 px, so 2.0 px is 2–4× anything observed and needs a stated rationale |
| Calibration yaw drift | 0, 0.1, 0.25, 0.5 ° | Camera extrinsic | **FROZEN** — anchored to the measured ladder: detection at 0.1°, harm at 0.25° |
| Calibration translation drift | 0, 0.025, 0.05, 0.10 m | Camera extrinsic | **PILOT** — no measured harm threshold exists yet |
| Message latency | 0, 50, 100, 200 ms | Measurement timestamp | **PILOT** — measured inter-sample p99 is ≤ 28.8 ms and the effective update interval is ≈ 450 ms, so the ladder may be mis-scaled relative to both |
| IID dropout | 0, 5, 15, 30 % | Per detection, per camera | **PILOT** |
| Burst dropout | 0.5, 1.0, 2.0 s | Per camera | **PILOT** — 0.5 s is barely one update at 2.2 Hz |
| Missing depth | 0, 10, 30 % of cells | Geometry rung, with the declared fallback | **PILOT**, contract B only; must be reported jointly with the fallback policy |
| Layout state | nominal, local change, global change | World | **BLOCKED** by `U3` |
| Commissioning budget | 0, 50, 100, 250, 500, 1000 unique sites per camera | Fitting stage | **PILOT**, contract B only; unique sites, not frames (A18). The current four-camera fit aggregates to 1101 points per camera, so 1000 is already near the available ceiling |

Injected calibration drift is a *fault* arm, not a nuisance arm: it belongs to the drift
lifecycle question and must never be enabled in the primary comparison of either contract.

---

## 10. Open items owned by this document

`U3` (layout-change definition; the two-world half is retired), `U11` (the 0.35 m
target height versus the robot's ≈0.20 m silhouette) and `U13` (whether an unavoidable
camera-poor region exists) originate here and are recorded with the rest in
[`03_assumptions.md`](03_assumptions.md) §8. Until they are resolved, R4 and R5 cannot be
bound, the layout ladder stays blocked, and the overlap, camera-poor and symmetry properties
in §1 stay UNMEASURED.
