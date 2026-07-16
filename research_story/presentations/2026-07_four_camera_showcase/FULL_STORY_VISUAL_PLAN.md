# Four-camera warehouse — complete visual story

## Narrative spine

The presentation should make one causal claim, in order: **calibration gives a
safe starting hypothesis; uncertainty-aware driving turns it into four measured
camera-specific trust fields; overlap data decides when sources may be selected
or combined; the resulting observation stream improves the robot's belief
without hiding uncertainty.**

Never jump directly from geometric coverage to a fusion or navigation benefit.
Each transition below has a distinct visual and an evidence gate.

| Act | Question the audience has | Essential visualization | Evidence label now | What earns the next claim |
| --- | --- | --- | --- | --- |
| 1. Orientation | What changed in the new facility? | Top-down map, four live RGB streams, media-only overview | **LIVE SIMULATOR / LAYOUT** | No data gate: these are runtime assets. |
| 2. Initialization | What does each camera know before driving? | A–D calibration-only prior small multiples, union map, overlap-count map | **MODEL / DAY-ZERO PRIOR** | Explicitly show that no detector records or ground truth trained the maps. |
| 3. Evidence collection | How is a camera prior challenged honestly? | Route-family map; trajectory/pose-covariance overlay; per-frame record ribbon | **PROTOCOL / DATA PENDING** | Valid A–D routes, repeated passes, operational/evaluation split. |
| 4. Per-camera learning | What did each camera actually learn? | For A–D: observations + covariance ellipses, GP mean, GP standard deviation, held-out calibration panel | **MEASURED IN SIM** only after D0/D1 | Held-out NLL/MAE/calibration and false-high-trust rate per camera. |
| 5. Overlap commissioning | When can two cameras be trusted together? | Overlap graph; synchronized-pair timeline; disagreement scatter/heatmap | **MEASURED IN SIM** only after D2 | At least 30 held-out synchronized pairs per claimed edge, at most 10% outliers. |
| 6. Combination | Select, hand over, or fuse? | Best-source timeline; source-switch markers; covariance before/after; selection-vs-fusion comparison | **EVALUATED IN SIM** only after D3/D4 | Compare fixed, score-only, trust-only, hysteretic, and conservative-combination policies. |
| 7. End-to-end result | Does the combined system help the robot? | Same-seed route montage; belief error/covariance timeline; success, breach, and recovery table | **CLOSED-LOOP EVIDENCE** only after D5/D6 | Shadow replay first, then opt-in live correction with matched baselines. |

## Required visual package

### Presentable immediately

1. Large-facility map with A–D camera positions and operating/no-go geometry.
2. Four labeled live Gazebo views plus the clearly excluded top-down media view.
3. Calibration-only A–D prior atlas, best-source map, and geometric overlap map.
4. System architecture: isolated camera streams → camera-specific priors →
   selection/consistency gate → covariance-aware belief correction.

### Generate while commissioning

1. **Route and coverage figure.** Plot each executed path by camera/route family,
   with repeated passes distinguished by faint traces and estimated-pose covariance
   ellipses. Show planned routes separately from executed routes.
2. **Per-camera evidence cards.** One card for each A–D field: record count,
   detections/misses, spatial coverage, freshness, and held-out split. Do not
   replace a missing card with a copied camera result.
3. **Four GP small multiples.** For every camera, pair posterior mean with
   posterior standard deviation on the identical floor map. Mark unvisited cells
   so spatial interpolation cannot look like data.
4. **Overlap gate figure.** Show only adjacent physical pairs that were actually
   synchronized. Report pair count, time offset, median/p90 disagreement,
   outlier rate, and a spatial disagreement heatmap.
5. **Decision trace.** Align selected source, trust, detector quality, age,
   pair-consistency result, and measurement covariance with the robot trajectory.
6. **Closed-loop matched comparison.** For matched seeds, show route, belief
   error, uncertainty, source changes, goals/breaches, and recovery after a
   controlled camera degradation. Report the whole distribution, not a highlight
   run.

## Talk sequence

1. “Here is the large warehouse and its four real streams.”
2. “Before driving, each view supplies a cautious, calibration-only hypothesis.”
3. “The robot drives routes that expose each hypothesis to detections *and*
   misses, while preserving uncertainty about where those observations occurred.”
4. “That evidence fits four separate GPs: a camera's shadow does not become
   another camera's shadow.”
5. “Overlap is a test bench: only compatible synchronized observations authorize
   a handover or conservative combination.”
6. “The system sends one source-aware observation and an honest covariance to
   the unchanged belief/planner interface.”
7. “The final claim is operational: matched closed-loop runs show whether this
   buys continuity and safety.”

## Current boundary

The first two acts are available today, including 99.2% calibration-only union
coverage and 42.2% geometric multi-camera overlap. Acts 3–7 are deliberately
visualized as a protocol and evidence checklist in the deck until full-world
routes and detector records exist. This prevents the presentation from
misrepresenting day-zero geometry as learned GP, fusion, or navigation results.
