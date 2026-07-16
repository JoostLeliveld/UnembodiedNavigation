# Media plan and shot list

The assets below already exist locally.  This plan uses them as evidence-aware
presentation material, rather than generating decorative figures that hide the
current validation gaps.  A path is relative to `docs/next_paper/`.

## Opening / meeting deck

| Slide | Message | Asset | Type and evidence label | Speaker note |
| --- | --- | --- | --- | --- |
| 1. Hook | “Reliable planning works — once the map exists.” | `../../paper_artifacts/figures/paired_mechanism_taskA.pdf` | Previous-paper context | Do not explain every planner term. Establish the map as a necessary input. |
| 2. Gap | “The map is costly and stale.” | `../../logs/geometry_visibility_prior/demo/hard_evidence.png` | Measured simulation | Use only the reliability/overconfidence takeaway; keep tiny diagnostics out of the talk. |
| 3. Day-zero question | “What can be known before the first drive?” | `../../logs/geometry_visibility_prior/demo/depth_occlusion_prior.png` | Simulated-sensor comparison | Say this motivates the sensing experiment; it does not prove real depth deployment. |
| 4. Proposed mechanism | “Prior mean plus prior strength.” | [storyline diagram](storyline.md#why-this-is-the-right-sequel) or the visual board | Proposed method | Explain that strength controls how readily driving corrects a cell. |
| 5. Learning loop | “Normal driving updates the map.” | `../../logs/geometry_visibility_prior/demo/gp_online_update.gif` | Measured simulation | Play 8–12 seconds; point out that misses are evidence too. |
| 6. Safety mechanism | “Uncertain pose spreads evidence; it is not silently thrown away.” | `../../logs/geometry_visibility_prior/demo/stereo_online_mechanism.png` | Mechanism illustration | Keep the claim narrow: hard gating is a negative control. |
| 7. Scale testbed | “The question persists at warehouse scale.” | `../../logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase/live_gazebo_views/overview/frame_000042.000.png` | Live simulator layout | Show the full facility and media-only overview camera; this is runtime evidence, not detector validation. |
| 8. Multi-view extension | “Four camera-specific priors cover the facility; shared regions make handover testable.” | `../assets/warehouse_full_4cam_map.png` and `../../logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase/overlap_handover_corridor.png` | Calibration-only model / layout | Label “day-zero coverage potential — not detector result.” |
| 9. Decisive experiment | “This is the minimal matrix that earns the claim.” | Conditions table from [storyline](storyline.md#minimal-defensible-experiment-matrix) | Proposed protocol | Emphasize the changed-layout and unvisited-cell split. |
| 10. Decision | “Commission first; fuse cameras later.” | `index.html` conclusion panel | Scope decision | Ask for approval to freeze the new campaign rather than more unconstrained extensions. |

## Recommended videos

### 1. Map learns from driving — primary clip

- Source: `../../logs/geometry_visibility_prior/demo/gp_online_update.gif`
- Length for talk: 8–12 seconds, loop once.
- On-screen caption: **Operational detections and misses refine a conservative
  day-zero map.**
- Evidence label: *measured simulator logs; map update not yet planner-integrated*.

### 2. Four-camera live views — contextual clip

- Source: `../../logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase/live_four_camera_montage.png`
- Length for talk: 10–15 seconds, with the live-stream labels visible.
- On-screen caption: **Four independent camera streams observe the same facility;
  overlap is where selection and conservative fusion must be validated.**
- Evidence label: *live simulator layout/runtime evidence; no learned reliability
  or fusion result claimed*.

### 3. Overlap corridor — optional appendix still

- Source: `../../logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase/overlap_handover_corridor.png`
- Length for talk: 6–10 seconds, with a verbal walkthrough.
- On-screen caption: **Coverage is not enough: a source switch must carry
  uncertainty.**
- Evidence label: *calibration-only layout demonstration; no per-camera
  reliability result claimed*.

## Visual rules

- Use one semantic colour system throughout: teal/green = well-supported
  reliable observation, amber = uncertain/needs data, magenta/red = unreliable
  or false-safe risk, navy = model/system plumbing.
- Put the evidence label in the lower-right corner of every non-paper figure.
- Keep reliability and **evidence confidence** visually distinct.  A high
  predicted reliability with weak support is precisely the cold-start risk.
- Show the known driveable/forbidden geometry separately from observation
  reliability.  The latter is never an obstacle map.
- For the final paper, regenerate every shown plot from a named config/run/artifact
  and write its provenance beside the figure.

## Local visual board

Open [index.html](index.html) in a browser from the repository.  It is a
16:9-style scrolling board with the exact narrative order above and embeds the
available local figures/videos without copying or mutating experimental output.
