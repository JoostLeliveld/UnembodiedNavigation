# Final Figure / Image Alignment Checklist (F86a v4)

Everything that must agree before F86a figures or paper claims are treated as
paper-ready. Use alongside `docs/F86_method_and_runtime_contract.md`,
`docs/CONSISTENCY_CHECKLIST.md`, and the root `CLAUDE.md` evidence standard.

## 0. Camera-view floor markings (paper figure)
- Floor overlay in the rendered camera view is **generated from the planner
  prisms** (`generate_driveable_overlay_sdf.py`), never hand-drawn.
- Colour semantics: **blue = outer driveable/workspace boundary**, **green =
  no-go region** = the complement of the driveable corridors (inter-aisle columns,
  corridor-to-corridor, split at the physical rack mid-gap so R2-R5 connectors stay
  driveable; R1 solid). Green is the no-go REGION, not the tight obstacle outline,
  and touches the driveable corridors. Never green for both.
- No mission-marker disks, red apron spots, black label stripes, pallet jack, or
  staging-clutter meshes in the camera view (start/goal live in the 2D route plot).
- Current GP = **`aws_gp_v7`** (locked camera z=4.8, y=−5.5; length_scale 0.90,
  noise_var 0.05, beta 0.5). Camera glyph in plots = (0, −5.5). GP pipeline figure =
  `gp_pipeline_aws_v7.pdf`. F87 rollout = `F87_offline_rollout_v7.png` (C1 NW-blind
  reaches; C2 south-visible through A1, Gate PASS). v6/v6b are stale (old camera).
- Alignment verified by `make_driveable_region_alignment.py` →
  `logs/paper_figures/driveable_region_alignment.{png,pdf}` (equal aspect, metric):
  blue boundary = prism-union bbox, green columns enclose the rack collision
  footprints.

## A. Geometry ↔ data chain
- [ ] World SDF geometry (F86a v4: continuous left shelf R1, R4 occluding crate
      stack, no monolith, no `low_crate_R4`) is the accepted geometry.
- [ ] `driveable_geometry_json` in `aws_f86a_camera_xy_config.yaml` matches the
      physical non-driveable layout. (Verified: R1 column x∈[-4.325,-3.775] and R4
      stack x∈[1.725,2.275] both lie between driveable prisms; no prism edit needed.)
- [ ] GP artifact `aws_gp_v6/yolo_score_raw_gp.npz` embeds `geometry_json` /
      `geometry_sha256` matching the current SDF collision prisms
      (CONSISTENCY_CHECKLIST C1 — must print OK).
- [ ] GP `camera_pos` matches the world-profile camera pose ±0.01 m
      (CONSISTENCY_CHECKLIST C2).
- [ ] Detector is `aws_yolo_simseg_v2` for both capture and runtime.

## B. Config ↔ artifact references
- [ ] `gp_artifact:` in every active config points at `aws_gp_v6` (not v5).
- [ ] `nogo_penalty_type: warning_band`, `nogo_weight: 2000`,
      `nogo_warning_band: 0.05`, `nogo_near_weight: 50` recorded in run config and
      logged by `experiment_logger`.
- [ ] Campaign config records the exact detector `.pt` and GP `.npz` paths.

## C. Figure scripts ↔ logs
- [ ] `make_f86_paper_figure.py` `CAMPAIGN`/`C1_DIR`/`C2_DIR` point at the **v4**
      run dirs (`logs/visibility_comparison/f86a_camera_xy_v4/...`), no stale
      `_archive_*` paths.
- [ ] `make_aws_problem_setup_figure.py` `DEFAULT_COV_RUN` / `DEFAULT_IMAGE` point
      at v4 logs and a v4-world camera frame.
- [ ] Map-panel overlays (shelf rectangles, driveable zones, camera marker) are
      drawn from the **same** geometry the planner used (parse from the config
      `driveable_geometry_json`, not a hand-copied list).
- [ ] GP background uses `P_conservative_plan_map` from `aws_gp_v6`.

## D. Convention consistency across figures
- [ ] Condition colours identical everywhere: **C1 = red (occluded route)**,
      **C2 = blue (observable route)**.
- [ ] Availability markers use `yolo_detected_after_threshold`, never
      `pixel_corr_accepted` (the NIS gate is disabled ⇒ that flag is ~always 1).
- [ ] Shared axis limits, GP colourbar range, and units across map panels.
- [ ] Timing hygiene: all means/curves start at `first_cmd_stamp`; goal success is
      `goal_region_success` not `completed`.

## E. Text ↔ figure ↔ logged metric
- [ ] Every numeric claim in the paper traces to a logged metric in a v4 run.
- [ ] Route-choice claim backed by the offline gate (C1→NW, C2→south) AND the
      closed-loop v4 logs.
- [ ] No claim of GP improving heading (GP affects camera (x,y) covariance only).

## F. Figures to regenerate after the world + GP change
- [ ] `problem_setup_camera.pdf` (panel a → Introduction).
- [ ] `problem_setup_snapshots.pdf` (panels b+c → Problem Statement).
- [ ] F86 main paper figure (`F86_paper_figure.{png,pdf}` + metrics `.md`).
- [ ] `make_f86_heading_compare.py` route-split + θ-variance audit.

---

## G. Suggested `.tex` edits (DO NOT auto-apply — user edits TeX)

The problem-setup figure is currently a single 3-panel PDF in
`thesis-report/sections/02_problem_statement.tex` (`fig:problem_setup`,
`figures/problem_setup_aws.pdf`). The plan splits it: panel (a) → Introduction,
panels (b)+(c) stay in Problem Statement. After regenerating the two PDFs
(`problem_setup_camera.pdf`, `problem_setup_snapshots.pdf`), apply these edits:

### `sections/01_introduction.tex` — add the camera-setup figure
```latex
\begin{figure}[tb]
    \centering
    \includegraphics[width=0.6\linewidth]{figures/problem_setup_camera.pdf}
    \caption{External-camera warehouse setup: a fixed external camera observes a
    mobile robot localized on the planar ground surface. Shelves and stacked
    crates occlude parts of the floor, reducing camera-based localization
    reliability in some regions.}
    \label{fig:setup_camera}
\end{figure}
```

### `sections/02_problem_statement.tex` — replace the existing figure block
Replace the current `fig:problem_setup` `\includegraphics{figures/problem_setup_aws.pdf}`
block with the two-panel snapshots figure (panel (a) moved to the introduction):
```latex
\begin{figure}[tb]
    \centering
    \includegraphics[width=\linewidth]{figures/problem_setup_snapshots.pdf}
    \caption{Two planning instants of a constant-covariance rollout. Panel~(a)
    shows the initial rollout; panel~(b) shows the robot near a region of reduced
    camera-update reliability. (The external-camera setup is shown in the
    introduction, Fig.~\ref{fig:setup_camera}.)}
    \label{fig:problem_setup}
\end{figure}
```
Note: re-letter the kept panels (b),(c) → (a),(b) in the snapshots figure caption,
matching whatever the regenerated `problem_setup_snapshots.pdf` labels them.
