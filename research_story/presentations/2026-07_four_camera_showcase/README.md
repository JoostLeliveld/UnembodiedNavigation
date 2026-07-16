# Four-camera warehouse reliability showcase (pitch deck)

Pitch for the enlarged four-camera system and its reliability-fusion architecture —
chapter [08](../../08_large_warehouse_scaling/) / [09](../../09_multicamera_handover_fusion/)
material. Map statistics come from the checked-in day-zero artifact
(`paper_artifacts/gp/warehouse_full_4cam_dayzero_v1/`), not invented coverage numbers; the
final scope slide separates implemented components from empirical claims that still need a
four-camera campaign (per the ch.09 gate).

```bash
python3 build_four_camera_showcase.py
```

Inputs: `docs/assets/warehouse_full_4cam_map.png`, the day-zero npz + manifest, and captured
frames in `logs/.../four_camera_showcase/live_gazebo_views/`.
Outputs (regenerable, gitignored):
`logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase/` — deck
`Four_Camera_Warehouse_Reliability_Showcase.pptx` (18 slides) + 4 rendered maps
(reliability atlas, best-camera map, overlap/handover corridor, live montage). The maps are
linked into the chapter 08/09 `figures/` views.

Status: PITCH (deck labels planned work as such). Relocated here from
`midterm_presentation/` on 2026-07-15 — the midterm package is a frozen deliverable and
holds no active builders.

For the full initialization → collection → per-camera GP → overlap → combination
→ closed-loop story, see [FULL_STORY_VISUAL_PLAN.md](FULL_STORY_VISUAL_PLAN.md).
The builder includes the first two acts as current evidence and renders the
remaining acts as explicitly labelled protocol/evidence gates until the
four-camera campaign produces data.
