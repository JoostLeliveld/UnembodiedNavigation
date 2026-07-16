# Four-camera warehouse — end-to-end walkthrough

This folder is the presentation-ready version of the complete story. Open the
numbered folders in order. Each part contains:

- `figures/` — only current world assets or explicitly labelled protocol
  templates;
- `TALKING_POINTS.md` — the message, evidence boundary, and hand-off to the
  next part.

| Part | Story beat | Evidence state |
| --- | --- | --- |
| [01](01_facility_and_live_streams/) | The physical upgrade: warehouse, A–D streams, overview | live simulator / layout |
| [02](02_day_zero_initialization/) | Four cautious calibration-only initial priors | day-zero model |
| [03](03_uncertainty_aware_collection/) | How driving produces honest records | protocol + executed pilot |
| [04](04_per_camera_gp_learning/) | How records earn four separate learned GPs | protocol + expected-kernel pilot fits |
| [05](05_overlap_selection_and_combination/) | When sources may be selected, handed over, or combined | geometry + C/D pilot gate |
| [06](06_closed_loop_evaluation/) | How to prove operational benefit on matched routes | protocol + shadow-replay safety result |
| [07](07_real_commissioning_execution/) | The real collection → GP → overlap → policy evidence | executed pilot; no closed-loop claim |

Regenerate the protocol diagrams after changing their narrative with:

```bash
python3 build_walkthrough_assets.py
```

The current live and day-zero figures are symlinks to the regenerated showcase
assets in `logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase/`.
They will refresh whenever the main showcase builder is re-run.
