# Archive payload snapshot — 2026-08-05

External payload root:

```text
/home/joostleliveld/Thesis/_archive/UnembodiedNavigation_paused_2026-08-05
```

| Bucket | Bytes | Files | Contents |
|---|---:|---:|---|
| `generated/` | 566,618,297 | 1,118 | Colcon logs, one-off launch log, generic checkpoint |
| `perception_datasets/` | 5,219,640,548 | 15,594 | Training, smoke, failed, diagnostic, and Meerhoven datasets |
| `perception_models/` | 48,146,785 | 69 | Superseded/diagnostic detector models |
| `studies/` | 158,116,143 | 316 | Paused geometry, Option A, observability, layout, and demo results |
| `visibility_comparison/` | 5,581,854,591 | 21,969 | Exploratory, retired-world, smoke, sweep, and showcase runs |
| **Total payload** | **11,574,381,369** | **39,066 payload files** | Moved, not deleted |

The active repository is **1,554,128,181 bytes** after relocation, down from roughly
13 GB before cleanup.

## Verification

- Same-filesystem moves completed without command errors.
- Canonical `honest_campaign_v1`, `whitenoise_campaign_v1`, `_paper_runs`, and current
  `spawn_grid_20260727` remain active.
- Current single-camera and four-camera detector models remain active.
- Core bias, belief-honesty, calibration-lifecycle, and closed-loop experiment paths
  remain active.
- Test suite after relocation: **891 passed, 2 failed, 3 warnings** — exactly the same two
  known detector-runtime contract failures as before relocation. The archive move caused
  no additional test failure.

## Restore pattern

Restore the archived directory to its original path, not to a new invented path. Example:

```text
archive/visibility_comparison/stack_capture2
→ logs/visibility_comparison/stack_capture2
```

Then re-run the relevant manifest/firewall/test gate before using the restored material.

