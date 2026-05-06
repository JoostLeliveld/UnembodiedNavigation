# UnembodiedNavigation — Claude Context

## What this project is

ROS 2 / Gazebo robotics thesis: Visibility-Aware EFE (Expected Free Energy) Navigation.
The robot uses active inference to plan paths that keep itself detectable by a fixed overhead camera,
using a GP-fitted observability map instead of constant observation noise.

## Build & run

```bash
# Source ROS 2 first (always required)
source /opt/ros/humble/setup.bash
source install/setup.bash   # after first build

# Build (use --symlink-install during dev so Python edits take effect without rebuild)
colcon build --symlink-install

# Build a single package
colcon build --symlink-install --packages-select experiments planning perception

# Run the primary campaign launch
ros2 launch experiments warehouse_primary_comparison.launch.py \
    planner:=visibility_aware_efe task:=shadow_tradeoff_a seed:=0
```

## Key directories

| Path | Purpose |
|------|---------|
| `src/experiments/` | Campaign orchestration, experiment logger, launch files |
| `src/planning/` | EFE planner, base_planner, CasADi optimizer |
| `src/perception/` | YOLO detector, GP inference node |
| `scripts/visibility_comparison/` | Offline scripts: campaign runner, GP verification, paper metrics |
| `src/experiments/data/visibility_gp/` | GP .npz artifacts |
| `logs/` | All run output (experiment.csv, run_summary.json per run) |

## IWAI campaign

36-run pre-registered experiment comparing 3 planner conditions across 3 tasks.

**Conditions:**
- C1 `constant_R_efe` — constant R₀, ignores GP
- C2 `visibility_aware_efe` — full GP-EFE with ambiguity term
- C3 `risk_only_ablation` — GP risk only, no ambiguity

**Tasks (all in `warehouse_occ_light.world.sdf`):**
- `shadow_tradeoff_a`: start (-2, 0.5) → goal (2, -0.5) — straight path through shadow
- `shadow_tradeoff_b`: start (-2, -1.0) → goal (2, -0.5) — diagonal through shadow
- `sanity_open`: start (-2, -1.5) → goal (2, -1.5) — fully visible, sanity check

**Run campaign:**
```bash
cd scripts/visibility_comparison
python3 run_iwai_campaign.py --config iwai_campaign_config.yaml
python3 run_iwai_campaign.py --config iwai_campaign_config.yaml --resume  # after interruption
python3 run_iwai_campaign.py --config iwai_campaign_config.yaml --dry-run  # preview only
```

**Compute paper metrics after campaign:**
```bash
python3 compute_paper_metrics.py \
    --campaign-log logs/visibility_comparison/iwai_campaign/campaign_log.json \
    --gp-artifact logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz \
    --out paper_metrics.csv
```

## GP artifacts — CRITICAL

The `.npz` files use these exact keys (NOT the names in old scripts):

| Key | Contents |
|-----|---------|
| `xs`, `ys` | Grid axes (shape (160,)) |
| `P_mean_map` | GP posterior mean, shape (160, 160) |
| `P_conservative_plan_map` | Conservative planning map ρ_plan (what the planner uses) |
| `X_train`, `p_train` | Training data |
| `camera_pos` | Camera (x, y, z) |

Legacy artifacts with `P_map` or `P_conservative_map` are not accepted in the paper runtime path.

**Artifact for IWAI campaign:**
`logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz`

**Camera pose (warehouse_occ_light):** `(-2.45, -2.45, 2.80)`, yaw 45°, pitch ~49° downward.

Note: this artifact shows near-zero visibility in the task region (P ≈ 0.001–0.05) because the `occ_light` world has heavy occlusion. This is expected — the planner still uses it for gradient guidance even at low absolute values.

## EFE objective and λ mapping

```
EFE = risk_scale × risk_term + ambiguity_scale × ambiguity_term + control_cost
risk_scale     = risk_weight_obs × observation_risk_scale  = 1.0 × 1.25 = 1.25
ambiguity_scale = ambiguity_weight × ambiguity_term_scale  = 3.0 × 1.00 = 3.0
```

These are set in `src/planning/planning/planners/base_planner.py` (lines ~711-712).
The paper's λ_risk and λ_amb map to these effective scales, not the raw parameters.

## Stopping logic

Three and only three valid completion reasons: `goal_reached`, `timeout_after_first_cmd`, `collision`.
- Timeout: 75 s after first nonzero command
- Goal: within 0.20 m radius, held for 2.0 s
- Collision: triggers `_finish_run("collision", stamp)` immediately (both contact and geometry)

## Key locked hyperparameters (campaign config)

```yaml
horizon: 40          dt: 0.25
observation_risk_scale: 1.25   ambiguity_term_scale: 1.0
optimizer_maxiter: 50          optimizer_maxfun: 300   # ~1.4 s/plan at T=40
goal_success_radius: 0.20      goal_success_hold_s: 2.0
run_timeout_after_first_cmd_s: 75.0
```

## YOLO model path

`/home/joostleliveld/Thesis/UnembodiedNavigation/logs/perception_models/yolo_simseg_smoke/model.pt`
(fine-tuned, 1 class: robot — NOT the COCO base `yolo11n-seg.pt` in the repo root)

## Common gotchas

- `auto_stop_on_goal` must be `'true'` (string) in launch args — was silently `false` before fix
- `risk_only_ablation` planner: `use_visibility_model=True`, `use_ambiguity=False`, `use_obs_risk=True`
- `dt` default in `visibility_launch_common.py` must come from `PAPER_LAUNCH_DEFAULTS`, not hardcoded `'0.2'`
- Collision callback `_contacts_cb` calls `_finish_run` after `_record_collision_event` — geometry collision check in `_log_once` calls it after writing the CSV row
