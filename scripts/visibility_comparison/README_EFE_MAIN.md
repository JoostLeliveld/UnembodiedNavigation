# EFE Main Profile

The current thesis planner profile is deliberately named `efe_main`.
Older one-off labels such as preference-only, horizon-only, derisk-aggressive,
or slow-tightening were exploratory and should not be kept as permanent method
names.

`efe_main` uses:

- broad, slow-tightening goal preferences: `180 -> 45 px`, power `1.2`
- horizon `50`, `dt=0.2`, `discount_gamma=0.98`
- direct GP-to-trust mapping: `visibility_trust_mode=direct`
- visibility-derived likelihood precision: `r_visible_uv=2.5`, `r_miss_uv=100`
- no direct visibility reward/cost: `visibility_weight=0`
- old single warm-start optimizer: `optimizer_multistart_seeds=false`
- odom-anchored belief yaw for the YOLO position-only pipeline

Run:

```bash
python3 scripts/visibility_comparison/run_efe_precision_sweep.py \
  --profiles efe_main \
  --seeds 0 \
  --yolo-model logs/perception_models/yolo_simseg_smoke/model.pt \
  --visibility-artifact-path logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz
```

Use `efe_seeded` only as a diagnostic check for optimizer basin misses. Use
`mpc_diagnostic` only as an explicitly labeled baseline; it is not a final
classical MPC claim.
