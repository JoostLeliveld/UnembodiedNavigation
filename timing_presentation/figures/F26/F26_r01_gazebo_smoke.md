# F26 - R01 Gazebo Smoke Diagnostic (STALLED — no runs completed)

Config: `scripts/visibility_comparison/aws_f26_r01_gazebo_smoke_config.yaml`

Changes vs F25: `nogo_safe_distance 0.13→0.30`, `local_optimizer_maxiter 60→25`.

## Outcome

**Global solve stalled — no runs completed, no figure generated.**

- `plan_samples.csv`: 0 rows (no plans ever produced)
- `experiment.csv`: 68 rows (robot telemetry logged while solver was running)
- Log duration before kill: ~133 s
- No `[hierarchical] global plan solved` message appeared in logs

## Root cause

`nogo_safe_distance=0.30 m` is too large for the AWS warehouse aisles under the
log-barrier formulation. With H=80 and multistart, the optimizer explored many
candidate routes but could not find a feasible global plan through passages that
are only ~1.6 m wide with 0.30 m forbidden-zone extensions on each side. The
solver ran indefinitely without converging.

Lesson: the correct safety margin increase is **incremental**. F25 crashed with
4 cm penetration; a jump from 0.13 to 0.30 (+0.17 m) overconstrains the aisles.
The next step is `nogo_safe_distance=0.20` (+0.07 m over F25), which adds a
meaningful buffer without narrowing the feasible region by 0.34 m total.

## Next step: F27

Config `aws_f27_r01_gazebo_smoke_config.yaml`:
- `nogo_safe_distance: 0.20` (from 0.13 in F25)
- `local_optimizer_maxiter: 25` (unchanged from F26 intent)

