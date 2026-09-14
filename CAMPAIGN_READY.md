# Final four-arm navigation campaign — ready to launch

Prepared 2026-09-15. Commit `6a757809`.

## Launch

    ./scripts/visibility_comparison/launch_final_v6_campaign.sh

It picks a fresh dated log root, refuses to start if a simulator is already
running, and passes through any extra flags (`--dry-run`, `--resume`). 80 runs,
about 4.5 h at the typical rate.

Nothing else needs editing first.

## What was wrong, and is now fixed

The first Gazebo pilot produced **zero commands for 270 s**. So did the three
runs of `three_uncertainty_navigation_campaign_v1` launched the same evening:
5 attempts, 0 usable, all `infra_invalid`.

`optimizer_control_block_steps = 2` averages each block of consecutive controls
and repeats the average. The route seeder emits alternating pure turns `[0, w]`
and pure drives `[v, 0]`; averaging a turn with a drive destroys both. The best
seed then misses the goal by more than the 0.35 m terminal tolerance, so no
candidate is admissible, selection falls through to raw cost, and a parked plan
wins — the clearance term is charged per step and a stationary robot drives past
nothing.

Set to 1, all four tasks reach the goal and solve faster. The value is now in the
enforced lock: a config with the old value emits a `RuntimeWarning`.

**`three_uncertainty_navigation_campaign_v1.yaml` has been corrected too** (a
backup of the original is in this session's scratchpad). Its runner had already
exited, so nothing was interrupted. Its 5 failed attempts should be discarded.

## Verified before launch

| check | result |
|---|---|
| four arms differ in exactly one argument | yes, `camera_network_artifact_path` |
| every artifact exists, hashes as declared | yes |
| world matches commissioning capture | byte-identical |
| Q in config / code default / lock doc | 0.02 / 0.08 in all three |
| lock warnings on the final config | none |
| `install/` vs `src/` | all packages symlink-installed, live |
| dry run | 80 runs, clean |
| disk | 12 GB projected, 35 GB free |

All four tasks solve to the goal offline: 0.000, 0.000, 0.000, 1.9e-15 m.

**$T_1$ splits by availability field.** $C_{00}$/$C_{01}$ take
`above_connector`; $C_{10}$/$C_{11}$ take `above_cross_aisle_2`; the groups
separate by 4.96 m. Cells sharing an availability field pick identical paths
whatever their covariance field, and both corridors are 32.65 m, so length
cannot account for the choice.

## Timeouts were breached and are now sized from measurement

`--first-cmd-timeout` was 270 s and `--run-timeout` 420 s, sized against a
"contended tail to ~220 s". Measured with `block_steps = 1`: 8 uncontended
solves, median 96 s, worst 238 s — and **304 s for one solve sharing the machine
with a live campaign**, which exceeds the old deadline. Slow solves were being
killed mid-optimization with no command and scored `infra_invalid`.

New defaults: `--first-cmd-timeout 480`, `--run-timeout 900`. Re-measure before
lowering either.

## Thesis text — proposed, not applied

Both files are in `papers/Thesis/`:

- `PROPOSED_04_additions.md` — the hard terminal constraint and per-step control
  resolution, neither currently stated in the experimental design.
- `PROPOSED_05_route_choice.md` — replaces the route-choice paragraph. The
  current 37.80 m / 34.10 m are not reproducible: they predate the planner repair
  and the task-set change, and no lock file records them. The measured result is
  stronger, because the two corridors are the same length.

`sections/05_results.tex` is already in finished voice with synthetic values
marked, so landing the campaign is a swap, not a rewrite. The abstract is still
empty.

## Known, not addressed

Three tasks report `optimizer_success: false`. That flag is the L-BFGS-B polish,
not the selected plan: the selected plan is a seed route that reaches the goal
and passes the hard gate. Read `selected_source` and
`terminal_goal_distance_pred` instead. You ruled investigating it out of scope.
