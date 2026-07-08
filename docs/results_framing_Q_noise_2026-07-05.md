# Mid-term results — canonical claims, run mapping, and mechanism
*2026-07-05. Single source of truth. Supersedes all earlier "collision / graze / crash" wording.*

## The one-line story
A learned, shared camera-reliability field lets the robot reach the goal more
reliably and stay in a regime where the safety filter can make progress. Without
observation the robot loses corrections, its belief uncertainty grows, and the
keep-in safety filter **safely halts it short** — it never crashes and never
leaves the driveable region.

## The three runs (which setting produced which result)
| | **Main** | **Robustness** | **Region diagnostic** |
|---|---|---|---|
| log dir | `honest_campaign_v1` | `whitenoise_campaign_v1` | `regiontest_v1` |
| config | `warehouse_visibility_campaign.yaml` (pinned `..._honest_v1.yaml`) | `..._whitenoise_v1.yaml` | `..._regiontest.yaml` |
| odometry noise | correlated + biased (AR(1) α≈0.8, slip_mean≠0) | white, zero-mean (α=0, slip_mean=0, std ~2×) | correlated + biased (= Main) |
| process noise Q (xy / θ) | **0.012 / 0.05** (fixed engineering estimate) | **0.010 / 0.08** (*derived* from injected variance) | 0.012 / 0.05 (= Main) |
| κ / mode | 1.0 / keep_in | 1.0 / keep_in | 1.0 / keep_in |
| graze terminates run? | yes (default) | yes (never fired) | **no** (`terminate_on_geom_collision:false`) |
| scope | 4 routes × 5 seeds × 2 cond = 40 | same, 40 | west route, C1, seeds 1–4 |
| result | C2 20/20 · C1 16/20 as-run | C2 20/20 · C1 19/20 | seed3 recovers → C1 west 2/5 |

Shared by all three: single overhead camera, heading dead-reckoned, one-shot
global plan, `use_belief_nogo_cost:true`, 5 seeds/cell. Q and κ were **never
changed** across any campaign (git-confirmed).

## Headline numbers
| | C2 (visibility-aware) | C1 (constant covariance) |
|---|---|---|
| **Main** (realistic noise, fixed Q) | 20/20 goal | **16/20** goal (17/20 under region-exit) |
| **Robustness** (white noise, derived Q) | 20/20 goal | 19/20 goal |
| region-exits (both) | 0 / 40 | 0 / 40 |
| physical contacts (both) | 0 / 40 | 0 / 40 |

Primary count is the full 40-run campaign as-run (**C1 16/20**). The **region-exit
refinement gives 17/20**: the 4 west-C1 runs, re-read from `regiontest_v1` (graze
non-terminal), have seed3 recover and reach the goal — which the old
graze-terminal metric wrongly failed. Report 16/20 as the headline and 17/20 as
the "under the safety-relevant region-exit criterion" refinement. Belief-vs-GT
accuracy is ~5 cm median / ~12 cm p95 in both campaigns, everywhere except C1 on
the west route (0.13 m mean).

## The claims you can make (each mapped to its evidence)
1. **Reliable goal-reaching.** C2 reaches the goal on every run in both noise
   regimes (20/20, 20/20); C1 fails only on the camera-poor west route; both are
   5/5 on the always-visible control route. *(Main + Robustness.)*
2. **Safety.** Neither method ever leaves the driveable region or makes physical
   contact: **0/40 region-exits, 0/40 contacts** in both campaigns. C1's failures
   are safe stalls ~2 m short of the goal, ≥30 cm from any wall. *(Main +
   Robustness + Region diagnostic.)*
3. **Mechanism (verified to the code).** Lose camera coverage → corrections stop →
   belief covariance grows (σ 0.007→0.21 trace) → the keep-in safety filter
   (`clearance − κ·σ_max`, κ=1.0) refuses to thread the narrow aisle gap it can no
   longer confirm is clear → the robot safely halts (83 logged
   `driveable_clearance_violation` stops). C2 stays observable → σ stays small →
   the same filter lets it through. *(Region diagnostic.)*
4. **Belief accuracy.** ~5 cm median / 12 cm p95 vs GT wherever the camera has
   coverage. *(Main + Robustness.)*
5. **Not a Q-tuning artifact.** Under a correctly-specified filter (white noise, Q
   derived from the injected variance, no free parameter) the effect persists
   (C2 20/20 vs C1 19/20); the gap narrows as the filter improves — the benefit is
   largest when localization is least ideal. Q is a fixed, shared, disclosed
   estimate (0.012, measured ~0.02), never fitted to GT. *(Robustness vs Main.)*

## Disclose these (makes it bulletproof)
- 5 seeds/cell → indicative, not statistically powered.
- The safe stall is a **safety–progress trade-off** in the keep-in gate (κ=2.0
  stalled *both* conditions; κ=1.0 is the setting used). "More caution" is not
  "better"; maintaining observability is what lets the robot progress safely.
- Earlier "sub-cm graze / crash" wording was the terminal-graze metric; corrected
  to 16–33 mm clearance grazes with **zero physical contact**, now reported as
  safe stalls under the region-exit criterion.

## Do NOT claim (no data)
- That it "would crash" without the gate or at low Q — untested; it could equally
  reach the goal (seed3 shows recovery is possible).
- That higher Q is "safer" — untested, and over-padding stalls everything.

## What "the keep-in gate" is (architecture note)
The local controller is a dumb geometric turn-then-go tracker (no obstacle
awareness). Its command is passed through a separate safety filter
(`_simple_plan_safe_to_execute`) that rolls it forward and vetoes it (publishes
0,0) if the **belief tube** (mean + κ·σ) would leave the keep-in region. The
controller stays dumb; the filter is what "knows to stop."
