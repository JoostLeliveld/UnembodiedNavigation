# Paper Evidence Summary — AWS benchmark (2026-06-03)

Honest, data-backed summary of what the final AWS campaigns support. Source data:
`logs/visibility_comparison/{paper_final_v1, paper_b5_v1, paper_generality_v1}`
(merged in `paper_combined_v1/`). Conditions: **C1** constant-observability EFE,
**C2** learned-observability (GP) EFE, **C3** GP covariance with ambiguity OFF
(ablation). Method locked in `docs/paper_runtime_contract.yaml` v0.4 +
`scripts/visibility_comparison/aws_paper_final_config.yaml`. 57 runs total.

## Headline result — two discriminating tasks (5 seeds each, all 100% clean)

| Task | Cond | route | path (m) | mean loc err (m) | shadow frac | det rate |
|---|---|---|---|---|---|---|
| F31_b1 | C1 constant | occluded | 4.64±0.12 | **0.22±0.02** | 0.19 | 0.81 |
| F31_b1 | C2 learned  | visible  | 6.81±0.10 | **0.07±0.01** | 0.00 | 1.00 |
| F31_b1 | C3 GP, amb-off | visible | 6.89±0.14 | **0.07±0.01** | 0.00 | 0.99 |
| b5     | C1 constant | occluded | 6.56±0.07 | **0.19±0.01** | 0.26 | 0.77 |
| b5     | C2 learned  | visible  | 8.54±0.11 | **0.09±0.01** | 0.00 | 1.00 |
| b5     | C3 GP, amb-off | visible | 8.52±0.06 | **0.09±0.01** | 0.00 | 1.00 |

**Supported claims:**
1. On two independent route-choice tasks with severe occlusion and a clear visible
   alternative, the learned-observability planner (C2) chooses the longer **visible**
   route while the constant-observability baseline (C1) takes the shorter **occluded**
   route. Robust across 5 seeds (no seed variance in route).
2. The visible route yields ~2–3× better localization (F31_b1: 0.07 vs 0.22 m;
   b5: 0.09 vs 0.19 m) and full detection (100% vs 77–81%); C1's belief error peaks
   at ~2.4 m in the occluded gap (single-run trace), i.e. near localization loss.
3. **Ablation (C3 ≡ C2):** turning the ambiguity term OFF does not change the route
   or localization — so the route choice is driven by the **GP-conditioned
   belief covariance feeding the belief-space (belief-nogo) feasibility term**, NOT
   by the information/ambiguity term. (Mechanism trace: along the occluded route the
   per-step belief-nogo posterior covariance is ~4× larger for C2/C3 than C1,
   0.165 vs 0.042 m major axis, tracking the p_vis dropoff —
   `logs/diagnostics/rollout_cov/cov_trace_mid_cross_lane.png`.)

## Control — no spurious behavior

`visible_aisle_sanity_aws` (visible→visible): C1 and C2 identical (7.8 m, err 0.10,
100% clean). The method does not alter behavior when there is no occlusion to react to.

## Honest generality / limitations (do NOT overclaim)

| Task | Result | Interpretation |
|---|---|---|
| b3 long-horizon | C1≡C2 both visible, 100% clean | No occluded shortcut tempted → no contrast (null) |
| b2 west-mirror | C1 33% / C2 67% clean (both crash some seeds) | Mild occlusion (p_vis~0.20) + narrow aisle → weak/noisy benefit |
| b4 occluded goal | C1/C2/C3 all commit to the occluded goal | **Safe-stop NOT supported** — planner has no explicit localization-safety abort |

**Scope the claim:** the advantage requires the discriminating regime — *severe*
occlusion AND a *clearly-safer visible alternative*. Where occlusion is mild (b2) or
no occluded shortcut exists (b3), there is no clean benefit. Where the goal itself is
unavoidably occluded (b4), the method commits rather than stopping — "stop safely when
localization-unsafe" is **future work** (needs an explicit predicted-covariance abort).

## Disclosures (per supervisor concerns)
- Global EFE solve is slow (~50–80 s, one-shot route planning) — scalability/solver
  limitation; route seeds (known warehouse aisles, condition-neutral) are needed for
  tractability, not to bias the choice.
- The binary route-flip can be optimizer-basin sensitive in narrow aisles (local minima).
- Heading from odometry; projection/calibration residual not propagated into the EKF.
- C1 retains the ambiguity term (it is not risk-only); only the GP covariance differs.

## Figures (staging: `logs/diagnostics/paper_figures_staging/`; b5 under `/b5`)
- `paired_mechanism_taskA.pdf` — headline: C1 occluded vs C2 visible over ρ_plan,
  C1 error spike to 2.2 m vs C2 flat, EFE term time-series.
- `compare_taskA.pdf` / `paths_per_seed.pdf` — 5-seed path overlays.
- `gp_pipeline.pdf` — scores → ρ_plan → ambiguity.
- `cov_trace_mid_cross_lane.png` — mechanism (covariance inflation vs p_vis).
- NOTE: script caption hardcodes "shadow-tradeoff task" (occ_light-era) — fix label
  before paper use. Figures are STAGED; `thesis-report/figures` not modified.
