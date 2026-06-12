# Robustness campaign — run, monitor, rebuild outputs, update paper

Everything needed to reproduce the C1-vs-C2 robustness campaign and the paper
figures/numbers it feeds. All paths are relative to the `UnembodiedNavigation/`
repo root.

## 0. Before launching (important)
The one-shot global EFE solve is **single-threaded** and takes ~120–220 s wall
under CPU contention (offline it is ~15–25 s). It loses a race against the
first-command timeout when the machine is loaded. So:
- **Reboot and keep only VS Code open, idle.** This was the difference between
  the first 8 runs (all clean) and the later contended failures.
- Optionally prefix with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2` to limit
  thread-storm oversubscription on the 12-core box.

## 1. Run the campaign
```bash
./run_keepin_campaign.sh                 # -> logs/visibility_comparison/robustness_campaign_v2
# or name it:
./run_keepin_campaign.sh my_campaign_v3
```
- 40 runs: 4 tasks × {C1,C2} × 5 seeds (defined in `aws_f31b1_final_config.yaml`).
- Timeouts are the runner defaults — `--first-cmd-timeout 270`, `--run-timeout 420`
  — sized so a contended solve completes with room to execute. **Do not pass a
  shorter `--run-timeout`** (the old 240 s guillotined slow solves: the dominant
  past failure mode).
- `--resume` is on, so re-launching after an interruption only fills the gaps.

Locked config knobs that matter: `optimizer_maxiter=60`, multistart over 2
lane-graph route seeds (no midpoint/direct seed), `nogo_mode=keep_in`,
`nogo_weight=2000`, `use_belief_nogo_cost=true`, `risk_weight_obs=1.0`,
`ambiguity_weight=1.0` (both untuned), goal-prior anneal 50→12 px.

## 2. Monitor / triage (safe to run while it is still going)
```bash
python3 scripts/visibility_comparison/monitor_campaign.py logs/visibility_comparison/robustness_campaign_v2
```
Reports per run: plan produced?, chosen route vs the **offline** expected route,
first-command stamp, outcome, validity, belief error. Then headline C1-vs-C2
counts and an **ANOMALIES** block:
- **compute-fail** = ran but produced no plan/command. The plan itself is never
  wrong when produced (route-mismatch is the check) — a compute-fail is the
  solve losing the timeout race. **Triage:** 1–2 → just re-run with `--resume`;
  many → stop, reboot fresh, relaunch.
- **route-mismatch** = online route ≠ offline route. Should be 0 (verified on the
  prior campaign). If >0, the runtime is seeing a different start/belief than
  offline — investigate before trusting results.
- **invalid** = `valid_run=False` for a non-outcome reason.

The route-mismatch check needs `logs/paper_figures/offline_plan_sanity.json`
(produced by step 3 / `plot_offline_plan_sanity.py`).

## 3. Rebuild ALL paper outputs from the finished campaign
```bash
./scripts/visibility_comparison/build_paper_outputs.sh logs/visibility_comparison/robustness_campaign_v2
```
Runs, in order:
1. `compute_paper_metrics.py` → `paper_artifacts/metrics/robustness_metrics.csv`
   (per-run `is_clean_success / is_collision / is_near_success / is_interrupted /
   is_invalid` flags = the paper outcome categories).
2. `make_robustness_spread.py --campaign-root <root>` →
   `paper_artifacts/figures/robustness_spread.png` (+ `.provenance.json`).
3. `plot_solve_diagnostics.py` → `logs/paper_figures/solve_diagnostics.png`
   (offline convergence/timing/stability — the "is it numerically hard" evidence).
4. `plot_offline_plan_sanity.py` → `logs/paper_figures/offline_plan_sanity.png`
   (+ `.json`, the route reference used by the monitor).
5. prints the headline C1-vs-C2 counts.

The paired mechanism figure (Fig. `fig:route_choice`) is rebuilt separately when
its representative run changes:
```bash
python3 scripts/paper_figures/make_paired_mechanism.py   # see its header for the run it pins
```

## 4. Update the paper with the new numbers
See `PAPER_UPDATE_PLAN.md` (same directory) for the exact sentences/files that
take the new counts. In short: the abstract counts, `tab:results`, the Experiments
multi-task paragraph, the conclusion, and `robustness_spread.png` are the only
places the campaign numbers appear; everything else (mechanism, method, appendix
config) is number-independent.
