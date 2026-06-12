# Paper update plan — dropping in the new campaign numbers

After `build_paper_outputs.sh` finishes, the campaign produces a fresh
`paper_artifacts/metrics/robustness_metrics.csv` and `robustness_spread.png`.
The numbers appear in only a few places; the mechanism/method/appendix are
number-independent. Paper repo: `thesis-report/` (separate git / Overleaf).

## Numbers to read off the metrics CSV
Aggregate per condition over the 4 tasks × 5 seeds (= 20 runs each):
- **clean successes** (`is_clean_success`), **collisions** (`is_collision`),
  **near-success** (`is_near_success`, the 0.40 m stop), **invalid/interrupted**
  (`is_invalid` / `is_interrupted`).
- **Per-task** counts, especially **b2** (how many seeds each condition reaches
  cleanly) — this carries the headline "largest separation on the hardest task".

`monitor_campaign.py <root>` prints the C1/C2 goal/collision tallies; the CSV
flags are authoritative for the paper categories (use the CSV, not the monitor's
quick tally, for the final sentence).

## Exact edit sites (current values from the prior 16/2 vs 12/8 campaign)

1. **Abstract** — `thesis-report/main.tex`
   Sentence: "C2 `16/20` clean and `2/20` collisions versus C1 `12/20` clean and
   `8/20` collisions." → replace all four counts.

2. **Experiments, multi-task paragraph** — `sections/05_experiments.tex`
   (`\subsection{Multi-task comparison}`). Update, in order:
   - "C2 reaches the goal cleanly in `16/20` runs, with one additional
     near-success stop and one run discarded as `infrastructure-invalid`,
     against `12/20` for C1, and collides in `2/20` runs against `8/20`."
     → all counts + adjust the "one near-success / one invalid" clause to the
       new `is_near_success` / `is_invalid` counts (may become zero on a clean
       fresh-machine run — then drop that clause).
   - "the long west-side approach (b2), where C1 commits to the camera-poor route
     and **fails on every seed** while C2 holds the observable route and **reaches
     four of five**." → recompute both from the per-task b2 counts. **Re-verify the
     qualitative claim**: only keep "fails on every seed" if C1/b2 clean successes
     = 0, and "reaches four of five" must match C2/b2 clean successes.

3. **Conclusion** — `sections/07_conclusion.tex`
   Mirror the headline counts; keep the association wording (no "drove into
   collision" causal phrasing — already fixed).

4. **Discussion** — `sections/06_discussion.tex`
   Aggregate-outcome paragraph: update any restated counts; the untuned-weights
   and numerically-tough-no-go points are number-independent and stay.

5. **Figure** — `figures/campaign/robustness_spread.png`
   Replace with the regenerated `paper_artifacts/figures/robustness_spread.png`
   (copy across). Caption is number-independent.

6. **Results table** (if `tab:results` is present in the experiments/appendix):
   regenerate its cells from the CSV columns; keep the metric set
   (clean / collision / near / det-rate / NLL / NEES / f_shadow) unchanged.

## Sanity gates before editing prose
- `monitor_campaign.py` shows **route-mismatch = 0** (plans match offline). If not,
  do NOT update the paper — investigate first.
- `is_invalid + is_interrupted` should be ~0 on a clean fresh-machine run. If a
  few remain, re-run those seeds with `--resume` rather than reporting them.
- If the new separation is *smaller* than 16/2 vs 12/8 (e.g. fewer C1 collisions
  on a cleaner machine), soften the abstract/discussion claims to match — the
  thesis axis is stability/observability, not a fixed collision count.

## What does NOT change with new numbers
Method (Sec. 3–4), the paired mechanism figure/story, the GP pipeline figure,
the appendix runtime-config table, the YOLO appendix, and all calibration numbers.
