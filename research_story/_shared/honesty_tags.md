# Evidence honesty tags

The vocabulary every chapter's `evidence.yaml` and every figure/slide uses to
state how strong a result is. Hoisted here (2026-07-21) from the retired
`docs/next_paper/evidence_ledger.md`, which `research_story/` supersedes as the
storyline of record; this file is now the single definition.

| Tag | Meaning |
|---|---|
| `established` | Existing, frozen or reproducible evidence supports a **narrow** claim. |
| `measured_in_sim` | Derived from logged simulator/detector data, but **not yet** a cross-layout or closed-loop paper result. |
| `model_plumbing` | First-principles, CAD-assisted, synthetic, or interface validation. Useful to choose experiments; **not** an empirical deployment claim. |
| `open` | Needs a new prospective experiment or evidence chain. |

## Presenting uncertainty well
Use the three-label legend on every meeting slide: **measured simulation**,
**model / plumbing**, **open experiment**. Stating this honestly makes the
research case stronger — the paper is exactly the work that turns a promising
chain into a validated deployment method.

## Hard firewall (restated)
Ground truth (`gt_*`, `eval_*`, oracle labels) and CAD/SDF geometry are
**evaluation-only** — they may score a result but can never train or feed an
operational/deployment model. See the leakage-firewall contracts in
[`../../docs/reliability_contracts/`](../../docs/reliability_contracts/).
