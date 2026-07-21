# Evidence classes

Every artifact, plot, table and result row carries an **evidence class** stating
what *kind* of evidence it is. This is orthogonal to the four
[`honesty_tags.md`](honesty_tags.md) (which state how *established/strong* a claim
is). Use both: a figure can be `REAL EXPERIMENT` + `measured_in_sim`, or
`MODEL ONLY` + `model_plumbing`.

Introduced 2026-07-21 as the reconciliation of the two-paper roadmap's Part I
governance layer onto this repo. See
[`../PROGRAMME_ROADMAP_2026-07-21.md`](../PROGRAMME_ROADMAP_2026-07-21.md).

## The eight classes

| Class | Meaning | Typical source |
|---|---|---|
| `CURRENT` | Valid evidence for the **active** system (current detector, GP artifact, runtime contract). | this-programme runs |
| `HISTORICAL` | Old-paper or deprecated-pipeline evidence; retired-world data. Cited, never a headline. | `_archive/`, submitted-paper figures |
| `REAL EXPERIMENT` | Actual detector + robot logs from a real (Gazebo) run. | `logs/…` captures/campaigns |
| `EVALUATION ONLY` | Ground truth used **only** for scoring (`gt_*`, oracle, CAD). Never trains/feeds an operational model. | Gazebo world pose, clearance, contacts |
| `CONTROLLED ABLATION` | Intentionally injected faults or synthetic uncertainty (e.g. WP3 Exp-A `Σ=α·Σ₀`; WP5/WP6 outage/drift injections). | fault/α sweeps |
| `MODEL ONLY` | Analytic / geometric / interface-validation prediction, not yet validated on real data. | geometry prior, projection round-trip |
| `DIAGNOSTIC` | Debugging evidence; not a claim. | `diag/`, throughput traces |
| `HYPOTHETICAL` | Unexecuted scenario (a layout/camera arrangement not run). | deck placeholders |

## The one rule

**Do not combine `CURRENT`, `HISTORICAL`, `CONTROLLED ABLATION` and `MODEL ONLY`
evidence in a single headline claim.** A headline is built from `CURRENT` +
`REAL EXPERIMENT` evidence, scored by `EVALUATION ONLY` ground truth. Everything
else is context, motivation, or ablation and must be labelled as such on the
slide/figure/row.

## Mapping onto honesty tags

| Evidence class | Compatible honesty tags |
|---|---|
| CURRENT + REAL EXPERIMENT | `measured_in_sim`, or `established` once frozen/reproducible |
| CONTROLLED ABLATION | `measured_in_sim` (labelled "controlled ablation, not operational evidence") |
| MODEL ONLY | `model_plumbing` |
| HYPOTHETICAL / open scenario | `open` |
| HISTORICAL | cite only; never upgrades a claim |

## Where classes are recorded

- Figures/slides: caption label (also `BELIEF` / `PIXEL` / `MODEL` / `GT — eval only`).
- `evidence.yaml` rows: alongside the honesty tag.
- Experiment registry rows (`docs/experiment_registry.md`): the evidence bundle
  should carry `ground_truth_access = evaluation_only` and the class of each metric.
