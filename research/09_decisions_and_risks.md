# Decisions, risks, and archive index

## Locked decisions

- `UnembodiedNavigation` is the only live research repository.
- The current publication is the correlated-error and belief-honesty paper.
- The reliability-source comparison is the next thesis chapter.
- Camera management follows source estimation and uses frozen reliability fields.
- Evidence is Gazebo-only, one robot, and 2-D position; no hardware claim is made.
- Null results, summaries, manifests, provenance, and decisive figures are permanent.
- Generated data is disposable; irreplaceable raw data moves only after hash verification.

## Recovery

The complete pre-consolidation tracked state is available at Git tag
`pre-research-consolidation-2026-08-06`. Removed tracked trees should be recovered from that
tag, not replaced with pointer READMEs.

Cold-storage manifests are stored both beside the archived content and under
`/home/joostleliveld/Thesis_Cold_Archive/manifests/`. Each records source, destination,
size, file count, SHA-256 manifest, reason, evidence references, date, and restore command.

## Principal risks

| Risk | Control | Residual limitation |
|---|---|---|
| Cleanup destroys current work | Four coherent commits plus recovery tag precede removal. | Untracked external files require separate manifests. |
| Research state forks again | Registry is machine authority; STATUS is generated. | Narrative prose can still become stale and must not carry status. |
| Oracle leakage | Operational/evaluation input split is validated. | Runtime code review remains necessary. |
| Campaign sprawl | One active experiment and offline promotion gates. | Exploratory work must remain outside active studies. |
| Cold archive becomes a junk drawer | Hash manifests and one archive index. | Storage health/backups are outside repository control. |
| Four cameras are overgeneralized | Report geometric and bias diversity only. | Optical/hardware diversity is not tested. |
| Better prediction does not help navigation | Treat C4 as open and preserve a null. | A null narrows the contribution to representation/estimation. |

## Artifact hashes

Detector, calibration, GP, and campaign hashes are canonical entries in `registry.yaml` and
are checked by the registry/hygiene tooling before campaigns and archive moves.
