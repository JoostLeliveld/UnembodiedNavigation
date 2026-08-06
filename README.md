# Unembodied Navigation

Research platform for observation-quality-aware navigation using fixed infrastructure
cameras.

## Current research dashboard

The scientific source of truth is [`research/`](research/README.md). The current paper asks
whether persistent camera-specific correlated error makes a conventional belief
overconfident, and whether honest uncertainty improves closed-loop navigation. Its only
active scientific gate is the matched 30-run campaign.

The broader constant/FOV/depth/GP/hybrid/DL comparison is the next thesis chapter. Camera
selection and fusion remain a separate downstream study.

- Human dashboard: [`research/README.md`](research/README.md)
- Generated status: [`research/STATUS.md`](research/STATUS.md)
- Canonical registry: [`research/registry.yaml`](research/registry.yaml)
- Current paper scope: [`research/papers/correlated_error_icra.md`](research/papers/correlated_error_icra.md)

## Build and validate

```bash
colcon build
source install/setup.bash
python3 scripts/research/validate_registry.py
python3 -m pytest -q
```

Campaign entry points and frozen artifacts are declared by the active experiment entry in
`research/registry.yaml`. Active logs belong under ignored `logs/`; curated evidence and
figures belong in `paper_artifacts/`.

## Directory map

| Path | Role |
|---|---|
| [`research/`](research/README.md) | Questions, claims, assumptions, status, paper scopes, and evidence registry. |
| [`src/`](src/) | Reusable ROS and scientific runtime packages. |
| [`experiments/`](experiments/) | Active studies only; every admitted study maps to registry claims and gates. |
| [`scripts/`](scripts/README.md) | Shared campaign, analysis, visualization, and hygiene tools. |
| [`config/`](config/) / [`schemas/`](schemas/) | Shared configuration and data contracts. |
| [`tests/`](tests/) | Unit, integration, contract, and registry tests. |
| [`docs/`](docs/README.md) | Technical/runtime documentation only. |
| [`paper_artifacts/`](paper_artifacts/README.md) | Locked summaries, provenance, and canonical figures. |
| `logs/` | Ignored active data and reproducible output. |

## Cold archive and recovery

Tracked history before consolidation is tagged
`pre-research-consolidation-2026-08-06`. Inactive raw data and former sibling repositories
are indexed through [`research/09_decisions_and_risks.md`](research/09_decisions_and_risks.md)
and stored outside the live workspace under `/home/joostleliveld/Thesis_Cold_Archive/`.
