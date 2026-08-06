#!/usr/bin/env python3
"""Promote decision figures out of ignored run output into the tracked `figures/` tree.

`logs/studies/` is gitignored, so every plot the project actually decides from was
invisible to git and one `rm -rf logs/` from gone. This copies them to
`figures/<EXPERIMENT-ID>/`, keyed by `registry.yaml`'s `study_path`, so the control plane
and the figure tree cannot drift apart, and writes a SHA-256 provenance sidecar per figure
recording where it came from.

Raw output stays in `logs/studies/`. This is a promotion step, not a move: rerunning a study
and re-promoting is the intended workflow.

    python3 scripts/research/promote_figures.py --dry-run
    python3 scripts/research/promote_figures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
import sys

import yaml

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
STUDIES = REPO / "logs" / "studies"
FIGURES = REPO / "figures"

#: Bulk image dumps that live beside real figures but are datasets/renders, not plots.
#: Matched against whole path components only -- substring matching would swallow real
#: studies such as `bayesian_filter_showcase`.
EXCLUDED_DIRS = frozenset({"images", "labels", "four_camera_showcase", "demos"})
EXCLUDED_PREFIXES = ("detector_dataset",)


def experiment_by_study() -> dict[str, str]:
    registry = yaml.safe_load((REPO / "research" / "registry.yaml").read_text())
    out = {}
    for entry in registry.get("experiments", []):
        study = entry.get("study_path")
        if study:
            out[Path(study).name] = entry["experiment_id"]
    return out


def is_figure(path: Path) -> bool:
    if path.suffix.lower() not in (".png", ".pdf"):
        return False
    parts = [p.lower() for p in path.parts]
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    return not any(part.startswith(EXCLUDED_PREFIXES) for part in parts)


def collect() -> dict[str, list[Path]]:
    """study name -> its figure files."""
    found: dict[str, list[Path]] = {}
    if not STUDIES.is_dir():
        return found
    for study_dir in sorted(p for p in STUDIES.iterdir() if p.is_dir()):
        figures = sorted(p for p in study_dir.rglob("*") if p.is_file() and is_figure(p))
        if figures:
            found[study_dir.name] = figures
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(dry_run: bool) -> dict[str, dict]:
    mapping = experiment_by_study()
    collected = collect()
    report: dict[str, dict] = {}

    for study, figures in collected.items():
        experiment = mapping.get(study)
        target_dir = FIGURES / experiment if experiment else FIGURES / "_unmapped" / study
        entries = []
        for source in figures:
            relative = source.relative_to(STUDIES)
            # Flatten study/exp/fig.png -> exp__fig.png so one directory stays browsable.
            flat = "__".join(relative.parts[1:]) if len(relative.parts) > 1 else relative.name
            destination = target_dir / flat
            entries.append({"source": str(source.relative_to(REPO)), "destination": destination})
            if dry_run:
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            (destination.with_suffix(destination.suffix + ".provenance.json")).write_text(
                json.dumps(
                    {
                        "promoted": date.today().isoformat(),
                        "experiment_id": experiment,
                        "study": study,
                        "source_path": str(source.relative_to(REPO)),
                        "source_sha256": sha256(source),
                        "note": "Promoted from ignored run output; regenerate by rerunning the study.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        report[study] = {"experiment_id": experiment, "count": len(entries), "entries": entries}

    missing = sorted(set(mapping.values()) - {r["experiment_id"] for r in report.values()})
    report["_experiments_without_figures"] = {"experiment_id": None, "count": 0, "ids": missing}
    return report


def write_index(report: dict[str, dict]) -> None:
    lines = [
        "# Figures",
        "",
        "Every plot this project makes decisions from. Grouped by `registry.yaml` experiment ID.",
        "",
        "Raw output stays in the ignored `logs/studies/`; this tree is the tracked, browsable",
        "copy. Each figure carries a `.provenance.json` naming its source and SHA-256.",
        "",
        "Regenerate with `python3 scripts/research/promote_figures.py`.",
        "",
        "| Experiment | Study | Figures |",
        "|---|---|---|",
    ]
    for study, info in sorted(report.items()):
        if study.startswith("_"):
            continue
        experiment = info["experiment_id"] or "_unmapped_"
        lines.append(f"| {experiment} | `{study}` | {info['count']} |")

    missing = report["_experiments_without_figures"]["ids"]
    lines += ["", "## Experiments with no figures", ""]
    lines += [f"- `{m}`" for m in missing] if missing else ["- none"]
    lines.append("")
    (FIGURES / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = promote(args.dry_run)
    total = sum(info["count"] for key, info in report.items() if not key.startswith("_"))
    for study, info in sorted(report.items()):
        if study.startswith("_"):
            continue
        label = info["experiment_id"] or "UNMAPPED"
        print(f"  {label:22} {study:38} {info['count']:3} figures")
    print(f"\n{total} figures" + (" (dry run, nothing written)" if args.dry_run else " promoted"))

    missing = report["_experiments_without_figures"]["ids"]
    if missing:
        print(f"\nexperiments with NO figures: {', '.join(missing)}")
    if not args.dry_run:
        write_index(report)
        print(f"-> {(FIGURES / 'README.md').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
