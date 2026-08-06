#!/usr/bin/env python3
"""Synchronize generated registry metadata blocks into active experiment READMEs."""

from __future__ import annotations

from pathlib import Path

import yaml

from validate_registry import REGISTRY, ROOT, load_registry


START = "<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->"
END = "<!-- RESEARCH-METADATA:END -->"
FIELDS = (
    "experiment_id", "status", "claim_ids", "assumption_ids",
    "reviewer_question_ids", "figure_ids", "dependencies", "operational_inputs",
    "evaluation_only_inputs", "primary_metric", "promotion_gate", "evidence_paths",
    "archive_rule", "next_action",
)


def block(experiment: dict) -> str:
    payload = {field: experiment[field] for field in FIELDS}
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip()
    return f"{START}\n\n```yaml\n{rendered}\n```\n\n{END}"


def sync(path: Path, generated: str) -> None:
    text = path.read_text(encoding="utf-8")
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        updated = before.rstrip() + "\n\n" + generated + after
    else:
        lines = text.splitlines()
        insertion = 1 if lines and lines[0].startswith("# ") else 0
        lines[insertion:insertion] = ["", generated, ""]
        updated = "\n".join(lines).rstrip() + "\n"
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    data = load_registry(REGISTRY)
    for experiment in data["experiments"]:
        readme = ROOT / experiment["study_path"] / "README.md"
        sync(readme, block(experiment))
        print(f"Synced {readme.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
