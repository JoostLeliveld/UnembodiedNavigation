#!/usr/bin/env python3
"""Validate the canonical research registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "research" / "registry.yaml"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("registry root must be a mapping")
    return data


def _ids(rows: list[dict[str, Any]], kind: str, errors: list[str], key: str = "id") -> set[str]:
    values = [str(row.get(key, "")) for row in rows]
    missing = [index for index, value in enumerate(values) if not value]
    if missing:
        errors.append(f"{kind}: missing {key} at indexes {missing}")
    duplicates = sorted({value for value in values if value and values.count(value) > 1})
    if duplicates:
        errors.append(f"{kind}: duplicate IDs: {duplicates}")
    return set(values) - {""}


def validate(data: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    allowed = set(data.get("allowed_statuses", []))
    if allowed != {"PLANNED", "READY", "ACTIVE", "BLOCKED", "COMPLETE", "LOCKED", "RETIRED"}:
        errors.append("allowed_statuses must contain exactly the seven canonical states")

    collections = {
        "questions": data.get("questions", []),
        "claims": data.get("claims", []),
        "assumptions": data.get("assumptions", []),
        "reviewer_questions": data.get("reviewer_questions", []),
        "figures": data.get("figures", []),
        "code_assets": data.get("code_assets", []),
        "maintenance_tasks": data.get("maintenance_tasks", []),
        "artifacts": data.get("artifacts", []),
    }
    known = {name: _ids(rows, name, errors) for name, rows in collections.items()}
    experiment_ids = _ids(data.get("experiments", []), "experiments", errors, "experiment_id")

    domain_statuses = {
        "assumptions": {"ACCEPTED", "TESTED", "DEFERRED"},
        "reviewer_questions": {"OPEN", "ANSWERED", "LIMITATION"},
    }
    for name, rows in collections.items():
        for row in rows:
            status = row.get("status")
            valid_statuses = domain_statuses.get(name, allowed)
            if status is not None and status not in valid_statuses:
                errors.append(f"{name}/{row.get('id')}: invalid status {status!r}")

    for claim in data.get("claims", []):
        if claim.get("question_id") not in known["questions"]:
            errors.append(f"claim/{claim.get('id')}: dangling question_id {claim.get('question_id')}")

    active_research = [row["experiment_id"] for row in data.get("experiments", []) if row.get("status") == "ACTIVE"]
    active_maintenance = [row["id"] for row in data.get("maintenance_tasks", []) if row.get("status") == "ACTIVE"]
    if len(active_research) > 1:
        errors.append(f"at most one ACTIVE research experiment is allowed: {active_research}")
    if len(active_maintenance) > 1:
        errors.append(f"at most one ACTIVE maintenance task is allowed: {active_maintenance}")

    focus = data.get("current_focus", {})
    if focus.get("research_experiment_id") not in experiment_ids:
        errors.append("exactly one valid current research focus must be declared")
    elif active_research != [focus.get("research_experiment_id")]:
        errors.append("current research focus must be the sole ACTIVE experiment")
    if focus.get("maintenance_task_id") not in known["maintenance_tasks"]:
        errors.append("current maintenance focus must reference a maintenance task")

    retired_assets = {
        row["id"] for row in data.get("code_assets", []) if row.get("status") == "RETIRED"
    }
    required_metadata = {
        "experiment_id", "status", "claim_ids", "assumption_ids",
        "reviewer_question_ids", "figure_ids", "dependencies",
        "operational_inputs", "evaluation_only_inputs", "primary_metric",
        "promotion_gate", "evidence_paths", "archive_rule", "next_action", "study_path",
    }
    refs = {
        "claim_ids": known["claims"],
        "assumption_ids": known["assumptions"],
        "reviewer_question_ids": known["reviewer_questions"],
        "figure_ids": known["figures"],
        "dependencies": known["code_assets"],
    }
    for experiment in data.get("experiments", []):
        eid = experiment.get("experiment_id")
        missing = sorted(required_metadata - set(experiment))
        if missing:
            errors.append(f"experiment/{eid}: missing metadata {missing}")
        if experiment.get("status") not in allowed:
            errors.append(f"experiment/{eid}: invalid status {experiment.get('status')!r}")
        for field, valid_ids in refs.items():
            dangling = sorted(set(experiment.get(field, [])) - valid_ids)
            if dangling:
                errors.append(f"experiment/{eid}: dangling {field}: {dangling}")
        if experiment.get("status") == "ACTIVE" and retired_assets.intersection(experiment.get("dependencies", [])):
            errors.append(f"experiment/{eid}: ACTIVE experiment depends on retired code")
        study_path = experiment.get("study_path")
        if study_path and not (root / study_path / "README.md").exists():
            errors.append(f"experiment/{eid}: missing study README {study_path}/README.md")
        operational = set(experiment.get("operational_interface_inputs", experiment.get("operational_inputs", [])))
        evaluation = set(experiment.get("evaluation_only_inputs", []))
        leaked = sorted(operational & evaluation)
        if leaked:
            errors.append(f"experiment/{eid}: evaluation-only inputs leak into operational interface: {leaked}")
        for evidence in experiment.get("evidence_paths", []):
            if not (root / evidence).exists():
                errors.append(f"experiment/{eid}: missing evidence path {evidence}")
        if experiment.get("status") == "COMPLETE":
            if not experiment.get("result_summary") or not experiment.get("provenance"):
                errors.append(f"experiment/{eid}: COMPLETE requires result_summary and provenance")
            for field in ("result_summary", "provenance"):
                path = experiment.get(field)
                if path and not (root / path).exists():
                    errors.append(f"experiment/{eid}: missing {field} path {path}")

    for artifact in data.get("artifacts", []):
        path = artifact.get("path")
        if not path or not (root / path).exists():
            errors.append(f"artifact/{artifact.get('id')}: missing path {path}")
        digest = str(artifact.get("sha256", ""))
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            errors.append(f"artifact/{artifact.get('id')}: invalid SHA-256")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    data = load_registry(args.registry)
    errors = validate(data, ROOT)
    if errors:
        print("Registry validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Registry valid: {args.registry.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
