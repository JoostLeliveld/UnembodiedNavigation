#!/usr/bin/env python3
"""Report repository hygiene risks that accumulate around research campaigns."""

from __future__ import annotations

import hashlib
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

from validate_registry import REGISTRY, ROOT, load_registry, validate


APPROVED_GENERATED_ROOTS = {"build", "install", "log", "logs"}
RETIRED_PATH_NAMES = ("modules/", "research_story/", "paused_archive/")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    data = load_registry(REGISTRY)
    findings = [f"ERROR: {item}" for item in validate(data, ROOT)]
    registered = {
        row["experiment_id"]: row for row in data.get("experiments", [])
    }
    registered_evidence = {Path(row["study_path"]).name for row in registered.values()}
    for directory in sorted((ROOT / "experiments").iterdir()):
        if directory.is_dir() and (directory / "README.md").exists() and directory.name not in registered_evidence:
            findings.append(f"WARN: unregistered experiment directory: {directory.relative_to(ROOT)}")

    files = tracked_files()
    for path in files:
        if path.exists() and path.stat().st_size > 25 * 1024 * 1024:
            findings.append(f"WARN: large tracked file ({path.stat().st_size / 1024**2:.1f} MiB): {path.relative_to(ROOT)}")
        if "__pycache__" in path.parts or path.name.endswith((".pyc", ".pyo")):
            findings.append(f"ERROR: generated Python cache is tracked: {path.relative_to(ROOT)}")

    active_text_roots = [ROOT / "research", ROOT / "experiments", ROOT / "src", ROOT / "scripts", ROOT / "docs", ROOT / "README.md"]
    for base in active_text_roots:
        candidates = [base] if base.is_file() else base.rglob("*") if base.exists() else []
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".yaml", ".yml", ".json"}:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for retired in RETIRED_PATH_NAMES:
                if retired in text:
                    findings.append(f"WARN: active reference to retired path {retired}: {path.relative_to(ROOT)}")
                    break

    digests: dict[str, list[Path]] = defaultdict(list)
    for pattern in ("model.pt", "best.pt", "last.pt", "*.torchscript"):
        for path in (ROOT / "logs" / "perception_models").rglob(pattern):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digests[digest].append(path)
    for digest, paths in digests.items():
        if len(paths) > 1:
            joined = ", ".join(str(path.relative_to(ROOT)) for path in paths)
            findings.append(f"WARN: duplicate model hash {digest}: {joined}")

    updated = date.fromisoformat(str(data["updated"]))
    if (date.today() - updated).days > 30:
        findings.append(f"WARN: registry status is stale ({updated})")

    if findings:
        print("Research hygiene report:")
        for item in findings:
            print(f"- {item}")
    else:
        print("Research hygiene report: clean")
    return 1 if any(item.startswith("ERROR:") for item in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
