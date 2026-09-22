#!/usr/bin/env python3
"""Fail when active thesis configuration contradicts the canonical method lock."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Historical implementations may remain importable for old artifact readers and tests. The
# audit is deliberately about active guidance, locks, launch defaults, and campaign configs.
ACTIVE_ROOTS = (
    ROOT / "AGENTS.md",
    ROOT / "docs",
    ROOT / "experiments",
    ROOT / "scripts" / "visibility_comparison",
    ROOT / "src" / "experiments",
)

CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}
FORBIDDEN_CONFIG = (
    re.compile(r"(?m)^\s*nogo_mode\s*:\s*keep_in\s*(?:#.*)?$"),
    re.compile(r"(?m)^\s*local_controller_type\s*:\s*turn_then_go(?:_recovery)?\s*(?:#.*)?$"),
)
FORBIDDEN_DEFAULTS = (
    re.compile(r"local_controller_type[^\n]{0,100}default_value=['\"]turn_then_go"),
    re.compile(r"['\"]local_controller_type['\"]\s*:\s*['\"]turn_then_go['\"]"),
    re.compile(r"_declare_if_not\(['\"]local_controller_type['\"],\s*['\"]turn_then_go['\"]"),
)


def files_under(path: Path):
    if path.is_file():
        yield path
        return
    for candidate in path.rglob("*"):
        if (candidate.is_file()
                and "archive" not in candidate.parts
                and "runtime_integrity" not in candidate.parts
                and "source_snapshot" not in candidate.parts):
            yield candidate


def main() -> int:
    conflicts: list[str] = []
    for root in ACTIVE_ROOTS:
        for path in files_under(root):
            if path.suffix == ".tex" or path.name == "THESIS_METHOD_CANONICAL_LOCK.md":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            patterns = list(FORBIDDEN_DEFAULTS)
            if path.suffix in CONFIG_SUFFIXES:
                patterns.extend(FORBIDDEN_CONFIG)
            for pattern in patterns:
                if pattern.search(text):
                    conflicts.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    if conflicts:
        print("Canonical-lock conflicts:")
        print("\n".join(f"- {item}" for item in conflicts))
        return 1
    print("PASS: no active campaign uses keep_in or turn_then_go and no launch default selects turn_then_go")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
