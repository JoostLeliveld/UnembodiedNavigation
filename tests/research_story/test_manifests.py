"""Validate the research_story chapter manifests.

Nothing validated these YAMLs before (the only automated check was the
dangling-symlink guard in research_story/_tools/link_media.py). This asserts
each chapter's evidence.yaml parses, carries the required keys, uses a known
status, has well-formed evidence entries, and does not point at a *source*
artifact (code/doc) that is missing on disk. Data-kind paths (campaign_runs,
datasets, fit_artifacts, ...) live under gitignored logs/ and are intentionally
not existence-checked here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "research_story"

# Statuses observed across chapters + registry (kept in sync with registry.yaml).
STATUSES = {
    "LOCKED",
    "ACTIVE",
    "PARTIAL",
    "PLUMBING",
    "PLUMBING_PLUS_PILOT",
    "PLANNED",
    "FUTURE",
}
REQUIRED_KEYS = {"chapter", "status", "world", "evidence"}
# Kinds whose `path` points at tracked source that must exist on disk.
SOURCE_KINDS = {
    "canonical_code",
    "code",
    "module",
    "runtime_code",
    "doc",
    "tests",
    "generator",
    "layout_doc",
    "contract",
}

MANIFESTS = sorted(STORY.glob("[0-9]*/evidence.yaml"))


def test_manifests_present() -> None:
    assert MANIFESTS, "no research_story/NN/evidence.yaml manifests found"


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_is_valid(manifest: Path) -> None:
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{manifest} is not a mapping"

    missing = REQUIRED_KEYS - set(data)
    assert not missing, f"{manifest} missing required keys: {sorted(missing)}"

    assert data["status"] in STATUSES, f"{manifest}: unknown status {data['status']!r}"

    evidence = data.get("evidence") or []
    for i, entry in enumerate(evidence):
        assert isinstance(entry, dict), f"{manifest}: evidence[{i}] is not a mapping"
        assert "path" in entry and "kind" in entry, (
            f"{manifest}: evidence[{i}] must have 'path' and 'kind'"
        )
        if entry["kind"] in SOURCE_KINDS:
            target = (manifest.parent / entry["path"]).resolve()
            assert target.exists(), (
                f"{manifest}: dangling {entry['kind']} path {entry['path']!r}"
            )
