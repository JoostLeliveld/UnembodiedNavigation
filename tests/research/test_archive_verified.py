from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "archive_verified.py"
SPEC = importlib.util.spec_from_file_location("archive_verified", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_inventory_hashes_files_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "payload"
    root.mkdir()
    (root / "file.txt").write_text("evidence", encoding="utf-8")
    (root / "link.txt").symlink_to("file.txt")

    inventory, size = MODULE.inventory(root)

    assert size == len("evidence")
    assert inventory["file.txt"]["sha256"] == "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
    assert inventory["link.txt"]["type"] == "symlink"
    assert inventory["link.txt"]["target"] == "file.txt"


def test_copy_inventory_matches_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "data.bin").write_bytes(b"\x00\x01\x02")

    MODULE.copy_source(source, destination)

    assert MODULE.inventory(source) == MODULE.inventory(destination)
