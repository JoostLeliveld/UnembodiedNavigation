from __future__ import annotations

from pathlib import Path
import sys

import pytest


PIXEL_GROUND = Path(__file__).resolve().parents[2] / "experiments" / "pixel_ground_path"
if str(PIXEL_GROUND) not in sys.path:
    sys.path.insert(0, str(PIXEL_GROUND))

from dataset_paths import METADATA_ROOT, _DATASET_RELATIVE, dataset_root  # noqa: E402


def test_archive_root_can_be_relocated_with_environment(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "cold"
    payload = archive / _DATASET_RELATIVE
    (payload / "labels").mkdir(parents=True)
    (payload / "localization_calibration_index.csv").write_text("sample_id\n", encoding="utf-8")
    monkeypatch.setenv("UNAV_COLD_ARCHIVE_ROOT", str(archive))

    assert dataset_root(tmp_path / "repo") == payload


def test_metadata_only_root_is_explicit_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UNAV_COLD_ARCHIVE_ROOT", str(tmp_path / "missing-cold"))
    repo = tmp_path / "workspace" / "repo"
    metadata = repo / METADATA_ROOT
    metadata.mkdir(parents=True)
    (metadata / "localization_calibration_index.csv").write_text(
        "sample_id\n", encoding="utf-8"
    )

    assert dataset_root(repo, require_payload=False) == metadata
    with pytest.raises(FileNotFoundError):
        dataset_root(repo, require_payload=True)


def test_missing_payload_error_names_relocation_variable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UNAV_COLD_ARCHIVE_ROOT", str(tmp_path / "missing-cold"))

    with pytest.raises(FileNotFoundError, match="UNAV_COLD_ARCHIVE_ROOT"):
        dataset_root(tmp_path / "repo")
