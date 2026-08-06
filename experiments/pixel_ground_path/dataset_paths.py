"""Where the 4-camera localization-calibration dataset lives, after the cold-archive move.

e1-e5 were written against a workspace path that no longer exists: the 2026-08-05 archive
pass moved `_archive/` (15.4 GB) out to `/home/joostleliveld/Thesis_Cold_Archive/`, taking
the dataset payload with it, and every one of those scripts kept the old absolute path
hard-coded.  They therefore died on import of their inputs rather than reproducing anything.

Two locations now matter and they hold different things:

* the **workspace metadata copy** (`logs/perception_datasets/.../merged`) restored on
  2026-08-05 -- manifests, `localization_calibration_index.csv`, `sample_qualifications.csv`,
  and no images or labels;
* the **cold-storage payload**, which additionally holds `images/` and `labels/`.

`e2` needs images (and a GPU); `e1`, `e1b`, `e3`, `e4` and `e5` need the label polygons.  So
the payload root is the one to resolve for a re-run, and the metadata copy is only enough for
index-level work.  Nothing here copies anything back into the workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Restored in-workspace copy: manifests and indices only, no images and no labels.
METADATA_ROOT = Path(
    "logs/perception_datasets/warehouse_yolo_dataset_4cam_v3_20260724/merged"
)

_DATASET_RELATIVE = Path(
    "workspace_repositories/_archive/UnembodiedNavigation_paused_2026-08-05/"
    "perception_datasets/warehouse_yolo_dataset_4cam_v3_20260724/merged"
)


def dataset_root(repo: Path, *, require_payload: bool = True) -> Path:
    """Resolve the dataset root, preferring one that actually carries labels.

    ``require_payload=False`` accepts the workspace metadata copy, which is enough to read
    the calibration index but not to re-derive anything from label polygons.  Raises
    ``FileNotFoundError`` naming every location tried, because a silent fallback to a
    metadata-only root produces an empty analysis rather than an error.
    """

    archive_root = Path(
        os.environ.get("UNAV_COLD_ARCHIVE_ROOT", "/home/joostleliveld/Thesis_Cold_Archive")
    )
    payload_roots = (
        archive_root / _DATASET_RELATIVE,
        repo.parent / "_archive/UnembodiedNavigation_paused_2026-08-05/perception_datasets/"
        "warehouse_yolo_dataset_4cam_v3_20260724/merged",
    )
    for root in payload_roots:
        if (root / "labels").is_dir() and (root / "localization_calibration_index.csv").is_file():
            return root
    if not require_payload:
        metadata = repo / METADATA_ROOT
        if (metadata / "localization_calibration_index.csv").is_file():
            return metadata
    tried = "\n  ".join(str(root) for root in (*payload_roots, repo / METADATA_ROOT))
    raise FileNotFoundError(
        "the 4-camera localization-calibration dataset was not found.  Tried:\n  "
        f"{tried}\nSet UNAV_COLD_ARCHIVE_ROOT when the archive is mounted elsewhere; "
        "see logs/perception_datasets/COLD_STORAGE.md."
    )
