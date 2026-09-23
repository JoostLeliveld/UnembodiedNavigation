#!/usr/bin/env python3
"""Select the reference-position dataset the pipeline runs on.

    THESIS_REFERENCE_DATASET=v8_uniform   (default) combined_recapture_v8, v8 lock
    THESIS_REFERENCE_DATASET=v5           combined_recapture_v5, v5 lock (superseded;
                                          only for like-for-like comparison with v5)

Every pipeline stage imports SOURCES, load_rows, load_records, image_path, LOCK_PATH and
CAMPAIGN_ROOT from here, so one variable moves the whole chain between datasets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from . import combined_recapture_v5 as _v5
    from . import combined_recapture_v8 as _v8
except ImportError:
    import combined_recapture_v5 as _v5
    import combined_recapture_v8 as _v8

REPO = Path(__file__).resolve().parents[2]
NAME = os.environ.get("THESIS_REFERENCE_DATASET", "v8_uniform")
_LOCKS = {
    "v5": REPO / "experiments/thesis_pipeline_lock/reference_position_campaign_lock.json",
    "v8_uniform": REPO / "experiments/thesis_pipeline_lock/reference_position_campaign_lock_v8_uniform.json",
}
_MODULES = {"v5": _v5, "v8_uniform": _v8}
if NAME not in _MODULES:
    raise RuntimeError(f"unknown THESIS_REFERENCE_DATASET={NAME!r}")

_module = _MODULES[NAME]
SOURCES = _module.SOURCES
load_rows = _module.load_rows
load_records = _module.load_records
image_path = _module.image_path
# THESIS_REFERENCE_LOCK overrides the lock for a dry run (e.g. v5 data with another
# detector). Campaign artifacts are always produced with the default lock of the dataset.
LOCK_PATH = Path(os.environ["THESIS_REFERENCE_LOCK"]) if os.environ.get(
    "THESIS_REFERENCE_LOCK") else _LOCKS[NAME]
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
CAMPAIGN_ROOT = REPO / LOCK["campaign_root"]
EXPECTED_WORKING_OPPORTUNITIES = int(LOCK["opportunity_accounting"].get(
    "expected_working_opportunities", 48_380))
EXPECTED_WORKING_UNIQUE_IMAGES = int(LOCK["opportunity_accounting"].get(
    "expected_working_unique_images", 14_570))
