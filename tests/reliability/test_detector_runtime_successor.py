from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "experiments/multicamera_commissioning_bigwarehouse/config"


def test_v1_is_pilot_only_and_v2_is_the_single_locked_deployment() -> None:
    v1 = yaml.safe_load((CONFIG_DIR / "detector_4cam_v1.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((CONFIG_DIR / "detector_4cam_v2.yaml").read_text(encoding="utf-8"))

    assert v1["runtime_pilot"]["evidence_selection"]["permitted"] is False
    assert v1["runtime_pilot"]["evidence_selection"]["selected_image_size"] is None
    assert v2["supersedes"] == v1["artifact_id"]
    assert v2["lock_status"] == "locked_before_evidence"
    selection = v2["runtime_pilot"]["evidence_selection"]
    assert selection["permitted"] is True
    assert selection["selected_image_size"] == 640
    assert v2["runtime_pilot"]["image_sizes"] == [640]
    assert v2["training"]["image_size"] == 640
    assert v2["runtime_pilot"]["model"] == v2["training"]["base_model"]
