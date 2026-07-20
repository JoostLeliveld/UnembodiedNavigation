from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "experiments/multicamera_commissioning_bigwarehouse/config"


def test_v1_is_pilot_only_v2_is_locked_and_v3_is_a_blocked_laptop_successor() -> None:
    v1 = yaml.safe_load((CONFIG_DIR / "detector_4cam_v1.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((CONFIG_DIR / "detector_4cam_v2.yaml").read_text(encoding="utf-8"))
    v3 = yaml.safe_load(
        (CONFIG_DIR / "detector_4cam_v3_laptop_640x360.yaml").read_text(encoding="utf-8")
    )

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

    assert v3["supersedes"] == v2["artifact_id"]
    assert v3["lock_status"] == "commissioning_candidate_blocked_from_evidence"
    assert v3["world"] == "warehouse_full_4cam_laptop_640x360.world.sdf"
    assert v3["laptop_commissioning"]["source_image_size"] == [640, 360]
    assert v3["laptop_commissioning"]["paper_source_image_size"] == [1280, 720]
    assert v3["laptop_commissioning"]["render_pixel_reduction_factor"] == 4
    assert v3["laptop_commissioning"]["renderer"] == "nvidia_prime_egl_required_on_hybrid_laptop"
    assert v3["laptop_commissioning"]["launch_argument"] == "use_nvidia_prime_offload:=true"
    assert v3["runtime_pilot"]["image_sizes"] == [640]
    assert all(
        camera["source_image_size"] == [640, 360]
        for camera in v3["cameras"].values()
    )
