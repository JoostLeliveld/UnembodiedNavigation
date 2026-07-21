from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "experiments/multicamera_commissioning_bigwarehouse/config"


def test_versioned_laptop_successors_keep_async_diagnostics_out_of_evidence() -> None:
    v1 = yaml.safe_load((CONFIG_DIR / "detector_4cam_v1.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((CONFIG_DIR / "detector_4cam_v2.yaml").read_text(encoding="utf-8"))
    v3 = yaml.safe_load(
        (CONFIG_DIR / "detector_4cam_v3_laptop_640x360.yaml").read_text(encoding="utf-8")
    )
    v4 = yaml.safe_load(
        (CONFIG_DIR / "detector_4cam_v4_laptop_async_microbatch.yaml").read_text(
            encoding="utf-8"
        )
    )
    v5 = yaml.safe_load(
        (CONFIG_DIR / "detector_4cam_v5_laptop_3hz_img416.yaml").read_text(
            encoding="utf-8"
        )
    )
    v6 = yaml.safe_load(
        (CONFIG_DIR / "detector_4cam_v6_laptop_direct_gz_torchscript416.yaml").read_text(
            encoding="utf-8"
        )
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

    assert v4["supersedes"] == v3["artifact_id"]
    assert v4["lock_status"] == "diagnostic_candidate_blocked_from_evidence"
    assert v4["runtime_pilot"]["evidence_selection"]["permitted"] is False
    assert v4["runtime_pilot"]["detector_mode"] == (
        "asynchronous_shared_model_microbatch_diagnostic_only"
    )
    assert v4["runtime_pilot"]["batch_size"] == "dynamic_1_to_4"
    assert v4["runtime_pilot"]["launch_switch"]["yolo_synchronization_mode"] == (
        "asynchronous"
    )
    assert v4["runtime_pilot"]["runtime_contract"] == (
        "absent_by_design_recorder_must_fail_closed"
    )

    assert v5["supersedes"] == v4["artifact_id"]
    assert v5["lock_status"] == "diagnostic_candidate_blocked_from_evidence"
    assert v5["world"] == "warehouse_full_4cam_laptop_640x360_3hz.world.sdf"
    assert v5["laptop_commissioning"]["source_sensor_update_rate_hz"] == 3
    assert v5["runtime_pilot"]["image_sizes"] == [416]
    assert v5["runtime_pilot"]["evidence_selection"]["permitted"] is False

    assert v6["supersedes"] == v5["artifact_id"]
    assert v6["lock_status"] == "commissioning_candidate_blocked_from_evidence"
    assert v6["runtime_pilot"]["image_sizes"] == [416]
    assert v6["runtime_pilot"]["launch_switch"]["yolo_input_transport"] == "direct_gz"
    assert v6["runtime_pilot"]["launch_switch"]["yolo_runtime_backend"] == "torchscript"
    assert len(v6["runtime_pilot"]["compiled_model_sha256"]) == 64
    assert min(v6["runtime_pilot"]["measured_rate_probe"]["output_wall_hz"].values()) >= 3.0
