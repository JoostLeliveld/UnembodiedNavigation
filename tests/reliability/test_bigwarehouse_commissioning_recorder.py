from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "record_operational_logs.py"


def _module():
    spec = importlib.util.spec_from_file_location("bigwarehouse_operational_recorder", RECORDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorder_uses_the_actual_bigwarehouse_camera_models_and_operational_schema() -> None:
    module = _module()

    assert module.DEFAULT_CAMERA_MODELS == {
        "camera_A": "external_camera",
        "camera_B": "external_camera_b",
        "camera_C": "external_camera_c",
        "camera_D": "external_camera_d",
    }
    assert module.DEFAULT_CAMERA_TOPICS["camera_B"] == "/perception/camera_observation/camera_B"
    assert "pred_world_x" in module.PERCEPTION_FIELDS
    assert "ground_truth" not in " ".join(module.PERCEPTION_FIELDS)
    assert "true_x" not in " ".join(module.PERCEPTION_FIELDS)
    assert "--use-sim-time" in module.main.__code__.co_consts


def test_recorder_calibrations_parse_all_fullwarehouse_camera_includes() -> None:
    module = _module()
    world = ROOT / "src" / "sim" / "gazebo_worlds" / "worlds" / "warehouse_full_4cam.world.sdf"

    expected = {
        "external_camera": (-6.0, -10.0, 6.1),
        "external_camera_b": (-6.0, 10.0, 6.1),
        "external_camera_c": (6.0, -10.0, 6.1),
        "external_camera_d": (6.0, 10.0, 6.1),
    }
    for include_name, pos in expected.items():
        camera = module.camera_model_from_world(world, include_name=include_name)
        assert tuple(camera.cam_pos) == pos
        assert camera.pixel_to_world(640.0, 360.0) is not None


def test_recorder_maps_spawn_local_odom_back_to_the_documented_warehouse_frame() -> None:
    module = _module()
    recorder = object.__new__(module.OperationalLogRecorder)
    recorder.odom_origin_x_m = -1.8
    recorder.odom_origin_y_m = -5.4
    recorder.odom_origin_yaw_rad = 1.5708
    recorder._origin_cos = __import__("math").cos(recorder.odom_origin_yaw_rad)
    recorder._origin_sin = __import__("math").sin(recorder.odom_origin_yaw_rad)

    world_x, world_y = recorder._odom_to_warehouse(11.6, 0.0)

    assert abs(world_x - (-1.8)) < 1e-4
    assert abs(world_y - 6.2) < 1e-4
