from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_current_and_multicamera_experiment_namespaces_are_separate() -> None:
    current = ROOT / "experiments" / "single_camera_current" / "README.md"
    extension = ROOT / "experiments" / "multicamera_fusion_extension" / "README.md"

    current_text = current.read_text(encoding="utf-8")
    extension_text = extension.read_text(encoding="utf-8")

    assert "scripts/visibility_comparison/warehouse_visibility_campaign.yaml" in current_text
    assert "15/20" in current_text
    assert "20/20" in current_text
    assert "must not modify the active C1/C2 campaign" in current_text

    assert "CameraObservation" in extension_text
    assert "CameraQuality" in extension_text
    assert "not a license" in extension_text
    assert "real camera-B logs" in extension_text


def test_experiment_overview_links_evidence_namespaces() -> None:
    overview = (ROOT / "experiments" / "README.md").read_text(encoding="utf-8")

    assert "single_camera_current/" in overview
    assert "multicamera_fusion_extension/" in overview


def test_detector_topic_parameters_keep_single_camera_defaults() -> None:
    detector = (ROOT / "src" / "perception" / "perception" / "nodes" / "yolo_robot_detector_node.py").read_text(
        encoding="utf-8"
    )

    assert "declare_parameter('image_topic', '/external_camera/image_raw')" in detector
    assert "declare_parameter('pixel_pose_topic', '/perception/pixel_pose')" in detector
    assert "declare_parameter('diagnostics_topic', DETECTION_DIAGNOSTICS_TOPIC)" in detector
    assert "declare_parameter('camera_observation_topic', '')" in detector
    assert "self.create_subscription(Image, self.image_topic" in detector
    assert "self.create_publisher(PoseStamped, self.pixel_pose_topic" in detector
