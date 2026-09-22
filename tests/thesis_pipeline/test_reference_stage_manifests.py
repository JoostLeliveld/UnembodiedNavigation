import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_reference_stage_manifests_are_hash_complete():
    pipeline = json.loads((REPO / "experiments/thesis_pipeline_lock/pipeline_lock.json").read_text())
    for stage_id in ("06_detector_gate", "07_correction_and_covariance", "08_planning_information"):
        stage = next(item for item in pipeline["stages"] if item["id"] == stage_id)
        assert stage["status"] == "locked"
        manifest_path = REPO / stage["manifest"]
        assert digest(manifest_path) == stage["manifest_sha256"]
        manifest = json.loads(manifest_path.read_text())
        assert manifest["final_audit_accessed"] is False
        for artifact in manifest["artifacts"].values():
            artifact_path = REPO / artifact["path"]
            assert digest(artifact_path) == artifact["sha256"]


def test_rproj_remains_evaluation_only():
    stage07 = json.loads((
        REPO / "experiments/thesis_pipeline_lock/stage07_reference_models_manifest.json"
    ).read_text())
    stage08 = json.loads((
        REPO / "experiments/thesis_pipeline_lock/stage08_reference_planning_manifest.json"
    ).read_text())
    evaluation = json.loads((
        REPO / "logs/thesis_final_pipeline_v1/recapture_v5/ddev_evaluation/manifest.json"
    ).read_text())
    assert stage07["replay_evaluation_only_models"] == ["Rproj_homography_pixel"]
    assert stage08["Rproj_navigation_arm"] is False
    assert evaluation["Rproj"]["navigation_arm"] is False
    assert evaluation["Rproj"]["planning_field"] is False


def test_stage09_is_pending_on_rate_qualification_and_audit_is_sealed():
    pipeline = json.loads((REPO / "experiments/thesis_pipeline_lock/pipeline_lock.json").read_text())
    stage09 = next(item for item in pipeline["stages"] if item["id"] == "09_navigation_campaign")
    assert stage09["status"] == "pending"
    assert "rate qualification failed" in stage09["blocking_reason"]
    assert stage09["final_audit_accessed"] is False
