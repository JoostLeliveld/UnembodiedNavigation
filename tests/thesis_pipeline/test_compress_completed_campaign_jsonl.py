import importlib.util
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments/thesis_pipeline_lock/compress_completed_campaign_jsonl.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compress_campaign", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_only_complete_validated_attempts_are_compressed(tmp_path):
    module = load_module()
    complete = tmp_path / "complete"
    pending = tmp_path / "pending"
    run = complete / "experiment_1"
    run.mkdir(parents=True)
    pending.mkdir()
    for path in (
        complete / "detector_outcomes.jsonl",
        complete / "manager_outcomes.jsonl",
        run / "runtime_event_deliveries.jsonl",
        run / "camera_opportunities.jsonl",
        run / "belief_predictions.jsonl",
        pending / "detector_outcomes.jsonl",
    ):
        path.write_text(json.dumps({"status": "session_stopped"}) + "\n")
    (tmp_path / "campaign_log.json").write_text(json.dumps({
        "complete": {
            "finished_at": "now", "attempt_evidence_complete": True,
            "run_log_dir": str(complete),
        },
        "pending": {
            "finished_at": None, "attempt_evidence_complete": None,
            "run_log_dir": str(pending),
        },
    }))

    attempts, campaign_complete = module.completed_attempts(tmp_path)
    assert attempts == {complete.resolve()}
    assert campaign_complete is False
    assert module.compress_attempt(complete) == 5
    for original in (
        complete / "detector_outcomes.jsonl",
        complete / "manager_outcomes.jsonl",
        run / "runtime_event_deliveries.jsonl",
        run / "camera_opportunities.jsonl",
        run / "belief_predictions.jsonl",
    ):
        compressed = original.with_name(original.name + ".zst")
        assert not original.exists()
        subprocess.run(["zstd", "-q", "-t", str(compressed)], check=True)
    assert (pending / "detector_outcomes.jsonl").is_file()
