import io
import json
from experiments.core.camera_opportunity_log import CameraOpportunityLog


def observation(hit=False, stamp=1., batch="batch1", camera="camera_A"):
    return json.dumps(dict(camera_id=camera, timestamp_s=stamp, source_batch_id=batch,
                           detection_valid=hit, bbox_xyxy=[1, 2, 3, 4] if hit else None))


def test_miss_and_duplicate_are_retained_and_distinguished():
    stream = io.StringIO(); log = CameraOpportunityLog(stream)
    first = log.append("camera_A", observation(), 1.2)
    repeat = log.append("camera_A", observation(), 1.4)
    other = log.append("camera_A", observation(True, 2., "batch2"), 2.3)
    assert first["observation"]["detection_valid"] is False
    assert not first["duplicate"] and repeat["duplicate"] and not other["duplicate"]
    assert len(stream.getvalue().splitlines()) == 3
    assert first["observation"]["timestamp_s"] == 1.
    assert first["receive_stamp_s"] == 1.2


def test_malformed_and_wrong_camera_deliveries_remain_auditable():
    stream = io.StringIO(); log = CameraOpportunityLog(stream)
    for payload in ("bad JSON", observation(camera="camera_B"), observation(batch="")):
        result = log.append("camera_A", payload, 3.)
        assert not result["valid_contract"] and result["reason"]
        assert result["raw_payload"] == payload
    assert len(stream.getvalue().splitlines()) == 3
