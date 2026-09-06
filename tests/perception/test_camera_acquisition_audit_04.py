"""Deterministic audit probes. Assertions describe observed behavior, including defects.
AST extraction executes unchanged methods without importing ROS/model runtimes.
"""
import ast
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/perception'))
from perception.core.four_camera_batch import CAMERA_ORDER, FourCameraBatcher, PendingFrame, BatchContractError


def method(path, cls, name, **env):
    tree = ast.parse((ROOT / path).read_text())
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), f], type_ignores=[])
    ns = dict(math=math, np=np, time=time, CAMERA_ORDER=CAMERA_ORDER, BatchContractError=BatchContractError, **env)
    exec(compile(ast.fix_missing_locations(code), str(ROOT / path), 'exec'), ns)
    return ns[name]

BATCH = 'src/perception/perception/nodes/batched_four_camera_yolo_node.py'
MANAGER = 'src/reliability/reliability/nodes/camera_manager_node.py'
SINGLE = 'src/perception/perception/nodes/yolo_robot_detector_node.py'
SCHEDULED = 'src/perception/perception/nodes/scheduled_camera_detector_node.py'


def frame(cid, stamp, wall=0, value=0):
    return PendingFrame(cid, stamp, stamp / 1e9, wall, np.full((2, 3, 3), value, np.uint8))


def receiver():
    callback = method(MANAGER, 'CameraManagerNode', '_observation_callback',
                      CameraObservation=NS(from_json=lambda x: x), ContractValidationError=ValueError)
    s = NS(camera_ids=list(CAMERA_ORDER), _pending_source_batches={}, _latest={},
           _ready_source_batch_stamp_s=-math.inf, _ready_source_batch_id=None,
           _last_decided_source_batch_id=None, require_source_batch_id=True,
           get_logger=lambda: NS(warn=lambda _: None))
    def send(cid, bid, stamp, value=0):
        callback(s, cid)(NS(data=NS(camera_id=cid, source_batch_id=bid, timestamp_s=stamp, value=value)))
    return s, send


def test_staggered_rounds_order_duplicates_and_delayed_input():
    b = FourCameraBatcher()
    for i, c in enumerate(reversed(CAMERA_ORDER)):
        d = b.offer(frame(c, 1_000_000_000, i * .01, CAMERA_ORDER.index(c)))
    assert [int(f.payload[0, 0, 0]) for f in d.batch] == list(range(5))
    assert b.offer(frame(CAMERA_ORDER[0], 1_000_000_000)).status == 'duplicate'
    assert b.offer(frame(CAMERA_ORDER[0], 999_999_999)).status == 'out_of_order'


def test_distinct_rounds_inside_tolerance_can_merge():
    b = FourCameraBatcher()
    b.offer(frame(CAMERA_ORDER[0], 1_000_000_000, value=1))
    for c in CAMERA_ORDER[1:]:
        d = b.offer(frame(c, 1_040_000_000, value=2))
    assert [int(f.payload[0, 0, 0]) for f in d.batch] == [1, 2, 2, 2, 2]


def test_equal_stamp_distinct_image_is_indistinguishable():
    b = FourCameraBatcher()
    b.offer(frame(CAMERA_ORDER[0], 100, value=1))
    assert b.offer(frame(CAMERA_ORDER[0], 100, value=2)).status == 'duplicate'


def test_clock_reset_and_restart_identity_collision():
    ids = []
    for b in [FourCameraBatcher(), FourCameraBatcher()]:
        for c in CAMERA_ORDER:
            d = b.offer(frame(c, 100_000_000_000))
        ids.append('strict:' + ','.join(f'{f.camera_id}@{f.stamp_ns}' for f in d.batch))
        assert all(b.offer(frame(c, 0)).status == 'out_of_order' for c in CAMERA_ORDER)
    assert ids[0] == ids[1]


def test_absent_camera_expiry_reports_healthy_cameras_and_requires_call():
    b = FourCameraBatcher()
    for c in CAMERA_ORDER[:-1]:
        b.offer(frame(c, 100))
    assert b.bucket_report == ((100, CAMERA_ORDER[:-1]),)
    assert b.expire(1) == CAMERA_ORDER[:-1]
    assert b.pending_camera_ids == ()


def test_burst_has_no_count_bound_and_success_silently_discards_old_round():
    b = FourCameraBatcher(max_stamp_skew_s=0)
    for i in range(1000):
        b.offer(frame(CAMERA_ORDER[0], i, wall=0))
    assert len(b.bucket_report) == 1000
    for c in CAMERA_ORDER[1:]:
        d = b.offer(frame(c, 999, wall=.1))
    assert d.batch is not None and d.dropped_camera_ids == ()
    assert b.bucket_report == ()


def test_partial_manager_batches_unbounded_until_complete():
    s, send = receiver()
    for i in range(1000):
        for c in CAMERA_ORDER[:-1]:
            send(c, str(i), i)
    assert len(s._pending_source_batches) == 1000
    assert s._ready_source_batch_id is None


def test_manager_equal_time_new_identity_and_reset_discarded():
    s, send = receiver()
    for bid, stamp in [('first', 100), ('distinct', 100), ('reset', 0)]:
        for c in CAMERA_ORDER:
            send(c, bid, stamp)
    assert s._ready_source_batch_id == 'first'


def test_duplicate_delivery_and_fast_ticks_do_not_repeat_active_decision():
    s, send = receiver()
    calls = []
    s.get_clock = lambda: NS(now=lambda: NS(nanoseconds=2_000_000_000))
    s._map_observations = lambda now: list(s._latest.values())
    s._publish_map_observations = lambda obs: None
    s.fusion_mode, s.active_pub = True, object()
    s._decide_fused = lambda *args, **kw: calls.append(kw['source_batch_id'])
    tick = method(MANAGER, 'CameraManagerNode', '_decide')
    for _ in range(3):
        for c in CAMERA_ORDER:
            send(c, 'physical', 1)
        for _ in range(20):
            tick(s)
    assert calls == ['physical']


def test_chunk_result_counts_can_compensate_and_shift_identity():
    predict = method(BATCH, 'BatchedFourCameraYoloNode', '_predict_batch')
    calls = []
    def fake(**kw):
        ids = [int(x[0, 0, 0]) for x in kw['source']]
        calls.append(ids)
        return {0: [0], 2: [2, 3, 99], 4: [4]}[ids[0]]
    s = NS(inference_chunk=2, image_size=960, predict_conf_floor=0, iou_threshold=.45,
           device='cpu', model=NS(predict=fake))
    result = predict(s, [frame(c, 0, value=i).payload for i, c in enumerate(CAMERA_ORDER)])
    assert calls == [[0, 1], [2, 3], [4]]
    assert result == [0, 2, 3, 99, 4]  # total count passes; camera B now receives C's result


def test_single_detector_repeated_delivery_reinfers_same_image():
    process = method(SINGLE, 'YoloRobotDetectorNode', '_process_image', image_msg_to_bgr8=lambda m: m.image)
    calls = []
    s = NS(debug_frame_dir='', get_clock=lambda: NS(now=lambda: NS(nanoseconds=10)),
           _predict=lambda im: calls.append(im) or [], _publish_diagnostics=lambda *a: None)
    msg = NS(image=np.zeros((2, 3, 3)), header=NS(stamp=1))
    process(s, msg, 0); process(s, msg, 0)
    assert len(calls) == 2


def test_scheduled_exception_claims_frame_without_terminal_observation():
    tick = method(SCHEDULED, 'ScheduledCameraDetector', '_tick')
    calls, outputs = [], []
    def fail(**kw):
        calls.append(kw); raise RuntimeError('controlled inference failure')
    s = NS(belief=(0, 0), cams=[('synthetic', '', '')], _cov_at=lambda *a: 1,
           selection_mode='coverage_best_with_fallback', min_cov=0,
           latest={'synthetic': object()}, _stamp_s=lambda m: 1,
           _last_processed_stamp_s={'synthetic': -math.inf},
           bridge=NS(imgmsg_to_cv2=lambda *a: np.zeros((2,3,3))),
           model=NS(predict=fail), imgsz=960, conf=.25, iou=.45, device='cpu',
           get_logger=lambda: NS(warn=lambda m: None), _publish_observation=lambda **kw: outputs.append(kw))
    tick(s); tick(s)
    assert len(calls) == 1 and outputs == []


def test_decoder_padding_and_rgb_coordinates():
    tree = ast.parse((ROOT / 'src/perception/perception/core/ros_image.py').read_text())
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    f.returns = None
    for arg in f.args.args: arg.annotation = None
    ns = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])), '<decoder>', 'exec'), ns)
    msg = NS(encoding='rgb8', height=2, width=1, step=4, data=bytes([1,2,3,99,4,5,6,88]))
    out = ns[f.name](msg)
    assert out.tolist() == [[[3,2,1]], [[6,5,4]]]
    assert out.flags.c_contiguous
    msg.step = 2
    with pytest.raises(ValueError): ns[f.name](msg)


def test_slow_inference_has_no_post_inference_age_rejection_and_partial_publish():
    from perception.core.four_camera_batch import MAX_FUTURE_IMAGE_STAMP_S, validate_batch_results
    process = method(BATCH, 'BatchedFourCameraYoloNode', '_process_frames',
                     MAX_FUTURE_IMAGE_STAMP_S=MAX_FUTURE_IMAGE_STAMP_S,
                     validate_batch_results=validate_batch_results,
                     _DirectGzImagePayload=type('Unused', (), {}),
                     image_msg_to_bgr8=lambda x: x, _BatchTiming=lambda **kw: NS(**kw))
    clock, published = [1.0], []
    def predict(images):
        clock[0] = 101.0  # controlled 100-second inference, no real sleep
        return [NS(boxes=[]) for _ in images]
    def publish(item, image, selection, timing, **kw):
        published.append((item.camera_id, clock[0] - item.stamp_ns / 1e9))
        if item.camera_id == CAMERA_ORDER[1]: raise RuntimeError('publication failure')
    s = NS(_clock_s=lambda: clock[0], torchscript_detection_only=False,
           _predict_batch=predict, _prepare_result=lambda r: {}, _publish_result=publish)
    with pytest.raises(RuntimeError, match='publication failure'):
        process(s, tuple(frame(c, 1_000_000_000) for c in CAMERA_ORDER))
    assert published == [(CAMERA_ORDER[0], 100.0), (CAMERA_ORDER[1], 100.0)]
