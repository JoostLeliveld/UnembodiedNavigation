from types import SimpleNamespace

from sim.wait_for_clock import WaitForClock
from sim.wait_for_odom import WaitForOdom
from sim.contact_evidence_node import ContactEvidenceNode


class _Logger:
    def info(self, _message):
        pass


def _clock(node, ns):
    msg = SimpleNamespace(clock=SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000))
    WaitForClock._cb(node, msg)


def test_clock_gate_requires_progress_and_restarts_count_on_rewind():
    node = SimpleNamespace(received=False, valid_count=0, last_stamp_ns=None, min_messages=3,
                           topic='/clock', get_logger=lambda: _Logger())
    _clock(node, 10)
    _clock(node, 10)
    _clock(node, 20)
    assert not node.received
    _clock(node, 5)
    _clock(node, 6)
    assert not node.received
    _clock(node, 7)
    assert node.received


def _odom(stamp_ns, frame='odom', child='base_footprint', x=0.0):
    vector = lambda **values: SimpleNamespace(x=values.get('x', 0.0), y=values.get('y', 0.0), z=values.get('z', 0.0))
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000,
                                                     nanosec=stamp_ns % 1_000_000_000),
                               frame_id=frame),
        child_frame_id=child,
        pose=SimpleNamespace(pose=SimpleNamespace(position=vector(x=x),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=vector(), angular=vector())),
    )


def test_odom_gate_rejects_wrong_frames_duplicates_and_nonfinite_values():
    node = SimpleNamespace(
        received=False, match_count=0, last_stamp_ns=None, min_messages=3,
        expected_frame_id='odom', expected_child_frame_id='base_footprint',
        require_pose_match=False, topic='/odom', get_logger=lambda: _Logger(),
    )
    WaitForOdom._cb(node, _odom(1, child='base_link'))
    WaitForOdom._cb(node, _odom(2))
    WaitForOdom._cb(node, _odom(2))
    WaitForOdom._cb(node, _odom(3, x=float('nan')))
    assert not node.received
    for stamp in (4, 5, 6):
        WaitForOdom._cb(node, _odom(stamp))
    assert node.received


def test_bringup_rejects_in_place_reset_and_wires_native_guard():
    from pathlib import Path
    launch = (Path(__file__).resolve().parents[2] / 'src/sim/launch/bringup_sim.launch.py').read_text()
    assert 'default_value="false"' in launch
    assert '_reject_unsafe_in_place_reset' in launch
    assert '/model/turtlebot3/cmd_vel_guard_input@geometry_msgs/msg/Twist' in launch
    assert '/model/turtlebot3/actuation_outcome@std_msgs/msg/String' in launch
    assert '/model/turtlebot3/tf@tf2_msgs/msg/TFMessage' in launch
    assert '/model/turtlebot3/odometry_tf@' not in launch
    assert '"timeout_s": 30.0' in launch
    assert '("turtlebot3", "base_footprint", "body_contact")' in launch
    assert '"status_topic": "/sim/contact_channel_status"' in launch


def test_active_robot_contact_sensor_references_converted_body_collision():
    from pathlib import Path
    robot = (Path(__file__).resolve().parents[2] /
             'src/sim/robot_description/urdf/warehouse_amr.urdf.xacro').read_text()
    assert '<sensor name="body_contact" type="contact">' in robot
    assert 'base_footprint_fixed_joint_lump__base_link_collision' in robot
    assert '<update_rate>60</update_rate>' in robot


def test_contact_heartbeat_never_interprets_silence_as_no_collision():
    import json
    node = object.__new__(ContactEvidenceNode)
    node.source_ids = ('native/sensor/a',)
    node.epoch = 'epoch'
    node.sequence = 0
    node.delivery_count = 0
    node.contact_count = 0
    node.last_source_stamp_ns = None
    node.last_receipt_wall_s = None
    messages = []
    node._publisher = SimpleNamespace(publish=messages.append)
    node._heartbeat()
    payload = json.loads(messages[-1].data)
    assert payload['state'] == 'configured_silent_requires_positive_control'
    assert payload['silence_is_no_contact'] is False
