#!/usr/bin/env python3
"""Owned, source-bound active-world command-silence and sustained-rest probe.

Run only after building the selected source tree. The script allocates the ROS
domain and Gazebo partition supplied on the command line, owns one launch
process group, records every received event in callback order, and terminates
only that process group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Contacts
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.convert import message_to_ordereddict
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = (
    'src/sim/launch/bringup_sim.launch.py',
    'src/sim/robot_description/urdf/warehouse_amr.urdf.xacro',
    'src/sim/sim/actuation_noise_node.py',
    'src/sim/sim/clock_throttle_node.py',
    'src/sim/sim/contact_evidence_node.py',
    'src/sim/sim/wait_for_clock.py',
    'src/sim/sim/wait_for_odom.py',
    'src/sim_command_guard/include/sim_command_guard/command_guard_core.hh',
    'src/sim_command_guard/src/command_guard_system.cc',
    'src/unav_common/unav_common/occlusion_geometry.py',
    'src/experiments/config/world_profiles.yaml',
    'src/experiments/experiments/core/visibility_launch_common.py',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Recorder(Node):
    def __init__(self, stream):
        super().__init__('simulator_integrity_private_probe')
        self.stream = stream
        self.sequence = 0
        self.clock_ns = None
        self.odom = []
        self.outcomes = []
        self.contacts = 0
        self.contact_status = []
        best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Clock, '/clock', self._clock, best_effort)
        self.create_subscription(Odometry, '/odom', self._odom, 100)
        self.create_subscription(String, '/sim/actuation_outcome', self._outcome, 100)
        self.create_subscription(Contacts, '/world_contacts', self._contact, 100)
        self.create_subscription(String, '/sim/contact_channel_status', self._contact_status, 100)
        self.command_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def record(self, topic, message, extra=None):
        self.sequence += 1
        payload = {
            'delivery_sequence': self.sequence,
            'topic': topic,
            'receipt_monotonic_ns': time.monotonic_ns(),
            'receipt_wall_ns': time.time_ns(),
            'latest_ros_clock_ns': self.clock_ns,
            'message': message_to_ordereddict(message),
        }
        if extra:
            payload.update(extra)
        self.stream.write(json.dumps(payload, separators=(',', ':')) + '\n')
        self.stream.flush()

    def _clock(self, message):
        self.clock_ns = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        self.record('/clock', message)

    def _odom(self, message):
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        item = {
            'stamp_ns': stamp_ns,
            'receipt_monotonic_ns': time.monotonic_ns(),
            'x_m': float(message.pose.pose.position.x),
            'y_m': float(message.pose.pose.position.y),
            'linear_x_mps': float(message.twist.twist.linear.x),
            'angular_z_radps': float(message.twist.twist.angular.z),
        }
        self.odom.append(item)
        self.record('/odom', message)

    def _outcome(self, message):
        try:
            decoded = json.loads(message.data)
        except json.JSONDecodeError:
            decoded = {'malformed_raw': message.data}
        decoded['receipt_monotonic_ns'] = time.monotonic_ns()
        self.outcomes.append(decoded)
        self.record('/sim/actuation_outcome', message, {'decoded': decoded})

    def _contact(self, message):
        self.contacts += len(message.contacts)
        self.record('/world_contacts', message)

    def _contact_status(self, message):
        try:
            decoded = json.loads(message.data)
        except json.JSONDecodeError:
            decoded = {'malformed_raw': message.data}
        self.contact_status.append(decoded)
        self.record('/sim/contact_channel_status', message, {'decoded': decoded})

    def publish_command(self, linear_mps, angular_radps):
        command = Twist()
        command.linear.x = float(linear_mps)
        command.angular.z = float(angular_radps)
        self.command_publisher.publish(command)
        self.record('/cmd_vel:probe_publish', command, {
            'units': {'linear_x': 'm/s', 'angular_z': 'rad/s'},
        })


def spin_until(node, predicate, timeout_wall_s):
    deadline = time.monotonic() + timeout_wall_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return True
    return False


def process_identity() -> dict:
    diff = subprocess.run(
        ['git', 'diff', '--binary', '--', *SOURCE_FILES], cwd=ROOT,
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return {
        'git_head': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip(),
        'selected_source_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'selected_source_files': {
            name: sha256(ROOT / name) for name in SOURCE_FILES
        },
        'installed_world': str((ROOT / 'install/sim/share/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf').resolve()),
        'installed_world_sha256': sha256(
            ROOT / 'install/sim/share/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
        ),
        'installed_guard_sha256': sha256(
            ROOT / 'install/sim_command_guard/lib/libsim_command_guard.so'
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--ros-domain-id', type=int, required=True)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--linear-mps', type=float, default=0.2)
    parser.add_argument('--angular-radps', type=float, default=0.0)
    parser.add_argument('--command-duration-sim-s', type=float, default=0.6)
    parser.add_argument('--rest-observation-sim-s', type=float, default=1.0)
    parser.add_argument('--max-rest-displacement-m', type=float, default=0.005)
    parser.add_argument('--max-rest-linear-speed-mps', type=float, default=0.01)
    parser.add_argument('--max-rest-angular-speed-radps', type=float, default=0.01)
    parser.add_argument('--wall-timeout-s', type=float, default=180.0)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(args.ros_domain_id)
    env['IGN_PARTITION'] = args.partition
    env['GZ_PARTITION'] = args.partition
    env['ROS_LOG_DIR'] = str(output / 'ros_logs')
    Path(env['ROS_LOG_DIR']).mkdir()
    os.environ.update({key: env[key] for key in ('ROS_DOMAIN_ID', 'IGN_PARTITION', 'GZ_PARTITION')})

    metadata = {
        'schema_version': 1,
        'scenario': 'command_silence_sustained_rest',
        'status': 'running',
        'units': {
            'position': 'm', 'linear_velocity': 'm/s',
            'angular_velocity': 'rad/s', 'sim_time': 'ns',
        },
        'parameters': vars(args),
        'isolation': {
            'ros_domain_id': args.ros_domain_id,
            'gazebo_partition': args.partition,
            'owned_process_group': True,
        },
        'source_identity': process_identity(),
        'explicitly_unexercised_cases': [
            'graceful_adapter_exit', 'forced_adapter_death', 'clock_bridge_loss',
            'physics_pause_resume', 'simulation_time_rewind',
            'known_contact_positive_control', 'clear_contact_silence',
            'gui_headless_render_parity',
        ],
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    launch_command = [
        'ros2', 'launch', 'sim', 'bringup_sim.launch.py',
        'world:=warehouse_v2.world.sdf', 'world_name:=warehouse_v2',
        'headless:=true', 'reset_world:=false', 'use_lidar:=false',
        'bridge_scan:=false', 'bridge_contacts:=true', 'bridge_camera_a:=false',
        'spawn_x:=0.55', 'spawn_y:=-7.50', 'spawn_z:=0.05', 'spawn_yaw:=1.5708',
    ]
    (output / 'command.json').write_text(json.dumps({
        'required_shell_setup': [
            'source /opt/ros/humble/setup.bash', 'source install/setup.bash'
        ],
        'probe_argv': [str(Path(__file__).resolve()), *os.sys.argv[1:]],
        'launch_argv': launch_command,
    }, indent=2) + '\n')

    with (output / 'launch.log').open('w') as launch_log, \
            (output / 'deliveries.jsonl').open('w') as deliveries:
        launch = subprocess.Popen(
            launch_command, cwd=ROOT, env=env, stdout=launch_log,
            stderr=subprocess.STDOUT, start_new_session=True, text=True,
        )
        node = None
        try:
            rclpy.init()
            node = Recorder(deliveries)
            ready = spin_until(node, lambda: bool(node.odom and node.clock_ns is not None), 60.0)
            if not ready:
                raise RuntimeError('startup did not produce both advancing clock and odometry')

            command_start_ns = node.clock_ns
            next_publish_ns = command_start_ns
            while node.clock_ns - command_start_ns < int(args.command_duration_sim_s * 1e9):
                if node.clock_ns >= next_publish_ns:
                    node.publish_command(args.linear_mps, args.angular_radps)
                    next_publish_ns += 100_000_000
                rclpy.spin_once(node, timeout_sec=0.02)
            final_command_clock_ns = node.clock_ns

            got_timeout = spin_until(
                node,
                lambda: any(event.get('reason') == 'physics_time_timeout' for event in node.outcomes),
                args.wall_timeout_s,
            )
            if not got_timeout:
                raise RuntimeError('no physics_time_timeout outcome was delivered')
            timeout_event = next(
                event for event in reversed(node.outcomes)
                if event.get('reason') == 'physics_time_timeout'
            )
            timeout_sim_ns = int(timeout_event['forwarded_sim_stamp_ns'])
            odom_at_timeout = min(node.odom, key=lambda item: abs(item['stamp_ns'] - timeout_sim_ns))
            rest_end_ns = timeout_sim_ns + int(args.rest_observation_sim_s * 1e9)
            if not spin_until(node, lambda: node.clock_ns is not None and node.clock_ns >= rest_end_ns,
                              args.wall_timeout_s):
                raise RuntimeError('post-timeout sustained-rest observation did not complete')
            rest = [item for item in node.odom if timeout_sim_ns <= item['stamp_ns'] <= rest_end_ns]
            if len(rest) < 3:
                raise RuntimeError(f'only {len(rest)} odometry samples in sustained-rest window')
            displacement = max(math.hypot(
                item['x_m'] - odom_at_timeout['x_m'],
                item['y_m'] - odom_at_timeout['y_m'],
            ) for item in rest)
            max_linear = max(abs(item['linear_x_mps']) for item in rest)
            max_angular = max(abs(item['angular_z_radps']) for item in rest)
            checks = {
                'timeout_elapsed_sim_s': (
                    timeout_sim_ns - final_command_clock_ns
                ) * 1e-9,
                'rest_sample_count': len(rest),
                'rest_observation_sim_s': args.rest_observation_sim_s,
                'max_rest_displacement_m': displacement,
                'max_abs_rest_linear_speed_mps': max_linear,
                'max_abs_rest_angular_speed_radps': max_angular,
                'displacement_pass': displacement <= args.max_rest_displacement_m,
                'linear_speed_pass': max_linear <= args.max_rest_linear_speed_mps,
                'angular_speed_pass': max_angular <= args.max_rest_angular_speed_radps,
            }
            checks['passed'] = all(checks[key] for key in (
                'displacement_pass', 'linear_speed_pass', 'angular_speed_pass'
            ))
            metadata.update({
                'status': 'passed' if checks['passed'] else 'failed',
                'checks': checks,
                'timeout_event': timeout_event,
                'delivery_count': node.sequence,
                'raw_contact_count': node.contacts,
                'last_contact_status': node.contact_status[-1] if node.contact_status else None,
            })
        except Exception as exc:  # noqa: BLE001
            metadata.update({'status': 'infrastructure_failure', 'error': repr(exc)})
            raise
        finally:
            if node is not None:
                node.destroy_node()
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except RuntimeError:
                pass
            os.killpg(launch.pid, signal.SIGINT)
            try:
                launch.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(launch.pid, signal.SIGTERM)
                try:
                    launch.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(launch.pid, signal.SIGKILL)
                    launch.wait()
            metadata['launch_returncode'] = launch.returncode
            (output / 'result.json').write_text(json.dumps(metadata, indent=2) + '\n')

    if metadata['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
