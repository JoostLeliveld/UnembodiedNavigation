"""Evaluate actual launch descriptions without executing them; inspect one named ledger.

Use ROS_LOG_DIR=/tmp/state_estimation_launch_logs. This constructs no ROS node,
launch service, DDS participant, simulator, or command publisher.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/fusion_on_fixed_routes'))
import aligned
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
from experiments.core.visibility_launch_common import (
    parse_common_launch_config, resolve_world_setup, build_agent_runtime_actions,
)

keys = ['state_correction_ekf', 'state_correction_mode', 'heading_update_mode', 'odom_topic',
        'use_odom_for_predict', 'use_encoder_noise', 'use_command_noise', 'use_pixel_correction',
        'require_state_correction_envelope', 'pixel_timeout_s', 'dt', 'process_noise_xy',
        'process_noise_theta', 'max_predict_speed_mps', 'state_max_predict_dt_s',
        'state_reject_inflate_m2', 'state_reanchor_m', 'min_state_cov', 'odom_yaw_offset_rad',
        'use_sim_time', 'input_topic', 'output_topic', 'enabled', 'linear_slip_mean',
        'linear_slip_std', 'angular_slip_std', 'correlation_alpha']
selection_path = ROOT / 'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/diagnostics/P0_selection.json'
selection = json.loads(selection_path.read_text())
run = ROOT / selection['run']
manifest = json.loads((run / 'run_manifest.json').read_text())
rows = aligned.assimilations(run)
row = next(x for x in rows if x['status'] == 'accepted')
spec = importlib.util.spec_from_file_location('audit_campaign', ROOT / 'scripts/visibility_comparison/run_visibility_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
cfg = campaign._load_config(ROOT / 'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml')
args = campaign._build_launch_cmd(cfg, 'fusion_network_traverse', 'P0', 210, Path('/tmp/audit_unused_log_destination'))
context = LaunchContext()
context.launch_configurations.update(dict(x.split(':=', 1) for x in args if ':=' in x))
# Apply only declarative defaults from the real entry point. Never execute its
# OpaqueFunction or any process/start/event-handler action.
entry_spec = importlib.util.spec_from_file_location('audit_launch_entry', ROOT / 'src/experiments/launch/warehouse_primary_comparison.launch.py')
entry = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(entry)
for entity in entry.generate_launch_description().entities:
    if isinstance(entity, DeclareLaunchArgument):
        entity.execute(context)
resolved = resolve_world_setup(parse_common_launch_config(context))
actions = build_agent_runtime_actions(resolved)
nodes = []
for action in actions:
    if isinstance(action, Node):
        raw_executable = action.node_executable
        executable = raw_executable if isinstance(raw_executable, str) else ''.join(
            x if isinstance(x, str) else x.perform(context) for x in raw_executable)
        if executable in ('efe_agent', 'encoder_noise_node'):
            evaluated = evaluate_parameters(context, action._Node__parameters)
            nodes.append({'executable': executable, 'parameters': [
                {k: v for k, v in group.items() if k in keys}
                for group in evaluated if isinstance(group, dict)]})
assert {n['executable'] for n in nodes} == {'efe_agent', 'encoder_noise_node'}
result = dict(scope='Read-only configuration and one terminal row; no drive accuracy scored',
              selection=selection['run'],
              manifest_sha256=hashlib.sha256((run / 'run_manifest.json').read_bytes()).hexdigest(),
              manifest_parameters={k: manifest[k] for k in keys if k in manifest},
              first_accepted_event_via_aligned=row, launch_command=args,
              resolved_launch_nodes=nodes)
print(json.dumps(result, indent=2, allow_nan=False))
