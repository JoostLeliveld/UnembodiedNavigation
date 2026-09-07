#!/usr/bin/env python3
"""Full-route preflight using the launch-resolved global planner (no ROS nodes run).

Every optimizer attempt is retained. This tests map feasibility and implementation,
not realized localization accuracy or the ranking of perception quality.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/icra_mpl')
import argparse
import ast
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in (
    'src/planning', 'src/reliability', 'src/unav_common', 'src/experiments')]
from study import digest, writejson
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.camera_network import projection_jacobian
from unav_common.lane_graph_routes import generate_route_seeds


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve(config, task, arm, seed):
    """Build the actual launch description, capture parameters; execute no actions."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.utilities import perform_substitutions
    from experiments.core import visibility_launch_common as common
    from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
    launch = module(REPO / 'src/experiments/launch/warehouse_primary_comparison.launch.py',
                    'network_probe_launch')
    runner = module(REPO / 'scripts/visibility_comparison/run_visibility_campaign.py',
                    'network_probe_campaign')
    cfg = runner._load_config(config)
    context = LaunchContext()
    for action in launch.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(context, action.default_value)
    command = runner._build_launch_cmd(cfg, task, arm, seed, Path('/tmp/network_route_graph_only'))
    for arg in command:
        if ':=' in arg:
            key, value = arg.split(':=', 1)
            context.launch_configurations[key] = value
    captured = {}
    original_node, original_build = common.Node, common.build_agent_runtime_actions

    def capture_node(*args, **kwargs):
        if kwargs.get('package') == 'planning' and kwargs.get('executable') == 'efe_agent':
            captured['parameters'] = dict(kwargs['parameters'][0])
        return original_node(*args, **kwargs)

    def capture_build(cfg):
        captured['resolved'] = cfg
        return original_build(cfg)

    try:
        common.Node, common.build_agent_runtime_actions = capture_node, capture_build
        launch._launch_setup(context)
    finally:
        common.Node, common.build_agent_runtime_actions = original_node, original_build
    # Literal defaults are read from the node so unexposed v_min/w limits agree.
    node_path = REPO / 'src/planning/planning/nodes/unicycle_planner_node.py'
    defaults = {}
    for call in ast.walk(ast.parse(node_path.read_text())):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == '_declare_if_not' and len(call.args) == 2):
            try:
                defaults[ast.literal_eval(call.args[0])] = ast.literal_eval(call.args[1])
            except (ValueError, TypeError):
                pass
    params = {**defaults, **captured['parameters']}
    proxy = SimpleNamespace(**params)
    proxy._camera_params = captured['resolved']['camera_params']
    proxy.optimizer_warm_start_shift_steps = max(1, round(1 / params['plan_rate'] / params['dt']))
    # Call the runtime's factory without constructing a ROS node.
    proxy.PLANNER_CLASS = lambda **kwargs: kwargs
    overrides = dict(horizon=params['global_horizon'], dt=params['global_dt'] or params['dt'],
                     use_ambiguity=params['global_use_ambiguity'],
                     optimizer_multistart=params['global_optimizer_multistart'])
    settings = UnicyclePlannerNode._build_planner_instance(
        proxy, lambda key: overrides.get(key, getattr(proxy, key)),
        lambda key, default: overrides.get(key, default), bool)
    captured['settings'] = settings
    captured['command'] = command
    return captured


class RecordedPlanner(UnicyclePlannerBase):
    """Record the same candidate evaluations used by the runtime selector."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attempts = []

    def _evaluate_candidate_controls(self, *args, **kwargs):
        result = super()._evaluate_candidate_controls(*args, **kwargs)
        record = {k: v for k, v in result.items() if k != 'controls_flat'}
        record['controls'] = result['controls_flat'].reshape(self.horizon, 2).tolist()
        self.attempts.append(record)
        return result


def run(args):
    if args.out.exists():
        raise RuntimeError('Choose a new output directory; preflights are immutable.')
    args.out.mkdir(parents=True)
    records = {}
    for arm in args.arms:
        captured = resolve(args.config, args.task, arm, args.seed)
        cfg, settings = captured['resolved'], captured['settings']
        state = np.array([cfg['spawn'][k] for k in ('x', 'y', 'yaw')], dtype=float)
        goal = np.array([cfg['goal_x'], cfg['goal_y']], dtype=float)
        # Explicit probe prior; live initialization must be recorded separately.
        P = np.diag([.05**2, .05**2, np.deg2rad(5)**2])
        seeds = generate_route_seeds(settings['driveable_geometry_json'], state[:2], goal)
        settings['optimizer_initial_routes_json'] = json.dumps(seeds)
        planner = RecordedPlanner(**settings)
        record = dict(state=state.tolist(), goal=goal.tolist(), initial_P=P.tolist(),
                      settings=settings, agent_parameters=captured['parameters'], seeds=seeds,
                      config_sha256=digest(args.config), launch_command=captured['command'])
        seed_results = []
        for route in planner.optimizer_initial_routes:
            controls = planner._controls_for_waypoints(state, route['waypoints'])
            result = planner.evaluate_rollout_controls(state, P, goal, controls)
            seed_results.append(dict(name=route['name'], **serializable(result)))
        record['seed_results'] = seed_results
        writejson(args.out / f'{arm}_preflight.json', record)
        print(arm, 'seeds', [(r['name'], r['rollout_valid'], r['terminal_goal_distance_pred'])
                            for r in seed_results], flush=True)
        if args.seeds_only:
            continue
        result = planner.plan(state, P, goal)
        J = [projection_jacobian(planner.camera.H, s) for s in result.states]
        record['result'] = serializable(asdict(result))
        record['maximum_chart_condition'] = float(max(np.linalg.cond(j) for j in J))
        record['attempts'] = planner.attempts
        writejson(args.out / f'{arm}_result.json', record)
        records[arm] = {k: record['result'][k] for k in (
            'rollout_valid', 'invalid_reason', 'selected_source', 'terminal_goal_distance_pred',
            'optimizer_success', 'solve_time_s', 'total_cost')}
        print(arm, json.dumps(records[arm]), flush=True)
    writejson(args.out / 'summary.json', dict(
        kind='offline_full_route_preflight_not_navigation', results=records,
        config=str(args.config.relative_to(REPO)), config_sha256=digest(args.config),
        sources={str(p.relative_to(REPO)): digest(p) for p in [Path(__file__),
            REPO / 'src/planning/planning/planners/base_planner.py',
            REPO / 'src/planning/planning/core/camera_network.py',
            REPO / 'src/unav_common/unav_common/lane_graph_routes.py',
            REPO / 'src/experiments/experiments/core/visibility_launch_common.py']}))


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'experiments/icra_commissioning/network_planner_pilot.yaml')
    parser.add_argument('--task', default='fusion_network_traverse')
    parser.add_argument('--arms', nargs='+', default=['P0', 'P1', 'P2'])
    parser.add_argument('--seed', type=int, default=210)
    parser.add_argument('--seeds-only', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.config, args.out = args.config.resolve(), args.out.resolve()
    run(args)
