"""Audit 14: evaluate assets/launches without executing ROS launch actions.

Writes only beside this script. No ROS node, simulator, network or experiment edits.
The independent geometry oracle uses Shapely polygons and full 3D rotations.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import xacro
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/fusion_on_fixed_routes'))
import aligned
from experiments.core.visibility_launch_common import parse_common_launch_config, resolve_world_setup, build_shared_nodes, build_agent_runtime_actions
from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy
from unav_common.robot_hull import VISUAL_HULL


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def text_value(value, context):
    if isinstance(value, str):
        return value
    return ''.join(v if isinstance(v, str) else v.perform(context) for v in value)


def declare(module, context):
    desc = module.generate_launch_description()
    for a in desc.entities:
        if isinstance(a, DeclareLaunchArgument):
            a.execute(context)
    return desc


def node_record(n, ctx):
    return {'package': text_value(n.node_package, ctx), 'executable': text_value(n.node_executable, ctx),
            'parameters': evaluate_parameters(ctx, n._Node__parameters),
            'remappings': [[text_value(a, ctx), text_value(b, ctx)] for a, b in n._Node__remappings]}


def pose(el):
    return np.array([float(x) for x in el.findtext('pose', '0 0 0 0 0 0').split()])


def transform(v):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler('xyz', v[3:]).as_matrix()
    out[:3, 3] = v[:3]
    return out


def robot_polygon(x, y, yaw):
    corners = np.array([[-.4, -.275], [.4, -.275], [.4, .275], [-.4, .275]])
    r = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return Polygon(corners @ r.T + [x, y])


def main():
    runner = load('audit14_runner', 'scripts/visibility_comparison/run_visibility_campaign.py')
    entry = load('audit14_entry', 'src/experiments/launch/warehouse_primary_comparison.launch.py')
    cfg0 = runner._load_config(ROOT / 'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml')
    args = runner._build_launch_cmd(cfg0, 'fusion_network_traverse', 'P0', 210, HERE / 'unused_log_destination')
    context = LaunchContext()
    context.launch_configurations.update(dict(x.split(':=', 1) for x in args if ':=' in x))
    declare(entry, context)
    cfg = resolve_world_setup(parse_common_launch_config(context))
    shared = build_shared_nodes(cfg)
    actions = build_agent_runtime_actions(cfg)
    bringup = load('audit14_bringup', 'src/sim/launch/bringup_sim.launch.py')
    gazebo = load('audit14_gazebo', 'src/sim/launch/gazebo.launch.py')
    robot = load('audit14_robot', 'src/sim/launch/robot_description.launch.py')
    sim_args = {k: text_value(v, context) for k, v in shared['bringup_sim'].launch_arguments}
    variants = {}
    for headless in ['true', 'false']:
        c = LaunchContext()
        c.launch_configurations.update({**sim_args, 'headless': headless})
        bdesc = declare(bringup, c)
        gdesc = declare(gazebo, c)
        rdesc = declare(robot, c)
        includes = [a for a in gdesc.entities if isinstance(a, IncludeLaunchDescription)]
        gz_args = {k: text_value(v, c) for k, v in includes[0].launch_arguments}
        env = {text_value(a.name, c): text_value(a.value, c) for a in gdesc.entities if isinstance(a, SetEnvironmentVariable)}
        rnode = next(a for a in rdesc.entities if isinstance(a, Node))
        description = evaluate_parameters(c, rnode._Node__parameters)[0]['robot_description']
        contact = bringup._make_contact_bridge(c)
        variants[headless] = {'arguments': gz_args, 'environment': env, 'robot_description_sha256': hashlib.sha256(description.encode()).hexdigest(),
                              'contact_bridge_count': len(contact), 'robot_model': c.launch_configurations['robot_model']}
    assert variants['true']['robot_description_sha256'] == variants['false']['robot_description_sha256']
    assert variants['true']['environment'] == variants['false']['environment']
    share = Path(get_package_share_directory('sim'))
    world_path = Path(cfg['world_path'])
    world = ET.parse(world_path).getroot().find('world')
    manifest = json.loads((ROOT / 'experiments/warehouse_v2_sketches/world_freeze_manifest.json').read_text())
    assert sha(world_path) == manifest['worlds']['A']['sha256']
    resources = variants['true']['environment']['GZ_SIM_RESOURCE_PATH'].split(':')
    model_records = []
    for incl in world.findall('include'):
        uri = incl.findtext('uri')
        if not uri or not uri.startswith('model://'):
            continue
        name = uri[8:]
        candidates = [Path(p) / name / 'model.sdf' for p in resources]
        found = [p for p in candidates if p.is_file()]
        assert found, uri
        mpath = found[0]
        model_records.append({'name': incl.findtext('name'), 'uri': uri, 'pose': pose(incl).tolist(),
                              'path': str(mpath), 'realpath': str(mpath.resolve()), 'sha256': sha(mpath),
                              'sensors': [{'name': s.get('name'), 'type': s.get('type'), 'pose': pose(s).tolist(),
                                           'topic': s.findtext('topic'), 'rate_hz': s.findtext('update_rate'), 'always_on': s.findtext('always_on'),
                                           'width': s.findtext('camera/image/width'), 'height': s.findtext('camera/image/height'),
                                           'hfov_rad': s.findtext('camera/horizontal_fov')} for s in ET.parse(mpath).getroot().iter('sensor')]})
    # Every native model:// reference, including material/mesh dependencies.
    deps = {}
    payloads = [world_path] + [Path(m['path']) for m in model_records]
    import re
    for p in payloads:
        for uri in re.findall(r'model://[^<\s"\']+', p.read_text()):
            ref = uri[8:]
            found = [Path(base) / ref for base in resources if (Path(base) / ref).exists()]
            deps[uri] = {'resolved': str(found[0].resolve()) if found else None,
                         'sha256': sha(found[0]) if found and found[0].is_file() else None}
    scene = scene_from_json(cfg['collision_geometry_json'])
    actual = {}
    contacts = []
    for m in world.findall('model'):
        for link in m.findall('link'):
            for s in link.findall("sensor[@type='contact']"):
                contacts.append({'model': m.get('name'), 'link': link.get('name'), 'sensor': s.get('name'),
                                 'collisions': [x.text for x in s.findall('contact/collision')], 'rate': s.findtext('update_rate'),
                                 'explicit_topic': s.findtext('topic')})
            for coll in link.findall('collision'):
                if coll.find('geometry/box') is None:
                    continue
                size = np.array([float(x) for x in coll.findtext('geometry/box/size').split()])
                t = transform(pose(m)) @ transform(pose(link)) @ transform(pose(coll))
                corners = np.array([[a, b, c, 1.] for a in [-size[0]/2, size[0]/2] for b in [-size[1]/2, size[1]/2] for c in [-size[2]/2, size[2]/2]]) @ t.T
                name = f"{m.get('name')}/{link.get('name')}:{coll.get('name')}"
                actual[name] = {'bounds': [*corners[:, :3].min(0), *corners[:, :3].max(0)], 'polygon': Polygon(corners[:, :2]).convex_hull}
    assert set(actual) == {p.name for p in scene.prisms}
    discrepancies = []
    for p in scene.prisms:
        bounds = np.array([p.xmin, p.ymin, p.zmin, p.xmax, p.ymax, p.zmax])
        discrepancies.append(np.max(abs(bounds - actual[p.name]['bounds'])))
    assert max(discrepancies) < 1e-12
    collision_union = unary_union([r['polygon'] for r in actual.values()])
    lanes = scene_from_json(cfg['driveable_geometry_json'])
    lane_union = unary_union([box(p.xmin, p.ymin, p.xmax, p.ymax) for p in lanes.prisms])
    spawn = cfg['spawn']
    footprint = robot_polygon(spawn['x'], spawn['y'], spawn['yaw'])
    sp = Point(spawn['x'], spawn['y'])
    spawn_results = {'pose': spawn, 'centre_inside_driveable': lane_union.covers(sp), 'body_inside_driveable': lane_union.covers(footprint),
                     'exact_body_clearance_m': footprint.distance(collision_union), 'circle_clearance_m': sp.distance(collision_union)-cfg['robot_collision_radius_m'],
                     'body_intersects': footprint.intersects(collision_union), 'lane_centre_clearance_m': sp.distance(lane_union.boundary)}
    assert spawn_results['centre_inside_driveable'] and not spawn_results['body_intersects']
    # Full in-place turn inside the selected start corridor, measured independently.
    turn = [{'yaw': float(a), 'body_clearance_m': robot_polygon(sp.x, sp.y, a).distance(collision_union),
             'intersects': robot_polygon(sp.x, sp.y, a).intersects(collision_union)} for a in np.linspace(0, 2*np.pi, 361)]
    assert not any(x['intersects'] for x in turn)
    # Narrow straight lane: exact rectangle fits; circumscribed disk may reject.
    narrow = box(-5., -.45, 5., .45)
    tight = [{'yaw': float(a), 'rect_fits_0_9m_lane': narrow.covers(robot_polygon(0, 0, a)),
              'lateral_half_extent_m': .4*abs(math.sin(a))+.275*abs(math.cos(a))} for a in [0., math.pi/4, math.atan2(.4, .275), math.pi/2]]
    # Active public robot TF has two parent edges for base_link.
    urdf = ET.fromstring(xacro.process_file(str(share / 'robot_description/urdf/warehouse_amr.urdf.xacro'), mappings={'use_lidar':'false'}).toxml())
    base_joint = urdf.find("joint[@name='base_joint']")
    diff = next(p for p in urdf.findall('gazebo/plugin') if 'DiffDrive' in p.get('name',''))
    tf = {'rsp_parent': base_joint.find('parent').get('link'), 'rsp_child': base_joint.find('child').get('link'),
          'diff_parent': diff.findtext('frame_id'), 'diff_child': diff.findtext('child_frame_id'),
          'rsp_origin': base_joint.find('origin').get('xyz'), 'joint_state_publisher_plugin_present': any('JointStatePublisher' in p.get('name','') for p in urdf.findall('gazebo/plugin'))}
    assert tf['rsp_child'] == tf['diff_child'] and tf['rsp_parent'] != tf['diff_parent']
    # The camera's configured SDF +X viewing axis is not its public TF +Z axis.
    cam = next(x for x in model_records if x['name']=='external_camera')
    rot = Rotation.from_euler('xyz', cam['pose'][3:]).as_matrix()
    tf['camera_A_sdf_forward_world'] = rot[:, 0].tolist()
    tf['camera_A_tf_z_world'] = rot[:, 2].tolist()
    tf['optical_axis_disagreement_deg'] = float(np.degrees(np.arccos(np.clip(rot[:, 0]@rot[:, 2], -1, 1))))
    native_files = ['src/sim/launch/bringup_sim.launch.py','src/sim/launch/gazebo.launch.py','src/sim/launch/robot_description.launch.py',
                    'src/sim/sim/reset_world.py','src/sim/sim/wait_for_clock.py','src/sim/sim/wait_for_odom.py','src/sim/sim/clock_throttle_node.py',
                    'src/sim/sim/encoder_noise_node.py','src/sim/sim/actuation_noise_node.py','src/sim/robot_description/urdf/warehouse_amr.urdf.xacro',
                    'src/experiments/experiments/core/visibility_launch_common.py','src/experiments/experiments/core/world_profiles.py',
                    'src/experiments/experiments/nodes/experiment_logger.py','src/unav_common/unav_common/geometry.py','src/unav_common/unav_common/robot_hull.py',
                    'src/unav_common/unav_common/occlusion_geometry.py','src/experiments/config/world_profiles.yaml','src/experiments/config/tasks.yaml']
    sources = {p: sha(ROOT/p) for p in native_files}
    selected_runs = []
    for campaign_name, arm in [('runtime','P0'),('runtime','P1'),('tracking','P1')]:
        selection_path = ROOT / f'logs/studies/icra_commissioning_20260905/network_navigation_{campaign_name}_evidence/selection.json'
        selection = json.loads(selection_path.read_text())
        rec = next(v for v in selection['runs'] if v['arm']==arm)
        run = ROOT / rec['run']
        for name, digest in rec['files'].items():
            assert sha(run/name) == digest, (run, name)
        table = aligned.rows(run)
        summary = json.loads((run/'run_summary.json').read_text())
        rm = json.loads((run/'run_manifest.json').read_text())
        logged_scene = scene_from_json(rm['collision_geometry_json'])
        assert [p.to_dict() for p in logged_scene.prisms] == [p.to_dict() for p in scene.prisms]
        relevant = ['stamp','odom_stamp','odom_map_stamp','odom_noisy_stamp','planner_belief_stamp','gt_stamp','gt_available','contact_topic_publishers',
                    'contact_messages_seen','collision_contact','collision_geom','collision_reason','first_crash_stamp','cmd_v','cmd_w']
        nonzero = [x for x in table if abs(float(x.get('cmd_v') or 0))>1e-4 or abs(float(x.get('cmd_w') or 0))>1e-4]
        selected_runs.append({'campaign':campaign_name,'arm':arm,'run':rec['run'],'selection_sha256':sha(selection_path),'verified_files':rec['files'],
                              'summary':{k:v for k,v in summary.items() if any(t in k for t in ['contact','collision','crash','gt_stamp']) or k in ['completion_reason','valid_run','termination_reference']},
                              'first_row':{k:table[0].get(k) for k in relevant}, 'first_command_row':{k:nonzero[0].get(k) for k in relevant},
                              'last_row':{k:table[-1].get(k) for k in relevant}, 'row_count':len(table),
                              'manifest_has_world_hash': 'world_sha256' in rm, 'manifest_has_robot_hash':'robot_description_sha256' in rm})
    result = {'scope':'Read-only launch expansion, synthetic geometry, exact manifest-selected log evidence; no accuracy rescore',
              'sources': sources, 'sim_share':str(share), 'world_path':str(world_path), 'world_realpath':str(world_path.resolve()), 'world_sha256':sha(world_path),
              'freeze_id':manifest['freeze_id'],'profile_freeze_id':cfg['profile']['geometry_freeze_id'], 'frozen_source_keys': list(manifest['sources']),
              'launch_command':args,'sim_launch_arguments':sim_args,'launch_variants':variants,'models':model_records,'model_uri_dependencies':deps,
              'resolved_nodes': [node_record(n, context) for n in actions if isinstance(n, Node)],
              'shared_nodes':{k:node_record(n, context) for k,n in shared.items() if isinstance(n,Node) and k in ['wait_for_odom','logger_node','encoder_noise_node','command_noise_node']},
              'contacts':contacts,'collision_prism_count':len(scene.prisms),'max_collision_bound_difference_m':max(discrepancies),
              'driveable_geometry':json.loads(cfg['driveable_geometry_json']), 'spawn':spawn_results,'start_full_turn_min_clearance_m': min(x['body_clearance_m'] for x in turn),
              'tight_turn_rectangles':tight,'physical_body_circumradius_m':math.hypot(.4,.275),'visual_hull_bounds_m':[VISUAL_HULL.min(0).tolist(),VISUAL_HULL.max(0).tolist()],
              'tf_contract':tf,'selected_runs':selected_runs}
    (HERE/'asset_results.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps({k:result[k] for k in ['world_realpath','world_sha256','collision_prism_count','max_collision_bound_difference_m','spawn','start_full_turn_min_clearance_m','tf_contract']}, indent=2))


if __name__ == '__main__':
    main()
