#!/usr/bin/env python3
"""Audit 13: isolated defect reproductions. No ROS graph or real campaign processes.

Run from repository root after sourcing install/setup.bash. All mutable fixtures
live in TemporaryDirectory; outputs are restricted to this audit's own files.
Assertions pin observed behavior, including bugs, rather than endorse it.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'scripts/visibility_comparison/run_visibility_campaign.py'
EXECUTED_SOURCES = {}
AMBIENT_IMPORTS = {}
for import_name in ('experiments.core.visibility_launch_common','experiments.core.manifest','planning.nodes.efe_agent_node',
                    'planning.nodes.unicycle_planner_node','perception.nodes.batched_four_camera_yolo_node',
                    'reliability.nodes.camera_manager_node','sim.actuation_noise_node','unav_common.manifest'):
    import_spec=importlib.util.find_spec(import_name)
    import_path=Path(import_spec.origin)
    AMBIENT_IMPORTS[import_name]=dict(import_path=str(import_path),resolved=str(import_path.resolve()),
                                    sha256=hashlib.sha256(import_path.read_bytes()).hexdigest())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    path=Path(path)
    source=path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    # Compile exactly the bytes hashed here, even if another chat edits the path.
    exec(compile(source,str(path),'exec'),obj.__dict__)
    rel=path.resolve().relative_to(ROOT)
    digest=hashlib.sha256(source).hexdigest()
    snapshot=OUT/'13_source_snapshot'/digest[:12]/rel
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():assert snapshot.read_bytes()==source
    else:snapshot.write_bytes(source)
    EXECUTED_SOURCES[name]=dict(path=str(rel),sha256=digest,snapshot=str(snapshot.relative_to(ROOT)))
    return obj


c = module('audit13_campaign', SOURCE)
from experiments.core import manifest as em
from unav_common import manifest as um
import experiments.core
lc=module('experiments.core.visibility_launch_common',ROOT/'src/experiments/experiments/core/visibility_launch_common.py')
experiments.core.visibility_launch_common=lc
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions

primary = module('audit13_primary', ROOT / 'src/experiments/launch/warehouse_primary_comparison.launch.py')
RESULTS = {}


def record(name, value):
    RESULTS[name] = value
    print(name + ': ' + json.dumps(value, default=str)[:600])


def fixture(tmp):
    detector = tmp / 'detector.pt'
    detector.write_bytes(b'detector version A')
    return dict(world='warehouse_v2.world.sdf', launch_file='warehouse_primary_comparison.launch.py',
                conditions={'C0': {}}, tasks={'fusion_network_traverse': {'conditions': ['C0'], 'seeds': [210]}},
                yolo_model=str(detector), horizon=20, dt=.25, goal_success_radius=.35,
                run_timeout_after_first_cmd_s=600, cleanup_mode='isolated', ros_domain_id_base=211)


def command(cfg, tmp, cond='C0'):
    return c._build_launch_cmd(cfg, 'fusion_network_traverse', cond, 210, tmp / 'runs')


def argsmap(cmd):
    return dict(a.split(':=', 1) for a in cmd[4:] if ':=' in a)


def provenance(cfg, path):
    cfg = deepcopy(cfg)
    cfg.update(_campaign_config_path=str(path), _campaign_config_sha256=sha(path),
               _yolo_model_sha256=sha(cfg['yolo_model']),
               _git_provenance={k: 'fixture-' + k for k in ('git_sha', 'git_status_sha256', 'git_diff_sha256', 'git_untracked_content_sha256')})
    return cfg


def resume_fixture(cfg, tmp):
    p = tmp / 'resume'
    p.mkdir(exist_ok=True)
    m = {k: v for k, v in cfg.items() if not k.startswith('_') and not isinstance(v, dict)}
    m.update(logging_schema_version=7, goal_termination_reference='planner_belief',
             yolo_model_sha256=cfg['_yolo_model_sha256'], campaign_config_sha256=cfg['_campaign_config_sha256'],
             **cfg['_git_provenance'])
    (p / 'run_manifest.json').write_text(json.dumps(m))
    entry = dict(task='fusion_network_traverse', condition='C0', seed=210, run_dir=str(p), outcome='goal_reached')
    return p, m, entry


def fake_evidence(p, summary):
    p.mkdir(parents=True, exist_ok=True)
    (p / 'run_summary.json').write_text(json.dumps(summary))
    (p / 'fusion_observations.csv').write_text('source_batch_id\nbatch-1\n')
    (p / 'correction_assimilations.csv').write_text('source_batch_id,status,reason\nbatch-1,dropped,supported policy refusal\n')


def dry_launch(cmd):
    """Resolve the real launch functions; replace all runtime action constructors."""
    ctx = LaunchContext()
    ctx.launch_configurations.update(argsmap(cmd))
    declarations = {}
    for item in primary.generate_launch_description().entities:
        if isinstance(item, DeclareLaunchArgument):
            declarations[item.name] = perform_substitutions(ctx, item.default_value or [])
            item.execute(ctx)
    captured = []
    resolved = {}
    def action(kind):
        def make(*args, **kwargs):
            if 'launch_arguments' in kwargs:
                kwargs['launch_arguments']=dict(kwargs['launch_arguments'])
            item = dict(kind=kind, args=args, **kwargs)
            captured.append(item)
            return item
        return make
    build = lc.build_agent_runtime_actions
    def capture_cfg(cfg):
        resolved.update(deepcopy(cfg))
        return build(cfg)
    with ExitStack() as stack:
        for name in ('Node', 'IncludeLaunchDescription', 'RegisterEventHandler', 'OnProcessExit', 'Shutdown'):
            stack.enter_context(patch.object(lc, name, action(name)))
        stack.enter_context(patch.object(lc, 'build_agent_runtime_actions', capture_cfg))
        active_actions = primary._launch_setup(ctx)
    reachable = set()
    def walk(value):
        if isinstance(value, dict):
            if value.get('kind') == 'Node':
                reachable.add(value.get('name', value.get('executable', '?')))
            for v in value.values(): walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value: walk(v)
    walk(active_actions)
    nodes = {v.get('name', v.get('executable', '?')): v for v in captured if v['kind'] == 'Node'}
    for node in nodes.values():
        node['parameters'] = {k: v for d in node.get('parameters', []) for k, v in d.items()}
        node['scheduled_for_execution'] = node.get('name', node.get('executable', '?')) in reachable
    return dict(context=dict(ctx.launch_configurations), declarations=declarations, resolved=resolved, nodes=nodes,
                includes=[v for v in captured if v['kind']=='IncludeLaunchDescription'])


def run_fake_main(cfg, path, logroot, *, rc=17, spawn_error=False, recorder_error=False):
    path.write_text(yaml.safe_dump(cfg))
    calls = []
    def popen(*a, **kw):
        calls.append(a[0])
        if spawn_error or (recorder_error and len(calls) == 2):
            raise OSError('audit synthetic spawn failure')
        return SimpleNamespace(pid=4321, poll=lambda: rc)
    cleanup = []
    text = io.StringIO()
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'argv', [str(SOURCE), '--config', str(path), '--log-root', str(logroot)]))
        stack.enter_context(patch.object(c.subprocess, 'Popen', popen))
        stack.enter_context(patch.object(c, 'git_provenance', lambda root: {}))
        stack.enter_context(patch.object(c.os, 'getpgid', lambda pid: pid))
        stack.enter_context(patch.object(c, '_terminate_process_group', lambda *a, **k: cleanup.append('group')))
        stack.enter_context(patch.object(c, '_cleanup_owned_run', lambda *a, **k: cleanup.append('token')))
        stack.enter_context(patch.object(c, '_force_fresh', lambda *a, **k: cleanup.append('global')))
        stack.enter_context(patch.dict(os.environ, {'CAMPAIGN_RECORD_CAMERA': '1'} if recorder_error else {}, clear=False))
        if not recorder_error:
            os.environ.pop('CAMPAIGN_RECORD_CAMERA', None)
        try:
            with redirect_stdout(text):
                result = c.main()
        except Exception as exc:
            result = type(exc).__name__ + ': ' + str(exc)
    ledger = c._load_run_log(logroot / 'campaign_log.json')
    return dict(returned=result, ledger=ledger, cleanup=cleanup, process_calls=len(calls))


def main():
    source_paths = [SOURCE, ROOT/'scripts/visibility_comparison/common.py',
        ROOT/'scripts/visibility_comparison/monitor_campaign.py',
        Path(lc.__file__).resolve(), Path(em.__file__).resolve(), Path(um.__file__).resolve(),
        ROOT/'src/experiments/experiments/core/tasks.py', ROOT/'src/experiments/experiments/core/world_profiles.py',
        ROOT/'src/experiments/experiments/nodes/experiment_logger.py',
        ROOT/'src/perception/perception/nodes/batched_four_camera_yolo_node.py',
        ROOT/'src/reliability/reliability/nodes/camera_manager_node.py',
        ROOT/'src/reliability/reliability/reference_calibration.py',
        ROOT/'src/planning/planning/nodes/unicycle_planner_node.py',
        ROOT/'src/planning/planning/nodes/efe_agent_node.py',
        ROOT/'src/sim/sim/actuation_noise_node.py',
        ROOT/'experiments/icra_commissioning/network_navigation_analysis.py']
    source_paths += sorted((ROOT/'src/experiments/launch').glob('*.py'))
    source_paths += sorted((ROOT/'src/sim/launch').glob('*.py'))
    source_paths += sorted((ROOT/'src').glob('*/setup.*'))
    hashes_before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    with tempfile.TemporaryDirectory(prefix='unav-audit13-') as directory:
        tmp = Path(directory)
        os.environ['ROS_LOG_DIR'] = str(tmp / 'launch_logs')
        cfg = fixture(tmp)
        config_path = tmp / 'campaign.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        # Schemas, typo keys, false and zero, independent override paths.
        for tier in ('global', 'task', 'condition', 'route'):
            bad = deepcopy(cfg)
            target = {'global': bad, 'task': bad['tasks']['fusion_network_traverse'], 'condition': bad['conditions']['C0']}
            if tier == 'route':
                bad['tasks']['fusion_network_traverse']['preselected_routes'] = {'C0': {'pixel_timout_s': 9}}
            else:
                target[tier]['pixel_timout_s'] = 9
            c._validate_config(bad, config_path)
            assert 'pixel_timout_s' not in argsmap(command(bad, tmp))
            record('typo_' + tier, 'accepted and omitted from launch')
        for key, val in {'tasks_yaml': str(tmp/'custom_tasks.yaml'), 'world_profiles': str(tmp/'custom_world.yaml'),
                         'state_correction_ekf': False, 'pixel_correction_nis_threshold': 0,
                         'pixel_timeout_s': None, 'manager_covariance_profile': '',
                         'enable_mission': False, 'yolo_predict_conf_floor': .2,
                         'simple_tracker_yaw_gate_rad': .1}.items():
            modified = deepcopy(cfg); modified[key] = val
            c._validate_config(modified, config_path)
            record('forward_' + key, dict(declared=val, argument=argsmap(command(modified,tmp)).get(key, '<omitted>')))
        precedence = deepcopy(cfg)
        precedence.update(v_max=.22, use_encoder_noise=True, pixel_timeout_s=.5)
        precedence['tasks']['fusion_network_traverse'].update(v_max=.3, use_encoder_noise=False, pixel_timeout_s=.6)
        precedence['conditions']['C0'].update(v_max=.4, use_encoder_noise=False, pixel_timeout_s=.7)
        precedence['tasks']['fusion_network_traverse']['preselected_routes']={'C0': {'v_max': 0, 'use_encoder_noise': False, 'pixel_timeout_s': 0}}
        built = argsmap(command(precedence,tmp))
        record('precedence', {k:dict(resolver=c._effective_value(precedence,'fusion_network_traverse','C0',k),launch=built.get(k)) for k in ('v_max','use_encoder_noise','pixel_timeout_s')})
        assert built['v_max']=='.22' or built['v_max']=='0.22'
        assert built['use_encoder_noise']=='true' and built['pixel_timeout_s']=='0'
        duplicate = deepcopy(cfg); duplicate['tasks']['fusion_network_traverse']['seeds']=[1,'1',1]
        c._validate_config(duplicate,config_path)
        keys=[c._run_key(*r) for r in c._build_run_matrix(duplicate)]
        assert len(set(keys))==1
        record('duplicate_seed_identity',keys)
        invalid_task = deepcopy(cfg)
        invalid_task['tasks']={'does_not_exist': {'conditions':['C0'], 'seeds':[210]}}
        c._validate_config(invalid_task,config_path)
        record('unknown_task_dry_validation','accepted by campaign validation')
        raw=config_path.read_text()+'\ndt: 0.5\n'
        config_path.write_text(raw)
        assert c._load_config(config_path)['dt']==.5
        record('duplicate_yaml_key','duplicate dt silently keeps last value 0.5')
        config_path.write_text(yaml.safe_dump(cfg))
        route=deepcopy(cfg);route.update(global_planner_mode='preselected_route',use_hierarchical=True,driveable_geometry_json='fixture',
                                      preselected_route_endpoint_tolerance_m=0,preselected_route_sample_step_m=0,preselected_route_clearance_m=0)
        route['tasks']['fusion_network_traverse']['preselected_routes']={'C0':dict(preselected_route_json='[[0,0],[1,0]]',preselected_route_sha256='fixture',preselected_route_source_path=str(config_path),preselected_route_source_sha256='fixture')}
        with patch.object(c,'_task_spec_from_yaml',return_value=dict(start=dict(x=0,y=0),goal=dict(x=1,y=0))),patch.object(c,'validate_preselected_route') as validator:
            c._validate_config(route,config_path)
            received=validator.call_args.kwargs
        record('zero_route_validation_fallback',{k:received[k] for k in ('endpoint_tolerance_m','sample_step_m','declared_clearance_m')})
        assert received['sample_step_m']==.04 and received['endpoint_tolerance_m']==.25

        # Reuse validation intentionally starts from a passing manifest fixture.
        bound = provenance(cfg,config_path)
        rp, manifest, entry = resume_fixture(bound,tmp)
        assert c._existing_entry_matches_config(entry,bound)==(True,'')
        record('resume_no_summary_or_csv', c._existing_entry_matches_config(entry,bound))
        for state in ('partial','malformed','invalid','failed'):
            payload={'partial':'{"completed":', 'malformed':'[]', 'invalid':json.dumps({'completed':True,'valid_run':False}),
                     'failed':json.dumps({'completed':False,'completion_reason':'interrupted'})}[state]
            (rp/'run_summary.json').write_text(payload)
            value=c._existing_entry_matches_config(entry,bound)
            assert value[0]
            record('resume_' + state,value)
        wrong=dict(manifest,task='other_task',seed=999,world='other.world.sdf',planner='wrong_planner')
        (rp/'run_manifest.json').write_text(json.dumps(wrong))
        record('resume_wrong_cell_identity', c._existing_entry_matches_config(entry,bound))
        assert c._existing_entry_matches_config(entry,bound)[0]
        (rp/'run_manifest.json').write_text(json.dumps(manifest))
        quoted=deepcopy(bound);quoted['use_command_noise']='false'
        off=dict(manifest,use_command_noise=False);(rp/'run_manifest.json').write_text(json.dumps(off))
        refused=c._existing_entry_matches_config(entry,quoted)
        on=dict(manifest,use_command_noise=True);(rp/'run_manifest.json').write_text(json.dumps(on))
        allowed=c._existing_entry_matches_config(entry,quoted)
        assert not refused[0] and allowed[0]
        record('quoted_false_resume',dict(correct_false_manifest=refused,incorrect_true_manifest=allowed))
        (rp/'run_manifest.json').write_text(json.dumps(manifest))
        Path(bound['yolo_model']).write_bytes(b'detector replaced B')
        assert c._existing_entry_matches_config(entry,bound)[0]
        refreshed=deepcopy(bound);refreshed['_yolo_model_sha256']=sha(bound['yolo_model'])
        assert not c._existing_entry_matches_config(entry,refreshed)[0]
        record('model_replacement_cached',dict(stale_cache=c._existing_entry_matches_config(entry,bound),fresh_process=c._existing_entry_matches_config(entry,refreshed)))
        nn=tmp/'mean.joblib'; nn.write_bytes(b'mean A')
        nncfg=deepcopy(bound);nncfg['manager_learned_correction_path']=str(nn)
        manifest['manager_learned_correction_path']=str(nn)
        (rp/'run_manifest.json').write_text(json.dumps(manifest))
        nn.write_bytes(b'mean B')
        assert c._existing_entry_matches_config(entry,nncfg)[0]
        record('resume_learned_mean_replaced',c._existing_entry_matches_config(entry,nncfg))
        legacy=deepcopy(bound); gp=tmp/'legacy.npz';gp.write_bytes(b'GP A');legacy['gp_artifact']=str(gp)
        legacy['conditions']={'C2':{}};legacy['tasks']['fusion_network_traverse']['conditions']=['C2']
        legentry=dict(entry,condition='C2'); manifest['visibility_artifact_path']=str(gp)
        manifest['visibility_artifact_sha256']=sha(gp);(rp/'run_manifest.json').write_text(json.dumps(manifest));gp.write_bytes(b'GP B')
        assert c._existing_entry_matches_config(legentry,legacy)[0]
        record('resume_legacy_gp_replaced',c._existing_entry_matches_config(legentry,legacy))
        net=tmp/'network.npz';net.write_bytes(b'network A')
        network=deepcopy(legacy);network.pop('gp_artifact');network['conditions']['C2']['camera_network_artifact_path']=str(net)
        manifest.update(camera_network_artifact_path=str(net),camera_network_artifact_sha256=sha(net),visibility_artifact_path='')
        (rp/'run_manifest.json').write_text(json.dumps(manifest));net.write_bytes(b'network B')
        assert not c._existing_entry_matches_config(legentry,network)[0]
        record('network_replacement_rejected',c._existing_entry_matches_config(legentry,network))
        # Fresh completion + stale attempt picking, all process boundaries faked.
        for name, summary in {'incomplete_goal':{'completed':False,'completion_reason':'goal_reached'},
            'invalid_goal':{'completed':True,'valid_run':False,'completion_reason':'goal_reached'},
            'interrupted':{'completed':False,'completion_reason':'interrupted'},
            'empty_summary':{}}.items():
            logroot=tmp/name
            p=logroot/'fusion_network_traverse/C0/seed210/experiment_20000101_000000'
            fake_evidence(p,summary)
            (p.parent/'experiment_20990101_000000').mkdir()
            result=run_fake_main(cfg,config_path,logroot)
            event=next(iter(result['ledger'].values()))
            record('fresh_' + name,dict(outcome=event['outcome'],completion_reason=event['completion_reason'],selected=Path(event['run_dir']).name,returncode_ignored=17,manifest_exists=(p/'run_manifest.json').exists()))
            assert event['outcome']!='infra_invalid' and event['run_dir']==str(p)
        for name, kw in [('spawn_failure', {'spawn_error':True}),('recorder_failure',{'recorder_error':True})]:
            result=run_fake_main(cfg,config_path,tmp/name,**kw)
            event=next(iter(result['ledger'].values()))
            assert not event.get('finished_at') and result['cleanup']==[]
            record(name,dict(error=result['returned'],outcome=event['outcome'],terminal_reason=event['completion_reason'],cleanup=result['cleanup'],process_calls=result['process_calls']))

        # Existing interrupted atomic replacement repair versus multiwriter durability.
        ledger=tmp/'ledger.json';old={'old':{'outcome':'goal_reached'}}
        c._save_run_log(ledger,old)
        with patch.object(c.os,'replace',side_effect=OSError('interrupt before rename')):
            try:c._save_run_log(ledger,{'replacement':{}})
            except OSError:pass
        assert c._load_run_log(ledger)==old
        record('ledger_interrupted_replace','old ledger preserved')
        a=c._load_run_log(ledger);b=c._load_run_log(ledger)
        a['completed_A']={'outcome':'goal_reached'};b['completed_B']={'outcome':'goal_reached'}
        c._save_run_log(ledger,a);c._save_run_log(ledger,b)
        assert 'completed_A' not in c._load_run_log(ledger)
        record('ledger_concurrent_writers',c._load_run_log(ledger))
        ledger.write_text('{"old":')
        assert c._load_run_log(ledger)=={}
        record('ledger_malformed_resume','malformed existing ledger becomes empty dictionary')
        seen=[]
        real_fsync=c.os.fsync
        def fsync(fd):
            import stat
            seen.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file');real_fsync(fd)
        with patch.object(c.os,'fsync',fsync):c._save_run_log(ledger,old)
        assert seen==['file']
        record('ledger_fsync_targets',seen)

        # Snapshot collisions and same-second identifiers.
        for part, value in [('a','source A'),('b','source B')]:
            (tmp/part).mkdir();(tmp/part/'same.yaml').write_text(value)
        snaps=em.snapshot_configs(str(tmp/'snapshots'),[str(tmp/'a/same.yaml'),str(tmp/'b/same.yaml')])
        assert len(snaps)==1 and Path(snaps['same.yaml']).read_text()=='source B'
        record('duplicate_config_basename',dict(snapshot_keys=list(snaps),retained=Path(snaps['same.yaml']).read_text()))
        with patch.object(um,'datetime') as date:
            date.now.return_value=datetime(2026,9,6,1,2,3)
            x=em.create_run_dir(str(tmp/'ids'));y=em.create_run_dir(str(tmp/'ids'))
        assert x==y
        record('same_second_run_ids',x)
        unsafe=tmp/'manifest_write';um.write_manifest(str(unsafe),{'complete':'old value'})
        def broken_dump(data,stream,**kwargs):
            stream.write('{"partial":');raise OSError('audit interrupted serialization')
        with patch.object(um.json,'dump',broken_dump):
            try:um.write_manifest(str(unsafe),{'new':'value'})
            except OSError:pass
        record('manifest_interrupted_rewrite',(unsafe/'run_manifest.json').read_text())
        assert (unsafe/'run_manifest.json').read_text()=='{"partial":'

        # Real git in an isolated fixture: tracked/untracked hashing and cached rewrites.
        gr=tmp/'git';gr.mkdir()
        def git(*args):
            return subprocess.run(['git','-C',str(gr),*args],check=True,capture_output=True,text=True)
        git('init','-q');git('config','user.name','Audit fixture');git('config','user.email','fixture@example.invalid')
        (gr/'tracked.py').write_text('value=1\n');git('add','tracked.py');git('commit','-qm','fixture')
        (gr/'tracked.py').write_text('value=2\n');(gr/'untracked.py').write_text('value=10\n')
        first=um.git_provenance(str(gr));em.write_manifest(str(tmp/'cached1'),{},str(gr))
        (gr/'tracked.py').write_text('value=3\n');(gr/'untracked.py').write_text('value=11\n')
        second=um.git_provenance(str(gr));em.write_manifest(str(tmp/'cached2'),{},str(gr))
        cached=json.loads((tmp/'cached2/run_manifest.json').read_text())
        assert first['git_diff_sha256']!=second['git_diff_sha256']
        assert first['git_untracked_content_sha256']!=second['git_untracked_content_sha256']
        assert cached==first
        record('dirty_source_and_cache',dict(fresh_tracked_change_detected=True,fresh_untracked_change_detected=True,second_manifest_keeps_first_hashes=True))
        installed=tmp/'installed';installed.mkdir();(installed/'audit13_runtime.py').write_text('VALUE="stale installed"\n')
        (gr/'audit13_runtime.py').write_text('VALUE="intended source"\n')
        before=um.git_provenance(str(gr))
        script='import audit13_runtime,json;print(json.dumps(dict(value=audit13_runtime.VALUE,path=audit13_runtime.__file__)))'
        imported=json.loads(subprocess.run([sys.executable,'-c',script],cwd=str(tmp),env=dict(os.environ,PYTHONPATH=str(installed)),check=True,capture_output=True,text=True).stdout)
        (installed/'audit13_runtime.py').write_text('VALUE="another installed version"\n')
        after=um.git_provenance(str(gr))
        assert imported['value']=='stale installed' and before==after
        record('installed_source_attestation_gap',dict(imported=imported,git_provenance_unchanged_when_installed_code_changes=True))
        from reliability.reference_calibration import ReferenceCalibration
        mean=tmp/'mean_checkpoint';mean.write_bytes(b'constant fixture identity')
        calpath=tmp/'reference.json'
        caldata=dict(schema='camera_reference_calibration.v1',frame='map_bev',reference='robot_ground_reference_xy',
            covariance_units='m2',mean_order='bbox_feature_nn_then_subtract_bias',mean_checkpoint_sha256=sha(mean),
            cameras={'camera_A':dict(bias_m=[0,0],R_m2=[[1,0],[0,1]])})
        calpath.write_text(json.dumps(caldata));calhash=sha(calpath)
        newdata=deepcopy(caldata);newdata['cameras']['camera_A']['R_m2']=[[2,0],[0,2]]
        read_text=Path.read_text
        def swap_read(path,*args,**kwargs):
            if path==calpath:path.write_text(json.dumps(newdata))
            return read_text(path,*args,**kwargs)
        with patch.object(Path,'read_text',swap_read):cal=ReferenceCalibration(calpath,mean,['camera_A'])
        assert cal.sha256==calhash and cal.covariance['camera_A'][0,0]==2
        record('calibration_hash_load_race',dict(recorded_hash_matches_loaded_file=cal.sha256==sha(calpath),loaded_variance=float(cal.covariance['camera_A'][0,0]),variance_of_recorded_bytes=1))

        # Simultaneous tokens protect killing, but do not lease the ROS domain.
        proc=tmp/'proc';proc.mkdir()
        for pid,token in [(8101,'ours'),(8102,'ours_other'),(8103,'theirs')]:
            (proc/str(pid)).mkdir();(proc/str(pid)/'environ').write_bytes(f'UNAV_CAMPAIGN_RUN_TOKEN={token}\0ROS_DOMAIN_ID=211\0'.encode())
        assert c._pids_with_run_token('ours',proc)==[8101]
        record('simultaneous_tokens',dict(ours=c._pids_with_run_token('ours',proc),theirs=c._pids_with_run_token('theirs',proc),campaign_A_domain=c._ros_domain_for_run(cfg,0),campaign_B_domain=c._ros_domain_for_run(deepcopy(cfg),0)))
        with patch.object(sys,'argv',[str(SOURCE),'--task-name','wrong']),redirect_stderr(io.StringIO()):
            try:c.main()
            except SystemExit as exc:record('invalid_runner_cli_exit',exc.code)

        # Actual launch parameter resolution, without executing Nodes or event handlers.
        active_path=ROOT/'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml'
        active=c._load_config(active_path);active['_campaign_config_path']=str(active_path)
        cmd=c._build_launch_cmd(active,'fusion_network_traverse','P0',210,tmp/'resolved')
        trace=dry_launch(cmd)
        (OUT/'13_resolved_launch.json').write_text(json.dumps(trace,indent=2,default=str)+'\n')
        badcmd=[a for a in cmd if not a.startswith('task:=')]+['task_name:=nonexistent_task']
        wrong=dry_launch(badcmd)
        record('invalid_launch_cli_task_alias',dict(argument='task_name:=nonexistent_task',resolved_task=wrong['resolved']['task_name']))
        wrongmode=dry_launch([a for a in cmd if not a.startswith('global_planner_mode:=')]+['global_planner_mode:=typo_efe'])
        assert wrongmode['resolved']['global_planner_mode']=='efe'
        record('invalid_launch_mode','typo_efe silently resolved to efe')
        altered=dry_launch([a for a in cmd if not a.startswith('state_max_predict_dt_s:=')]+['state_max_predict_dt_s:=0.125'])
        vals={n:d['parameters']['state_max_predict_dt_s'] for n,d in altered['nodes'].items() if 'state_max_predict_dt_s' in d['parameters']}
        assert set(vals.values())=={.125}
        record('declared_prediction_cap',vals)
        ekfoff=dry_launch([a for a in cmd if not a.startswith('state_correction_ekf:=')]+['state_correction_ekf:=false'])
        assert ekfoff['nodes']['visibility_aware_efe_agent']['parameters']['state_correction_ekf'] is False
        record('explicit_false_ekf','false reaches planner')
        zero=dry_launch([a for a in cmd if not a.startswith('max_predict_speed_mps:=')]+['max_predict_speed_mps:=0'])
        record('multicam_zero_speed_cap',zero['nodes']['visibility_aware_efe_agent']['parameters']['max_predict_speed_mps'])
        # A malformed retained summary is distinguished from absent JSON only by parsing,
        # not by shape. The resume function above never reads either representation.
        sp=tmp/'summary_shape';sp.mkdir();(sp/'run_summary.json').write_text('[]')
        assert c._read_run_summary(sp)==[]
        record('summary_wrong_shape','JSON list accepted as summary object')
        # Same-time/drop reason contract regression, executed unchanged.
        refusal=tmp/'refusal';fake_evidence(refusal,{'completed':True})
        assert c._verify_correction_assimilations(refusal)==(True,'')
        record('reasoned_refusal_contract',c._verify_correction_assimilations(refusal))
        # Defaulted task files and cwd-dependent model paths are visible at launch.
        record('path_resolution_policy',dict(runner_subprocess_has_explicit_cwd=False,
            tasks_yaml=trace['resolved']['tasks_yaml'],world_path=trace['resolved']['world_path'],
            manager_mean_path=trace['nodes']['camera_manager_active']['parameters']['learned_correction_path'],
            detector_model_path=trace['nodes']['batched_four_camera_yolo']['parameters']['model_path']))
        # Real registered metadata inspection: no measurements/accuracy reanalysis.
        selection_path=ROOT/'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json'
        selection=json.loads(selection_path.read_text()); selected=selection['runs'][0]
        assert selected['key']=='fusion_network_traverse__P0__seed210'
        run=ROOT/selected['run'];real_manifest=json.loads((run/'run_manifest.json').read_text())
        hashes={name:sha(run/name)==expected for name,expected in selected['files'].items()}
        assert all(hashes.values())
        actualcfg=deepcopy(active);actualcfg['_git_provenance']={k:v for k,v in real_manifest.items() if k.startswith('git_')}
        actualcfg['_campaign_config_sha256']=sha(active_path);actualcfg['_yolo_model_sha256']=sha(active['yolo_model'])
        record('registered_run_manifest_reuse_check',dict(selection_sha256=sha(selection_path),selected=selected['key'],all_selected_hashes_match=all(hashes.values()),same_recorded_git_resume=c._existing_entry_matches_config(selected['event'],actualcfg)))
        aligned=module('audit13_aligned',ROOT/'experiments/fusion_on_fixed_routes/aligned.py')
        assert aligned.rows(run)
        record('registered_loader','opened exact selected run through aligned.rows; no scores calculated')
        record('registered_correction_accounting',c._verify_correction_assimilations(run))
        matrix_checks={}
        for yaml_path in sorted((ROOT/'experiments/icra_commissioning').glob('*.yaml')):
            payload=yaml.safe_load(yaml_path.read_text())
            if not isinstance(payload,dict) or 'launch_file' not in payload:continue
            try:
                loaded=c._load_config(yaml_path);matrix=c._build_run_matrix(loaded)
                for t,arm,seed in matrix:c._build_launch_cmd(loaded,t,arm,seed,tmp/'all_yaml_dry')
                matrix_checks[yaml_path.name]=dict(cells=len(matrix),validation_and_command_build='passed',cleanup_mode=loaded.get('cleanup_mode','legacy_global'),ros_domain_id_base=loaded.get('ros_domain_id_base'))
            except Exception as exc:
                matrix_checks[yaml_path.name]=dict(error=type(exc).__name__+': '+str(exc))
        record('campaign_yaml_dry_checks',matrix_checks)
        # Every AST-named setting, with declaration/default/consumer/resume references.
        build_inventory(trace, active, cmd, real_manifest)
        build_node_inventory(trace,real_manifest)
        audit_registered_provenance(selection_path,selected,real_manifest)
    hashes_after={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    output=dict(scope='No live processes; fake process and action boundaries; real files only in temporary fixtures',
                probes=RESULTS,executed_source_bytes=EXECUTED_SOURCES,source_hashes=hashes_before,
                changed_during_probe={k:(v,hashes_after[k]) for k,v in hashes_before.items() if v!=hashes_after[k]})
    (OUT/'13_probe_results.json').write_text(json.dumps(output,indent=2,default=str)+'\n')


def build_inventory(trace, active, cmd, actual):
    """All parsed settings, plus ignored campaign declarations, with source indices."""
    files=[SOURCE,Path(lc.__file__).resolve(),ROOT/'src/experiments/launch/warehouse_primary_comparison.launch.py',ROOT/'src/experiments/experiments/nodes/experiment_logger.py']
    trees={p:ast.parse(p.read_text()) for p in files}
    resume=next(n for n in trees[SOURCE].body if isinstance(n,ast.FunctionDef) and n.name=='_existing_entry_matches_config')
    compare_sets={}
    for n in ast.walk(resume):
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Tuple) and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id in ('numeric_keys','bool_keys','string_keys'):
                for e in n.value.elts:compare_sets[e.value]=n.targets[0].id
    args=argsmap(cmd)
    nodes=trace['nodes'];rows=[]
    keys={k for k in set(trace['resolved'])|set(active)|set(active['conditions']['P0'])|set(args) if not k.startswith('_')}
    for key in sorted(keys):
        references={}
        for p,tree in trees.items():
            refs=sorted({n.lineno for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value==key})
            if refs:references[str(p.relative_to(ROOT))]=refs
        runtime={name:{key:node['parameters'][key]} for name,node in nodes.items() if key in node['parameters'] and node['scheduled_for_execution']}
        if key.startswith('manager_'):
            short=lc._MANAGER_PARAM_NAMES.get(key,key[len('manager_'):])
            if short in nodes.get('camera_manager_active',{}).get('parameters',{}):runtime['camera_manager_active']={short:nodes['camera_manager_active']['parameters'][short]}
        if key in active['conditions']['P0']:tier='condition P0';declared=active['conditions']['P0'][key]
        elif key in active:tier='campaign';declared=active[key]
        else:tier='default/derived';declared=None
        rows.append(dict(setting=key,declaration_tier=tier,declared_value=declared,
            launch_argument=args.get(key,'<omitted>'),launch_declared_default=trace['declarations'].get(key,'<not declared>'),
            parsed_or_resolved_value=trace['resolved'].get(key,'<not parsed>'),
            runtime_same_name_or_manager_alias=runtime,registered_manifest_value=actual.get(key,'<absent>'),
            resume_numeric_bool_string_check=compare_sets.get(key,'no generic comparison; see custom checks'),references=references))
    (OUT/'13_setting_inventory.json').write_text(json.dumps(rows,indent=2,default=str)+'\n')
    record('setting_inventory',dict(settings=len(rows),parsed=len(trace['resolved']),node_parameter_sets=list(nodes)))


def build_node_inventory(trace,actual):
    paths={
        'wait_for_odom':['src/sim/sim/wait_for_odom.py'],
        'actuation_noise_node':['src/sim/sim/actuation_noise_node.py'],
        'encoder_noise_node':['src/sim/sim/encoder_noise_node.py'],
        'goal_mission_node':['src/experiments/experiments/nodes/goal_mission_node.py'],
        'goal_marker_node':['src/experiments/experiments/nodes/goal_marker_node.py'],
        'experiment_logger':['src/experiments/experiments/nodes/experiment_logger.py'],
        'visibility_aware_efe_agent':['src/planning/planning/nodes/unicycle_planner_node.py','src/planning/planning/nodes/efe_agent_node.py'],
        'batched_four_camera_yolo':['src/perception/perception/nodes/batched_four_camera_yolo_node.py'],
        'camera_manager_active':['src/reliability/reliability/nodes/camera_manager_node.py']}
    aliases={'model_path':'yolo_model','image_size':'yolo_imgsz','class_name':'yolo_target_class','class_id':'yolo_class_id',
             'confidence_threshold':'yolo_conf_threshold','iou_threshold':'yolo_iou_threshold','device':'yolo_device',
             'use_masks':'yolo_use_masks','max_batch_stamp_skew_s':'yolo_max_batch_stamp_skew_s','min_bbox_area_px':'yolo_min_bbox_area_px'}
    records=[]
    for name,node in trace['nodes'].items():
        if not node['scheduled_for_execution']:continue
        defaults={};references={}
        for rel in paths.get(name,[]):
            tree=ast.parse((ROOT/rel).read_text())
            for n in ast.walk(tree):
                declare = (isinstance(n,ast.Call) and
                    ((isinstance(n.func,ast.Attribute) and n.func.attr=='declare_parameter') or
                     (isinstance(n.func,ast.Name) and n.func.id=='_declare_if_not')))
                if declare and len(n.args)>=2 and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):
                    defaults[n.args[0].value]=dict(expression=ast.unparse(n.args[1]),file=rel,line=n.lineno)
                if isinstance(n,ast.Constant) and isinstance(n.value,str):references.setdefault(n.value,[]).append(f'{rel}:{n.lineno}')
        for key in sorted(set(defaults)|set(node['parameters'])):
            mkey=('manager_'+key if name=='camera_manager_active' else
                  'command_noise_'+key if name=='actuation_noise_node' and ('slip' in key or 'additive' in key or key=='correlation_alpha') else
                  'encoder_noise_'+key if name=='encoder_noise_node' and ('slip' in key or 'additive' in key or key=='correlation_alpha') else
                  aliases.get(key,key) if name=='batched_four_camera_yolo' else key)
            if key=='correction_propagation_drift_std_m_per_s':mkey='manager_correction_propagation_drift_std'
            records.append(dict(node=name,parameter=key,node_default=defaults.get(key),
                supplied_by_launch=key in node['parameters'],launch_value=node['parameters'].get(key,'<uses node declaration/default>'),
                matching_manifest_key=mkey,registered_manifest_value=actual.get(mkey,'<absent>'),source_references=references.get(key,[])))
    (OUT/'13_node_parameter_inventory.json').write_text(json.dumps(records,indent=2,default=str)+'\n')
    record('node_inventory',dict(parameters=len(records),note='AST literal/direct-helper declarations plus launch overrides; dynamic loops and internal clamps require linked source inspection'))


def audit_registered_provenance(selection_path,selected,manifest):
    evidence=selection_path.parent
    protocol=json.loads((evidence/'protocol.json').read_text())
    checks=[]
    for rel,expected in protocol['sources'].items():
        snapshot=evidence/'source_snapshot'/rel
        checks.append(dict(path=rel,protocol_hash=expected,snapshot_exists=snapshot.is_file(),
            snapshot_matches=sha(snapshot)==expected if snapshot.is_file() else None,
            current_matches=sha(ROOT/rel)==expected if (ROOT/rel).is_file() else None))
    artifacts={}
    for key in ('yolo_model','camera_network_artifact_path','manager_commissioned_calibration_path','manager_commissioned_world_covariance_path','manager_learned_correction_path'):
        p=Path(manifest[key]);p=p if p.is_absolute() else ROOT/p
        hashkey={'yolo_model':'yolo_model_sha256','camera_network_artifact_path':'camera_network_artifact_sha256','manager_commissioned_calibration_path':'manager_commissioned_calibration_sha256','manager_commissioned_world_covariance_path':'manager_commissioned_world_covariance_sha256','manager_learned_correction_path':'manager_learned_correction_sha256'}[key]
        artifacts[key]=dict(path=str(p.relative_to(ROOT)),current_sha256=sha(p),recorded=manifest.get(hashkey),matches=sha(p)==manifest.get(hashkey))
    imports=AMBIENT_IMPORTS
    from ament_index_python.packages import get_package_share_directory
    shares={p:get_package_share_directory(p) for p in ('experiments','sim','perception','planning','reliability')}
    output=dict(selection=str(selection_path.relative_to(ROOT)),selected=selected['key'],protocol_sources=checks,
                artifacts=artifacts,current_import_resolution=imports,current_package_shares=shares,
                execution_attestation='Current symlink/import resolution and matching snapshot bytes cannot prove historical in-memory imports; no loaded-module attestation recorded.')
    (OUT/'13_registered_provenance.json').write_text(json.dumps(output,indent=2)+'\n')
    record('registered_provenance',dict(protocol_sources=len(checks),snapshots_matching=sum(x['snapshot_matches'] is True for x in checks),current_sources_matching=sum(x['current_matches'] is True for x in checks),artifact_hashes=artifacts))


if __name__=='__main__':
    main()
