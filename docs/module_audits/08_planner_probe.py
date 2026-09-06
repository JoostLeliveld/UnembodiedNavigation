"""Independent numerical/boundary audit; no ROS graph, simulation, or log mutation.

Assertions distinguish demonstrated implementation behaviour (including defects)
from an acceptance test for a repaired system. Run from the repository root:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/08_planner_probe.py
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import conftest
sys.path[:0] = [str(ROOT / 'tests/planning'), str(ROOT / 'experiments/icra_commissioning')]
import numpy as np
import casadi as ca
import scipy
from planning.core.camera_network import CameraNetworkModel, projection_jacobian
from planning.core.casadi_efe import unicycle_step_ca, unicycle_jacobian_ca, unicycle_process_noise_ca
from planning.core.dynamics import unicycle_step, unicycle_jacobian, unicycle_process_noise
from planning.core.efe_utils import finite_diff_jacobian
from planning.core.visibility_gp_map import GPVisibilityMapConfig, GPVisibilityMapModel
from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel
from planning.planners.base_planner import UnicyclePlannerBase, PlanResult
from planning.nodes.efe_agent_node import EfeAgentNode
from geometry_msgs.msg import PoseStamped
from test_runtime_transactions import command_node
from test_planner_node_correction_wiring import _Clock, _Logger, stamp

SOURCES = [
    'src/planning/planning/core/' + name + '.py' for name in
    ('camera_network', 'casadi_efe', 'efe_utils', 'rollout', 'visibility_gp_map', 'nogo_cost', 'dynamics')
] + ['src/planning/planning/planners/base_planner.py',
     'src/planning/planning/nodes/efe_agent_node.py',
     'src/planning/planning/nodes/unicycle_planner_node.py',
     'src/unav_common/unav_common/camera_model.py',
     'src/unav_common/unav_common/occlusion_geometry.py',
     'src/experiments/experiments/core/visibility_launch_common.py',
     'src/experiments/launch/warehouse_primary_comparison.launch.py',
     'scripts/visibility_comparison/run_visibility_campaign.py'] + [
    'experiments/icra_commissioning/' + name for name in
    ('export_network_planner.py', 'verify_network_planner.py', 'future.py', 'network_route_probe.py',
     'network_navigation_runtime_pilot.yaml', 'network_navigation_recovery_pilot.yaml')]
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
HASHES = {p: digest(ROOT / p) for p in SOURCES}
OUT = dict(scope='synthetic software/model checks; saved planned routes, no drive metrics',
           versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, casadi=ca.__version__),
           sources=HASHES, cases={})
C = OUT['cases']

def save_case(name, **values):
    C[name] = values
    print(name, flush=True)

def field(path, n=1, score=.5, q=.5, spatial=False):
    xs = np.array([-2., -.5, 1., 3.]); ys = np.array([-2., 0., 2.])
    X, Y = np.meshgrid(xs, ys)
    scores = np.full((n, len(ys), len(xs)), score)
    if spatial:
        scores = np.stack([.4 + i * .03 + .04 * X + .06 * Y for i in range(n)])
    R = np.repeat(np.array([[[.04, .012], [.012, .09]]]), n, axis=0)
    if spatial:
        R = np.array([r * (1 + .6 * i) for i, r in enumerate(R)])
    metadata = dict(schema='camera_network.iwai.v1', reference='robot_ground_reference_xy',
                    frame='map_bev', covariance_units='m2', score_target='detector_score_with_miss_zero',
                    availability_target='valid_detection_finite_ground_projection')
    np.savez(path, xs=xs, ys=ys, camera_ids=[f'camera_{i}' for i in range(n)], score=scores,
             availability=np.full_like(scores, q), R_cond_m2=R, R_miss_proxy_m2=R + 25 * np.eye(2),
             metadata_json=json.dumps(metadata))
    return CameraNetworkModel(path)

def planner(path=None, **overrides):
    settings = dict(horizon=5, dt=.25, v_min=0., v_max=.22, w_min=-1., w_max=1.,
        control_weight=.02, process_noise_xy=.01, process_noise_theta=.02, obs_noise_uv=2.5,
        goal_sigma_uv=30., risk_weight_obs=1., ambiguity_weight=1., optimizer_maxiter=3,
        optimizer_gtol=1e-5, optimizer_warm_start=False, seed=210,
        camera_params=dict(cam_pos=(-5.,-5.,5.), look_at=(0.,0.,0.),
                           img_width=1280, img_height=720, fov_h_rad=1.2))
    if path:
        settings.update(use_visibility_model=True, camera_network_artifact_path=str(path))
    settings.update(overrides)
    return UnicyclePlannerBase(**settings)

def ca_eval(p):
    goal = np.array([1.2, 1., 0.]); obs = p._goal_obs(goal)
    fn = p._get_casadi_valgrad(goal, obs, use_observation_risk=p.use_obs_risk,
                             use_ambiguity_term=p.use_ambiguity)
    return fn, goal, obs

def node(global_mode=False):
    n = command_node()
    n.use_hierarchical = bool(global_mode); n._hier_phase = 'GLOBAL' if global_mode else 'LOCAL'
    n.global_planner_mode = 'efe'; n._global_solve_done = False
    n._global_goal_xy = None; n.goal_replan_move_m = 1.
    n._waypoints = []; n._wp_idx = 0; n.waypoint_spacing_m = .2
    n.optimizer_route_seed_mode = 'explicit'; n.driveable_geometry_json = ''
    n.goal_msg = PoseStamped(); n.goal_msg.pose.position.x = 2.
    n._goal_received_logged = True; n._update_goal_progress_origin = lambda _: None
    n._snapshot_plan_inputs = lambda: {'goal': n.goal_msg, 'pixel_stamp': None, 'state': None}
    n.belief_m = np.zeros(3); n.belief_S = np.eye(3) * .01; n.belief_stamp = stamp(9.9)
    n._resolve_belief_for_planning = lambda: (n.belief_m.copy(), n.belief_S.copy(), {})
    n._fatal_stop_triggered = False; n.get_logger = lambda: _Logger()
    n._publish_plan_and_metrics = lambda *a, **kw: None
    n._save_global_plan_artifacts = lambda *a, **kw: None
    n._pending_plan_started_at = None; n.latency_compensate_plan_handoff = False
    n.planner = NS(_controls_for_waypoints=lambda *a: np.zeros(4), prev_controls_flat=None)
    n.v_min = 0.; n.v_max = .22; n.w_min = -1.; n.w_max = 1.
    n._validate_plan_frames = lambda *a: ('map_bev', 'map_bev')
    n._current_goal_progress_index = lambda *a: 0.
    n._publish_planner_diagnostics = lambda *a, **kw: None
    n._warn_on_plan_health = lambda *a, **kw: None
    n._log_plan_debug_once = lambda *a, **kw: None
    n.debug_runtime = False
    return n

with tempfile.TemporaryDirectory(prefix='planner_audit08_') as td:
    td = Path(td)
    P = np.array([[.2, .01, .02], [.01, .3, -.01], [.02, -.01, .05]])
    state = np.array([.2, .4, .1])
    # Independent oracle: posterior information for each explicitly enumerated mask.
    for ncam in (1, 2, 5):
        for q in (0., .4, 1.):
            net = field(td / f'f{ncam}_{q}.npz', ncam, q=q)
            exact = np.zeros((3, 3))
            for mask in itertools.product((0, 1), repeat=ncam):
                J = np.linalg.inv(P)
                for hit, R in zip(mask, net.R):
                    if hit:
                        J[:2, :2] += np.linalg.inv(R)
                exact += np.prod([q if h else 1-q for h in mask]) * np.linalg.inv(J)
            actual = net.forecast_posterior(state, P)
            info = net.forecast_posterior(state, P, 'information')
            np.testing.assert_allclose(actual, exact, atol=1e-12)
            assert np.linalg.eigvalsh(actual-info).min() > -1e-12
            if q in (0., 1.): np.testing.assert_allclose(actual, info, atol=1e-12)
            save_case(f'outcomes_{ncam}_{q}', branch_xy_trace=float(np.trace(actual[:2,:2])),
                      information_xy_trace=float(np.trace(info[:2,:2])), min_eigenvalue=float(np.linalg.eigvalsh(actual).min()))
    net = field(td/'order.npz', 3, spatial=True)
    rev = CameraNetworkModel(net.path, cameras=list(reversed(net.camera_ids)))
    H = np.array([[120., 5., 640.], [2., 140., 400.], [.03, .02, 1.]])
    m = ca.MX.sym('m08', 3); S = ca.MX.sym('P08', 3, 3)
    proxy_fn = ca.Function('proxy08', [m, S], [net.make_proxy_covariance_casadi(H)(m, S)])
    errors = []
    for x in (state, [3., 0., 0.], [3.+1e-10, 0., 0.], [8., 8., 0.]):
        diag = net.planning_diagnostics(x, P, H)
        np.testing.assert_allclose(diag['R_plan'], rev.planning_diagnostics(x, P, H)['R_plan'], atol=1e-10)
        errors.append(np.max(np.abs(diag['R_plan']-np.asarray(proxy_fn(x, P)))))
    np.testing.assert_allclose(net.query(state)['score'], [.432, .462, .492], atol=1e-12)
    save_case('order_layout_numpy_casadi', max_covariance_difference=float(max(errors)))
    # Zero score retains the explicitly designed miss precision for every camera.
    one = field(td/'one.npz', 1, score=0., q=0.)
    five = field(td/'five.npz', 5, score=0., q=0.)
    np.testing.assert_allclose(five.proxy_ground_covariance(np.zeros(5)), one.R_miss[0]/5)
    save_case('no_available_cameras_proxy', one_proxy=one.proxy_ground_covariance([0.]).tolist(),
        five_proxy=five.proxy_ground_covariance(np.zeros(5)).tolist(), forecast_preserves_P=True,
        interpretation='finite designed cost endpoint, not a measurement update')
    try: CameraNetworkModel(one.path, cameras=[])
    except ValueError as e: save_case('empty_camera_set', rejected=str(e))
    const = field(td/'constant.npz', score=.8, q=.8)
    edge_scores = [float(const.query([x, 0., 0.])['score'][0]) for x in [3.-1e-10, 3., 3.+1e-10]]
    save_case('grid_edge_discontinuity', x=[3.-1e-10, 3., 3.+1e-10], scores=edge_scores)
    # Synthetic supported and unsupported adjacent nodes: interpolation carries support between them.
    with np.load(const.path) as d: payload = {k:d[k].copy() for k in d.files}
    payload['score'][:, :, -1] = 0.; np.savez(td/'support.npz', **payload)
    unsupported = CameraNetworkModel(td/'support.npz')
    save_case('support_boundary_interpolation', midpoint_score=unsupported.query([2., 0., 0.])['score'].tolist(),
              unsupported_vertex_score=unsupported.query([3., 0., 0.])['score'].tolist())
    sig_before = net.signature; before = np.asarray(proxy_fn(state, P))
    net.fields['score'][:] = .95
    after_numpy = net.planning_diagnostics(state, P, H)['R_plan']
    after_ca = np.asarray(proxy_fn(state, P))
    assert net.signature == sig_before and np.allclose(before, after_ca) and not np.allclose(after_numpy, after_ca)
    save_case('mutable_network_cache', signature_unchanged=True,
              maximum_numpy_symbolic_difference=float(np.max(np.abs(after_numpy-after_ca))))
    # Projection quotient rule, including full off-diagonal covariance and chart information.
    f = lambda s: (H @ np.r_[s[:2], 1.])[:2] / (H @ np.r_[s[:2], 1.])[2]
    D = finite_diff_jacobian(f, state)[:,:2]; J = projection_jacobian(H, state)
    np.testing.assert_allclose(D, J, rtol=1e-8, atol=1e-7)
    Ruv = J @ one.R[0] @ J.T
    np.testing.assert_allclose(J.T @ np.linalg.solve(Ruv, J), np.linalg.inv(one.R[0]), atol=1e-12)
    save_case('projection_frame_units', maximum_fd_difference=float(np.max(np.abs(D-J))),
              information_congruence_max_error=float(np.max(np.abs(J.T @ np.linalg.solve(Ruv, J)-np.linalg.inv(one.R[0])))))
    Hsing = np.array([[100.,0.,2.],[0.,100.,1.],[1.,0.,0.]])
    singular_fn = ca.Function('singular08', [m,S], [one.make_proxy_covariance_casadi(Hsing)(m,S)])
    try: one.planning_diagnostics([0.,0.,0.], P, Hsing)
    except ValueError as e: rejection = str(e)
    save_case('projection_singularity', numpy_rejection=rejection,
              casadi_covariance_finite=bool(np.isfinite(np.asarray(singular_fn([0.,0.,0.], P))).all()))
    # Legacy path has different outside-domain models in optimization and candidate selection.
    np.savez(td/'legacy.npz', xs=[-1.,0.,1.], ys=[-1.,0.,1.],
             P_conservative_plan_map=np.full((3,3), .8), P_mean_map=np.full((3,3), .9))
    legacy = planner(use_visibility_model=True, visibility_artifact_path=str(td/'legacy.npz'))
    lf = ca.Function('legacy08', [m], [legacy.visibility_model.make_prob_state_casadi()(m)])
    save_case('legacy_outside', numpy_score=legacy.visibility_probability([2.,0.,0.]),
              casadi_score=float(lf([2.,0.,0.])))

    net = field(td/'objective.npz', n=2, spatial=True)
    p = planner(net.path)
    fn, goal, obs = ca_eval(p); u = np.tile([.17, .06], p.horizon)
    val, grad = fn(u, state, P, obs, goal[:2], 0.)
    delta = 1e-5
    fd = np.array([(fn(u+np.eye(len(u))[i]*delta, state,P,obs,goal[:2],0.)[0]-
                    fn(u-np.eye(len(u))[i]*delta, state,P,obs,goal[:2],0.)[0])/(2*delta) for i in range(len(u))])
    np.testing.assert_allclose(grad, fd, atol=1e-5, rtol=1e-4)
    raw = p._evaluate_controls(u, state,P,goal,obs,None)
    heff = sum(p.discount_gamma**t for t in range(p.horizon))
    np.testing.assert_allclose(val,raw/heff,rtol=1e-7)
    save_case('active_objective_gradient_and_scale', casadi_objective=val, numpy_reported_sum=raw,
        effective_horizon=heff, max_gradient_error=float(np.max(np.abs(grad-fd))))
    # Record the P supplied at each cost evaluation; independently recurse motion only.
    seen = []; original = p.planning_visibility_diagnostics
    def observe(m_, P_):
        seen.append(P_.copy()); return original(m_, P_)
    p.planning_visibility_diagnostics = observe
    p._evaluate_controls(u, state,P,goal,obs,None)
    mm = state.copy(); PP = P.copy(); error = 0.
    for step, control in enumerate(u.reshape(-1,2)):
        F = unicycle_jacobian(mm,control,p.dt)
        Q = unicycle_process_noise(p.process_noise_xy,p.process_noise_theta,p.dt,theta=mm[2],v=control[0])
        PP = F@PP@F.T+Q; mm = unicycle_step(mm,control,p.dt)
        error = max(error,float(np.max(np.abs(PP-seen[step]))))
    save_case('objective_motion_only_covariance', max_difference=error,
              final_prior_xy_trace=float(np.trace(seen[-1][:2,:2])), camera_feedback=False)
    # Exact derivative of the implemented discrete mean; Q agrees between NumPy and CasADi.
    ctrl = np.array([.22, 1.]); sm = ca.MX.sym('sm08',3); cu = ca.MX.sym('cu08',2)
    dynamics = ca.Function('dyn08', [sm,cu], [unicycle_step_ca(sm,cu,1.), unicycle_jacobian_ca(sm,cu,1.),
                          unicycle_process_noise_ca(.01,.02,1.,sm[2],cu[0])])
    cm, cF, cQ = map(np.asarray, dynamics(state,ctrl))
    np.testing.assert_allclose(cF,finite_diff_jacobian(lambda x:unicycle_step(x,ctrl,1.),state),atol=1e-9)
    np.testing.assert_allclose(cQ,unicycle_process_noise(.01,.02,1.,theta=state[2],v=ctrl[0]),atol=1e-15)
    exact = np.array([.22*math.sin(1.),.22*(1-math.cos(1.)),1.])
    euler = unicycle_step(np.zeros(3),ctrl,1.)
    save_case('dynamics_derivatives_and_turn_approximation', jacobian_fd_error=float(np.max(np.abs(cF-unicycle_jacobian(state,ctrl,1.)))),
        exact_constant_twist=exact.tolist(), euler=euler.tolist(), xy_gap_m=float(np.linalg.norm(euler[:2]-exact[:2])))
    # Invalid heading P passes the network query and can give a finite, valid candidate.
    badP = np.diag([.1,.1,-.02])
    p.planning_visibility_diagnostics = original
    bad_eval = p.evaluate_rollout_controls(state,badP,goal[:2],u)
    rejected = False
    try: net.forecast_posterior(state,badP)
    except ValueError: rejected=True
    save_case('invalid_prior_covariance', min_eigenvalue=float(np.linalg.eigvalsh(badP).min()),
        cost_finite=bool(np.isfinite(bad_eval['total_cost'])), rollout_valid=bad_eval['rollout_valid'], forecast_rejects=rejected)
    # Cache omits Q settings; rebuild demonstrates a different objective after changing Q.
    p.process_noise_theta = .7
    cached, _, _ = ca_eval(p); oldval = cached(u,state,P,obs,goal[:2],0.)[0]
    p._casadi_valgrad_cache.clear(); rebuilt, _, _ = ca_eval(p)
    newval = rebuilt(u,state,P,obs,goal[:2],0.)[0]
    assert np.isclose(oldval,val) and not np.isclose(newval,val)
    save_case('process_noise_cache_omission', before=val, after_cached=oldval, after_rebuilt=newval)
    # Mixture flag changes optimization but fixed-control selection still uses the blend.
    mp = planner(use_visibility_model=True, visibility_artifact_path=str(td/'legacy.npz'), use_hit_miss_mixture=True)
    mf, mg, mo = ca_eval(mp); mu = np.tile([.1,.03],mp.horizon)
    mixture = mf(mu,state,P,mo,mg[:2],0.)[0]
    blend = mp._evaluate_controls(mu,state,P,mg,mo,None)/heff
    save_case('optional_mixture_selector_mismatch', optimized_objective=mixture, selection_objective=blend)
    # Real tiny-horizon solve: no possible route can reach the mission goal.
    tiny = planner(horizon=2,dt=.25,optimizer_terminal_goal_tolerance_m=.35,optimizer_maxiter=1)
    partial = tiny.plan(np.zeros(3),np.eye(3)*.01,[2.,0.])
    assert partial.rollout_valid and partial.terminal_goal_distance_pred>.35
    n = node(True); n.global_planner = NS(plan=lambda *a,**kw:partial)
    n._plan_once()
    save_case('incomplete_real_solve_handoff', optimizer_success=partial.optimizer_success,
        status=partial.optimizer_status, returned_goal_gap_m=partial.terminal_goal_distance_pred,
        phase=n._hier_phase, waypoints=n._waypoints, unvalidated_appended_goal=True)
    # Finite x with nonfinite objective is not excluded from selection.
    pnan = planner(horizon=2, use_obs_risk=False, use_ambiguity=False)
    original_eval = pnan._evaluate_candidate_controls
    count=[0]
    def corrupt_first(*a,**kw):
        r=original_eval(*a,**kw); count[0]+=1
        if count[0]==1: r['total_cost']=math.nan; r['scaled_total']=math.nan
        return r
    pnan._evaluate_candidate_controls=corrupt_first
    with patch('planning.planners.base_planner.minimize',return_value=NS(x=np.tile([.1,0.],2),success=False,status=1,nit=1,nfev=1,message='limit')):
        winner=pnan.plan(np.zeros(3),np.eye(3)*.01,[2.,0.])
    assert math.isnan(winner.total_cost)
    save_case('nonfinite_objective_wins', returned_nan_cost=True, rollout_valid=winner.rollout_valid,
              controls_finite=bool(np.isfinite(winner.controls).all()))

    n = node()
    variants = dict(missing_fields=NS(controls=np.array([[.2,0.]])),
        invalid_rollout=NS(controls=np.array([[.2,0.]]),rollout_valid=False),
        nonfinite_controls=NS(controls=np.array([[math.nan,0.]])),
        out_of_bounds=NS(controls=np.array([[2.,3.]])),
        nonfinite_cost=NS(controls=np.array([[.2,0.]]),total_cost=math.nan),
        nonfinite_states=NS(controls=np.array([[.2,0.]]),states=np.full((2,3),math.nan)),
        nonfinite_clearance=NS(controls=np.array([[.2,0.]]),min_predicted_obstacle_distance_m=-math.inf))
    save_case('direct_result_validation', decisions={k:n._result_safe_to_execute(v) for k,v in variants.items()})
    for label, result in [('invalid', NS(states=np.array([[0.,0.,0.],[.1,0.,0.]]),rollout_valid=False)),
                          ('nonfinite',NS(states=np.full((2,3),math.nan),rollout_valid=False))]:
        n = node(True); n.global_planner=NS(plan=lambda *a,**kw:result); n._plan_once()
        save_case('global_handoff_'+label, phase=n._hier_phase, waypoints=n._waypoints)

    # Controlled slow solves: real thread blocks before returning; inputs change concurrently.
    for global_mode in (True,False):
        for change in ('belief','goal','config','clock'):
            n=node(global_mode); entered=threading.Event(); release=threading.Event(); errors=[]; origin={}
            def slow(m0,S0,g,**kw):
                origin.update(state=m0.tolist(),goal=list(g))
                entered.set()
                if not release.wait(2.): raise RuntimeError('test barrier timeout')
                return NS(states=np.array([[0.,0.,0.],[.1,0.,0.]]),controls=np.array([[.2,0.]]),
                          rollout_valid=True,min_predicted_obstacle_distance_m=1.,terminal_goal_distance_pred=1.9)
            if global_mode: n.global_planner=NS(plan=slow,process_noise_theta=.02)
            else: n.planner.plan=slow
            def run():
                try:n._plan_once()
                except BaseException as e:errors.append(repr(e))
            thread=threading.Thread(target=run);thread.start();assert entered.wait(2.)
            if change=='belief':
                with n._data_lock:n.belief_m=np.array([1.,2.,1.]);n.belief_S=np.eye(3);n.belief_stamp=stamp(10.)
            if change=='goal':
                g=PoseStamped();g.pose.position.x=-2.;n._goal_cb(g)
            if change=='config':n.w_max=.1
            if change=='clock':n._clock.seconds=30.
            release.set();thread.join(2.);assert not thread.is_alive() and not errors,errors
            save_case(('global' if global_mode else 'direct')+'_slow_'+change, origin=origin,
                current_state=n.belief_m.tolist(),current_goal_x=n.goal_msg.pose.position.x,
                route_installed=(n._hier_phase=='LOCAL') if global_mode else None,
                commands=[[v.linear.x,v.angular.z] for v in n.cmd_pub.messages],pending_start_missing=n._pending_plan_started_at is None)

OUT['source_changes_during_probe'] = {p:dict(before=h,after=digest(ROOT/p)) for p,h in HASHES.items() if digest(ROOT/p)!=h}
OUT['probe_sha256'] = digest(__file__)
Path(__file__).with_name('08_planner_probe_results.json').write_text(json.dumps(OUT,indent=2)+'\n')
print('COMPLETED',len(C),'cases; source changes:',OUT['source_changes_during_probe'],flush=True)
