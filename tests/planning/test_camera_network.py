"""Network algebra, planner wiring and gradients; synthetic fixtures only."""
import json
import numpy as np
import pytest
from planning.core.camera_network import CameraNetworkModel, projection_jacobian


def write_network(path, score=.5, availability=.4, spatial=False):
    xs=ys=np.array([-3.,0.,3.]); X,Y=np.meshgrid(xs,ys)
    rho=np.stack([np.full((3,3),score),np.full((3,3),score)])
    if spatial: rho=np.stack([.5+.05*X+.02*Y,.55-.03*X+.04*Y])
    R=np.array([[[.01,.006],[.006,.16]],[[.12,-.008],[-.008,.015]]])
    meta=dict(schema='camera_network.iwai.v1',reference='robot_ground_reference_xy',
        frame='map_bev',covariance_units='m2',score_target='detector_score_with_miss_zero',
        availability_target='valid_detection_finite_ground_projection',evidence='synthetic_test_fixture')
    np.savez(path,xs=xs,ys=ys,camera_ids=['camera_A','camera_B'],score=rho,
        availability=np.full((2,3,3),availability),R_cond_m2=R,R_miss_proxy_m2=R+25*np.eye(2),
        metadata_json=json.dumps(meta))
    return CameraNetworkModel(path)


def test_complementarity_preserves_full_directional_information(tmp_path):
    net=write_network(tmp_path/'field.npz',score=1.)
    covariance=net.proxy_ground_covariance(np.ones(2))
    expected=np.linalg.solve(np.linalg.solve(net.R[0],np.eye(2))+
        np.linalg.solve(net.R[1],np.eye(2)),np.eye(2))
    np.testing.assert_allclose(covariance,expected)
    assert abs(covariance[0,1])>1e-6
    for R in net.R: assert np.linalg.eigvalsh(R-covariance).min()>0
    reverse=CameraNetworkModel(net.path,cameras=['camera_B','camera_A'])
    np.testing.assert_allclose(reverse.proxy_ground_covariance(np.ones(2)),covariance)


def test_single_camera_is_the_matrix_iwai_precision_blend(tmp_path):
    net=write_network(tmp_path/'field.npz')
    one=CameraNetworkModel(net.path,cameras=['camera_A'])
    expected=np.linalg.solve(.3*one.precision[0]+.7*one.miss_precision[0],np.eye(2))
    np.testing.assert_allclose(one.proxy_ground_covariance([.3]),expected)


def test_misses_do_not_create_measurements_in_the_reference(tmp_path):
    net=write_network(tmp_path/'field.npz',score=.9,availability=0.)
    P=np.diag([.2,.3,.05])
    np.testing.assert_allclose(net.forecast_posterior([0.,0.,0.],P),P)
    np.testing.assert_allclose(net.forecast_posterior([0.,0.,0.],P,'information'),P)
    # A finite IWAI cost proxy remains distinct from the no-observation branch.
    assert np.isfinite(net.proxy_ground_covariance([0.,0.])).all()


def test_branch_reference_exposes_expected_information_optimism(tmp_path):
    net=write_network(tmp_path/'field.npz')
    P=np.array([[.2,.01,.006],[.01,.3,-.002],[.006,-.002,.05]])
    branch=net.forecast_posterior([0.,0.,0.],P)
    information=net.forecast_posterior([0.,0.,0.],P,'information')
    assert np.linalg.eigvalsh(branch).min()>0
    assert np.linalg.eigvalsh(branch-information).min()>-1e-12
    assert np.trace(branch[:2,:2])>np.trace(information[:2,:2])


def test_all_hits_matches_the_joint_linear_update_including_heading_cross_covariance(tmp_path):
    net=write_network(tmp_path/'field.npz',availability=1.)
    P=np.array([[.2,.01,.03],[.01,.3,-.01],[.03,-.01,.05]])
    joint_precision=np.linalg.solve(P,np.eye(3))
    joint_precision[:2,:2]+=net.precision.sum(axis=0)
    expected=np.linalg.solve(joint_precision,np.eye(3))
    np.testing.assert_allclose(net.forecast_posterior([0.,0.,.4],P),expected,atol=1e-12)
    assert expected[2,2]<P[2,2]


def test_numpy_and_casadi_match_inside_and_outside_support(tmp_path):
    ca=pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',spatial=True)
    H=np.array([[120.,5.,640.],[2.,140.,400.],[.03,.02,1.]])
    m=ca.MX.sym('m',3);P=ca.MX.sym('P',3,3)
    func=ca.Function('network_cov',[m,P],[net.make_proxy_covariance_casadi(H)(m,P)])
    cov=np.diag([.05,.04,.03])
    for state in ([.35,.42,.1],[4.,5.,.1],[2.9,.1,.1]):
        actual=np.asarray(func(state,cov))
        expected=net.planning_diagnostics(state,cov,H)['R_plan']
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-8)
    np.testing.assert_array_equal(net.query([4.,5.,0.])['availability'],[0.,0.])
    with pytest.raises(ValueError):net.query([0.,0.,0.,.9])


def make_planner(path):
    from planning.planners.base_planner import UnicyclePlannerBase
    return UnicyclePlannerBase(horizon=5,dt=.25,v_min=0.,v_max=.5,w_min=-1.,w_max=1.,
        control_weight=.02,process_noise_xy=.01,process_noise_theta=.02,obs_noise_uv=2.5,
        goal_sigma_uv=30.,risk_weight_obs=1.,ambiguity_weight=1.,optimizer_maxiter=15,
        optimizer_gtol=1e-5,optimizer_warm_start=False,seed=513,use_visibility_model=True,
        camera_network_artifact_path=str(path),camera_params=dict(cam_pos=(-5.,-5.,5.),
            look_at=(0.,0.,0.),img_width=1280,img_height=720,fov_h_rad=1.2))


def test_real_planner_uses_network_cost_and_correct_gradient(tmp_path):
    pytest.importorskip('casadi')
    net=write_network(tmp_path/'field.npz',spatial=True)
    planner=make_planner(net.path)
    state=np.array([.35,.42,.1]);P=np.diag([.05,.04,.03]);goal=np.array([1.2,1.,0.])
    goal_obs=planner._goal_obs(goal)
    evaluate=planner._get_casadi_valgrad(goal,goal_obs,use_observation_risk=True,use_ambiguity_term=True)
    u=np.tile([.22,.06],5)
    value,gradient=evaluate(u,state,P,goal_obs,goal[:2],0.)
    numerical=[]
    for i in range(len(u)):
        delta=np.zeros_like(u);delta[i]=1e-5
        numerical.append((evaluate(u+delta,state,P,goal_obs,goal[:2],0.)[0]-
            evaluate(u-delta,state,P,goal_obs,goal[:2],0.)[0])/2e-5)
    np.testing.assert_allclose(gradient,numerical,rtol=2e-4,atol=2e-5)
    # Existing diagnostic totals are undiscounted-horizon sums; the optimizer
    # normalizes by the effective discounted horizon. Compare the same scale.
    numpy_value=planner._evaluate_controls(u,state,P,goal,goal_obs,None)
    H_eff=sum(planner.discount_gamma**t for t in range(planner.horizon))
    np.testing.assert_allclose(value,numpy_value/H_eff,rtol=1e-7,atol=1e-6)
    with pytest.raises(RuntimeError,match='not a fresh measurement'):
        planner.observation_model_with_visibility(state,P)
    solved=planner.plan(state,P,goal[:2])
    assert np.isfinite(solved.total_cost)
    assert np.isfinite(solved.controls).all()


def test_invalid_covariance_and_unknown_masks_fail(tmp_path):
    net=write_network(tmp_path/'field.npz')
    with pytest.raises(ValueError,match='unknown camera'):
        CameraNetworkModel(net.path,cameras=['camera_Z'])
    with np.load(net.path) as data: payload={k:data[k] for k in data.files}
    payload['R_cond_m2'][0,0,0]=-1
    np.savez(tmp_path/'bad.npz',**payload)
    with pytest.raises(ValueError,match='positive definite'):
        CameraNetworkModel(tmp_path/'bad.npz')
