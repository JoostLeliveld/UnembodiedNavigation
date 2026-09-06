"""Independent synthetic checks for audit 07; no ROS graph, models or logs modified.

Run from the repository root. JSON records observed defects as well as verified
invariants; a successful run does not mean those defects have been repaired.
NumPy information/Joseph equations and a SciPy convex optimizer are independent
oracles for the tuple-based implementation. All quantities are synthetic except
the explicitly named frozen camera covariance matrices and source hashes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
for rel in ("src/reliability", "src/unav_common", "src/planning", "src/state"):
    sys.path.insert(0, str(ROOT / rel))
from reliability.contracts import CameraObservation, CameraQuality
from reliability.fusion import (
    MapObservation, independent_measurement_fusion_2d, joint_network_estimate_2d,
    distance_angle_weighted_fusion_2d, map_observations_from_json,
    map_observations_to_json, sequential_kalman_update_2d,
)
from reliability.bias_floor import apply_belief_floor, bias_floor_matrix, combine_floors
from reliability.camera_manager import CameraManager, CameraManagerConfig
from reliability.nodes import camera_manager_node as M
from reliability.replay import ReplayConfig, ReplayMode, _with_provider_quality

RESULTS = {}
RULES = M.SUPPORTED_FUSION_RULES
IDS = [f"camera_{c}" for c in "ABCDE"]
POSITIONS = {c: (0., 0., 5.) for c in IDS}


def obs(c, xy=(1., 2.), R=None, t=10.):
    return MapObservation(camera_id=c, timestamp_s=t, xy_m=xy,
                          covariance_m2=np.eye(2)*.01 if R is None else R,
                          quality=CameraQuality(camera_id=c))


def independent(items):
    W = np.linalg.inv(np.array([o.covariance_m2 for o in items]))
    J = W.sum(axis=0)
    R = np.linalg.solve(J, np.eye(2))
    z = np.linalg.solve(J, np.einsum("nij,nj->i", W, [o.xy_m for o in items]))
    return z, R


def robust_objective(items):
    W = np.linalg.inv(np.array([o.covariance_m2 for o in items]))
    z = np.array([o.xy_m for o in items])
    def objective(x):
        e = x-z
        d = np.sqrt(np.einsum("ni,nij,nj->n", e, W, e))
        w = np.minimum(1., 2.5/np.maximum(d, 1e-100))
        value = np.where(d <= 2.5, .5*d*d, 2.5*(d-1.25)).sum()
        gradient = np.einsum("n,nij,nj->i", w, W, e)
        return value, gradient
    return objective


def robust_covariance(items, z):
    Rs = np.array([o.covariance_m2 for o in items])
    W = np.linalg.inv(Rs)
    e = np.array([o.xy_m for o in items])-z
    d = np.sqrt(np.einsum("ni,nij,nj->n", e, W, e))
    w = np.minimum(1., 2.5/np.maximum(d, 1e-100))
    A = np.einsum("n,nij->ij", w, W)
    B = sum(wi**2 * Wi @ (Ri+np.outer(ei,ei)) @ Wi
            for wi,Wi,Ri,ei in zip(w,W,Rs,e))
    inv = np.linalg.solve(A,np.eye(2))
    neff = w.sum()**2/(w@w)
    return neff*inv@B@inv, w, neff


def posterior(m, P, items):
    H = np.eye(3)[:2]
    P = np.array(P, float); m = np.array(m, float)
    for o in items:
        R = np.array(o.covariance_m2)
        K = np.linalg.solve(H@P@H.T+R, H@P).T
        m = m+K@(np.array(o.xy_m)-H@m)
        A = np.eye(3)-K@H
        P = A@P@A.T+K@R@K.T
    return m,P


def result(items, rule, gate=1000., floors=None):
    r = M._gated_fusion(items, disagreement_gate_m=gate, rule=rule,
                        camera_positions_m=POSITIONS, belief_floors=floors)
    return dict(mean=r.mean_xy, covariance=r.covariance_m2,
                used=r.accepted_camera_ids, rejected=r.rejected_camera_ids)


def error(fn):
    try:
        return dict(returned=fn())
    except Exception as exc:
        return dict(error=type(exc).__name__, message=str(exc))


def rule_checks():
    R = np.array([[.01,.003],[.003,.04]])
    cases = {
        "zero": [],
        "one": [obs(IDS[0], R=R)],
        "identical_three": [obs(c,R=R) for c in IDS[:3]],
        "unequal_full_covariances": [
            obs(IDS[0], (.02,.03), [[.01,.004],[.004,.02]]),
            obs(IDS[1], (.06,-.01), [[.03,-.012],[-.012,.01]]),
            obs(IDS[2], (-.02,.02), [[.025,.009],[.009,.04]])],
        "symmetric_disagreement": [obs(c,(x,0.)) for c,x in zip(IDS,[-.1,0.,.1])],
        "one_extreme_outlier": [obs(c,(x,0.)) for c,x in zip(IDS,[0.,0.,2.])],
        "all_metric_gate_rejected": [obs(IDS[0],(-1.,0.)),obs(IDS[1],(1.,0.))],
    }
    out = {}
    for name,items in cases.items():
        out[name] = dict(inputs=[o.to_dict() for o in items], rules={})
        for rule in RULES:
            if not items:
                out[name]["rules"][rule] = error(lambda: result(items,rule))
                continue
            got = result(items,rule)
            out[name]["rules"][rule] = got
            if rule == "independent":
                z,Rref = independent(items)
                np.testing.assert_allclose(got["mean"],z,atol=1e-12)
                np.testing.assert_allclose(got["covariance"],Rref,atol=1e-12)
            if rule == "distance_angle":
                z = np.array([o.xy_m for o in items]); rho2=(z*z).sum(axis=1)
                raw = 5/np.sqrt(rho2+25)/(rho2+1e-6); w=raw/raw.sum()
                np.testing.assert_allclose(got["mean"],w@z,atol=1e-12)
                np.testing.assert_allclose(got["covariance"],sum(wi**2*np.array(o.covariance_m2)
                    for wi,o in zip(w,items)),atol=1e-12)
            if rule == "joint_network":
                cov,w,neff = robust_covariance(items,np.array(got["mean"]))
                np.testing.assert_allclose(got["covariance"],cov,atol=1e-12)
                f = robust_objective(items)
                opt = minimize(f,got["mean"],jac=True,method="BFGS",options={"gtol":1e-10})
                got.update(weights=w,effective_camera_count=neff,
                           optimizer_mean=opt.x, optimizer_gradient_norm=np.linalg.norm(f(opt.x)[1]),
                           mean_difference_m=np.linalg.norm(opt.x-got["mean"]))
            gated = result(items,rule,gate=.6)
            got["with_runtime_metric_gate"] = gated
    # Analytic Huber case: two good zeros, one far camera, sigma=.1 => mean=.125 m.
    np.testing.assert_allclose(out["one_extreme_outlier"]["rules"]["joint_network"]["mean"],
                               [.125,0.],atol=2e-9)
    for rule in RULES:
        np.testing.assert_allclose(out["one"]["rules"][rule]["covariance"],R,atol=1e-12)
    RESULTS["rules"] = out
    # Explicit all-zero geometry fallback, which otherwise lies outside the camera installation.
    grazers={c:(0.,0.,0.) for c in IDS[:2]}
    RESULTS["all_grazing_equal_weight_fallback"] = distance_angle_weighted_fusion_2d(
        [obs(c) for c in IDS[:2]],grazers)
    RESULTS["unequal_R_at_identical_mean"] = {
        "one_good": result([obs(IDS[0])],"joint_network"),
        "plus_uninformative":result([obs(IDS[0]),obs(IDS[1],R=np.eye(2)*1e5)],"joint_network")}
    return cases


def ordering_and_identity(cases):
    items=cases["unequal_full_covariances"]
    perms={}
    for rule in RULES:
        answers=[result(list(p),rule) for p in itertools.permutations(items)]
        means=np.array([r["mean"] for r in answers]); covs=np.array([r["covariance"] for r in answers])
        perms[rule]=dict(max_mean_difference_m=np.ptp(means,axis=0).max(),
                         max_covariance_difference_m2=np.ptp(covs,axis=0).max())
        assert perms[rule]["max_mean_difference_m"]<1e-12
        assert perms[rule]["max_covariance_difference_m2"]<1e-12
    tied=[obs(IDS[0],(0.,0.)),obs(IDS[1],(.2,0.))]
    RESULTS["permutations"] = perms
    RESULTS["best_single_tie"]=[result(p,"best_single") for p in [tied,tied[::-1]]]
    repeated=[obs(IDS[0]),obs(IDS[0])]
    RESULTS["duplicate_raw_helpers"]={rule:result(repeated,rule) for rule in RULES}
    manager=CameraManager(CameraManagerConfig(min_spatial_trust=0.,max_measurement_age_s=1.25))
    unique,_,_=manager.eligible_observations(timestamp_s=10.1,observations=repeated)
    assert len(unique)==1
    RESULTS["manager_duplicate_count"]=len(unique)
    # Callback executes the actual method with real CameraObservation JSON.
    def receiver():
        return NS(camera_ids=IDS[:2],_pending_source_batches={},_latest={},
                  _ready_source_batch_stamp_s=-math.inf,_ready_source_batch_id=None,
                  require_source_batch_id=True,get_logger=lambda:NS(warn=lambda msg:None))
    def deliver(s,c,x):
        o=CameraObservation(camera_id=c,source_batch_id="batch",timestamp_s=10.,
            detection_valid=True,pixel_uv=(x,500.),detector_score=.9)
        M.CameraManagerNode._observation_callback(s,c)(NS(data=o.to_json()))
    outcomes=[]
    for order in [(100.,200.),(200.,100.)]:
        s=receiver()
        for x in order: deliver(s,IDS[0],x)
        deliver(s,IDS[1],300.)
        outcomes.append(s._latest[IDS[0]].pixel_uv)
    RESULTS["conflicting_duplicate_callback_order"]=outcomes


def timing():
    items=[obs(IDS[0],(1.,2.),t=10.),obs(IDS[1],(1.0088,2.),t=10.04)]
    histories={"exact_supported":[(10.,(0.,0.,0.)),(10.04,(.0088,0.,0.))],
               "one_stale_pose":[(9.9,(0.,0.,0.))],
               "one_future_pose":[(10.3,(.066,0.,0.))],
               "missing":[],
               "belief_correction_jump":[(10.,(0.,0.,0.)),(10.04,(.4,0.,0.))],
               "wrong_frame_rotation":[(10.,(0.,0.,0.)),(10.04,(.0088,0.,0.))]}
    out={}
    for name,h in histories.items():
        aligned,rejected,target=M.align_observations_to_common_time(items,h,
            max_pose_delta_s=.35,drift_std_m_per_s=.05)
        out[name]=dict(observations=[o.to_dict() for o in aligned],rejected=rejected,target=target)
    np.testing.assert_allclose(out["exact_supported"]["observations"][0]["xy_m"],[1.0088,2.])
    assert out["missing"]["rejected"]==[IDS[0]]
    assert items[0].timestamp_s==10. and items[0].xy_m==(1.,2.)
    # Exactly representable binary times make the nearest-neighbor tie unambiguous.
    tied=[obs(IDS[0],(0.,0.),t=10.),obs(IDS[1],(.006875,0.),t=10.+1/32)]
    a=(10.+1/64,(.0034375,0.,0.)); b=(10.+3/64,(.0103125,0.,0.))
    ties=[]
    for h in [[(10.,(0.,0.,0.)),a,b],[(10.,(0.,0.,0.)),b,a]]:
        ties.append(M.align_observations_to_common_time(tied,h,max_pose_delta_s=.35,
                    drift_std_m_per_s=.05)[0][0].xy_m)
    out["same_history_different_callback_order"]=ties
    out["frame_oracle_90_degree_spawn"]=dict(odom_delta=[.0088,0.],
        map_delta=[0.,.0088],received_frame="odom",published_frame="map_bev")
    # A rigid coordinate offset cancels; rotation must still be applied to a displacement.
    out["missing_equal_time"]=[o.to_dict() for o in M.align_observations_to_common_time(
        [obs(c,t=10.) for c in IDS[:2]],[],max_pose_delta_s=.35,drift_std_m_per_s=.05)[0]]
    RESULTS["time_compensation"]=out


def floors_and_malformed():
    fs=[bias_floor_matrix(20.,a) for a in (0.,.7,1.4)]
    outputs=[]
    for perm in itertools.permutations(range(3)):
        C=np.array(combine_floors([fs[i] for i in perm]))
        assert min(np.linalg.eigvalsh(C-np.array(f)).min() for f in fs)>-1e-12
        outputs.append(dict(order=perm,covariance=C))
    RESULTS["three_floor_permutations"]=outputs
    F=np.eye(2)*.01
    one=[obs(IDS[0],(0.,0.),R=np.eye(2)*.001)]
    fused=result(one,"independent",floors={IDS[0]:F})
    P=np.eye(3); m=np.zeros(3)
    for _ in range(100):
        m,P=posterior(m,P,[obs("fused",fused["mean"],fused["covariance"])])
    RESULTS["measurement_floor_is_not_posterior_floor"]=dict(floor=F,
        fused_covariance=fused["covariance"],posterior_after_100=P,
        expected_planar_variance=1/(1+100/.01))
    malformed={}
    for name,R in [("indefinite",[[.01,.02],[.02,.01]]),
                   ("asymmetric",[[.01,.001],[0.,.01]]),
                   ("singular",[[.01,.01],[.01,.01]]),
                   ("nonfinite",[[math.inf,0.],[0.,.01]])]:
        malformed[name]=error(lambda R=R:obs(IDS[0],R=R).to_dict())
        assert "error" in malformed[name]
    for name,t in [("nan_timestamp",math.nan),("negative_timestamp",-1.)]:
        malformed[name]=error(lambda t=t:obs("",t=t).to_dict())
    malformed["wrong_frame_direct_batch"]=error(lambda:map_observations_from_json(
        map_observations_to_json(one,frame_id="camera_optical"))[1])
    for variance in [1e-6,1e-7,1e6]:
        malformed[f"valid_SPD_variance_{variance}"]=error(lambda v=variance:
            independent_measurement_fusion_2d([obs(IDS[0],R=np.eye(2)*v)]))
    malformed["indefinite_belief_silently_floored"]=error(lambda:apply_belief_floor(
        ((-1.,0.),(0.,.001)),F))
    malformed["one_zero_slope"]=error(lambda:combine_floors([
        bias_floor_matrix(10.,.3,along_slope=0.,across_slope=.00035)]))
    malformed["bootstrap_min_one_helper"]=M._largest_agreeing_group(one,.91)
    RESULTS["malformed_and_limiting_inputs"]=malformed
    class Provider:
        def query(self,c,xy,t):
            return CameraQuality(camera_id=c,p_available=.9,source_model="audit_fake_gp")
    original=obs(IDS[0],R=[[.016,.0006],[.0006,.009]])
    cfg=ReplayConfig(mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,quality_providers={IDS[0]:Provider()})
    RESULTS["GP_replaces_full_R"]=dict(before=original.to_dict(),after=
        _with_provider_quality(original,cfg,(5.,6.),10.1).to_dict())


def equivalence_and_dependence(cases):
    items=cases["unequal_full_covariances"]
    m=np.array([.01,-.02,.1]); P=np.array([[.05,.01,.005],[.01,.04,-.004],[.005,-.004,.03]])
    Q=np.diag([.0001,.0001,.0004]); prior=P+Q
    direct=posterior(m,prior,items)
    z,R=independent_measurement_fusion_2d(items)
    fused=posterior(m,prior,[obs("fused",z,R)])
    np.testing.assert_allclose(direct[0],fused[0],atol=1e-12)
    np.testing.assert_allclose(direct[1],fused[1],atol=1e-12)
    rz,rR=joint_network_estimate_2d(items)
    robust=posterior(m,prior,[obs("fused",rz,rR)])
    out=dict(initial_mean=m,initial_covariance=P,common_Q=Q,direct=direct,
        independent_fused=fused,robust_fused=robust,
        independent_max_difference=max(np.max(abs(direct[0]-fused[0])),np.max(abs(direct[1]-fused[1]))))
    # Shared-error oracle: equal camera R=.04I, cross-camera C=.03I.
    R=np.eye(2)*.04; C=np.eye(2)*.03; n=3
    out["correlated_cameras"]=dict(marginal_R=R,cross_camera_C=C,
        independent_report=R/n,true_covariance_of_average=(R-C)/n+C,
        joint_identical_report=joint_network_estimate_2d([obs(c,R=R) for c in IDS[:n]])[1])
    # Common compensation uses the same random motion U and the prior's prediction
    # uses that U too. These off-diagonal blocks are not supplied to current fusion.
    U=np.eye(2)*.0025
    out["shared_motion"]=dict(camera_marginal_after_alignment=R+U,
        cross_camera_after_alignment=U,prior_measurement_cross_covariance=U,
        naive_average_covariance=(R+U)/n,true_average_covariance=R/n+U)
    gated=[obs(IDS[0],(-.25,0.)),obs(IDS[1],(.25,0.))]
    out["sequential_gate_order"]=[sequential_kalman_update_2d((0.,0.),((.05,0.),(0.,.05)),p,
        nis_gate=9.21).__dict__ for p in [gated,gated[::-1]]]
    # Scalar staggered case, Q=.01 over the interval. Direct observation at t0,
    # propagation, then t1; moving the old observation and independently updating
    # a prior also propagated with that Q counts shared information incorrectly.
    prior_scalar=.1; r=.04; q=.01
    p0=1/(1/prior_scalar+1/r); direct_staggered=1/(1/(p0+q)+1/r)
    naive_common=1/(1/(prior_scalar+q)+1/(r+q)+1/r)
    out["staggered_Q"]=dict(initial_variance=prior_scalar,R=r,Q_interval=q,
        direct_final_variance=direct_staggered,naive_common_time_variance=naive_common)
    RESULTS["equivalence_and_dependence"]=out


def convergence():
    calibration=json.loads((ROOT/"logs/studies/icra_commissioning_20260905/network_planner/reference_calibration.json").read_text())
    z=[[-.17727081586261784,-.19941694910930402],[-.22061737111128787,-.04943132470561806],
       [.004093433062157981,.04814761686704322],[.4459502550909534,.26339572729237776],
       [.10996130649704644,.44006413291369645]]
    items=[obs(c,xy,calibration["cameras"][c]["R_m2"]) for c,xy in zip(IDS,z)]
    got=result(items,"joint_network",gate=.6)
    assert len(got["used"])==5
    f=robust_objective(items)
    opt=minimize(f,got["mean"],jac=True,method="BFGS",options={"gtol":1e-10})
    longer=joint_network_estimate_2d(items,max_iterations=1000)
    np.testing.assert_allclose(longer[0],opt.x,atol=2e-8)
    RESULTS["IRLS_iteration_cap"]=dict(inputs=[o.to_dict() for o in items],runtime=got,
        independent_optimizer_mean=opt.x,optimizer_gradient_norm=np.linalg.norm(f(opt.x)[1]),
        runtime_gradient_norm=np.linalg.norm(f(np.array(got["mean"]))[1]),
        runtime_mean_difference_m=np.linalg.norm(np.array(got["mean"])-opt.x),
        objective_excess=f(np.array(got["mean"]))[0]-f(opt.x)[0],longer_solve=longer)


def source_identity():
    paths=["src/reliability/reliability/"+f+".py" for f in ["fusion","contracts","camera_manager",
        "covariance_mapping","bias_floor","replay","projection","reference_calibration"]]
    paths += ["src/reliability/reliability/nodes/camera_manager_node.py",
        "src/planning/planning/nodes/unicycle_planner_node.py",
        "src/planning/planning/core/camera_network.py","experiments/fusion_on_fixed_routes/aligned.py",
        "experiments/icra_commissioning/network_navigation_runtime_pilot.yaml"]
    protocol=json.loads((ROOT/"logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/protocol.json").read_text())
    out={}
    for p in paths:
        sha=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        frozen=protocol["sources"].get(p)
        out[p]=dict(sha256=sha,registered_protocol_sha256=frozen,
                    matches_protocol=None if frozen is None else sha==frozen)
    RESULTS["source_identity"]=out


def jsonable(x):
    if isinstance(x,np.ndarray):return jsonable(x.tolist())
    if isinstance(x,np.generic):return jsonable(x.item())
    if isinstance(x,float) and not math.isfinite(x):return str(x)
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
    return x


if __name__=="__main__":
    source_identity()
    cases=rule_checks()
    ordering_and_identity(cases)
    timing()
    floors_and_malformed()
    equivalence_and_dependence(cases)
    convergence()
    print(json.dumps(jsonable(RESULTS),indent=2,allow_nan=False))
