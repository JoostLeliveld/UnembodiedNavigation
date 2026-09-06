"""Audit 07 actual-method probes using existing ROS-free node fixtures.

No ROS graph or simulator is started. Motion is a declared synthetic common Q;
we test aggregation/wiring, not the dynamics implementation owned by audit 01.
"""
from pathlib import Path
import importlib.util
import json
import math
import sys
from types import SimpleNamespace as NS

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
for rel in ("src/reliability","src/planning","src/unav_common","src/state",
            "src/perception","src/experiments","src/sim","tests/planning"):
    sys.path.insert(0,str(ROOT/rel))
from test_planner_node_state_correction import make_state_node, state_msg
from test_planner_node_correction_wiring import stamp
from reliability.nodes import camera_manager_node as M
from reliability.camera_manager import CameraManager, CameraManagerConfig
from reliability.contracts import CameraQuality
from reliability.fusion import MapObservation, map_observations_to_json

spec=importlib.util.spec_from_file_location("audit07_math",Path(__file__).with_name("07_fusion_probe.py"))
F=importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
OUT={}


def node(gate=0.):
    n=make_state_node(belief_stamp_s=9.9,now_s=10.,nis_threshold=gate)
    n.heading_update_mode="coupled"
    n.state_reanchor_m=0.; n.state_reject_inflate_m2=0.
    n._seen_map_observation_stamps={}; n._seen_state_source_batch_ids=set()
    n.require_state_correction_envelope=True
    n._resolve_plan_frame_id=lambda:"map_bev"
    n.belief_m=np.array([.01,-.02,.1])
    n.belief_S=np.array([[.05,.01,.005],[.01,.04,-.004],[.005,-.004,.03]])
    n.replays=[]
    def replay(m,P,start,end,u,dt):
        n.replays.append(float(dt))
        # One same-time batch gets Q exactly once, independent of camera count.
        Q=np.diag([.0001,.0001,.0004]) if dt>1e-9 else np.zeros((3,3))
        return m.copy(),P+Q,{}
    n._replay_cmd_log_interval=replay
    n.errors=[]
    n._fatal_experiment_stop=lambda context,exc:n.errors.append((context,str(exc)))
    return n


def envelope(xy,R,batch="synthetic",t=9.95,**extra):
    payload=dict(schema_version=1,frame_id="map_bev",source_batch_id=batch,
        correction_stamp=t,common_capture_stamp=t,xy=list(xy),covariance_m2=np.array(R).tolist(),
        accepted_camera_ids=["camera_A","camera_B"])
    payload.update(extra)
    return NS(data=json.dumps(payload))


def snapshot(n):
    return dict(mean=n.belief_m,covariance=n.belief_S,
        stamp=n._stamp_to_float(n.belief_stamp),seen_batches=sorted(n._seen_state_source_batch_ids),
        terminal=[json.loads(s) for s in n.correction_assimilation_pub.published],
        errors=list(n.errors),replay_intervals=list(n.replays))


items=[F.obs("camera_A",(.02,.03),[[.01,.004],[.004,.02]],t=9.95),
       F.obs("camera_B",(.06,-.01),[[.03,-.012],[-.012,.01]],t=9.95),
       F.obs("camera_C",(-.02,.02),[[.025,.009],[.009,.04]],t=9.95)]
direct=node(); direct.state_correction_mode="per_camera"
direct._map_observations_cb(NS(data=map_observations_to_json(items,frame_id="map_bev")))
assert not direct.errors
one=node(); z,R=F.independent(items)
one._state_correction_envelope_cb(envelope(z,R))
assert not one.errors
np.testing.assert_allclose(direct.belief_m,one.belief_m,atol=1e-12)
np.testing.assert_allclose(direct.belief_S,one.belief_S,atol=1e-12)
np.testing.assert_allclose(one.belief_S,F.posterior(node().belief_m,
    node().belief_S+np.diag([.0001,.0001,.0004]),items)[1],atol=1e-12)
robust=node(); z,R=F.joint_network_estimate_2d(items)
robust._state_correction_envelope_cb(envelope(z,R))
OUT["same_event_same_prior_R_Q_actual_nodes"]=dict(direct=snapshot(direct),
    independent=snapshot(one),robust=snapshot(robust),
    maximum_independent_covariance_difference=np.max(abs(direct.belief_S-one.belief_S)))

before=direct.belief_S.copy()
direct._map_observations_cb(NS(data=map_observations_to_json(items,frame_id="map_bev")))
assert np.array_equal(direct.belief_S,before)
OUT["repeated_direct_delivery"]=dict(changed_covariance=False,
    terminal_records=len(direct.correction_assimilation_pub.published))

# Wrong-frame values cross the actual callback and mutate the robot belief.
bad_frame=node(); bad_frame.state_correction_mode="per_camera"
bad_frame._map_observations_cb(NS(data=map_observations_to_json(items,frame_id="camera_optical")))
assert not bad_frame.errors
np.testing.assert_allclose(bad_frame.belief_m,one.belief_m,atol=1e-12)
OUT["wrong_frame_direct_callback"]=snapshot(bad_frame)

# Fused schema/frame/SPD repair remains effective. Extra semantic fields are unchecked.
validations={}
for name,extra in [("schema",dict(schema_version=99)),("frame",dict(frame_id="camera_optical")),
    ("indefinite",dict(covariance_m2=[[.01,.1],[.1,.01]])),
    ("asymmetric",dict(covariance_m2=[[.01,.001],[0.,.01]])),
    ("duplicate_camera_list",dict(accepted_camera_ids=["camera_A","camera_A"])),
    ("empty_camera_list",dict(accepted_camera_ids=[])),
    ("common_time_after_correction",dict(common_capture_stamp=11.)),
    ("boolean_schema",dict(schema_version=True))]:
    n=node(); n._state_correction_envelope_cb(envelope([.02,.03],np.eye(2)*.02,**extra))
    validations[name]=snapshot(n)
    if name in {"schema","frame","indefinite","asymmetric"}:
        assert n.errors and not n.correction_assimilation_pub.published
OUT["fused_envelope_validation"]=validations

# Actual manager dispatch preserves original capture evidence beside aligned values.
class Pub:
    def __init__(self,fail_once=False):self.messages=[]; self.fail_once=fail_once
    def publish(self,msg):
        if self.fail_once:
            self.fail_once=False
            raise RuntimeError("synthetic publisher failure")
        self.messages.append(msg)


def manager(items,**changes):
    h=[(9.95,(0.,0.,0.)),(9.97,(.0044,0.,0.)),(9.99,(.0088,0.,0.))]
    n=NS(authority="active",manager=CameraManager(CameraManagerConfig(
        min_spatial_trust=0.,min_association_confidence=0.,max_measurement_age_s=1.25)),
        fusion_max_timestamp_spread_s=.05,_belief_query_history=h.copy(),_odom_history=h.copy(),
        reliability_query_max_time_delta_s=.35,propagation_drift_std=.05,
        _bootstrap_camera_ids=set(),bootstrap_min_cameras=2,bootstrap_max_disagreement_m=.91,
        fusion_disagreement_gate_m=.6,fusion_rule="joint_network",
        camera_models={o.camera_id:NS(cam_pos=(0.,0.,5.)) for o in items},
        _bias_floors=lambda _:None,timestamp_compensation=False,correction_residual_interval_s=.05,
        frame_id="map_bev",fusion_common_mode_std_m=0.,covariance_profile="synthetic_full_R",
        _gate_rejections={},_reliability_query_source_by_camera={},_silhouette_status_by_camera={},
        _detection_extras_by_camera={},decision_pub=Pub(),selected_pub=Pub(),active_pub=Pub(),
        fused_correction_pub=Pub(),_ready_source_batch_id="synthetic_batch",
        _last_decided_source_batch_id=None,get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_000_000_000)),
        fusion_mode=True,_map_observations=lambda _:items,_publish_map_observations=lambda _:None)
    for k,v in changes.items():setattr(n,k,v)
    n._decide_fused=lambda now,obs,source_batch_id:M.CameraManagerNode._decide_fused(
        n,now,obs,source_batch_id=source_batch_id)
    return n


staggered=[F.obs("camera_A",(1.,2.),t=9.95),F.obs("camera_B",(1.0088,2.),t=9.99)]
n=manager(staggered); M.CameraManagerNode._decide(n)
payload=json.loads(n.decision_pub.messages[0].data)
assert payload["observations"][0]["obs_stamp"]==9.95
assert payload["observations"][0]["xy"]==[1.,2.]
np.testing.assert_allclose(payload["observations"][0]["aligned_xy"],[1.0088,2.])
assert payload["fused_stamp"]==9.99
OUT["manager_capture_vs_common_time_payload"]=payload

noisy=manager(staggered,_odom_history=[(9.95,(0.,0.,0.)),(9.99,(0.,0.,0.))],
    _belief_query_history=[(9.95,(0.,0.,0.)),(9.99,(.4,0.,0.))])
M.CameraManagerNode._decide(noisy)
OUT["manager_two_valid_odom_poses_ignored_for_belief_jump"]=json.loads(noisy.decision_pub.messages[0].data)
assert OUT["manager_two_valid_odom_poses_ignored_for_belief_jump"]["observations"][0]["aligned_xy"]==[1.4,2.]

partial=manager(items,selected_pub=Pub(fail_once=True))
try:M.CameraManagerNode._decide(partial)
except RuntimeError:pass
first=dict(envelopes=len(partial.fused_correction_pub.messages),
    decisions=len(partial.decision_pub.messages),decided_id=partial._last_decided_source_batch_id)
M.CameraManagerNode._decide(partial)
OUT["manager_partial_publication"]=dict(after_failure=first,
    envelopes_after_retry=len(partial.fused_correction_pub.messages),
    envelope_ids=[json.loads(msg.data)["source_batch_id"] for msg in partial.fused_correction_pub.messages])
assert first==dict(envelopes=1,decisions=0,decided_id=None)
assert len(partial.fused_correction_pub.messages)==2

missing=node(); pub=missing.pixel_correction_diag_pub; normal_publish=pub.publish
def fail_publish(_):raise RuntimeError("synthetic diagnostic publish failure after commit")
pub.publish=fail_publish
msg=envelope([.02,.03],np.eye(2)*.02)
missing._state_correction_envelope_cb(msg)
OUT["filter_partial_publication_after_commit"]=snapshot(missing)
assert missing.errors and missing._stamp_to_float(missing.belief_stamp)==9.95
assert not missing.correction_assimilation_pub.published and not missing._seen_state_source_batch_ids
# Do not resume a failed experiment: this second delivery is an isolated recovery probe.
pub.publish=normal_publish
missing._state_correction_envelope_cb(msg)
OUT["filter_retry_after_commit_failure"]=snapshot(missing)
assert json.loads(missing.correction_assimilation_pub.published[0])["status"]=="dropped"

# Re-run two already reported cross-workstream defects without editing their evidence.
spec=importlib.util.spec_from_file_location("prior_review07",ROOT/"experiments/estimator_consistency/module_review_20260906/probe.py")
prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
OUT["reverified_01_untracked_bootstrap"]=prior.untracked_bootstrap()
OUT["reverified_10_same_clock_batches"]=prior.same_clock_batches()

if __name__=="__main__":
    print(json.dumps(F.jsonable(OUT),indent=2,allow_nan=False))
