"""Run actual simulator/consumer callbacks with controlled clocks and fake I/O.

No rclpy.init, DDS graph or simulator. Assertions reproduce observed defects,
not desired invariants. AST extraction changes only the Node I/O base class.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace as NS

from builtin_interfaces.msg import Time as Stamp
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Contact, Contacts, Entity
from ros_gz_interfaces.srv import ControlWorld
from rclpy.clock import Clock as RclpyClock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE_ROOT=Path(os.environ.get('AUDIT14_SOURCE_ROOT',ROOT))
OUTPUT_NAME=os.environ.get('AUDIT14_OUTPUT_NAME','callback_current_results.json')
WALL_TIME=[10.0]
time=NS(monotonic=lambda:WALL_TIME[0])


class Publisher:
    def __init__(self): self.messages=[]
    def publish(self,m): self.messages.append(copy.deepcopy(m))


class FakeNode:
    def __init__(self,name):
        self.params={}; self.now_s=10.; self.logs=[]; self.pubs={}; self.timers=[]
    def declare_parameter(self,k,v): self.params[k]=v
    def get_parameter(self,k): return NS(value=self.params[k])
    def get_logger(self): return NS(**{k:lambda s:self.logs.append(s) for k in ['info','warn','warning','error']})
    def get_clock(self): return NS(now=lambda:NS(nanoseconds=round(self.now_s*1e9)))
    def create_publisher(self,typ,topic,qos):
        self.pubs[topic]=Publisher(); return self.pubs[topic]
    def create_subscription(self,*a,**k): return None
    def create_timer(self,*a,**k): self.timers.append((a,k)); return None
    def create_client(self,*a):
        self.requests=[]
        def call(req):
            self.requests.append(req)
            return NS(done=lambda:True,result=lambda:NS(success=False))
        return NS(wait_for_service=lambda **k:True,call_async=call)


def load_class(rel, class_name, keep_methods=None):
    path=SOURCE_ROOT/rel; tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name not in ['main']]
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name)
    if keep_methods:
        cls.body=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in keep_methods]
    nodes.append(cls)
    env={**globals(),'Node':FakeNode,'Any':object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
    return env[class_name]


def stamp(s):
    ns=round(s*1e9); return Stamp(sec=ns//10**9,nanosec=ns%10**9)


def odom(s,x=0.,y=0.,yaw=0.,v=.2,w=0.,frame='odom'):
    m=Odometry();m.header.stamp=stamp(s);m.header.frame_id=frame;m.child_frame_id='base_link'
    m.pose.pose.position.x=float(x);m.pose.pose.position.y=float(y)
    m.pose.pose.orientation.z=math.sin(yaw/2);m.pose.pose.orientation.w=math.cos(yaw/2)
    m.twist.twist.linear.x=float(v);m.twist.twist.angular.z=float(w);return m


def contact(s,a,b):
    m=Contacts();m.header.stamp=stamp(s)
    c=Contact();c.collision1=Entity(name=a);c.collision2=Entity(name=b);m.contacts=[c];return m


def main():
    results={}
    W=load_class('src/sim/sim/wait_for_clock.py','WaitForClock')
    n=W();m=Clock();m.clock=stamp(0.);n._cb(m)
    assert n.received
    results['clock_zero_is_ready']={'ready':n.received,'message_time_s':0.}
    W=load_class('src/sim/sim/wait_for_odom.py','WaitForOdom')
    n=W();n._cb(odom(100.,x=42.,frame='foreign_frame'))
    assert n.received
    n2=W();n2.require_pose_match=True;n2.min_messages=3
    for _ in range(3):n2._cb(odom(100.,v=0.,frame='foreign_frame'))
    assert n2.received
    results['odom_readiness_has_no_epoch_frame_or_uniqueness']={'wrong_pose_default_ready':n.received,'three_identical_old_wrong_frame_messages_pose_gate_ready':n2.received,'now_s':n2.now_s,'source_stamp_s':100.}
    R=load_class('src/sim/sim/reset_world.py','ResetWorld')
    n=R();n._tick();n._tick();assert n.done
    results['reset_failure_is_done']={'done':n.done,'service_success':False,'logs':n.logs,'requested_reset_all':n.requests[0].world_control.reset.all}
    T=load_class('src/sim/sim/clock_throttle_node.py','ClockThrottleNode')
    n=T();m=Clock();m.clock=stamp(100.);n._clock_cb(m)
    for wall_s in [10.,10.02,10.04]:
        WALL_TIME[0]=wall_s;n._publish_latest()
    WALL_TIME[0]=900.;n._publish_latest()
    m=Clock();m.clock=stamp(1.);n._clock_cb(m);n._publish_latest()
    outputs=[p.clock.sec+p.clock.nanosec*1e-9 for p in n._pub.messages]
    expected=[100.,100.,1.] if hasattr(n,'duplicate_heartbeat_s') else [100.,100.,100.,100.,1.]
    assert outputs==expected
    results['clock_republishes_after_source_loss_and_forwards_rewind']={'outputs_s':outputs,'new_upstream_samples':2,'wall_now_s':WALL_TIME[0], 'duplicate_heartbeat_s':getattr(n,'duplicate_heartbeat_s',None)}
    E=load_class('src/sim/sim/encoder_noise_node.py','EncoderNoiseNode')
    n=E();n.enabled=False;n._odom_cb(odom(100.,x=5.));n._odom_cb(odom(100.1,x=5.02))
    state_before=(n._pose_x,n._pose_y,n._pose_theta,copy.deepcopy(n._pose_cov),list(n._linear_scale_jacobian))
    n._odom_cb(odom(.1,x=0.))
    assert n._pose_x==state_before[0] and n._pose_cov==state_before[3]
    n._odom_cb(odom(.2,x=.02))
    out=n._pub.messages[-1]
    assert abs(out.pose.pose.position.x-5.04)<1e-10
    fresh=E();fresh.enabled=False;fresh._odom_cb(odom(.1,x=0.));fresh._odom_cb(odom(.2,x=.02))
    results['encoder_rewind_retains_prior_pose_noise_covariance']={'before_x_m':state_before[0],'after_reset_x_m':out.pose.pose.position.x,'after_stamp_s':.2,
        'fresh_instance_x_m':fresh._pub.messages[-1].pose.pose.position.x,'reset_cleared_covariance':False,'reset_cleared_scale_jacobian':False}
    bad=E();bad.enabled=False;bad._odom_cb(odom(1.));bad._odom_cb(odom(1.1,v=math.nan))
    results['encoder_nonfinite_velocity_is_published']={'nan_pose':math.isnan(bad._pub.messages[-1].pose.pose.position.x),'nan_velocity':math.isnan(bad._pub.messages[-1].twist.twist.linear.x)}
    assert all(results['encoder_nonfinite_velocity_is_published'].values())
    A=load_class('src/sim/sim/actuation_noise_node.py','ActuationNoiseNode')
    def command():
        m=Twist();m.linear.x=.2;return m
    n=A();n.enabled=False;n._cmd_cb(command());n._watchdog_tick()
    assert n._pub.messages[-1].linear.x==.2
    n.now_s=10.5;n._watchdog_tick();at_boundary=n._pub.messages[-1].linear.x
    n.now_s=10.6;n._watchdog_tick();after=n._pub.messages[-1].linear.x
    n.now_s=10.61;n._cmd_cb(command());resumed=n._pub.messages[-1].linear.x
    results['paused_watchdog_and_stale_twist_restart']={'at_0_5s':at_boundary,'after_0_6s':after,'delayed_unstamped_input_resumes':resumed}
    n=A();n.enabled=False;n._cmd_cb(command());n.now_s=1.;n._cmd_cb(command());n._watchdog_tick()
    assert n._pub.messages[-1].linear.x==.2
    results['receipt_before_watchdog_hides_reset']={'output_v':n._pub.messages[-1].linear.x,'last_receipt':n._last_input_stamp_s}
    n=A();n.now_s=100.;n._watchdog_tick();assert not n._pub.messages
    results['fresh_adapter_does_not_confirm_zero']={'output_messages':len(n._pub.messages)}
    L=load_class('src/experiments/experiments/nodes/experiment_logger.py','ExperimentLogger',['_stamp_to_float','_contacts_cb','_record_collision_event'])
    def logger():
        n=L('audit');n._contact_messages_seen=0;n._contact_collision_seen=False;n._geom_collision_seen=False
        n._first_crash_stamp=math.nan;n._collision_reason='';n.finished=[]
        n._finish_run=lambda reason,s:n.finished.append({'reason':reason,'stamp':s})
        return n
    cases={
        'known_contact':contact(9.9,'turtlebot3::base_footprint::body_collision','warehouse_shell::wall::collision'),
        'known_empty_message':Contacts(),
        'ground_contact':contact(9.9,'turtlebot3::wheel_left_link::collision','ground_plane::link::collision'),
        'wrong_robot_substring':contact(9.9,'turtlebot3_clone::base::collision','warehouse_shell::wall::collision'),
        'unrelated_contact':contact(9.9,'other_robot::base::collision','warehouse_shell::wall::collision'),
        'missing_obstacle_identity':contact(9.9,'turtlebot3::base::collision',''),
        'ground_substring_masks_obstacle':contact(9.9,'turtlebot3::base::collision','not_ground_plane_rack::link::collision'),
        'old_epoch_contact':contact(100.,'turtlebot3::base::collision','warehouse_shell::wall::collision'),
        'zero_timestamp_contact':contact(0.,'turtlebot3::base::collision','warehouse_shell::wall::collision'),
    }
    for name,m in cases.items():
        n=logger();n._contacts_cb(m)
        results[name]={'messages_seen':n._contact_messages_seen,'contact_flag':n._contact_collision_seen,'finish_calls':n.finished,
                       'reason':n._collision_reason,'receipt_time_s':n.now_s,'payload_positions':len(m.contacts[0].positions) if m.contacts else 0}
    assert results['known_contact']['contact_flag'] and not results['ground_contact']['contact_flag']
    assert results['wrong_robot_substring']['contact_flag'] and results['missing_obstacle_identity']['contact_flag']
    assert results['old_epoch_contact']['finish_calls'][0]['stamp']==100.
    n=logger();results['no_messages']={'messages_seen':0,'contact_flag':n._contact_collision_seen}
    from tf2_ros import Buffer
    b=Buffer()
    def tr(parent,child,z=0.):
        t=TransformStamped();t.header.frame_id=parent;t.child_frame_id=child;t.header.stamp=stamp(10.)
        t.transform.translation.z=z;t.transform.rotation.w=1.;return t
    b.set_transform_static(tr('map_bev','odom'),'audit_static')
    b.set_transform_static(tr('base_footprint','base_link',.01),'audit_rsp')
    # Conditional after repairing the currently wrong native TF bridge topic.
    b.set_transform(tr('odom','base_link'),'audit_diffdrive')
    try:b.lookup_transform('map_bev','base_footprint',Time())
    except Exception as exc:
        results['tf_parent_conflict_after_topic_repair']={'exception':type(exc).__name__,'message':str(exc),'frames':b.all_frames_as_yaml()}
    assert 'tf_parent_conflict_after_topic_repair' in results
    sources={str(p.relative_to(SOURCE_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*SOURCE_ROOT.glob('src/sim/sim/*py'),SOURCE_ROOT/'src/experiments/experiments/nodes/experiment_logger.py']}
    result={'scope':'Synthetic callback inputs, fake transport and finish hook; no physical outcome measured','results':results,'sources':sources}
    (HERE/OUTPUT_NAME).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
