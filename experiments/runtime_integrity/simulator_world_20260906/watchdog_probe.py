"""Bounded private ROS/native watchdog probe; no camera renderer or campaign.

All ROS data topics, node names, Gazebo partition and world are unique. Only
owned process groups are signalled. Robot/colliders, physics and active adapter
noise defaults are unchanged. Ground truth is recorded, never used to drive.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

from physics_probe import HERE, ROOT, stop


def main():
    token='audit14_watchdog_'+uuid.uuid4().hex
    prefix='/'+token
    env={**os.environ,'IGN_PARTITION':token,'GZ_PARTITION':token,
         'IGN_IP':'127.0.0.1','GZ_IP':'127.0.0.1','ROS_LOCALHOST_ONLY':'1',
         'ROS_DOMAIN_ID':'227','ROS_LOG_DIR':str(HERE/'private_logs'/token/'ros'),
         'IGN_LOG_PATH':str(HERE/'private_logs'),'GZ_LOG_PATH':str(HERE/'private_logs')}
    os.environ.update(env)
    import rclpy
    from ament_index_python.packages import get_package_prefix
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    fixture=HERE/'watchdog.fixture.sdf'
    root=ET.parse(HERE/'transport.fixture.sdf').getroot()
    root.find('world').set('name',token)
    ET.ElementTree(root).write(fixture,encoding='unicode')
    children=[];handles=[];commands=[];events=[];actions=[];phases=[]
    last={};clock_bridge=None;noise=None;observer=None
    def executable(pkg,name):return str(Path(get_package_prefix(pkg))/'lib'/pkg/name)
    def launch(label,cmd):
        f=(HERE/f'watchdog.{label}.txt').open('w');handles.append(f)
        p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);commands.append({'label':label,'pid':p.pid,'cmd':cmd});return p
    def bridge(label,topic,kind,ros_topic,direction='['):
        return launch(label,[executable('ros_gz_bridge','parameter_bridge'),
                    topic+'@'+kind[0]+direction+kind[1],
                    '--ros-args','-r','__node:='+token+'_'+label,'-r',topic+':='+ros_topic])
    def adapter(label):
        return launch(label,[executable('sim','actuation_noise_node'),'--ros-args',
                    '-r','__node:='+token+'_'+label,'-r','/clock:='+prefix+'/clock',
                    '-p','use_sim_time:=true','-p','seed:=210',
                    '-p','input_topic:='+prefix+'/cmd_input','-p','output_topic:='+prefix+'/cmd_output',
                    '-p','diagnostics_topic:='+prefix+'/diagnostics'])
    def capture(kind,value):
        record={'kind':kind,'receipt_monotonic_s':time.monotonic(),**value}
        last[kind]=record;events.append(record)
    def seconds(stamp):return stamp.sec+stamp.nanosec*1e-9
    def pump(duration):
        end=time.monotonic()+duration
        while time.monotonic()<end:rclpy.spin_once(observer,timeout_sec=.02)
    def mark(label):
        phases.append({'label':label,'wall_monotonic_s':time.monotonic(),
                       'last':copy.deepcopy(last),'event_count':len(events)})
    def command():
        m=Twist();m.linear.x=.2;pub.publish(m)
        actions.append({'action':'ROS Twist input','wall_monotonic_s':time.monotonic(),'linear_x':.2,'angular_z':0.})
    def control(req):
        cmd=['ign','service','-s',f'/world/{token}/control','--reqtype','ignition.msgs.WorldControl',
             '--reptype','ignition.msgs.Boolean','--timeout','1000','--req',req]
        start=time.monotonic();p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=4)
        actions.append({'action':'native control','request':req,'wall_monotonic_s':start,
                        'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        assert p.returncode==0 and 'data: true' in p.stdout,p.stdout+p.stderr
    try:
        raw=(HERE/'watchdog.ground_truth.jsonl').open('w');handles.append(raw)
        children.append(subprocess.Popen(['ign','topic','-e','--json-output','-t',f'/world/{token}/dynamic_pose/info','-d','30'],
                                         env=env,stdout=raw,stderr=subprocess.DEVNULL,start_new_session=True))
        launch('sim',['ign','gazebo','-s','-v','4','--iterations','18000','--record-path',str(HERE/'private_logs'/token),str(fixture),'--force-version','6'])
        clock_bridge=bridge('clock_bridge',f'/world/{token}/clock',('rosgraph_msgs/msg/Clock','gz.msgs.Clock'),prefix+'/clock_full')
        bridge('cmd_bridge','/model/turtlebot3/cmd_vel',('geometry_msgs/msg/Twist','gz.msgs.Twist'),prefix+'/cmd_output',']')
        bridge('odom_bridge','/model/turtlebot3/odometry',('nav_msgs/msg/Odometry','gz.msgs.Odometry'),prefix+'/odom')
        launch('clock_throttle',[executable('sim','clock_throttle_node'),'--ros-args',
                    '-r','__node:='+token+'_clock_throttle','-p','input_topic:='+prefix+'/clock_full','-p','output_topic:='+prefix+'/clock'])
        noise=adapter('adapter_initial')
        rclpy.init()
        observer=rclpy.create_node(token+'_observer',enable_rosout=False)
        pub=observer.create_publisher(Twist,prefix+'/cmd_input',1)
        observer.create_subscription(Twist,prefix+'/cmd_output',lambda m:capture('output',{'v':m.linear.x,'w':m.angular.z}),10)
        observer.create_subscription(Clock,prefix+'/clock',lambda m:capture('clock',{'stamp_s':seconds(m.clock)}),QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
        observer.create_subscription(Odometry,prefix+'/odom',lambda m:capture('odom',{'stamp_s':seconds(m.header.stamp),'frame':m.header.frame_id,'child':m.child_frame_id,'x':m.pose.pose.position.x,'y':m.pose.pose.position.y,'v':m.twist.twist.linear.x,'w':m.twist.twist.angular.z}),10)
        pump(2.);control('pause: false');pump(.8)
        assert 'odom' in last and 'clock' in last and observer.count_subscribers(prefix+'/cmd_input')==1
        mark('ready')
        command();pump(.9);mark('ordinary_silence_after_0_9_wall_s')
        assert last['output']['v']==0. and last['output']['w']==0.,'ordinary watchdog did not stop'
        command();pump(.08);control('pause: true');pump(.7);mark('paused_after_command')
        control('pause: false');pump(.7);mark('resumed_after_pause')
        command();pump(.10);mark('before_clock_bridge_loss')
        os.killpg(clock_bridge.pid,signal.SIGSTOP)
        actions.append({'action':'SIGSTOP owned clock bridge','pid':clock_bridge.pid,'wall_monotonic_s':time.monotonic()})
        pump(1.0);mark('clock_bridge_lost_physics_running')
        os.killpg(clock_bridge.pid,signal.SIGCONT)
        actions.append({'action':'SIGCONT owned clock bridge','pid':clock_bridge.pid,'wall_monotonic_s':time.monotonic()})
        pump(.8);mark('clock_bridge_restored')
        command();pump(.10);mark('before_reset');control('reset: {all: true}');pump(.25);control('pause: false');pump(.7);mark('after_reset_no_new_command')
        command();pump(.10);mark('before_adapter_death')
        stop(noise);noise=adapter('adapter_restarted');pump(1.2);mark('fresh_adapter_without_new_input')
        pub.publish(Twist());pump(.3);mark('explicit_final_zero')
    finally:
        if clock_bridge and clock_bridge.poll() is None:
            os.killpg(clock_bridge.pid,signal.SIGCONT)
        if observer is not None:observer.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        for p in reversed(children):stop(p)
        for f in handles:f.close()
        (HERE/'watchdog.events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
        results={'scope':'Private native physics plus real ROS bridges, clock throttle and noisy adapter; no planner, rendering or existing campaign resources.',
                 'partition':token,'world':token,'ros_domain':227,'ros_topic_prefix':prefix,
                 'commands':commands,'actions':actions,'phases':phases,
                 'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
                 'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/sim/sim/actuation_noise_node.py','src/sim/sim/clock_throttle_node.py']},
                 'raw_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'watchdog.events.jsonl',HERE/'watchdog.ground_truth.jsonl']}}
        (HERE/'watchdog_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'partition':token,'phases':phases},indent=2))


if __name__=='__main__':main()
