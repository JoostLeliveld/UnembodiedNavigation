"""Private Sim6 transport, pause/reset and command-hold diagnostic.

No ROS, camera rendering, shared world/topic access or blanket cleanup. All native
traffic is scoped by a freshly generated Gazebo transport partition.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from physics_probe import stop, HERE, ROOT


def main():
    token='audit14_transport_'+uuid.uuid4().hex
    name=token
    env={**os.environ,'IGN_PARTITION':token,'GZ_PARTITION':token,'IGN_IP':'127.0.0.1','GZ_IP':'127.0.0.1',
         'IGN_LOG_PATH':str(HERE/'private_logs'),'GZ_LOG_PATH':str(HERE/'private_logs')}
    root=ET.parse(HERE/'clear.fixture.sdf').getroot();w=root.find('world');w.set('name',name)
    original=ET.parse(ROOT/'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf').getroot().find('world')
    scene=next(p for p in original.findall('plugin') if 'SceneBroadcaster' in p.get('name',''))
    w.append(copy.deepcopy(scene))
    fixture=HERE/'transport.fixture.sdf';ET.ElementTree(root).write(fixture,encoding='unicode')
    outputs={};children=[];handles=[];actions=[]
    def run(args):
        started=time.monotonic()
        p=subprocess.run(['ign',*args],env=env,capture_output=True,text=True,timeout=4)
        actions.append({'args':args,'wall_monotonic':started,'duration_s':time.monotonic()-started,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        return p
    def control(req):return run(['service','-s',f'/world/{name}/control','--reqtype','ignition.msgs.WorldControl','--reptype','ignition.msgs.Boolean','--timeout','1000','--req',req])
    def subscribe(label,topic):
        f=(HERE/f'transport.{label}.jsonl').open('w');e=(HERE/f'transport.{label}.stderr.txt').open('w');handles.extend([f,e])
        cmd=['ign','topic','-e','--json-output','-t',topic,'-d','15']
        children.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=e,start_new_session=True))
        outputs[label]=topic
    log=(HERE/'transport.sim.txt').open('w');handles.append(log)
    server=None
    try:
        for label,topic in {'odom':'/model/turtlebot3/odometry','native_tf':'/model/turtlebot3/tf',
                            'configured_tf':'/model/turtlebot3/odometry_tf','joint_states':'/model/turtlebot3/joint_states',
                            'ground_truth':f'/world/{name}/dynamic_pose/info','clock':f'/world/{name}/clock'}.items():subscribe(label,topic)
        cmd=['ign','gazebo','-s','-v','4','--iterations','8000','--record-path',str(HERE/'private_logs'/token),str(fixture),'--force-version','6']
        server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(1.0)
        initial=run(['topic','-l']).stdout.splitlines()
        control('pause: false')
        time.sleep(.5)
        run(['topic','-t','/model/turtlebot3/cmd_vel','-m','ignition.msgs.Twist','-p','linear: {x: 0.2}'])
        time.sleep(.8)
        control('pause: true')
        time.sleep(.4)
        control('pause: false')
        time.sleep(.5)
        control('reset: {all: true}')
        time.sleep(.5)
        control('pause: false')
        time.sleep(.5)
    finally:
        if server:stop(server)
        for p in children:stop(p)
        for f in handles:f.close()
    (HERE/'transport_actions.json').write_text(json.dumps({'partition':token,'world':name,'command':cmd,'actions':actions,'initial_advertised_topics':initial},indent=2)+'\n')
    data={}
    for label,topic in outputs.items():
        path=HERE/f'transport.{label}.jsonl'
        records=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]
        data[label]={'topic':topic,'count':len(records),'first':records[0] if records else None,'last':records[-1] if records else None}
        if label=='odom':
            stamps=[float(r.get('header',{}).get('stamp',{}).get('sec',0))+float(r.get('header',{}).get('stamp',{}).get('nsec',0))*1e-9 for r in records]
            rewinds=[i for i in range(1,len(stamps)) if stamps[i]<stamps[i-1]]
            data[label]['rewinds']=[{'index':i,'before':records[i-1],'after':records[i]} for i in rewinds]
            data[label]['max_reported_linear_velocity']=max((r.get('twist',{}).get('linear',{}).get('x',0) for r in records),default=0.)
    result={'partition':token,'world':name,'command':cmd,'server_returncode':server.returncode,'actions':actions,'initial_advertised_topics':initial,'streams':data,
            'scope':'Single native Twist and private physics fixture; no planner or adapter was connected. A held target after source silence is a downstream property.'}
    (HERE/'transport_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'partition':token,'server_returncode':server.returncode,'streams':{k:{'count':v['count'],'topic':v['topic']} for k,v in data.items()},'actions':actions},indent=2))


if __name__=='__main__':main()
