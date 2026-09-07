"""Small private simulator probe, no ROS, rendering, campaign or global cleanup.

Extracts unchanged active ground and colliders plus converted warehouse robot.
Uses a unique Gazebo partition, a unique world name, bounded own child groups.
This is a diagnostic fixture, never an experimental warehouse replacement.
"""
from __future__ import annotations
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)


def main():
    token='audit14_'+uuid.uuid4().hex
    env={**os.environ,'IGN_PARTITION':token,'GZ_PARTITION':token,'IGN_IP':'127.0.0.1','GZ_IP':'127.0.0.1',
         'IGN_LOG_PATH':str(HERE/'private_logs'), 'GZ_LOG_PATH':str(HERE/'private_logs')}
    original=ET.parse(ROOT/'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf').getroot().find('world')
    robot=ET.parse(HERE/'warehouse_amr.converted.sdf').getroot().find('model')
    results=[]
    for case,xyz in [('clear',[-7.9,-8.7,.05]),('contact',[-5.31,5.393,.05])]:
        name=token+'_'+case
        root=ET.Element('sdf',version='1.7');world=ET.SubElement(root,'world',name=name)
        for c in original:
            if c.tag=='physics' or c.tag=='plugin' and any(v in c.get('name','') for v in ['Physics','UserCommands','Contact']):
                world.append(copy.deepcopy(c))
        for model_name in ['ground_plane','warehouse_shell','warehouse_v2_occluders']:
            m=copy.deepcopy(original.find(f"model[@name='{model_name}']"))
            for link in list(m.findall('link')):
                for vis in list(link.findall('visual')):link.remove(vis)
                if link.find('collision') is None:m.remove(link)
            world.append(m)
        r=copy.deepcopy(robot);r.set('name','turtlebot3')
        p=r.find('pose')
        if p is None:p=ET.SubElement(r,'pose')
        p.text=' '.join(map(str,[*xyz,0,0,0]))
        world.append(r)
        path=HERE/f'{case}.fixture.sdf';ET.ElementTree(root).write(path,encoding='unicode')
        topic=f'/world/{name}/model/warehouse_v2_occluders/link/obs_A2b3e/sensor/contact/contact'
        # A sensor emits no empty samples in some versions. Presence is separately
        # checked via own partition topic listing, and clear must not be inferred
        # solely from subscriber silence.
        echo_cmd=['ign','topic','-e','--json-output','-t',topic,'-d','5']
        sim_cmd=['ign','gazebo','-r','-s','-v','4','--iterations','2500','--record-path',str(HERE/'private_logs'/name),str(path),'--force-version','6']
        out=HERE/f'{case}.sim.txt';raw=HERE/f'{case}.contacts.jsonl';err=HERE/f'{case}.subscriber.stderr.txt'
        with out.open('w') as so, raw.open('w') as eo, err.open('w') as ee:
            ep=subprocess.Popen(echo_cmd,env=env,stdout=eo,stderr=ee,start_new_session=True)
            sp=None
            try:
                sp=subprocess.Popen(sim_cmd,env=env,stdout=so,stderr=subprocess.STDOUT,start_new_session=True)
                sp.wait(timeout=15)
                ep.wait(timeout=7)
            except subprocess.TimeoutExpired:
                pass
            finally:
                if sp is not None:stop(sp)
                stop(ep)
        results.append({'case':case,'partition':token,'world':name,'robot_pose':xyz,'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'command':sim_cmd,'returncode':sp.returncode if sp else None,'sensor_topic':topic,'subscriber_command':echo_cmd,
                        'subscriber_returncode':ep.returncode,'contact_bytes':raw.stat().st_size,
                        'errors':[line for line in out.read_text(errors='replace').splitlines() if any(s in line.lower() for s in ['error','failed','permission','physics engine','loaded system'])][:35]})
    (HERE/'physics_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
