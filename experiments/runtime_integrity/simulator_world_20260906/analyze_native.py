"""Summarize only Audit 14's private native streams, preserving source identities."""
import json
import hashlib
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(label):
    return [json.loads(l) for l in (HERE/label).read_text().splitlines() if l.startswith('{')]


def seconds(s):return float(s.get('sec',0))+float(s.get('nsec',0))*1e-9
def header(r):return seconds(r.get('header',{}).get('stamp',{}))


def rate(records):
    stamps=[header(r) for r in records]
    dt=[b-a for a,b in zip(stamps,stamps[1:]) if b>a+1e-8]
    return {'count':len(records),'first_stamp_s':stamps[0] if stamps else None,'last_stamp_s':stamps[-1] if stamps else None,
            'median_positive_interval_s':statistics.median(dt) if dt else None,
            'min_positive_interval_s':min(dt) if dt else None,'max_positive_interval_s':max(dt) if dt else None,
            'unique_positive_intervals_s':sorted(set(round(x,6) for x in dt))[:30]}


def robot(r):return next(p for p in r['pose'] if p.get('name')=='turtlebot3')


def main():
    contact=read('contact.contacts.jsonl');clear=read('clear.contacts.jsonl')
    gt=read('transport.ground_truth.jsonl');od=read('transport.odom.jsonl');clock=read('transport.clock.jsonl')
    grewind=next(i for i in range(1,len(gt)) if header(gt[i])<header(gt[i-1]))
    orewind=next(i for i in range(1,len(od)) if header(od[i])<header(od[i-1]))
    groups=[];current=[]
    for r in clock:
        if current and seconds(r['sim'])!=seconds(current[-1]['sim']):groups.append(current);current=[]
        current.append(r)
    groups.append(current)
    paused=max((g for g in groups if seconds(g[0]['sim'])>0),key=lambda g:seconds(g[-1]['system'])-seconds(g[0]['system']))
    recs={}
    for label,records in [('contact',contact),('ground_truth',gt),('odom',od),('native_tf',read('transport.native_tf.jsonl'))]:
        recs[label]=rate(records)
    recs['native_tf']['rate_note']='stamp is on each pose, not the Pose_V outer header'
    out={'contact_stream':recs['contact'],'clear_stream_count':len(clear),'first_contact_payload':contact[0],
         'last_contact_payload':contact[-1],'native_cadence':{k:v for k,v in recs.items() if k!='contact'},
         'reset':{'gt_before':{'stamp':header(gt[grewind-1]),'robot':robot(gt[grewind-1])},
                  'gt_after':{'stamp':header(gt[grewind]),'robot':robot(gt[grewind])},
                  'gt_last':{'stamp':header(gt[-1]),'robot':robot(gt[-1])},
                  'odom_before':od[orewind-1],'odom_after':od[orewind],'odom_last':od[-1]},
         'pause':{'sim_stamp_s':seconds(paused[0]['sim']),'native_clock_messages':len(paused),
                  'system_clock_span_s':seconds(paused[-1]['system'])-seconds(paused[0]['system'])},
         'raw_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.jsonl')}}
    assert len(contact)==2500 and len(clear)==0
    assert abs(recs['contact']['median_positive_interval_s']-.001)<1e-10
    assert contact[0]['contact'][0]['collision2']['name'].startswith('turtlebot3::base_footprint::')
    assert abs(od[-1]['twist']['linear']['x']-.2)<1e-6
    (HERE/'native_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
