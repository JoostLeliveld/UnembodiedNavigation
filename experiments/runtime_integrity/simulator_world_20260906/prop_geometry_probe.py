"""Independent 2-D audit of collision-bearing included props in active world."""
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
asset=json.loads((HERE/'asset_results.json').read_text())
world=ET.parse(asset['world_path']).getroot().find('world')
models={m['name']:m for m in asset['models']}


def local_boxes(model):
    root=ET.parse(model['path']).getroot().find('model');out=[]
    for link in root.findall('link'):
        lp=[float(x) for x in link.findtext('pose','0 0 0 0 0 0').split()]
        for c in link.findall('collision'):
            cp=[float(x) for x in c.findtext('pose','0 0 0 0 0 0').split()]
            sz=c.findtext('geometry/box/size')
            if sz:
                sx,sy,sz=map(float,sz.split());yaw=lp[5]+cp[5]
                g=rotate(box(-sx/2,-sy/2,sx/2,sy/2),yaw*180/math.pi,use_radians=False)
                g=translate(g,lp[0]+cp[0],lp[1]+cp[1]);out.append((g,lp[2]+cp[2]-sz/2,lp[2]+cp[2]+sz/2,c.get('name')))
    return out


def clip_z(poly, bound, keep_above):
    out=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        ain=a[2]>=bound if keep_above else a[2]<=bound
        bin=b[2]>=bound if keep_above else b[2]<=bound
        if ain:out.append(a)
        if ain != bin:
            t=(bound-a[2])/(b[2]-a[2])
            out.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),bound))
    return out


def mesh_triangles(model):
    root=ET.parse(model['path']).getroot().find('model');out=[]
    for link in root.findall('link'):
        lp=[float(x) for x in link.findtext('pose','0 0 0 0 0 0').split()]
        for c in link.findall('collision'):
            uri=c.findtext('geometry/mesh/uri')
            if not uri:continue
            name=uri.split('/')[2];rel='/'.join(uri.split('/')[3:])
            base=next(m['realpath'] for m in asset['models'] if Path(m['realpath']).parent.name==name)
            path=Path(base).parent/rel
            r=ET.parse(path).getroot();ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
            unit=float(r.find('c:asset/c:unit',ns).get('meter','1'))
            a=next(a for a in r.findall('.//c:float_array',ns) if 'POSITION-array' in a.get('id',''))
            vals=list(map(float,a.text.split()));xyz=list(zip(vals[::3],vals[1::3],vals[2::3]))
            xyz=[(x*unit+lp[0],y*unit+lp[1],z*unit+lp[2]) for x,y,z in xyz]
            triangles=[]
            for tris in r.findall('.//c:triangles',ns):
                inputs=tris.findall('c:input',ns);stride=1+max(int(i.get('offset','0')) for i in inputs)
                voff=int(next(i for i in inputs if i.get('semantic')=='VERTEX').get('offset','0'))
                idx=list(map(int,tris.findtext('c:p',namespaces=ns).split()))
                for k in range(0,len(idx),3*stride):
                    poly=[xyz[idx[k+j*stride+voff]] for j in range(3)]
                    poly=clip_z(clip_z(poly,.07,True),.34,False)
                    if len(poly)>=3:
                        g=Polygon([(q[0],q[1]) for q in poly])
                        if g.area>0:triangles.append(g)
            if triangles:out.append((unary_union(triangles),.07,.34,c.get('name')))
    return out


drive=unary_union([box(p['xmin'],p['ymin'],p['xmax'],p['ymax']) for p in asset['driveable_geometry']['prisms']])
run=ROOT/'logs/studies/icra_commissioning_20260905/network_navigation_runtime_pilot/fusion_network_traverse/P0/seed210/experiment_20260906_213015'
with (run/'global_plan.csv').open() as f: route=LineString([(float(r['x']),float(r['y'])) for r in csv.DictReader(f)])
body=box(-.4,-.275,.4,.275)
records=[]
for name in ['forklift_parked','pallet_jack','bin_office','pallet_loose_1','pallet_loose_2']:
    model=models[name]; geoms=local_boxes(model)+mesh_triangles(model)
    yaw=model['pose'][5]; parts=[]
    for g,z0,z1,cname in geoms:
        g=translate(rotate(g,yaw*180/math.pi,use_radians=False),model['pose'][0],model['pose'][1])
        if z0+model['pose'][2] <= .34 and z1+model['pose'][2] >= .07:parts.append(g)
    footprint=unary_union(parts)
    spawn=translate(body,-7.9,-8.7)
    intersections=[];mind=math.inf
    for i in range(721):
        b=translate(rotate(body,i*.5,use_radians=False),-7.9,-8.7)
        mind=min(mind,b.distance(footprint));intersections.append(b.intersects(footprint))
    records.append({'name':name,'source_model':Path(model['realpath']).parent.name,'include_pose':model['pose'],
                    'collision_part_count_at_robot_height':len(parts),'collision_footprint_bounds':list(footprint.bounds),
                    'driveable_intersection_area_m2':footprint.intersection(drive).area,
                    'spawn_body_clearance_m':spawn.distance(footprint),'spawn_any_half_degree_yaw_intersection':any(intersections),
                    'spawn_min_half_degree_yaw_clearance_m':mind,'selected_plan_centerline_clearance_m':route.distance(footprint),
                    'selected_plan_circle_clearance_m':route.distance(footprint)-math.hypot(.4,.275)})
result={'scope':'Exact clipping/projecting of DAE collision triangles through the robot vertical slab; exact 2-D boxes for SDF box collisions. Static included props only; lamps are above robot and omitted.',
        'planner_collision_models':['warehouse_shell','warehouse_v2_occluders'],'records':records}
(HERE/'prop_geometry_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
