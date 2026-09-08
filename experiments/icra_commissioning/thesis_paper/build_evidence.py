#!/usr/bin/env python3
"""Build thesis tables/figures from explicitly selected, checked evidence."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/thesis_mpl')
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
# Lives in UnembodiedNavigation so it stays version-controlled with the study
# artifacts it reads; it WRITES into the paper repo, which carries only what
# the document compiles from. REPO is this repo, BASE is papers/Thesis.
REPO=Path(__file__).resolve().parents[3]
BASE=REPO.parent/'papers'/'Thesis'
sys.path.insert(0,str(REPO/'experiments/icra_commissioning'))
from field_study import load_data,FIELD_OUT
from field_driving import load_run
from replay import aligned
import xml.etree.ElementTree as ET
import joblib
OUT=REPO/'logs/studies/icra_commissioning_20260905';E=OUT/'thesis_evidence'
G=BASE/'generated';F=BASE/'figures';G.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':8,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})
COL=['#2367a2','#d27a24','#25856a','#9a4b8b','#697581','#c23d36','#755c3b']
NAMES={'constant':'Full constant','diagonal':'Diagonal','isotropic':'Isotropic','geometry':'Geometry',
 'spatial':'Spatial','confidence':'Calibrated score','confidence_bias':'Persistent bias (diagnostic)',
 'geometry_xy':'Geometry XY','geometry_heading':'Geometry + heading','local_xy':'Local XY',
 'local_heading':'Local + heading','gp_xy':'GP','gp_integrated':'GP integrated','local_joint':'Joint local'}
sources={}
def read(p):
 sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def savefig(fig,name):
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.svg'));plt.close(fig)
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def table(name,headers,rows,fmt=None):
 fmt=fmt or 'l'+'r'*(len(headers)-1)
 text='\\begin{tabular}{@{}'+fmt+'@{}}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
 text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
 (G/(name+'.tex')).write_text(text)
def box(ax,xy,w,h,title,body,color=COL[0]):
 x,y=xy;ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc='#f5f8fa',ec=color,lw=1))
 ax.text(x+w/2,y+h*.74,title,ha='center',va='center',fontsize=11,fontweight='bold',color=color)
 ax.text(x+w/2,y+h*.35,body,ha='center',va='center',fontsize=9,linespacing=1.5)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.4,color='#4c5660'))

# Architecture: intentionally distinguishes implementation from unestablished outcome.
fig,ax=plt.subplots(figsize=(10.2,5.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,(.02,.7),.28,.25,'IWAI: published study','Detector-score GP\nEFE route choice',COL[4])
box(ax,(.37,.7),.28,.25,'Commissioning','Reference residuals + misses\nGrouped fitting and selection')
box(ax,(.72,.7),.26,.25,'Two sensor interfaces','Current observation\nFuture configuration')
box(ax,(.05,.23),.26,.26,'Measurement + covariance','Frozen corrected position\nConditional error model')
box(ax,(.38,.23),.25,.26,'Sequential estimation','Fresh multi-camera updates\nAccuracy and consistency')
box(ax,(.71,.23),.27,.26,'Future belief prediction','Availability + quality outcomes\nHeld-out route evaluation')
arrow(ax,(.3,.82),(.37,.82));arrow(ax,(.65,.82),(.72,.82));arrow(ax,(.50,.70),(.18,.49));arrow(ax,(.84,.70),(.84,.49));arrow(ax,(.31,.36),(.38,.36));arrow(ax,(.63,.36),(.71,.36))
ax.text(.5,.055,'New closed-loop navigation benefit requires a separate matched comparison.',ha='center',fontsize=10,color=COL[5])
savefig(fig,'system_overview')
fig,ax=plt.subplots(figsize=(10.2,4.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
for x,title,body in [(.02,'Image + identity','Capture stamp\nCamera and batch ID'),(.36,'Frozen detector','Original-image bbox\nConfidence; preserve misses'),(.70,'Ground projection','BBox bottom centre\nIntermediate ground point')]:box(ax,(x,.66),.28,.27,title,body)
arrow(ax,(.30,.79),(.36,.79));arrow(ax,(.64,.79),(.70,.79))
box(ax,(.70,.12),.28,.30,'Frozen metric correction','Raw ray-frame NN offset\nRobot-reference XY (m)')
box(ax,(.36,.12),.28,.30,'Residual calibration','Shared mean offset\nConditional ground R (m²)')
box(ax,(.02,.12),.28,.30,'Robot filter','One fresh update per frame\nNo update on a miss')
arrow(ax,(.84,.66),(.84,.42));arrow(ax,(.70,.27),(.64,.27));arrow(ax,(.36,.27),(.30,.27))
ax.text(.50,.52,'Reference pose: offline training and evaluation only',ha='center',fontsize=10,color=COL[5]);savefig(fig,'perception_contract')
fig,ax=plt.subplots(figsize=(7.7,3.5));q=np.linspace(0,1,301)
for j,R in enumerate([.01,.1,1.]):
 ax.plot(q,q*R/(1+R)+(1-q),color=COL[j],label=f'Branch average, R={R:g}')
 ax.plot(q,1/(1+q/R),'--',color=COL[j],label=f'Average information, R={R:g}')
ax.set(xlabel='Probability of receiving one measurement',ylabel='Posterior variance / prior variance',ylim=(0,1.03));ax.legend(ncol=2);ax.grid(alpha=.15);savefig(fig,'branch_averaging')

static=read(E/'static.json');selection=read(E/'selection.json')
data,geometry,constant=load_data(FIELD_OUT);field=joblib.load(FIELD_OUT/'field.joblib')
for p in [FIELD_OUT/'manifest.json',FIELD_OUT/'field.joblib',OUT/'manifest.json',OUT/'models.joblib']:
 sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
ev=data['role']=='evaluation';poses=data['pose'][ev];hits=data['hits'][ev];groups=data['group'][ev]
fig,axes=plt.subplots(2,5,figsize=(11.2,5.1),sharex=True,sharey=True,layout='constrained')
for j,c in enumerate('ABCDE'):
 for row,key in enumerate(['raw_error','error']):
  ax=axes[row,j];err=data[key][ev,j];valid=np.isfinite(err).all(1);xy=poses[valid,:2];ee=err[valid]
  # Heading-averaged error at each independent sampled XY; vectors scaled 10x.
  up=np.unique(xy,axis=0);means=np.array([ee[np.all(xy==p,axis=1)].mean(0) for p in up])
  ax.scatter(up[:,0],up[:,1],s=4,color='#d8dfe4')
  z=ax.quiver(up[:,0],up[:,1],means[:,0],means[:,1],angles='xy',scale_units='xy',scale=.1,color=COL[row],width=.004)
  ax.set(aspect='equal',xlim=(-12,13),ylim=(-11,12));ax.set_title(f'Camera {c}; n={valid.sum()}')
  if j==0:ax.set_ylabel(('Raw' if row==0 else 'Corrected + offset')+'\nY (m)')
  if row==1:ax.set_xlabel('X (m)')
  if j==4:ax.quiverkey(z,.7,.08,.1,'10 cm',coordinates='axes',labelpos='N',fontproperties={'size':8})
savefig(fig,'static_bias')
fig,axes=plt.subplots(1,2,figsize=(9.7,3.6),layout='constrained')
for j,k in enumerate(['constant','geometry','confidence']):
 d=static['covariance'][k];xx=np.array(list(d['coverage']),float);yy=np.array(list(d['coverage'].values()))
 axes[0].plot(xx*100,yy*100,'o-',color=COL[j],label=NAMES[k])
axes[0].plot([50,100],[50,100],'k--',lw=1);axes[0].set(xlabel='Nominal ellipse probability (%)',ylabel='Empirical containment (%)');axes[0].legend()
kinds=list(static['covariance']);vals=[static['covariance'][k]['group_mean_nll'] for k in kinds]
ci=np.array([static['covariance'][k]['group_bootstrap_nll_ci95'] for k in kinds]);axes[1].errorbar(vals,np.arange(len(vals)),xerr=np.maximum(0,np.array([np.array(vals)-ci[:,0],ci[:,1]-np.array(vals)])),fmt='o',color=COL[0],capsize=3)
axes[1].set(yticks=np.arange(len(kinds)),yticklabels=[NAMES[k] for k in kinds],xlabel='Equal-tile Gaussian score (lower is better)');axes[1].invert_yaxis()
savefig(fig,'static_covariance')
table('static_covariance',['Model','Median (cm)','95th pct. (cm)','95\\% coverage'],[[NAMES[k],f"{d['median_cm']:.2f}",f"{d['p95_cm']:.2f}",f"{100*d['coverage']['0.95']:.1f}\\%"] for k,d in static['covariance'].items()])

fig,axes=plt.subplots(1,2,figsize=(9.7,3.9),layout='constrained');kinds=list(static['availability'])
rng=np.random.default_rng(609);avrows=[]
for j,k in enumerate(kinds):
 d=static['availability'][k];tilevals=np.array(list(d['tile_scores'].values()));boot=rng.choice(tilevals,(2000,len(tilevals)),replace=True).mean(1);lo,hi=np.quantile(boot,[.025,.975]);v=d['equal_tile_brier']
 axes[0].errorbar(v,j,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=COL[j%7],capsize=3)
 avrows.append([NAMES[k],f'{v:.4f}',f'{lo:.4f}--{hi:.4f}'])
axes[0].set(yticks=np.arange(len(kinds)),yticklabels=[NAMES[k] for k in kinds],xlabel='Equal-tile Brier score (lower is better)');axes[0].invert_yaxis()
for j,k in enumerate(['constant','geometry_xy','gp_xy']):
 p=field.availability[k].predict(poses).ravel();target=hits.ravel();xs=[];ys=[]
 for lo in np.arange(0,1,.1):
  keep=(p>=lo)&(p<lo+.1)
  if keep.sum()>=15:xs.append(p[keep].mean());ys.append(target[keep].mean())
 axes[1].plot(xs,ys,'o-',color=COL[j],label=NAMES[k])
axes[1].plot([0,1],[0,1],'k--',lw=1);axes[1].set(xlabel='Mean predicted hit probability',ylabel='Observed hit fraction',xlim=(0,1),ylim=(0,1));axes[1].legend()
savefig(fig,'availability_calibration');table('static_availability',['Model','Brier score','Tile-bootstrap interval'],avrows)
fig,axes=plt.subplots(2,5,figsize=(11.2,5),sharex=True,sharey=True,layout='constrained')
pred=field.availability['gp_xy'].predict(poses)
for j,c in enumerate('ABCDE'):
 xy=poses[:,:2];unique=np.unique(xy,axis=0)
 for r,values in enumerate([hits[:,j].astype(float),pred[:,j]]):
  vals=np.array([values[np.all(xy==p,axis=1)].mean() for p in unique]);ax=axes[r,j]
  im=ax.scatter(unique[:,0],unique[:,1],c=vals,cmap='viridis',vmin=0,vmax=1,s=18)
  ax.set(aspect='equal',title=f'Camera {c}',xlim=(-12,13),ylim=(-11,12))
  if j==0:ax.set_ylabel(('Observed' if r==0 else 'Predicted GP')+'\nY (m)')
  if r==1:ax.set_xlabel('X (m)')
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Heading-averaged availability');savefig(fig,'availability_maps')

# Driving artifacts are loaded by explicit manifest, never by listing a result directory.
runs=[]
for entry in selection['runs']:
 p=E/'driving'/entry['key']/'results.json'
 if not p.exists():continue
 load_run(entry)
 r=read(p);r['_key']=entry['key'];runs.append(r)
if not runs:raise RuntimeError('No verified analyzed drive available')
methods=['constant','geometry','confidence','confidence_bias'];rows=[]
for r in runs:
 for k in methods:
  d=r['replay']['full/'+k];rows.append([('Overlap' if r['task']=='fusion_overlap_rich' else 'Traverse')+f" {r['seed']}",NAMES[k],f"{d['median_cm']:.2f}",f"{d['p95_cm']:.2f}",f"{100*d['coverage']['0.95']:.1f}"])
table('driving_results',['Run','Model','Median','95th pct.','Coverage'],rows,fmt='llrrr')
account=[]
for r in runs:
 a=r['accounting'];account.append([('Overlap' if r['task']=='fusion_overlap_rich' else 'Traverse')+f" {r['seed']}",a['fresh_readings'],a['batches'],f"{100*a['dropped_fraction']:.2f}",f"{a['longest_gap_s']:.1f}",tex(a['outcome'])])
table('accounting',['Run','Readings','Batches','Dropped (\\%)','Gap (s)','Outcome'],account,'lrrrrl')
fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),layout='constrained')
for j,r in enumerate(runs):
 label=('Overlap' if r['task']=='fusion_overlap_rich' else 'Traverse')+f" {r['seed']}"
 axes[0].plot(range(4),[r['replay']['full/'+k]['median_cm'] for k in methods],'-o',color=COL[j%7],label=label)
 axes[1].plot(range(4),[100*r['replay']['full/'+k]['coverage']['0.95'] for k in methods],'-o',color=COL[j%7])
for ax in axes:ax.set(xticks=range(4),xticklabels=['Constant','Geometry','Score','Bias state']);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Run-level median position error (cm)');axes[0].legend(fontsize=7)
axes[1].set_ylabel('95% ellipse containment (%)');axes[1].axhline(95,color='k',ls='--',lw=1)
savefig(fig,'driving_comparison')
fig,axes=plt.subplots(1,5,figsize=(11.2,2.7),sharey=True,layout='constrained')
for j,c in enumerate('ABCDE'):
 for n,r in enumerate(runs):
  lag=[v for v in r['temporal'] if v['camera']=='camera_'+c]
  axes[j].plot([v['lag_s'] for v in lag],[np.mean(v['correlation']) for v in lag],'-o',color=COL[n%7],ms=3)
 axes[j].set(title=f'Camera {c}',xlabel='Lag (s)',ylim=(-.2,1.05));axes[j].axhline(0,color='k',lw=.5)
axes[0].set_ylabel('Mean component residual correlation');savefig(fig,'temporal_correlation')
fig,axes=plt.subplots(1,2,figsize=(9.5,3.6),layout='constrained');frm=[]
for n,r in enumerate(runs):
 pts=[v for v in r['forecasts'] if v['horizon_s']==5 and v['method']=='gp_xy' and v['approximation']=='branch']
 x=np.sqrt([v['predicted_trace_m2'] for v in pts])*100;y=np.sqrt([v['actual_filter_trace_m2'] for v in pts])*100
 axes[0].scatter(x,y,s=12,alpha=.5,color=COL[n%7],label=('Overlap' if r['task']=='fusion_overlap_rich' else 'Traverse')+f" {r['seed']}")
 for k in ['constant','geometry_xy','gp_xy','local_joint']:
  p=[v for v in r['forecasts'] if v['horizon_s']==5 and v['method']==k and v['approximation']=='branch']
  if p:
   ratio=np.mean([v['predicted_trace_m2'] for v in p])/np.mean([v['actual_squared_error_m2'] for v in p])
   axes[1].scatter(['constant','geometry_xy','gp_xy','local_joint'].index(k),ratio,s=28,color=COL[n%7]);frm.append([r['_key'],k,ratio])
lim=max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]);axes[0].plot([0,lim],[0,lim],'k--',lw=1)
axes[0].set(xlabel='Predicted root position trace (cm)',ylabel='Realized filter root position trace (cm)');axes[0].legend(fontsize=7)
axes[1].axhline(1,color='k',ls='--',lw=1);axes[1].set(xticks=range(4),xticklabels=['Constant','Geometry','GP','Joint local'],ylabel='Mean predicted trace / mean squared error',yscale='log')
savefig(fig,'forecast_diagnostic')
# Show all individual-camera baselines; do not select a visually favorable camera.
fig,ax=plt.subplots(figsize=(8.5,3.4),layout='constrained')
for j,r in enumerate(runs):
 keys=['single/camera_'+c for c in 'ABCDE']+['full/constant'];ax.plot(range(6),[r['replay'][k]['p95_cm'] for k in keys],'-o',color=COL[j%7],label=('Overlap' if r['task']=='fusion_overlap_rich' else 'Traverse')+f" {r['seed']}")
ax.set(xticks=range(6),xticklabels=['A','B','C','D','E','All cameras'],ylabel='Run-level 95th-percentile error (cm)',yscale='log');ax.legend(ncol=2,fontsize=7);ax.grid(axis='y',alpha=.15);savefig(fig,'camera_complementarity')

# Recompute the residual-calibration budget from grouped input, retaining the historical output for comparison.
from study import load as load_static, score
from model import CameraModel
rawdata,_=load_static(OUT);fitrows=[r for r in rawdata if r['role']=='covariance_fit'];evalrows=[r for r in rawdata if r['role']=='evaluation']
fitgroups=sorted({r['group'] for r in fitrows});budget=[]
for fraction in [.25,.5,1.]:
 for seed in range(5):
  chosen=set(np.random.default_rng(seed).choice(fitgroups,max(5,int(fraction*len(fitgroups))),replace=False))
  subset=[r for r in fitrows if r['group'] in chosen]
  if any(sum(r['camera']==c for r in subset)<15 for c in ['camera_'+x for x in 'ABCDE']):continue
  mm={c:CameraModel('constant').fit([r for r in subset if r['camera']==c]) for c in ['camera_'+x for x in 'ABCDE']}
  errors=[];Cs=[]
  for r in evalrows:
   z,C=mm[r['camera']].predict([r]);errors.append(z[0]-r['truth']);Cs.append(C[0])
  budget.append(dict(fraction=fraction,seed=seed,poses=len({tuple(r['truth']) for r in subset}),score=score(np.array(errors),np.array(Cs),[r['group'] for r in evalrows])))
previous=read(OUT/'results.json')
for a,b in zip(budget,previous['budget']):
 assert a['fraction']==b['fraction'] and a['seed']==b['seed'] and a['poses']==b['poses']
 np.testing.assert_allclose(a['score']['coverage']['0.95'],b['score']['coverage']['0.95'])
previous={'budget':budget}
fig,ax=plt.subplots(figsize=(7.5,3.3),layout='constrained')
for j,fraction in enumerate([.25,.5,1.]):
 b=[b for b in previous['budget'] if b['fraction']==fraction]
 ax.scatter([b['poses'] for b in b],[b['score']['coverage']['0.95']*100 for b in b],color=COL[j],label=f'{fraction:.0%} of covariance-fit tiles',s=35)
ax.axhline(95,color='k',ls='--',lw=1);ax.set(xlabel='Independent XY positions in covariance fit',ylabel='95% ellipse containment (%)');ax.legend();savefig(fig,'commissioning_budget')
# Pooled residual ellipses: visualize remaining mean explicitly, never recenter the deployed update.
fig,axes=plt.subplots(2,3,figsize=(8,5.4),layout='constrained')
for j,c in enumerate('ABCDE'):
 ax=axes.flat[j];err=data['error'][ev,j];valid=np.isfinite(err).all(1);err=err[valid]*100
 C=np.cov(err.T,bias=True);predicted=data['R'][ev,j][valid].mean(0)*10000
 for center,cov,color,label in [(err.mean(0),C,COL[1],'Empirical mean + scatter'),(np.zeros(2),predicted,COL[0],'Mean assigned R')]:
  val,vec=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))
  ax.add_patch(Ellipse(center,2*np.sqrt(5.991*val[-1]),2*np.sqrt(5.991*val[0]),angle=angle,fill=False,color=color,lw=1.5,label=label))
 ax.scatter(err[:,0],err[:,1],s=2,alpha=.18,color=COL[4]);ax.autoscale_view();ax.set(aspect='equal',title=f'Camera {c}; n={len(err)}',xlabel='X residual (cm)',ylabel='Y residual (cm)');ax.axhline(0,color='#aaaaaa',lw=.5);ax.axvline(0,color='#aaaaaa',lw=.5)
axes.flat[5].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[5].legend(handles,labels,loc='center',frameon=False,fontsize=9)
savefig(fig,'pooled_ellipses')
# Ground-reference trajectories with collision-box footprints from the actual warehouse SDF.
fig,axes=plt.subplots(1,2,figsize=(9,4.2),sharex=True,sharey=True,layout='constrained')
world=REPO/'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
sources[str(world)]=hashlib.sha256(world.read_bytes()).hexdigest()
root=ET.parse(world).getroot();occluders=root.find(".//model[@name='warehouse_v2_occluders']")
assert occluders is not None and occluders.find('pose') is None
for link in occluders.findall('link'):
 pose=np.fromstring(link.findtext('pose'),sep=' ');size=np.fromstring(link.findtext('collision/geometry/box/size'),sep=' ')
 assert np.allclose(pose[3:],0) and link.find('collision/pose') is None
 for ax in axes:ax.add_patch(plt.Rectangle(pose[:2]-size[:2]/2,*size[:2],fc='#eeeeee',ec='#cccccc',lw=.5,zorder=0))
for n,r in enumerate(runs):
 entry=next(e for e in selection['runs'] if e['key']==r['_key']);m,summary,truth,odom,readings,batches,accounting=load_run(entry)
 ax=axes[0 if r['task']=='fusion_overlap_rich' else 1]
 times=np.arange(summary['first_cmd_stamp'],summary['stop_stamp'],.2);x,y=truth.at(times)
 ax.plot(x,y,color=COL[n%7],label=f"Seed {r['seed']}")
 ax.plot(x[-1],y[-1],'x' if accounting['outcome']=='collision' else 'o',color=COL[n%7],ms=7)
for j,ax in enumerate(axes):
 for k,c in enumerate('ABCDE'):
  cam=geometry['camera_'+c];xy=np.asarray(cam.cam_pos)[:2];ax.plot(*xy,'s',color='#333333',ms=4);ax.annotate(c,xy,xytext=(4,3),textcoords='offset points',fontsize=9)
 ax.set(aspect='equal',title=['Overlap-rich route','Network-traverse route'][j],xlabel='X (m)',xlim=(-13,14),ylim=(-11,11));ax.legend(loc='upper left',fontsize=8)
axes[0].set_ylabel('Y (m)');savefig(fig,'recorded_routes')
# Predeclared first traverse execution: live belief and heading at their own stamps.
failure=next((e for e in selection['runs'] if e['task']=='fusion_network_traverse' and e['seed']==110),None)
if failure:
 run=REPO/failure['run'];m,summary,truth,odom,readings,batches,accounting=load_run(failure)
 tab=aligned.rows(run);live=aligned.aligned_error_cm(run,'belief',tab);t=live['stamp']
 keep=aligned.landed_mask(t)&np.isfinite(live['aligned_cm'])&(t>=summary['first_cmd_stamp'])&(t<=summary['stop_stamp'])
 P=np.array([[[float(r['planner_cov_x']),float(r['planner_cov_xy'])],[float(r['planner_cov_xy']),float(r['planner_cov_y'])]] for r in tab])
 radius=100*np.sqrt(5.991*np.maximum(0,np.linalg.eigvalsh(P)[:,-1]))
 yaw=np.array([float(r['planner_belief_yaw']) for r in tab]);refyaw=truth.yaw_at(t);yawerror=np.degrees(np.abs((yaw-refyaw+np.pi)%(2*np.pi)-np.pi))
 ass=aligned.assimilations(run);accepted=sorted([summary['first_cmd_stamp'],summary['stop_stamp']]+[a['belief_stamp_after'] for a in ass if a['accepted'] and summary['first_cmd_stamp']<=a['belief_stamp_after']<=summary['stop_stamp']])
 gaps=np.diff(accepted);longest=int(np.argmax(gaps));start=summary['first_cmd_stamp']
 fig,axes=plt.subplots(2,1,figsize=(8.3,4.5),sharex=True,layout='constrained')
 axes[0].plot(t[keep]-start,live['aligned_cm'][keep],color=COL[5],label='Live position error')
 axes[0].plot(t[keep]-start,radius[keep],color=COL[0],label='95% ellipse major semi-axis')
 axes[0].set_ylabel('Position (cm)');axes[0].legend()
 axes[1].plot(t[keep]-start,yawerror[keep],color=COL[1]);axes[1].set(xlabel='Time after first command (s)',ylabel='Absolute heading error (deg)')
 for ax in axes:ax.axvspan(accepted[longest]-start,accepted[longest+1]-start,color='#bbbbbb',alpha=.3);ax.grid(alpha=.15)
 savefig(fig,'live_failure_episode')
 (G/'failure_episode.json').write_text(json.dumps(dict(run=failure['run'],samples=int(keep.sum()),gap_start_s=accepted[longest],gap_end_s=accepted[longest+1],longest_gap_s=float(gaps[longest]),reference_stamp_source=accounting['gt_stamp_source'],collision_reason=summary['collision_reason'],scope='Temporal association, not a causal attribution of collision'),indent=2)+'\n')
# Describe run-level patterns without treating frames as independent trials.
paragraphs=[]
for task in ['fusion_overlap_rich','fusion_network_traverse']:
 subset=[r for r in runs if r['task']==task]
 if not subset:continue
 n=len(subset);name='overlap-rich' if task=='fusion_overlap_rich' else 'network-traverse'
 wins=sum(r['replay']['full/confidence']['median_cm']<r['replay']['full/constant']['median_cm'] for r in subset)
 values=[100*r['replay']['full/confidence']['coverage']['0.95'] for r in subset]
 paragraphs.append(f'On the {name} route, the calibrated-score model has lower median replay error than full constant covariance in {wins} of {n} analyzed executions. Its nominal 95\% position-ellipse containment spans {min(values):.1f}--{max(values):.1f}\%. These are paired pilot observations, not a significance claim.')
(G/'run_findings.tex').write_text('\n\n'.join(paragraphs)+'\n')
(G/'numbers.tex').write_text('\\newcommand{\\StaticOpportunities}{'+str(static['opportunities'])+'}\n'+'\\newcommand{\\StaticGroups}{'+str(static['groups'])+'}\n'+'\\newcommand{\\DriveCount}{'+str(len(runs))+'}\n'+'\\newcommand{\\StaticAccepted}{'+str(static['accepted'])+'}\n')
(G/'forecast_ratios.json').write_text(json.dumps(frm,indent=2)+'\n')
(G/'evidence_manifest.json').write_text(json.dumps(dict(status=selection['status'],sources=sources,
 figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob('*.pdf'))},
 analyzed_runs=[r['_key'] for r in runs],pending=selection['pending'],invalid=selection['invalid'],
 builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print('Built figures and tables for',len(runs),'verified diagnostic runs')
