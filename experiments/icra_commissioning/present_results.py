#!/usr/bin/env python3
"""Generate the framing and measured-results package; no manuscript generation."""
import json,sys,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse,FancyBboxPatch
from scipy.stats import chi2
from study import REPO,OUT,CAPTURE,ARTIFACT,load,readcsv,writejson,digest
sys.path.insert(0,str(REPO/'experiments/deck_figures'))
import style as D
from replay import load as load_drive, LearnedBoxCorrection,camera_models

FIG=OUT/'presentable_results';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':13,'axes.titlesize':15,'axes.labelsize':13,'legend.fontsize':11,
                     'xtick.labelsize':11,'ytick.labelsize':11,'pdf.fonttype':42})
C={'constant':'#476582','geometry':'#238e83','confidence':'#bd487a','confidence_bias':'#825db4',
   'image':'#d79c36','spatial':'#e16b3f','raw':'#e16b3f','nn':'#2a78d6'}
LABEL={'constant':'Full constant','geometry':'Geometry','confidence':'Confidence','confidence_bias':'Confidence + bias state',
       'image':'Crop statistics','spatial':'Spatial'}
PAGES=[]

def canvas(title,subtitle):
    fig=plt.figure(figsize=(13.333,7.5),facecolor='#fcfcfb')
    heading=fig.text(.055,.95,title,fontsize=23,weight='bold',va='top',color='#14283c')
    fig.canvas.draw()
    while heading.get_window_extent(fig.canvas.get_renderer()).width>fig.bbox.width*.89:
        heading.set_fontsize(heading.get_fontsize()-.5)
        fig.canvas.draw()
    fig.text(.055,.895,subtitle,fontsize=12.5,va='top',color='#566372')
    return fig

def save(fig,name,note):
    fig.text(.055,.035,note,fontsize=10,color='#647080',va='bottom')
    fig.savefig(FIG/f'{name}.pdf');fig.savefig(FIG/f'{name}.png',dpi=160)
    PAGES.append((name,fig));return fig

def ellipse(ax,mean,cov,color,ls='-',label=None):
    vals,vec=np.linalg.eigh(cov);theta=np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))
    ax.add_patch(Ellipse(mean,2*np.sqrt(chi2.ppf(.95,2)*vals[-1]),2*np.sqrt(chi2.ppf(.95,2)*vals[0]),
       angle=theta,fill=False,edgecolor=color,lw=2,ls=ls,label=label))

def main():
    result=json.loads((OUT/'results.json').read_text());replay=json.loads((OUT/'replay_results.json').read_text())
    future=json.loads((OUT/'future_results.json').read_text());image=json.loads((OUT/'image_results.json').read_text())
    gen=json.loads((OUT/'generalization_results.json').read_text());data,counts=load(OUT)
    models=joblib.load(OUT/'models.joblib');rows=[r for r in data if r['role']=='evaluation']
    # 1: Research framing and evidence levels.
    fig=canvas('The link to test: camera commissioning → future localization quality',
        'Current evidence supports a measurement-model comparison and a temporal-consistency diagnosis.')
    ax=fig.add_axes([.05,.35,.9,.4]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    names=['Commissioning','Corrected observation','Multi-camera fusion','Future belief','Navigation']
    desc=['Grouped positions\n+ camera outcomes','Reference position\n+ conditional covariance','Fresh frames\n+ fixed robot Q','No future image\n+ joint hit/miss outcomes','Existing planner\n+ matched executions']
    status=['MEASURED','MEASURED','ONE-DRIVE REPLAY','ONE-ROUTE DIAGNOSTIC','BASELINE REPRODUCTION']
    for i in range(5):
        x=i*.203
        ax.add_patch(FancyBboxPatch((x,.26),.182,.62,boxstyle='round,pad=.008',facecolor='#eef3f6',edgecolor='#cbd6df'))
        ax.text(x+.091,.75,names[i],ha='center',fontsize=13,weight='bold')
        ax.text(x+.091,.54,desc[i],ha='center',va='center',fontsize=11.5)
        ax.text(x+.091,.32,status[i],ha='center',fontsize=8.3,color=C['constant'])
        if i<4:ax.annotate('',xy=(x+.199,.57),xytext=(x+.185,.57),arrowprops={'arrowstyle':'->','lw':2})
    fig.text(.065,.26,'Working finding',weight='bold',fontsize=14,color=C['confidence'])
    fig.text(.065,.205,'A useful per-frame error model can still give an overconfident trajectory.',fontsize=21)
    fig.text(.065,.125,'Next claim to earn: the same commissioned model predicts route quality and improves navigation.',fontsize=15)
    save(fig,'01_framing','Development evidence in one simulated installation. No submission-ready or physical-navigation claim.')
    # 2: correction and calibration, all same samples.
    fig=canvas('The existing correction removes much of the typical offset; tails remain',
        'Same 1,172 returned camera readings, 14 held-out development tiles; reference is the commanded static robot position.')
    a=fig.add_axes([.08,.2,.37,.59]);b=fig.add_axes([.57,.2,.36,.59])
    for key,label in [('raw','Raw bbox → floor'),('z','Frozen NN correction')]:
        vals=np.sort([100*np.linalg.norm(r[key]-r['truth']) for r in rows])
        a.plot(vals,np.arange(1,len(vals)+1)/len(vals),lw=2.5,label=f'{label} (median {np.median(vals):.1f} cm)',color=C['raw' if key=='raw' else 'nn'])
    a.set(xlim=(0,80),ylim=(0,1),xlabel='Camera-reading position error (cm)',ylabel='Fraction of readings');a.legend(loc='lower right',fontsize=10)
    a.axhline(.95,ls=':',color='#aaa')
    scores={k:result['scores'][f'evaluation/{k}'] for k in ['constant','geometry','confidence']};scores['image']=image['score']
    for k,s in scores.items():
        b.plot([100*float(q) for q in s['coverage']],[100*v for v in s['coverage'].values()],'-o',lw=2,label=LABEL[k],color=C[k])
    b.plot([50,100],[50,100],'k--',lw=1);b.set(xlabel='Nominal ellipse probability (%)',ylabel='Empirical containment (%)',xlim=(49,100),ylim=(45,101));b.legend(loc='lower right',fontsize=10)
    save(fig,'02_correction_and_calibration','All covariance arms share NN + per-camera offset and acceptance. Separate selection tiles tune covariance scale/shrinkage. CDF includes tails beyond 80 cm.')
    # 3: spatial vectors camera B with shared scale.
    fig=canvas('Where the residual remains: camera B on held-out floor positions',
        'Arrows are mean observed residuals across available headings at each position, not repeated-trial bias estimates.')
    lay=D.layout(); axes=[fig.add_axes([.055,.14,.43,.65]),fig.add_axes([.515,.14,.43,.65])]
    rr=[r for r in rows if r['camera']=='camera_B'];group=defaultdict(list)
    for r in rr:group[tuple(r['truth'])].append(r)
    for ax,key,title in zip(axes,['raw','z'],['Raw bottom-centre projection','Frozen NN reference estimate']):
        D.draw_warehouse(ax,lay,rack_alpha=.5)
        xy=np.array(list(group));e=np.array([np.mean([r[key]-r['truth'] for r in v],axis=0) for v in group.values()])
        mag=np.linalg.norm(e,axis=1);shrink=np.minimum(1,2.5/np.maximum(mag*3,1e-9))
        ax.quiver(xy[:,0],xy[:,1],e[:,0]*3*shrink,e[:,1]*3*shrink,mag*100,cmap='magma',clim=(0,50),angles='xy',scale_units='xy',scale=1,width=.006,zorder=8)
        ax.set_title(title,pad=13);ax.quiverkey(ax.collections[-1],.12,.02,.6,'20 cm (×3)',coordinates='axes',labelpos='E')
    save(fig,'03_spatial_residuals',f'{len(rr)} camera B readings across {len(group)} positions. Shared arrow gain ×3; displayed arrows capped at 2.5 m. Color 0–50 cm.')
    # 4: empirical versus predicted ellipse by score regime.
    fig=canvas('Calibration needs the residual centre and the ellipse, not only a score',
        'Camera B; the confidence model is frozen before evaluation. Dashed ellipses describe observed scatter across configurations.')
    camrows=[r for r in rows if r['camera']=='camera_B'];m=models['camera_B','confidence'];cells=m._cells(camrows)
    for cell in range(3):
        ax=fig.add_axes([.065+cell*.313,.23,.27,.53]);r=[r for r,c in zip(camrows,cells) if c==cell]
        z,R=m.predict(r);e=100*(z-np.array([v['truth'] for v in r]));Cemp=np.cov(e.T,bias=True)
        ax.scatter(e[:,0],e[:,1],s=14,alpha=.4,color=C['confidence']);ellipse(ax,[0,0],10000*R.mean(axis=0),C['confidence'],label='Predicted 95%')
        ellipse(ax,e.mean(axis=0),Cemp,'#34485c','--',label='Empirical Gaussian 95%')
        ax.plot(*e.mean(axis=0),'x',color='#34485c',ms=10,mew=2)
        cap=max(10,float(np.max(np.abs(e)))*1.08,
                1.08*np.sqrt(chi2.ppf(.95,2)*np.linalg.eigvalsh(10000*R.mean(axis=0)).max()))
        ax.set(xlim=(-cap,cap),ylim=(-cap,cap),aspect='equal',xlabel='Residual x (cm)',ylabel='Residual y (cm)',title=f'Score regime {cell+1}  ·  n={len(r)}')
        if cell==0:ax.legend(loc='upper left',bbox_to_anchor=(0,-.26),fontsize=9)
    save(fig,'04_residual_ellipses','Cross marks show remaining signed means. Covariance fitting is centred; the same per-camera bias correction is applied in inference and evaluation.')
    # 5: static model comparison and an independent configuration application.
    fig=canvas('Detector score remains competitive; this image probe adds no clear gain',
        'The crop-statistics branch changes uncertainty only. The second capture uses the same camera installation and frozen models.')
    axes=[fig.add_axes([.16,.22,.30,.55]),fig.add_axes([.60,.22,.32,.55])]
    kk=['constant','geometry','confidence','image']; vals=[scores[k]['group_mean_nll'] for k in kk]
    lo=[scores[k]['group_bootstrap_nll_ci95'][0] for k in kk];hi=[scores[k]['group_bootstrap_nll_ci95'][1] for k in kk]
    axes[0].errorbar(vals,np.arange(4),xerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',ecolor='#8293a1',capsize=4)
    axes[0].scatter(vals,np.arange(4),c=[C[k] for k in kk],s=100,zorder=3)
    axes[0].set(yticks=np.arange(4),yticklabels=[LABEL[k] for k in kk],xlabel='Gaussian error score (lower is better)',title='14 tiles; descriptive tile-bootstrap intervals');axes[0].invert_yaxis()
    for k in ['constant','geometry','confidence']:
        s=gen['scores'][k];axes[1].scatter(s['rms_sigma_cm'],100*s['coverage']['0.95'],s=110,color=C[k],label=LABEL[k])
        axes[1].annotate(LABEL[k],(s['rms_sigma_cm'],100*s['coverage']['0.95']),xytext=(5,-16 if k=='constant' else 7),textcoords='offset points',fontsize=11)
    axes[1].axhline(95,color='#999',ls='--');axes[1].set(xlabel='Predicted RMS sigma (cm)',ylabel='95% containment (%)',ylim=(94.5,100),xlim=(16,34),title=f'Second static capture: {gen["readings"]:,} readings')
    save(fig,'05_model_comparison','Second capture: previously examined dense configurations, not new-camera transfer. Image probe: five crop statistics, three regimes per camera; no new NN mean.')
    # 6: camera complementarity selected by model, not reference error.
    selection=json.loads((OUT/'driving_manifest.json').read_text());entry=selection['runs'][0]
    mean=LearnedBoxCorrection(REPO/ARTIFACT);geometry=camera_models(json.loads((REPO/CAPTURE/'capture_manifest.json').read_text()))
    manifest,truth,odom,readings,ass=load_drive(entry,mean,geometry)
    batches=defaultdict(list)
    for r in readings:batches[r['batch']].append(r)
    candidates=[]
    for rs in batches.values():
        if len(rs)<2 or np.ptp([r['t'] for r in rs])>1e-6:continue
        zz=[];RR=[]
        for r in rs:
            z,R=models[r['camera'],'confidence'].predict([r]);zz.append(z[0]);RR.append(R[0])
        info=sum(np.linalg.solve(R,np.eye(2)) for R in RR);fR=np.linalg.solve(info,np.eye(2))
        gain=min(np.trace(R) for R in RR)/np.trace(fR)
        candidates.append((gain,rs,zz,RR,fR))
    gain,rs,zz,RR,fR=max(candidates,key=lambda t:t[0]);fused=fR@sum(np.linalg.solve(R,z) for R,z in zip(RR,zz));ref=rs[0]['truth']
    fig=canvas('Overlapping cameras can constrain different directions',
        'One synchronized diagnostic batch, chosen by predicted trace reduction before looking at reference error.')
    ax=fig.add_axes([.07,.19,.47,.61]);positions=[]
    for r,z,R in zip(rs,zz,RR):
        col=D.CAM_COLOUR[r['camera'][-1]];pos=100*(z-ref);positions.append(pos)
        ellipse(ax,pos,R*10000,col,label=r['camera']);ax.plot(*pos,'o',color=col)
    ellipse(ax,100*(fused-ref),fR*10000,'#222',label='Independent fused');ax.plot(*((fused-ref)*100),'s',color='#222');ax.plot(0,0,'*',ms=15,color='black',label='Capture-time reference')
    ax.autoscale_view();ax.set(aspect='equal',xlabel='x relative to reference (cm)',ylabel='y relative to reference (cm)');ax.legend(fontsize=9)
    a=fig.add_axes([.63,.2,.29,.59]);a.axis('off')
    a.text(0,.85,f'{len(rs)} cameras',fontsize=28,weight='bold');a.text(0,.68,f'{gain:.2f}× predicted trace reduction',fontsize=16)
    a.text(0,.58,'versus the best single view',fontsize=12)
    a.text(0,.36,f'{100*np.linalg.norm(fused-ref):.2f} cm fused reading error',fontsize=18,weight='bold')
    a.text(0,.14,'This example illustrates geometry.\nIt does not establish independence\nor calibrated multi-camera fusion.',fontsize=13,linespacing=1.5)
    save(fig,'06_complementarity',f'Source: {manifest["run_id"]}; capture {rs[0]["t"]:.3f} s. 95% working-Gaussian ellipses; no robot prior in this illustration.')
    # 7: sequential diagnosis, consistent population.
    fig=canvas('The same per-frame model becomes overconfident when errors persist',
        'Same recorded observations, mean, fixed robot Q and 1,559 evaluation timestamps; one drive, seed 10.')
    a=fig.add_axes([.08,.22,.37,.57]);b=fig.add_axes([.58,.22,.34,.57])
    for cam in ['camera_A','camera_B','camera_D','camera_E']:
        t=[r for r in replay['temporal'] if r['camera']==cam]
        a.plot([r['median_lag_s'] for r in t],[r['correlation'][0] for r in t],'-o',label=cam,color=D.CAM_COLOUR[cam[-1]])
    a.axhline(0,color='#aaa',ls='--');a.set(xlabel='Within-run lag (s)',ylabel='Whitened x residual correlation',ylim=(-.5,1));a.legend(fontsize=9)
    for kind in ['constant','geometry','confidence','confidence_bias']:
        for interval in [0,1.]:
            s=next(v['score'] for v in replay['scores'] if v['kind']==kind and v['interval_s']==interval and v['cameras']=='all')
            b.scatter(s['median_cm'],100*s['coverage']['0.95'],s=110,marker='o' if interval==0 else '^',color=C[kind],facecolor=C[kind] if interval==0 else 'none')
            if interval==0:b.annotate(LABEL[kind],(s['median_cm'],100*s['coverage']['0.95']),xytext=(6,-12 if kind=='geometry' else 5),textcoords='offset points',fontsize=10)
    b.axhline(95,color='#555',ls='--');b.set(xlabel='Median belief position error (cm)',ylabel='95% position-ellipse containment (%)',ylim=(75,103),xlim=(2.5,7.7))
    fig.text(.59,.13,'● Full rate    △ ≥1 s between camera updates',fontsize=11)
    save(fig,'07_temporal_fusion','Bias-state ablation: 50% persistent marginal error, 2 s decay; chosen as a diagnostic, not tuned on a final test. Subsampling does not establish independence.')
    # 8: ahead predictions and cadence.
    fig=canvas('Future-quality forecasts depend strongly on the information cadence',
        'Three-second forecasts along one recorded route. Future images are unavailable to the predictor; route controls are prescribed.')
    a=fig.add_axes([.075,.23,.51,.56]);b=fig.add_axes([.69,.23,.25,.56])
    for cadence,ls in [(.2,'-'),(1.,'--')]:
        rr=[r for r in future['rows'] if r['horizon_s']==3 and r['cadence_s']==cadence and r['method']=='branch']
        a.plot([r['start_s'] for r in rr],[100*np.sqrt(r['predicted_trace_m2']) for r in rr],ls,lw=2,label=f'Joint branch average, {1/cadence:g} Hz')
    a.plot([r['start_s'] for r in rr],[100*np.sqrt(r['realized_squared_error_m2']) for r in rr],color='#bd487a',alpha=.6,label='Realized belief error')
    a.set(xlabel='Forecast start time (s)',ylabel='Predicted RMS / observed error (cm)');a.legend(fontsize=10)
    ss=[r for r in future['summary'] if r['horizon_s']==3 and r['method']=='branch']
    b.bar(['5 Hz','1 Hz'],[r['predicted_rms_cm'] for r in ss],color=[C['constant'],C['geometry']]);b.axhline(ss[0]['realized_rms_cm'],color=C['confidence'],ls='--',label='Realized RMS')
    b.set(ylabel='RMS position error (cm)',title=f'{ss[0]["windows"]} dependent windows');b.legend(fontsize=10)
    save(fig,'08_future_quality','Forecast-only cadence sensitivity. The estimator cadence is fixed. Good pooled RMS at 1 Hz is not independent validation or a route-ranking result.')
    # 9: commissioning cost and required gap, numerical cost included.
    fig=canvas('Commissioning cost must include the mean model, not only covariance',
        'Whole-tile subsets; five fixed subset seeds. The same evaluation positions are used at every budget.')
    a=fig.add_axes([.075,.24,.4,.54]);b=fig.add_axes([.58,.2,.35,.6]);b.axis('off')
    for frac in [.25,.5,1.]:
        rs=[r for r in result['budget'] if r['fraction']==frac]
        a.scatter([r['poses'] for r in rs],[100*r['score']['coverage']['0.95'] for r in rs],s=85,alpha=.7,label=f'{100*frac:g}% of fit tiles')
    a.axhline(95,color='#555',ls='--');a.set(xlabel='Covariance-fit positions',ylabel='95% ellipse containment (%)');a.legend(fontsize=10)
    b.text(0,.92,'3,249 boxes',fontsize=30,weight='bold',color=C['nn']);b.text(0,.82,'already used to train the fixed mean network',fontsize=12)
    b.text(0,.61,'1,172 + 819 boxes',fontsize=26,weight='bold',color=C['constant']);b.text(0,.51,'covariance fit + separate model selection',fontsize=12)
    b.text(0,.3,'1,172 evaluation boxes',fontsize=22,weight='bold');b.text(0,.2,'14 development tiles; not an untouched final test',fontsize=12)
    save(fig,'09_commissioning_budget','Budget panel is the untuned full constant covariance: no hidden selection-budget tuning. Counts refer to returned boxes; misses are retained separately.')
    # 10: a fresh execution tests frozen models without pooling unequal drives.
    valpath=OUT/'validation_replay/replay_results.json'
    if valpath.exists():
        val=json.loads(valpath.read_text())
        fig=canvas('The new execution changes the calibration verdict',
            'Frozen models replayed on fresh CPU observations. Compare models within each column pair; do not pool the two drives.')
        ax=fig.add_axes([.075,.28,.85,.48]);ax.axis('off');cells=[]
        for kind in ['constant','geometry','confidence','confidence_bias']:
            ss=[]
            for source in [replay,val]:
                s=next(v['score'] for v in source['scores'] if v['kind']==kind and v['interval_s']==0 and v['cameras']=='all')
                ss.extend([f"{s['median_cm']:.2f} cm",f"{100*s['coverage']['0.95']:.1f}%"])
            cells.append([LABEL[kind],*ss])
        table=ax.table(cellText=cells,colLabels=['Model','Drive 1: median','Drive 1: 95% coverage','New drive: median','New drive: 95% coverage'],
            cellLoc='center',colWidths=[.27,.17,.20,.17,.19],bbox=[0,0,1,1])
        table.auto_set_font_size(False);table.set_fontsize(12)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('#d9e0e5');cell.set_facecolor('#eaf0f4' if row==0 else '#fcfcfb')
            if row==0:cell.set_text_props(weight='bold',fontsize=10.5)
        fig.text(.075,.17,'The temporal-bias model is not a consistent winner. Keep the accuracy–consistency trade-off visible.',fontsize=16)
        save(fig,'10_fresh_execution','Drive 1: 1,559 timestamps. New drive: 835 timestamps, terminated by collision. Both nominal seed 10; different executions/rates/routes. No across-run significance claim.')
    # 11: runtime result if a completed frozen manifest has been provided.
    baseline=OUT/'baseline_report.json'
    if baseline.exists():
        s=json.loads(baseline.read_text());fig=canvas('The reproduced NN baseline collided during navigation',
            'New CPU execution; standalone baseline reproduction. This is not a navigation comparison between sensor models.')
        a=fig.add_axes([.06,.16,.45,.64]);D.draw_warehouse(a,lay,rack_alpha=.5)
        r=s['trajectory'];a.plot(r['gt_x'],r['gt_y'],color='#283845',lw=2,label='Reference path');a.plot(r['x'],r['y'],color=C['nn'],lw=1.4,label='Planner belief');a.legend(fontsize=10)
        b=fig.add_axes([.58,.15,.35,.66]);b.axis('off');lines=[('Outcome',s['outcome']),('Median belief error',f"{s['median_cm']:.2f} cm"),('95th percentile',f"{s['p95_cm']:.2f} cm"),('Dropped corrections',f"{100*s['dropped_fraction']:.1f}%"),('Longest correction gap',f"{s['longest_gap_s']:.2f} s"),('Capture/update accounting',s['accounting'])]
        for i,(k,v) in enumerate(lines):b.text(0,.93-i*.14,k,fontsize=12,color='#607080');b.text(0,.88-i*.14,v,fontsize=18,weight='bold')
        save(fig,'11_runtime_baseline',f"Source: {s['run_id']}; CPU timing differs from the archived GPU drive. One execution is reproduction, not replication.")
    with PdfPages(FIG/'commissioning_results.pdf') as pdf:
        for name,fig in PAGES:pdf.savefig(fig);plt.close(fig)
    source_files=['manifest.json','models.joblib','results.json','replay_results.json','future_results.json',
                  'image_results.json','generalization_results.json','baseline_report.json',
                  'validation_manifest.json','validation_replay/replay_results.json']
    writejson(FIG/'sources.json',dict(inputs={f:digest(OUT/f) for f in source_files if (OUT/f).exists()},
        generator_sha256=digest(Path(__file__)),
        pages=[name for name,_ in PAGES],status='development_results'))
    print('wrote',len(PAGES),'pages',FIG/'commissioning_results.pdf')

if __name__=='__main__':main()
