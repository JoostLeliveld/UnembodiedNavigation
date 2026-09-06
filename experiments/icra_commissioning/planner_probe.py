#!/usr/bin/env python3
"""Recorded planner integration probe: fitted fields and optimization, no driving.

Produces model-prediction figures, never camera-error or navigation-performance
figures. The same short warehouse-lane problem and starting belief are used for
all three fields. Full closed-loop comparisons remain a separate experiment.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/icra_mpl')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/p) for p in ('src/planning','src/reliability','src/unav_common',
    'src/experiments','experiments/camera_observation_characterization')]
from export_network_planner import DEST,KINDS
from study import OUT,digest,writejson
from derive_interpretations import camera_models
from planning.core.camera_network import CameraNetworkModel,projection_jacobian
from planning.planners.base_planner import UnicyclePlannerBase
from experiments.core.world_profiles import serialize_driveable_geometry_from_profile


def run(root,out):
    if (out/'results.json').exists():raise RuntimeError('probe already recorded; choose a new --out')
    manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest['sources'].items():
        if digest(REPO/p)!=h:raise RuntimeError(f'changed export input {p}')
    paths={k:REPO/v['path'] for k,v in manifest['artifacts'].items()}
    for k,p in paths.items():
        if digest(p)!=manifest['artifacts'][k]['sha256']:raise RuntimeError(f'changed field {k}')
    static=json.loads((OUT/'manifest.json').read_text())
    capture_path=REPO/static['capture']/'capture_manifest.json'
    capture=json.loads(capture_path.read_text())
    geometry=camera_models(capture);camera=geometry['camera_A']
    profile=yaml.safe_load(Path(capture['world_profiles_path']).read_text())['worlds']['warehouse_v2.world.sdf']
    driveable=serialize_driveable_geometry_from_profile(profile)
    state=np.array([-4.8,-7.2,0.]);goal=np.array([-2.8,-7.2]);P=np.diag([.05**2,.05**2,np.deg2rad(5)**2])
    settings=dict(horizon=40,dt=.25,v_min=0.,v_max=.22,w_min=-.8,w_max=.8,
        control_weight=.02,process_noise_xy=.01,process_noise_theta=.02,obs_noise_uv=2.5,
        goal_sigma_uv=30.,risk_weight_obs=1.,ambiguity_weight=1.,optimizer_maxiter=80,
        optimizer_gtol=1e-5,optimizer_warm_start=False,optimizer_multistart=True,
        optimizer_ftol=1e-7,seed=513,use_visibility_model=True,use_hit_miss_mixture=False,
        use_nogo_cost=True,nogo_mode='keep_in',driveable_geometry_json=driveable,
        # Current AMR body is 0.800 x 0.550 m (warehouse_amr.urdf.xacro).
        # The previous drive pilot used 0.275 m; do not inherit that width-only disc.
        nogo_weight=40.,nogo_safe_distance=.55,robot_collision_radius_m=float(np.hypot(.4,.275)),
        use_belief_nogo_cost=False,
        camera_params=dict(cam_pos=camera.cam_pos.tolist(),look_at=camera.look_at.tolist(),
            img_width=camera.img_width,img_height=camera.img_height,fov_h_rad=camera.fov_h_rad))
    source_files=[Path(__file__).resolve(),REPO/'src/planning/planning/core/camera_network.py',
        REPO/'src/planning/planning/core/casadi_efe.py',REPO/'src/planning/planning/planners/base_planner.py',
        REPO/'src/planning/planning/core/dynamics.py',capture_path,Path(capture['world_profiles_path'])]
    source_files.append(REPO/'src/sim/robot_description/urdf/warehouse_amr.urdf.xacro')
    protocol=dict(kind='planner_implementation_probe_not_navigation_evaluation',
        initial_predicted_state=state.tolist(),goal_xy=goal.tolist(),initial_P=P.tolist(),
        settings=settings,network_manifest_sha256=digest(root/'manifest.json'),
        sources={str(p.relative_to(REPO)):digest(p) for p in source_files},
        scenario='short south lane, selected for integration; no images, sensor samples, or robot execution',
        anchor='camera_A remains fixed across all network arms; it is a cost chart, not an extra sensor')
    out.mkdir(parents=True,exist_ok=True);writejson(out/'protocol.json',protocol)
    results={};networks={};proxy_maps={};trajectories={}
    for kind in KINDS:
        network=CameraNetworkModel(paths[kind]);networks[kind]=network
        planner=UnicyclePlannerBase(camera_network_artifact_path=str(paths[kind]),**settings)
        result=planner.plan(state,P,goal)
        if not np.isfinite(result.total_cost) or not np.isfinite(result.controls).all():
            raise RuntimeError(f'nonfinite optimization {kind}')
        jac=[projection_jacobian(camera.H,s) for s in result.states]
        prior=planner.planning_visibility_diagnostics(state,P)
        branch=network.forecast_posterior(state,P);info=network.forecast_posterior(state,P,'information')
        results[kind]=dict(artifact_sha256=network.sha256,total_cost_diagnostic_sum=float(result.total_cost),
            risk=float(result.risk_cost),ambiguity=float(result.ambiguity_cost),control=float(result.control_cost),
            obstacle=float(result.obstacle_cost),optimizer_success=bool(result.optimizer_success),
            optimizer_message=result.optimizer_message,iterations=int(result.optimizer_nit),
            solve_seconds=float(result.solve_time_s),terminal_goal_distance_m=float(np.linalg.norm(result.states[-1,:2]-goal)),
            maximum_chart_jacobian_condition=float(max(np.linalg.cond(j) for j in jac)),
            start_scores=prior['network_score'].tolist(),start_availability=prior['network_availability'].tolist(),
            start_branch_position_trace_m2=float(np.trace(branch[:2,:2])),
            start_information_position_trace_m2=float(np.trace(info[:2,:2])),
            selected_source=result.selected_source,backend=result.backend,rollout_valid=bool(result.rollout_valid),
            invalid_reason=result.invalid_reason)
        trajectories[kind]=result.states
        print(kind,json.dumps(results[kind]),flush=True)
        score=network.fields['score']
        cov=np.array([network.proxy_ground_covariance(score[:,iy,ix])
            for iy in range(len(network.ys)) for ix in range(len(network.xs))])
        proxy_maps[kind]=100*np.sqrt(np.trace(cov,axis1=1,axis2=2)/2).reshape(score.shape[1:])
    # Stored forecasts sample one independent detection opportunity per query.
    # These are NOT repeated updates along a hypothetical drive.
    query=np.column_stack([np.linspace(-7.5,10.,100),np.full(100,-6.5),np.zeros(100)])
    forecast={}
    for kind,network in networks.items():
        forecast[kind]={mode:np.array([np.trace(network.forecast_posterior(s,P,mode)[:2,:2])
            for s in query]) for mode in ('branch','information')}
    np.savez_compressed(out/'predictions.npz',query=query,**{f'plan_{k}':v for k,v in trajectories.items()},
        **{f'{k}_{m}':v for k,a in forecast.items() for m,v in a.items()})
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(10.4,3.65),layout='constrained',sharex=True,sharey=True)
    for ax,kind in zip(axes,KINDS):
        network=networks[kind]
        im=ax.pcolormesh(network.xs,network.ys,np.clip(proxy_maps[kind],0,40),vmin=0,vmax=40,cmap='viridis_r',shading='nearest',rasterized=True)
        for c,cam in geometry.items():
            ax.plot(*cam.cam_pos[:2],'^',ms=5,color='#e4a75a',mec='black',mew=.3)
            ax.annotate(c[-1],cam.cam_pos[:2],xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.set(title={'uniform':'Uniform score','geometry':'Geometry score','gp':'Commissioned GP score'}[kind],xlabel='World x [m]',aspect='equal')
    axes[0].set_ylabel('World y [m]')
    fig.colorbar(im,ax=axes,label='IWAI metric proxy scale [cm], capped at 40',shrink=.85)
    fig.suptitle('Fitted network fields: model predictions, not localization error',fontsize=11)
    for suffix in ('svg','png','pdf'):fig.savefig(out/f'network_fields.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.7),layout='constrained')
    colors={'uniform':'#67778a','geometry':'#cf8740','gp':'#237966'}
    for kind in KINDS:
        states=trajectories[kind]
        axes[0].plot(states[:,0],states[:,1],label=kind,color=colors[kind])
    axes[0].plot(*state[:2],'ko',ms=4);axes[0].plot(*goal,'k*',ms=9)
    axes[0].set(title='Same short-lane optimization problem',xlabel='World x [m]',ylabel='World y [m]',aspect='equal')
    axes[0].set_ylim(-7.9,-6.4)
    axes[0].axhline(-7.8,color='#9ca6af',lw=.8,ls=':',label='Declared lane boundary')
    axes[0].legend(frameon=False,loc='upper right',fontsize=8);axes[0].grid(alpha=.2)
    for mode,style,label in [('branch','-','Hit/miss branch reference'),('information','--','Expected-information approximation')]:
        axes[1].plot(query[:,0],100*np.sqrt(forecast['gp'][mode]),style,color=colors['gp'],label=label)
    axes[1].set(title='One future opportunity; fixed incoming belief',xlabel='Query x [m] along y = -6.5 m',ylabel=r'Model $\sqrt{\mathrm{tr}(P_{xy}^{+})}$ [cm]')
    axes[1].legend(frameon=False,fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('Planner integration probe: predicted paths and assumed-model uncertainty',fontsize=11)
    for suffix in ('svg','png','pdf'):fig.savefig(out/f'planner_probe.{suffix}',dpi=180)
    plt.close(fig)
    writejson(out/'results.json',dict(protocol_sha256=digest(out/'protocol.json'),results=results,
        interpretation='software integration only; no observed localization error, route ranking validation, or closed-loop result',
        files={name:digest(out/name) for name in ('predictions.npz','network_fields.svg','network_fields.pdf','network_fields.png',
            'planner_probe.svg','planner_probe.pdf','planner_probe.png')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network-root',type=Path,default=DEST)
    parser.add_argument('--out',type=Path,default=DEST/'probe_reviewed')
    args=parser.parse_args();run(args.network_root.resolve(),args.out.resolve())
