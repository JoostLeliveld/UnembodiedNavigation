"""Change only objective evaluation/gradient; retain IWAI plan() and geometry.

The goal is world XY with an isotropic 0.15 m standard deviation. The objective
uses KL risk plus posterior XY entropy. This explicit EFE extension is not
algebraically identical to the published observation-entropy objective.
"""
from paths import *
import json,pickle,types
import numpy as np
from speed_planner import planner,camera_models
from gp_forecast import build,event_offsets

from speed_profile import validate_config,PROFILE

class NetworkPlanner:
    def __init__(self,cfg):
        if 'id' in cfg:validate_config(cfg)
        self.cfg=cfg
        self.model=pickle.loads((O/'final_model/final_process.pkl').read_bytes())
        self.artifact=json.loads((HERE/'gp.json').read_text());self.artifact['variant']=cfg['arm']
        cams=camera_models(json.loads((R/'logs/perception_datasets/warehouse_v2_icra_p1_geometry_20260904/capture_manifest.json').read_text()))
        self.p=planner(cams[cfg.get('planner_camera','camera_A')],cfg['seed'])
        self.fn,self.trace=build(self.model,self.p,self.artifact,independent=True)
        self.p._evaluate_controls=types.MethodType(self._evaluate,self.p)
        self.p._get_casadi_valgrad=types.MethodType(self._valgrad,self.p)

    def _call(self,u,goal):
        return self.fn(self.mean,self.cov,np.asarray(u).reshape(-1,2).T,self.offsets,goal)

    def _evaluate(self,p,u,m,S,goal_state,goal_obs,goal_cov,return_metrics=False,**kw):
        value,grad,parts=self._call(u,goal_state[:2]);v=float(value)
        metrics=dict(zip(['risk_cost','ambiguity_cost','control_cost','obstacle_cost'],np.array(parts).ravel()))
        return (v,metrics) if return_metrics else v

    def _valgrad(self,p,*args,**kw):
        def evaluate(u,m,S,goal_obs,goal_xy,progress):
            val,grad,_=self._call(u,goal_xy)
            return float(val),np.array(grad).ravel()
        return evaluate

    def plan(self,mean,cov,goal,progress_index=0,source_age_s=0.):
        self.mean=np.asarray(mean).copy();self.cov=np.asarray(cov).copy()
        if self.mean.shape!=(13,) or self.cov.shape!=(13,13):raise ValueError('Full joint state required')
        self.offsets=event_offsets(source_age_s,self.artifact['opportunity_period_s'],self.p.horizon,self.p.dt)
        result=self.p.plan(self.mean[:3],self.cov[:3,:3],np.asarray(goal),progress_index=progress_index)
        m,P,q=self.trace(self.mean,self.cov,result.controls.T,self.offsets,goal)
        self.last_trace=dict(pose_mean=np.array(m).T.tolist(),joint_covariance=np.array(P).T.reshape(-1,13,13).transpose(0,2,1).tolist(),
            q=np.array(q).T.tolist(),event_offsets_s=self.offsets.tolist(),source_age_s=source_age_s,
            full_input_mean=self.mean.tolist(),full_input_covariance=self.cov.tolist(),
            field_variant=self.cfg['arm'],field_semantics='usable-reading probability' if self.cfg['arm']=='Commissioned' else 'IWAI detector-score reliability',approximation='independent future readings; sigma-point field query; periodic source opportunities; Bernoulli moment for Commissioned, precision-blended proxy for Uniform/IWAI')
        return result
