"""Current-belief projection onto the selected global plan; no reference input."""
import numpy as np
from speed_profile import LOCAL_GOAL_AHEAD,HANDOFF_FORWARD
class RouteHandoff:
 def __init__(self,states,goal):
  xy=np.asarray(states)[:,:2];keep=np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-6];self.xy=xy[keep]
  if np.linalg.norm(self.xy[-1]-goal)>1e-6:self.xy=np.vstack([self.xy,goal])
  self.ds=np.linalg.norm(np.diff(self.xy,axis=0),axis=1);self.arc=np.r_[0,np.cumsum(self.ds)];self.progress=0.
 def query(self,pose):
  seg=np.diff(self.xy,axis=0);alpha=np.clip(np.sum((np.asarray(pose)[:2]-self.xy[:-1])*seg,axis=1)/np.maximum(self.ds**2,1e-12),0,1);at=self.arc[:-1]+alpha*self.ds;dist=np.linalg.norm(self.xy[:-1]+alpha[:,None]*seg-np.asarray(pose)[:2],axis=1)
  dist[(at<self.progress-.2)|(at>self.progress+HANDOFF_FORWARD)]=np.inf;k=int(np.argmin(dist));self.progress=max(self.progress,float(at[k]));target_s=min(self.progress+LOCAL_GOAL_AHEAD,self.arc[-1]);j=min(np.searchsorted(self.arc,target_s,side='right')-1,len(self.ds)-1);target=self.xy[j]+(target_s-self.arc[j])/self.ds[j]*(self.xy[j+1]-self.xy[j])
  return target,dict(global_progress_m=self.progress,global_length_m=float(self.arc[-1]),global_path_deviation_m=float(dist[k]),local_goal=target.tolist())
