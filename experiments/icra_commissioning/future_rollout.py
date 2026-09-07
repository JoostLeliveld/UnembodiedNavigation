"""Prescribed-control forecasts on one shared motion grid for all camera arms.

The grid contains every control change and the union of compared observation
opportunities. Camera cadence changes only which grid events perform a forecast
update. It cannot change the Euler motion/Q discretization between arms.
"""
from __future__ import annotations

import numpy as np

from planning.core.dynamics import unicycle_step,unicycle_jacobian,unicycle_process_noise
from planning.core.plan_validation import validate_covariance


def _ns(value):
    value=float(value)
    if not np.isfinite(value):raise ValueError('nonfinite forecast time')
    integer=int(round(value*1e9))
    if abs(integer)>np.iinfo(np.int64).max:raise ValueError('forecast time exceeds nanosecond range')
    return integer


def _times_ns(values):
    values=np.asarray(values,float)
    if values.ndim!=1 or not len(values):raise ValueError('nonempty one-dimensional timestamps required')
    times=np.array([_ns(t) for t in values],dtype=np.int64)
    if np.any(np.diff(times)<=0):raise ValueError('forecast timestamps must be strictly increasing')
    return times


def observation_times(start,end,cadence):
    first,last,step=_ns(start),_ns(end),_ns(cadence)
    if last<=first or step<=0:raise ValueError('positive forecast interval and cadence required')
    return np.array(range(first+step,last+1,step),dtype=np.int64)*1e-9


def common_motion_grid(start,end,control_times,cadences):
    """All compared opportunity boundaries, all controls, and the exact endpoints."""
    first,last=_ns(start),_ns(end)
    controls=_times_ns(control_times)
    if first>=last or controls[0]>first or controls[-1]<last:
        raise ValueError('forecast interval lacks prescribed-control endpoint support')
    events={first,last,*map(int,controls[(controls>first)&(controls<last)])}
    for cadence in cadences:
        events.update(_ns(t) for t in observation_times(start,end,cadence))
    return np.array(sorted(events),dtype=np.int64)*1e-9


def forecast_window(state,covariance,*,start,end,control_times,controls,motion_grid,
                    opportunities,forecast,method,process_noise_xy=.01,process_noise_theta=.02):
    """Propagate one arm; a forecast returns (posterior, usable probability, support)."""
    if method not in ('branch','information'):raise ValueError('unsupported forecast approximation')
    state=np.asarray(state,float).copy()
    if state.shape!=(3,) or not np.isfinite(state).all():raise ValueError('invalid forecast initial state')
    covariance=validate_covariance(covariance)
    times=_times_ns(control_times);controls=np.asarray(controls,float)
    if controls.shape!=(len(times),2) or not np.isfinite(controls).all():raise ValueError('invalid prescribed controls')
    first,last=_ns(start),_ns(end);grid=_times_ns(motion_grid)
    if times[0]>first or times[-1]<last:raise ValueError('unsupported control interval')
    if grid[0]!=first or grid[-1]!=last:raise ValueError('motion grid does not match forecast endpoints')
    required=set(map(int,times[(times>first)&(times<last)]))
    if not required<=set(map(int,grid)):raise ValueError('motion grid omits a control change')
    opportunity_list=[_ns(t) for t in opportunities]
    if len(set(opportunity_list))!=len(opportunity_list):raise ValueError('duplicate forecast opportunity')
    event_set=set(opportunity_list)
    if not event_set<=set(map(int,grid[1:])):raise ValueError('opportunities must belong to the common motion grid')
    if not np.isfinite([process_noise_xy,process_noise_theta]).all() or min(process_noise_xy,process_noise_theta)<0:
        raise ValueError('invalid forecast process noise')
    records=[]
    for previous,t in zip(grid[:-1],grid[1:]):
        index=int(np.searchsorted(times,previous,side='right')-1)
        control=controls[index];dt=float(t-previous)*1e-9
        F=unicycle_jacobian(state,control,dt)
        Q=unicycle_process_noise(process_noise_xy,process_noise_theta,dt,theta=state[2],v=control[0])
        covariance=F@covariance@F.T+Q
        state=unicycle_step(state,control,dt)
        covariance=validate_covariance(covariance,name='forecast predicted covariance')
        if not np.isfinite(state).all():raise ValueError('nonfinite prescribed motion')
        if int(t) in event_set:
            covariance,q,support=forecast(state.copy(),covariance.copy(),method)
            covariance=validate_covariance(covariance,name='forecast posterior covariance')
            if not np.isfinite(q) or not 0<=q<=1:raise ValueError('invalid usable-observation probability')
            records.append(dict(time_s=float(t)*1e-9,usable_probability=float(q),support=support))
    return dict(state=state,covariance=covariance,opportunities=records,
                start_s=float(first)*1e-9,end_s=float(last)*1e-9,motion_steps=len(grid)-1)
