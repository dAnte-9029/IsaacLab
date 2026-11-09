"""Pure CPU reconstruction helpers for archived Unified trajectories."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore


def immutable_store(directory):
    """Closed archives only: immutable URI avoids creating/updating SQLite SHM.

    Nonempty WAL is refused, never silently ignored. No database copy is needed.
    """
    path=Path(directory).resolve()/'rollout_feedback.sqlite3'
    wal=Path(str(path)+'-wal')
    if not path.is_file() or (wal.exists() and wal.stat().st_size):
        raise ValueError(f'Missing database or nonempty WAL; needs consistent external snapshot: {path}')
    store=object.__new__(FeedbackStore)
    store.path=path; store.readonly=True; store.simulated=False
    store.db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    return store


def rotate_inverse(q,v):
    q=np.asarray(q,np.float32);v=np.asarray(v,np.float32)
    t=2*np.cross(q[...,1:],v)
    return v-q[...,0,None]*t+np.cross(q[...,1:],t)


def observation(control, cfg, length, joint_ids):
    """POST state k -> actor input for raw action k+1 (same physical episode)."""
    r=np.asarray(control['root_state'],np.float32);q=r[3:7];ref=control['reference']
    rot=lambda v:rotate_inverse(q,v)
    phase=np.float32(control['phase'])
    x=np.concatenate([rot(r[7:10])/10,rot(r[10:13])/10,rot([0,0,-1]),
        rot(ref['error'])/cfg['tracking_scale_m'],rot(ref['tangent']),
        *[rot(np.asarray(p,np.float32)-r[:3])/d for p,d in zip(ref['preview'],cfg['preview_distances_m'])],
        np.asarray([ref[k] for k in ('kappa','gamma','kappa_prime','gamma_prime')],np.float32),
        [np.sin(phase),np.cos(phase)],control['command'],[np.float32(control['actuator_state']['_freq'])/5],
        np.asarray(control['joint_pos'],np.float32)[joint_ids],
        [1-np.float32(control['projection']['progress'])/np.float32(length)]])
    assert x.shape==(39,)
    return np.nan_to_num(x,nan=0,posinf=5,neginf=-5).clip(-5,5).astype(np.float32)


@torch.no_grad()
def actor(obs, state):
    x=torch.as_tensor(np.asarray(obs),dtype=torch.float32)
    for i in (0,2,4,6):
        x=F.linear(x,state[f'actor.{i}.weight'],state[f'actor.{i}.bias'])
        if i!=6:x=F.elu(x)
    return x


def first_event(mask,t):
    ix=np.flatnonzero(mask)
    return float(t[ix[0]]) if len(ix) else None


def trend_event(error,t,direction):
    """First 0.5 s window: |ey| >= .5 m at start, net change >= .5 m,
    at least 80% interval differences in the specified direction.
    Derivative includes moving frame; no vy == dey/dt assumption.
    """
    a=np.abs(error); width=30
    for i in range(len(a)-width):
        delta=np.diff(a[i:i+width+1])*direction
        if a[i]>=.5 and direction*(a[i+width]-a[i])>=.5 and np.mean(delta>0)>=.8:
            return dict(start_s=float(t[i]),end_s=float(t[i+width]),abs_error_start_m=float(a[i]),abs_error_end_m=float(a[i+width]))
    return None


def failure_reasons(s,c):
    return [n for n,b in [('path_error',s['path_error_m']>c['max_path_error_m']),
        ('height_low',s['height_m']<c['min_height_m']),('height_high',s['height_m']>c['max_height_m']),
        ('tilt',s['tilt_cos']<c['min_tilt_cos']),('body_rate',s['body_rate_rad_s']>c['max_body_rate_rad_s']),
        ('nonfinite',not s['finite'])] if b]


def geometry_profile(path):
    """Geometry-only labels on a 0.2 m grid; absolute activation thresholds.

    kappa=d heading/ds: active > .002 rad/m; gamma active > .02 rad.
    Sustained: connected same-sign active interval >=10 m. Reverse: at least
    two active sign runs of >=2 m with opposite signs (zero gaps allowed).
    Local reversal window: +/-5 m from transition between accepted runs.
    """
    grid=np.linspace(0,path.spec.length_m,int(round(path.spec.length_m/.2))+1)
    q=path.query(grid);result={'grid':grid};active=[]
    for key,threshold in [('kappa',.002),('gamma',.02)]:
        sign=np.where(np.abs(q[key])>threshold,np.sign(q[key]),0)
        edges=np.r_[0,np.flatnonzero(np.diff(sign)!=0)+1,len(sign)]
        runs=[(int(sign[a]),float(grid[a]),float(grid[min(b,len(grid)-1)])) for a,b in zip(edges[:-1],edges[1:]) if sign[a]!=0]
        accepted=[r for r in runs if r[2]-r[1]>=2]
        reversals=[.5*(a[2]+b[1]) for a,b in zip(accepted[:-1],accepted[1:]) if a[0]!=b[0]]
        sustained=np.zeros(len(grid),bool)
        for _,a,b in runs:
            if b-a>=10:sustained|=(grid>=a)&(grid<=b)
        result[key+'_sustained']=sustained
        result[key+'_reversal']=np.any([abs(grid-r)<=5 for r in reversals],axis=0) if reversals else np.zeros(len(grid),bool)
        result[key+'_has_reversal']=bool(reversals);result[key+'_max_same_sign_m']=max([b-a for _,a,b in runs],default=0.)
        active.append(sign!=0)
    result['kappa_sustained40']=np.zeros(len(grid),bool)
    sign=np.where(abs(q['kappa'])>.002,np.sign(q['kappa']),0)
    edges=np.r_[0,np.flatnonzero(np.diff(sign)!=0)+1,len(sign)]
    for a,b in zip(edges[:-1],edges[1:]):
        if sign[a] and grid[min(b,len(grid)-1)]-grid[a]>=40:
            result['kappa_sustained40'][a:b]=True
    result['flat_sustained40']=result['kappa_sustained40'] & (abs(q['gamma'])<=.02)
    result['negative_turn']=q['kappa']<-.002
    result['positive_turn']=q['kappa']>.002
    result['straight']=~(active[0]|active[1]);result['coupled']=active[0]&active[1]
    result['climb']=q['gamma']>.02;result['descent']=q['gamma']<-.02
    result['primary']=('coupled' if np.any(result['coupled']) else 'reverse' if result['kappa_has_reversal'] or result['gamma_has_reversal'] else 'sustained_turn' if np.any(result['kappa_sustained']) else 'sustained_vertical' if np.any(result['gamma_sustained']) else 'straight_or_short')
    return result
