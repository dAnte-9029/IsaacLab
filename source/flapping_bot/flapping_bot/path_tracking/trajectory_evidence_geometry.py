"""Offline geometric demands and revisable evidence lookup, never a speed task."""
import numpy as np
from scipy.interpolate import PPoly
from .unified_trajectory import Trajectory


def peak(spline, length):
    roots = PPoly.from_spline(spline.derivative()).roots(extrapolate=False)
    points = np.r_[0., length, spline.t, roots[np.isfinite(roots) & (roots >= 0) & (roots <= length)]]
    return float(np.max(np.abs(spline(points))))


def features(path: Trajectory):
    """Exact four spline peaks; spatial descriptors on <=0.2 m path grid.

    Active intervals exceed 10% of their own peak; this describes duration,
    not a feasibility threshold. Retain profiles to avoid peak-only screening.
    """
    s=path.s; q=path.query(s); ds=np.diff(s)
    result={key:peak(spline,path.spec.length_m) for key,spline in
            [('kappa',path.k),('gamma',path.g),('kappa_prime',path.k.derivative()),('gamma_prime',path.g.derivative())]}
    result.update(length_m=path.spec.length_m,net_climb_m=float(q['position'][-1,2]-q['position'][0,2]),
        climb_excursion_m=float(np.ptp(q['position'][:,2])),curvature_3d_max_per_m=float(q['curvature'].max()),
        feature_grid_max_step_m=float(ds.max()))
    active=[]
    for key in ['kappa','gamma']:
        mask=np.abs(q[key])>max(1e-12,.1*result[key]); active.append(mask)
        edges=np.diff(np.r_[False,mask,False].astype(int))
        intervals=[[float(s[start]),float(s[min(end,len(s)-1)])] for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
        result[key+'_active_intervals_m']=intervals
        result[key+'_max_active_length_m']=max([b-a for a,b in intervals],default=0.)
        flat=(np.abs(q[key])>=result[key]-1e-10)&(np.abs(q[key+'_prime'])<1e-10) if result[key]>1e-12 else np.zeros(len(s),dtype=bool)
        result[key+'_plateau_total_m']=float(np.sum(ds*(flat[:-1]&flat[1:])))
        result[key+'_signed_integral']=float(np.trapz(q[key],s))
    result['coupled_length_m']=float(np.sum(ds*(active[0][:-1]&active[1][:-1])))
    result['coupling_abs_integral']=float(np.trapz(np.abs(q['kappa']*q['gamma']),s))
    return result
