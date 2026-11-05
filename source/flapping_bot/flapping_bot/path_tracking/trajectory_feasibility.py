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


def contract_implies(observed, target):
    """Only exact, fully specified contracts are currently proven equivalent.

    No ordering is inferred from a tolerance alone: projections, stopping time,
    recovery and safety predicates must also agree.
    """
    return bool(observed) and observed == target


def append_evidence(rules, record):
    """Append a separately evaluated witness; never replace earlier outcomes.

    Caller owns validation of actions/outcome against the supplied full contract.
    This API records evidence, it does not certify an unevaluated policy.
    """
    from copy import deepcopy
    if rules.get('schema_version') != 2:
        raise ValueError('Migrate evidence to schema_version=2 first')
    for key in ('backend', 'trajectory', 'physical_conditions', 'evaluation_contract',
                'witness', 'source', 'status'):
        if not record.get(key):
            raise ValueError('Missing evidence field: '+key)
    if record['status'] not in ('VERIFIED FEASIBLE', 'NOT FOUND'):
        raise ValueError('Unsupported evidence outcome')
    if not all(record['witness'].get(k) is not None for k in
               ('controller_version', 'controller', 'actions', 'result')):
        raise ValueError('Incomplete control witness')
    rules['evidence'].append(deepcopy(record))


def screen_evidence(path, backend, rules, conditions, evaluation_contract):
    """Existential plant feasibility, conditional on exact physics and contract.

    conditions contains physics only, never the proposed controller. Seed,
    trajectory origin and heading are strictly matched; no symmetry is assumed.
    NOT FOUND describes stored searches only, never physical impossibility.
    """
    if rules.get('schema_version') != 2:
        raise ValueError('Migrate evidence to schema_version=2 first')
    if not evaluation_contract:
        raise ValueError('An explicit target evaluation contract is required')
    descriptor=features(path)
    if not all(np.isfinite(v).all() for v in path.query(path.s).values()):
        return dict(status='HARD EXCLUSION',hard_exclude=True,reason='nonfinite geometry')
    matches=[r for r in rules['evidence'] if r['backend']==backend
             and r['trajectory']==path.spec.to_dict() and r['physical_conditions']==conditions]
    compatible=[r for r in matches if contract_implies(r['evaluation_contract'], evaluation_contract)]
    successes=[r for r in compatible if r['status']=='VERIFIED FEASIBLE']
    related=[r for r in matches if r['status']=='VERIFIED FEASIBLE' and r not in compatible]
    if successes:
        status='VERIFIED FEASIBLE';priority='supported_recipe'
    elif related:
        status='RELATED EVIDENCE / TARGET UNVERIFIED';priority='needs_target_validation'
    elif any(r['status']=='NOT FOUND' for r in compatible):
        status='NOT FOUND';priority='lower_priority_revisable'
    else:
        status='UNTESTED / UNCERTAIN';priority='needs_evidence'
    return dict(status=status,priority=priority,hard_exclude=False,features=descriptor,
                evaluation_contract=evaluation_contract,
                evidence=matches,successful_witnesses=successes,related_evidence=related,
                limits=rules['plants'][backend]['limits'])


def training_priority(path, backend, rules, conditions, evaluation_contract):
    """Feature proximity is a proposal heuristic, never a feasibility theorem."""
    result=screen_evidence(path,backend,rules,conditions,evaluation_contract)
    if result['status']!='UNTESTED / UNCERTAIN':return result
    f=result['features'];sources=[]
    policy=rules.get('feature_priority_policy',{})
    if not policy.get('enabled',True):return result
    keys=['kappa','gamma','kappa_prime','gamma_prime','length_m',
          'kappa_max_active_length_m','gamma_max_active_length_m',
          'climb_excursion_m','coupled_length_m','coupling_abs_integral']
    keys=[k for k in policy.get('fields',keys) if k!='net_climb_m']
    for record in rules['evidence']:
        if (record['backend']!=backend or record['status']!='VERIFIED FEASIBLE'
                or record['physical_conditions']!=conditions
                or not contract_implies(record['evaluation_contract'],evaluation_contract)):continue
        old=record['features']
        if (all(f[k]<=old[k]+1e-10 for k in keys)
                and abs(f['net_climb_m'])<=abs(old['net_climb_m'])+1e-10):
            sources.append(record['source'])
    if sources:
        result['priority']='candidate_near_success_requires_validation'
        result['priority_sources']=sources
        result['warning']='Feature-wise interpolation, sign/order changes and shorter tasks remain unverified.'
    return result


def main():
    import argparse
    import json
    from pathlib import Path
    from .unified_trajectory import TrajectorySpec
    p=argparse.ArgumentParser()
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--conditions',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--backend',choices=['L3D','I3D'],required=True)
    a=p.parse_args()
    try:
        path=Trajectory(TrajectorySpec(**json.loads(a.trajectory.read_text())))
    except (ValueError,TypeError) as exc:
        print(json.dumps(dict(status='HARD EXCLUSION',hard_exclude=True,reason='Invalid mathematical representation: '+str(exc))))
        return
    print(json.dumps(training_priority(path,a.backend,json.loads(a.rules.read_text()),json.loads(a.conditions.read_text()),json.loads(a.contract.read_text())),indent=2))

if __name__=='__main__':main()
