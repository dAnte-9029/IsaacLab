"""Focused independent geometry/decision checks; no simulator imports."""
import numpy as np
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
from flapping_bot.path_tracking.trajectory_feasibility import features,screen_evidence


def test_curvature_matches_tangent_finite_difference():
    p=Trajectory(TrajectorySpec((0.,)*4+(80.,)*4,(.01,.02,-.01,.03),(0.,.2,-.1,.1),80.))
    s=np.linspace(1,79,111);h=1e-4;q=p.query(s)
    fd=(p.query(s+h)['tangent']-p.query(s-h)['tangent'])/(2*h)
    np.testing.assert_allclose(np.linalg.norm(fd,axis=1),q['curvature'],atol=1e-10)
    np.testing.assert_allclose(q['curvature'],np.sqrt(np.cos(q['gamma'])**2*q['kappa']**2+q['gamma_prime']**2),atol=1e-14)


def test_duration_and_climb_not_only_peaks():
    def path(length):return Trajectory(TrajectorySpec((0.,)*4+(length,)*4,(.01,)*4,(.1,)*4,length))
    a,b=features(path(40.)),features(path(80.))
    assert a['kappa']==b['kappa'] and a['gamma']==b['gamma']
    assert np.isclose(b['net_climb_m'],2*a['net_climb_m'])
    assert b['coupled_length_m']==2*a['coupled_length_m']


def fixture():
    from copy import deepcopy
    path=Trajectory(TrajectorySpec((0.,)*4+(40.,)*4,(0.,)*4,(0.,)*4,40.))
    physical=dict(plant='native',wind='zero',initial=[9.,0.,3.],phase=0.,seed=29)
    contract=dict(kind='offline',completion_fraction=.98)
    record=dict(backend='L3D',trajectory=path.spec.to_dict(),physical_conditions=physical,
        evaluation_contract=contract,status='VERIFIED FEASIBLE',source='A',features=features(path),
        witness=dict(controller_version='A',controller={'gain':1},actions={'source':'traceA'},result={'passed':True}))
    rules=dict(schema_version=2,plants={b:{'limits':{'kappa':None}} for b in ['L3D','I3D']},evidence=[deepcopy(record)])
    return path,physical,contract,record,rules


def test_success_survives_controller_change_and_later_failure():
    from copy import deepcopy
    from flapping_bot.path_tracking.trajectory_feasibility import append_evidence
    path,physical,contract,record,rules=fixture()
    failed=deepcopy(record);failed.update(status='NOT FOUND',source='B')
    failed['witness']=dict(controller_version='PPO_B',controller={'gain':99},actions={'source':'traceB'},result={'passed':False})
    append_evidence(rules,failed)
    result=screen_evidence(path,'L3D',rules,physical,contract)
    assert result['status']=='VERIFIED FEASIBLE'
    assert len(result['evidence'])==2 and len(result['successful_witnesses'])==1
    rules['evidence'].reverse()
    assert screen_evidence(path,'L3D',rules,physical,contract)['status']=='VERIFIED FEASIBLE'
    rules['evidence']=[failed]
    result=screen_evidence(path,'L3D',rules,physical,contract)
    assert result['status']=='NOT FOUND' and not result['hard_exclude']


def test_physics_and_contract_isolation_and_rl_append():
    from copy import deepcopy
    from flapping_bot.path_tracking.trajectory_feasibility import append_evidence
    path,physical,contract,record,rules=fixture()
    for key,value in [('plant','other'),('wind','gust'),('initial',[8.,0.,3.]),('phase',1.),('seed',30)]:
        changed=dict(physical);changed[key]=value
        assert screen_evidence(path,'L3D',rules,changed,contract)['status']=='UNTESTED / UNCERTAIN'
    assert screen_evidence(path,'I3D',rules,physical,contract)['status']=='UNTESTED / UNCERTAIN'
    for key,value in [('origin',(1.,0.,20.)),('heading_rad',1.),('seed',1)]:
        spec=path.spec.to_dict();spec[key]=value
        assert screen_evidence(Trajectory(TrajectorySpec(**spec)),'L3D',rules,physical,contract)['status']=='UNTESTED / UNCERTAIN'
    rl=dict(kind='rl',completion_tolerance_m=.5)
    assert screen_evidence(path,'L3D',rules,physical,rl)['status']=='RELATED EVIDENCE / TARGET UNVERIFIED'
    new=deepcopy(record);new['evaluation_contract']=rl;new['witness']['controller_version']='PPO'
    append_evidence(rules,new)
    assert screen_evidence(path,'L3D',rules,physical,rl)['status']=='VERIFIED FEASIBLE'
    assert len(rules['evidence'])==2


def test_feature_priority_never_upgrades_an_unexecuted_recipe():
    from flapping_bot.path_tracking.trajectory_feasibility import training_priority
    path,physical,contract,record,rules=fixture()
    shorter=Trajectory(TrajectorySpec((0.,)*4+(30.,)*4,(0.,)*4,(0.,)*4,30.))
    result=training_priority(shorter,'L3D',rules,physical,contract)
    assert result['status']=='UNTESTED / UNCERTAIN'
    assert result['priority']=='candidate_near_success_requires_validation'


def test_historical_migration_preserves_all_sources_and_outcomes():
    import json
    from pathlib import Path
    from collections import Counter,defaultdict
    root=Path(__file__).resolve().parents[1]/'logs/unified_trajectory_phase2_v1'
    old=json.loads((root/'screening_rules.json').read_text())
    new=json.loads((root/'screening_rules_v2.json').read_text())
    assert len(old['evidence'])==len(new['evidence'])==88
    assert Counter(e['status'] for e in new['evidence'])=={'VERIFIED FEASIBLE':35,'NOT FOUND':53}
    groups=defaultdict(list)
    for index,(a,b) in enumerate(zip(old['evidence'],new['evidence'])):
        assert b['legacy_record']==a and b['source']==a['source']
        assert b['provenance']['legacy_index']==index
        results=json.loads(Path(b['source']).read_text())
        assert results[b['provenance']['case_index']]==b['witness']['result']
        path=Trajectory(TrajectorySpec(**b['trajectory']))
        result=screen_evidence(path,b['backend'],new,b['physical_conditions'],b['evaluation_contract'])
        if b['status']=='VERIFIED FEASIBLE':assert result['status']=='VERIFIED FEASIBLE'
        rl=json.loads((root/'rl_default_contract_v1.json').read_text())
        assert screen_evidence(path,b['backend'],new,b['physical_conditions'],rl)['status']!='VERIFIED FEASIBLE'
        groups[json.dumps([b['backend'],b['trajectory']],sort_keys=True)].append(b)
    mixed=[rows for rows in groups.values() if len({r['status'] for r in rows})>1]
    assert mixed  # Real historical controller successes/failures coexist and are retained.
    for rows in mixed:
        assert any(r['witness']['controller']!=rows[0]['witness']['controller'] for r in rows)
