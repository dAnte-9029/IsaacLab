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


def test_failed_search_is_not_hard_rejection_and_success_is_conditional():
    path=Trajectory(TrajectorySpec((0.,)*4+(40.,)*4,(0.,)*4,(0.,)*4,40.))
    rules=dict(plants={'L3D':{'limits':{'kappa':None}}},evidence=[dict(backend='L3D',trajectory=path.spec.to_dict(),conditions={'controller':'a'},status='NOT FOUND',source='run')])
    result=screen_evidence(path,'L3D',rules,{'controller':'a'})
    assert result['status']=='NOT FOUND' and not result['hard_exclude']
    rules['evidence'][0]['status']='VERIFIED FEASIBLE'
    assert screen_evidence(path,'L3D',rules,{'controller':'a'})['status']=='VERIFIED FEASIBLE'
    assert screen_evidence(path,'L3D',rules,{'controller':'b'})['status']=='UNTESTED / UNCERTAIN'


def test_feature_priority_never_upgrades_an_unexecuted_recipe():
    from flapping_bot.path_tracking.trajectory_feasibility import training_priority
    original=Trajectory(TrajectorySpec((0.,)*4+(40.,)*4,(.01,)*4,(.1,)*4,40.))
    shorter=Trajectory(TrajectorySpec((0.,)*4+(30.,)*4,(.005,)*4,(.05,)*4,30.))
    rules=dict(plants={'L3D':{'limits':{'kappa':None}}},evidence=[dict(backend='L3D',trajectory=original.spec.to_dict(),conditions={},status='VERIFIED FEASIBLE',source='run',features=features(original))])
    result=training_priority(shorter,'L3D',rules,{})
    assert result['status']=='UNTESTED / UNCERTAIN'
    assert result['priority']=='candidate_near_success_requires_validation'
    assert not result['hard_exclude']
