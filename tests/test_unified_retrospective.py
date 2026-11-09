import sqlite3
import numpy as np
import pytest
from flapping_bot.analysis.unified_retrospective import immutable_store,trend_event,geometry_profile,rotate_inverse
from flapping_bot.analysis.unified_ppo_contract import evaluation_tasks
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec


def test_immutable_database_has_no_side_effects(tmp_path):
    path=tmp_path/'rollout_feedback.sqlite3';db=sqlite3.connect(path);db.execute('create table fixture(x)');db.commit();db.close()
    before=path.read_bytes();names=list(tmp_path.iterdir());s=immutable_store(tmp_path)
    assert s.db.execute('pragma integrity_check').fetchone()[0]=='ok';s.close()
    assert path.read_bytes()==before and list(tmp_path.iterdir())==names
    (tmp_path/'rollout_feedback.sqlite3-wal').write_bytes(b'nonempty')
    with pytest.raises(ValueError,match='WAL'):immutable_store(tmp_path)


def test_path_frame_and_geometry():
    tasks=evaluation_tasks();path=Trajectory(TrajectorySpec(**tasks[1]['trajectory']))
    q=path.query(np.array([0.,20.,40.,60.]));basis=np.stack([q['tangent'],q['normal'],q['up']],axis=1)
    np.testing.assert_allclose(basis@basis.transpose(0,2,1),np.broadcast_to(np.eye(3),(4,3,3)),atol=1e-14)
    assert np.all(np.linalg.det(basis)>0)
    g=geometry_profile(path);assert g['kappa_max_same_sign_m']>40 and g['flat_sustained40'].any() and g['negative_turn'].any()
    assert not g['kappa_has_reversal']
    reverse=geometry_profile(Trajectory(TrajectorySpec(**tasks[2]['trajectory'])))
    assert reverse['kappa_has_reversal'] and reverse['kappa_reversal'].any()
    np.testing.assert_allclose(rotate_inverse([1,0,0,0],[1,2,3]),[1,2,3])


def test_trend_thresholds_and_constant_motion():
    t=np.arange(121)/60
    assert trend_event(np.ones(121),t,1) is None
    a=.5+1.1*t;event=trend_event(a,t,1)
    assert event['start_s']==0 and event['end_s']==.5
    assert trend_event(a,t,-1) is None
    assert trend_event(a[::-1],t,-1)['start_s']==0
