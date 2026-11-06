"""Mechanism/integrity checks; all manufactured feedback uses simulation namespace."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import os
import numpy as np
import pytest
from flapping_bot.path_tracking.online_trajectory import (
    OfflineEvidence, FeedbackStore, RolloutEvaluator, OnlineSampler, OnlineConfig, vector)
from flapping_bot.path_tracking.unified_trajectory import SamplingConfig, TrajectorySpec, sample_trajectory

ROOT=Path(__file__).resolve().parents[1]
RULES=Path(os.environ.get('UNIFIED_RULES',ROOT.parent/'control-oriented-3d-wing-aero/logs/unified_trajectory_phase2_v1/screening_rules_v2.json'))
PLANT=dict(backend='L3D',twist_deg=35.,scope='synthetic fixture')
INITIAL=dict(position=[0.,0.,20.],speed=8.)
CONTRACT=dict(kind='unified_terminal_v1',completion_tolerance_m=.5,max_path_error_m=8.,min_height_m=1.,
              max_height_m=60.,min_tilt_cos=.1,max_body_rate_rad_s=20.,max_steps=4)
CONTROL=dict(command=[0.]*4,applied=[0.]*4,joint_pos=[0.]*5,joint_vel=[0.]*5,root_state=[0.]*13,phase=0.)


def make(tmp_path, cfg=OnlineConfig(region_count=16,capacity=64)):
    store=FeedbackStore(tmp_path,simulated=True)
    return OnlineSampler(OfflineEvidence(RULES),store,PLANT,CONTRACT,'A',cfg),RolloutEvaluator(store)


def episode(sampler,evaluator,spec,success=False,*,versions=('A',),mode='training',env_id=0,instance_id=None):
    evaluator.begin(env_id,spec,PLANT,INITIAL,CONTRACT,versions[0],'test',evaluation_mode=mode,instance_id=instance_id)
    for step,version in enumerate(versions,1):
        last=step==len(versions)
        result=evaluator.observe(env_id,progress_m=60. if last and success else 10.,
            path_error_m=9. if last and not success else 0.,height_m=20.,tilt_cos=1.,body_rate_rad_s=0.,
            finite=True,step=step,control=CONTROL,policy_version=version,execution_mode=mode)
    sampler.feedback(result)
    return result


def test_offline_is_readonly_and_exact():
    before=RULES.read_bytes(); offline=OfflineEvidence(RULES)
    assert offline.count==88
    r=next(x for x in offline.records if x['status']=='VERIFIED FEASIBLE'); spec=TrajectorySpec(**r['trajectory'])
    assert offline.query(spec,r['backend'],r['physical_conditions'],r['evaluation_contract'])['witnesses']
    assert offline.query(spec,r['backend'],r['physical_conditions'],CONTRACT)['status']=='RELATED EVIDENCE / TARGET UNVERIFIED'
    for backend in ['DeLaurier', 'L3D' if r['backend']=='I3D' else 'I3D']:
        assert not offline.query(spec,backend,r['physical_conditions'],r['evaluation_contract'])['witnesses']
    assert not offline.query(spec,r['backend'],dict(r['physical_conditions'],twist_deg=0),r['evaluation_contract'])['witnesses']
    assert RULES.read_bytes()==before


def test_mixed_unknown_training_and_frozen_are_separate(tmp_path):
    sampler,evaluator=make(tmp_path); spec=sampler.regions[3]
    mixed=episode(sampler,evaluator,spec,True,versions=('A','B'))
    unknown=episode(sampler,evaluator,spec,True,versions=(None,))
    assert mixed['policy_execution']=='mixed' and mixed['frozen_checkpoint'] is None
    assert unknown['policy_execution']=='unknown' and unknown['frozen_checkpoint'] is None
    assert sampler.statistics(spec,'A')['count']==sampler.statistics(spec,'B')['count']==0
    assert sampler.statistics(spec,stream='training')['successes']==2
    assert len(sampler.store.query(spec,PLANT,INITIAL,CONTRACT)['witnesses'])==2
    for _ in range(8): episode(sampler,evaluator,spec,True,mode='frozen')
    assert sampler.statistics(spec,'A')['rate']==1
    old=sampler.exploration.copy(); sampler.set_policy_version('B')
    np.testing.assert_array_equal(old,sampler.exploration)
    assert sampler.statistics(spec,'B')['rate'] is None
    for _ in range(8): episode(sampler,evaluator,spec,False,versions=('B',),mode='frozen')
    assert sampler.statistics(spec,'B')['rate']==0
    assert sampler.forgetting.max()>0
    # Even a purported frozen evaluation crossing versions is not checkpoint evidence.
    mixed=episode(sampler,evaluator,spec,True,versions=('A','B'),mode='frozen')
    assert mixed['frozen_checkpoint'] is None
    assert sampler.statistics(spec,'B')['successes']==0
    trace=list(sampler.store.trace(mixed['instance_id']))
    assert [x['policy_version'] for x in trace]==['A','B']


@pytest.mark.parametrize('field',RolloutEvaluator.CONTROL_FIELDS)
def test_nonfinite_control_can_never_certify(tmp_path,field):
    sampler,evaluator=make(tmp_path); spec=sampler.regions[1]
    evaluator.begin(0,spec,PLANT,INITIAL,CONTRACT,'A','test')
    control=deepcopy(CONTROL); control[field]=float('nan')
    result=evaluator.observe(0,progress_m=60.,path_error_m=0.,height_m=20.,tilt_cos=1.,body_rate_rad_s=0.,
        finite=True,step=1,control=control,policy_version='A')
    assert result['outcome']=='invalid_data'
    assert not sampler.store.query(spec,PLANT,INITIAL,CONTRACT)['witnesses']
    assert list(sampler.store.trace(result['instance_id']))[0]['control'][field]=='nan'


def test_restart_append_only_duplicates_and_partial_reset(tmp_path):
    sampler,evaluator=make(tmp_path); spec=sampler.regions[2]
    record=episode(sampler,evaluator,spec,True,instance_id='permanent')
    for _ in range(80): episode(sampler,evaluator,spec)
    assert len(sampler.capability.training)==64
    assert not hasattr(sampler.store,'_records') and not hasattr(sampler.store,'_by_id')
    assert all('trace' not in r for r in sampler.store.iter_records())
    before=sampler.exploration.copy(); sampler.store.close()
    restarted,ev=make(tmp_path)
    np.testing.assert_allclose(restarted.exploration,before)
    assert restarted.store.query(spec,PLANT,INITIAL,CONTRACT)['witnesses']
    with pytest.raises(Exception): episode(restarted,ev,spec,True,instance_id='permanent')
    with pytest.raises(ValueError): restarted.feedback(record)
    with pytest.raises(ValueError): restarted.feedback(dict(record,instance_id='forged'))
    real=FeedbackStore(tmp_path); assert not real.query(spec,PLANT,INITIAL,CONTRACT)['witnesses']; real.close()
    for i in [0,1]: ev.begin(i,spec,PLANT,INITIAL,CONTRACT,'A','test')
    protected=deepcopy(ev.active[1]); aborted=ev.abort(0)
    assert aborted['outcome']=='manual_reset' and ev.active[1]==protected
    assert ev.abort(0) is None
    result=None
    for step in range(1,5):
        result=ev.observe(1,progress_m=2.,path_error_m=0.,height_m=20.,tilt_cos=1.,body_rate_rad_s=0.,
                          finite=True,step=step,control=CONTROL)
    assert result['outcome']=='timeout'


def test_incremental_rate_limit_and_actual_branches(tmp_path):
    sampler,evaluator=make(tmp_path); spec=sampler.regions[3]
    initial=sampler.exploration.copy()
    for _ in range(7): episode(sampler,evaluator,spec,True)
    np.testing.assert_array_equal(initial,sampler.exploration)
    episode(sampler,evaluator,spec,True)
    assert .5*np.abs(initial-sampler.exploration).sum()<=sampler.cfg.update_rate
    # Sampling must neither enumerate persisted summaries nor reconstruct capability.
    sampler.store.iter_records=lambda: (_ for _ in ()).throw(AssertionError('history replay'))
    samples=sampler.sample_batch(range(128)); sources={src for _,src in samples}
    assert {'base','retention','retention_fallback_base','exploration_kernel','exploration_candidates'}<=sources
    assert all(s.length_m==60 for s,_ in samples)
    # Force incompatible local altitude: fallback itself may also be rejected, so use length mismatch instead.
    sampler.regions=[replace(s,length_m=61.) for s in sampler.regions]
    sampler.anchors=[replace(s,length_m=61.) if s else None for s in sampler.anchors]
    sampler.witness_count=0
    sources={src for _,src in sampler.sample_batch(range(64))}
    assert 'exploration_fallback_base' in sources and 'retention_fallback_base' in sources


def test_geometric_features_do_not_prove_order_equivalence():
    spec=sample_trajectory(7,SamplingConfig(required_reversals=('kappa','gamma'))).spec
    spec=replace(spec,gamma_coeff=tuple(0. for _ in spec.gamma_coeff))
    other=replace(spec,kappa_coeff=tuple(-v for v in spec.kappa_coeff))
    np.testing.assert_allclose(vector(spec),vector(other),atol=1e-12)
    assert spec.to_dict()!=other.to_dict()


def test_same_checkpoint_changes_are_not_forgetting_and_legacy_is_not_dropped(tmp_path):
    sampler,evaluator=make(tmp_path/'new'); spec=sampler.regions[3]
    for _ in range(8): episode(sampler,evaluator,spec,True,mode='frozen')
    for _ in range(8): episode(sampler,evaluator,spec,False,mode='frozen')
    assert not sampler.forgetting.any()
    assert not sampler.store.query(spec,dict(PLANT,twist_deg=0),INITIAL,CONTRACT)['witnesses']
    assert not sampler.store.query(spec,PLANT,INITIAL,dict(CONTRACT,max_steps=5))['witnesses']
    legacy=tmp_path/'legacy'; legacy.mkdir(); original='{"status":"VERIFIED FEASIBLE"}\n'
    (legacy/'rollout_feedback.jsonl').write_text(original)
    with pytest.raises(ValueError,match='Legacy v1'): FeedbackStore(legacy)
    assert (legacy/'rollout_feedback.jsonl').read_text()==original
