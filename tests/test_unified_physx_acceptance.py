"""Smoke snapshot guards only; no PhysX, optimizer or PPO construction."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from flapping_bot.analysis.unified_ppo_contract import SMOKE_SCHEMA, require_checkpoint_use
from flapping_bot.path_tracking.online_trajectory import OnlineSampler
from test_unified_pretraining import function


def fixture():
    return dict(schema=SMOKE_SCHEMA,smoke_fixture=True,training_updates=0)


@pytest.mark.parametrize('resume',[False,True])
def test_training_rejects_smoke_before_environment(resume):
    from pathlib import Path
    from types import SimpleNamespace as NS
    calls=[]
    train=function('train',torch=NS(set_num_threads=lambda _:None,manual_seed=lambda _:None,
        load=lambda *a,**kw:fixture()), require_checkpoint_use=require_checkpoint_use,
        build_env=lambda *a,**kw:calls.append('build'))
    args=NS(checkpoint=Path('fixture.pt'),seed=0,threads=1,resume=resume)
    with pytest.raises(ValueError,match='evaluation-only'): train(args,Path('.'))
    assert calls==[]


def test_smoke_requires_explicit_entry():
    with pytest.raises(ValueError,match='evaluation-only'): require_checkpoint_use(fixture())
    require_checkpoint_use(fixture(),smoke_evaluation=True)
    for key,value in [('training_updates',1),('optimizer',{}),('smoke_fixture',False)]:
        bad=fixture(); bad[key]=value
        with pytest.raises(ValueError): require_checkpoint_use(bad,smoke_evaluation=True)
    with pytest.raises(ValueError): require_checkpoint_use({'schema':'unified_li_ppo_v1'},smoke_evaluation=True)


def test_smoke_feedback_is_rejected_before_any_state_access():
    receive=function('receive_evaluation')
    state={'pending_evaluation':True}; before=deepcopy(state)
    with pytest.raises(ValueError,match='Smoke evaluation'):
        receive(None,state,None,{'smoke_fixture':True},'v',256)
    assert state==before
    # No sampler/store state is touched, even if the caller bypasses the receiver.
    with pytest.raises(ValueError,match='independently frozen'):
        OnlineSampler.feedback_evaluation(None,{'smoke_fixture':True},None)


def test_environment_and_evidence_timeout_agree(tmp_path):
    from test_unified_final_repairs import tensor_env
    env,methods=tensor_env(tmp_path)
    env.progress[:]=1.
    for step in range(1,5):
        env.episode_length_buf[:]=step
        done,timeout=methods['_get_dones'](env)
        assert not done.item()
        assert timeout.item()==(step==4)
    assert len(env.records)==1 and env.records[0]['outcome']=='timeout'
    methods['_get_dones'](env)
    assert len(env.records)==1
    env.rollout_evaluator.store.close()
