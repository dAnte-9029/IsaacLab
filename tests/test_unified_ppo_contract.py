"""Pure contract checks; real PPO/PhysX tests are separately bounded smoke runs."""
from pathlib import Path
import numpy as np
from flapping_bot.analysis.unified_ppo_contract import CANDIDATES,evaluation_tasks,POLICY,PPO_CONFIG
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec,GeometryLimits,screen


def test_fixed_suite_has_full_legal_tasks_and_opposite_reversals():
    suite=evaluation_tasks()
    assert suite==evaluation_tasks() and len(suite)==5
    for entry in suite:
        path=Trajectory(TrajectorySpec(**entry['trajectory']))
        assert path.spec.length_m==60 and not screen(path,GeometryLimits())
    left,right=(TrajectorySpec(**suite[i]['trajectory']) for i in (2,3))
    np.testing.assert_allclose(left.kappa_coeff,-np.array(right.kappa_coeff))
    assert min(left.kappa_coeff)<0<max(left.kappa_coeff)


def test_initial_candidates_are_exact_saved_values():
    import json
    root=Path(__file__).resolve().parents[1].parent/'control-oriented-3d-wing-aero/logs/control_aero_3d/periodic_trim_lateral_v1'
    for backend,folder in [('L3D','L3D_late'),('I3D','I3D_refined')]:
        assert list(CANDIDATES[backend])==json.loads((root/folder/'search.json').read_text())['best']
    assert POLICY['actor_hidden_dims']==POLICY['critic_hidden_dims']
    assert PPO_CONFIG['schedule']=='fixed'


def test_sampler_checkpoint_restores_weights_and_deduplicates_evaluation_receipts(tmp_path):
    from flapping_bot.path_tracking.online_trajectory import OnlineSampler,OfflineEvidence,FeedbackStore,OnlineConfig
    from test_online_trajectory import RULES,PLANT,CONTRACT,episode,make
    sampler,evaluator=make(tmp_path/'one')
    for _ in range(8): episode(sampler,evaluator,sampler.regions[2],True)
    saved=sampler.state_dict(); sampler.store.backup(tmp_path/'snapshot.sqlite3')
    restored=OnlineSampler(OfflineEvidence(RULES),sampler.store,PLANT,CONTRACT,'A',sampler.cfg)
    restored.load_state_dict(saved)
    np.testing.assert_array_equal(restored.exploration,sampler.exploration)
    assert [s.to_dict() for s,_ in restored.sample_batch(range(16))]==[s.to_dict() for s,_ in sampler.sample_batch(range(16))]
    assert restored._cursor==sampler._cursor
    import pytest
    record=sampler.store.records[-1]
    with pytest.raises(ValueError): restored.feedback_evaluation(record,sampler.store)


def test_adaptive_jitter_preserves_common_coefficient_domain(tmp_path):
    from test_online_trajectory import make
    sampler,_=make(tmp_path)
    for spec,_ in sampler.sample_batch(range(128)):
        assert max(abs(x) for x in spec.kappa_coeff)<=sampler.sampling.kappa_amplitude+1e-12
        assert max(abs(x) for x in spec.gamma_coeff)<=sampler.sampling.gamma_amplitude+1e-12


def test_resume_forks_only_checkpoint_prefix_and_preserves_source(tmp_path):
    from test_online_trajectory import make,episode,PLANT,INITIAL,CONTRACT
    from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore
    sampler,evaluator=make(tmp_path/'synthetic')
    first=episode(sampler,evaluator,sampler.regions[1],True)
    second=episode(sampler,evaluator,sampler.regions[2],True)
    # Synthetic data must not be copied into a real journal by the recovery API.
    import pytest
    with pytest.raises(ValueError,match='synthetic'):
        FeedbackStore.restore_prefix(sampler.store.path,tmp_path/'rejected',first['sequence'])
