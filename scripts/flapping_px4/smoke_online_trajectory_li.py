"""Real L/I CPU-native wiring, short fixed actions; no learner or injected success."""
import argparse
from pathlib import Path
import json
from time import perf_counter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['L3D','I3D'],required=True)
    p.add_argument('--plant-assets',type=Path,required=True)
    p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cpu',kit_args=(
        f'--ext-folder {args.plant_assets.resolve()}/source/flapping_bot/native_extensions '
        '--enable omni.flapping_bot.holonomic_constraint')).app
    env=None
    try:
        import numpy as np
        import torch
        from flapping_bot.direct.flapping_bot.unified_trajectory_env import UnifiedTrajectoryEnv,UnifiedTrajectoryEnvCfg
        from flapping_bot.physics.control_aero_3d import ControlAero3DConfig
        from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import _configure_robot_asset
        from flapping_bot.path_tracking.online_trajectory import FeedbackStore
        torch.set_num_threads(1)
        root=Path(__file__).resolve().parents[2]
        cfg=UnifiedTrajectoryEnvCfg(); cfg.seed=29; cfg.sim.device='cpu'
        cfg.scene.num_envs=4; cfg.scene.env_spacing=10.
        cfg.randomize_straight_line_heading=cfg.randomize_flap_phase_at_reset=False
        cfg.wind_enabled=cfg.wind_ou_enabled=False
        cfg.episode_length_s=.4
        cfg.control_aero_3d=ControlAero3DConfig(backend=args.backend,twist_amplitude_deg=35.,sections_per_wing=10)
        cfg.delaurier_wing_geom_csv=args.plant_assets.resolve()/'outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv'
        cfg.online_sampling=True; cfg.online_rules_path=str(args.rules.resolve())
        cfg.online_feedback_directory=str(args.output.resolve()/'evidence')
        cfg.online_policy_version='smoke-A'
        _configure_robot_asset(cfg,asset_path=root/'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf',
                               usd_dir=args.output.resolve()/'assets')
        env=UnifiedTrajectoryEnv(cfg)
        assert env._control_aero_model is not None and env.cfg.control_aero_3d.backend==args.backend
        obs,_=env.reset(); sampler=env.online_sampler; evaluator=env.rollout_evaluator
        assert sampler.plant['backend']==args.backend
        first_tasks=list(env.task_records); reset_ms=[]; online_ms=[]; baseline_ms=[]
        # Check exact contract boundary against a real successful offline witness.
        old=next(r for r in sampler.offline.records if r['backend']==args.backend and r['status']=='VERIFIED FEASIBLE')
        from flapping_bot.path_tracking.unified_trajectory import TrajectorySpec
        assert not sampler.offline.query(TrajectorySpec(**old['trajectory']),args.backend,
            sampler.plant,sampler.contract)['witnesses']
        action=env._actions.clone(); action[:,1]=.03
        for step in range(32):
            if step==6: env.set_policy_version('smoke-B')
            if step==8:
                protected=evaluator.active[1]['record']['instance_id']; task=env.task_records[1]
                cache=env.path_cache.values[1].clone(); t=perf_counter()
                env._reset_idx(torch.tensor([0])); reset_ms.append(1000*(perf_counter()-t))
                assert evaluator.active[1]['record']['instance_id']==protected and env.task_records[1]==task
                torch.testing.assert_close(cache,env.path_cache.values[1])
            t=perf_counter(); obs,reward,done,timeout,_=env.step(action); online_ms.append(1000*(perf_counter()-t))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert torch.isfinite(env._robot.data.root_state_w).all()
            assert torch.isfinite(env._act_cmd).all() and torch.isfinite(env._robot.data.joint_pos).all()
            for i,path in enumerate(env.paths):
                expected=path.query(float(env.progress[i])+np.array(cfg.preview_distances_m))['position']
                np.testing.assert_allclose(env.reference['preview'][i].cpu().numpy(),expected,atol=3e-5)
        for i in range(4):
            record=evaluator.abort(i)
            if record is not None: sampler.feedback(record)
        records=evaluator.store.records
        assert any(r['outcome']=='timeout' for r in records)
        assert any(r['outcome']=='manual_reset' for r in records)
        assert any(r['policy_execution']=='mixed' for r in records)
        assert all(r['outcome']!='completed' for r in records)
        assert all(r['frozen_checkpoint'] is None for r in records)
        assert all(r['control_data_finite'] for r in records)
        trace=next(r for r in records if r['steps']>0)
        assert list(evaluator.store.trace(trace['instance_id']))
        # Same plant and task environment, online hooks disabled for short overhead comparison.
        env.online_sampler=None; env.rollout_evaluator=None; env.reset()
        for step in range(32):
            t=perf_counter(); env.step(action); baseline_ms.append(1000*(perf_counter()-t))
        evaluator.store.close()
        reopened=FeedbackStore(args.output/'evidence')
        assert len(reopened.records)==len(records); reopened.close()
        report=dict(passed=True,backend=args.backend,plant_base='4303c1ad',twist_deg=35.,sections=10,
            num_envs=4,online_steps=32,baseline_steps=32,physics_dt=env.physics_dt,control_dt=env.step_dt,
            online_step_median_ms=float(np.median(online_ms[3:])),baseline_step_median_ms=float(np.median(baseline_ms[3:])),
            online_total_ms=sum(online_ms),baseline_total_ms=sum(baseline_ms),partial_reset_ms=reset_ms,
            terminal_outcomes=[r['outcome'] for r in records],executions=[r['policy_execution'] for r in records],
            checked=['actual L/I model','finite actual actuators/body/joints','distance preview','partial reset isolation',
                     'mixed per-action versions','timeout/manual reset evidence','no inherited offline success','restart summary/trace'],
            physical_success_claim=False,first_tasks=first_tasks)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))
        print('ONLINE_LI_SMOKE_PASSED',json.dumps({k:v for k,v in report.items() if k!='first_tasks'}),flush=True)
    finally:
        if env is not None: env.close()
        app.close()

if __name__=='__main__': main()
