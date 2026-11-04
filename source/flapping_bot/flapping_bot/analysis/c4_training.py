"""CPU-native six-task C4 training orchestration and resumable optimizer state."""

from dataclasses import asdict
import copy
import json
import hashlib
from pathlib import Path

import torch

C4_TRAINING_SCHEMA = 'pure_rl_c4_joint_training_v1'
TASK_NAMES = ('c1', 'c2c', 'c3a', 'c3b', 'c3c', 'c4')
TASK_WEIGHTS = (.09, .12, .09, .15, .15, .40)


def prepare_agent_config(source: dict, *, seed: int, steps: int, minibatches: int, save_interval: int) -> dict:
    """Reuse the source architecture, rejecting unsupported input semantics."""
    import copy
    config = copy.deepcopy(source)
    if (config['policy']['class_name'] != 'PureRLSplitActorCritic'
            or config.get('empirical_normalization')
            or config['policy'].get('actor_obs_normalization')
            or config['policy'].get('critic_obs_normalization')
            or config.get('clip_actions') is not None):
        raise ValueError('C4 requires the recorded split actor and fixed normalization')
    if min(steps, minibatches, save_interval) < 1:
        raise ValueError('Rollout, minibatch and save settings must be positive')
    config.update(seed=seed, device='cpu', num_steps_per_env=steps, save_interval=save_interval,
                  obs_groups={'policy': ['policy'], 'critic': ['policy']}, logger='tensorboard')
    config['algorithm']['num_mini_batches'] = minibatches
    config['algorithm']['normalize_advantage_per_mini_batch'] = False
    return config


def train_c4(*, checkpoint: Path, output: Path, root: Path, resume: bool, seed: int,
             num_envs: int, iterations: int, steps: int, minibatches: int, save_interval: int,
             retention_variant: str | None = None, memory_path: Path | None = None,
             memory_coefficient: float = 1.0, collect_memory_steps: int = 0,
             anchoring_mode: str | None = None, anchors_path: Path | None = None) -> dict:
    """Train after Kit startup; resume optimizer/progress with fresh simulation episodes."""
    import csv
    import types
    import yaml
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_yaml
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLC4JointEnvCfg,
    )
    from flapping_bot.px4_like.pure_rl_split_actor import register_pure_rl_split_actor_critic
    from flapping_bot.px4_like.rl_training_utils import maybe_enable_ppo_warm_start_guard, maybe_enable_task_aware_ppo
    from flapping_bot.path_tracking.c4_curriculum import C4Curriculum, C4CurriculumConfig
    from flapping_bot.px4_like.c4_retention import (
        BehaviorMemory, C4UpdateDiagnostics, TrainingEpisodeTracker, attach_behavior_memory,
    )
    from flapping_bot.px4_like.behavioral_anchoring import register_anchored_ppo, TailAnchors

    if anchoring_mode and (not resume or retention_variant != 'D' or collect_memory_steps):
        raise ValueError('Anchoring requires D resume, with no memory collection')

    if retention_variant not in (None, 'A', 'B', 'C', 'D'):
        raise ValueError('Unknown retention variant')
    if collect_memory_steps and (resume or retention_variant is not None or memory_path is not None):
        raise ValueError('Memory collection is a separate frozen-source operation')
    if (retention_variant in ('C', 'D')) != (memory_path is not None):
        raise ValueError('Only variants C/D require a frozen memory file')
    curriculum = C4Curriculum(seed=seed + 49999) if retention_variant in ('B', 'D') else None
    memory = None
    memory_payload = None
    if memory_path is not None:
        memory_payload = torch.load(memory_path, map_location='cpu', weights_only=True)
        memory = BehaviorMemory(memory_payload, seed=seed + 67867967)
        if not resume and memory_payload['provenance']['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('Memory teacher differs from weights-only initialization')

    source_agent = yaml.safe_load((checkpoint.parent / 'params/agent.yaml').read_text())
    source_provenance = json.loads((checkpoint.parent / 'curriculum_source.json').read_text())
    if str(source_provenance.get('heading_canonical_observation')).lower() != 'true':
        raise ValueError('Missing heading-canonical checkpoint provenance')
    agent = prepare_agent_config(source_agent, seed=seed, steps=steps, minibatches=minibatches,
                                 save_interval=save_interval)
    if anchoring_mode:
        register_anchored_ppo()
        agent['algorithm']['class_name'] = 'AnchoredPPO'
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC4JointEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 5.0
    cfg.seed = seed
    cfg.log_dir = str(output)
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(
        asset_path=str(root / 'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf'),
        usd_dir=str(output / 'generated_assets')))
    contract = dict(schema=C4_TRAINING_SCHEMA, task_weights=list(TASK_WEIGHTS), seed=seed,
                    num_envs=num_envs, steps=steps, minibatches=minibatches,
                    short_preview=list(cfg.pure_rl_spatial_preview_times_s),
                    c4_preview=list(cfg.pure_rl_c4_preview_times_s),
                    sampling=json.loads(json.dumps(asdict(cfg.pure_rl_continuous_curve_sampling))))
    if retention_variant is not None:
        contract['retention'] = dict(schema='c4_retention_pilot_v1', variant=retention_variant,
                                     curriculum=asdict(C4CurriculumConfig()) if curriculum else None,
                                     memory_sha256=hashlib.sha256(memory_path.read_bytes()).hexdigest() if memory_path else None,
                                     memory_coefficient=memory_coefficient if memory else 0.0,
                                     diagnostic_interval=5)
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = (loaded.get('infos') or {}).get('c4_training')
    anchor_payload = None
    if anchoring_mode:
        if not state:
            raise ValueError('Missing optimizer/schedule/curriculum resume state')
        previous_anchor = state.get('anchoring')
        if iterations + (previous_anchor['updates'] if previous_anchor else 0) > 25:
            raise ValueError('This experiment is capped at 25 additional PPO iterations')
        if previous_anchor:
            if previous_anchor['mode'] != anchoring_mode:
                raise ValueError('Cannot switch anchoring mode on resume')
            anchor_payload = previous_anchor['payload']
            contract['anchoring'] = state['contract']['anchoring']
        else:
            if anchors_path is None:
                raise ValueError('Initial D-50 continuation needs fixed anchors')
            anchor_payload = torch.load(anchors_path, map_location='cpu', weights_only=True)
            if anchor_payload['source_checkpoint'] != str(checkpoint):
                raise ValueError('Anchor teacher must be this D-50 source')
            # Validate the entire historical D contract before adding the sole intervention.
            if state['contract'] != contract:
                raise ValueError('D-50 source training contract mismatch')
            contract['anchoring'] = dict(mode=anchoring_mode, schema=anchor_payload['anchor_schema'],
                                         epsilon=anchor_payload['epsilon'], coefficient=1.0,
                                         family_weights=[1 / 6] * 6, legacy_memory_loss_enabled=False)
        TailAnchors(anchor_payload)
    expected_contract = dict(contract)
    if anchoring_mode and not state.get('anchoring'):
        expected_contract.pop('anchoring')
    if resume and (not state or state.get('contract') != expected_contract):
        raise ValueError('Resume requires a C4 checkpoint with matching training contract')
    if not resume and state:
        raise ValueError('Use --resume for a C4 training checkpoint')

    class TrainingEnv(FlappingBotStraightFlightEnv):
        def __init__(self, config):
            self._continuous_curve_sampler = curriculum.sample if curriculum else None
            self.retention_tracker = None
            super().__init__(config)

        def _get_rewards(self):
            reward = super()._get_rewards()
            family = self._pure_rl_spatial_path.task_family_id
            log = self.extras.setdefault('log', {})
            if self.retention_tracker is not None:
                episodes = self.retention_tracker.observe(self, curriculum)
                if episodes:
                    with (output / 'retention_episodes.jsonl').open('a') as stream:
                        for episode in episodes:
                            stream.write(json.dumps(episode, allow_nan=False) + '\n')
            for key in ('candidate_count', 'accepted_count', 'rejected_count', 'exhausted_count'):
                log[f'C4Sampling/{key}'] = float(self._continuous_sampling_diagnostics.get(key, 0))
            for group, name in enumerate(TASK_NAMES):
                mask = family == group
                count = mask.sum()
                log[f'C4Train/{name}/env_count'] = count
                values = {'horizontal_error_m': self._eval_pure_rl_cross_track_error_m.abs(),
                          'vertical_error_m': self._eval_pure_rl_height_error_m.abs(),
                          'speed_mps': self._eval_pure_rl_along_track_velocity_mps,
                          'frequency_gap_hz': (self._requested_frequency_hz - self._applied_frequency_hz).abs(),
                          'tail_limit': self._eval_pure_rl_tail_limit_active.float(),
                          'completion_reached': self._eval_pure_rl_reached_all_events.float()}
                for key, value in values.items():
                    log[f'C4Train/{name}/{key}'] = value[mask].sum() / count.clamp_min(1)
                if retention_variant is not None:
                    for key, value in (self._debug_last_pure_rl_reward_terms or {}).items():
                        if value.shape == family.shape:
                            log[f'C4Reward/{name}/{key}'] = value[mask].sum() / count.clamp_min(1)
            done = self.reset_terminated | self.reset_time_outs
            if not hasattr(self, '_training_completed_once'):
                self._training_completed_once = torch.zeros_like(done)
            self._training_completed_once |= self._eval_pure_rl_reached_all_events
            if done.any():
                path = output / 'episodes.csv'
                exists = path.exists()
                with path.open('a', newline='') as stream:
                    writer = csv.writer(stream)
                    if not exists:
                        writer.writerow(('task', 'steps', 'survived', 'completion_reached', 'progress_m'))
                    for i in done.nonzero().flatten().tolist():
                        writer.writerow((TASK_NAMES[int(family[i])], int(self.episode_length_buf[i]),
                                         bool(self.reset_time_outs[i] and not self.reset_terminated[i]),
                                         bool(self._training_completed_once[i]),
                                         float(self._eval_pure_rl_along_track_progress_m[i])))
                self._training_completed_once[done] = False
            return reward

    register_pure_rl_split_actor_critic()
    env = TrainingEnv(cfg)
    try:
        assert env.device == 'cpu' and not cfg.scene.replicate_physics
        assert env._native_holonomic.get_active_joint_count() == num_envs
        if retention_variant is not None or collect_memory_steps:
            env.retention_tracker = TrainingEpisodeTracker(num_envs, env.max_episode_length, collect=bool(collect_memory_steps))
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, copy.deepcopy(agent), log_dir=str(output), device='cpu')
        runner.alg.policy.load_state_dict(loaded['model_state_dict'], strict=True)
        if collect_memory_steps:
            observations = wrapped.get_observations()
            runner.alg.policy.eval()
            with torch.inference_mode():
                for _ in range(collect_memory_steps):
                    actions = runner.alg.policy.act_inference(observations)
                    env.retention_tracker.add_inputs(observations['policy'], actions)
                    observations, _, _, _ = wrapped.step(actions)
            provenance = dict(checkpoint=str(checkpoint),
                              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              seed=seed, transitions=num_envs * collect_memory_steps, contract=contract,
                              note='Deterministic training rollouts; no injected recovery states or evaluation data')
            payload = env.retention_tracker.payload(provenance)
            BehaviorMemory(payload, seed=0)
            torch.save(payload, output / 'memory.pt')
            return dict(schema=payload['schema'], samples=len(payload['observations']),
                        samples_by_family=[int((payload['families'] == i).sum()) for i in range(6)],
                        accepted_episodes=payload['accepted_episodes'], rejected_episodes=payload['rejected_episodes'],
                        provenance=provenance, promotion_eligible=False)
        before = [p.detach().clone() for p in runner.alg.policy.actor.parameters()]
        assert maybe_enable_ppo_warm_start_guard(runner=runner)
        probe = None
        if retention_variant is not None:
            probe = C4UpdateDiagnostics(runner.alg, output_path=output / 'update_diagnostics.jsonl',
                                        interval=5, minimum_samples=16)
            runner.alg.update = probe.update
        if memory is not None and not anchoring_mode:
            attach_behavior_memory(runner.alg, memory, memory_coefficient)
        assert maybe_enable_task_aware_ppo(runner=runner)
        guard, adapter = runner.alg.ppo_warm_start_guard, runner.alg.task_aware_ppo
        if resume:
            runner.alg.optimizer.load_state_dict(loaded['optimizer_state_dict'])
            runner.current_learning_iteration = state['completed_iterations']
            guard.iteration, adapter.iteration = state['guard_iteration'], state['task_iteration']
            guard._set_learning_rate(state['learning_rate'])
            runner.alg.num_learning_epochs = state['num_learning_epochs']
            torch.set_rng_state(state['torch_rng_state'])
            if curriculum is not None:
                curriculum.load_state_dict(state['curriculum'])
            if memory is not None:
                memory.load_state_dict(state['memory'])
            if probe is not None:
                probe.iteration = state['probe_iteration']
        else:
            runner.current_learning_iteration = 0
            assert len(runner.alg.optimizer.state) == 0
        start = runner.current_learning_iteration
        if anchoring_mode:
            runner.alg.configure_anchors(anchor_payload, anchoring_mode)
            if state.get('anchoring'):
                runner.alg.anchor_updates = state['anchoring']['updates']
            elif any(not torch.equal(value, runner.alg.policy.actor.state_dict()[key])
                     for key, value in anchor_payload['reference_actor'].items()):
                raise ValueError('Initial actor is not the frozen anchor reference')
            # Equality checks are against the actual in-memory initialized objects.
            assert all(torch.equal(v, runner.alg.policy.state_dict()[k]) for k, v in loaded['model_state_dict'].items())
            from flapping_bot.analysis.behavioral_anchoring_experiment import assert_state_equal
            assert_state_equal(runner.alg.optimizer.state_dict(), loaded['optimizer_state_dict'])
            assert_state_equal(curriculum.state_dict(), state['curriculum'])
            (output / 'initialization.json').write_text(json.dumps(dict(
                actor_critic_std_equal=True, optimizer_equal=True, curriculum_equal=True,
                source=str(checkpoint), source_completed_iterations=start,
                learning_rate=runner.alg.learning_rate, schedule=runner.alg.schedule,
                legacy_memory_loss_enabled=runner.alg.symmetry is not None,
                initial_anchor_metrics=runner.alg.anchors.metrics(runner.alg.policy.actor)), indent=2))
        if curriculum is not None:
            original_update = runner.alg.update
            def update_curriculum():
                result = original_update()
                report = curriculum.update()
                with (output / 'curriculum.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(iteration=adapter.iteration, **report)) + '\n')
                return result
            runner.alg.update = update_curriculum
        original_save = runner.save
        def save_with_state(self, path, infos=None):
            training_state = dict(contract=contract, completed_iterations=adapter.iteration,
                                  guard_iteration=guard.iteration, task_iteration=adapter.iteration,
                                  learning_rate=runner.alg.learning_rate,
                                  num_learning_epochs=runner.alg.num_learning_epochs,
                                  torch_rng_state=torch.get_rng_state())
            if retention_variant is not None:
                training_state.update(curriculum=curriculum.state_dict() if curriculum else None,
                                      memory=memory.state_dict() if memory else None,
                                      probe_iteration=probe.iteration)
            if anchoring_mode:
                training_state['anchoring'] = dict(mode=anchoring_mode, payload=anchor_payload,
                                                   updates=runner.alg.anchor_updates)
            original_save(path, infos={**(infos or {}), 'c4_training': training_state})
        runner.save = types.MethodType(save_with_state, runner)
        if anchoring_mode:
            previous_update = runner.alg.update
            def report_anchor_update():
                result = previous_update()
                with (output / 'anchoring.jsonl').open('a') as stream:
                    stream.write(json.dumps(runner.alg.anchor_report, allow_nan=False) + '\n')
                if runner.alg.anchor_updates in (12, 25):
                    runner.save(str(output / f'update_{runner.alg.anchor_updates}.pt'))
                return result
            runner.alg.update = report_anchor_update
        dump_yaml(str(output / 'params/env.yaml'), cfg)
        (output / 'params/agent.yaml').write_text(yaml.safe_dump(agent))
        (output / 'curriculum_source.json').write_text(json.dumps({
            'heading_canonical_observation': 'true', 'policy_class_name': 'PureRLSplitActorCritic',
            'source_checkpoint_path': str(checkpoint), 'target_stage': 'c4joint', 'training_contract': contract}, indent=2))
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        delta = sum((a-b).square().sum() for a,b in zip(runner.alg.policy.actor.parameters(), before)).sqrt()
        if not all(torch.isfinite(p).all() for p in runner.alg.policy.parameters()):
            raise RuntimeError('Nonfinite policy after training')
        return dict(schema=C4_TRAINING_SCHEMA, start_iteration=start, completed_iterations=adapter.iteration,
                    guard_iteration=guard.iteration, actor_parameter_delta_norm=float(delta),
                    optimizer_state_entries=len(runner.alg.optimizer.state), resume=resume,
                    memory_auxiliary_calls=memory.calls if memory and not anchoring_mode else 0,
                    curriculum_updates=curriculum.updates if curriculum else 0,
                    promotion_eligible=False, contract=contract)
    finally:
        import omni.physx
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()
