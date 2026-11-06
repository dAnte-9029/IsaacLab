"""Independent geometric trajectory task on the unchanged CPU-native plant."""
from dataclasses import replace
import json

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from flapping_bot.path_tracking.unified_trajectory import GeometryLimits, SamplingConfig, Trajectory, sample_trajectory
from flapping_bot.path_tracking.unified_trajectory_cache import TrajectoryCache
from .straight_flight_env import FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg


@configclass
class UnifiedTrajectoryEnvCfg(FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg):
    use_pure_rl_actor_observation: bool = False
    use_pure_rl_curriculum1_reward: bool = False
    pure_rl_reward_telemetry_enabled: bool = False
    observation_space: int = 39
    online_sampling: bool = False
    feedback_enabled: bool = False
    online_identity_metadata: object = None
    evaluation_tasks: object = None
    online_rules_path: str = ""
    online_feedback_directory: str = ""
    online_policy_version: str | None = None
    online_evaluation_mode: str = "training"
    online_sampler_config: object = None
    sampling: SamplingConfig = SamplingConfig()
    geometry_limits: GeometryLimits = GeometryLimits()
    preview_distances_m: tuple[float, ...] = (2., 5., 10.)
    height_cmd: float = 20.
    reset_forward_speed_mps: float = 8.  # initial condition only, never a tracking target
    episode_length_s: float = 120.  # resource timeout, independently configurable from route length
    tracking_scale_m: float = 2.
    progress_weight: float = 1.
    tracking_weight: float = 1.
    action_change_weight: float = .01
    completion_bonus: float = 10.
    failure_penalty: float = 5.
    max_path_error_m: float = 8.
    max_body_rate_rad_s: float = 20.
    completion_tolerance_m: float = .5
    projection_margin_m: float = .5
    reference_cache_spacing_m: float = .2


class UnifiedTrajectoryEnv(FlappingBotStraightFlightEnv):
    cfg: UnifiedTrajectoryEnvCfg

    def __init__(self, cfg: UnifiedTrajectoryEnvCfg, **kwargs):
        cfg.sampling.validate()
        if (cfg.sim.device != "cpu" or cfg.pure_rl_spatial_stage_id is not None
                or cfg.pure_rl_longitudinal_stage_id is not None or cfg.pure_rl_c4_joint_training_enabled
                or cfg.use_pure_rl_actor_observation or cfg.use_pure_rl_curriculum1_reward):
            raise ValueError("Unified task requires CPU-native plant and independent task logic")
        if (not cfg.preview_distances_m or any(not np.isfinite(x) or x <= 0 for x in cfg.preview_distances_m)
                or any(b <= a for a, b in zip(cfg.preview_distances_m, cfg.preview_distances_m[1:]))):
            raise ValueError("Preview distances must be finite, positive and increasing")
        cfg.observation_space = 30 + 3*len(cfg.preview_distances_m)
        super().__init__(cfg, **kwargs)
        self.path_cache = TrajectoryCache(self.num_envs, cfg.sampling.length_m,
                                         cfg.reference_cache_spacing_m, self.device)
        self.preview_distances = torch.tensor(cfg.preview_distances_m, device=self.device)
        self.paths = [None]*self.num_envs
        self.task_records = [None]*self.num_envs
        self.task_sampling_sources = [None]*self.num_envs
        self.reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.progress = torch.zeros(self.num_envs, device=self.device)
        self.high_water = self.progress.clone()
        self.previous_high_water = self.progress.clone()
        self.previous_action = torch.zeros_like(self._actions)
        self.previous_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.reference = {}
        self._reference_valid = False
        self._current_policy_version = cfg.online_policy_version
        self._evaluation_mode = cfg.online_evaluation_mode
        self._action_versions = [None]*self.num_envs
        self._action_modes = ["training"]*self.num_envs
        self._terminal_records = []
        self._task_events = []
        self._evaluation_finished = np.zeros(self.num_envs, dtype=bool)
        self.online_sampler = None
        self.rollout_evaluator = None
        if cfg.online_sampling or cfg.feedback_enabled:
            from flapping_bot.path_tracking.online_trajectory import (
                OfflineEvidence, FeedbackStore, RolloutEvaluator, OnlineSampler, OnlineConfig)
            if not cfg.online_rules_path or not cfg.online_feedback_directory:
                raise ValueError("Online mode requires evidence path and journal directory")
            # Runtime paths, task source and env count cannot change physical identity.
            snapshot = cfg.to_dict()
            for key in list(snapshot):
                if key.startswith('online_') or key in ('feedback_enabled','evaluation_tasks','seed','log_dir'):
                    del snapshot[key]
            snapshot['scene'].pop('num_envs', None)
            snapshot['robot']['spawn'].pop('usd_dir', None)
            snapshot = json.loads(json.dumps(snapshot, default=lambda obj:
                f'{obj.__module__}.{obj.__qualname__}' if callable(obj) else str(obj)))
            backend_cfg = getattr(cfg, 'control_aero_3d', None)
            plant = dict(backend=getattr(backend_cfg, 'backend', 'DeLaurier'), configuration=snapshot,
                         physics_dt=float(self.physics_dt), control_dt=float(self.step_dt),
                         identity=cfg.online_identity_metadata or {'status':'explicit identity unavailable'})
            contract = dict(kind='unified_terminal_v1', reference='base_link world z-up',
                projection='cache local movement + margin', projection_margin_m=cfg.projection_margin_m,
                cache_spacing_m=cfg.reference_cache_spacing_m,
                completion_tolerance_m=cfg.completion_tolerance_m, max_path_error_m=cfg.max_path_error_m,
                min_height_m=cfg.terminate_ground_height, max_height_m=cfg.geometry_limits.max_height,
                min_tilt_cos=float(np.cos(np.deg2rad(cfg.terminate_tilt_deg))),
                max_body_rate_rad_s=cfg.max_body_rate_rad_s, max_steps=int(self.max_episode_length-1),
                evaluation_dt_s=float(self.step_dt), failure_precedes_completion=True,
                completion_precedes_timeout=True)
            store = FeedbackStore(cfg.online_feedback_directory)
            self.rollout_evaluator = RolloutEvaluator(store)
            self.plant_identity, self.evaluation_contract = plant, contract
            self.online_sampler = OnlineSampler(OfflineEvidence(cfg.online_rules_path), store, plant,
                contract, cfg.online_policy_version, cfg.online_sampler_config or OnlineConfig(),
                cfg.sampling, cfg.geometry_limits) if cfg.online_sampling else None

    def accept_terminal(self, record):
        self._terminal_records.append(record)
        if self.online_sampler is not None: self.online_sampler.feedback(record)

    def drain_terminals(self):
        records, self._terminal_records = self._terminal_records, []
        return records

    def drain_tasks(self):
        records, self._task_events = self._task_events, []
        return records

    def set_policy_version(self, version: str | None, *, evaluation_mode: str = 'training'):
        """Runner calls before actions from a new PPO checkpoint; no forced reset."""
        if evaluation_mode not in ('training', 'frozen'):
            raise ValueError('Expected training or explicitly frozen evaluation')
        self._current_policy_version = version
        self._evaluation_mode = evaluation_mode
        if self.online_sampler is not None:
            self.online_sampler.set_policy_version(version)

    def _reset_idx(self, env_ids):
        if self.rollout_evaluator is not None:
            for i in torch.as_tensor(env_ids).tolist():
                record = self.rollout_evaluator.abort(i)
                if record is not None: self.accept_terminal(record)
        # Parent owns only robot/actuator/PhysX reset. All legacy task stages are None.
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for i in ids.tolist():
            seed = int(np.random.SeedSequence([int(self.cfg.seed or 0), i, int(self.reset_counts[i])]).generate_state(1)[0])
            origin = self._robot.data.root_pos_w[i].detach().cpu().numpy()
            # Screen heights relative to env origin, then translate the complete route.
            local_origin = origin-self.scene.env_origins[i].cpu().numpy()
            heading = float(self._straight_line_heading_rad[i])
            source = 'base'
            if self.cfg.evaluation_tasks is not None:
                from flapping_bot.path_tracking.unified_trajectory import TrajectorySpec
                spec = TrajectorySpec(**self.cfg.evaluation_tasks[i]['trajectory'])
                path = Trajectory(replace(spec, origin=tuple(local_origin), heading_rad=heading))
                source = 'fixed_evaluation'
            elif self.online_sampler is None:
                path = sample_trajectory(seed, self.cfg.sampling, self.cfg.geometry_limits, local_origin, heading)
            else:
                spec, source = self.online_sampler.sample(seed, local_origin, heading)
                path = Trajectory(spec)
            path = Trajectory(replace(path.spec, origin=tuple(origin)))
            self.paths[i] = path
            self.task_records[i] = path.spec.to_dict()
            self.task_sampling_sources[i] = source
            if not self._evaluation_finished[i]:
                self._task_events.append(dict(env_id=i, trajectory=path.spec.to_dict(), source=source))
            if self.rollout_evaluator is not None and not self._evaluation_finished[i]:
                d = self._robot.data
                initial = dict(root_state=d.root_state_w[i].detach().cpu().tolist(),
                    joint_pos=d.joint_pos[i].detach().cpu().tolist(), joint_vel=d.joint_vel[i].detach().cpu().tolist(),
                    phase=float(self._phase[i]), frequency=float(self._freq[i]),
                    action=self._actions[i].detach().cpu().tolist(),
                    applied_action=self._act_cmd[i].detach().cpu().tolist(),
                    wind=self._wind_w[i].detach().cpu().tolist(),
                    env_origin=self.scene.env_origins[i].detach().cpu().tolist(),
                    scene=dict(num_envs=self.num_envs, env_index=i, env_spacing_m=self.cfg.scene.env_spacing))
                self.task_sampling_sources[i] = source
                self.rollout_evaluator.begin(i, path.spec, self.plant_identity, initial,
                    self.evaluation_contract, self._current_policy_version, source,
                    evaluation_mode=self._evaluation_mode)
            self.reset_counts[i] += 1
        self.path_cache.update_rows(ids, [self.paths[i] for i in ids.tolist()])
        self.progress[ids] = 0
        self.high_water[ids] = 0
        self.previous_high_water[ids] = 0
        self.previous_action[ids] = self._actions[ids]
        self.previous_position[ids] = self._robot.data.root_pos_w[ids]
        self._reference_valid = False

    def _pre_physics_step(self, actions):
        self.previous_high_water.copy_(self.high_water)
        self.previous_action.copy_(self._actions)
        self.previous_position.copy_(self._robot.data.root_pos_w)
        self._reference_valid = False
        # Capture provenance at the action boundary, not at reset or termination.
        self._action_versions = [self._current_policy_version]*self.num_envs
        self._action_modes = [self._evaluation_mode]*self.num_envs
        self._raw_actions_finite = torch.isfinite(actions).all(-1)
        super()._pre_physics_step(actions)

    def _update_reference(self):
        if self._reference_valid:
            return
        position = self._robot.data.root_pos_w
        movement = torch.linalg.vector_norm(position-self.previous_position, dim=-1)
        self.progress.copy_(self.path_cache.project(position, self.progress, movement+self.cfg.projection_margin_m))
        self.reference = {key: value.to(position.dtype) for key, value in self.path_cache.query(self.progress).items()}
        self.reference["preview"] = self.path_cache.positions(
            self.progress[:, None]+self.preview_distances[None, :]).to(position.dtype)
        self.reference["error"] = self.reference["position"]-self._robot.data.root_pos_w
        self.high_water.copy_(torch.maximum(self.high_water, self.progress))
        self._reference_valid = True

    def _get_observations(self):
        self._update_reference()
        q = self._robot.data.root_quat_w
        rotate = lambda x: quat_apply_inverse(q, x)
        preview = torch.cat([rotate(p-self._robot.data.root_pos_w)/self.cfg.preview_distances_m[j]
                             for j, p in enumerate(self.reference["preview"].unbind(1))], dim=-1)
        geometry = torch.stack([self.reference[key] for key in ("kappa", "gamma", "kappa_prime", "gamma_prime")], -1)
        tail = self._robot.data.joint_pos[:, [int(self._joint_ids[j]) for j in
                                            (self._IDX_RUDDER, self._IDX_LEFT_TAIL, self._IDX_RIGHT_TAIL)]]
        obs = torch.cat([self._robot.data.root_lin_vel_b/10, self._robot.data.root_ang_vel_b/10,
                         self._robot.data.projected_gravity_b, rotate(self.reference["error"])/self.cfg.tracking_scale_m,
                         rotate(self.reference["tangent"]), preview, geometry,
                         torch.stack([torch.sin(self._phase), torch.cos(self._phase)], -1),
                         self._actions, (self._freq/5).unsqueeze(-1), tail,
                         (1-self.progress/self.cfg.sampling.length_m).unsqueeze(-1)], -1)
        return {"policy": torch.nan_to_num(obs, nan=0., posinf=5., neginf=-5.).clamp(-5, 5)}

    def _get_dones(self):
        self._update_reference()
        distance = torch.linalg.vector_norm(self.reference["error"], dim=-1)
        finite = torch.isfinite(self._robot.data.root_state_w).all(-1)
        height = self._robot.data.root_pos_w[:, 2]-self.scene.env_origins[:, 2]
        tilt_cos = -self._robot.data.projected_gravity_b[:, 2]
        self.failed = (~finite | (height < self.cfg.terminate_ground_height)
                       | (height > self.cfg.geometry_limits.max_height)
                       | (distance > self.cfg.max_path_error_m)
                       | (tilt_cos < np.cos(np.deg2rad(self.cfg.terminate_tilt_deg)))
                       | (torch.linalg.vector_norm(self._robot.data.root_ang_vel_b, dim=-1) > self.cfg.max_body_rate_rad_s))
        self.completed = ((self.progress >= self.cfg.sampling.length_m-self.cfg.completion_tolerance_m)
                          & (distance <= self.cfg.completion_tolerance_m) & ~self.failed)
        timeout = self.episode_length_buf >= self.max_episode_length-1
        self.extras["unified"] = {"completed": self.completed.clone(), "failed": self.failed.clone(),
                                  "progress_m": self.progress.clone(), "path_error_m": distance.clone(),
                                  "timeout": timeout.clone()}
        if self.rollout_evaluator is not None:
            for i in range(self.num_envs):
                record = self.rollout_evaluator.observe(i, progress_m=float(self.progress[i]),
                    path_error_m=float(distance[i]), height_m=float(height[i]), tilt_cos=float(tilt_cos[i]),
                    body_rate_rad_s=float(torch.linalg.vector_norm(self._robot.data.root_ang_vel_b[i])),
                    finite=bool(finite[i] & self._raw_actions_finite[i]), step=int(self.episode_length_buf[i]),
                    policy_version=self._action_versions[i], execution_mode=self._action_modes[i],
                    control=dict(command=self._actions[i].detach().cpu().tolist(),
                        applied=self._act_cmd[i].detach().cpu().tolist(),
                        joint_pos=self._robot.data.joint_pos[i].detach().cpu().tolist(),
                        joint_vel=self._robot.data.joint_vel[i].detach().cpu().tolist(),
                        root_state=self._robot.data.root_state_w[i].detach().cpu().tolist(),
                        path_error_xyz=self.reference['error'][i].detach().cpu().tolist(),
                        phase=float(self._phase[i])))
                if record is not None:
                    self.accept_terminal(record)
                    if self.cfg.evaluation_tasks is not None: self._evaluation_finished[i] = True
            self.rollout_evaluator.store.flush()
        return self.failed | self.completed, timeout & ~self.completed & ~self.failed

    def _get_rewards(self):
        self._update_reference()
        error2 = torch.sum(self.reference["error"]**2, -1)
        tracking = torch.exp(-error2/self.cfg.tracking_scale_m**2)
        # Only new arc length earns credit: reversing and revisiting cannot farm progress.
        forward = (self.high_water-self.previous_high_water).clamp_min(0)*tracking
        action_change = torch.mean((self._actions-self.previous_action)**2, -1)
        reward = (self.cfg.progress_weight*forward + self.cfg.tracking_weight*(tracking-1)*self.step_dt
                  - self.cfg.action_change_weight*action_change
                  + self.cfg.completion_bonus*self.completed - self.cfg.failure_penalty*self.failed)
        self.extras["log"] = {"Unified/path_error_m": torch.sqrt(error2).mean(),
                              "Unified/new_progress_m": forward.mean(), "Unified/reward": reward.mean()}
        return torch.nan_to_num(reward, nan=-self.cfg.failure_penalty)
