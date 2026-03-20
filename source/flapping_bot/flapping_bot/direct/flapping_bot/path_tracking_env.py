"""Generic path-tracking environment."""

from __future__ import annotations

import math

import torch

from ...path_tracking import Mission, MissionGeneratorCfg, PathManager, PathManagerCfg, sample_mission
from ...px4_like.path_tracking_controller import PX4LikePathTrackingController, PX4LikePathTrackingControllerCfg

PATH_TRACKING_RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = None


def _wrap_pi(angle_rad: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


def _compute_alignment_error(*, tangent_xy: torch.Tensor, ground_vel_xy: torch.Tensor) -> torch.Tensor:
    """Return absolute track-alignment error in radians."""
    tangent_heading = torch.atan2(tangent_xy[:, 1], tangent_xy[:, 0])
    ground_speed = torch.linalg.norm(ground_vel_xy, dim=1)
    ground_heading = torch.atan2(ground_vel_xy[:, 1], ground_vel_xy[:, 0])
    align_error = torch.abs(_wrap_pi(ground_heading - tangent_heading))
    return torch.where(ground_speed > 1.0e-3, align_error, torch.zeros_like(align_error))


try:
    from isaaclab.utils import configclass
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

    from .straight_flight_env import FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg, FlappingBotStraightFlightEnv
except ModuleNotFoundError as exc:
    PATH_TRACKING_RUNTIME_IMPORT_ERROR = exc
    PATH_TRACKING_RUNTIME_AVAILABLE = False

    class FlappingBotPathTrackingEnvCfg:
        """Headless fallback config placeholder for path-tracking tests."""


    class FlappingBotPathTrackingEnv:
        """Headless fallback env placeholder for path-tracking tests."""

else:
    PATH_TRACKING_RUNTIME_AVAILABLE = True

    @configclass
    class FlappingBotPathTrackingEnvCfg(FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg):
        """Configuration for the generic path-tracking environment."""

        observation_space: int = 87
        episode_length_s: float = 18.0

        teacher_guidance_enabled: bool = True
        teacher_guidance_delta_init: float = 0.15
        teacher_guidance_delta_final: float = 2.0
        teacher_guidance_anneal_steps: int = 160_000
        teacher_guidance_schedule_steps: tuple[int, ...] = (0, 20_000, 80_000, 160_000)
        teacher_guidance_schedule_deltas: tuple[float, ...] = (0.15, 0.25, 0.75, 2.0)
        teacher_guidance_disable_after_steps: int = -1

        mission_seed: int = 0
        mission_num_segments_min: int = 2
        mission_num_segments_max: int = 4
        mission_allow_straight: bool = True
        mission_allow_turn: bool = True
        mission_allow_loiter: bool = True
        mission_allow_climb_on_straight: bool = True

        path_manager_max_roll_deg: float = 35.0
        path_manager_max_flight_path_angle_deg: float = 10.0

        terminate_path_error_m: float = 15.0
        terminate_height_error_m: float = 5.0


    class FlappingBotPathTrackingEnv(FlappingBotStraightFlightEnv):
        """Path-tracking RL environment backed by canonical mission primitives."""

        def __init__(self, cfg: FlappingBotPathTrackingEnvCfg, render_mode: str | None = None, **kwargs):
            self._path_reset_counter: int = 0
            self._missions: list[Mission | None] = []
            self._path_managers: list[PathManager | None] = []

            self._path_closest_point_xyz: torch.Tensor | None = None
            self._path_tangent_xy: torch.Tensor | None = None
            self._path_curvature_m_inv: torch.Tensor | None = None
            self._path_progress_s: torch.Tensor | None = None
            self._path_progress_prev_s: torch.Tensor | None = None
            self._path_delta_s: torch.Tensor | None = None
            self._path_height_sp_m: torch.Tensor | None = None
            self._path_lateral_error_m: torch.Tensor | None = None
            self._path_height_error_m: torch.Tensor | None = None
            self._path_align_error_rad: torch.Tensor | None = None
            self._path_preview_points_xyz: torch.Tensor | None = None
            self._path_preview_points_body_xyz: torch.Tensor | None = None
            self._path_action_delta: torch.Tensor | None = None
            self._path_query_dirty: bool = True

            super().__init__(cfg, render_mode, **kwargs)

            self._ensure_path_buffers()

            if bool(self.cfg.teacher_guidance_enabled):
                self._teacher_controller = PX4LikePathTrackingController(
                    PX4LikePathTrackingControllerCfg(
                        control_dt_s=float(self.step_dt),
                        height_sp_m=float(self.cfg.height_cmd),
                        pitch_trim_deg=float(self.cfg.pitch_cmd_deg),
                        freq_trim_hz=float(self.cfg.reset_flap_hz),
                        min_flap_hz=float(self.cfg.min_flap_hz),
                        max_flap_hz=float(self.cfg.max_flap_hz),
                        enable_tecs=True,
                        speed_sp_mps=float(self.cfg.vx_cmd),
                    ),
                    device=self.device,
                )

        def _ensure_path_buffers(self) -> None:
            if self._path_closest_point_xyz is not None:
                return

            num_envs = int(self.num_envs)
            device = self.device
            self._path_closest_point_xyz = torch.zeros((num_envs, 3), device=device)
            self._path_tangent_xy = torch.zeros((num_envs, 2), device=device)
            self._path_curvature_m_inv = torch.zeros((num_envs,), device=device)
            self._path_progress_s = torch.zeros((num_envs,), device=device)
            self._path_progress_prev_s = torch.zeros((num_envs,), device=device)
            self._path_delta_s = torch.zeros((num_envs,), device=device)
            self._path_height_sp_m = torch.full((num_envs,), float(self.cfg.height_cmd), device=device)
            self._path_lateral_error_m = torch.zeros((num_envs,), device=device)
            self._path_height_error_m = torch.zeros((num_envs,), device=device)
            self._path_align_error_rad = torch.zeros((num_envs,), device=device)
            self._path_preview_points_xyz = torch.zeros((num_envs, 5, 3), device=device)
            self._path_preview_points_body_xyz = torch.zeros((num_envs, 5, 3), device=device)
            self._path_action_delta = torch.zeros((num_envs, 4), device=device)

            if len(self._missions) != num_envs:
                self._missions = [None] * num_envs
            if len(self._path_managers) != num_envs:
                self._path_managers = [None] * num_envs

        def _normalize_env_ids(self, env_ids: torch.Tensor | list[int] | None) -> torch.Tensor:
            if env_ids is None:
                return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if isinstance(env_ids, list):
                return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            return env_ids.to(device=self.device, dtype=torch.long)

        def _sample_mission_for_env(self, env_id: int) -> Mission:
            seed = int(self.cfg.mission_seed) + int(self._path_reset_counter) * 7919 + int(env_id)
            self._path_reset_counter += 1
            return sample_mission(
                MissionGeneratorCfg(
                    seed=seed,
                    num_segments_min=int(self.cfg.mission_num_segments_min),
                    num_segments_max=int(self.cfg.mission_num_segments_max),
                    allow_straight=bool(self.cfg.mission_allow_straight),
                    allow_turn=bool(self.cfg.mission_allow_turn),
                    allow_loiter=bool(self.cfg.mission_allow_loiter),
                    allow_climb_on_straight=bool(self.cfg.mission_allow_climb_on_straight),
                )
            )

        def _make_path_manager(self, mission: Mission, env_id: int) -> PathManager:
            return PathManager(
                PathManagerCfg(
                    max_roll_deg=float(self.cfg.path_manager_max_roll_deg),
                    max_flight_path_angle_deg=float(self.cfg.path_manager_max_flight_path_angle_deg),
                    initial_altitude_m=float(self._height_cmd[env_id].item()),
                ),
                mission,
            )

        def _pre_physics_step(self, actions: torch.Tensor):
            prev_act_cmd = None if self._act_cmd is None else self._act_cmd.clone()
            super()._pre_physics_step(actions)
            self._ensure_path_buffers()
            assert self._path_action_delta is not None
            if prev_act_cmd is None:
                self._path_action_delta.zero_()
            else:
                self._path_action_delta.copy_(self._act_cmd - prev_act_cmd)
            self._path_query_dirty = True

        def _reset_idx(self, env_ids: torch.Tensor | list[int]):
            env_ids = self._normalize_env_ids(env_ids)
            super()._reset_idx(env_ids)
            self._ensure_path_buffers()

            assert self._path_progress_prev_s is not None
            assert self._path_delta_s is not None
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._path_align_error_rad is not None
            assert self._path_preview_points_xyz is not None
            assert self._path_preview_points_body_xyz is not None
            assert self._path_curvature_m_inv is not None
            assert self._path_progress_s is not None
            assert self._path_height_sp_m is not None
            assert self._path_closest_point_xyz is not None
            assert self._path_tangent_xy is not None
            assert self._path_action_delta is not None

            for env_id in env_ids.tolist():
                mission = self._sample_mission_for_env(env_id)
                self._missions[env_id] = mission
                self._path_managers[env_id] = self._make_path_manager(mission, env_id)

            self._path_progress_prev_s[env_ids] = 0.0
            self._path_delta_s[env_ids] = 0.0
            self._path_lateral_error_m[env_ids] = 0.0
            self._path_height_error_m[env_ids] = 0.0
            self._path_align_error_rad[env_ids] = 0.0
            self._path_curvature_m_inv[env_ids] = 0.0
            self._path_progress_s[env_ids] = 0.0
            self._path_height_sp_m[env_ids] = self._height_cmd[env_ids]
            self._path_closest_point_xyz[env_ids] = 0.0
            self._path_tangent_xy[env_ids, 0] = 1.0
            self._path_tangent_xy[env_ids, 1] = 0.0
            self._path_preview_points_xyz[env_ids] = 0.0
            self._path_preview_points_body_xyz[env_ids] = 0.0
            self._path_action_delta[env_ids] = 0.0
            self._path_query_dirty = True

        def _refresh_path_state(self) -> None:
            if not self._path_query_dirty:
                return

            self._ensure_path_buffers()
            assert self._path_closest_point_xyz is not None
            assert self._path_tangent_xy is not None
            assert self._path_curvature_m_inv is not None
            assert self._path_progress_s is not None
            assert self._path_progress_prev_s is not None
            assert self._path_delta_s is not None
            assert self._path_height_sp_m is not None
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._path_align_error_rad is not None
            assert self._path_preview_points_xyz is not None
            assert self._path_preview_points_body_xyz is not None

            pos_local = self._robot.data.root_pos_w - self.scene.env_origins
            ground_vel_local = self._robot.data.root_lin_vel_w

            for env_id in range(self.num_envs):
                manager = self._path_managers[env_id]
                if manager is None:
                    mission = self._sample_mission_for_env(env_id)
                    self._missions[env_id] = mission
                    manager = self._make_path_manager(mission, env_id)
                    self._path_managers[env_id] = manager

                query = manager.query(
                    position_xy=(float(pos_local[env_id, 0].item()), float(pos_local[env_id, 1].item())),
                    altitude_m=float(pos_local[env_id, 2].item()),
                    speed_mps=float(torch.linalg.norm(ground_vel_local[env_id, 0:2]).item()),
                )

                closest_point_xyz = getattr(query, "closest_point_xyz", None)
                if closest_point_xyz is None:
                    height_sp_m = float(getattr(query, "height_sp_m", pos_local[env_id, 2].item()))
                    preview_points_xyz = getattr(query, "preview_points_xyz")
                    first_preview = preview_points_xyz[0]
                    closest_point_xyz = (float(first_preview[0]), float(first_preview[1]), height_sp_m)
                height_sp_m = float(getattr(query, "height_sp_m", closest_point_xyz[2]))
                tangent_xy = getattr(query, "tangent_xy", None)
                if tangent_xy is None:
                    preview_points_xyz = getattr(query, "preview_points_xyz")
                    if len(preview_points_xyz) >= 2:
                        dx = float(preview_points_xyz[1][0]) - float(preview_points_xyz[0][0])
                        dy = float(preview_points_xyz[1][1]) - float(preview_points_xyz[0][1])
                        tangent_xy = (dx, dy)
                    else:
                        tangent_xy = (1.0, 0.0)

                tangent = torch.tensor(tangent_xy, device=self.device, dtype=pos_local.dtype)
                tangent_norm = torch.linalg.norm(tangent).clamp_min(1.0e-6)
                tangent = tangent / tangent_norm
                closest_xy = torch.tensor(closest_point_xyz[:2], device=self.device, dtype=pos_local.dtype)
                preview_points = torch.tensor(query.preview_points_xyz, device=self.device, dtype=pos_local.dtype)

                signed_lateral_error = getattr(query, "signed_lateral_error_m", None)
                if signed_lateral_error is None:
                    offset_xy = pos_local[env_id, 0:2] - closest_xy
                    signed_lateral_error = float(tangent[0] * offset_xy[1] - tangent[1] * offset_xy[0])

                self._path_closest_point_xyz[env_id] = torch.tensor(closest_point_xyz, device=self.device, dtype=pos_local.dtype)
                self._path_tangent_xy[env_id] = tangent
                self._path_curvature_m_inv[env_id] = float(query.curvature_m_inv)
                self._path_progress_s[env_id] = float(query.progress_s)
                self._path_height_sp_m[env_id] = height_sp_m
                self._path_lateral_error_m[env_id] = float(signed_lateral_error)
                self._path_preview_points_xyz[env_id] = preview_points

            self._path_delta_s.copy_(torch.clamp(self._path_progress_s - self._path_progress_prev_s, min=0.0))
            self._path_progress_prev_s.copy_(self._path_progress_s)
            self._path_height_error_m.copy_(pos_local[:, 2] - self._path_height_sp_m)
            self._path_align_error_rad.copy_(
                _compute_alignment_error(tangent_xy=self._path_tangent_xy, ground_vel_xy=ground_vel_local[:, 0:2])
            )

            rel_preview_xyz = self._path_preview_points_xyz - pos_local.unsqueeze(1)
            quat_batch = self._robot.data.root_quat_w.repeat_interleave(self._path_preview_points_xyz.shape[1], dim=0)
            self._path_preview_points_body_xyz.copy_(
                quat_apply_inverse(quat_batch, rel_preview_xyz.reshape(-1, 3)).reshape(self.num_envs, -1, 3)
            )
            self._path_query_dirty = False

        def _compute_teacher_actions(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if self._teacher_controller is None:
                raise RuntimeError("Teacher controller is not initialized.")

            self._refresh_path_state()
            assert self._path_closest_point_xyz is not None
            assert self._path_tangent_xy is not None
            assert self._path_curvature_m_inv is not None
            assert self._path_progress_s is not None
            assert self._path_height_sp_m is not None
            assert self._path_preview_points_xyz is not None

            pos_local = self._robot.data.root_pos_w - self.scene.env_origins
            ground_vel_local = self._robot.data.root_lin_vel_w
            roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
            ang_vel_body = self._robot.data.root_ang_vel_b
            if bool(self.cfg.teacher_guidance_use_wind_truth):
                wind_xy = self._wind_w[:, 0:2]
            else:
                wind_xy = torch.zeros((self.num_envs, 2), device=self.device)

            return self._teacher_controller.compute_actions_from_query(
                path_query={
                    "closest_point_xyz": self._path_closest_point_xyz,
                    "tangent_xy": self._path_tangent_xy,
                    "curvature_m_inv": self._path_curvature_m_inv,
                    "progress_s": self._path_progress_s,
                    "height_sp_m": self._path_height_sp_m,
                    "preview_points_xyz": self._path_preview_points_xyz,
                },
                pos_local=pos_local,
                ground_vel_local=ground_vel_local,
                wind_vel_local=wind_xy,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                ang_vel_body=ang_vel_body,
            )

        def _get_observations(self) -> dict[str, torch.Tensor]:
            self._refresh_path_state()
            assert self._path_height_sp_m is not None
            assert self._path_preview_points_body_xyz is not None
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._path_curvature_m_inv is not None
            assert self._path_progress_s is not None

            lin_vel_b = self._robot.data.root_lin_vel_b
            ang_vel_b = self._robot.data.root_ang_vel_b
            g_b = self._robot.data.projected_gravity_b
            jpos = self._robot.data.joint_pos[:, self._joint_ids]

            z_s = self._path_height_error_m.unsqueeze(1) / 20.0
            lin_s = lin_vel_b[:, 0:3] / 10.0
            ang_s = ang_vel_b[:, 0:3] / 10.0
            g_s = g_b[:, 0:3]

            l_idx, r_idx = self._IDX_LEFT_TAIL, self._IDX_RIGHT_TAIL
            l_den = torch.maximum(self._joint_upper_limits[l_idx].abs(), self._joint_lower_limits[l_idx].abs())
            r_den = torch.maximum(self._joint_upper_limits[r_idx].abs(), self._joint_lower_limits[r_idx].abs())
            tail_s = torch.stack([jpos[:, l_idx] / l_den, jpos[:, r_idx] / r_den], dim=1)

            freq_s = self._freq.unsqueeze(1) / float(self.cfg.max_flap_hz)
            forward_speed_s = lin_vel_b[:, 0].unsqueeze(1) / 10.0

            def _roll_and_set(buf: torch.Tensor, new: torch.Tensor):
                buf.copy_(torch.roll(buf, shifts=-1, dims=1))
                buf[:, -1, :] = new

            if (~self._hist_valid).any():
                ids = (~self._hist_valid).nonzero(as_tuple=False).squeeze(-1)
                if ids.numel() > 0:
                    self._hist_z[ids] = z_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_z, 1)
                    self._hist_lin[ids] = lin_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_lin, 1)
                    self._hist_ang[ids] = ang_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_ang, 1)
                    self._hist_gb[ids] = g_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_gb, 1)
                    self._hist_tail[ids] = tail_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_tail, 1)
                    self._hist_freq[ids] = freq_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_freq, 1)
                    self._hist_vx[ids] = forward_speed_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_vx, 1)
                    self._hist_valid[ids] = True

            _roll_and_set(self._hist_z, z_s)
            _roll_and_set(self._hist_lin, lin_s)
            _roll_and_set(self._hist_ang, ang_s)
            _roll_and_set(self._hist_gb, g_s)
            _roll_and_set(self._hist_tail, tail_s)
            _roll_and_set(self._hist_freq, freq_s)
            _roll_and_set(self._hist_vx, forward_speed_s)

            base_obs = torch.cat(
                [
                    self._hist_z.reshape(self.num_envs, -1),
                    self._hist_lin.reshape(self.num_envs, -1),
                    self._hist_ang.reshape(self.num_envs, -1),
                    self._hist_gb.reshape(self.num_envs, -1),
                    self._hist_tail.reshape(self.num_envs, -1),
                    self._hist_freq.reshape(self.num_envs, -1),
                    self._hist_vx.reshape(self.num_envs, -1),
                ],
                dim=1,
            )
            preview_obs = _build_preview_observation(
                {
                    "preview_points_body_xyz": self._path_preview_points_body_xyz,
                    "lateral_error_m": self._path_lateral_error_m,
                    "height_error_m": self._path_height_error_m,
                    "curvature_m_inv": self._path_curvature_m_inv,
                    "progress_s": self._path_progress_s,
                }
            )
            return {"policy": torch.cat((base_obs, preview_obs), dim=1)}

        def _get_rewards(self) -> torch.Tensor:
            self._refresh_path_state()
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._path_align_error_rad is not None
            assert self._path_delta_s is not None
            assert self._path_action_delta is not None

            airspeed = torch.linalg.norm(self._robot.data.root_lin_vel_w - self._wind_w, dim=1)
            g_b = self._robot.data.projected_gravity_b
            tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
            ang_rate = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
            return _compute_tracking_reward(
                lateral_error=torch.abs(self._path_lateral_error_m),
                height_error=torch.abs(self._path_height_error_m),
                align_error=self._path_align_error_rad,
                delta_s=self._path_delta_s,
                airspeed=airspeed,
                action=self._act_cmd,
                action_delta=self._path_action_delta,
                tilt=tilt,
                ang_rate=ang_rate,
            )

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            self._refresh_path_state()
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None

            pos_w = self._robot.data.root_pos_w - self.scene.env_origins
            fell = pos_w[:, 2] <= float(self.cfg.terminate_ground_height)

            g_b = self._robot.data.projected_gravity_b
            tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
            tilt_thr = torch.sin(torch.deg2rad(torch.tensor(self.cfg.terminate_tilt_deg, device=self.device)))
            fell = fell | (tilt > tilt_thr)
            fell = fell | (torch.abs(self._path_lateral_error_m) > float(self.cfg.terminate_path_error_m))
            fell = fell | (torch.abs(self._path_height_error_m) > float(self.cfg.terminate_height_error_m))

            timed_out = self.episode_length_buf >= self.max_episode_length - 1
            return fell, timed_out


def _build_preview_observation(query: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build a flat preview-based observation vector."""
    preview = query["preview_points_body_xyz"].reshape(query["preview_points_body_xyz"].shape[0], -1)
    extras = torch.stack(
        [
            query["lateral_error_m"],
            query["height_error_m"],
            query["curvature_m_inv"],
            query["progress_s"],
        ],
        dim=-1,
    )
    return torch.cat((preview, extras), dim=-1)


def _compute_tracking_reward(
    *,
    lateral_error: torch.Tensor,
    height_error: torch.Tensor,
    align_error: torch.Tensor,
    delta_s: torch.Tensor,
    airspeed: torch.Tensor,
    action: torch.Tensor,
    action_delta: torch.Tensor,
    tilt: torch.Tensor,
    ang_rate: torch.Tensor,
) -> torch.Tensor:
    """Compute a tracking-plus-progress reward."""
    low_speed_penalty = 0.2 * torch.clamp(4.0 - airspeed, min=0.0)
    action_penalty = 0.01 * torch.sum(action**2, dim=1)
    action_delta_penalty = 0.01 * torch.sum(action_delta**2, dim=1)
    tilt_penalty = 0.05 * tilt
    ang_rate_penalty = 0.02 * ang_rate
    return (
        delta_s
        - 0.2 * lateral_error
        - 0.2 * height_error
        - 0.1 * align_error
        - low_speed_penalty
        - action_penalty
        - action_delta_penalty
        - tilt_penalty
        - ang_rate_penalty
    )


def _teacher_recovery_mask(**kwargs) -> torch.Tensor:
    """Thin wrapper placeholder for recovery-teacher masking."""
    from ...px4_like.rl_training_utils import compute_recovery_teacher_mask

    return compute_recovery_teacher_mask(**kwargs)
