"""Generic path-tracking environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from ...path_tracking import Mission, MissionGeneratorCfg, MissionSegment, PathManager, PathManagerCfg, sample_mission
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


def _compute_loiter_radial_error(
    *,
    position_xy: torch.Tensor,
    center_xy: torch.Tensor,
    radius_m: torch.Tensor,
) -> torch.Tensor:
    """Return signed radial error relative to a target loiter circle."""
    return torch.linalg.norm(position_xy - center_xy, dim=1) - radius_m


def _compute_loiter_angular_progress(
    *,
    start_angle_rad: torch.Tensor,
    current_angle_rad: torch.Tensor,
    turn_direction: torch.Tensor,
    total_turns: torch.Tensor,
    reference_progress_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return normalized loiter angular progress in ``[0, 1]``."""
    two_pi = current_angle_rad.new_tensor(2.0 * math.pi)
    signed_base_delta_rad = torch.remainder(
        (current_angle_rad - start_angle_rad) * turn_direction.to(dtype=current_angle_rad.dtype),
        two_pi,
    )
    total_sweep_rad = torch.clamp(total_turns.to(dtype=current_angle_rad.dtype), min=0.0) * two_pi

    if reference_progress_ratio is not None:
        resolved_delta_rad = signed_base_delta_rad.clone()
        reference_progress_ratio = torch.clamp(reference_progress_ratio.to(dtype=current_angle_rad.dtype), min=0.0, max=1.0)
        reference_delta_rad = reference_progress_ratio * total_sweep_rad
        max_extra_turns = max(int(torch.ceil(total_turns.detach().max()).item()), 1)
        for turn_idx in range(1, max_extra_turns + 1):
            candidate_delta_rad = signed_base_delta_rad + turn_idx * two_pi
            valid_mask = candidate_delta_rad <= total_sweep_rad + 1.0e-6
            closer_mask = torch.abs(candidate_delta_rad - reference_delta_rad) < torch.abs(
                resolved_delta_rad - reference_delta_rad
            )
            resolved_delta_rad = torch.where(valid_mask & closer_mask, candidate_delta_rad, resolved_delta_rad)
        signed_base_delta_rad = resolved_delta_rad

    return torch.clamp(signed_base_delta_rad / torch.clamp(total_sweep_rad, min=1.0e-6), min=0.0, max=1.0)


def _compute_loiter_milestone_mask(*, progress_ratio: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return whether the normalized loiter progress reached one milestone."""
    return progress_ratio >= float(threshold)


def _compute_loiter_milestone_bonus(
    *,
    previous_progress_ratio: torch.Tensor,
    progress_ratio: torch.Tensor,
    quarter_turn_bonus: float,
    half_turn_bonus: float,
    three_quarter_turn_bonus: float,
) -> torch.Tensor:
    """Return one-shot loiter milestone bonuses for newly crossed thresholds."""
    previous_progress_ratio = torch.clamp(previous_progress_ratio, min=0.0, max=1.0)
    progress_ratio = torch.clamp(progress_ratio, min=0.0, max=1.0)
    bonus = torch.zeros_like(progress_ratio)
    for threshold, reward in (
        (0.25, float(quarter_turn_bonus)),
        (0.50, float(half_turn_bonus)),
        (0.75, float(three_quarter_turn_bonus)),
    ):
        if reward <= 0.0:
            continue
        crossed = (previous_progress_ratio < threshold) & (progress_ratio >= threshold)
        bonus = bonus + crossed.to(dtype=progress_ratio.dtype) * reward
    return bonus


def _compute_loiter_tracking_quality(
    *,
    loiter_radial_error: torch.Tensor,
    align_error: torch.Tensor,
    loiter_tracking_radial_scale_m: float,
    loiter_tracking_align_scale_rad: float,
) -> torch.Tensor:
    """Return a normalized loiter tracking quality in ``(0, 1]``."""
    loiter_radial_term = loiter_radial_error / max(float(loiter_tracking_radial_scale_m), 1.0e-6)
    loiter_align_term = torch.abs(align_error) / max(float(loiter_tracking_align_scale_rad), 1.0e-6)
    return torch.exp(-(loiter_radial_term + loiter_align_term))


def _compute_mission_seed(*, base_seed: int, path_reset_counter: int, env_id: int, increment_per_reset: bool) -> int:
    """Compute the deterministic mission seed for one environment reset."""
    if increment_per_reset:
        return int(base_seed) + int(path_reset_counter) * 7919 + int(env_id)
    return int(base_seed) + int(env_id)


def _compute_curriculum_schedule_step(*, common_step_counter: int, num_envs: int) -> int:
    """Convert vector-env steps into the total sample-count schedule used by path curricula."""
    return max(int(common_step_counter), 0) * max(int(num_envs), 1)


@dataclass(frozen=True)
class _LoiterCurriculumStage:
    """Resolved loiter curriculum parameters for the current training step."""

    mode: str
    loiter_turns: float
    straight_rehearsal_prob: float


try:
    from isaaclab.utils import configclass
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

    from .straight_flight_env import FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg, FlappingBotStraightFlightEnv
except ModuleNotFoundError as exc:
    PATH_TRACKING_RUNTIME_IMPORT_ERROR = exc
    PATH_TRACKING_RUNTIME_AVAILABLE = False

    class FlappingBotPathTrackingEnvCfg:
        """Headless fallback config placeholder for path-tracking tests."""

        tail_horizontal_tail_incidence_bias_deg: float = 0.0
        tail_fixed_horizontal_effectiveness: float = 0.5
        tail_elevon_effectiveness: float = 1.2
        tail_elevon_alpha_limit_deg: float = 25.0
        tail_horizontal_tail_q_scale: float = 1.0
        base_body_com_override_x_m: float | None = -0.10
        reset_pitch_deg: float = 4.0
        reset_flap_hz: float = 3.4
        reset_elevon_pitch_deg: float = -18.0


    class FlappingBotPathTrackingWeakTeacherRLEnvCfg(FlappingBotPathTrackingEnvCfg):
        """Headless fallback weak-teacher config placeholder."""


    class FlappingBotPathTrackingPureRLEnvCfg(FlappingBotPathTrackingEnvCfg):
        """Headless fallback pure-RL config placeholder."""


    class FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg(FlappingBotPathTrackingWeakTeacherRLEnvCfg):
        """Headless fallback primitive weak-teacher config placeholder."""


    class FlappingBotPathTrackingPrimitivePureRLEnvCfg(FlappingBotPathTrackingPureRLEnvCfg):
        """Headless fallback primitive pure-RL config placeholder."""


    class FlappingBotPathTrackingEnv:
        """Headless fallback env placeholder for path-tracking tests."""

else:
    PATH_TRACKING_RUNTIME_AVAILABLE = True

    @configclass
    class FlappingBotPathTrackingEnvCfg(FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg):
        """Configuration for the generic path-tracking environment."""

        observation_space: int = 96
        episode_length_s: float = 18.0
        completion_ratio: float = 0.98
        completion_bonus: float = 5.0

        teacher_guidance_enabled: bool = True
        teacher_guidance_delta_init: float = 0.15
        teacher_guidance_delta_final: float = 2.0
        teacher_guidance_anneal_steps: int = 160_000
        teacher_guidance_schedule_steps: tuple[int, ...] = (0, 20_000, 80_000, 160_000)
        teacher_guidance_schedule_deltas: tuple[float, ...] = (0.15, 0.25, 0.75, 2.0)
        teacher_guidance_disable_after_steps: int = -1
        curve_teacher_delta_scale: float = 1.0
        curve_teacher_curvature_ref_m_inv: float = 0.05
        loiter_teacher_delta_scale: float = 1.0
        teacher_use_tecs_load_factor_compensation: bool = True
        # Keep this in the same order of magnitude as PX4's FW_T_RLL2THR.
        # Smaller single-digit gains were effectively neutral in this TECS scaling.
        teacher_tecs_roll_throttle_compensation: float = 30.0
        teacher_tecs_load_factor_clamp_max: float = 2.0
        teacher_tecs_load_factor_use_roll_sp: bool = True
        teacher_tecs_load_factor_pitch_compensation_gain: float = 0.75
        teacher_pitch_kp: float = 2.5
        teacher_inner_pitch_ki: float = 0.9
        teacher_use_tecs_bank_aware_speed_sp: bool = True
        teacher_tecs_bank_aware_speed_scale: float = 1.0
        teacher_tecs_bank_aware_speed_clamp_mps: float = 2.0
        # Heuristic maneuver-speed floor for turns/loiter: this is above straight-flight cruise
        # on purpose, so TECS carries more kinetic energy before the vehicle drops into recovery.
        teacher_use_tecs_bank_aware_min_airspeed: bool = True
        teacher_tecs_bank_aware_min_airspeed_mps: float = 8.0
        teacher_tecs_bank_aware_min_airspeed_scale: float = 1.0
        teacher_tecs_bank_aware_min_airspeed_clamp_mps: float = 2.0
        tail_horizontal_tail_incidence_bias_deg: float = 0.0
        tail_fixed_horizontal_effectiveness: float = 0.5
        tail_elevon_effectiveness: float = 1.2
        tail_elevon_alpha_limit_deg: float = 25.0
        tail_horizontal_tail_q_scale: float = 1.0
        base_body_com_override_x_m: float | None = -0.10
        reset_pitch_deg: float = 4.0
        reset_flap_hz: float = 3.4
        reset_elevon_pitch_deg: float = -18.0

        wind_enabled: bool = False
        wind_xy_mps: tuple[float, float] = (0.0, 0.0)
        randomize_wind: bool = False
        wind_x_range_mps: tuple[float, float] = (0.0, 0.0)
        wind_y_range_mps: tuple[float, float] = (0.0, 0.0)
        wind_ou_enabled: bool = False
        wind_ou_sigma_xy_mps: tuple[float, float] = (0.0, 0.0)
        wind_curriculum_enabled: bool = False
        act_lpf_tau_s: float = 0.0
        act_rate_limit_per_s: float = 0.0

        mission_seed: int = 0
        mission_num_segments_min: int = 2
        mission_num_segments_max: int = 4
        mission_allow_straight: bool = True
        mission_allow_turn: bool = True
        mission_allow_loiter: bool = True
        mission_allow_climb_on_straight: bool = True
        mission_increment_seed_per_reset: bool = True
        mission_curriculum_enabled: bool = False
        mission_curriculum_stage_steps: tuple[int, ...] = ()
        mission_curriculum_stage_modes: tuple[str, ...] = ()

        path_manager_max_roll_deg: float = 35.0
        path_manager_max_flight_path_angle_deg: float = 10.0
        loiter_turns: float = 1.0
        loiter_curriculum_enabled: bool = False
        loiter_curriculum_stage_steps: tuple[int, ...] = ()
        loiter_curriculum_stage_modes: tuple[str, ...] = ()
        loiter_curriculum_stage_turns: tuple[float, ...] = ()
        loiter_curriculum_straight_rehearsal_prob: float = 0.0
        path_episode_speed_ref_mps: float = 6.0
        path_episode_completion_margin_s: float = 3.0
        path_episode_max_s: float = 36.0

        terminate_path_error_m: float = 15.0
        terminate_height_error_m: float = 5.0
        no_progress_warmup_s: float = 2.0
        no_progress_window_s: float = 2.0
        no_progress_min_delta_s_m: float = 0.5
        no_progress_penalty: float = 0.5
        recovery_teacher_enabled: bool = False
        recovery_teacher_loiter_only: bool = True
        recovery_teacher_delta_scale: float = 0.0
        recovery_teacher_lateral_error_trigger_m: float = 2.0
        recovery_teacher_height_error_trigger_m: float = 1.5
        recovery_teacher_min_safe_airspeed_mps: float = 4.0
        recovery_teacher_tilt_trigger_deg: float = 55.0
        recovery_teacher_ang_rate_trigger_deg_s: float = 120.0
        teacher_action_gap_penalty: float = 0.0
        curve_progress_bonus: float = 0.25
        curve_tracking_bonus: float = 0.6
        curve_tracking_lateral_scale_m: float = 2.0
        curve_tracking_height_scale_m: float = 1.0
        curve_tracking_align_scale_rad: float = 0.35
        loiter_progress_bonus: float = 0.2
        loiter_tracking_bonus: float = 0.8
        loiter_tracking_radial_scale_m: float = 1.0
        loiter_tracking_align_scale_rad: float = 0.25
        loiter_radial_drift_penalty: float = 5.0
        loiter_quarter_turn_bonus: float = 1.0
        loiter_half_turn_bonus: float = 0.0
        loiter_three_quarter_turn_bonus: float = 0.0
        loiter_teacher_action_gap_penalty_scale: float = 1.0
        tracking_bonus_progress_gate_m: float = 0.05


    @configclass
    class FlappingBotPathTrackingWeakTeacherRLEnvCfg(FlappingBotPathTrackingEnvCfg):
        """Weaker teacher schedule for path-tracking continuation."""

        teacher_guidance_mode: str = "residual"
        teacher_guidance_zero_actor_init: bool = True
        teacher_guidance_schedule_steps: tuple[int, ...] = (0, 10_000, 30_000, 60_000)
        teacher_guidance_schedule_deltas: tuple[float, ...] = (0.35, 0.75, 1.5, 2.0)
        recovery_teacher_enabled: bool = True
        recovery_teacher_loiter_only: bool = True
        recovery_teacher_delta_scale: float = 0.0
        recovery_teacher_lateral_error_trigger_m: float = 0.5
        recovery_teacher_tilt_trigger_deg: float = 35.0
        teacher_action_gap_penalty: float = 0.5
        curve_teacher_delta_scale: float = 0.6
        loiter_teacher_delta_scale: float = 0.7
        loiter_teacher_action_gap_penalty_scale: float = 1.0


    @configclass
    class FlappingBotPathTrackingPureRLEnvCfg(FlappingBotPathTrackingEnvCfg):
        """Pure-RL continuation config for generic path tracking."""

        teacher_guidance_enabled: bool = False
        teacher_guidance_disable_after_steps: int = 0


    @configclass
    class FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg(FlappingBotPathTrackingWeakTeacherRLEnvCfg):
        """Primitive-stage weak-teacher config for no-wind single-segment curriculum."""

        teacher_guidance_mode: str = "residual"
        teacher_guidance_zero_actor_init: bool = True
        mission_num_segments_min: int = 1
        mission_num_segments_max: int = 1
        mission_allow_straight: bool = True
        mission_allow_turn: bool = True
        mission_allow_loiter: bool = True
        mission_allow_climb_on_straight: bool = False
        mission_curriculum_enabled: bool = True
        mission_curriculum_stage_steps: tuple[int, ...] = (0, 12_000, 24_000, 36_000)
        mission_curriculum_stage_modes: tuple[str, ...] = (
            "turn_only",
            "loiter_quarter",
            "loiter_half",
            "loiter_full_with_straight_rehearsal",
        )
        loiter_curriculum_enabled: bool = True
        loiter_curriculum_stage_steps: tuple[int, ...] = (0, 12_000, 24_000, 36_000)
        loiter_curriculum_stage_modes: tuple[str, ...] = (
            "turn_only",
            "loiter_quarter",
            "loiter_half",
            "loiter_full_with_straight_rehearsal",
        )
        loiter_curriculum_stage_turns: tuple[float, ...] = (1.0, 0.25, 0.5, 1.0)
        loiter_curriculum_straight_rehearsal_prob: float = 0.2


    @configclass
    class FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg(FlappingBotPathTrackingEnvCfg):
        """Primitive-stage teacher-RL config with absolute-action semantics."""

        mission_num_segments_min: int = 1
        mission_num_segments_max: int = 1
        mission_allow_straight: bool = True
        mission_allow_turn: bool = True
        mission_allow_loiter: bool = True
        mission_allow_climb_on_straight: bool = False
        mission_curriculum_enabled: bool = True
        mission_curriculum_stage_steps: tuple[int, ...] = (0, 12_000, 24_000, 36_000)
        mission_curriculum_stage_modes: tuple[str, ...] = (
            "turn_only",
            "loiter_quarter",
            "loiter_half",
            "loiter_full_with_straight_rehearsal",
        )
        loiter_curriculum_enabled: bool = True
        loiter_curriculum_stage_steps: tuple[int, ...] = (0, 12_000, 24_000, 36_000)
        loiter_curriculum_stage_modes: tuple[str, ...] = (
            "turn_only",
            "loiter_quarter",
            "loiter_half",
            "loiter_full_with_straight_rehearsal",
        )
        loiter_curriculum_stage_turns: tuple[float, ...] = (1.0, 0.25, 0.5, 1.0)
        loiter_curriculum_straight_rehearsal_prob: float = 0.2


    @configclass
    class FlappingBotPathTrackingPrimitivePureRLEnvCfg(FlappingBotPathTrackingPureRLEnvCfg):
        """Primitive-stage pure-RL config for no-wind single-segment curriculum."""

        mission_num_segments_min: int = 1
        mission_num_segments_max: int = 1
        mission_allow_straight: bool = True
        mission_allow_turn: bool = True
        mission_allow_loiter: bool = True
        mission_allow_climb_on_straight: bool = False
        mission_curriculum_enabled: bool = True
        mission_curriculum_stage_steps: tuple[int, ...] = (0, 400_000, 800_000, 1_200_000)
        mission_curriculum_stage_modes: tuple[str, ...] = (
            "turn_only",
            "loiter_quarter",
            "loiter_half",
            "loiter_full_with_straight_rehearsal",
        )
        loiter_curriculum_enabled: bool = True
        loiter_curriculum_stage_steps: tuple[int, ...] = (0, 400_000, 800_000, 1_200_000)
        loiter_curriculum_stage_modes: tuple[str, ...] = (
            "turn_only",
            "loiter_quarter",
            "loiter_half",
            "loiter_full_with_straight_rehearsal",
        )
        loiter_curriculum_stage_turns: tuple[float, ...] = (1.0, 0.25, 0.5, 1.0)
        loiter_curriculum_straight_rehearsal_prob: float = 0.2


    class FlappingBotPathTrackingEnv(FlappingBotStraightFlightEnv):
        """Path-tracking RL environment backed by canonical mission primitives."""

        def __init__(self, cfg: FlappingBotPathTrackingEnvCfg, render_mode: str | None = None, **kwargs):
            self._path_reset_counter: int = 0
            self._missions: list[Mission | None] = []
            self._path_managers: list[PathManager | None] = []

            self._path_closest_point_xyz: torch.Tensor | None = None
            self._path_closest_point_body_xyz: torch.Tensor | None = None
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
            self._path_episode_horizon_steps: torch.Tensor | None = None
            self._path_loiter_turns: torch.Tensor | None = None
            self._path_is_loiter: torch.Tensor | None = None
            self._path_loiter_radial_error_m: torch.Tensor | None = None
            self._path_loiter_radial_error_prev_m: torch.Tensor | None = None
            self._path_loiter_progress_ratio: torch.Tensor | None = None
            self._path_loiter_progress_prev: torch.Tensor | None = None
            self._path_loiter_quarter_turn_complete: torch.Tensor | None = None
            self._teacher_recovery_active_mask: torch.Tensor | None = None
            self._path_query_dirty: bool = True
            self._eval_abs_lateral_error_m: torch.Tensor | None = None
            self._eval_abs_height_error_m: torch.Tensor | None = None
            self._eval_abs_align_error_deg: torch.Tensor | None = None
            self._eval_progress_ratio: torch.Tensor | None = None
            self._eval_loiter_radial_error_m: torch.Tensor | None = None
            self._eval_loiter_progress_ratio: torch.Tensor | None = None
            self._eval_loiter_quarter_turn_complete: torch.Tensor | None = None
            self._eval_stalled: torch.Tensor | None = None
            self._stall_window_start_step: torch.Tensor | None = None
            self._stall_window_start_progress_s: torch.Tensor | None = None
            self._stalled: torch.Tensor | None = None
            self.reset_stalled: torch.Tensor | None = None

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
                        use_tecs_load_factor_compensation=bool(
                            self.cfg.teacher_use_tecs_load_factor_compensation
                        ),
                        tecs_roll_throttle_compensation=float(self.cfg.teacher_tecs_roll_throttle_compensation),
                        tecs_load_factor_clamp_max=float(self.cfg.teacher_tecs_load_factor_clamp_max),
                        tecs_load_factor_use_roll_sp=bool(self.cfg.teacher_tecs_load_factor_use_roll_sp),
                        load_factor_pitch_compensation_gain=float(
                            self.cfg.teacher_tecs_load_factor_pitch_compensation_gain
                        ),
                        pitch_kp=float(self.cfg.teacher_pitch_kp),
                        inner_pitch_ki=float(self.cfg.teacher_inner_pitch_ki),
                        use_tecs_bank_aware_speed_sp=bool(self.cfg.teacher_use_tecs_bank_aware_speed_sp),
                        tecs_bank_aware_speed_scale=float(self.cfg.teacher_tecs_bank_aware_speed_scale),
                        tecs_bank_aware_speed_clamp_mps=float(self.cfg.teacher_tecs_bank_aware_speed_clamp_mps),
                        use_tecs_bank_aware_min_airspeed=bool(self.cfg.teacher_use_tecs_bank_aware_min_airspeed),
                        tecs_bank_aware_min_airspeed_mps=float(self.cfg.teacher_tecs_bank_aware_min_airspeed_mps),
                        tecs_bank_aware_min_airspeed_scale=float(self.cfg.teacher_tecs_bank_aware_min_airspeed_scale),
                        tecs_bank_aware_min_airspeed_clamp_mps=float(
                            self.cfg.teacher_tecs_bank_aware_min_airspeed_clamp_mps
                        ),
                        initial_elevon_pitch_action=float(self.cfg.reset_elevon_pitch_deg)
                        / max(float(self.cfg.elevon_max_deg), 1.0e-6),
                        initial_elevon_roll_action=float(self.cfg.reset_elevon_roll_deg)
                        / max(float(self.cfg.elevon_max_deg), 1.0e-6),
                    ),
                    device=self.device,
                )

        def _ensure_path_buffers(self) -> None:
            if self._path_closest_point_xyz is not None:
                return

            num_envs = int(self.num_envs)
            device = self.device
            self._path_closest_point_xyz = torch.zeros((num_envs, 3), device=device)
            self._path_closest_point_body_xyz = torch.zeros((num_envs, 3), device=device)
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
            self._path_episode_horizon_steps = torch.full(
                (num_envs,),
                int(self.max_episode_length),
                device=device,
                dtype=torch.long,
            )
            self._path_loiter_turns = torch.full((num_envs,), float(self.cfg.loiter_turns), device=device)
            self._path_is_loiter = torch.zeros((num_envs,), device=device, dtype=torch.bool)
            self._path_loiter_radial_error_m = torch.full((num_envs,), float("nan"), device=device)
            self._path_loiter_radial_error_prev_m = torch.zeros((num_envs,), device=device)
            self._path_loiter_progress_ratio = torch.full((num_envs,), float("nan"), device=device)
            self._path_loiter_progress_prev = torch.zeros((num_envs,), device=device)
            self._path_loiter_quarter_turn_complete = torch.zeros((num_envs,), device=device, dtype=torch.bool)
            self._teacher_recovery_active_mask = torch.zeros((num_envs,), device=device, dtype=torch.bool)
            self._eval_abs_lateral_error_m = torch.zeros((num_envs,), device=device)
            self._eval_abs_height_error_m = torch.zeros((num_envs,), device=device)
            self._eval_abs_align_error_deg = torch.zeros((num_envs,), device=device)
            self._eval_progress_ratio = torch.zeros((num_envs,), device=device)
            self._eval_loiter_radial_error_m = torch.full((num_envs,), float("nan"), device=device)
            self._eval_loiter_progress_ratio = torch.full((num_envs,), float("nan"), device=device)
            self._eval_loiter_quarter_turn_complete = torch.zeros((num_envs,), device=device, dtype=torch.bool)
            self._eval_stalled = torch.zeros((num_envs,), device=device, dtype=torch.bool)
            self._stall_window_start_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
            self._stall_window_start_progress_s = torch.zeros((num_envs,), device=device)
            self._stalled = torch.zeros((num_envs,), device=device, dtype=torch.bool)
            self.reset_stalled = torch.zeros((num_envs,), device=device, dtype=torch.bool)

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
            seed = _compute_mission_seed(
                base_seed=int(self.cfg.mission_seed),
                path_reset_counter=int(self._path_reset_counter),
                env_id=int(env_id),
                increment_per_reset=bool(self.cfg.mission_increment_seed_per_reset),
            )
            self._path_reset_counter += 1
            schedule_step = _compute_curriculum_schedule_step(
                common_step_counter=int(self.common_step_counter),
                num_envs=int(self.num_envs),
            )
            base_flags = (
                bool(self.cfg.mission_allow_straight),
                bool(self.cfg.mission_allow_turn),
                bool(self.cfg.mission_allow_loiter),
            )
            explicit_single_primitive = sum(int(flag) for flag in base_flags) == 1

            allow_straight, allow_turn, allow_loiter = _resolve_path_tracking_curriculum(
                step=schedule_step,
                enabled=bool(self.cfg.mission_curriculum_enabled) and not explicit_single_primitive,
                stage_steps=tuple(int(v) for v in self.cfg.mission_curriculum_stage_steps),
                stage_modes=tuple(str(v) for v in self.cfg.mission_curriculum_stage_modes),
                allow_straight=bool(self.cfg.mission_allow_straight),
                allow_turn=bool(self.cfg.mission_allow_turn),
                allow_loiter=bool(self.cfg.mission_allow_loiter),
            )
            loiter_stage = _resolve_loiter_curriculum_stage(
                step=schedule_step,
                enabled=bool(self.cfg.loiter_curriculum_enabled) and not explicit_single_primitive,
                stage_steps=tuple(int(v) for v in self.cfg.loiter_curriculum_stage_steps),
                stage_modes=tuple(str(v) for v in self.cfg.loiter_curriculum_stage_modes),
                stage_turns=tuple(float(v) for v in self.cfg.loiter_curriculum_stage_turns),
                default_loiter_turns=float(self.cfg.loiter_turns),
                straight_rehearsal_prob=float(self.cfg.loiter_curriculum_straight_rehearsal_prob),
            )
            assert self._path_loiter_turns is not None
            self._path_loiter_turns[env_id] = float(loiter_stage.loiter_turns)

            if (
                int(self.cfg.mission_num_segments_min) == 1
                and int(self.cfg.mission_num_segments_max) == 1
                and allow_loiter
                and loiter_stage.straight_rehearsal_prob > 0.0
            ):
                rng = random.Random(seed ^ 104_729)
                if allow_straight and rng.random() < loiter_stage.straight_rehearsal_prob:
                    return Mission([MissionSegment(kind="straight", altitude_changes=False)])
                return Mission([MissionSegment(kind="loiter", altitude_changes=False)])

            return sample_mission(
                MissionGeneratorCfg(
                    seed=seed,
                    num_segments_min=int(self.cfg.mission_num_segments_min),
                    num_segments_max=int(self.cfg.mission_num_segments_max),
                    allow_straight=allow_straight,
                    allow_turn=allow_turn,
                    allow_loiter=allow_loiter,
                    allow_climb_on_straight=bool(self.cfg.mission_allow_climb_on_straight),
                )
            )

        def _make_path_manager(self, mission: Mission, env_id: int) -> PathManager:
            assert self._path_loiter_turns is not None
            return PathManager(
                PathManagerCfg(
                    max_roll_deg=float(self.cfg.path_manager_max_roll_deg),
                    max_flight_path_angle_deg=float(self.cfg.path_manager_max_flight_path_angle_deg),
                    loiter_turns=float(self._path_loiter_turns[env_id].item()),
                    initial_altitude_m=float(self._height_cmd[env_id].item()),
                ),
                mission,
            )

        def _pre_physics_step(self, actions: torch.Tensor):
            prev_act_cmd = None if self._act_cmd is None else self._act_cmd.clone()
            super()._pre_physics_step(actions)
            self._ensure_path_buffers()
            if self._teacher_recovery_active_mask is not None:
                self.extras.setdefault("log", {})
                self.extras["log"]["Teacher/recovery_frac"] = float(
                    self._teacher_recovery_active_mask.float().mean().item()
                )
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
            assert self._path_closest_point_body_xyz is not None
            assert self._path_action_delta is not None
            assert self._path_episode_horizon_steps is not None
            assert self._path_loiter_turns is not None
            assert self._path_is_loiter is not None
            assert self._path_loiter_radial_error_m is not None
            assert self._path_loiter_radial_error_prev_m is not None
            assert self._path_loiter_progress_ratio is not None
            assert self._path_loiter_progress_prev is not None
            assert self._path_loiter_quarter_turn_complete is not None
            assert self._teacher_recovery_active_mask is not None
            assert self._stall_window_start_step is not None
            assert self._stall_window_start_progress_s is not None
            assert self._stalled is not None
            assert self.reset_stalled is not None

            for env_id in env_ids.tolist():
                mission = self._sample_mission_for_env(env_id)
                self._missions[env_id] = mission
                manager = self._make_path_manager(mission, env_id)
                self._path_managers[env_id] = manager
                episode_length_s = _compute_path_episode_length_s(
                    current_episode_length_s=float(self.cfg.episode_length_s),
                    path_total_length_m=float(manager.total_length_m),
                    speed_ref_mps=float(self.cfg.path_episode_speed_ref_mps),
                    freeze_steps=int(self.cfg.freeze_steps_after_reset),
                    sim_dt=float(self.cfg.sim.dt),
                    completion_margin_s=float(self.cfg.path_episode_completion_margin_s),
                    max_episode_length_s=float(self.cfg.path_episode_max_s),
                )
                self._path_episode_horizon_steps[env_id] = max(
                    int(math.ceil(episode_length_s / max(float(self.step_dt), 1.0e-6))),
                    1,
                )

            self._path_progress_prev_s[env_ids] = 0.0
            self._path_delta_s[env_ids] = 0.0
            self._path_lateral_error_m[env_ids] = 0.0
            self._path_height_error_m[env_ids] = 0.0
            self._path_align_error_rad[env_ids] = 0.0
            self._path_curvature_m_inv[env_ids] = 0.0
            self._path_progress_s[env_ids] = 0.0
            self._path_height_sp_m[env_ids] = self._height_cmd[env_ids]
            self._path_closest_point_xyz[env_ids] = 0.0
            self._path_closest_point_body_xyz[env_ids] = 0.0
            self._path_tangent_xy[env_ids, 0] = 1.0
            self._path_tangent_xy[env_ids, 1] = 0.0
            self._path_preview_points_xyz[env_ids] = 0.0
            self._path_preview_points_body_xyz[env_ids] = 0.0
            self._path_action_delta[env_ids] = 0.0
            self._path_is_loiter[env_ids] = False
            self._path_loiter_radial_error_m[env_ids] = float("nan")
            self._path_loiter_radial_error_prev_m[env_ids] = 0.0
            self._path_loiter_progress_ratio[env_ids] = float("nan")
            self._path_loiter_progress_prev[env_ids] = 0.0
            self._path_loiter_quarter_turn_complete[env_ids] = False
            self._teacher_recovery_active_mask[env_ids] = False
            self._stall_window_start_step[env_ids] = -1
            self._stall_window_start_progress_s[env_ids] = 0.0
            self._stalled[env_ids] = False
            self._path_query_dirty = True

        def _refresh_path_state(self) -> None:
            if not self._path_query_dirty:
                return

            self._ensure_path_buffers()
            assert self._path_closest_point_xyz is not None
            assert self._path_tangent_xy is not None
            assert self._path_closest_point_body_xyz is not None
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
            assert self._path_is_loiter is not None
            assert self._path_loiter_radial_error_m is not None
            assert self._path_loiter_radial_error_prev_m is not None
            assert self._path_loiter_progress_ratio is not None
            assert self._path_loiter_quarter_turn_complete is not None

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
                is_loiter = str(getattr(query, "segment_kind", "")) == "loiter"
                self._path_is_loiter[env_id] = is_loiter
                if is_loiter and query.loiter_center_xy is not None and query.loiter_radius_m is not None:
                    position_xy = pos_local[env_id : env_id + 1, 0:2]
                    center_xy = torch.tensor([query.loiter_center_xy], device=self.device, dtype=pos_local.dtype)
                    radius_m = torch.tensor([float(query.loiter_radius_m)], device=self.device, dtype=pos_local.dtype)
                    loiter_radial_error = torch.abs(
                        _compute_loiter_radial_error(position_xy=position_xy, center_xy=center_xy, radius_m=radius_m)
                    )
                    loiter_total_turns = float(getattr(query, "loiter_total_turns", 1.0) or 1.0)
                    loiter_turn_direction = int(getattr(query, "loiter_turn_direction", 1) or 1)
                    segment_progress_ratio = float(getattr(query, "segment_progress_ratio", 0.0))
                    start_angle_rad = torch.tensor(
                        [float(query.loiter_start_angle_rad or 0.0)], device=self.device, dtype=pos_local.dtype
                    )
                    relative_xy = position_xy - center_xy
                    current_angle_rad = torch.atan2(relative_xy[:, 1], relative_xy[:, 0])
                    loiter_progress_ratio = _compute_loiter_angular_progress(
                        start_angle_rad=start_angle_rad,
                        current_angle_rad=current_angle_rad,
                        turn_direction=torch.tensor([loiter_turn_direction], device=self.device, dtype=torch.long),
                        total_turns=torch.tensor([loiter_total_turns], device=self.device, dtype=pos_local.dtype),
                        reference_progress_ratio=torch.tensor(
                            [segment_progress_ratio], device=self.device, dtype=pos_local.dtype
                        ),
                    )
                    self._path_loiter_radial_error_m[env_id] = loiter_radial_error[0]
                    self._path_loiter_progress_ratio[env_id] = loiter_progress_ratio[0]
                    self._path_loiter_quarter_turn_complete[env_id] = _compute_loiter_milestone_mask(
                        progress_ratio=loiter_progress_ratio, threshold=0.25
                    )[0]
                else:
                    self._path_loiter_radial_error_m[env_id] = float("nan")
                    self._path_loiter_progress_ratio[env_id] = float("nan")
                    self._path_loiter_quarter_turn_complete[env_id] = False

            self._path_delta_s.copy_(torch.clamp(self._path_progress_s - self._path_progress_prev_s, min=0.0))
            self._path_progress_prev_s.copy_(self._path_progress_s)
            self._path_height_error_m.copy_(pos_local[:, 2] - self._path_height_sp_m)
            self._path_align_error_rad.copy_(
                _compute_alignment_error(tangent_xy=self._path_tangent_xy, ground_vel_xy=ground_vel_local[:, 0:2])
            )

            rel_preview_xyz = self._path_preview_points_xyz - pos_local.unsqueeze(1)
            rel_closest_xyz = self._path_closest_point_xyz - pos_local
            quat_batch = self._robot.data.root_quat_w.repeat_interleave(self._path_preview_points_xyz.shape[1], dim=0)
            self._path_closest_point_body_xyz.copy_(quat_apply_inverse(self._robot.data.root_quat_w, rel_closest_xyz))
            self._path_preview_points_body_xyz.copy_(
                quat_apply_inverse(quat_batch, rel_preview_xyz.reshape(-1, 3)).reshape(self.num_envs, -1, 3)
            )
            self._path_query_dirty = False

        def _capture_eval_metrics(self) -> None:
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._path_align_error_rad is not None
            assert self._path_progress_s is not None
            assert self._path_loiter_radial_error_m is not None
            assert self._path_loiter_radial_error_prev_m is not None
            assert self._path_loiter_progress_ratio is not None
            assert self._path_loiter_quarter_turn_complete is not None
            assert self._eval_abs_lateral_error_m is not None
            assert self._eval_abs_height_error_m is not None
            assert self._eval_abs_align_error_deg is not None
            assert self._eval_progress_ratio is not None
            assert self._eval_loiter_radial_error_m is not None
            assert self._eval_loiter_progress_ratio is not None
            assert self._eval_loiter_quarter_turn_complete is not None
            assert self._eval_stalled is not None
            assert self._stalled is not None

            self._eval_abs_lateral_error_m.copy_(torch.abs(self._path_lateral_error_m))
            self._eval_abs_height_error_m.copy_(torch.abs(self._path_height_error_m))
            self._eval_abs_align_error_deg.copy_(torch.rad2deg(torch.abs(self._path_align_error_rad)))
            self._eval_loiter_radial_error_m.copy_(self._path_loiter_radial_error_m)
            self._eval_loiter_progress_ratio.copy_(self._path_loiter_progress_ratio)
            self._eval_loiter_quarter_turn_complete.copy_(self._path_loiter_quarter_turn_complete)
            self._eval_stalled.copy_(self._stalled)

            for env_id, manager in enumerate(self._path_managers):
                total_length = float(manager.total_length_m) if manager is not None else 1.0
                progress_ratio = float(self._path_progress_s[env_id].item()) / max(total_length, 1.0e-6)
                self._eval_progress_ratio[env_id] = min(max(progress_ratio, 0.0), 1.0)

        def _update_stall_flags(self) -> torch.Tensor:
            assert self._path_progress_s is not None
            assert self._stall_window_start_step is not None
            assert self._stall_window_start_progress_s is not None
            assert self._stalled is not None

            warmup_steps = max(0, int(math.ceil(float(self.cfg.no_progress_warmup_s) / max(float(self.step_dt), 1.0e-6))))
            window_steps = max(1, int(math.ceil(float(self.cfg.no_progress_window_s) / max(float(self.step_dt), 1.0e-6))))
            min_delta_s_m = float(self.cfg.no_progress_min_delta_s_m)

            episode_step = self.episode_length_buf
            active = episode_step >= warmup_steps
            newly_active = active & (self._stall_window_start_step < 0)
            if torch.any(newly_active):
                self._stall_window_start_step[newly_active] = episode_step[newly_active]
                self._stall_window_start_progress_s[newly_active] = self._path_progress_s[newly_active]

            window_ready = active & (self._stall_window_start_step >= 0)
            window_ready = window_ready & ((episode_step - self._stall_window_start_step) >= window_steps)
            progress_delta = self._path_progress_s - self._stall_window_start_progress_s
            path_total_length_m = torch.ones_like(self._path_progress_s)
            for env_id, manager in enumerate(self._path_managers):
                if manager is not None:
                    path_total_length_m[env_id] = float(manager.total_length_m)

            self._stalled.copy_(
                _compute_stall_mask(
                    window_ready=window_ready,
                    progress_delta=progress_delta,
                    progress_s=self._path_progress_s,
                    path_total_length_m=path_total_length_m,
                    min_delta_s_m=min_delta_s_m,
                )
            )

            rollout_mask = window_ready & (~self._stalled)
            if torch.any(rollout_mask):
                self._stall_window_start_step[rollout_mask] = episode_step[rollout_mask]
                self._stall_window_start_progress_s[rollout_mask] = self._path_progress_s[rollout_mask]

            return self._stalled

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

        def _get_teacher_delta(self) -> float | torch.Tensor:
            from ...px4_like.rl_training_utils import compute_teacher_guidance_delta

            base_delta = compute_teacher_guidance_delta(
                int(self.common_step_counter),
                enabled=bool(self.cfg.teacher_guidance_enabled),
                delta_init=float(self.cfg.teacher_guidance_delta_init),
                delta_final=float(self.cfg.teacher_guidance_delta_final),
                anneal_steps=int(self.cfg.teacher_guidance_anneal_steps),
                schedule_steps=tuple(int(v) for v in self.cfg.teacher_guidance_schedule_steps),
                schedule_deltas=tuple(float(v) for v in self.cfg.teacher_guidance_schedule_deltas),
                disable_after_steps=int(self.cfg.teacher_guidance_disable_after_steps),
            )
            self._refresh_path_state()
            assert self._path_curvature_m_inv is not None
            assert self._path_is_loiter is not None
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._teacher_recovery_active_mask is not None
            curve_delta = _compute_curve_aware_teacher_delta(
                base_delta=base_delta,
                curvature_m_inv=self._path_curvature_m_inv,
                curvature_ref_m_inv=float(self.cfg.curve_teacher_curvature_ref_m_inv),
                min_scale=float(self.cfg.curve_teacher_delta_scale),
            )
            delta = _compute_segment_aware_teacher_delta(
                base_delta=curve_delta,
                is_loiter=self._path_is_loiter,
                loiter_min_scale=float(self.cfg.loiter_teacher_delta_scale),
            )
            if not bool(self.cfg.recovery_teacher_enabled):
                self._teacher_recovery_active_mask.zero_()
                return delta

            airspeed = torch.linalg.norm(self._robot.data.root_lin_vel_w - self._wind_w, dim=1)
            g_b = self._robot.data.projected_gravity_b
            tilt_sin = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2).clamp(0.0, 1.0)
            tilt_deg = torch.rad2deg(torch.asin(tilt_sin))
            ang_rate_deg_s = torch.rad2deg(torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1))
            recovery_mask = _teacher_recovery_mask(
                lateral_error=torch.abs(self._path_lateral_error_m),
                height_error=torch.abs(self._path_height_error_m),
                airspeed=airspeed,
                tilt_deg=tilt_deg,
                ang_rate_deg_s=ang_rate_deg_s,
                lateral_error_trigger_m=float(self.cfg.recovery_teacher_lateral_error_trigger_m),
                height_error_trigger_m=float(self.cfg.recovery_teacher_height_error_trigger_m),
                min_safe_airspeed_mps=float(self.cfg.recovery_teacher_min_safe_airspeed_mps),
                tilt_trigger_deg=float(self.cfg.recovery_teacher_tilt_trigger_deg),
                ang_rate_trigger_deg_s=float(self.cfg.recovery_teacher_ang_rate_trigger_deg_s),
            )
            if bool(self.cfg.recovery_teacher_loiter_only):
                recovery_mask = recovery_mask & self._path_is_loiter
            self._teacher_recovery_active_mask.copy_(recovery_mask)
            return _apply_recovery_teacher_delta(
                base_delta=delta,
                recovery_mask=recovery_mask,
                recovery_scale=float(self.cfg.recovery_teacher_delta_scale),
            )

        def _get_observations(self) -> dict[str, torch.Tensor]:
            self._refresh_path_state()
            assert self._path_height_sp_m is not None
            assert self._path_closest_point_body_xyz is not None
            assert self._path_preview_points_body_xyz is not None
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._path_curvature_m_inv is not None
            assert self._path_progress_s is not None
            assert self._path_align_error_rad is not None

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
                    "closest_point_body_xyz": self._path_closest_point_body_xyz,
                    "tangent_xy": self._path_tangent_xy,
                    "preview_points_body_xyz": self._path_preview_points_body_xyz,
                    "lateral_error_m": self._path_lateral_error_m,
                    "height_error_m": self._path_height_error_m,
                    "align_error_rad": self._path_align_error_rad,
                    "curvature_m_inv": self._path_curvature_m_inv,
                    "previous_action": self._act_cmd,
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
            assert self._path_curvature_m_inv is not None
            assert self._path_progress_s is not None
            assert self._path_is_loiter is not None
            assert self._path_loiter_radial_error_m is not None
            assert self._path_loiter_radial_error_prev_m is not None
            assert self._path_loiter_progress_ratio is not None
            assert self._path_loiter_progress_prev is not None
            assert self._path_loiter_quarter_turn_complete is not None
            assert self._stalled is not None
            assert self._teacher_action_gap_abs is not None

            airspeed = torch.linalg.norm(self._robot.data.root_lin_vel_w - self._wind_w, dim=1)
            pos_w = self._robot.data.root_pos_w - self.scene.env_origins
            g_b = self._robot.data.projected_gravity_b
            tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
            ang_rate = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
            tilt_thr = torch.sin(torch.deg2rad(torch.tensor(self.cfg.terminate_tilt_deg, device=self.device)))
            terminated = pos_w[:, 2] <= float(self.cfg.terminate_ground_height)
            terminated = terminated | (tilt > tilt_thr)
            terminated = terminated | (torch.abs(self._path_lateral_error_m) > float(self.cfg.terminate_path_error_m))
            terminated = terminated | (torch.abs(self._path_height_error_m) > float(self.cfg.terminate_height_error_m))
            path_total_length_m = torch.ones_like(self._path_progress_s)
            for env_id, manager in enumerate(self._path_managers):
                if manager is not None:
                    path_total_length_m[env_id] = float(manager.total_length_m)
            completed = _compute_path_completion_mask(
                progress_s=self._path_progress_s,
                path_total_length_m=path_total_length_m,
                completion_ratio=float(self.cfg.completion_ratio),
            )
            terminated = terminated & (~completed)
            loiter_progress_ratio = torch.where(
                self._path_is_loiter,
                torch.nan_to_num(self._path_loiter_progress_ratio, nan=0.0),
                torch.zeros_like(self._path_loiter_progress_prev),
            )
            loiter_radial_error = torch.where(
                self._path_is_loiter,
                torch.nan_to_num(self._path_loiter_radial_error_m, nan=0.0),
                torch.zeros_like(self._path_loiter_progress_prev),
            )
            loiter_radial_error_delta = torch.clamp(loiter_radial_error - self._path_loiter_radial_error_prev_m, min=0.0)
            loiter_milestone_bonus_reward = _compute_loiter_milestone_bonus(
                previous_progress_ratio=self._path_loiter_progress_prev,
                progress_ratio=loiter_progress_ratio,
                quarter_turn_bonus=float(self.cfg.loiter_quarter_turn_bonus),
                half_turn_bonus=float(self.cfg.loiter_half_turn_bonus),
                three_quarter_turn_bonus=float(self.cfg.loiter_three_quarter_turn_bonus),
            )
            reward = _compute_tracking_reward(
                lateral_error=torch.abs(self._path_lateral_error_m),
                height_error=torch.abs(self._path_height_error_m),
                align_error=self._path_align_error_rad,
                delta_s=self._path_delta_s,
                curvature_m_inv=self._path_curvature_m_inv,
                airspeed=airspeed,
                action=self._act_cmd,
                action_delta=self._path_action_delta,
                tilt=tilt,
                ang_rate=ang_rate,
                terminated=terminated,
                is_loiter=self._path_is_loiter,
                loiter_radial_error=loiter_radial_error,
                loiter_progress_ratio=loiter_progress_ratio,
                loiter_radial_error_delta=loiter_radial_error_delta,
                loiter_milestone_bonus_reward=loiter_milestone_bonus_reward,
                completed=completed,
                stalled=self._stalled,
                no_progress_penalty=float(self.cfg.no_progress_penalty),
                teacher_action_gap_abs=self._teacher_action_gap_abs,
                teacher_action_gap_penalty=float(self.cfg.teacher_action_gap_penalty),
                loiter_teacher_action_gap_penalty_scale=float(self.cfg.loiter_teacher_action_gap_penalty_scale),
                completion_bonus=float(self.cfg.completion_bonus),
                curve_progress_bonus=float(self.cfg.curve_progress_bonus),
                curve_tracking_bonus=float(self.cfg.curve_tracking_bonus),
                curve_tracking_lateral_scale_m=float(self.cfg.curve_tracking_lateral_scale_m),
                curve_tracking_height_scale_m=float(self.cfg.curve_tracking_height_scale_m),
                curve_tracking_align_scale_rad=float(self.cfg.curve_tracking_align_scale_rad),
                loiter_progress_bonus=float(self.cfg.loiter_progress_bonus),
                loiter_tracking_bonus=float(self.cfg.loiter_tracking_bonus),
                loiter_tracking_radial_scale_m=float(self.cfg.loiter_tracking_radial_scale_m),
                loiter_tracking_align_scale_rad=float(self.cfg.loiter_tracking_align_scale_rad),
                loiter_radial_drift_penalty=float(self.cfg.loiter_radial_drift_penalty),
                tracking_bonus_progress_gate_m=float(self.cfg.tracking_bonus_progress_gate_m),
            )
            self._path_loiter_progress_prev.copy_(loiter_progress_ratio)
            self._path_loiter_radial_error_prev_m.copy_(loiter_radial_error)
            return reward

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            self._refresh_path_state()
            assert self._path_lateral_error_m is not None
            assert self._path_height_error_m is not None
            assert self._path_progress_s is not None
            assert self._path_episode_horizon_steps is not None
            assert self.reset_stalled is not None
            self.reset_stalled.copy_(self._update_stall_flags())
            self._capture_eval_metrics()

            path_total_length_m = torch.ones_like(self._path_progress_s)
            for env_id, manager in enumerate(self._path_managers):
                if manager is not None:
                    path_total_length_m[env_id] = float(manager.total_length_m)
            completed = _compute_path_completion_mask(
                progress_s=self._path_progress_s,
                path_total_length_m=path_total_length_m,
                completion_ratio=float(self.cfg.completion_ratio),
            )
            pos_w = self._robot.data.root_pos_w - self.scene.env_origins
            fell = pos_w[:, 2] <= float(self.cfg.terminate_ground_height)

            g_b = self._robot.data.projected_gravity_b
            tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
            tilt_thr = torch.sin(torch.deg2rad(torch.tensor(self.cfg.terminate_tilt_deg, device=self.device)))
            fell = fell | (tilt > tilt_thr)
            fell = fell | (torch.abs(self._path_lateral_error_m) > float(self.cfg.terminate_path_error_m))
            fell = fell | (torch.abs(self._path_height_error_m) > float(self.cfg.terminate_height_error_m))
            fell = fell | self.reset_stalled
            fell = fell & (~completed)

            timed_out = self.episode_length_buf >= (self._path_episode_horizon_steps - 1)
            timed_out = timed_out | completed
            return fell, timed_out


def _build_preview_observation(query: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build a flat preview-based observation vector."""
    current_geometry = torch.cat(
        (
            query["closest_point_body_xyz"],
            query["tangent_xy"],
            query["curvature_m_inv"].unsqueeze(-1),
            query["lateral_error_m"].unsqueeze(-1),
            query["height_error_m"].unsqueeze(-1),
            query["align_error_rad"].unsqueeze(-1),
        ),
        dim=-1,
    )
    preview = query["preview_points_body_xyz"].reshape(query["preview_points_body_xyz"].shape[0], -1)
    return torch.cat((current_geometry, preview, query["previous_action"]), dim=-1)


def _compute_tracking_reward(
    *,
    lateral_error: torch.Tensor,
    height_error: torch.Tensor,
    align_error: torch.Tensor,
    delta_s: torch.Tensor,
    curvature_m_inv: torch.Tensor | None = None,
    airspeed: torch.Tensor,
    action: torch.Tensor,
    action_delta: torch.Tensor,
    tilt: torch.Tensor,
    ang_rate: torch.Tensor,
    terminated: torch.Tensor,
    is_loiter: torch.Tensor | None = None,
    loiter_radial_error: torch.Tensor | None = None,
    loiter_progress_ratio: torch.Tensor | None = None,
    loiter_radial_error_delta: torch.Tensor | None = None,
    loiter_milestone_bonus_reward: torch.Tensor | None = None,
    completed: torch.Tensor | None = None,
    stalled: torch.Tensor | None = None,
    no_progress_penalty: float = 0.0,
    teacher_action_gap_abs: torch.Tensor | None = None,
    teacher_action_gap_penalty: float = 0.0,
    loiter_teacher_action_gap_penalty_scale: float = 1.0,
    completion_bonus: float = 0.0,
    curve_progress_bonus: float = 0.0,
    curve_tracking_bonus: float = 0.0,
    curve_tracking_lateral_scale_m: float = 2.0,
    curve_tracking_height_scale_m: float = 1.0,
    curve_tracking_align_scale_rad: float = 0.35,
    loiter_progress_bonus: float = 0.0,
    loiter_tracking_bonus: float = 0.0,
    loiter_tracking_radial_scale_m: float = 1.0,
    loiter_tracking_align_scale_rad: float = 0.25,
    loiter_radial_drift_penalty: float = 0.0,
    tracking_bonus_progress_gate_m: float = 0.05,
) -> torch.Tensor:
    """Compute a tracking-plus-progress reward."""
    low_speed_penalty = 0.2 * torch.clamp(4.0 - airspeed, min=0.0)
    action_penalty = 0.01 * torch.sum(action**2, dim=1)
    action_delta_penalty = 0.01 * torch.sum(action_delta**2, dim=1)
    tilt_penalty = 0.05 * tilt
    ang_rate_penalty = 0.02 * ang_rate
    termination_penalty = 5.0 * terminated.float()
    stall_penalty = 0.0
    if stalled is not None and no_progress_penalty > 0.0:
        stall_penalty = float(no_progress_penalty) * stalled.float()
    teacher_gap_penalty = 0.0
    if teacher_action_gap_abs is not None and teacher_action_gap_penalty > 0.0:
        teacher_gap_scale = torch.ones_like(lateral_error)
        if is_loiter is not None:
            loiter_gap_scale = max(float(loiter_teacher_action_gap_penalty_scale), 1.0)
            teacher_gap_scale = teacher_gap_scale + is_loiter.to(dtype=lateral_error.dtype) * (loiter_gap_scale - 1.0)
        teacher_gap_penalty = float(teacher_action_gap_penalty) * teacher_action_gap_abs.mean(dim=1) * teacher_gap_scale
    curve_weight = 0.0
    progress_reward = delta_s
    curve_progress_reward = 0.0
    curve_tracking_reward = 0.0
    tracking_progress_gate = torch.clamp(delta_s / max(float(tracking_bonus_progress_gate_m), 1.0e-6), min=0.0, max=1.0)
    if curvature_m_inv is not None:
        curve_weight = torch.clamp(torch.abs(curvature_m_inv) / 0.05, min=0.0, max=1.0)
        if curve_progress_bonus > 0.0:
            curve_progress_reward = float(curve_progress_bonus) * curve_weight * delta_s
        if curve_tracking_bonus > 0.0:
            lateral_term = lateral_error / max(float(curve_tracking_lateral_scale_m), 1.0e-6)
            height_term = height_error / max(float(curve_tracking_height_scale_m), 1.0e-6)
            align_term = torch.abs(align_error) / max(float(curve_tracking_align_scale_rad), 1.0e-6)
            curve_tracking_reward = float(curve_tracking_bonus) * curve_weight * tracking_progress_gate * torch.exp(
                -(lateral_term + height_term + align_term)
            )
    loiter_progress_reward = 0.0
    loiter_tracking_reward = 0.0
    loiter_milestone_reward = 0.0
    loiter_radial_drift_cost = 0.0
    if is_loiter is not None:
        loiter_weight = is_loiter.to(dtype=lateral_error.dtype)
        loiter_tracking_quality = 1.0
        if loiter_radial_error is not None:
            loiter_tracking_quality = _compute_loiter_tracking_quality(
                loiter_radial_error=loiter_radial_error,
                align_error=align_error,
                loiter_tracking_radial_scale_m=loiter_tracking_radial_scale_m,
                loiter_tracking_align_scale_rad=loiter_tracking_align_scale_rad,
            )
            progress_reward = progress_reward * ((1.0 - loiter_weight) + loiter_weight * loiter_tracking_quality)
            curve_progress_reward = curve_progress_reward * ((1.0 - loiter_weight) + loiter_weight * loiter_tracking_quality)
        if loiter_progress_bonus > 0.0:
            loiter_progress_reward = float(loiter_progress_bonus) * loiter_weight * loiter_tracking_quality * delta_s
        if loiter_radial_error is not None and loiter_tracking_bonus > 0.0:
            loiter_tracking_reward = (
                float(loiter_tracking_bonus) * loiter_weight * tracking_progress_gate * loiter_tracking_quality
            )
        if loiter_radial_error_delta is not None and loiter_radial_drift_penalty > 0.0:
            loiter_radial_drift_cost = (
                float(loiter_radial_drift_penalty) * loiter_weight * torch.clamp(loiter_radial_error_delta, min=0.0)
            )
        if loiter_milestone_bonus_reward is not None:
            loiter_milestone_reward = loiter_weight * loiter_milestone_bonus_reward
    completion_reward = 0.0
    if completed is not None and completion_bonus > 0.0:
        completion_reward = float(completion_bonus) * completed.float()
    return (
        progress_reward
        + curve_progress_reward
        + curve_tracking_reward
        + loiter_progress_reward
        + loiter_tracking_reward
        + loiter_milestone_reward
        + completion_reward
        - 0.2 * lateral_error
        - 0.2 * height_error
        - 0.1 * align_error
        - low_speed_penalty
        - action_penalty
        - action_delta_penalty
        - tilt_penalty
        - ang_rate_penalty
        - termination_penalty
        - stall_penalty
        - teacher_gap_penalty
        - loiter_radial_drift_cost
    )


def _compute_stall_mask(
    *,
    window_ready: torch.Tensor,
    progress_delta: torch.Tensor,
    progress_s: torch.Tensor,
    path_total_length_m: torch.Tensor,
    min_delta_s_m: float,
    completion_ratio: float = 0.98,
) -> torch.Tensor:
    """Return stalled env mask while exempting missions that already reached completion."""
    completed = _compute_path_completion_mask(
        progress_s=progress_s,
        path_total_length_m=path_total_length_m,
        completion_ratio=completion_ratio,
    )
    stalled = torch.zeros_like(window_ready, dtype=torch.bool)
    stalled[window_ready] = (progress_delta[window_ready] < float(min_delta_s_m)) & (~completed[window_ready])
    return stalled


def _compute_path_completion_mask(
    *,
    progress_s: torch.Tensor,
    path_total_length_m: torch.Tensor,
    completion_ratio: float,
) -> torch.Tensor:
    """Return completed env mask from path progress."""
    return progress_s >= torch.clamp(path_total_length_m * float(completion_ratio), min=0.0)


def _compute_path_episode_length_s(
    *,
    current_episode_length_s: float,
    path_total_length_m: float,
    speed_ref_mps: float,
    freeze_steps: int,
    sim_dt: float,
    completion_margin_s: float,
    max_episode_length_s: float = 0.0,
) -> float:
    """Estimate a training horizon long enough to finish the current path."""
    speed_ref_mps = max(abs(float(speed_ref_mps)), 1.0)
    freeze_time_s = max(int(freeze_steps), 0) * max(float(sim_dt), 1.0e-6)
    required_episode_length_s = freeze_time_s + max(float(path_total_length_m), 0.0) / speed_ref_mps + max(
        float(completion_margin_s), 0.0
    )
    episode_length_s = max(float(current_episode_length_s), required_episode_length_s)
    if float(max_episode_length_s) > 0.0:
        episode_length_s = min(episode_length_s, float(max_episode_length_s))
    return episode_length_s


def _resolve_curriculum_active_mode(
    *,
    step: int,
    enabled: bool,
    stage_steps: tuple[int, ...],
    stage_modes: tuple[str, ...],
    default_mode: str,
) -> str:
    """Return the active curriculum mode for the current step."""
    if not enabled or len(stage_steps) == 0 or len(stage_modes) == 0:
        return str(default_mode)
    if len(stage_steps) != len(stage_modes):
        raise ValueError("curriculum stage_steps and stage_modes must have the same length.")

    active_mode = stage_modes[0]
    for stage_step, stage_mode in zip(stage_steps, stage_modes, strict=True):
        if int(step) < int(stage_step):
            break
        active_mode = stage_mode
    return str(active_mode)


def _resolve_loiter_curriculum_stage(
    *,
    step: int,
    enabled: bool,
    stage_steps: tuple[int, ...],
    stage_modes: tuple[str, ...],
    default_loiter_turns: float,
    straight_rehearsal_prob: float,
    stage_turns: tuple[float, ...] = (),
) -> _LoiterCurriculumStage:
    """Resolve loiter turn fraction and rehearsal probability for the current stage."""
    active_mode = _resolve_curriculum_active_mode(
        step=step,
        enabled=enabled,
        stage_steps=stage_steps,
        stage_modes=stage_modes,
        default_mode="all",
    )
    if len(stage_turns) > 0 and len(stage_turns) != len(stage_modes):
        raise ValueError("loiter_curriculum_stage_turns and loiter_curriculum_stage_modes must have the same length.")

    loiter_turns = max(float(default_loiter_turns), 0.0)
    if len(stage_turns) > 0 and enabled and len(stage_modes) > 0:
        active_idx = 0
        for stage_idx, stage_step in enumerate(stage_steps):
            if int(step) < int(stage_step):
                break
            active_idx = stage_idx
        loiter_turns = min(loiter_turns, max(float(stage_turns[active_idx]), 0.0))
    elif active_mode == "loiter_quarter":
        loiter_turns = min(loiter_turns, 0.25)
    elif active_mode == "loiter_half":
        loiter_turns = min(loiter_turns, 0.5)

    rehearsal_prob = 0.0
    if active_mode == "loiter_full_with_straight_rehearsal":
        rehearsal_prob = min(max(float(straight_rehearsal_prob), 0.0), 1.0)

    return _LoiterCurriculumStage(
        mode=active_mode,
        loiter_turns=loiter_turns,
        straight_rehearsal_prob=rehearsal_prob,
    )


def _resolve_path_tracking_curriculum(
    *,
    step: int,
    enabled: bool,
    stage_steps: tuple[int, ...],
    stage_modes: tuple[str, ...],
    allow_straight: bool,
    allow_turn: bool,
    allow_loiter: bool,
) -> tuple[bool, bool, bool]:
    """Resolve mission primitive availability for the current training stage."""
    base_flags = (bool(allow_straight), bool(allow_turn), bool(allow_loiter))
    active_mode = _resolve_curriculum_active_mode(
        step=step,
        enabled=enabled,
        stage_steps=stage_steps,
        stage_modes=stage_modes,
        default_mode="all",
    )

    if active_mode == "all":
        return base_flags
    if active_mode == "turn_only":
        resolved = (False, bool(allow_turn), False)
        return resolved if any(resolved) else base_flags
    if active_mode == "straight_turn":
        resolved = (bool(allow_straight), bool(allow_turn), False)
        return resolved if any(resolved) else base_flags
    if active_mode == "straight_only":
        resolved = (bool(allow_straight), False, False)
        return resolved if any(resolved) else base_flags
    if active_mode == "loiter_only":
        resolved = (False, False, bool(allow_loiter))
        return resolved if any(resolved) else base_flags
    if active_mode == "loiter_quarter":
        resolved = (False, False, bool(allow_loiter))
        return resolved if any(resolved) else base_flags
    if active_mode == "loiter_half":
        resolved = (False, False, bool(allow_loiter))
        return resolved if any(resolved) else base_flags
    if active_mode == "loiter_full_with_straight_rehearsal":
        resolved = (bool(allow_straight), False, bool(allow_loiter))
        return resolved if any(resolved) else base_flags
    if active_mode == "turn_loiter":
        resolved = (False, bool(allow_turn), bool(allow_loiter))
        return resolved if any(resolved) else base_flags
    raise ValueError(f"unsupported mission curriculum mode: {active_mode}")


def _compute_curve_aware_teacher_delta(
    *,
    base_delta: float,
    curvature_m_inv: torch.Tensor,
    curvature_ref_m_inv: float,
    min_scale: float,
) -> torch.Tensor:
    """Tighten teacher envelope on curved segments while leaving straight segments unchanged."""
    curvature_ref_m_inv = max(float(curvature_ref_m_inv), 1.0e-6)
    min_scale = min(max(float(min_scale), 0.0), 1.0)
    curve_weight = torch.clamp(torch.abs(curvature_m_inv) / curvature_ref_m_inv, min=0.0, max=1.0)
    delta_scale = 1.0 - curve_weight * (1.0 - min_scale)
    return (float(base_delta) * delta_scale).unsqueeze(-1)


def _compute_segment_aware_teacher_delta(
    *,
    base_delta: torch.Tensor,
    is_loiter: torch.Tensor,
    loiter_min_scale: float,
) -> torch.Tensor:
    """Tighten teacher envelope further on loiter while leaving non-loiter segments unchanged."""
    loiter_min_scale = min(max(float(loiter_min_scale), 0.0), 1.0)
    delta = base_delta.clone()
    loiter_mask = is_loiter.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
    return delta * (1.0 - loiter_mask * (1.0 - loiter_min_scale))


def _apply_recovery_teacher_delta(
    *,
    base_delta: torch.Tensor,
    recovery_mask: torch.Tensor,
    recovery_scale: float,
) -> torch.Tensor:
    """Tighten teacher envelope on masked recovery states."""
    recovery_scale = min(max(float(recovery_scale), 0.0), 1.0)
    delta = base_delta.clone()
    recovery_weight = recovery_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
    return delta * (1.0 - recovery_weight * (1.0 - recovery_scale))


def _teacher_recovery_mask(**kwargs) -> torch.Tensor:
    """Thin wrapper placeholder for recovery-teacher masking."""
    from ...px4_like.rl_training_utils import compute_recovery_teacher_mask

    return compute_recovery_teacher_mask(**kwargs)
