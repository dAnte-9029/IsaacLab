"""Direct RL environment for straight-flight flapping-wing control.

This environment is designed to work with IsaacLab's standard RSL-RL scripts:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK_ID> --headless

Key design choices for long-horizon iteration:
- Low-dimensional actions: throttle + rudder + elevon pitch/roll commands.
- Tail aerodynamics depends on deflection angle (not only joint velocity).
- Optional wing aerodynamic backend: simple QSM (fast) or DeLaurier (1993) strip theory (slower, more detailed).
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, quat_from_euler_xyz

from ...assets import FlappingBotCfg
from ...physics import (
    DeLaurierParams,
    compute_aero_wrench_delaurier1993,
    compute_area_weighted_quarter_chord_link_points,
    FlappingQSMCfg,
    QuasiSteadyWingModel,
    TailAeroCfg,
    TailAeroModel,
    WingQSMCfg,
    WingGeometry,
    build_wing_geometry_from_csv,
)
from ...px4_like.rl_training_utils import (
    apply_teacher_guided_actions,
    linear_anneal,
    piecewise_linear_anneal,
    resolve_teacher_guidance_mode,
    teacher_guidance_is_active,
)
from ...px4_like.straight_line_controller import PX4LikeStraightLineController, PX4LikeStraightLineControllerCfg
from ...scenes import FlappingRoomSceneCfg
from .state_source_contract import (
    TeacherStateInputs,
    resolve_imu_source,
    resolve_teacher_state_inputs,
)
from .startup_phase import advance_flap_phase

Tensor = torch.Tensor


@configclass
class FlappingBotStraightFlightEnvCfg(DirectRLEnvCfg):
    """Base configuration for straight-flight training."""

    # episode / control
    episode_length_s: float = 12.0
    decimation: int = 2
    action_space: int = 4  # [throttle, rudder, elevon_pitch, elevon_roll]
    observation_space: int = 68  # keep same stacking layout as FlappingBotEnv
    state_space: int = 0
    action_scale: float = 1.0

    # commands
    vx_cmd: float = 7.0
    height_cmd: float = 10.0
    randomize_commands: bool = False
    vx_cmd_range: tuple[float, float] = (2.0, 8.0)
    height_cmd_range: tuple[float, float] = (8.0, 12.0)
    wind_enabled: bool = False
    wind_xy_mps: tuple[float, float] = (0.0, 0.0)
    randomize_wind: bool = False
    wind_x_range_mps: tuple[float, float] = (0.0, 0.0)
    wind_y_range_mps: tuple[float, float] = (0.0, 0.0)
    # Optional time-varying gust model (OU / first-order Gauss-Markov) around a mean wind.
    wind_ou_enabled: bool = False
    wind_ou_tau_s: float = 2.0
    wind_ou_sigma_xy_mps: tuple[float, float] = (0.0, 0.0)
    wind_ou_clip_to_range: bool = False
    wind_curriculum_enabled: bool = False
    wind_curriculum_steps: int = 120_000
    wind_curriculum_min_scale: float = 0.0
    wind_curriculum_max_scale: float = 1.0
    wind_curriculum_zero_prob_start: float = 1.0
    wind_curriculum_zero_prob_end: float = 0.1

    # teacher-guided RL in the normalized action space [-1, 1]
    teacher_guidance_enabled: bool = False
    teacher_guidance_mode: str = "envelope"
    teacher_guidance_zero_actor_init: bool = False
    teacher_guidance_delta_init: float = 0.20
    teacher_guidance_delta_final: float = 2.0
    teacher_guidance_anneal_steps: int = 120_000
    teacher_guidance_schedule_steps: tuple[int, ...] = ()
    teacher_guidance_schedule_deltas: tuple[float, ...] = ()
    teacher_guidance_disable_after_steps: int = -1
    teacher_guidance_use_wind_truth: bool = True
    teacher_state_source: str = "truth"
    policy_state_source: str = "truth"
    imu_source: str = "synthetic"
    teacher_line_start_xy: tuple[float, float] = (0.0, 0.0)
    teacher_line_end_xy: tuple[float, float] = (120.0, 0.0)

    # action filtering (normalized action space [-1, 1])
    act_lpf_tau_s: float = 0.1  # 0: off; first-order low-pass time constant (s)
    act_rate_limit_per_s: float = 2.0  # 0: off; max |delta a| per second in normalized units

    # grouped frame stacking (per-signal)
    stack_gb: int = 6
    stack_ang: int = 6
    stack_vx: int = 6
    stack_lin: int = 4
    stack_z: int = 4
    stack_tail: int = 4
    stack_freq: int = 2

    freeze_steps_after_reset: int = 240  # keep bot at spawn pose for N physics steps (0 disables)

    # reset pose / initial conditions
    # Note: for x-forward, y-left, z-up, a negative rotation about +Y corresponds to a nose-up pitch.
    reset_pitch_deg: float = 8.0
    reset_flap_hz: float = 4.0
    # A small negative elevon pitch command helps counter the default wing pitching moment in open-loop rollouts.
    reset_elevon_pitch_deg: float = -18.0
    reset_forward_speed_mps: float | None = 8.0
    reset_rudder_deg: float = 0.0
    reset_elevon_roll_deg: float = 0.0

    # attitude targets for straight flight
    # Note: pitch is defined positive nose-down. A positive "nose-up" target corresponds to a *negative* pitch angle.
    pitch_cmd_deg: float = 10.0
    roll_cmd_deg: float = 0.0

    # reward weights (tuned for stable long-horizon flight learning)
    w_height: float = 0.45
    w_vx: float = 0.45
    w_att: float = 0.20
    w_vy: float = 0.08
    w_y: float = 0.05
    w_act: float = 0.01
    w_ang: float = 0.05
    # attitude shaping scales
    att_pitch_err_deg: float = 10.0
    att_roll_err_deg: float = 10.0

    # UI
    ui_window_class_type = None
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        eye=(0.0, -8.0, 4.0),
        lookat=(0.0, 0.0, 2.0),
    )

    # physics
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 240.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
    )

    # scene & robot
    scene: InteractiveSceneCfg = FlappingRoomSceneCfg(num_envs=256, env_spacing=5.0)
    robot: ArticulationCfg = FlappingBotCfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        # Keep gravity enabled; we decouple flapping inertial coupling via mass scaling instead.
        spawn=FlappingBotCfg.spawn.replace(
            rigid_props=FlappingBotCfg.spawn.rigid_props.replace(disable_gravity=False),
        ),
    )

    # joints
    controlled_joints: Tuple[str, ...] = (
        "left_wing",
        "right_wing",
        "rudder",
        "left_tail",
        "right_tail",
    )
    joint_limit_softness: float = 0.98

    # termination
    terminate_ground_height: float = 0.05
    terminate_tilt_deg: float = 75.0
    terminate_abs_y: float = 20.0

    # flapping frequency action mapping
    # Keep the initial action range fairly tight around typical trimmed conditions.
    # A very wide range makes early exploration extremely unstable.
    min_flap_hz: float = 2.0
    max_flap_hz: float = 5.0

    # joint actuation model
    # If True, directly write joint positions/velocities each physics step (kinematic override).
    # This decouples wing flapping kinematics from rigid-body reaction dynamics and improves stability for
    # aero-driven flight experiments.
    use_kinematic_joint_override: bool = True

    # dynamics decoupling (mass/inertia)
    # Reduce inertial coupling from moving wing/tail links by scaling their masses/inertias and (optionally)
    # redistributing the removed mass onto the base link to keep total mass roughly constant.
    override_appendage_masses: bool = True
    appendage_mass_scale: float = 0.0  # 0 => clamp to appendage_min_mass_kg
    appendage_min_mass_kg: float = 1.0e-4
    redistribute_removed_mass_to_base: bool = True

    # elevon/rudder command mapping (desired symmetric range, intersected with joint limits)
    # Keep commanded elevon authority near the URDF physical limit; the effective
    # command is still intersected with the softened joint limits at runtime.
    elevon_max_deg: float = 41.0
    rudder_max_deg: float = 25.0
    elevon_pitch_mix: float = 1.0
    elevon_roll_mix: float = 1.0
    elevon_trim_deg: float = 0.0
    # Symmetric elevon bias applied in the tail aerodynamic model (deg). This captures a fixed trim/incidence
    # offset without consuming the action range.
    tail_elevator_bias_deg: float = 0.0
    tail_horizontal_tail_incidence_bias_deg: float = 0.0
    tail_fixed_horizontal_effectiveness: float = 0.5
    tail_elevon_effectiveness: float = 1.2
    tail_elevon_alpha_limit_deg: float = 25.0
    tail_horizontal_tail_q_scale: float = 1.0
    base_body_com_override_x_m: float | None = -0.10
    total_mass_kg_override: float | None = None

    # virtual roll control (decoupled from the aerodynamic tail model)
    # Differential elevons now generate a physical roll moment, so this surrogate is disabled by default.
    # tau_x += gain * q_dyn * roll_deflection - damping * p
    virtual_roll_moment_gain: float = 0.0
    virtual_roll_moment_damping: float = 0.0
    # virtual pitch control (decoupled from visual model)
    # tau_y += gain * q_dyn * elevator_deflection - damping * q
    virtual_pitch_moment_gain: float = 0.0
    virtual_pitch_moment_damping: float = 0.0

    # simple wing QSM (Stage A)
    qsm_wings: FlappingQSMCfg = FlappingQSMCfg(
        wings=[
            WingQSMCfg(
                name="left_wing",
                joint_name="left_wing",
                hinge_axis_body=(1.0, 0.0, 0.0),
                lever_arm_body=(0.02, 0.18, 0.01),
                area=0.165624,
                lift_coefficient=1.2,
                drag_coefficient=0.18,
                effective_radius_fraction=0.75,
                hinge_damping=0.01,
            ),
            WingQSMCfg(
                name="right_wing",
                joint_name="right_wing",
                hinge_axis_body=(-1.0, 0.0, 0.0),
                lever_arm_body=(0.02, -0.18, 0.01),
                area=0.165624,
                lift_coefficient=1.2,
                drag_coefficient=0.18,
                effective_radius_fraction=0.75,
                hinge_damping=0.01,
            ),
        ],
        air_density=1.225,
    )

    # tail aero (fixed horizontal + left/right elevons + fixed vertical + rudder)
    tail_aero: TailAeroCfg = TailAeroCfg()

    # diagnostics: selectively apply aerodynamic components
    enable_wing_aero: bool = True
    enable_tail_aero: bool = True

    # DeLaurier wing model (Stage B)
    use_delaurier_wings: bool = False
    delaurier_wing_geom_csv: Path = Path("outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv")
    delaurier_num_strips: int = 80
    delaurier_min_airspeed: float = 0.5  # m/s clamp for stability
    delaurier_theta_w_deg: float = 0.0
    delaurier_params: DeLaurierParams = DeLaurierParams(
        alpha0_rad=0.0,
        eta_s=0.65,
        cd_cf=1.95,
        alpha_stall_min_rad=math.radians(-12.0),
        alpha_stall_max_rad=math.radians(12.0),
        xi=0.0,
        c_mac=0.0,
        nu=1.5e-5,
        cd_f=0.028,
    )
    delaurier_enable_separation: bool = False
    # Induced drag correction (simple Oswald efficiency model).
    # DeLaurier strip theory as used here does not include a finite-wing induced drag term, which can lead to
    # unrealistic positive chordwise force and runaway acceleration in free-flight simulations.
    # Set to 0 to disable.
    delaurier_induced_drag_efficiency: float = 0.0
    # Simple quadratic parasite drag applied at the base link (acts opposite body velocity).
    # Use as a stabilizing term to prevent unbounded acceleration when combined aero models are missing body drag.
    fuselage_drag_cda: float = 0.0  # m^2 effective Cd*A

    # Prescribed twist used with DeLaurier (qd_scaled proxy)
    twist_f_ref_hz: float = 4.0
    twist_eta_max_deg: float = 10.0
    twist_eta_limit_deg: float = 10.0
    twist_sign_left: float = 1.0
    twist_sign_right: float = 1.0


@configclass
class FlappingBotStraightFlightSimpleEnvCfg(FlappingBotStraightFlightEnvCfg):
    """Fast baseline: simple wings + tail aero."""

    use_delaurier_wings: bool = False


@configclass
class FlappingBotStraightFlightDeLaurierEnvCfg(FlappingBotStraightFlightEnvCfg):
    """Detailed wings: DeLaurier wings + tail aero."""

    use_delaurier_wings: bool = True


@configclass
class FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg(FlappingBotStraightFlightDeLaurierEnvCfg):
    """Teacher-guided RL defaults for DeLaurier straight-flight training."""

    teacher_guidance_enabled: bool = True
    teacher_guidance_delta_init: float = 0.15
    teacher_guidance_delta_final: float = 2.0
    teacher_guidance_anneal_steps: int = 160_000
    teacher_guidance_schedule_steps: tuple[int, ...] = (0, 20_000, 80_000, 160_000)
    teacher_guidance_schedule_deltas: tuple[float, ...] = (0.15, 0.25, 0.75, 2.0)
    teacher_guidance_disable_after_steps: int = -1

    wind_enabled: bool = True
    randomize_wind: bool = True
    wind_x_range_mps: tuple[float, float] = (-0.5, 0.5)
    wind_y_range_mps: tuple[float, float] = (-2.5, 2.5)
    wind_ou_enabled: bool = True
    wind_ou_tau_s: float = 2.0
    wind_ou_sigma_xy_mps: tuple[float, float] = (0.2, 0.8)
    wind_curriculum_enabled: bool = True
    wind_curriculum_steps: int = 160_000
    wind_curriculum_zero_prob_start: float = 1.0
    wind_curriculum_zero_prob_end: float = 0.15


@configclass
class FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg(FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg):
    """Weaker teacher schedule: policy gets a wider action envelope sooner."""

    teacher_guidance_schedule_steps: tuple[int, ...] = (0, 10_000, 30_000, 60_000)
    teacher_guidance_schedule_deltas: tuple[float, ...] = (0.35, 0.75, 1.5, 2.0)


@configclass
class FlappingBotStraightFlightDeLaurierPureRLEnvCfg(FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg):
    """Pure-RL continuation config with the same wind curriculum but no teacher envelope."""

    teacher_guidance_enabled: bool = False
    teacher_guidance_disable_after_steps: int = 0


class FlappingBotStraightFlightEnv(DirectRLEnv):
    cfg: FlappingBotStraightFlightEnvCfg

    def __init__(self, cfg: FlappingBotStraightFlightEnvCfg, render_mode: str | None = None, **kwargs):
        # runtime buffers
        self._robot: Articulation | None = None
        self._joint_ids: list[int] = []
        self._resolved_joint_names: list[str] = []
        self._joint_lower_limits: Tensor | None = None
        self._joint_upper_limits: Tensor | None = None
        self._default_joint_pos: Tensor | None = None
        self._joint_targets: Tensor | None = None
        self._actions: Tensor | None = None
        self._act_lpf: Tensor | None = None
        self._act_cmd: Tensor | None = None
        self._freeze_steps: Tensor | None = None
        self._spawn_root_state: Tensor | None = None
        self._mass_total: Tensor | None = None

        # wing phase and frequency
        self._phase: Tensor | None = None  # (N,)
        self._freq: Tensor | None = None  # (N,)

        # tail command buffers
        self._elevon_pitch_cmd: Tensor | None = None  # (N,)
        self._elevon_roll_cmd: Tensor | None = None  # (N,)
        self._left_elevon_cmd: Tensor | None = None  # (N,)
        self._right_elevon_cmd: Tensor | None = None  # (N,)
        self._elevator_cmd: Tensor | None = None  # (N,) equivalent symmetric elevon deflection for aero
        self._rudder_cmd: Tensor | None = None  # (N,)
        self._roll_cmd: Tensor | None = None  # (N,) equivalent differential elevon deflection for virtual roll moment

        # command buffers
        self._vx_cmd: Tensor | None = None
        self._height_cmd: Tensor | None = None
        self._wind_w: Tensor | None = None
        self._wind_mean_w: Tensor | None = None
        self._teacher_state_inputs: TeacherStateInputs | None = None
        self._resolved_imu_source: str | None = None
        self._teacher_controller: PX4LikeStraightLineController | None = None
        self._teacher_actions: Tensor | None = None
        self._teacher_action_gap_abs: Tensor | None = None
        self._teacher_delta: float | Tensor = 0.0
        self._wind_curriculum_scale: float = 0.0
        self._debug_last_teacher_actions: Tensor | None = None
        self._debug_last_teacher_diag: dict[str, Tensor] = {}
        self._debug_last_exec_action: Tensor | None = None
        self._debug_last_exec_freq_hz: Tensor | None = None
        self._debug_last_exec_rudder_rad: Tensor | None = None
        self._debug_last_exec_left_elevon_rad: Tensor | None = None
        self._debug_last_exec_right_elevon_rad: Tensor | None = None

        # indices
        self._IDX_LEFT_WING = None
        self._IDX_RIGHT_WING = None
        self._IDX_RUDDER = None
        self._IDX_LEFT_TAIL = None
        self._IDX_RIGHT_TAIL = None

        # wing kinematics cache (commanded)
        self._wing_amp: float = 0.0
        self._wing_mid_L: float = 0.0
        self._wing_mid_R: float = 0.0
        self._q_cmd: Tensor | None = None
        self._qd_cmd: Tensor | None = None
        self._qdd_cmd: Tensor | None = None

        # Debug caches (filled in _apply_action) for offline analysis and scripts.
        self._debug_last_wing_force_b: Tensor | None = None
        self._debug_last_tail_force_b: Tensor | None = None
        self._debug_last_force_b: Tensor | None = None
        self._debug_last_torque_b: Tensor | None = None

        # aero models
        self._qsm_wing_model: QuasiSteadyWingModel | None = None
        self._tail_model: TailAeroModel | None = None
        self._wing_geom: WingGeometry | None = None
        self._wing_geom_R: float | None = None
        self._wing_area: float | None = None
        self._delaurier_params: DeLaurierParams | None = None
        self._wing_body_ids: list[int] = []
        self._base_body_ids: list[int] = []
        self._A_w2l_batch: Tensor | None = None  # (2,3,3) for left/right
        self._wing_application_point_link: Tensor | None = None  # (2,3) for left/right

        super().__init__(cfg, render_mode, **kwargs)
        self._teacher_state_inputs = resolve_teacher_state_inputs(
            self.cfg.teacher_state_source,
            self.cfg.policy_state_source,
            self.cfg.teacher_guidance_use_wind_truth,
        )
        self._resolved_imu_source = resolve_imu_source(self.cfg.imu_source)

        # resolve joints available in current URDF
        available = list(self._robot.joint_names)
        req = [n for n in self.cfg.controlled_joints if n in available]
        if not req:
            raise RuntimeError("No controlled joints resolved; check URDF names.")
        joint_ids, joint_names = self._robot.find_joints(req, preserve_order=True)
        self._joint_ids = joint_ids
        self._resolved_joint_names = list(joint_names)
        name_to_idx = {n: i for i, n in enumerate(self._resolved_joint_names)}

        # limits (softened)
        jlim = self._robot.data.joint_pos_limits[0, self._joint_ids].to(device=self.device)
        lower, upper = jlim[:, 0], jlim[:, 1]
        span = upper - lower
        margin = (1.0 - self.cfg.joint_limit_softness) * span * 0.5
        self._joint_lower_limits = lower + margin
        self._joint_upper_limits = upper - margin
        self._default_joint_pos = self._robot.data.default_joint_pos[0, self._joint_ids].to(device=self.device)

        # action buffer
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._act_lpf = torch.zeros_like(self._actions)
        self._act_cmd = torch.zeros_like(self._actions)
        self._teacher_actions = torch.zeros_like(self._actions)
        self._teacher_action_gap_abs = torch.zeros_like(self._actions)
        self._debug_last_teacher_actions = torch.zeros_like(self._actions)
        self._debug_last_exec_action = torch.zeros_like(self._actions)
        self._debug_last_exec_freq_hz = torch.zeros(self.num_envs, device=self.device)
        self._debug_last_exec_rudder_rad = torch.zeros(self.num_envs, device=self.device)
        self._debug_last_exec_left_elevon_rad = torch.zeros(self.num_envs, device=self.device)
        self._debug_last_exec_right_elevon_rad = torch.zeros(self.num_envs, device=self.device)

        # grouped frame history buffers (N, K, D)
        N = self.num_envs
        self._hist_valid = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._hist_z = torch.zeros(N, self.cfg.stack_z, 1, device=self.device)
        self._hist_lin = torch.zeros(N, self.cfg.stack_lin, 3, device=self.device)
        self._hist_ang = torch.zeros(N, self.cfg.stack_ang, 3, device=self.device)
        self._hist_gb = torch.zeros(N, self.cfg.stack_gb, 3, device=self.device)
        self._hist_tail = torch.zeros(N, self.cfg.stack_tail, 2, device=self.device)
        self._hist_freq = torch.zeros(N, self.cfg.stack_freq, 1, device=self.device)
        self._hist_vx = torch.zeros(N, self.cfg.stack_vx, 1, device=self.device)
        self._freeze_steps = torch.zeros(N, dtype=torch.int32, device=self.device)
        self._spawn_root_state = torch.zeros(N, 13, device=self.device)

        # debug caches
        self._debug_last_wing_force_b = torch.zeros(N, 3, device=self.device)
        self._debug_last_tail_force_b = torch.zeros(N, 3, device=self.device)
        self._debug_last_force_b = torch.zeros(N, 3, device=self.device)
        self._debug_last_torque_b = torch.zeros(N, 3, device=self.device)
        self._mass_total = self._robot.data.default_mass.sum(dim=1).to(device=self.device)

        # joint targets
        self._joint_targets = self._default_joint_pos.expand(self.num_envs, -1).clone()
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

        # control indices
        self._IDX_LEFT_WING = name_to_idx.get("left_wing")
        self._IDX_RIGHT_WING = name_to_idx.get("right_wing")
        self._IDX_RUDDER = name_to_idx.get("rudder")
        self._IDX_LEFT_TAIL = name_to_idx.get("left_tail")
        self._IDX_RIGHT_TAIL = name_to_idx.get("right_tail")
        for idx, name in (
            (self._IDX_LEFT_WING, "left_wing"),
            (self._IDX_RIGHT_WING, "right_wing"),
            (self._IDX_RUDDER, "rudder"),
            (self._IDX_LEFT_TAIL, "left_tail"),
            (self._IDX_RIGHT_TAIL, "right_tail"),
        ):
            if idx is None:
                raise RuntimeError(f"Required joint '{name}' not found; update URDF or env mapping.")

        # base body ids used for applying net aerodynamic wrench
        base_body_ids, _ = self._robot.find_bodies(["base_link"], preserve_order=True)
        self._base_body_ids = base_body_ids
        self._override_appendage_mass_properties()
        self._override_total_mass_properties()
        self._override_base_body_com()

        # wing phase/frequency
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._freq = torch.full((self.num_envs,), float(self.cfg.min_flap_hz), device=self.device)

        # initialize commands
        self._vx_cmd = torch.full((self.num_envs,), float(self.cfg.vx_cmd), device=self.device)
        self._height_cmd = torch.full((self.num_envs,), float(self.cfg.height_cmd), device=self.device)
        self._wind_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._wind_mean_w = torch.zeros((self.num_envs, 3), device=self.device)
        if bool(self.cfg.wind_enabled) and not bool(self.cfg.randomize_wind):
            self._wind_mean_w[:, 0] = float(self.cfg.wind_xy_mps[0])
            self._wind_mean_w[:, 1] = float(self.cfg.wind_xy_mps[1])
        self._wind_w.copy_(self._wind_mean_w)

        # tail commands
        self._elevon_pitch_cmd = torch.zeros(self.num_envs, device=self.device)
        self._elevon_roll_cmd = torch.zeros(self.num_envs, device=self.device)
        self._left_elevon_cmd = torch.zeros(self.num_envs, device=self.device)
        self._right_elevon_cmd = torch.zeros(self.num_envs, device=self.device)
        self._elevator_cmd = torch.zeros(self.num_envs, device=self.device)
        self._rudder_cmd = torch.zeros(self.num_envs, device=self.device)
        self._roll_cmd = torch.zeros(self.num_envs, device=self.device)

        # wing amplitude (constant from joint limits)
        target_lower = torch.tensor(-math.radians(30.0), device=self.device)
        target_upper = torch.tensor(math.radians(30.0), device=self.device)
        l_lower = torch.maximum(self._joint_lower_limits[self._IDX_LEFT_WING], target_lower)
        l_upper = torch.minimum(self._joint_upper_limits[self._IDX_LEFT_WING], target_upper)
        r_lower = torch.maximum(self._joint_lower_limits[self._IDX_RIGHT_WING], target_lower)
        r_upper = torch.minimum(self._joint_upper_limits[self._IDX_RIGHT_WING], target_upper)
        mid_L = 0.5 * (l_lower + l_upper)
        mid_R = 0.5 * (r_lower + r_upper)
        amp = torch.minimum(0.5 * (l_upper - l_lower), 0.5 * (r_upper - r_lower))
        self._wing_mid_L = float(mid_L.item())
        self._wing_mid_R = float(mid_R.item())
        self._wing_amp = float(amp.item())

        # aero models
        tail_aero_cfg = replace(
            self.cfg.tail_aero,
            horizontal_tail_incidence_bias_deg=float(self.cfg.tail_horizontal_tail_incidence_bias_deg),
            fixed_horizontal_effectiveness=float(self.cfg.tail_fixed_horizontal_effectiveness),
            elevon_effectiveness=float(self.cfg.tail_elevon_effectiveness),
            elevon_alpha_limit_deg=float(self.cfg.tail_elevon_alpha_limit_deg),
            horizontal_tail_q_scale=float(self.cfg.tail_horizontal_tail_q_scale),
        )
        self._tail_model = TailAeroModel(tail_aero_cfg, self.device)
        if not bool(self.cfg.use_delaurier_wings):
            self._qsm_wing_model = QuasiSteadyWingModel(self.cfg.qsm_wings, self.device)
        else:
            # Build wing geometry for DeLaurier. Use chord distribution from CSV.
            self._wing_geom, info = build_wing_geometry_from_csv(
                Path(self.cfg.delaurier_wing_geom_csv),
                N=int(self.cfg.delaurier_num_strips),
                device=self.device,
                dtype=torch.float32,
                dhat=0.0,
                aspect_ratio=None,
            )
            self._wing_geom_R = float(info.R)
            # Planform area for induced-drag correction (one wing).
            self._wing_area = float(torch.sum(self._wing_geom.c * self._wing_geom.dx).item())
            self._delaurier_params = self.cfg.delaurier_params
            self._wing_application_point_link = compute_area_weighted_quarter_chord_link_points(self._wing_geom)
            # body ids for moment arms
            wing_body_ids, _ = self._robot.find_bodies(["left_wing", "right_wing"], preserve_order=True)
            self._wing_body_ids = wing_body_ids

            # Constant Wang axes mapping (same as wind-tunnel defaults): (wang_x, wang_y, wang_z) in link coords.
            # x: span -> link +y, y: normal -> link +z, z: chord -> link +x
            A_l2w_L = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=self.device)
            # Mirror span axis for the right wing (link +y is opposite span direction on mirrored side).
            A_l2w_R = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=self.device)
            self._A_w2l_batch = torch.stack((A_l2w_L.T, A_l2w_R.T), dim=0)  # (2,3,3)

        self._teacher_delta = float(self.cfg.teacher_guidance_delta_final)
        self._wind_curriculum_scale = 1.0 if bool(self.cfg.wind_enabled) else 0.0
        if bool(self.cfg.teacher_guidance_enabled):
            self._teacher_controller = PX4LikeStraightLineController(
                PX4LikeStraightLineControllerCfg(
                    line_start_xy=tuple(float(v) for v in self.cfg.teacher_line_start_xy),
                    line_end_xy=tuple(float(v) for v in self.cfg.teacher_line_end_xy),
                    wind_xy=(0.0, 0.0),
                    control_dt_s=float(self.step_dt),
                    height_sp_m=float(self.cfg.height_cmd),
                    pitch_trim_deg=float(self.cfg.pitch_cmd_deg),
                    freq_trim_hz=float(self.cfg.reset_flap_hz),
                    min_flap_hz=float(self.cfg.min_flap_hz),
                    max_flap_hz=float(self.cfg.max_flap_hz),
                    enable_tecs=True,
                    speed_sp_mps=float(self.cfg.vx_cmd),
                    initial_elevon_pitch_action=float(self.cfg.reset_elevon_pitch_deg)
                    / max(float(self.cfg.elevon_max_deg), 1.0e-6),
                    initial_elevon_roll_action=float(self.cfg.reset_elevon_roll_deg)
                    / max(float(self.cfg.elevon_max_deg), 1.0e-6),
                ),
                device=self.device,
            )

    def _override_appendage_mass_properties(self) -> None:
        """Optionally scale wing/tail masses to reduce rigid-body reaction torques from prescribed joint motion."""
        if self._mass_total is None:
            # Should be created in __init__, but guard to keep the method side-effect safe.
            self._mass_total = self._robot.data.default_mass.sum(dim=1).to(device=self.device)

        if not bool(self.cfg.override_appendage_masses):
            return

        # PhysX view APIs for masses/inertias expect CPU tensors.
        env_ids = torch.arange(self.num_envs, device="cpu", dtype=torch.int64)
        masses = self._robot.root_physx_view.get_masses().clone()  # (N,B) CPU
        inertias = self._robot.root_physx_view.get_inertias().clone()  # (N,B,9) CPU

        base_ids, _ = self._robot.find_bodies(["base_link"], preserve_order=True)
        wing_ids, _ = self._robot.find_bodies(["left_wing", "right_wing"], preserve_order=True)
        tail_ids, _ = self._robot.find_bodies(["left_tail", "right_tail", "rudder"], preserve_order=True)
        if len(base_ids) != 1:
            raise RuntimeError("Expected exactly one base_link body for mass override.")

        base_id = int(base_ids[0])
        appendage_ids = [int(i) for i in (wing_ids + tail_ids)]
        if len(appendage_ids) == 0:
            self._mass_total = masses.sum(dim=1).to(device=self.device)
            return

        scale = float(self.cfg.appendage_mass_scale)
        min_mass = float(self.cfg.appendage_min_mass_kg)

        masses_new = masses.clone()
        append_old = masses[:, appendage_ids]
        append_new = torch.clamp(append_old * scale, min=min_mass)
        masses_new[:, appendage_ids] = append_new

        if bool(self.cfg.redistribute_removed_mass_to_base):
            removed_sum = (append_old - append_new).sum(dim=1)
            masses_new[:, base_id] = masses[:, base_id] + removed_sum

        # Scale inertia tensors with the mass ratios (simple approximation).
        ratios = torch.ones_like(masses)
        ratios[:, appendage_ids] = append_new / torch.clamp(append_old, min=1.0e-9)
        ratios[:, base_id] = masses_new[:, base_id] / torch.clamp(masses[:, base_id], min=1.0e-9)
        inertias_new = inertias.clone()
        ids_all = appendage_ids + [base_id]
        inertias_new[:, ids_all] = inertias[:, ids_all] * ratios[:, ids_all].unsqueeze(-1)

        self._robot.root_physx_view.set_masses(masses_new, env_ids)
        self._robot.root_physx_view.set_inertias(inertias_new, env_ids)

        # Cache total mass on the simulation device (used for gravity compensation).
        self._mass_total = masses_new.sum(dim=1).to(device=self.device)

    def _override_base_body_com(self) -> None:
        """Optionally override the base-body COM x offset in the base-link frame."""
        override_x = self.cfg.base_body_com_override_x_m
        if override_x is None:
            return
        if len(self._base_body_ids) != 1:
            raise RuntimeError("Expected exactly one base_link body for COM override.")

        env_ids = torch.arange(self.num_envs, device="cpu", dtype=torch.int64)
        coms = self._robot.root_physx_view.get_coms().clone()
        base_id = int(self._base_body_ids[0])
        coms[:, base_id, 0] = float(override_x)
        self._robot.root_physx_view.set_coms(coms, env_ids)

    def _override_total_mass_properties(self) -> None:
        """Optionally scale all rigid-body masses/inertias to match a desired total vehicle mass."""
        target_mass_kg = self.cfg.total_mass_kg_override
        if target_mass_kg is None:
            return

        desired_mass = float(target_mass_kg)
        if desired_mass <= 0.0:
            raise ValueError("cfg.total_mass_kg_override must be positive when provided.")

        env_ids = torch.arange(self.num_envs, device="cpu", dtype=torch.int64)
        masses = self._robot.root_physx_view.get_masses().clone()
        inertias = self._robot.root_physx_view.get_inertias().clone()
        total_now = masses.sum(dim=1, keepdim=True)
        scale = desired_mass / torch.clamp(total_now, min=1.0e-9)
        masses_new = masses * scale
        inertias_new = inertias * scale.unsqueeze(-1)

        self._robot.root_physx_view.set_masses(masses_new, env_ids)
        self._robot.root_physx_view.set_inertias(inertias_new, env_ids)

        self._mass_total = masses_new.sum(dim=1).to(device=self.device)
        if hasattr(self._robot.data, "default_mass"):
            self._robot.data.default_mass = masses_new.to(device=self.device)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _get_wind_curriculum_scale(self) -> float:
        if not bool(self.cfg.wind_enabled):
            return 0.0
        if not bool(self.cfg.wind_curriculum_enabled):
            return 1.0
        return linear_anneal(
            int(self.common_step_counter),
            start=float(self.cfg.wind_curriculum_min_scale),
            end=float(self.cfg.wind_curriculum_max_scale),
            duration_steps=int(self.cfg.wind_curriculum_steps),
        )

    def _get_wind_zero_prob(self) -> float:
        if not bool(self.cfg.wind_curriculum_enabled):
            return 0.0
        return linear_anneal(
            int(self.common_step_counter),
            start=float(self.cfg.wind_curriculum_zero_prob_start),
            end=float(self.cfg.wind_curriculum_zero_prob_end),
            duration_steps=int(self.cfg.wind_curriculum_steps),
        )

    def _teacher_guidance_active(self) -> bool:
        return teacher_guidance_is_active(
            int(self.common_step_counter),
            enabled=bool(self.cfg.teacher_guidance_enabled),
            disable_after_steps=int(self.cfg.teacher_guidance_disable_after_steps),
        )

    def _get_teacher_delta(self) -> float:
        if not self._teacher_guidance_active():
            return float(self.cfg.teacher_guidance_delta_final)

        if len(self.cfg.teacher_guidance_schedule_steps) > 0 or len(self.cfg.teacher_guidance_schedule_deltas) > 0:
            if len(self.cfg.teacher_guidance_schedule_steps) != len(self.cfg.teacher_guidance_schedule_deltas):
                raise ValueError("teacher_guidance_schedule_steps and teacher_guidance_schedule_deltas must have the same length.")
            return piecewise_linear_anneal(
                int(self.common_step_counter),
                steps=tuple(int(v) for v in self.cfg.teacher_guidance_schedule_steps),
                values=tuple(float(v) for v in self.cfg.teacher_guidance_schedule_deltas),
            )

        return linear_anneal(
            int(self.common_step_counter),
            start=float(self.cfg.teacher_guidance_delta_init),
            end=float(self.cfg.teacher_guidance_delta_final),
            duration_steps=int(self.cfg.teacher_guidance_anneal_steps),
        )

    def _compute_teacher_actions(self) -> tuple[Tensor, dict[str, Tensor]]:
        if self._teacher_controller is None:
            raise RuntimeError("Teacher controller is not initialized.")

        pos_local = self._robot.data.root_pos_w - self.scene.env_origins
        ground_vel_local = self._robot.data.root_lin_vel_w
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel_body = self._robot.data.root_ang_vel_b
        state_inputs = resolve_teacher_state_inputs(
            self.cfg.teacher_state_source,
            self.cfg.policy_state_source,
            self.cfg.teacher_guidance_use_wind_truth,
        )
        self._teacher_state_inputs = state_inputs
        if state_inputs.teacher_uses_truth_wind:
            wind_xy = self._wind_w[:, 0:2]
        else:
            wind_xy = torch.zeros((self.num_envs, 2), device=self.device)

        return self._teacher_controller.compute_actions(
            pos_local=pos_local,
            ground_vel_local=ground_vel_local,
            wind_vel_local=wind_xy,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            ang_vel_body=ang_vel_body,
        )

    def _update_wind_process(self) -> None:
        """Update world-frame wind state, optionally with OU gust dynamics."""
        assert self._wind_w is not None
        assert self._wind_mean_w is not None

        self._wind_curriculum_scale = self._get_wind_curriculum_scale()

        if not bool(self.cfg.wind_enabled):
            self._wind_mean_w.zero_()
            self._wind_w.zero_()
            return

        # Keep deterministic mean (constant or sampled at reset) when OU is disabled.
        if not bool(self.cfg.wind_ou_enabled):
            self._wind_w.copy_(self._wind_mean_w)
            self._wind_w[:, 2] = 0.0
            return

        tau = max(float(self.cfg.wind_ou_tau_s), 1.0e-3)
        alpha = math.exp(-float(self.step_dt) / tau)
        noise_scale = math.sqrt(max(1.0 - alpha * alpha, 0.0))
        sigma_x = self._wind_curriculum_scale * max(float(self.cfg.wind_ou_sigma_xy_mps[0]), 0.0)
        sigma_y = self._wind_curriculum_scale * max(float(self.cfg.wind_ou_sigma_xy_mps[1]), 0.0)
        sigma_xy = torch.tensor((sigma_x, sigma_y), device=self.device).view(1, 2)

        self._wind_w[:, 0:2] = self._wind_mean_w[:, 0:2] + alpha * (self._wind_w[:, 0:2] - self._wind_mean_w[:, 0:2])
        if noise_scale > 0.0 and (sigma_x > 0.0 or sigma_y > 0.0):
            noise = torch.randn((self.num_envs, 2), device=self.device)
            self._wind_w[:, 0:2] += noise * (noise_scale * sigma_xy)
        self._wind_w[:, 2] = 0.0

        if bool(self.cfg.wind_ou_clip_to_range):
            x_low, x_high = sorted(
                (
                    self._wind_curriculum_scale * float(self.cfg.wind_x_range_mps[0]),
                    self._wind_curriculum_scale * float(self.cfg.wind_x_range_mps[1]),
                )
            )
            y_low, y_high = sorted(
                (
                    self._wind_curriculum_scale * float(self.cfg.wind_y_range_mps[0]),
                    self._wind_curriculum_scale * float(self.cfg.wind_y_range_mps[1]),
                )
            )
            self._wind_w[:, 0] = torch.clamp(self._wind_w[:, 0], min=x_low, max=x_high)
            self._wind_w[:, 1] = torch.clamp(self._wind_w[:, 1], min=y_low, max=y_high)

    def _pre_physics_step(self, actions: Tensor):
        self._update_wind_process()
        # raw actions in [-1, 1]
        self._actions = actions.clamp(-1.0, 1.0)
        act_exec = self._actions
        teacher_active = self._teacher_guidance_active()
        teacher_guidance_mode = resolve_teacher_guidance_mode(self.cfg.teacher_guidance_mode)
        teacher_diag: dict[str, Tensor] = {}
        if teacher_active:
            self._teacher_delta = self._get_teacher_delta()
            teacher_actions, teacher_diag = self._compute_teacher_actions()
            self._teacher_actions.copy_(teacher_actions)
            act_exec = apply_teacher_guided_actions(
                self._teacher_actions,
                self._actions,
                delta=self._teacher_delta,
                mode=teacher_guidance_mode,
            )
            teacher_requested_gap_abs = torch.abs(self._actions - self._teacher_actions)
            teacher_exec_gap_abs = torch.abs(act_exec - self._teacher_actions)
            if teacher_guidance_mode == "residual":
                self._teacher_action_gap_abs.copy_(teacher_exec_gap_abs)
            else:
                self._teacher_action_gap_abs.copy_(teacher_requested_gap_abs)
            teacher_freq_hz = float(teacher_diag["freq_hz"].mean().item()) if "freq_hz" in teacher_diag else float("nan")
        else:
            self._teacher_delta = float(self.cfg.teacher_guidance_delta_final)
            self._teacher_actions.zero_()
            self._teacher_action_gap_abs.zero_()
            teacher_requested_gap_abs = torch.zeros_like(self._teacher_action_gap_abs)
            teacher_exec_gap_abs = torch.zeros_like(self._teacher_action_gap_abs)
            teacher_freq_hz = float("nan")
        if self._debug_last_teacher_actions is not None:
            self._debug_last_teacher_actions.copy_(self._teacher_actions)
        self._debug_last_teacher_diag = {
            key: value.detach().clone()
            for key, value in teacher_diag.items()
            if torch.is_tensor(value)
        }
        # low-pass filter
        if self.cfg.act_lpf_tau_s > 0.0:
            alpha = float(self.step_dt) / (self.cfg.act_lpf_tau_s + float(self.step_dt))
            self._act_lpf = self._act_lpf + alpha * (act_exec - self._act_lpf)
        else:
            self._act_lpf = act_exec
        # slew-rate limit
        if self.cfg.act_rate_limit_per_s > 0.0:
            max_delta = self.cfg.act_rate_limit_per_s * float(self.step_dt)
            delta = torch.clamp(self._act_lpf - self._act_cmd, min=-max_delta, max=max_delta)
            self._act_cmd = self._act_cmd + delta
        else:
            self._act_cmd = self._act_lpf

        # frequency from action 0 in [min_flap_hz, max_flap_hz]
        a0 = self._act_cmd[:, 0]
        f = 0.5 * (a0 + 1.0) * (self.cfg.max_flap_hz - self.cfg.min_flap_hz) + self.cfg.min_flap_hz
        self._freq = torch.clamp(f, min=float(self.cfg.min_flap_hz), max=float(self.cfg.max_flap_hz))

        # rudder (action 1) -> virtual rudder channel
        rud_lim = torch.deg2rad(torch.tensor(float(self.cfg.rudder_max_deg), device=self.device))
        self._rudder_cmd = (rud_lim * self._act_cmd[:, 1]).clamp(-rud_lim, rud_lim)

        elevon_lim = torch.deg2rad(torch.tensor(float(self.cfg.elevon_max_deg), device=self.device))
        self._elevon_pitch_cmd = (elevon_lim * self._act_cmd[:, 2]).clamp(-elevon_lim, elevon_lim)
        self._elevon_roll_cmd = (elevon_lim * self._act_cmd[:, 3]).clamp(-elevon_lim, elevon_lim)

        trim = torch.deg2rad(torch.tensor(float(self.cfg.elevon_trim_deg), device=self.device))
        mixed_pitch = float(self.cfg.elevon_pitch_mix) * self._elevon_pitch_cmd
        mixed_roll = float(self.cfg.elevon_roll_mix) * self._elevon_roll_cmd
        left_raw = trim + mixed_pitch + mixed_roll
        right_raw = trim + mixed_pitch - mixed_roll

        self.extras["log"] = {
            "Teacher/delta": float(self._teacher_delta.mean().item()) if torch.is_tensor(self._teacher_delta) else float(self._teacher_delta),
            "Teacher/mean_abs_gap": float(self._teacher_action_gap_abs.mean().item()),
            "Teacher/mean_requested_abs_gap": float(teacher_requested_gap_abs.mean().item()),
            "Teacher/mean_exec_abs_gap": float(teacher_exec_gap_abs.mean().item()),
            "Teacher/enabled": float(bool(self.cfg.teacher_guidance_enabled)),
            "Teacher/active": float(bool(teacher_active)),
            "Teacher/mode_residual": float(teacher_guidance_mode == "residual"),
            "Teacher/freq_hz": teacher_freq_hz,
            "Wind/curriculum_scale": float(self._wind_curriculum_scale),
            "Wind/mean_x_mps": float(self._wind_mean_w[:, 0].mean().item()),
            "Wind/mean_y_mps": float(self._wind_mean_w[:, 1].mean().item()),
        }

        l_lower = torch.maximum(self._joint_lower_limits[self._IDX_LEFT_TAIL], -elevon_lim)
        l_upper = torch.minimum(self._joint_upper_limits[self._IDX_LEFT_TAIL], elevon_lim)
        r_lower = torch.maximum(self._joint_lower_limits[self._IDX_RIGHT_TAIL], -elevon_lim)
        r_upper = torch.minimum(self._joint_upper_limits[self._IDX_RIGHT_TAIL], elevon_lim)
        self._left_elevon_cmd = left_raw.clamp(l_lower, l_upper)
        self._right_elevon_cmd = right_raw.clamp(r_lower, r_upper)

        self._elevator_cmd = 0.5 * (self._left_elevon_cmd + self._right_elevon_cmd)
        self._roll_cmd = 0.5 * (self._left_elevon_cmd - self._right_elevon_cmd)
        if self._debug_last_exec_action is not None:
            self._debug_last_exec_action.copy_(self._act_cmd)
        if self._debug_last_exec_freq_hz is not None:
            self._debug_last_exec_freq_hz.copy_(self._freq)
        if self._debug_last_exec_rudder_rad is not None:
            self._debug_last_exec_rudder_rad.copy_(self._rudder_cmd)
        if self._debug_last_exec_left_elevon_rad is not None:
            self._debug_last_exec_left_elevon_rad.copy_(self._left_elevon_cmd)
        if self._debug_last_exec_right_elevon_rad is not None:
            self._debug_last_exec_right_elevon_rad.copy_(self._right_elevon_cmd)

    def _apply_action(self):
        # advance phase (per-physics step)
        two_pi = 6.283185307179586
        self._phase = advance_flap_phase(
            phase=self._phase,
            freq_hz=self._freq,
            physics_dt_s=float(self.physics_dt),
            freeze_steps=self._freeze_steps,
        )

        # commanded wing joint targets
        # Use a cosine waveform so that phase=0 starts at max deflection with zero velocity.
        c = torch.cos(self._phase)
        s = torch.sin(self._phase)
        flap_off = self._freq <= 1.0e-3
        if flap_off.any():
            c = c.clone()
            s = s.clone()
            c[flap_off] = 0.0
            s[flap_off] = 0.0
        amp = float(self._wing_amp)
        # cache commanded wing kinematics (for DeLaurier backend)
        w = two_pi * self._freq
        self._q_cmd = amp * c
        self._qd_cmd = -amp * w * s
        self._qdd_cmd = -amp * (w * w) * c

        left_cmd = self._wing_mid_L + self._q_cmd
        right_cmd = self._wing_mid_R + self._q_cmd

        jt = self._joint_targets
        jt[:, self._IDX_LEFT_WING] = left_cmd
        jt[:, self._IDX_RIGHT_WING] = right_cmd
        jt[:, self._IDX_RUDDER] = self._rudder_cmd
        # visualization joints for tail (treated as left/right elevons)
        jt[:, self._IDX_LEFT_TAIL] = self._left_elevon_cmd
        jt[:, self._IDX_RIGHT_TAIL] = self._right_elevon_cmd
        if bool(self.cfg.use_kinematic_joint_override):
            jvel = torch.zeros_like(jt)
            self._robot.write_joint_state_to_sim(jt, jvel, joint_ids=self._joint_ids)
        else:
            self._robot.set_joint_position_target(jt, joint_ids=self._joint_ids)

        # hold root pose for a few steps after reset to avoid immediate drop
        if self.cfg.freeze_steps_after_reset > 0:
            hold_ids = torch.nonzero(self._freeze_steps > 0, as_tuple=False).squeeze(-1)
            if hold_ids.numel() > 0:
                self._robot.write_root_state_to_sim(self._spawn_root_state[hold_ids], env_ids=hold_ids)
                self._freeze_steps[hold_ids] = torch.clamp_min(self._freeze_steps[hold_ids] - 1, 0)

        # aerodynamic wrench: wings + tail -> base body
        assert self._wind_w is not None
        quat_w = self._robot.data.root_quat_w
        v_b = self._robot.data.root_lin_vel_b
        w_b = self._robot.data.root_ang_vel_b
        wind_b = quat_apply_inverse(quat_w, self._wind_w)
        v_air_b = v_b - wind_b

        if bool(self.cfg.enable_tail_aero):
            # tail (deflection-based)
            ele_bias = math.radians(float(self.cfg.tail_elevator_bias_deg))
            base_body_com_pos_b = self._robot.data.body_com_pos_b[:, int(self._base_body_ids[0]), :]
            f_tail, tau_tail = self._tail_model.compute_wrench(
                root_lin_vel_b=v_air_b,
                root_ang_vel_b=w_b,
                left_elevon_rad=self._left_elevon_cmd + ele_bias,
                right_elevon_rad=self._right_elevon_cmd + ele_bias,
                rudder_rad=self._rudder_cmd,
                base_com_pos_b=base_body_com_pos_b,
            )
            speed = torch.linalg.norm(v_air_b, dim=1)
            q_dyn = 0.5 * float(self.cfg.qsm_wings.air_density) * (speed * speed)
            tau_roll_virtual = (
                float(self.cfg.virtual_roll_moment_gain) * q_dyn * self._roll_cmd
                - float(self.cfg.virtual_roll_moment_damping) * w_b[:, 0]
            )
            tau_pitch_virtual = (
                float(self.cfg.virtual_pitch_moment_gain) * q_dyn * self._elevator_cmd
                - float(self.cfg.virtual_pitch_moment_damping) * w_b[:, 1]
            )
            tau_tail = tau_tail.clone()
            tau_tail[:, 0] += tau_roll_virtual
            tau_tail[:, 1] += tau_pitch_virtual
        else:
            f_tail = torch.zeros_like(v_b)
            tau_tail = torch.zeros_like(v_b)

        if bool(self.cfg.enable_wing_aero):
            if not bool(self.cfg.use_delaurier_wings):
                # simple wing QSM
                jpos_all = self._robot.data.joint_pos[:, self._joint_ids]
                jvel_all = self._robot.data.joint_vel[:, self._joint_ids]
                jpos = torch.stack((jpos_all[:, self._IDX_LEFT_WING], jpos_all[:, self._IDX_RIGHT_WING]), dim=1)
                jvel = torch.stack((jvel_all[:, self._IDX_LEFT_WING], jvel_all[:, self._IDX_RIGHT_WING]), dim=1)
                f_w, tau_w, _ = self._qsm_wing_model.compute_forces(jpos, jvel, v_air_b, w_b)
                f_w_sum = torch.sum(f_w, dim=1)
                tau_w_sum = torch.sum(tau_w, dim=1)
            else:
                f_w_sum, tau_w_sum = self._compute_wing_delaurier_wrench(v_air_b)
        else:
            f_w_sum = torch.zeros_like(v_b)
            tau_w_sum = torch.zeros_like(v_b)

        # Quadratic parasite drag at the base (opposes air-relative body velocity).
        f_drag = torch.zeros_like(v_b)
        cda = float(self.cfg.fuselage_drag_cda)
        if cda > 0.0:
            rho = float(self.cfg.qsm_wings.air_density)
            speed = torch.linalg.norm(v_air_b, dim=1, keepdim=True)
            f_drag = -0.5 * rho * cda * speed * v_air_b

        # debug caches (body frame)
        self._debug_last_wing_force_b.copy_(f_w_sum)
        self._debug_last_tail_force_b.copy_(f_tail)
        self._debug_last_force_b.copy_(f_w_sum + f_tail + f_drag)
        self._debug_last_torque_b.copy_(tau_w_sum + tau_tail)

        f_sum = (f_w_sum + f_tail + f_drag).unsqueeze(1)  # (N,1,3)
        t_sum = (tau_w_sum + tau_tail).unsqueeze(1)  # (N,1,3)
        self._robot.set_external_force_and_torque(
            forces=f_sum, torques=t_sum, body_ids=self._base_body_ids, is_global=False
        )

    def _compute_wing_delaurier_wrench(self, v_air_b: Tensor) -> tuple[Tensor, Tensor]:
        """Compute net wing wrench about the base in the body frame (DeLaurier backend)."""
        assert self._wing_geom is not None
        assert self._wing_area is not None
        assert self._delaurier_params is not None
        assert self._A_w2l_batch is not None
        assert self._wing_application_point_link is not None
        assert self._q_cmd is not None and self._qd_cmd is not None and self._qdd_cmd is not None

        # Batch with two wings per env: (env0_L, env0_R, env1_L, env1_R, ...)
        N_env = self.num_envs
        B = 2 * N_env
        N_strip = int(self._wing_geom.x_mid.numel())
        y = self._wing_geom.x_mid.view(1, N_strip).expand(B, N_strip)

        q = torch.repeat_interleave(self._q_cmd, 2)  # (B,)
        qd = torch.repeat_interleave(self._qd_cmd, 2)
        qdd = torch.repeat_interleave(self._qdd_cmd, 2)
        w = torch.repeat_interleave(2.0 * torch.pi * self._freq, 2)  # (B,)

        # theta_a (flapping-axis angle relative to the freestream) per env -> per wing.
        # In free-flight we approximate this as the body-frame angle-of-attack based on the velocity vector.
        # This matches the wind-tunnel convention when the vehicle is trimmed (v_z small, pitch≈flight-path angle),
        # and avoids large sign errors during dives/climbs where pitch!=AOA.
        quat_w = self._robot.data.root_quat_w  # (N,4)
        vx_b = torch.clamp(v_air_b[:, 0], min=1.0e-3)
        theta_a_env = torch.atan2(-v_air_b[:, 2], vx_b)  # +theta_a => nose-up relative wind
        theta_a = torch.repeat_interleave(theta_a_env, 2)  # (B,)
        theta_bar = theta_a + math.radians(float(self.cfg.delaurier_theta_w_deg))

        # Airspeed approximation: body-forward velocity component clamped (matches wind-tunnel trim scans).
        vx = torch.clamp(v_air_b[:, 0], min=float(self.cfg.delaurier_min_airspeed))
        U = torch.repeat_interleave(vx, 2)  # (B,)

        h = -q.view(B, 1) * y
        hdot = -qd.view(B, 1) * y
        hddot = -qdd.view(B, 1) * y

        # Prescribed twist: qd_scaled proxy (full-span uniform)
        eta_max = torch.deg2rad(torch.tensor(float(self.cfg.twist_eta_max_deg), device=self.device))
        eta_lim = torch.deg2rad(torch.tensor(float(self.cfg.twist_eta_limit_deg), device=self.device))
        f_ref = float(self.cfg.twist_f_ref_hz)
        qd_ref = float(self._wing_amp) * (2.0 * math.pi * f_ref)
        qd_ref = max(qd_ref, 1e-6)
        s = (qd / qd_ref).clamp(-1.0, 1.0)
        # apply per-wing sign convention
        sgn_L = float(self.cfg.twist_sign_left)
        sgn_R = float(self.cfg.twist_sign_right)
        sgn = torch.tensor([sgn_L, sgn_R], device=self.device, dtype=torch.float32).repeat(N_env)
        eta_tip = (eta_max * sgn * s).clamp(-eta_lim, eta_lim)
        k = eta_max / qd_ref
        qddd = -(w * w) * qd
        etad_tip = (k * sgn) * qdd
        etadd_tip = (k * sgn) * qddd

        theta = (theta_bar + eta_tip).view(B, 1).expand(B, N_strip)
        thetad = etad_tip.view(B, 1).expand(B, N_strip)
        thetadd = etadd_tip.view(B, 1).expand(B, N_strip)

        omega_ref = w.view(B, 1).expand(B, N_strip)

        F_c, _tau_c, _p_in, _sep = compute_aero_wrench_delaurier1993(
            h,
            hdot,
            hddot,
            theta,
            thetad,
            thetadd,
            self._wing_geom,
            rho=float(self.cfg.qsm_wings.air_density),
            U=U.view(B, 1),
            theta_a=theta_a.view(B, 1),
            theta_bar=theta_bar.view(B, 1),
            omega_ref=omega_ref,
            params=self._delaurier_params,
            enable_separation=bool(self.cfg.delaurier_enable_separation),
            return_terms=False,
        )  # (B,3) in Wang co-rotating frame

        # Wang->link for each wing (repeat per env)
        A_w2l = self._A_w2l_batch.repeat(N_env, 1, 1)  # (B,3,3)
        F_l = torch.bmm(A_w2l, F_c.view(B, 3, 1)).view(B, 3)

        # Link->world
        q_w_link = self._robot.data.body_quat_w[:, self._wing_body_ids, :].reshape(B, 4)
        p_wing_origin_w = self._robot.data.body_pos_w[:, self._wing_body_ids, :].reshape(B, 3)
        wing_application_point_link = self._wing_application_point_link.repeat(N_env, 1)
        p_wing_w = p_wing_origin_w + quat_apply(q_w_link, wing_application_point_link)
        F_w = quat_apply(q_w_link, F_l)

        # World wrench about the base rigid-body COM.
        p_base_w = self._robot.data.root_com_pos_w
        r_w = p_wing_w - torch.repeat_interleave(p_base_w, 2, dim=0)
        tau_w = torch.linalg.cross(r_w, F_w)
        F_w_sum = F_w.view(N_env, 2, 3).sum(dim=1)
        tau_w_sum = tau_w.view(N_env, 2, 3).sum(dim=1)

        # World->body
        F_b = quat_apply_inverse(quat_w, F_w_sum)
        tau_b = quat_apply_inverse(quat_w, tau_w_sum)

        # Induced drag correction (finite wing): D_i = L^2 / (q S pi AR e), applied opposite air-relative velocity.
        e = float(self.cfg.delaurier_induced_drag_efficiency)
        if e > 0.0:
            AR = float(self._wing_geom.aspect_ratio)
            S = float(self._wing_area)
            if AR > 1.0e-6 and S > 1.0e-9:
                speed = torch.linalg.norm(v_air_b, dim=1)
                speed_safe = torch.clamp(speed, min=1.0e-6)
                v_dir = v_air_b / speed_safe.unsqueeze(1)
                q_dyn = 0.5 * float(self.cfg.qsm_wings.air_density) * (speed_safe * speed_safe)

                # Lift magnitude: component of the wing force perpendicular to velocity direction.
                f_para_mag = torch.sum(F_b * v_dir, dim=1)
                f_para = f_para_mag.unsqueeze(1) * v_dir
                f_perp = F_b - f_para
                lift_mag = torch.linalg.norm(f_perp, dim=1)

                cd_k = 1.0 / (math.pi * AR * e)
                denom = torch.clamp(q_dyn * S, min=1.0e-6)
                D_i = cd_k * (lift_mag * lift_mag) / denom
                F_b = F_b - D_i.unsqueeze(1) * v_dir

        return F_b, tau_b

    # ------------------------------------------------------------------
    # Reset / Observations / Rewards / Dones
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Tensor | list[int]):
        if isinstance(env_ids, list):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        # commands: randomize or set defaults per env
        if self.cfg.randomize_commands:
            vl, vh = self.cfg.vx_cmd_range
            zl, zh = self.cfg.height_cmd_range
            self._vx_cmd[env_ids] = torch.rand_like(self._vx_cmd[env_ids]) * (vh - vl) + vl
            self._height_cmd[env_ids] = torch.rand_like(self._height_cmd[env_ids]) * (zh - zl) + zl
        else:
            self._vx_cmd[env_ids] = float(self.cfg.vx_cmd)
            self._height_cmd[env_ids] = float(self.cfg.height_cmd)
        # wind mean: world-frame constant over each episode; optional OU gusts evolve around it.
        assert self._wind_w is not None
        assert self._wind_mean_w is not None
        if not bool(self.cfg.wind_enabled):
            self._wind_mean_w[env_ids] = 0.0
        elif bool(self.cfg.randomize_wind):
            scale = self._get_wind_curriculum_scale()
            zero_prob = self._get_wind_zero_prob()
            xl = scale * float(self.cfg.wind_x_range_mps[0])
            xh = scale * float(self.cfg.wind_x_range_mps[1])
            yl = scale * float(self.cfg.wind_y_range_mps[0])
            yh = scale * float(self.cfg.wind_y_range_mps[1])
            self._wind_mean_w[env_ids, 0] = torch.rand_like(self._vx_cmd[env_ids]) * (xh - xl) + xl
            self._wind_mean_w[env_ids, 1] = torch.rand_like(self._vx_cmd[env_ids]) * (yh - yl) + yl
            self._wind_mean_w[env_ids, 2] = 0.0
            if zero_prob > 0.0:
                zero_mask = torch.rand((env_ids.shape[0],), device=self.device) < zero_prob
                self._wind_mean_w[env_ids[zero_mask], 0:2] = 0.0
        else:
            scale = self._get_wind_curriculum_scale()
            self._wind_mean_w[env_ids, 0] = scale * float(self.cfg.wind_xy_mps[0])
            self._wind_mean_w[env_ids, 1] = scale * float(self.cfg.wind_xy_mps[1])
            self._wind_mean_w[env_ids, 2] = 0.0
        self._wind_w[env_ids] = self._wind_mean_w[env_ids]

        # root state: spawn at local (0,0,height_cmd) relative to env origin, with initial vx ~ vx_cmd
        n = env_ids.shape[0]
        env_origins = self.scene.env_origins[env_ids]
        pos = env_origins.clone()
        pos[:, 2] += self._height_cmd[env_ids]
        # Isaac uses a right-handed convention: positive rotation about +Y pitches the nose *down*.
        # Users typically specify +pitch as "nose up", so we negate it here to match the wind-tunnel scripts.
        pitch = -torch.deg2rad(torch.full((n,), float(self.cfg.reset_pitch_deg), device=self.device))
        rot = quat_from_euler_xyz(
            roll=torch.zeros_like(pitch),
            pitch=pitch,
            yaw=torch.zeros_like(pitch),
        )
        lin_vel = torch.zeros(n, 3, device=self.device)
        reset_forward_speed = self._vx_cmd[env_ids]
        if self.cfg.reset_forward_speed_mps is not None:
            reset_forward_speed = torch.full(
                (n,),
                float(self.cfg.reset_forward_speed_mps),
                device=self.device,
            )
        lin_vel[:, 0] = reset_forward_speed
        ang_vel = torch.zeros(n, 3, device=self.device)
        root_state = torch.cat([pos, rot, lin_vel, ang_vel], dim=1)
        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._spawn_root_state[env_ids] = root_state

        # joints to default and zero velocity
        jpos = self._default_joint_pos.expand(n, -1).clone()
        # initialize tail joints from mixed reset elevon commands
        elevon_lim = torch.deg2rad(torch.tensor(float(self.cfg.elevon_max_deg), device=self.device))
        rudder_lim = torch.deg2rad(torch.tensor(float(self.cfg.rudder_max_deg), device=self.device))
        rudder_lower = torch.maximum(self._joint_lower_limits[self._IDX_RUDDER], -rudder_lim)
        rudder_upper = torch.minimum(self._joint_upper_limits[self._IDX_RUDDER], rudder_lim)
        rudder0 = torch.full((n,), math.radians(float(self.cfg.reset_rudder_deg)), device=self.device).clamp(
            rudder_lower, rudder_upper
        )
        l_lower = torch.maximum(self._joint_lower_limits[self._IDX_LEFT_TAIL], -elevon_lim)
        l_upper = torch.minimum(self._joint_upper_limits[self._IDX_LEFT_TAIL], elevon_lim)
        r_lower = torch.maximum(self._joint_lower_limits[self._IDX_RIGHT_TAIL], -elevon_lim)
        r_upper = torch.minimum(self._joint_upper_limits[self._IDX_RIGHT_TAIL], elevon_lim)
        trim0 = math.radians(float(self.cfg.elevon_trim_deg))
        pit0 = math.radians(float(self.cfg.reset_elevon_pitch_deg))
        rol0 = math.radians(float(self.cfg.reset_elevon_roll_deg))
        left0 = trim0 + float(self.cfg.elevon_pitch_mix) * pit0 + float(self.cfg.elevon_roll_mix) * rol0
        right0 = trim0 + float(self.cfg.elevon_pitch_mix) * pit0 - float(self.cfg.elevon_roll_mix) * rol0
        left0 = torch.full((n,), left0, device=self.device).clamp(l_lower, l_upper)
        right0 = torch.full((n,), right0, device=self.device).clamp(r_lower, r_upper)
        jpos[:, self._IDX_RUDDER] = rudder0
        jpos[:, self._IDX_LEFT_TAIL] = left0
        jpos[:, self._IDX_RIGHT_TAIL] = right0
        jvel = torch.zeros_like(jpos)
        self._robot.write_joint_state_to_sim(jpos, jvel, joint_ids=self._joint_ids, env_ids=env_ids)

        # clear actions and phases
        self._actions[env_ids] = 0.0
        self._act_lpf[env_ids] = 0.0
        self._act_cmd[env_ids] = 0.0
        self._phase[env_ids] = 0.0
        # start near trim frequency to avoid immediate drop before the policy stabilizes
        f0 = float(self.cfg.reset_flap_hz)
        f0 = max(float(self.cfg.min_flap_hz), min(float(self.cfg.max_flap_hz), f0))
        self._freq[env_ids] = f0
        # initialize action history to match the reset frequency so action filtering doesn't create a large transient
        a0 = 2.0 * (f0 - float(self.cfg.min_flap_hz)) / (float(self.cfg.max_flap_hz) - float(self.cfg.min_flap_hz)) - 1.0
        self._actions[env_ids, 0] = a0
        self._act_lpf[env_ids, 0] = a0
        self._act_cmd[env_ids, 0] = a0
        # initialize action history for tail channels to avoid large transients
        rud_norm = float(self.cfg.reset_rudder_deg) / max(float(self.cfg.rudder_max_deg), 1.0e-6)
        ele_norm = float(self.cfg.reset_elevon_pitch_deg) / max(float(self.cfg.elevon_max_deg), 1.0e-6)
        rol_norm = float(self.cfg.reset_elevon_roll_deg) / max(float(self.cfg.elevon_max_deg), 1.0e-6)
        self._actions[env_ids, 1] = max(-1.0, min(1.0, rud_norm))
        self._act_lpf[env_ids, 1] = self._actions[env_ids, 1]
        self._act_cmd[env_ids, 1] = self._actions[env_ids, 1]
        self._actions[env_ids, 2] = max(-1.0, min(1.0, ele_norm))
        self._act_lpf[env_ids, 2] = self._actions[env_ids, 2]
        self._act_cmd[env_ids, 2] = self._actions[env_ids, 2]
        self._actions[env_ids, 3] = max(-1.0, min(1.0, rol_norm))
        self._act_lpf[env_ids, 3] = self._actions[env_ids, 3]
        self._act_cmd[env_ids, 3] = self._actions[env_ids, 3]

        # initialize tail commands
        self._left_elevon_cmd[env_ids] = left0
        self._right_elevon_cmd[env_ids] = right0
        self._elevon_pitch_cmd[env_ids] = torch.full((n,), pit0, device=self.device)
        self._elevon_roll_cmd[env_ids] = torch.full((n,), rol0, device=self.device)
        self._elevator_cmd[env_ids] = 0.5 * (left0 + right0)
        self._rudder_cmd[env_ids] = math.radians(float(self.cfg.reset_rudder_deg))
        self._roll_cmd[env_ids] = 0.5 * (left0 - right0)
        if self._teacher_controller is not None:
            self._teacher_controller.reset(env_ids)
        self._hist_valid[env_ids] = False
        self._freeze_steps[env_ids] = int(self.cfg.freeze_steps_after_reset)

    def _get_observations(self) -> dict[str, Tensor]:
        pos_w = self._robot.data.root_pos_w - self.scene.env_origins
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        g_b = self._robot.data.projected_gravity_b
        jpos = self._robot.data.joint_pos[:, self._joint_ids]

        # normalized/scaled observations + command targets
        z_s = (pos_w[:, 2] - self._height_cmd).unsqueeze(1) / 20.0
        lin_s = lin_vel_b[:, 0:3] / 10.0
        ang_s = ang_vel_b[:, 0:3] / 10.0
        g_s = g_b[:, 0:3]

        # tail normalized by max magnitude of soft limits
        l_idx, r_idx = self._IDX_LEFT_TAIL, self._IDX_RIGHT_TAIL
        l_den = torch.maximum(self._joint_upper_limits[l_idx].abs(), self._joint_lower_limits[l_idx].abs())
        r_den = torch.maximum(self._joint_upper_limits[r_idx].abs(), self._joint_lower_limits[r_idx].abs())
        tail_s = torch.stack([jpos[:, l_idx] / l_den, jpos[:, r_idx] / r_den], dim=1)

        freq_s = self._freq.unsqueeze(1) / float(self.cfg.max_flap_hz)
        vx_err_s = (lin_vel_b[:, 0] - self._vx_cmd).unsqueeze(1) / 5.0

        def _roll_and_set(buf: Tensor, new: Tensor):
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
                self._hist_vx[ids] = vx_err_s[ids].unsqueeze(1).repeat(1, self.cfg.stack_vx, 1)
                self._hist_valid[ids] = True

        _roll_and_set(self._hist_z, z_s)
        _roll_and_set(self._hist_lin, lin_s)
        _roll_and_set(self._hist_ang, ang_s)
        _roll_and_set(self._hist_gb, g_s)
        _roll_and_set(self._hist_tail, tail_s)
        _roll_and_set(self._hist_freq, freq_s)
        _roll_and_set(self._hist_vx, vx_err_s)

        obs = torch.cat(
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
        return {"policy": obs}

    def _get_rewards(self) -> Tensor:
        pos_w = self._robot.data.root_pos_w - self.scene.env_origins
        # Height tracking
        height = pos_w[:, 2]
        height_error = height - self._height_cmd
        r_height = 1.0 - torch.tanh(torch.abs(height_error) / 0.75)

        # Forward speed tracking (body-x)
        vx = self._robot.data.root_lin_vel_b[:, 0]
        vx_error = vx - self._vx_cmd
        r_vx = 1.0 - torch.tanh(torch.abs(vx_error) / 0.75)

        # Lateral drift penalty
        vy = self._robot.data.root_lin_vel_b[:, 1]
        p_vy = torch.tanh(torch.abs(vy) / 1.0)
        y = pos_w[:, 1]
        p_y = torch.tanh(torch.abs(y) / float(self.cfg.terminate_abs_y))

        # Attitude tracking (roll ~ 0, pitch ~ commanded trim).
        roll, pitch, _yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        pitch_cmd = -math.radians(float(self.cfg.pitch_cmd_deg))  # positive nose-up -> negative pitch angle
        roll_cmd = math.radians(float(self.cfg.roll_cmd_deg))
        pitch_err = pitch - pitch_cmd
        roll_err = roll - roll_cmd
        pitch_scale = math.radians(float(self.cfg.att_pitch_err_deg))
        roll_scale = math.radians(float(self.cfg.att_roll_err_deg))
        r_pitch = 1.0 - torch.tanh(torch.abs(pitch_err) / max(pitch_scale, 1.0e-6))
        r_roll = 1.0 - torch.tanh(torch.abs(roll_err) / max(roll_scale, 1.0e-6))
        r_att = 0.5 * (r_pitch + r_roll)

        # Angular velocity penalty
        ang = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        p_ang = torch.tanh(ang)

        # Action penalty
        p_act = torch.sum(self._act_cmd ** 2, dim=1)

        return (
            float(self.cfg.w_height) * r_height
            + float(self.cfg.w_vx) * r_vx
            + float(self.cfg.w_att) * r_att
            - float(self.cfg.w_vy) * p_vy
            - float(self.cfg.w_y) * p_y
            - float(self.cfg.w_act) * p_act
            - float(self.cfg.w_ang) * p_ang
        )

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        pos_w = self._robot.data.root_pos_w - self.scene.env_origins
        height = pos_w[:, 2]
        fell = height <= float(self.cfg.terminate_ground_height)

        # tilt termination
        g_b = self._robot.data.projected_gravity_b
        tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
        tilt_thr = torch.sin(torch.deg2rad(torch.tensor(self.cfg.terminate_tilt_deg, device=self.device)))
        fell = fell | (tilt > tilt_thr)

        # lateral bound
        y = pos_w[:, 1]
        fell = fell | (torch.abs(y) > float(self.cfg.terminate_abs_y))

        # time out
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, timed_out
