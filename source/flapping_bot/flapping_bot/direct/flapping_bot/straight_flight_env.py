"""Direct RL environment for straight-flight flapping-wing control.

This environment is designed to work with IsaacLab's standard RSL-RL scripts:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK_ID> --headless

Key design choices for long-horizon iteration:
- Low-dimensional actions: throttle + rudder + elevon pitch/roll commands.
- Tail aerodynamics depends on deflection angle (not only joint velocity).
- Optional wing aerodynamic backend: simple QSM (fast) or DeLaurier (1993) strip theory (slower, more detailed).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

from ...assets import (
    IDEAL_COUPLED_WING_DRIVE,
    IDEAL_DRIVER_EFFORT_LIMIT_NM,
    IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
    IDEAL_TORQUE_COUPLED_WING_DRIVE,
    IDEAL_TORQUE_DAMPING_RATIO,
    IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2,
    IDEAL_TORQUE_NATURAL_FREQUENCY_HZ,
    KINEMATIC_WING_OVERRIDE,
    NATIVE_HOLONOMIC_WING_DRIVE,
    PRESCRIBED_COUPLED_WING_DRIVE,
    SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
    FlappingBotCfg,
    IdealCoupledFlappingBotCfg,
    IdealInverseDynamicsPhaseCoupledFlappingBotCfg,
    IdealTorqueCoupledFlappingBotCfg,
    NativeHolonomicCoupledFlappingBotCfg,
    PrescribedCoupledFlappingBotCfg,
    SinusoidalPhaseSpeedCoupledFlappingBotCfg,
    apply_hard_opposed_wing_mimic,
    apply_native_holonomic_wing_constraint,
    apply_quintic_amplitude_ramp,
    compute_aerodynamic_joint_hinge_torques,
    compute_common_aerodynamic_hinge_torque,
    compute_ideal_torque_drive_effort,
    cloned_native_holonomic_joint_paths,
    require_native_holonomic_extension,
    validate_wing_drive_variant,
)
from ...physics import (
    ACTUAL_JOINT_ACCELERATION,
    ACTUAL_MOTION_BASE_EQUIVALENT,
    ACTUAL_PER_WING_LINK,
    COMMANDED_BASE_EQUIVALENT,
    DeLaurierParams,
    DeLaurierStripLoads,
    DeLaurierStripWrench,
    DeLaurierTwistKinematics,
    FULL_WING_LINK_WRENCH,
    IDEAL_INVERSE_DYNAMICS_PER_WING_LINK,
    NATIVE_HOLONOMIC_PER_WING_LINK,
    IDEAL_TORQUE_PER_WING_LINK,
    PRESCRIBED_ACCELERATION,
    PRESCRIBED_PER_WING_LINK,
    SINUSOIDAL_PHASE_PER_WING_LINK,
    build_prescribed_dof_position_limits,
    body_air_velocity_to_delaurier_section_velocity,
    compute_aero_wrench_delaurier1993,
    compute_area_weighted_quarter_chord_link_points,
    compute_delaurier_axis_incidence,
    compute_delaurier_dynamic_twist,
    compute_delaurier_strip_loads,
    compute_legacy_qd_scaled_twist,
    compute_opposed_wing_kinematics,
    compute_desired_common_acceleration,
    estimate_common_constraint_load,
    compute_sinusoidal_constraint_effort,
    FlappingQSMCfg,
    build_measured_wing_multibody_tensors,
    integrate_delaurier_strip_wrench,
    map_opposed_joint_states_to_physical_wing_kinematics,
    MEASURED_MULTIBODY_DEFAULT_PLACEHOLDER_MASS_KG,
    MEASURED_WING_MULTIBODY_PLANT,
    NEAR_SINGLE_RIGID_BODY_PLANT,
    QuasiSteadyWingModel,
    resolve_delaurier_phase,
    resolve_wing_aero_acceleration,
    reduce_common_inverse_dynamics,
    IdealFrequencyPhaseState,
    IdealFrequencyPhaseStep,
    IdealInverseDynamicsPhaseDriveConfig,
    SinusoidalPhaseDriveState,
    SinusoidalPhaseDriveStep,
    SinusoidalPhaseSpeedDriveConfig,
    TailAeroCfg,
    TailAeroModel,
    WING_LINK_FORCE_ONLY,
    WING_LINK_MOMENT_ONLY,
    transform_wang_wrench_to_link,
    translate_wing_root_wrench_to_com_link,
    translate_wrench_moment,
    validate_wing_aero_coupling_mode,
    validate_wing_aero_acceleration_source,
    validate_wing_link_aero_load_mode,
    validate_delaurier_dynamic_twist_mode,
    validate_flapping_bot_plant_variant,
    WingQSMCfg,
    WingGeometry,
    build_wing_geometry_from_csv,
    step_sinusoidal_phase_speed_drive,
    step_ideal_frequency_phase,
)
from ...px4_like.rl_training_utils import (
    apply_teacher_guided_actions,
    linear_anneal,
    piecewise_linear_anneal,
    resolve_teacher_guidance_mode,
    teacher_guidance_is_active,
)
from ...px4_like.state_estimation import SensorStateEstimator, SensorSuiteCfg, StateEstimatorCfg
from ...px4_like.straight_line_controller import PX4LikeStraightLineController, PX4LikeStraightLineControllerCfg
from ...px4_like import build_imu_provider, resolve_base_body_com_offset_b
from ...scenes import FlappingRoomSceneCfg
from .action_contract import (
    ACTUAL_JOINT_TAIL_AERO_DEFLECTION,
    COMMAND_TAIL_AERO_DEFLECTION,
    DIRECT_TAIL_SURFACE_ACTION,
    MIXED_ELEVON_ACTION,
    apply_frequency_slew_governor,
    frequency_hz_to_normalized_action,
    joint_position_to_normalized_action,
    normalized_action_to_frequency_hz,
    normalized_action_to_joint_position,
    validate_action_interface,
    validate_tail_aero_deflection_source,
)
from .pure_rl_observation import (
    PURE_RL_RAW_OBSERVATION_LAYOUT,
    build_raw_sensor_frame,
    compute_preview_query_progress_m,
    normalize_actor_observation,
    transform_world_preview_points_to_body,
)
from .pure_rl_curriculum_contract import PURE_RL_SHARED_CONTRACT
from .pure_rl_longitudinal_path import (
    CLIMB_TASK_ID,
    DESCENT_TASK_ID,
    LEVEL_TASK_ID,
    PureRLLongitudinalPathBatch,
    PureRLLongitudinalPathQuery,
    PureRLLongitudinalStageConfig,
    query_longitudinal_path,
    resolve_longitudinal_stage,
    sample_longitudinal_path_batch,
    write_longitudinal_path_batch_rows_,
)
from .pure_rl_reward import (
    PURE_RL_CURRICULUM1_TERMINATION_CONFIG,
    PureRLRewardConfig,
    compute_pure_rl_path_reward_terms,
    compute_pure_rl_reward_terms,
    compute_pure_rl_termination_terms,
)
from .state_source_contract import (
    TeacherStateInputs,
    resolve_imu_source,
    resolve_teacher_state_inputs,
)
from .startup_phase import (
    MECHANICAL_SINE_NEUTRAL_UPSTROKE,
    advance_flap_phase,
    compute_prescribed_flap_kinematics,
    map_symmetric_flap_coordinate_to_joint_space,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class _DeLaurierWingWrenchResult:
    """Per-wing and equivalent total DeLaurier wrenches.

    ``force_link_n`` and ``moment_link_about_com_nm`` have shape ``(N,2,3)``
    in each wing link's FLU frame. ``net_force_b_n`` and
    ``net_moment_b_about_base_com_nm`` have shape ``(N,3)`` in the base-link
    FLU frame.
    """

    net_force_b_n: Tensor
    net_moment_b_about_base_com_nm: Tensor
    force_link_n: Tensor
    moment_link_about_com_nm: Tensor


@configclass
class FlappingBotStraightFlightEnvCfg(DirectRLEnvCfg):
    """Base configuration for straight-flight training."""

    # episode / control
    episode_length_s: float = 12.0
    decimation: int = 2
    action_space: int = 4
    # Baseline: [frequency, rudder, elevon_pitch, elevon_roll]. The measured
    # PureRL task explicitly selects [frequency, rudder, left_elevon,
    # right_elevon] without changing controller-facing configurations.
    action_interface: str = MIXED_ELEVON_ACTION
    observation_space: int = 68  # keep same stacking layout as FlappingBotEnv
    state_space: int = 0
    action_scale: float = 1.0
    use_pure_rl_actor_observation: bool = False
    randomize_straight_line_heading: bool = False
    randomize_flap_phase_at_reset: bool = False
    pure_rl_eval_heading_schedule_rad: tuple[float, ...] | None = None
    pure_rl_eval_flap_phase_schedule_rad: tuple[float, ...] | None = None
    pure_rl_eval_longitudinal_task_schedule: tuple[int, ...] | None = None
    pure_rl_eval_longitudinal_slope_deg_schedule: tuple[float, ...] | None = None
    pure_rl_eval_entry_length_m_schedule: tuple[float, ...] | None = None
    pure_rl_eval_slope_length_m_schedule: tuple[float, ...] | None = None
    pure_rl_longitudinal_stage_id: str | None = None
    pure_rl_preview_minimum_speed_mps: float = 1.0
    pure_rl_preview_maximum_speed_mps: float = 12.0
    use_pure_rl_curriculum1_reward: bool = False
    pure_rl_reward_telemetry_enabled: bool = False
    pure_rl_reward_cfg: PureRLRewardConfig = PureRLRewardConfig()

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
    frequency_governor_enabled: bool = False
    frequency_governor_maximum_rise_rate_hz_per_s: float = 2.0
    frequency_governor_maximum_fall_rate_hz_per_s: float = 2.0

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

    # Legacy reward weights retained for controller-facing environments.
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
    pure_rl_terminate_abs_height_error_m: float = 3.0

    # flapping frequency action mapping
    # Keep the initial action range fairly tight around typical trimmed conditions.
    # A very wide range makes early exploration extremely unstable.
    min_flap_hz: float = 2.0
    max_flap_hz: float = 5.0

    # joint actuation model
    wing_drive_variant: str = KINEMATIC_WING_OVERRIDE
    # If True, directly write joint positions/velocities each physics step (kinematic override).
    # This decouples wing flapping kinematics from rigid-body reaction dynamics and improves stability for
    # aero-driven flight experiments.
    use_kinematic_joint_override: bool = True
    # The prescribed mechanism uses a moving PhysX position-limit band on the
    # common coordinate. Zero requests an equality constraint.
    prescribed_joint_limit_half_width_rad: float = 0.0
    # Minimal ideal trajectory servo for the measured two-wing common
    # coordinate. The effort feedback uses physics-step-aware discrete gains;
    # these runtime values define inertia feedforward and startup behavior.
    ideal_torque_equivalent_inertia_kg_m2: float = IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2
    ideal_torque_aero_feedforward_scale: float = 1.0
    ideal_torque_ramp_cycles: float = 2.0
    # Experimental nonlinear phase-to-wing constraint. These are numerical
    # mechanism settings, not identified motor parameters.
    sinusoidal_constraint_natural_frequency_hz: float = 50.0
    sinusoidal_constraint_damping_ratio: float = 1.0
    sinusoidal_phase_inertia_kg_m2: float = 0.00435938889653
    sinusoidal_speed_settling_time_s: float = 0.15
    sinusoidal_speed_damping_ratio: float = 1.0
    sinusoidal_effort_limit_nm: float = IDEAL_DRIVER_EFFORT_LIMIT_NM
    # Scheme B: ideal load-independent frequency source with PhysX reduced
    # inverse dynamics. These remain numerical mechanism settings.
    ideal_inverse_frequency_settling_time_s: float = 0.15
    ideal_inverse_tracking_natural_frequency_hz: float = 50.0
    ideal_inverse_tracking_damping_ratio: float = 1.0
    ideal_inverse_effort_limit_nm: float = IDEAL_DRIVER_EFFORT_LIMIT_NM

    # dynamics decoupling (mass/inertia)
    # Plant selection is explicit so the established near-single-rigid-body
    # baseline remains available while measured wing inertia is introduced.
    plant_variant: str = NEAR_SINGLE_RIGID_BODY_PLANT
    measured_multibody_placeholder_mass_kg: float = MEASURED_MULTIBODY_DEFAULT_PLACEHOLDER_MASS_KG
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
    # Measured whole-aircraft properties expressed in the base_link body frame.
    # With appendage inertial coupling suppressed below, base_link represents the
    # near-single-rigid-body plant used by the aerodynamic simulation.
    base_body_com_override_x_m: float | None = None
    base_body_com_override_m: tuple[float, float, float] | None = (-0.12154, 0.00541, -0.01298)
    total_mass_kg_override: float | None = 0.90415
    base_body_inertia_diag_override_kg_m2: tuple[float, float, float] | None = (0.02329, 0.02573, 0.04270)

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
    # Baseline compatibility uses commanded angles. The direct-surface PureRL
    # task selects actual PhysX joint positions so the aerodynamic calculation
    # observes implicit-drive lag.
    tail_aero_deflection_source: str = COMMAND_TAIL_AERO_DEFLECTION

    # diagnostics: selectively apply aerodynamic components
    enable_wing_aero: bool = True
    enable_tail_aero: bool = True
    # The established baseline computes commanded wing kinematics and applies
    # one equivalent wing wrench at the base COM. The explicit multibody mode
    # uses actual joint motion and applies one wrench to each wing link.
    wing_aero_coupling_mode: str = COMMANDED_BASE_EQUIVALENT
    # Diagnostic source for the acceleration-dependent DeLaurier heave input.
    # It is used only by actual_per_wing_link; the default preserves the
    # implemented PhysX joint-acceleration behavior.
    wing_aero_acceleration_source: str = ACTUAL_JOINT_ACCELERATION
    # Diagnostic ablation for loads applied to each wing COM. The default
    # applies the complete computed wrench.
    wing_link_aero_load_mode: str = FULL_WING_LINK_WRENCH

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
    # ``strip_integrated`` is the physical default. The legacy fixed quarter-
    # chord closure remains available for A/B regression only.
    wing_moment_mode: str = "strip_integrated"
    delaurier_include_aerodynamic_center_moment: bool = True
    delaurier_include_apparent_mass_moment: bool = True
    delaurier_store_strip_diagnostics: bool = False
    # Opt-in diagnostics for the native ideal mechanism. The load is a
    # multibody inverse-dynamics estimate in the common wing coordinate, not a
    # PhysX constraint multiplier or motor-shaft quantity.
    native_holonomic_load_diagnostics: bool = False
    # Compute a non-applied DeLaurier wrench using actual PhysX joint
    # acceleration alongside the prescribed-acceleration plant calculation.
    delaurier_shadow_actual_acceleration: bool = False
    # Induced drag correction (simple Oswald efficiency model).
    # DeLaurier strip theory as used here does not include a finite-wing induced drag term, which can lead to
    # unrealistic positive chordwise force and runaway acceleration in free-flight simulations.
    # Set to 0 to disable.
    delaurier_induced_drag_efficiency: float = 0.0
    # Simple quadratic parasite drag applied at the base link (acts opposite body velocity).
    # Use as a stabilizing term to prevent unbounded acceleration when combined aero models are missing body drag.
    fuselage_drag_cda: float = 0.0  # m^2 effective Cd*A

    # Prescribed DeLaurier dynamic twist.  The default remains the rigid-wing
    # prior; amplitudes are configured in degrees and converted to radians at
    # the environment-to-physics boundary.
    dynamic_twist_mode: str = "disabled"
    dynamic_twist_tip_amplitude_deg: float = 0.0
    dynamic_twist_phase_direction: float = 1.0
    dynamic_twist_phase_offset_deg: float = -90.0

    # Mechanical phase: q=A*sin(phase), phase=0 is neutral and starts upstroke.
    # ``legacy_cosine_endpoint_zero`` remains available for baseline replay;
    # pair it with dynamic_twist_phase_offset_deg=0.
    flap_phase_convention: str = MECHANICAL_SINE_NEUTRAL_UPSTROKE

    # Historical full-span-uniform qd-scaled proxy.  These fields are consumed
    # only when ``dynamic_twist_mode='legacy_qd_scaled_proxy'``.
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
class FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg(FlappingBotStraightFlightEnvCfg):
    """Measured body/two-wing PhysX plant with the legacy drive path unchanged."""

    plant_variant: str = MEASURED_WING_MULTIBODY_PLANT
    override_appendage_masses: bool = False
    redistribute_removed_mass_to_base: bool = False
    total_mass_kg_override: float | None = None
    base_body_com_override_x_m: float | None = None
    base_body_com_override_m: tuple[float, float, float] | None = None
    base_body_inertia_diag_override_kg_m2: tuple[float, float, float] | None = None


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg
):
    """Measured plant driven by one ideal left-wing PD actuator and hard coupling."""

    wing_drive_variant: str = IDEAL_COUPLED_WING_DRIVE
    use_kinematic_joint_override: bool = False
    robot: ArticulationCfg = IdealCoupledFlappingBotCfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=IdealCoupledFlappingBotCfg.spawn.replace(
            rigid_props=IdealCoupledFlappingBotCfg.spawn.rigid_props.replace(disable_gravity=False),
        ),
    )


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg
):
    """Measured multibody plant with actual-motion per-wing DeLaurier loads."""

    use_delaurier_wings: bool = True
    wing_aero_coupling_mode: str = ACTUAL_PER_WING_LINK


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg
):
    """Measured plant with one explicit ideal common-coordinate effort servo."""

    wing_drive_variant: str = IDEAL_TORQUE_COUPLED_WING_DRIVE
    use_kinematic_joint_override: bool = False
    robot: ArticulationCfg = IdealTorqueCoupledFlappingBotCfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=IdealTorqueCoupledFlappingBotCfg.spawn.replace(
            rigid_props=IdealTorqueCoupledFlappingBotCfg.spawn.rigid_props.replace(disable_gravity=False),
        ),
    )


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg
):
    """Ideal-torque measured plant with actual-state per-wing DeLaurier loads."""

    use_delaurier_wings: bool = True
    wing_aero_coupling_mode: str = IDEAL_TORQUE_PER_WING_LINK
    wing_aero_acceleration_source: str = PRESCRIBED_ACCELERATION


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg
):
    """Measured plant with passive wings constrained to a prescribed common coordinate."""

    wing_drive_variant: str = PRESCRIBED_COUPLED_WING_DRIVE
    use_kinematic_joint_override: bool = False
    robot: ArticulationCfg = PrescribedCoupledFlappingBotCfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=PrescribedCoupledFlappingBotCfg.spawn.replace(
            rigid_props=PrescribedCoupledFlappingBotCfg.spawn.rigid_props.replace(disable_gravity=False),
        ),
    )


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg
):
    """Prescribed measured multibody plant with per-wing DeLaurier loads."""

    use_delaurier_wings: bool = True
    wing_aero_coupling_mode: str = PRESCRIBED_PER_WING_LINK


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg
):
    """No-aerodynamics technical gate for the nonlinear phase mechanism."""

    wing_drive_variant: str = SINUSOIDAL_PHASE_SPEED_WING_DRIVE
    use_kinematic_joint_override: bool = False
    min_flap_hz: float = 0.0
    max_flap_hz: float = 5.0
    reset_flap_hz: float = 0.0
    enable_wing_aero: bool = False
    enable_tail_aero: bool = False
    robot: ArticulationCfg = SinusoidalPhaseSpeedCoupledFlappingBotCfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=SinusoidalPhaseSpeedCoupledFlappingBotCfg.spawn.replace(
            rigid_props=SinusoidalPhaseSpeedCoupledFlappingBotCfg.spawn.rigid_props.replace(
                disable_gravity=False
            ),
        ),
    )


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg
):
    """Sinusoidal phase mechanism with actual-state per-wing DeLaurier loads."""

    enable_wing_aero: bool = True
    use_delaurier_wings: bool = True
    wing_aero_coupling_mode: str = SINUSOIDAL_PHASE_PER_WING_LINK
    wing_aero_acceleration_source: str = ACTUAL_JOINT_ACCELERATION


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg
):
    """Measured plant with an ideal frequency source and reduced inverse dynamics."""

    wing_drive_variant: str = IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE
    use_kinematic_joint_override: bool = False
    min_flap_hz: float = 0.0
    max_flap_hz: float = 5.0
    reset_flap_hz: float = 0.0
    enable_wing_aero: bool = False
    enable_tail_aero: bool = False
    robot: ArticulationCfg = IdealInverseDynamicsPhaseCoupledFlappingBotCfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=IdealInverseDynamicsPhaseCoupledFlappingBotCfg.spawn.replace(
            rigid_props=IdealInverseDynamicsPhaseCoupledFlappingBotCfg.spawn.rigid_props.replace(
                disable_gravity=False
            ),
        ),
    )


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg
):
    """Scheme-B measured plant with per-wing DeLaurier loads."""

    enable_wing_aero: bool = True
    use_delaurier_wings: bool = True
    wing_aero_coupling_mode: str = IDEAL_INVERSE_DYNAMICS_PER_WING_LINK
    wing_aero_acceleration_source: str = PRESCRIBED_ACCELERATION


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg
):
    """Measured plant with a native ideal sinusoidal mechanism constraint."""

    # Keep the 1/120 s controller cadence while using the physics step that
    # passed the 2/3/4/5 Hz aerodynamic trajectory matrix.
    decimation: int = 4
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 480.0,
        render_interval=decimation,
        device="cpu",
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
    )
    wing_drive_variant: str = NATIVE_HOLONOMIC_WING_DRIVE
    use_kinematic_joint_override: bool = False
    min_flap_hz: float = 0.0
    max_flap_hz: float = 5.0
    reset_flap_hz: float = 0.0
    enable_wing_aero: bool = False
    enable_tail_aero: bool = False
    scene: InteractiveSceneCfg = FlappingRoomSceneCfg(
        num_envs=256,
        env_spacing=5.0,
        replicate_physics=False,
    )
    robot: ArticulationCfg = NativeHolonomicCoupledFlappingBotCfg.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=NativeHolonomicCoupledFlappingBotCfg.spawn.replace(
            rigid_props=NativeHolonomicCoupledFlappingBotCfg.spawn.rigid_props.replace(
                disable_gravity=False,
                retain_accelerations=False,
            ),
        ),
    )


@configclass
class FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg
):
    """Native holonomic measured plant with per-wing DeLaurier loads."""

    enable_wing_aero: bool = True
    use_delaurier_wings: bool = True
    wing_aero_coupling_mode: str = NATIVE_HOLONOMIC_PER_WING_LINK
    wing_aero_acceleration_source: str = PRESCRIBED_ACCELERATION


@configclass
class FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg(
    FlappingBotStraightFlightEnvCfg
):
    """Legacy commanded-kinematics/base-wrench DeLaurier comparison plant."""

    use_delaurier_wings: bool = True


@configclass
class FlappingBotStraightFlightDeLaurierEnvCfg(
    FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg
):
    """Default measured-wing native-holonomic DeLaurier flight plant."""

    min_flap_hz: float = 2.0
    max_flap_hz: float = 5.0
    reset_flap_hz: float = 4.0
    enable_tail_aero: bool = True


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


@configclass
class FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg(FlappingBotStraightFlightDeLaurierPureRLEnvCfg):
    """Canonical curriculum-1 PureRL defaults for the measured native plant."""

    # The policy produces one command every eight 480 Hz physics steps.
    decimation: int = PURE_RL_SHARED_CONTRACT.policy_decimation
    sim: SimulationCfg = SimulationCfg(
        dt=PURE_RL_SHARED_CONTRACT.physics_dt_s,
        render_interval=decimation,
        device="cpu",
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
    )
    action_interface: str = DIRECT_TAIL_SURFACE_ACTION
    tail_aero_deflection_source: str = ACTUAL_JOINT_TAIL_AERO_DEFLECTION
    observation_space: int = PURE_RL_SHARED_CONTRACT.observation_dim
    use_pure_rl_actor_observation: bool = True
    pure_rl_preview_minimum_speed_mps: float = 1.0
    pure_rl_preview_maximum_speed_mps: float = 12.0
    randomize_straight_line_heading: bool = True
    randomize_flap_phase_at_reset: bool = True
    randomize_commands: bool = False
    teacher_guidance_enabled: bool = False
    use_pure_rl_curriculum1_reward: bool = True
    pure_rl_reward_telemetry_enabled: bool = True
    pure_rl_reward_cfg: PureRLRewardConfig = PureRLRewardConfig()
    pure_rl_longitudinal_stage_id: str | None = None
    terminate_ground_height: float = 0.05
    terminate_tilt_deg: float = 75.0
    terminate_abs_y: float = 3.0
    pure_rl_terminate_abs_height_error_m: float = 3.0
    freeze_steps_after_reset: int = 0

    # Tail commands remain direct. Frequency alone receives a physical slew governor.
    act_lpf_tau_s: float = 0.0
    act_rate_limit_per_s: float = 0.0
    frequency_governor_enabled: bool = True
    frequency_governor_maximum_rise_rate_hz_per_s: float = (
        PURE_RL_SHARED_CONTRACT.frequency_governor_rise_hz_per_s
    )
    frequency_governor_maximum_fall_rate_hz_per_s: float = (
        PURE_RL_SHARED_CONTRACT.frequency_governor_fall_hz_per_s
    )

    # Start the pure-RL smoke experiment without wind. Wind and dynamics randomization should be added only after
    # the policy can maintain basic height, speed, and attitude in the measured nominal model.
    wind_enabled: bool = False
    randomize_wind: bool = False
    wind_ou_enabled: bool = False
    wind_curriculum_enabled: bool = False
    min_flap_hz: float = PURE_RL_SHARED_CONTRACT.minimum_flap_frequency_hz
    max_flap_hz: float = PURE_RL_SHARED_CONTRACT.maximum_flap_frequency_hz


@configclass
class FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg(
    FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg
):
    """Measured PureRL longitudinal curriculum stage C2a."""

    pure_rl_longitudinal_stage_id: str = "c2a"


@configclass
class FlappingBotStraightFlightDeLaurierMeasuredPureRLC2bEnvCfg(
    FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg
):
    """Measured PureRL longitudinal curriculum stage C2b."""

    pure_rl_longitudinal_stage_id: str = "c2b"


@configclass
class FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg(
    FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg
):
    """Measured PureRL longitudinal curriculum stage C2c."""

    pure_rl_longitudinal_stage_id: str = "c2c"


class FlappingBotStraightFlightEnv(DirectRLEnv):
    cfg: FlappingBotStraightFlightEnvCfg

    def __init__(self, cfg: FlappingBotStraightFlightEnvCfg, render_mode: str | None = None, **kwargs):
        action_interface = validate_action_interface(cfg.action_interface)
        validate_tail_aero_deflection_source(cfg.tail_aero_deflection_source)
        longitudinal_stage = (
            resolve_longitudinal_stage(cfg.pure_rl_longitudinal_stage_id)
            if cfg.pure_rl_longitudinal_stage_id is not None
            else None
        )
        if longitudinal_stage is not None:
            if not bool(cfg.use_pure_rl_actor_observation) or not bool(cfg.use_pure_rl_curriculum1_reward):
                raise ValueError("PureRL longitudinal stages require the PureRL observation and reward contracts.")
            if bool(cfg.wind_enabled) or bool(cfg.randomize_wind) or bool(cfg.wind_ou_enabled):
                raise ValueError("PureRL C2 longitudinal stages require wind to remain disabled.")
        if bool(cfg.use_pure_rl_actor_observation):
            if action_interface != DIRECT_TAIL_SURFACE_ACTION:
                raise ValueError("PureRL actor observations require the direct-tail-surface action interface.")
            if int(cfg.observation_space) != PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim:
                raise ValueError(
                    "PureRL actor observation_space must equal "
                    f"{PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim}."
                )
            preview_minimum = float(cfg.pure_rl_preview_minimum_speed_mps)
            preview_maximum = float(cfg.pure_rl_preview_maximum_speed_mps)
            if not math.isfinite(preview_minimum) or not math.isfinite(preview_maximum):
                raise ValueError("PureRL preview-speed bounds must be finite.")
            if preview_minimum < 0.0 or preview_maximum < preview_minimum:
                raise ValueError("PureRL preview-speed bounds must satisfy 0 <= minimum <= maximum.")
            if float(cfg.min_flap_hz) != 0.0 or float(cfg.max_flap_hz) != 5.0:
                raise ValueError("The fixed PureRL frequency observation contract requires a 0--5 Hz action range.")
        eval_heading_schedule = cfg.pure_rl_eval_heading_schedule_rad
        eval_phase_schedule = cfg.pure_rl_eval_flap_phase_schedule_rad
        eval_longitudinal_schedules = (
            cfg.pure_rl_eval_longitudinal_task_schedule,
            cfg.pure_rl_eval_longitudinal_slope_deg_schedule,
            cfg.pure_rl_eval_entry_length_m_schedule,
            cfg.pure_rl_eval_slope_length_m_schedule,
        )
        if (eval_heading_schedule is None) != (eval_phase_schedule is None):
            raise ValueError("PureRL evaluation heading and flap-phase schedules must be configured together.")
        if eval_heading_schedule is not None:
            if len(eval_heading_schedule) == 0 or len(eval_heading_schedule) != len(eval_phase_schedule):
                raise ValueError("PureRL evaluation schedules must be non-empty and have equal lengths.")
            if bool(cfg.randomize_straight_line_heading) or bool(cfg.randomize_flap_phase_at_reset):
                raise ValueError("PureRL evaluation schedules require heading and flap-phase randomization to be disabled.")
        configured_longitudinal_schedules = tuple(value is not None for value in eval_longitudinal_schedules)
        if any(configured_longitudinal_schedules) and not all(configured_longitudinal_schedules):
            raise ValueError("PureRL longitudinal evaluation schedules must be configured together.")
        if all(configured_longitudinal_schedules):
            if longitudinal_stage is None or eval_heading_schedule is None:
                raise ValueError("Longitudinal evaluation schedules require a C2 stage and heading/phase schedules.")
            schedule_length = len(eval_heading_schedule)
            if any(len(value) != schedule_length for value in eval_longitudinal_schedules if value is not None):
                raise ValueError("All PureRL evaluation schedules must have equal lengths.")
            task_schedule = cfg.pure_rl_eval_longitudinal_task_schedule
            slope_schedule = cfg.pure_rl_eval_longitudinal_slope_deg_schedule
            entry_schedule = cfg.pure_rl_eval_entry_length_m_schedule
            length_schedule = cfg.pure_rl_eval_slope_length_m_schedule
            assert task_schedule is not None
            assert slope_schedule is not None
            assert entry_schedule is not None
            assert length_schedule is not None
            if any(task not in (LEVEL_TASK_ID, CLIMB_TASK_ID, DESCENT_TASK_ID) for task in task_schedule):
                raise ValueError("Longitudinal evaluation task IDs must be level, climb, or descent.")
            for task, slope_deg in zip(task_schedule, slope_schedule):
                if not math.isfinite(float(slope_deg)):
                    raise ValueError("Longitudinal evaluation slopes must be finite.")
                if task == LEVEL_TASK_ID and float(slope_deg) != 0.0:
                    raise ValueError("Level evaluation cases require zero slope.")
                if task == CLIMB_TASK_ID and float(slope_deg) <= 0.0:
                    raise ValueError("Climb evaluation cases require positive slope.")
                if task == DESCENT_TASK_ID and float(slope_deg) >= 0.0:
                    raise ValueError("Descent evaluation cases require negative slope.")
            if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in entry_schedule):
                raise ValueError("Longitudinal evaluation entry lengths must be finite and positive.")
            if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in length_schedule):
                raise ValueError("Longitudinal evaluation slope lengths must be finite and positive.")
        if bool(cfg.use_pure_rl_curriculum1_reward):
            if action_interface != DIRECT_TAIL_SURFACE_ACTION:
                raise ValueError("PureRL curriculum-1 reward requires the direct-tail-surface action interface.")
            if float(cfg.min_flap_hz) != 0.0 or float(cfg.max_flap_hz) != float(
                cfg.pure_rl_reward_cfg.maximum_flap_frequency_hz
            ):
                raise ValueError("PureRL reward frequency scale must match the configured 0--5 Hz action range.")
            if float(cfg.terminate_abs_y) <= 0.0 or float(cfg.pure_rl_terminate_abs_height_error_m) <= 0.0:
                raise ValueError("PureRL route-relative termination thresholds must be positive.")
        if action_interface == DIRECT_TAIL_SURFACE_ACTION and bool(cfg.teacher_guidance_enabled):
            raise ValueError(
                "teacher_guidance_enabled is incompatible with the direct-tail-surface action interface."
            )
        wing_drive_variant = validate_wing_drive_variant(cfg.wing_drive_variant)
        wing_aero_coupling_mode = validate_wing_aero_coupling_mode(cfg.wing_aero_coupling_mode)
        validate_wing_aero_acceleration_source(cfg.wing_aero_acceleration_source)
        validate_wing_link_aero_load_mode(cfg.wing_link_aero_load_mode)
        if wing_drive_variant in {
            IDEAL_COUPLED_WING_DRIVE,
            PRESCRIBED_COUPLED_WING_DRIVE,
            IDEAL_TORQUE_COUPLED_WING_DRIVE,
            SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
            IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
            NATIVE_HOLONOMIC_WING_DRIVE,
        }:
            if cfg.plant_variant != MEASURED_WING_MULTIBODY_PLANT:
                raise ValueError(f"{wing_drive_variant} requires plant_variant='measured_wing_multibody'.")
            if bool(cfg.use_kinematic_joint_override):
                raise ValueError(f"{wing_drive_variant} cannot use the per-step kinematic joint override.")
        if float(cfg.prescribed_joint_limit_half_width_rad) < 0.0:
            raise ValueError("prescribed_joint_limit_half_width_rad must be nonnegative.")
        if float(cfg.ideal_torque_equivalent_inertia_kg_m2) <= 0.0:
            raise ValueError("ideal_torque_equivalent_inertia_kg_m2 must be positive.")
        if float(cfg.ideal_torque_aero_feedforward_scale) < 0.0:
            raise ValueError("ideal_torque_aero_feedforward_scale must be nonnegative.")
        if float(cfg.ideal_torque_ramp_cycles) <= 0.0:
            raise ValueError("ideal_torque_ramp_cycles must be positive.")
        if float(cfg.sinusoidal_constraint_natural_frequency_hz) <= 0.0:
            raise ValueError("sinusoidal_constraint_natural_frequency_hz must be positive.")
        if float(cfg.sinusoidal_constraint_damping_ratio) <= 0.0:
            raise ValueError("sinusoidal_constraint_damping_ratio must be positive.")
        if float(cfg.sinusoidal_phase_inertia_kg_m2) <= 0.0:
            raise ValueError("sinusoidal_phase_inertia_kg_m2 must be positive.")
        if float(cfg.sinusoidal_speed_settling_time_s) <= 0.0:
            raise ValueError("sinusoidal_speed_settling_time_s must be positive.")
        if float(cfg.sinusoidal_speed_damping_ratio) <= 0.0:
            raise ValueError("sinusoidal_speed_damping_ratio must be positive.")
        if float(cfg.sinusoidal_effort_limit_nm) <= 0.0:
            raise ValueError("sinusoidal_effort_limit_nm must be positive.")
        if float(cfg.ideal_inverse_frequency_settling_time_s) <= 0.0:
            raise ValueError("ideal_inverse_frequency_settling_time_s must be positive.")
        if float(cfg.ideal_inverse_tracking_natural_frequency_hz) <= 0.0:
            raise ValueError("ideal_inverse_tracking_natural_frequency_hz must be positive.")
        if not math.isclose(
            float(cfg.ideal_inverse_tracking_damping_ratio),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("ideal_inverse_tracking_damping_ratio must equal one.")
        if float(cfg.ideal_inverse_effort_limit_nm) <= 0.0:
            raise ValueError("ideal_inverse_effort_limit_nm must be positive.")
        if wing_drive_variant == SINUSOIDAL_PHASE_SPEED_WING_DRIVE:
            if bool(cfg.enable_tail_aero):
                raise ValueError(
                    "sinusoidal_phase_speed_drive aerodynamic validation currently excludes tail aerodynamics."
                )
            if (
                bool(cfg.enable_wing_aero)
                and wing_aero_coupling_mode != SINUSOIDAL_PHASE_PER_WING_LINK
            ):
                raise ValueError(
                    "sinusoidal_phase_speed_drive wing aerodynamics require "
                    "wing_aero_coupling_mode='sinusoidal_phase_per_wing_link'."
                )
        if wing_drive_variant == IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE:
            if bool(cfg.enable_tail_aero):
                raise ValueError(
                    "ideal_inverse_dynamics_phase_drive validation currently excludes tail aerodynamics."
                )
            if (
                bool(cfg.enable_wing_aero)
                and wing_aero_coupling_mode != IDEAL_INVERSE_DYNAMICS_PER_WING_LINK
            ):
                raise ValueError(
                    "ideal_inverse_dynamics_phase_drive wing aerodynamics require "
                    "wing_aero_coupling_mode='ideal_inverse_dynamics_per_wing_link'."
                )
        if wing_drive_variant == NATIVE_HOLONOMIC_WING_DRIVE:
            if str(cfg.sim.device).startswith("cuda"):
                raise ValueError(
                    "native_holonomic_drive currently requires CPU PhysX; its custom "
                    "PxConstraint is not compatible with a direct-GPU scene."
                )
            if bool(cfg.scene.replicate_physics):
                raise ValueError(
                    "native_holonomic_drive requires scene.replicate_physics=False because "
                    "external custom joints cannot resolve PhysX fast-replicated bodies."
                )
            if bool(cfg.enable_wing_aero) and wing_aero_coupling_mode != NATIVE_HOLONOMIC_PER_WING_LINK:
                raise ValueError(
                    "native_holonomic_drive wing aerodynamics require "
                    "wing_aero_coupling_mode='native_holonomic_per_wing_link'."
                )
        if bool(cfg.native_holonomic_load_diagnostics) and wing_drive_variant != NATIVE_HOLONOMIC_WING_DRIVE:
            raise ValueError(
                "native_holonomic_load_diagnostics requires wing_drive_variant='native_holonomic_drive'."
            )
        if (
            bool(cfg.native_holonomic_load_diagnostics)
            and str(cfg.wing_link_aero_load_mode) != FULL_WING_LINK_WRENCH
        ):
            raise ValueError(
                "native_holonomic_load_diagnostics requires the full wing-link wrench so the "
                "inverse-dynamics estimate matches the applied load."
            )
        if bool(cfg.delaurier_shadow_actual_acceleration):
            if wing_drive_variant != NATIVE_HOLONOMIC_WING_DRIVE:
                raise ValueError(
                    "delaurier_shadow_actual_acceleration requires wing_drive_variant='native_holonomic_drive'."
                )
            if not bool(cfg.enable_wing_aero) or not bool(cfg.use_delaurier_wings):
                raise ValueError(
                    "delaurier_shadow_actual_acceleration requires enabled DeLaurier wing aerodynamics."
                )
            if str(cfg.wing_link_aero_load_mode) != FULL_WING_LINK_WRENCH:
                raise ValueError(
                    "delaurier_shadow_actual_acceleration requires the full wing-link wrench."
                )
        if wing_aero_coupling_mode in {
            ACTUAL_MOTION_BASE_EQUIVALENT,
            ACTUAL_PER_WING_LINK,
        }:
            if cfg.plant_variant != MEASURED_WING_MULTIBODY_PLANT:
                raise ValueError("Actual-motion wing aerodynamics require plant_variant='measured_wing_multibody'.")
            if wing_drive_variant != IDEAL_COUPLED_WING_DRIVE:
                raise ValueError("Actual-motion wing aerodynamics require wing_drive_variant='ideal_coupled_drive'.")
            if not bool(cfg.use_delaurier_wings):
                raise ValueError("Actual-motion wing aerodynamics require use_delaurier_wings=True.")
            if str(cfg.wing_moment_mode) != "strip_integrated":
                raise ValueError("Actual-motion wing aerodynamics require wing_moment_mode='strip_integrated'.")
            if str(cfg.dynamic_twist_mode) == "legacy_qd_scaled_proxy":
                raise ValueError(
                    "actual_per_wing_link does not support dynamic_twist_mode='legacy_qd_scaled_proxy'."
                )
            if float(cfg.delaurier_induced_drag_efficiency) != 0.0:
                raise ValueError(
                    "actual_per_wing_link requires delaurier_induced_drag_efficiency=0 until a per-wing "
                    "induced-drag distribution is defined."
                )
        if wing_aero_coupling_mode == PRESCRIBED_PER_WING_LINK:
            if cfg.plant_variant != MEASURED_WING_MULTIBODY_PLANT:
                raise ValueError("prescribed_per_wing_link requires plant_variant='measured_wing_multibody'.")
            if wing_drive_variant != PRESCRIBED_COUPLED_WING_DRIVE:
                raise ValueError("prescribed_per_wing_link requires wing_drive_variant='prescribed_coupled_drive'.")
            if not bool(cfg.use_delaurier_wings):
                raise ValueError("prescribed_per_wing_link requires use_delaurier_wings=True.")
            if str(cfg.wing_moment_mode) != "strip_integrated":
                raise ValueError("prescribed_per_wing_link requires wing_moment_mode='strip_integrated'.")
            if str(cfg.dynamic_twist_mode) == "legacy_qd_scaled_proxy":
                raise ValueError(
                    "prescribed_per_wing_link does not support dynamic_twist_mode='legacy_qd_scaled_proxy'."
                )
            if float(cfg.delaurier_induced_drag_efficiency) != 0.0:
                raise ValueError(
                    "prescribed_per_wing_link requires delaurier_induced_drag_efficiency=0 until a per-wing "
                    "induced-drag distribution is defined."
                )
        if wing_aero_coupling_mode == IDEAL_TORQUE_PER_WING_LINK:
            if cfg.plant_variant != MEASURED_WING_MULTIBODY_PLANT:
                raise ValueError("ideal_torque_per_wing_link requires plant_variant='measured_wing_multibody'.")
            if wing_drive_variant != IDEAL_TORQUE_COUPLED_WING_DRIVE:
                raise ValueError(
                    "ideal_torque_per_wing_link requires wing_drive_variant='ideal_torque_coupled_drive'."
                )
            if not bool(cfg.use_delaurier_wings):
                raise ValueError("ideal_torque_per_wing_link requires use_delaurier_wings=True.")
            if str(cfg.wing_moment_mode) != "strip_integrated":
                raise ValueError("ideal_torque_per_wing_link requires wing_moment_mode='strip_integrated'.")
            if str(cfg.dynamic_twist_mode) == "legacy_qd_scaled_proxy":
                raise ValueError(
                    "ideal_torque_per_wing_link does not support dynamic_twist_mode='legacy_qd_scaled_proxy'."
                )
            if float(cfg.delaurier_induced_drag_efficiency) != 0.0:
                raise ValueError(
                    "ideal_torque_per_wing_link requires delaurier_induced_drag_efficiency=0 until a per-wing "
                    "induced-drag distribution is defined."
                )
        if wing_aero_coupling_mode == SINUSOIDAL_PHASE_PER_WING_LINK:
            if cfg.plant_variant != MEASURED_WING_MULTIBODY_PLANT:
                raise ValueError(
                    "sinusoidal_phase_per_wing_link requires plant_variant='measured_wing_multibody'."
                )
            if wing_drive_variant != SINUSOIDAL_PHASE_SPEED_WING_DRIVE:
                raise ValueError(
                    "sinusoidal_phase_per_wing_link requires "
                    "wing_drive_variant='sinusoidal_phase_speed_drive'."
                )
            if not bool(cfg.enable_wing_aero) or not bool(cfg.use_delaurier_wings):
                raise ValueError(
                    "sinusoidal_phase_per_wing_link requires enabled DeLaurier wing aerodynamics."
                )
            if str(cfg.wing_moment_mode) != "strip_integrated":
                raise ValueError(
                    "sinusoidal_phase_per_wing_link requires wing_moment_mode='strip_integrated'."
                )
            if str(cfg.dynamic_twist_mode) == "legacy_qd_scaled_proxy":
                raise ValueError(
                    "sinusoidal_phase_per_wing_link does not support "
                    "dynamic_twist_mode='legacy_qd_scaled_proxy'."
                )
            if float(cfg.delaurier_induced_drag_efficiency) != 0.0:
                raise ValueError(
                    "sinusoidal_phase_per_wing_link requires delaurier_induced_drag_efficiency=0 "
                    "until a per-wing induced-drag distribution is defined."
                )
        if wing_aero_coupling_mode == IDEAL_INVERSE_DYNAMICS_PER_WING_LINK:
            if cfg.plant_variant != MEASURED_WING_MULTIBODY_PLANT:
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link requires plant_variant='measured_wing_multibody'."
                )
            if wing_drive_variant != IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE:
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link requires "
                    "wing_drive_variant='ideal_inverse_dynamics_phase_drive'."
                )
            if not bool(cfg.enable_wing_aero) or not bool(cfg.use_delaurier_wings):
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link requires enabled DeLaurier wing aerodynamics."
                )
            if str(cfg.wing_aero_acceleration_source) != PRESCRIBED_ACCELERATION:
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link requires prescribed_acceleration "
                    "for the DeLaurier apparent-mass input."
                )
            if str(cfg.wing_moment_mode) != "strip_integrated":
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link requires wing_moment_mode='strip_integrated'."
                )
            if str(cfg.dynamic_twist_mode) == "legacy_qd_scaled_proxy":
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link does not support "
                    "dynamic_twist_mode='legacy_qd_scaled_proxy'."
                )
            if float(cfg.delaurier_induced_drag_efficiency) != 0.0:
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link requires delaurier_induced_drag_efficiency=0 "
                    "until a per-wing induced-drag distribution is defined."
                )
            if str(cfg.wing_link_aero_load_mode) != FULL_WING_LINK_WRENCH:
                raise ValueError(
                    "ideal_inverse_dynamics_per_wing_link currently requires the full wing-link wrench "
                    "so inverse-dynamics feedforward matches the applied load."
                )
        if wing_aero_coupling_mode == NATIVE_HOLONOMIC_PER_WING_LINK:
            if cfg.plant_variant != MEASURED_WING_MULTIBODY_PLANT:
                raise ValueError(
                    "native_holonomic_per_wing_link requires plant_variant='measured_wing_multibody'."
                )
            if wing_drive_variant != NATIVE_HOLONOMIC_WING_DRIVE:
                raise ValueError(
                    "native_holonomic_per_wing_link requires wing_drive_variant='native_holonomic_drive'."
                )
            if not bool(cfg.enable_wing_aero) or not bool(cfg.use_delaurier_wings):
                raise ValueError(
                    "native_holonomic_per_wing_link requires enabled DeLaurier wing aerodynamics."
                )
            if str(cfg.wing_aero_acceleration_source) != PRESCRIBED_ACCELERATION:
                raise ValueError(
                    "native_holonomic_per_wing_link requires prescribed_acceleration for the "
                    "DeLaurier apparent-mass input."
                )
            if str(cfg.wing_moment_mode) != "strip_integrated":
                raise ValueError(
                    "native_holonomic_per_wing_link requires wing_moment_mode='strip_integrated'."
                )
            if str(cfg.dynamic_twist_mode) == "legacy_qd_scaled_proxy":
                raise ValueError(
                    "native_holonomic_per_wing_link does not support "
                    "dynamic_twist_mode='legacy_qd_scaled_proxy'."
                )
            if float(cfg.delaurier_induced_drag_efficiency) != 0.0:
                raise ValueError(
                    "native_holonomic_per_wing_link requires delaurier_induced_drag_efficiency=0 "
                    "until a per-wing induced-drag distribution is defined."
                )

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
        self._nominal_dof_position_limits_cpu: Tensor | None = None
        self._native_holonomic = (
            require_native_holonomic_extension()
            if wing_drive_variant == NATIVE_HOLONOMIC_WING_DRIVE
            else None
        )
        self._native_holonomic_joint_paths: list[str] = []
        self._pure_rl_longitudinal_stage: PureRLLongitudinalStageConfig | None = longitudinal_stage
        self._pure_rl_longitudinal_path: PureRLLongitudinalPathBatch | None = None

        # wing phase and frequency
        self._phase: Tensor | None = None  # (N,)
        self._freq: Tensor | None = None  # (N,)
        self._ideal_torque_elapsed_s: Tensor | None = None  # (N,)
        self._phase_throttle: Tensor | None = None  # (N,), normalized 0--1
        self._phase_target_frequency_hz: Tensor | None = None  # (N,)
        self._requested_frequency_hz: Tensor | None = None  # (N,)
        self._applied_frequency_hz: Tensor | None = None  # (N,)
        self._frequency_slew_hz_per_s: Tensor | None = None  # (N,)
        self._frequency_governor_limited: Tensor | None = None  # (N,)
        self._phase_acceleration_rad_s2: Tensor | None = None  # (N,)
        self._sinusoidal_phase_drive_state: SinusoidalPhaseDriveState | None = None
        self._sinusoidal_phase_drive_cfg: SinusoidalPhaseSpeedDriveConfig | None = None
        self._ideal_inverse_phase_state: IdealFrequencyPhaseState | None = None
        self._ideal_inverse_phase_cfg: IdealInverseDynamicsPhaseDriveConfig | None = None

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
        self._debug_last_actual_rudder_rad: Tensor | None = None
        self._debug_last_actual_left_elevon_rad: Tensor | None = None
        self._debug_last_actual_right_elevon_rad: Tensor | None = None
        self._runtime_imu_provider = None
        self._runtime_imu_sensor = None
        self._runtime_state_estimator: SensorStateEstimator | None = None
        self._runtime_estimated_state: dict[str, Tensor] | None = None
        self._runtime_estimator_diag: dict[str, Tensor] = {}
        self._straight_line_heading_rad: Tensor | None = None
        self._straight_line_tangent_w: Tensor | None = None
        self._straight_line_normal_w: Tensor | None = None
        self._pure_rl_history_valid: Tensor | None = None
        self._pure_rl_previous_orientation_wxyz: Tensor | None = None
        self._pure_rl_sensor_history: Tensor | None = None
        self._pure_rl_action_history: Tensor | None = None
        self._pure_rl_previous_reward_action: Tensor | None = None
        self._eval_pure_rl_cross_track_error_m: Tensor | None = None
        self._eval_pure_rl_height_error_m: Tensor | None = None
        self._eval_pure_rl_along_track_progress_m: Tensor | None = None
        self._eval_pure_rl_along_track_velocity_mps: Tensor | None = None
        self._eval_pure_rl_lateral_normal_velocity_mps: Tensor | None = None
        self._eval_pure_rl_vertical_normal_velocity_mps: Tensor | None = None
        self._eval_pure_rl_active_slope_rad: Tensor | None = None
        self._eval_pure_rl_reached_recovery: Tensor | None = None
        self._eval_pure_rl_tilt_rad: Tensor | None = None
        self._eval_pure_rl_angular_rate_rad_s: Tensor | None = None
        self._eval_pure_rl_actual_flap_frequency_hz: Tensor | None = None
        self._eval_pure_rl_frequency_limit_active: Tensor | None = None
        self._eval_pure_rl_tail_limit_active: Tensor | None = None
        self._eval_pure_rl_normalized_action_delta: Tensor | None = None
        self._eval_pure_rl_frequency_slew_hz_per_s: Tensor | None = None
        self._eval_pure_rl_frequency_governor_limited: Tensor | None = None
        self._eval_pure_rl_ground_termination: Tensor | None = None
        self._eval_pure_rl_tilt_termination: Tensor | None = None
        self._eval_pure_rl_cross_track_termination: Tensor | None = None
        self._eval_pure_rl_height_termination: Tensor | None = None
        self._pure_rl_termination_cfg = replace(
            PURE_RL_CURRICULUM1_TERMINATION_CONFIG,
            ground_height_m=float(cfg.terminate_ground_height),
            maximum_tilt_rad=math.radians(float(cfg.terminate_tilt_deg)),
            maximum_cross_track_error_m=float(cfg.terminate_abs_y),
            maximum_height_error_m=float(cfg.pure_rl_terminate_abs_height_error_m),
        )

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
        self._debug_last_wing_moment_b_about_base_com_nm: Tensor | None = None
        self._debug_last_tail_force_b: Tensor | None = None
        self._debug_last_tail_moment_b_about_base_com_nm: Tensor | None = None
        self._debug_last_force_b: Tensor | None = None
        self._debug_last_torque_b: Tensor | None = None
        self._debug_last_wing_force_link_n: Tensor | None = None
        self._debug_last_wing_moment_link_about_com_nm: Tensor | None = None
        self._debug_last_wing_aero_position_rad: Tensor | None = None
        self._debug_last_wing_aero_velocity_rad_s: Tensor | None = None
        self._debug_last_wing_aero_acceleration_rad_s2: Tensor | None = None
        self._debug_last_wing_actual_acceleration_rad_s2: Tensor | None = None
        self._debug_last_shadow_actual_accel_wing_force_b_n: Tensor | None = None
        self._debug_last_shadow_actual_accel_wing_moment_b_nm: Tensor | None = None
        self._debug_last_native_common_inertia_kg_m2: Tensor | None = None
        self._debug_last_native_inertia_torque_nm: Tensor | None = None
        self._debug_last_native_bias_torque_nm: Tensor | None = None
        self._debug_last_native_external_load_torque_nm: Tensor | None = None
        self._debug_last_native_constraint_torque_estimate_nm: Tensor | None = None
        self._debug_last_native_constraint_power_estimate_w: Tensor | None = None

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
        self._debug_last_delaurier_strip_loads: DeLaurierStripLoads | None = None
        self._debug_last_delaurier_strip_wrench: DeLaurierStripWrench | None = None
        self._debug_last_delaurier_twist_kinematics: DeLaurierTwistKinematics | None = None

        super().__init__(cfg, render_mode, **kwargs)
        if self._native_holonomic is not None:
            active_joint_count = int(self._native_holonomic.get_active_joint_count())
            expected_joint_count = len(self._native_holonomic_joint_paths)
            if active_joint_count != expected_joint_count:
                raise RuntimeError(
                    "PhysX did not instantiate the expected native holonomic constraints: "
                    f"active={active_joint_count}, expected={expected_joint_count}."
                )
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
        self._nominal_dof_position_limits_cpu = self._robot.root_physx_view.get_dof_limits().clone()

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
        self._debug_last_actual_rudder_rad = torch.zeros(self.num_envs, device=self.device)
        self._debug_last_actual_left_elevon_rad = torch.zeros(self.num_envs, device=self.device)
        self._debug_last_actual_right_elevon_rad = torch.zeros(self.num_envs, device=self.device)

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
        self._straight_line_heading_rad = torch.zeros(N, device=self.device)
        self._straight_line_tangent_w = torch.zeros(N, 3, device=self.device)
        self._straight_line_tangent_w[:, 0] = 1.0
        self._straight_line_normal_w = torch.zeros(N, 3, device=self.device)
        self._straight_line_normal_w[:, 1] = 1.0
        if self._pure_rl_longitudinal_stage is not None:
            self._pure_rl_longitudinal_path = PureRLLongitudinalPathBatch(
                task_id=torch.zeros(N, dtype=torch.int64, device=self.device),
                heading_rad=self._straight_line_heading_rad,
                signed_slope_rad=torch.zeros(N, device=self.device),
                entry_length_m=torch.zeros(N, device=self.device),
                slope_length_m=torch.zeros(N, device=self.device),
                initial_altitude_m=torch.zeros(N, device=self.device),
            )
        if bool(self.cfg.use_pure_rl_actor_observation):
            self._pure_rl_history_valid = torch.zeros(N, dtype=torch.bool, device=self.device)
            self._pure_rl_previous_orientation_wxyz = torch.zeros(N, 4, device=self.device)
            self._pure_rl_previous_orientation_wxyz[:, 0] = 1.0
            self._pure_rl_sensor_history = torch.zeros(
                N,
                PURE_RL_RAW_OBSERVATION_LAYOUT.sensor_history_steps,
                PURE_RL_RAW_OBSERVATION_LAYOUT.sensor_frame_dim,
                device=self.device,
            )
            self._pure_rl_action_history = torch.zeros(
                N,
                PURE_RL_RAW_OBSERVATION_LAYOUT.action_history_steps,
                PURE_RL_RAW_OBSERVATION_LAYOUT.action_dim,
                device=self.device,
            )
        if bool(self.cfg.use_pure_rl_curriculum1_reward):
            self._pure_rl_previous_reward_action = torch.zeros_like(self._actions)
            self._eval_pure_rl_cross_track_error_m = torch.zeros(N, device=self.device)
            self._eval_pure_rl_height_error_m = torch.zeros(N, device=self.device)
            self._eval_pure_rl_along_track_progress_m = torch.zeros(N, device=self.device)
            self._eval_pure_rl_along_track_velocity_mps = torch.zeros(N, device=self.device)
            if self._pure_rl_longitudinal_stage is not None:
                self._eval_pure_rl_lateral_normal_velocity_mps = torch.zeros(N, device=self.device)
                self._eval_pure_rl_vertical_normal_velocity_mps = torch.zeros(N, device=self.device)
                self._eval_pure_rl_active_slope_rad = torch.zeros(N, device=self.device)
                self._eval_pure_rl_reached_recovery = torch.zeros(N, dtype=torch.bool, device=self.device)
            self._eval_pure_rl_tilt_rad = torch.zeros(N, device=self.device)
            self._eval_pure_rl_angular_rate_rad_s = torch.zeros(N, device=self.device)
            self._eval_pure_rl_actual_flap_frequency_hz = torch.zeros(N, device=self.device)
            self._eval_pure_rl_frequency_limit_active = torch.zeros(N, dtype=torch.bool, device=self.device)
            self._eval_pure_rl_tail_limit_active = torch.zeros(N, dtype=torch.bool, device=self.device)
            self._eval_pure_rl_normalized_action_delta = torch.zeros(N, device=self.device)
            self._eval_pure_rl_frequency_slew_hz_per_s = torch.zeros(N, device=self.device)
            self._eval_pure_rl_frequency_governor_limited = torch.zeros(
                N, dtype=torch.bool, device=self.device
            )
            self._eval_pure_rl_ground_termination = torch.zeros(N, dtype=torch.bool, device=self.device)
            self._eval_pure_rl_tilt_termination = torch.zeros(N, dtype=torch.bool, device=self.device)
            self._eval_pure_rl_cross_track_termination = torch.zeros(N, dtype=torch.bool, device=self.device)
            self._eval_pure_rl_height_termination = torch.zeros(N, dtype=torch.bool, device=self.device)

        # debug caches
        self._debug_last_wing_force_b = torch.zeros(N, 3, device=self.device)
        self._debug_last_wing_moment_b_about_base_com_nm = torch.zeros(N, 3, device=self.device)
        self._debug_last_tail_force_b = torch.zeros(N, 3, device=self.device)
        self._debug_last_tail_moment_b_about_base_com_nm = torch.zeros(N, 3, device=self.device)
        self._debug_last_force_b = torch.zeros(N, 3, device=self.device)
        self._debug_last_torque_b = torch.zeros(N, 3, device=self.device)
        self._debug_last_wing_force_link_n = torch.zeros(N, 2, 3, device=self.device)
        self._debug_last_wing_moment_link_about_com_nm = torch.zeros(N, 2, 3, device=self.device)
        self._debug_last_wing_aero_position_rad = torch.zeros(N, 2, device=self.device)
        self._debug_last_wing_aero_velocity_rad_s = torch.zeros(N, 2, device=self.device)
        self._debug_last_wing_aero_acceleration_rad_s2 = torch.zeros(N, 2, device=self.device)
        self._debug_last_wing_actual_acceleration_rad_s2 = torch.zeros(N, 2, device=self.device)
        self._debug_last_shadow_actual_accel_wing_force_b_n = torch.zeros(N, 3, device=self.device)
        self._debug_last_shadow_actual_accel_wing_moment_b_nm = torch.zeros(N, 3, device=self.device)
        self._debug_last_native_common_inertia_kg_m2 = torch.zeros(N, device=self.device)
        self._debug_last_native_inertia_torque_nm = torch.zeros(N, device=self.device)
        self._debug_last_native_bias_torque_nm = torch.zeros(N, device=self.device)
        self._debug_last_native_external_load_torque_nm = torch.zeros(N, device=self.device)
        self._debug_last_native_constraint_torque_estimate_nm = torch.zeros(N, device=self.device)
        self._debug_last_native_constraint_power_estimate_w = torch.zeros(N, device=self.device)
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
        self._configure_plant_mass_properties()

        # wing phase/frequency
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._ideal_torque_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._freq = torch.full((self.num_envs,), float(self.cfg.min_flap_hz), device=self.device)
        self._phase_throttle = torch.zeros(self.num_envs, device=self.device)
        self._phase_target_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self._requested_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self._applied_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self._frequency_slew_hz_per_s = torch.zeros(self.num_envs, device=self.device)
        self._frequency_governor_limited = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._phase_acceleration_rad_s2 = torch.zeros(self.num_envs, device=self.device)

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
        if (
            validate_wing_drive_variant(self.cfg.wing_drive_variant)
            in {
                PRESCRIBED_COUPLED_WING_DRIVE,
                IDEAL_TORQUE_COUPLED_WING_DRIVE,
                SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
                IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
                NATIVE_HOLONOMIC_WING_DRIVE,
            }
            and abs(self._wing_mid_L + self._wing_mid_R) > 1.0e-6
        ):
            raise ValueError(
                f"{self.cfg.wing_drive_variant} requires mirrored wing joint midpoints whose sum is zero."
            )
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) == SINUSOIDAL_PHASE_SPEED_WING_DRIVE:
            standalone_phase_cfg = SinusoidalPhaseSpeedDriveConfig(
                amplitude_rad=self._wing_amp,
                max_frequency_hz=float(self.cfg.max_flap_hz),
            )
            self._sinusoidal_phase_drive_cfg = SinusoidalPhaseSpeedDriveConfig(
                amplitude_rad=self._wing_amp,
                max_frequency_hz=float(self.cfg.max_flap_hz),
                common_joint_inertia_kg_m2=0.0,
                constant_phase_inertia_kg_m2=float(self.cfg.sinusoidal_phase_inertia_kg_m2),
                phase_inertia_floor_ratio=standalone_phase_cfg.phase_inertia_floor_ratio,
                phase_viscous_damping_nm_s=0.0,
                frequency_settling_time_s=float(self.cfg.sinusoidal_speed_settling_time_s),
                frequency_damping_ratio=float(self.cfg.sinusoidal_speed_damping_ratio),
                effort_limit_nm=float(self.cfg.sinusoidal_effort_limit_nm),
            )
            phase_rate = 2.0 * math.pi * self._freq
            self._sinusoidal_phase_drive_state = SinusoidalPhaseDriveState(
                phase_rad=self._phase,
                phase_rate_rad_s=phase_rate,
                speed_error_integral_rad=torch.zeros_like(self._phase),
            )
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) in {
            IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
            NATIVE_HOLONOMIC_WING_DRIVE,
        }:
            self._ideal_inverse_phase_cfg = IdealInverseDynamicsPhaseDriveConfig(
                amplitude_rad=self._wing_amp,
                max_frequency_hz=float(self.cfg.max_flap_hz),
                frequency_settling_time_s=float(
                    self.cfg.ideal_inverse_frequency_settling_time_s
                ),
                tracking_natural_frequency_hz=float(
                    self.cfg.ideal_inverse_tracking_natural_frequency_hz
                ),
                tracking_damping_ratio=float(
                    self.cfg.ideal_inverse_tracking_damping_ratio
                ),
                effort_limit_nm=float(self.cfg.ideal_inverse_effort_limit_nm),
            )
            self._ideal_inverse_phase_state = IdealFrequencyPhaseState(
                phase_rad=self._phase,
                frequency_hz=self._freq,
            )

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
        self._initialize_runtime_state_estimation()

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

    def _configure_plant_mass_properties(self) -> None:
        """Apply the selected runtime plant mass partition."""

        plant_variant = validate_flapping_bot_plant_variant(self.cfg.plant_variant)
        if plant_variant == NEAR_SINGLE_RIGID_BODY_PLANT:
            self._override_appendage_mass_properties()
            self._override_total_mass_properties()
            self._override_base_body_com()
            self._override_base_body_inertia()
            return
        if plant_variant != MEASURED_WING_MULTIBODY_PLANT:
            raise RuntimeError(f"Unhandled plant variant: {plant_variant!r}.")

        env_ids = torch.arange(self.num_envs, device="cpu", dtype=torch.int64)
        masses = self._robot.root_physx_view.get_masses().clone()
        inertias = self._robot.root_physx_view.get_inertias().clone()
        coms = self._robot.root_physx_view.get_coms().clone()
        masses_new, inertias_new, coms_new = build_measured_wing_multibody_tensors(
            body_names=self._robot.body_names,
            masses_kg=masses,
            inertias_kg_m2=inertias,
            com_poses_link=coms,
            placeholder_mass_kg=float(self.cfg.measured_multibody_placeholder_mass_kg),
        )

        self._robot.root_physx_view.set_masses(masses_new, env_ids)
        self._robot.root_physx_view.set_inertias(inertias_new, env_ids)
        self._robot.root_physx_view.set_coms(coms_new, env_ids)
        self._mass_total = masses_new.sum(dim=1).to(device=self.device)
        self._robot.data.default_mass = masses_new.to(device=self.device)
        self._robot.data.default_inertia = inertias_new.to(device=self.device)

    def _override_base_body_com(self) -> None:
        """Optionally override the base-body COM offset in the base-link frame."""
        override_xyz = self.cfg.base_body_com_override_m
        override_x = self.cfg.base_body_com_override_x_m
        if override_xyz is None and override_x is None:
            return
        if len(self._base_body_ids) != 1:
            raise RuntimeError("Expected exactly one base_link body for COM override.")

        env_ids = torch.arange(self.num_envs, device="cpu", dtype=torch.int64)
        coms = self._robot.root_physx_view.get_coms().clone()
        base_id = int(self._base_body_ids[0])
        if override_xyz is not None:
            if len(override_xyz) != 3:
                raise ValueError("cfg.base_body_com_override_m must contain exactly three values when provided.")
            coms[:, base_id, 0:3] = torch.tensor(
                tuple(float(v) for v in override_xyz),
                dtype=coms.dtype,
                device=coms.device,
            )
        else:
            coms[:, base_id, 0] = float(override_x)
        self._robot.root_physx_view.set_coms(coms, env_ids)

    def _override_base_body_inertia(self) -> None:
        """Optionally override the base-body inertia diagonal about its COM in the base-link frame."""
        inertia_diag = self.cfg.base_body_inertia_diag_override_kg_m2
        if inertia_diag is None:
            return
        if len(inertia_diag) != 3:
            raise ValueError("cfg.base_body_inertia_diag_override_kg_m2 must contain exactly three values when provided.")
        if len(self._base_body_ids) != 1:
            raise RuntimeError("Expected exactly one base_link body for inertia override.")

        ixx, iyy, izz = (float(v) for v in inertia_diag)
        if ixx <= 0.0 or iyy <= 0.0 or izz <= 0.0:
            raise ValueError("cfg.base_body_inertia_diag_override_kg_m2 values must be positive.")

        env_ids = torch.arange(self.num_envs, device="cpu", dtype=torch.int64)
        inertias = self._robot.root_physx_view.get_inertias().clone()
        base_id = int(self._base_body_ids[0])
        inertia_flat = torch.tensor(
            (ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz),
            dtype=inertias.dtype,
            device=inertias.device,
        )
        inertias[:, base_id, :] = inertia_flat
        self._robot.root_physx_view.set_inertias(inertias, env_ids)

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
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) in {
            IDEAL_COUPLED_WING_DRIVE,
            IDEAL_TORQUE_COUPLED_WING_DRIVE,
            SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
            IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
            NATIVE_HOLONOMIC_WING_DRIVE,
        }:
            source_robot_path = f"{self.scene.env_prim_paths[0]}/Robot"
            apply_hard_opposed_wing_mimic(
                self._robot.stage,
                articulation_root_path=source_robot_path,
            )
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) == NATIVE_HOLONOMIC_WING_DRIVE:
            assert self._native_holonomic is not None
            self._native_holonomic_joint_paths = cloned_native_holonomic_joint_paths(
                self.scene.env_prim_paths
            )
            for env_prim_path, constraint_path in zip(
                self.scene.env_prim_paths,
                self._native_holonomic_joint_paths,
                strict=True,
            ):
                apply_native_holonomic_wing_constraint(
                    self._robot.stage,
                    articulation_root_path=f"{env_prim_path}/Robot",
                    constraint_prim_path=constraint_path,
                    native_module=self._native_holonomic,
                )
            missing_paths = [
                path
                for path in self._native_holonomic_joint_paths
                if not self._robot.stage.GetPrimAtPath(path)
            ]
            if missing_paths:
                raise RuntimeError(
                    "Failed to author per-environment native holonomic joints: "
                    f"{missing_paths[:3]}."
                )
            zeros = [0.0] * len(self._native_holonomic_joint_paths)
            self._native_holonomic.set_targets(
                self._native_holonomic_joint_paths,
                zeros,
                zeros,
            )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _runtime_requires_estimated_state(self) -> bool:
        assert self._teacher_state_inputs is not None
        return (
            self._teacher_state_inputs.teacher_state_source == "estimated"
            or self._teacher_state_inputs.policy_state_source == "estimated"
        )

    def _teacher_uses_estimated_state(self) -> bool:
        assert self._teacher_state_inputs is not None
        return self._teacher_state_inputs.teacher_state_source == "estimated"

    def _policy_uses_estimated_state(self) -> bool:
        assert self._teacher_state_inputs is not None
        return self._teacher_state_inputs.policy_state_source == "estimated"

    def _initialize_runtime_state_estimation(self) -> None:
        if not self._runtime_requires_estimated_state():
            return
        self._runtime_imu_provider = build_imu_provider(self._resolved_imu_source or self.cfg.imu_source)
        self._runtime_state_estimator = SensorStateEstimator(
            sensor_cfg=SensorSuiteCfg(),
            estimator_cfg=StateEstimatorCfg(),
            num_envs=int(self.num_envs),
            device=self.device,
            control_dt_s=float(self.step_dt),
        )
        self._runtime_imu_sensor = self._create_runtime_imu_sensor()

    def _create_runtime_imu_sensor(self):
        if getattr(self._runtime_imu_provider, "backend_name", "") != "isaacsim":
            return None

        try:
            from ...px4_like import IsaacSimImuSensorSpec
        except ModuleNotFoundError:
            from ...px4_like import IsaacSimImuSensorSpec

        body_name = str(self._robot.body_names[0]) if len(self._robot.body_names) > 0 else "base_link"
        prim_path = f"{self.cfg.robot.prim_path}/{body_name}"
        offset_pos_b = resolve_base_body_com_offset_b(self._robot, self._base_body_ids)
        sensor = self._runtime_imu_provider.create_sensor(
            IsaacSimImuSensorSpec(
                prim_path=prim_path,
                update_period=float(self.step_dt),
                offset_pos_b=offset_pos_b,
            )
        )
        if hasattr(sensor, "is_initialized") and not sensor.is_initialized:
            sensor._initialize_impl()
            sensor._is_initialized = True
        sensor.reset()
        sensor.update(dt=float(self.step_dt), force_recompute=True)
        return sensor

    def _reset_runtime_state_estimation(self, env_ids: Tensor | None = None) -> None:
        if self._runtime_state_estimator is None:
            return
        if self._runtime_imu_sensor is not None:
            self._runtime_imu_sensor.reset()
            self._runtime_imu_sensor.update(dt=float(self.step_dt), force_recompute=True)

        pos_local = self._robot.data.root_pos_w - self.scene.env_origins
        vel_w = self._robot.data.root_lin_vel_w
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        wind_w = self._wind_w if self._wind_w is not None else torch.zeros_like(vel_w)
        airspeed = torch.linalg.norm(vel_w - wind_w, dim=1)
        self._runtime_state_estimator.reset(
            env_ids=env_ids,
            pos_local_true=pos_local,
            vel_local_true=vel_w,
            roll_true=roll,
            pitch_true=pitch,
            yaw_true=yaw,
            airspeed_true=airspeed,
            wind_local_true=wind_w,
        )
        self._runtime_estimated_state = None
        self._runtime_estimator_diag = {}

    def _refresh_runtime_estimated_state(self) -> None:
        if self._runtime_state_estimator is None:
            return

        pos_local = self._robot.data.root_pos_w - self.scene.env_origins
        vel_w = self._robot.data.root_lin_vel_w
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel_b = self._robot.data.root_ang_vel_b
        wind_w = self._wind_w if self._wind_w is not None else torch.zeros_like(vel_w)
        airspeed = torch.linalg.norm(vel_w - wind_w, dim=1)

        imu_measurement = None
        if self._runtime_imu_sensor is not None and self._runtime_imu_provider is not None:
            self._runtime_imu_sensor.update(dt=float(self.step_dt), force_recompute=True)
            imu_measurement = self._runtime_imu_provider.build_from_sensor(self._runtime_imu_sensor)

        self._runtime_estimated_state, self._runtime_estimator_diag = self._runtime_state_estimator.step(
            pos_local_true=pos_local,
            vel_local_true=vel_w,
            roll_true=roll,
            pitch_true=pitch,
            yaw_true=yaw,
            ang_vel_body_true=ang_vel_b,
            airspeed_true=airspeed,
            imu_measurement=imu_measurement,
        )

    def _get_runtime_estimated_quat_w(self) -> Tensor:
        if self._runtime_estimated_state is None:
            raise RuntimeError("Estimated runtime state is unavailable.")
        return quat_from_euler_xyz(
            roll=self._runtime_estimated_state["roll"],
            pitch=self._runtime_estimated_state["pitch"],
            yaw=self._runtime_estimated_state["yaw"],
        )

    def _get_runtime_teacher_inputs(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._teacher_uses_estimated_state() and self._runtime_estimated_state is not None:
            estimated = self._runtime_estimated_state
            return (
                estimated["pos_local"],
                estimated["ground_vel_local"],
                estimated["roll"],
                estimated["pitch"],
                estimated["yaw"],
                estimated["ang_vel_body"],
                estimated["wind_xy"],
            )

        pos_local = self._robot.data.root_pos_w - self.scene.env_origins
        ground_vel_local = self._robot.data.root_lin_vel_w
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel_body = self._robot.data.root_ang_vel_b
        if self._teacher_state_inputs is not None and self._teacher_state_inputs.teacher_uses_truth_wind:
            wind_xy = self._wind_w[:, 0:2]
        else:
            wind_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=ground_vel_local.dtype)
        return pos_local, ground_vel_local, roll, pitch, yaw, ang_vel_body, wind_xy

    def _get_policy_observation_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self._policy_uses_estimated_state() and self._runtime_estimated_state is not None:
            quat_est = self._get_runtime_estimated_quat_w()
            gravity_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=quat_est.dtype)
            gravity_w[:, 2] = -1.0
            return (
                self._runtime_estimated_state["pos_local"],
                quat_apply_inverse(quat_est, self._runtime_estimated_state["ground_vel_local"]),
                self._runtime_estimated_state["ang_vel_body"],
                quat_apply_inverse(quat_est, gravity_w),
            )

        return (
            self._robot.data.root_pos_w - self.scene.env_origins,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
        )

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

        state_inputs = resolve_teacher_state_inputs(
            self.cfg.teacher_state_source,
            self.cfg.policy_state_source,
            self.cfg.teacher_guidance_use_wind_truth,
        )
        self._teacher_state_inputs = state_inputs
        pos_local, ground_vel_local, roll, pitch, yaw, ang_vel_body, wind_xy = self._get_runtime_teacher_inputs()

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
            self._act_lpf.add_(alpha * (act_exec - self._act_lpf))
        else:
            self._act_lpf.copy_(act_exec)
        # slew-rate limit
        if self.cfg.act_rate_limit_per_s > 0.0:
            max_delta = self.cfg.act_rate_limit_per_s * float(self.step_dt)
            delta = torch.clamp(self._act_lpf - self._act_cmd, min=-max_delta, max=max_delta)
            self._act_cmd.add_(delta)
        else:
            self._act_cmd.copy_(self._act_lpf)

        # frequency from action 0 in [min_flap_hz, max_flap_hz]
        a0 = self._act_cmd[:, 0]
        requested_frequency_hz = normalized_action_to_frequency_hz(
            a0,
            minimum_frequency_hz=float(self.cfg.min_flap_hz),
            maximum_frequency_hz=float(self.cfg.max_flap_hz),
        )
        assert self._requested_frequency_hz is not None
        assert self._applied_frequency_hz is not None
        assert self._frequency_slew_hz_per_s is not None
        assert self._frequency_governor_limited is not None
        self._requested_frequency_hz.copy_(requested_frequency_hz)
        if bool(self.cfg.frequency_governor_enabled):
            governor_step = apply_frequency_slew_governor(
                requested_frequency_hz,
                previous_frequency_hz=self._applied_frequency_hz,
                policy_step_dt_s=float(self.step_dt),
                maximum_rise_rate_hz_per_s=float(
                    self.cfg.frequency_governor_maximum_rise_rate_hz_per_s
                ),
                maximum_fall_rate_hz_per_s=float(
                    self.cfg.frequency_governor_maximum_fall_rate_hz_per_s
                ),
            )
            frequency_setpoint = governor_step.applied_frequency_hz
            self._frequency_slew_hz_per_s.copy_(governor_step.slew_hz_per_s)
            self._frequency_governor_limited.copy_(governor_step.limited)
        else:
            frequency_setpoint = requested_frequency_hz
            self._frequency_slew_hz_per_s.copy_(
                (frequency_setpoint - self._applied_frequency_hz) / float(self.step_dt)
            )
            self._frequency_governor_limited.zero_()
        self._applied_frequency_hz.copy_(frequency_setpoint)
        self._act_cmd[:, 0].copy_(
            frequency_hz_to_normalized_action(
                frequency_setpoint,
                minimum_frequency_hz=float(self.cfg.min_flap_hz),
                maximum_frequency_hz=float(self.cfg.max_flap_hz),
            )
        )
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) in {
            SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
            IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
            NATIVE_HOLONOMIC_WING_DRIVE,
        }:
            assert self._phase_throttle is not None
            assert self._phase_target_frequency_hz is not None
            self._phase_target_frequency_hz.copy_(frequency_setpoint)
            self._phase_throttle.copy_(
                torch.clamp(
                    frequency_setpoint / float(self.cfg.max_flap_hz),
                    min=0.0,
                    max=1.0,
                )
            )
        else:
            self._freq = frequency_setpoint

        action_interface = validate_action_interface(self.cfg.action_interface)
        if action_interface == DIRECT_TAIL_SURFACE_ACTION:
            self._rudder_cmd = normalized_action_to_joint_position(
                self._act_cmd[:, 1],
                lower_limit_rad=self._joint_lower_limits[self._IDX_RUDDER],
                upper_limit_rad=self._joint_upper_limits[self._IDX_RUDDER],
            )
            self._left_elevon_cmd = normalized_action_to_joint_position(
                self._act_cmd[:, 2],
                lower_limit_rad=self._joint_lower_limits[self._IDX_LEFT_TAIL],
                upper_limit_rad=self._joint_upper_limits[self._IDX_LEFT_TAIL],
            )
            self._right_elevon_cmd = normalized_action_to_joint_position(
                self._act_cmd[:, 3],
                lower_limit_rad=self._joint_lower_limits[self._IDX_RIGHT_TAIL],
                upper_limit_rad=self._joint_upper_limits[self._IDX_RIGHT_TAIL],
            )
            self._elevon_pitch_cmd = 0.5 * (
                self._left_elevon_cmd + self._right_elevon_cmd
            )
            self._elevon_roll_cmd = 0.5 * (
                self._left_elevon_cmd - self._right_elevon_cmd
            )
        else:
            # Established controller-facing [rudder, pitch, roll] allocation.
            rud_lim = torch.deg2rad(
                torch.tensor(float(self.cfg.rudder_max_deg), device=self.device)
            )
            self._rudder_cmd = (rud_lim * self._act_cmd[:, 1]).clamp(-rud_lim, rud_lim)

            elevon_lim = torch.deg2rad(
                torch.tensor(float(self.cfg.elevon_max_deg), device=self.device)
            )
            self._elevon_pitch_cmd = (elevon_lim * self._act_cmd[:, 2]).clamp(
                -elevon_lim, elevon_lim
            )
            self._elevon_roll_cmd = (elevon_lim * self._act_cmd[:, 3]).clamp(
                -elevon_lim, elevon_lim
            )

            trim = torch.deg2rad(
                torch.tensor(float(self.cfg.elevon_trim_deg), device=self.device)
            )
            mixed_pitch = float(self.cfg.elevon_pitch_mix) * self._elevon_pitch_cmd
            mixed_roll = float(self.cfg.elevon_roll_mix) * self._elevon_roll_cmd
            left_raw = trim + mixed_pitch + mixed_roll
            right_raw = trim + mixed_pitch - mixed_roll

            l_lower = torch.maximum(self._joint_lower_limits[self._IDX_LEFT_TAIL], -elevon_lim)
            l_upper = torch.minimum(self._joint_upper_limits[self._IDX_LEFT_TAIL], elevon_lim)
            r_lower = torch.maximum(self._joint_lower_limits[self._IDX_RIGHT_TAIL], -elevon_lim)
            r_upper = torch.minimum(self._joint_upper_limits[self._IDX_RIGHT_TAIL], elevon_lim)
            self._left_elevon_cmd = left_raw.clamp(l_lower, l_upper)
            self._right_elevon_cmd = right_raw.clamp(r_lower, r_upper)

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
        wing_drive_variant = validate_wing_drive_variant(self.cfg.wing_drive_variant)
        ideal_torque_drive_effort: Tensor | None = None
        sinusoidal_phase_step: SinusoidalPhaseDriveStep | None = None
        ideal_inverse_phase_step: IdealFrequencyPhaseStep | None = None
        ideal_inverse_desired_acceleration: Tensor | None = None
        next_phase = advance_flap_phase(
            phase=self._phase,
            freq_hz=self._freq,
            physics_dt_s=float(self.physics_dt),
            freeze_steps=self._freeze_steps,
        )
        if wing_drive_variant in {
            IDEAL_TORQUE_COUPLED_WING_DRIVE,
            SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
            IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
            NATIVE_HOLONOMIC_WING_DRIVE,
        }:
            reference_phase = self._phase
        else:
            self._phase = next_phase
            reference_phase = self._phase

        # Cache commanded wing kinematics for the articulation and DeLaurier
        # backend. The default phase zero is the neutral pose starting upstroke.
        self._q_cmd, self._qd_cmd, self._qdd_cmd = compute_prescribed_flap_kinematics(
            phase=reference_phase,
            freq_hz=self._freq,
            amplitude_rad=self._wing_amp,
            convention=self.cfg.flap_phase_convention,
        )
        if wing_drive_variant == SINUSOIDAL_PHASE_SPEED_WING_DRIVE:
            assert self._sinusoidal_phase_drive_state is not None
            assert self._phase_acceleration_rad_s2 is not None
            phase_kinematics = compute_opposed_wing_kinematics(
                phase_rad=self._sinusoidal_phase_drive_state.phase_rad,
                phase_rate_rad_s=self._sinusoidal_phase_drive_state.phase_rate_rad_s,
                phase_acceleration_rad_s2=self._phase_acceleration_rad_s2,
                amplitude_rad=self._wing_amp,
                left_joint_mid_rad=self._wing_mid_L,
                right_joint_mid_rad=self._wing_mid_R,
            )
            self._q_cmd = phase_kinematics.common_position_rad
            self._qd_cmd = phase_kinematics.common_velocity_rad_s
            self._qdd_cmd = phase_kinematics.common_acceleration_rad_s2
        if wing_drive_variant in {
            IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
            NATIVE_HOLONOMIC_WING_DRIVE,
        }:
            assert self._ideal_inverse_phase_state is not None
            assert self._ideal_inverse_phase_cfg is not None
            assert self._phase_throttle is not None
            ideal_inverse_phase_step = step_ideal_frequency_phase(
                state=self._ideal_inverse_phase_state,
                throttle_01=self._phase_throttle,
                physics_dt_s=float(self.physics_dt),
                config=self._ideal_inverse_phase_cfg,
                left_joint_mid_rad=self._wing_mid_L,
                right_joint_mid_rad=self._wing_mid_R,
            )
            self._q_cmd = ideal_inverse_phase_step.kinematics.common_position_rad
            self._qd_cmd = ideal_inverse_phase_step.kinematics.common_velocity_rad_s
            self._qdd_cmd = ideal_inverse_phase_step.kinematics.common_acceleration_rad_s2
        if wing_drive_variant == IDEAL_TORQUE_COUPLED_WING_DRIVE:
            assert self._ideal_torque_elapsed_s is not None
            ramp_duration_s = float(self.cfg.ideal_torque_ramp_cycles) / self._freq
            self._q_cmd, self._qd_cmd, self._qdd_cmd = apply_quintic_amplitude_ramp(
                position_rad=self._q_cmd,
                velocity_rad_s=self._qd_cmd,
                acceleration_rad_s2=self._qdd_cmd,
                elapsed_time_s=self._ideal_torque_elapsed_s,
                duration_s=ramp_duration_s,
            )

        left_cmd, right_cmd, left_qd_cmd, right_qd_cmd = map_symmetric_flap_coordinate_to_joint_space(
            flap_position_rad=self._q_cmd,
            flap_velocity_rad_s=self._qd_cmd,
            left_joint_mid_rad=self._wing_mid_L,
            right_joint_mid_rad=self._wing_mid_R,
        )

        jt = self._joint_targets
        jt[:, self._IDX_LEFT_WING] = left_cmd
        jt[:, self._IDX_RIGHT_WING] = right_cmd
        jt[:, self._IDX_RUDDER] = self._rudder_cmd
        # visualization joints for tail (treated as left/right elevons)
        jt[:, self._IDX_LEFT_TAIL] = self._left_elevon_cmd
        jt[:, self._IDX_RIGHT_TAIL] = self._right_elevon_cmd
        if bool(self.cfg.use_kinematic_joint_override):
            jvel = torch.zeros_like(jt)
            jvel[:, self._IDX_LEFT_WING] = left_qd_cmd
            jvel[:, self._IDX_RIGHT_WING] = right_qd_cmd
            self._robot.write_joint_state_to_sim(jt, jvel, joint_ids=self._joint_ids)
        elif wing_drive_variant == IDEAL_COUPLED_WING_DRIVE:
            left_driver_joint_id = [int(self._joint_ids[self._IDX_LEFT_WING])]
            self._robot.set_joint_position_target(
                jt[:, self._IDX_LEFT_WING].unsqueeze(-1),
                joint_ids=left_driver_joint_id,
            )
            self._robot.set_joint_velocity_target(
                left_qd_cmd.unsqueeze(-1),
                joint_ids=left_driver_joint_id,
            )
            non_wing_local_ids = [self._IDX_RUDDER, self._IDX_LEFT_TAIL, self._IDX_RIGHT_TAIL]
            non_wing_joint_ids = [int(self._joint_ids[index]) for index in non_wing_local_ids]
            self._robot.set_joint_position_target(
                jt[:, non_wing_local_ids],
                joint_ids=non_wing_joint_ids,
            )
            q_left = self._robot.data.joint_pos[:, left_driver_joint_id[0]]
            q_right = self._robot.data.joint_pos[:, int(self._joint_ids[self._IDX_RIGHT_WING])]
            qd_left = self._robot.data.joint_vel[:, left_driver_joint_id[0]]
            drive_torque = self._robot.data.applied_torque[:, left_driver_joint_id[0]]
            self.extras.setdefault("log", {}).update(
                {
                    "WingDrive/mean_abs_sync_error_rad": float(torch.mean(torch.abs(q_left + q_right)).item()),
                    "WingDrive/mean_abs_tracking_error_rad": float(torch.mean(torch.abs(q_left - left_cmd)).item()),
                    "WingDrive/mean_abs_velocity_error_rad_s": float(
                        torch.mean(torch.abs(qd_left - left_qd_cmd)).item()
                    ),
                    "WingDrive/mean_abs_driver_torque_Nm": float(torch.mean(torch.abs(drive_torque)).item()),
                }
            )
        elif wing_drive_variant == PRESCRIBED_COUPLED_WING_DRIVE:
            assert self._nominal_dof_position_limits_cpu is not None
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            prescribed_limits_cpu = build_prescribed_dof_position_limits(
                nominal_limits_rad=self._nominal_dof_position_limits_cpu,
                target_position_rad=left_cmd.detach().to(
                    device=self._nominal_dof_position_limits_cpu.device,
                    dtype=self._nominal_dof_position_limits_cpu.dtype,
                ),
                joint_id=left_joint_id,
                half_width_rad=float(self.cfg.prescribed_joint_limit_half_width_rad),
            )
            prescribed_limits_cpu = build_prescribed_dof_position_limits(
                nominal_limits_rad=prescribed_limits_cpu,
                target_position_rad=right_cmd.detach().to(
                    device=prescribed_limits_cpu.device,
                    dtype=prescribed_limits_cpu.dtype,
                ),
                joint_id=right_joint_id,
                half_width_rad=float(self.cfg.prescribed_joint_limit_half_width_rad),
            )
            physx_env_ids_cpu = torch.arange(self.num_envs, dtype=torch.int32, device="cpu")
            self._robot.root_physx_view.set_dof_limits(
                prescribed_limits_cpu,
                indices=physx_env_ids_cpu,
            )
            non_wing_local_ids = [self._IDX_RUDDER, self._IDX_LEFT_TAIL, self._IDX_RIGHT_TAIL]
            non_wing_joint_ids = [int(self._joint_ids[index]) for index in non_wing_local_ids]
            self._robot.set_joint_position_target(
                jt[:, non_wing_local_ids],
                joint_ids=non_wing_joint_ids,
            )
            q_left = self._robot.data.joint_pos[:, left_joint_id]
            q_right = self._robot.data.joint_pos[:, right_joint_id]
            qd_left = self._robot.data.joint_vel[:, left_joint_id]
            self.extras.setdefault("log", {}).update(
                {
                    "WingDrive/mean_abs_sync_error_rad": float(torch.mean(torch.abs(q_left + q_right)).item()),
                    "WingDrive/mean_abs_tracking_error_rad": float(torch.mean(torch.abs(q_left - left_cmd)).item()),
                    "WingDrive/mean_abs_velocity_error_rad_s": float(
                        torch.mean(torch.abs(qd_left - left_qd_cmd)).item()
                    ),
                }
            )
        elif wing_drive_variant == NATIVE_HOLONOMIC_WING_DRIVE:
            assert ideal_inverse_phase_step is not None
            assert self._native_holonomic is not None
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            self._native_holonomic.set_targets(
                self._native_holonomic_joint_paths,
                left_cmd.detach().to(device="cpu", dtype=torch.float64).tolist(),
                left_qd_cmd.detach().to(device="cpu", dtype=torch.float64).tolist(),
            )
            non_wing_local_ids = [
                self._IDX_RUDDER,
                self._IDX_LEFT_TAIL,
                self._IDX_RIGHT_TAIL,
            ]
            non_wing_joint_ids = [int(self._joint_ids[index]) for index in non_wing_local_ids]
            self._robot.set_joint_position_target(
                jt[:, non_wing_local_ids],
                joint_ids=non_wing_joint_ids,
            )
            common_position = self._robot.data.joint_pos[:, left_joint_id] - float(self._wing_mid_L)
            common_velocity = self._robot.data.joint_vel[:, left_joint_id]
            q_right_physical = -(
                self._robot.data.joint_pos[:, right_joint_id] - float(self._wing_mid_R)
            )
            self.extras.setdefault("log", {}).update(
                {
                    "WingDrive/mean_abs_sync_error_rad": float(
                        torch.mean(torch.abs(common_position - q_right_physical)).item()
                    ),
                    "WingDrive/mean_abs_tracking_error_rad": float(
                        torch.mean(torch.abs(common_position - self._q_cmd)).item()
                    ),
                    "WingDrive/mean_abs_velocity_error_rad_s": float(
                        torch.mean(torch.abs(common_velocity - self._qd_cmd)).item()
                    ),
                    "WingDrive/actual_frequency_hz": float(
                        torch.mean(self._ideal_inverse_phase_state.frequency_hz).item()
                    ),
                    "WingDrive/target_frequency_hz": float(
                        torch.mean(ideal_inverse_phase_step.target_frequency_hz).item()
                    ),
                }
            )
        elif wing_drive_variant == IDEAL_TORQUE_COUPLED_WING_DRIVE:
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            common_position = self._robot.data.joint_pos[:, left_joint_id] - float(self._wing_mid_L)
            common_velocity = self._robot.data.joint_vel[:, left_joint_id]
            ideal_torque_drive_effort = compute_ideal_torque_drive_effort(
                position_rad=common_position,
                velocity_rad_s=common_velocity,
                reference_position_rad=self._q_cmd,
                reference_velocity_rad_s=self._qd_cmd,
                reference_acceleration_rad_s2=self._qdd_cmd,
                equivalent_inertia_kg_m2=float(self.cfg.ideal_torque_equivalent_inertia_kg_m2),
                natural_frequency_hz=IDEAL_TORQUE_NATURAL_FREQUENCY_HZ,
                damping_ratio=IDEAL_TORQUE_DAMPING_RATIO,
                effort_limit_nm=IDEAL_DRIVER_EFFORT_LIMIT_NM,
                physics_dt_s=float(self.physics_dt),
            )
            non_wing_local_ids = [self._IDX_RUDDER, self._IDX_LEFT_TAIL, self._IDX_RIGHT_TAIL]
            non_wing_joint_ids = [int(self._joint_ids[index]) for index in non_wing_local_ids]
            self._robot.set_joint_position_target(
                jt[:, non_wing_local_ids],
                joint_ids=non_wing_joint_ids,
            )
            q_right_physical = -(
                self._robot.data.joint_pos[:, right_joint_id] - float(self._wing_mid_R)
            )
            self.extras.setdefault("log", {}).update(
                {
                    "WingDrive/mean_abs_sync_error_rad": float(
                        torch.mean(torch.abs(common_position - q_right_physical)).item()
                    ),
                    "WingDrive/mean_abs_tracking_error_rad": float(
                        torch.mean(torch.abs(common_position - self._q_cmd)).item()
                    ),
                    "WingDrive/mean_abs_velocity_error_rad_s": float(
                        torch.mean(torch.abs(common_velocity - self._qd_cmd)).item()
                    ),
                    "WingDrive/mean_abs_driver_torque_Nm": float(
                        torch.mean(torch.abs(ideal_torque_drive_effort)).item()
                    ),
                }
            )
        elif wing_drive_variant == SINUSOIDAL_PHASE_SPEED_WING_DRIVE:
            assert self._sinusoidal_phase_drive_state is not None
            assert self._sinusoidal_phase_drive_cfg is not None
            assert self._phase_throttle is not None
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            common_position = (
                self._robot.data.joint_pos[:, left_joint_id] - float(self._wing_mid_L)
            )
            common_velocity = self._robot.data.joint_vel[:, left_joint_id]
            constraint = compute_sinusoidal_constraint_effort(
                actual_common_position_rad=common_position,
                actual_common_velocity_rad_s=common_velocity,
                reference_common_position_rad=self._q_cmd,
                reference_common_velocity_rad_s=self._qd_cmd,
                phase_rad=self._sinusoidal_phase_drive_state.phase_rad,
                amplitude_rad=self._wing_amp,
                equivalent_common_inertia_kg_m2=float(
                    self.cfg.ideal_torque_equivalent_inertia_kg_m2
                ),
                natural_frequency_hz=float(
                    self.cfg.sinusoidal_constraint_natural_frequency_hz
                ),
                damping_ratio=float(self.cfg.sinusoidal_constraint_damping_ratio),
                effort_limit_nm=float(self.cfg.sinusoidal_effort_limit_nm),
                physics_dt_s=float(self.physics_dt),
            )
            sinusoidal_phase_step = step_sinusoidal_phase_speed_drive(
                state=self._sinusoidal_phase_drive_state,
                throttle_01=self._phase_throttle,
                common_joint_external_torque_nm=-constraint.common_wing_effort_nm,
                physics_dt_s=float(self.physics_dt),
                config=self._sinusoidal_phase_drive_cfg,
                left_joint_mid_rad=self._wing_mid_L,
                right_joint_mid_rad=self._wing_mid_R,
            )
            self._qdd_cmd = sinusoidal_phase_step.kinematics.common_acceleration_rad_s2
            non_wing_local_ids = [
                self._IDX_RUDDER,
                self._IDX_LEFT_TAIL,
                self._IDX_RIGHT_TAIL,
            ]
            non_wing_joint_ids = [
                int(self._joint_ids[index]) for index in non_wing_local_ids
            ]
            self._robot.set_joint_position_target(
                jt[:, non_wing_local_ids],
                joint_ids=non_wing_joint_ids,
            )
            self._robot.set_joint_effort_target(
                constraint.common_wing_effort_nm.unsqueeze(-1),
                joint_ids=[left_joint_id],
            )
            self._robot.set_joint_effort_target(
                torch.zeros_like(constraint.common_wing_effort_nm).unsqueeze(-1),
                joint_ids=[right_joint_id],
            )
            q_right_physical = -(
                self._robot.data.joint_pos[:, right_joint_id] - float(self._wing_mid_R)
            )
            constraint_power_residual = (
                constraint.common_wing_effort_nm * common_velocity
                + constraint.phase_reaction_torque_nm
                * self._sinusoidal_phase_drive_state.phase_rate_rad_s
                - constraint.common_wing_effort_nm
                * constraint.velocity_error_rad_s
            )
            self.extras.setdefault("log", {}).update(
                {
                    "WingDrive/mean_abs_sync_error_rad": float(
                        torch.mean(torch.abs(common_position - q_right_physical)).item()
                    ),
                    "WingDrive/mean_abs_tracking_error_rad": float(
                        torch.mean(torch.abs(constraint.position_error_rad)).item()
                    ),
                    "WingDrive/mean_abs_velocity_error_rad_s": float(
                        torch.mean(torch.abs(constraint.velocity_error_rad_s)).item()
                    ),
                    "WingDrive/mean_abs_driver_torque_Nm": float(
                        torch.mean(torch.abs(constraint.common_wing_effort_nm)).item()
                    ),
                    "WingDrive/actual_frequency_hz": float(
                        torch.mean(sinusoidal_phase_step.actual_frequency_hz).item()
                    ),
                    "WingDrive/target_frequency_hz": float(
                        torch.mean(sinusoidal_phase_step.target_frequency_hz).item()
                    ),
                    "WingDrive/constraint_saturation_fraction": float(
                        constraint.saturated.to(dtype=torch.float32).mean().item()
                    ),
                    "WingDrive/phase_drive_saturation_fraction": float(
                        sinusoidal_phase_step.drive_saturated.to(dtype=torch.float32)
                        .mean()
                        .item()
                    ),
                    "WingDrive/constraint_power_residual_W": float(
                        torch.mean(torch.abs(constraint_power_residual)).item()
                    ),
                }
            )
        elif wing_drive_variant == IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE:
            assert ideal_inverse_phase_step is not None
            assert self._ideal_inverse_phase_cfg is not None
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            common_position = (
                self._robot.data.joint_pos[:, left_joint_id] - float(self._wing_mid_L)
            )
            common_velocity = self._robot.data.joint_vel[:, left_joint_id]
            ideal_inverse_desired_acceleration = compute_desired_common_acceleration(
                actual_position_rad=common_position,
                actual_velocity_rad_s=common_velocity,
                reference_position_rad=self._q_cmd,
                reference_velocity_rad_s=self._qd_cmd,
                reference_acceleration_rad_s2=self._qdd_cmd,
                natural_frequency_hz=self._ideal_inverse_phase_cfg.tracking_natural_frequency_hz,
                damping_ratio=self._ideal_inverse_phase_cfg.tracking_damping_ratio,
                physics_dt_s=float(self.physics_dt),
            )
            non_wing_local_ids = [
                self._IDX_RUDDER,
                self._IDX_LEFT_TAIL,
                self._IDX_RIGHT_TAIL,
            ]
            non_wing_joint_ids = [
                int(self._joint_ids[index]) for index in non_wing_local_ids
            ]
            self._robot.set_joint_position_target(
                jt[:, non_wing_local_ids],
                joint_ids=non_wing_joint_ids,
            )
            q_right_physical = -(
                self._robot.data.joint_pos[:, right_joint_id] - float(self._wing_mid_R)
            )
            self.extras.setdefault("log", {}).update(
                {
                    "WingDrive/mean_abs_sync_error_rad": float(
                        torch.mean(torch.abs(common_position - q_right_physical)).item()
                    ),
                    "WingDrive/mean_abs_tracking_error_rad": float(
                        torch.mean(torch.abs(common_position - self._q_cmd)).item()
                    ),
                    "WingDrive/mean_abs_velocity_error_rad_s": float(
                        torch.mean(torch.abs(common_velocity - self._qd_cmd)).item()
                    ),
                    "WingDrive/actual_frequency_hz": float(
                        torch.mean(self._ideal_inverse_phase_state.frequency_hz).item()
                    ),
                    "WingDrive/target_frequency_hz": float(
                        torch.mean(ideal_inverse_phase_step.target_frequency_hz).item()
                    ),
                }
            )
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

        controlled_joint_position = self._robot.data.joint_pos[:, self._joint_ids]
        actual_rudder_rad = controlled_joint_position[:, self._IDX_RUDDER]
        actual_left_elevon_rad = controlled_joint_position[:, self._IDX_LEFT_TAIL]
        actual_right_elevon_rad = controlled_joint_position[:, self._IDX_RIGHT_TAIL]
        self._debug_last_actual_rudder_rad.copy_(actual_rudder_rad)
        self._debug_last_actual_left_elevon_rad.copy_(actual_left_elevon_rad)
        self._debug_last_actual_right_elevon_rad.copy_(actual_right_elevon_rad)

        if bool(self.cfg.enable_tail_aero):
            tail_deflection_source = validate_tail_aero_deflection_source(
                self.cfg.tail_aero_deflection_source
            )
            if tail_deflection_source == ACTUAL_JOINT_TAIL_AERO_DEFLECTION:
                tail_rudder_rad = actual_rudder_rad
                tail_left_elevon_rad = actual_left_elevon_rad
                tail_right_elevon_rad = actual_right_elevon_rad
            else:
                tail_rudder_rad = self._rudder_cmd
                tail_left_elevon_rad = self._left_elevon_cmd
                tail_right_elevon_rad = self._right_elevon_cmd
            ele_bias = math.radians(float(self.cfg.tail_elevator_bias_deg))
            base_body_com_pos_b = self._robot.data.body_com_pos_b[:, int(self._base_body_ids[0]), :]
            f_tail, tau_tail = self._tail_model.compute_wrench(
                root_lin_vel_b=v_air_b,
                root_ang_vel_b=w_b,
                left_elevon_rad=tail_left_elevon_rad + ele_bias,
                right_elevon_rad=tail_right_elevon_rad + ele_bias,
                rudder_rad=tail_rudder_rad,
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

        delaurier_wrench: _DeLaurierWingWrenchResult | None = None
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
                delaurier_wrench = self._compute_wing_delaurier_wrench(v_air_b)
                f_w_sum = delaurier_wrench.net_force_b_n
                tau_w_sum = delaurier_wrench.net_moment_b_about_base_com_nm
                if bool(self.cfg.delaurier_shadow_actual_acceleration):
                    shadow_wrench = self._compute_wing_delaurier_wrench(
                        v_air_b,
                        acceleration_source=ACTUAL_JOINT_ACCELERATION,
                        store_diagnostics=False,
                    )
                    self._debug_last_shadow_actual_accel_wing_force_b_n.copy_(
                        shadow_wrench.net_force_b_n
                    )
                    self._debug_last_shadow_actual_accel_wing_moment_b_nm.copy_(
                        shadow_wrench.net_moment_b_about_base_com_nm
                    )
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
        self._debug_last_wing_moment_b_about_base_com_nm.copy_(tau_w_sum)
        self._debug_last_tail_force_b.copy_(f_tail)
        self._debug_last_tail_moment_b_about_base_com_nm.copy_(tau_tail)
        self._debug_last_force_b.copy_(f_w_sum + f_tail + f_drag)
        self._debug_last_torque_b.copy_(tau_w_sum + tau_tail)

        wing_aero_coupling_mode = validate_wing_aero_coupling_mode(self.cfg.wing_aero_coupling_mode)
        if delaurier_wrench is None:
            wing_force_link = torch.zeros(
                self.num_envs,
                2,
                3,
                device=self.device,
                dtype=v_b.dtype,
            )
            wing_moment_link_about_com = torch.zeros_like(wing_force_link)
        else:
            wing_force_link = delaurier_wrench.force_link_n
            wing_moment_link_about_com = delaurier_wrench.moment_link_about_com_nm
        self._debug_last_wing_force_link_n.copy_(wing_force_link)
        self._debug_last_wing_moment_link_about_com_nm.copy_(wing_moment_link_about_com)

        if wing_aero_coupling_mode in {
            ACTUAL_PER_WING_LINK,
            PRESCRIBED_PER_WING_LINK,
            IDEAL_TORQUE_PER_WING_LINK,
            SINUSOIDAL_PHASE_PER_WING_LINK,
            IDEAL_INVERSE_DYNAMICS_PER_WING_LINK,
            NATIVE_HOLONOMIC_PER_WING_LINK,
        }:
            wing_link_aero_load_mode = validate_wing_link_aero_load_mode(
                self.cfg.wing_link_aero_load_mode
            )
            applied_wing_force_link = wing_force_link
            applied_wing_moment_link_about_com = wing_moment_link_about_com
            if wing_link_aero_load_mode == WING_LINK_FORCE_ONLY:
                applied_wing_moment_link_about_com = torch.zeros_like(
                    wing_moment_link_about_com
                )
            elif wing_link_aero_load_mode == WING_LINK_MOMENT_ONLY:
                applied_wing_force_link = torch.zeros_like(wing_force_link)
            # One call is required because Isaac Lab stores one global/local
            # wrench-frame flag for the complete articulation. Each row below
            # is expressed in its selected body's local FLU link frame.
            forces_link = torch.cat(
                ((f_tail + f_drag).unsqueeze(1), applied_wing_force_link),
                dim=1,
            )
            body_ids = [int(self._base_body_ids[0]), *(int(index) for index in self._wing_body_ids)]
            torques_link_about_com = torch.cat(
                (
                    tau_tail.unsqueeze(1),
                    applied_wing_moment_link_about_com,
                ),
                dim=1,
            )
            self._robot.set_external_force_and_torque(
                forces=forces_link,
                torques=torques_link_about_com,
                body_ids=body_ids,
                is_global=False,
            )
        else:
            f_sum = (f_w_sum + f_tail + f_drag).unsqueeze(1)  # (N,1,3)
            t_sum = (tau_w_sum + tau_tail).unsqueeze(1)  # (N,1,3)
            self._robot.set_external_force_and_torque(
                forces=f_sum,
                torques=t_sum,
                body_ids=self._base_body_ids,
                is_global=False,
            )
        if (
            wing_drive_variant == NATIVE_HOLONOMIC_WING_DRIVE
            and bool(self.cfg.native_holonomic_load_diagnostics)
        ):
            self._update_native_holonomic_load_diagnostics(
                root_quaternion_w=quat_w,
                net_external_force_b_n=f_w_sum + f_tail + f_drag,
                net_external_moment_b_about_base_com_nm=tau_w_sum + tau_tail,
                delaurier_wrench=delaurier_wrench,
            )
        if wing_drive_variant == IDEAL_TORQUE_COUPLED_WING_DRIVE:
            assert ideal_torque_drive_effort is not None
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            if delaurier_wrench is not None:
                common_aero_torque = compute_common_aerodynamic_hinge_torque(
                    force_link_n=delaurier_wrench.force_link_n,
                    moment_link_about_com_nm=delaurier_wrench.moment_link_about_com_nm,
                    wing_com_position_link_m=self._robot.data.body_com_pos_b[:, self._wing_body_ids, :],
                )
                ideal_torque_drive_effort = (
                    ideal_torque_drive_effort
                    - float(self.cfg.ideal_torque_aero_feedforward_scale) * common_aero_torque
                )
            ideal_torque_drive_effort = torch.clamp(
                ideal_torque_drive_effort,
                min=-IDEAL_DRIVER_EFFORT_LIMIT_NM,
                max=IDEAL_DRIVER_EFFORT_LIMIT_NM,
            )
            self._robot.set_joint_effort_target(
                ideal_torque_drive_effort.unsqueeze(-1),
                joint_ids=[left_joint_id],
            )
            self._robot.set_joint_effort_target(
                torch.zeros_like(ideal_torque_drive_effort).unsqueeze(-1),
                joint_ids=[right_joint_id],
            )
            self.extras.setdefault("log", {})["WingDrive/mean_abs_driver_torque_Nm"] = float(
                torch.mean(torch.abs(ideal_torque_drive_effort)).item()
            )
            self._phase = next_phase
            self._ideal_torque_elapsed_s += (
                (self._freeze_steps <= 0).to(dtype=self._ideal_torque_elapsed_s.dtype)
                * float(self.physics_dt)
            )
        elif wing_drive_variant == SINUSOIDAL_PHASE_SPEED_WING_DRIVE:
            assert sinusoidal_phase_step is not None
            assert self._phase_acceleration_rad_s2 is not None
            self._sinusoidal_phase_drive_state = sinusoidal_phase_step.next_state
            self._phase = sinusoidal_phase_step.next_state.phase_rad
            self._freq = (
                sinusoidal_phase_step.next_state.phase_rate_rad_s
                / (2.0 * math.pi)
            )
            self._phase_acceleration_rad_s2 = (
                sinusoidal_phase_step.phase_acceleration_rad_s2
            )
        elif wing_drive_variant == NATIVE_HOLONOMIC_WING_DRIVE:
            assert ideal_inverse_phase_step is not None
            assert self._ideal_inverse_phase_state is not None
            self._ideal_inverse_phase_state = ideal_inverse_phase_step.next_state
            self._phase = ideal_inverse_phase_step.next_state.phase_rad
            self._freq = ideal_inverse_phase_step.next_state.frequency_hz
            self._phase_acceleration_rad_s2 = ideal_inverse_phase_step.phase_acceleration_rad_s2
        elif wing_drive_variant == IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE:
            assert ideal_inverse_phase_step is not None
            assert ideal_inverse_desired_acceleration is not None
            assert self._ideal_inverse_phase_cfg is not None
            assert self._ideal_inverse_phase_state is not None
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            generalized_mass = self._robot.root_physx_view.get_generalized_mass_matrices().to(
                device=self.device
            )
            coriolis_bias = (
                self._robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces().to(
                    device=self.device
                )
            )
            gravity_bias = self._robot.root_physx_view.get_gravity_compensation_forces().to(
                device=self.device
            )
            generalized_bias = coriolis_bias + gravity_bias
            joint_count = len(self._robot.joint_names)
            base_dof_count = generalized_mass.shape[-1] - joint_count
            external_generalized = torch.zeros_like(generalized_bias)
            if base_dof_count == 6:
                external_generalized[:, :3] = quat_apply(
                    quat_w,
                    f_w_sum + f_tail + f_drag,
                )
                external_generalized[:, 3:6] = quat_apply(
                    quat_w,
                    tau_w_sum + tau_tail,
                )
            elif base_dof_count != 0:
                raise RuntimeError(
                    "PhysX generalized coordinates must contain either zero or six base DOFs."
                )
            if delaurier_wrench is not None:
                wing_hinge_torque = compute_aerodynamic_joint_hinge_torques(
                    force_link_n=delaurier_wrench.force_link_n,
                    moment_link_about_com_nm=delaurier_wrench.moment_link_about_com_nm,
                    wing_com_position_link_m=self._robot.data.body_com_pos_b[:, self._wing_body_ids, :],
                )
                external_generalized[:, base_dof_count + left_joint_id] = wing_hinge_torque[:, 0]
                external_generalized[:, base_dof_count + right_joint_id] = wing_hinge_torque[:, 1]
            joint_direction = torch.zeros(
                joint_count,
                device=generalized_mass.device,
                dtype=generalized_mass.dtype,
            )
            joint_direction[left_joint_id] = 1.0
            joint_direction[right_joint_id] = -1.0
            inverse_dynamics = reduce_common_inverse_dynamics(
                generalized_mass_matrix=generalized_mass,
                generalized_bias_effort=generalized_bias,
                external_generalized_effort=external_generalized,
                joint_direction=joint_direction,
                desired_common_acceleration_rad_s2=ideal_inverse_desired_acceleration.to(
                    dtype=generalized_mass.dtype
                ),
                effort_limit_nm=self._ideal_inverse_phase_cfg.effort_limit_nm,
            )
            self._robot.set_joint_effort_target(
                inverse_dynamics.effort_nm.unsqueeze(-1),
                joint_ids=[left_joint_id],
            )
            self._robot.set_joint_effort_target(
                torch.zeros_like(inverse_dynamics.effort_nm).unsqueeze(-1),
                joint_ids=[right_joint_id],
            )
            common_velocity = self._robot.data.joint_vel[:, left_joint_id]
            self.extras.setdefault("log", {}).update(
                {
                    "WingDrive/mean_abs_driver_torque_Nm": float(
                        torch.mean(torch.abs(inverse_dynamics.effort_nm)).item()
                    ),
                    "WingDrive/mean_abs_inverse_inertia_torque_Nm": float(
                        torch.mean(
                            torch.abs(
                                inverse_dynamics.common_inertia_kg_m2
                                * inverse_dynamics.desired_acceleration_rad_s2
                            )
                        ).item()
                    ),
                    "WingDrive/mean_abs_inverse_bias_torque_Nm": float(
                        torch.mean(torch.abs(inverse_dynamics.common_bias_effort_nm)).item()
                    ),
                    "WingDrive/mean_abs_aero_feedforward_torque_Nm": float(
                        torch.mean(torch.abs(inverse_dynamics.common_external_effort_nm)).item()
                    ),
                    "WingDrive/inverse_effort_saturation_fraction": float(
                        inverse_dynamics.saturated.to(dtype=torch.float32).mean().item()
                    ),
                    "WingDrive/mean_abs_mechanism_power_W": float(
                        torch.mean(torch.abs(inverse_dynamics.effort_nm * common_velocity)).item()
                    ),
                }
            )
            self._ideal_inverse_phase_state = ideal_inverse_phase_step.next_state
            self._phase = ideal_inverse_phase_step.next_state.phase_rad
            self._freq = ideal_inverse_phase_step.next_state.frequency_hz
            self._phase_acceleration_rad_s2 = ideal_inverse_phase_step.phase_acceleration_rad_s2

    def _update_native_holonomic_load_diagnostics(
        self,
        *,
        root_quaternion_w: Tensor,
        net_external_force_b_n: Tensor,
        net_external_moment_b_about_base_com_nm: Tensor,
        delaurier_wrench: _DeLaurierWingWrenchResult | None,
    ) -> None:
        """Cache an ideal common-coordinate multibody inverse-dynamics estimate.

        Root force and moment inputs are body-FLU quantities about the base COM.
        The cached torque is in N m and positive in the left-upstroke common
        coordinate. It estimates ideal wing-mechanism output load and is not a
        PhysX solver multiplier or a motor-shaft quantity.
        """

        assert self._qdd_cmd is not None
        generalized_mass = self._robot.root_physx_view.get_generalized_mass_matrices().to(
            device=self.device
        )
        coriolis_bias = (
            self._robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces().to(
                device=self.device
            )
        )
        gravity_bias = self._robot.root_physx_view.get_gravity_compensation_forces().to(
            device=self.device
        )
        generalized_bias = coriolis_bias + gravity_bias
        joint_count = len(self._robot.joint_names)
        base_dof_count = generalized_mass.shape[-1] - joint_count
        external_generalized = torch.zeros_like(generalized_bias)
        if base_dof_count == 6:
            external_generalized[:, :3] = quat_apply(
                root_quaternion_w,
                net_external_force_b_n,
            )
            external_generalized[:, 3:6] = quat_apply(
                root_quaternion_w,
                net_external_moment_b_about_base_com_nm,
            )
        elif base_dof_count != 0:
            raise RuntimeError(
                "PhysX generalized coordinates must contain either zero or six base DOFs."
            )

        left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
        right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
        if delaurier_wrench is not None:
            wing_hinge_torque = compute_aerodynamic_joint_hinge_torques(
                force_link_n=delaurier_wrench.force_link_n,
                moment_link_about_com_nm=delaurier_wrench.moment_link_about_com_nm,
                wing_com_position_link_m=self._robot.data.body_com_pos_b[:, self._wing_body_ids, :],
            )
            external_generalized[:, base_dof_count + left_joint_id] = wing_hinge_torque[:, 0]
            external_generalized[:, base_dof_count + right_joint_id] = wing_hinge_torque[:, 1]

        joint_direction = torch.zeros(
            joint_count,
            device=generalized_mass.device,
            dtype=generalized_mass.dtype,
        )
        joint_direction[left_joint_id] = 1.0
        joint_direction[right_joint_id] = -1.0
        actual_common_velocity = 0.5 * (
            self._robot.data.joint_vel[:, left_joint_id]
            - self._robot.data.joint_vel[:, right_joint_id]
        )
        estimate = estimate_common_constraint_load(
            generalized_mass_matrix=generalized_mass,
            generalized_bias_effort=generalized_bias,
            external_generalized_effort=external_generalized,
            joint_direction=joint_direction,
            prescribed_common_acceleration_rad_s2=self._qdd_cmd.to(
                dtype=generalized_mass.dtype
            ),
            actual_common_velocity_rad_s=actual_common_velocity.to(
                dtype=generalized_mass.dtype
            ),
        )
        self._debug_last_native_common_inertia_kg_m2.copy_(estimate.common_inertia_kg_m2)
        self._debug_last_native_inertia_torque_nm.copy_(estimate.inertia_torque_nm)
        self._debug_last_native_bias_torque_nm.copy_(estimate.bias_torque_nm)
        self._debug_last_native_external_load_torque_nm.copy_(estimate.external_load_torque_nm)
        self._debug_last_native_constraint_torque_estimate_nm.copy_(
            estimate.equivalent_constraint_torque_nm
        )
        self._debug_last_native_constraint_power_estimate_w.copy_(estimate.mechanical_power_w)
        self.extras.setdefault("log", {}).update(
            {
                "WingDrive/mean_abs_inverse_dynamics_constraint_torque_estimate_Nm": float(
                    torch.mean(torch.abs(estimate.equivalent_constraint_torque_nm)).item()
                ),
                "WingDrive/mean_abs_inverse_dynamics_constraint_power_estimate_W": float(
                    torch.mean(torch.abs(estimate.mechanical_power_w)).item()
                ),
            }
        )

    def _compute_wing_delaurier_wrench(
        self,
        v_air_b: Tensor,
        *,
        acceleration_source: str | None = None,
        store_diagnostics: bool = True,
    ) -> _DeLaurierWingWrenchResult:
        """Compute the net DeLaurier wing wrench about the base COM in body frame.

        ``legacy_fixed_quarter_chord`` preserves the previous aggregate-force
        closure. ``strip_integrated`` maps the force and axial moment of each
        wing from the Wang frame, then translates the full wing wrench from the
        wing-root pitching-axis origin to the base COM in world frame.
        """
        assert self._wing_geom is not None
        assert self._wing_area is not None
        assert self._delaurier_params is not None
        assert self._A_w2l_batch is not None
        assert self._q_cmd is not None and self._qd_cmd is not None and self._qdd_cmd is not None

        # Batch with two wings per env: (env0_L, env0_R, env1_L, env1_R, ...)
        N_env = self.num_envs
        B = 2 * N_env
        N_strip = int(self._wing_geom.x_mid.numel())
        y = self._wing_geom.x_mid.view(1, N_strip).expand(B, N_strip)

        wing_aero_coupling_mode = validate_wing_aero_coupling_mode(self.cfg.wing_aero_coupling_mode)
        if wing_aero_coupling_mode in {
            ACTUAL_MOTION_BASE_EQUIVALENT,
            ACTUAL_PER_WING_LINK,
            IDEAL_TORQUE_PER_WING_LINK,
            SINUSOIDAL_PHASE_PER_WING_LINK,
            IDEAL_INVERSE_DYNAMICS_PER_WING_LINK,
            NATIVE_HOLONOMIC_PER_WING_LINK,
        }:
            left_joint_id = int(self._joint_ids[self._IDX_LEFT_WING])
            right_joint_id = int(self._joint_ids[self._IDX_RIGHT_WING])
            physical_kinematics = map_opposed_joint_states_to_physical_wing_kinematics(
                joint_position_rad=torch.stack(
                    (
                        self._robot.data.joint_pos[:, left_joint_id],
                        self._robot.data.joint_pos[:, right_joint_id],
                    ),
                    dim=1,
                ),
                joint_velocity_rad_s=torch.stack(
                    (
                        self._robot.data.joint_vel[:, left_joint_id],
                        self._robot.data.joint_vel[:, right_joint_id],
                    ),
                    dim=1,
                ),
                joint_acceleration_rad_s2=torch.stack(
                    (
                        self._robot.data.joint_acc[:, left_joint_id],
                        self._robot.data.joint_acc[:, right_joint_id],
                    ),
                    dim=1,
                ),
                left_joint_mid_rad=self._wing_mid_L,
                right_joint_mid_rad=self._wing_mid_R,
            )
            q = physical_kinematics.position_rad.reshape(B)
            qd = physical_kinematics.velocity_rad_s.reshape(B)
            commanded_q = self._q_cmd.unsqueeze(1)
            commanded_qd = self._qd_cmd.unsqueeze(1)
            commanded_qdd = self._qdd_cmd.unsqueeze(1).expand_as(
                physical_kinematics.acceleration_rad_s2
            )
            resolved_acceleration_source = validate_wing_aero_acceleration_source(
                self.cfg.wing_aero_acceleration_source
                if acceleration_source is None
                else acceleration_source
            )
            selected_qdd = resolve_wing_aero_acceleration(
                source=resolved_acceleration_source,
                actual_acceleration_rad_s2=physical_kinematics.acceleration_rad_s2,
                prescribed_acceleration_rad_s2=commanded_qdd,
            )
            qdd = selected_qdd.reshape(B)
            actual_qdd = physical_kinematics.acceleration_rad_s2
            if store_diagnostics:
                self.extras.setdefault("log", {}).update(
                    {
                        "WingAero/mean_abs_position_input_error_rad": float(
                            torch.mean(torch.abs(physical_kinematics.position_rad - commanded_q)).item()
                        ),
                        "WingAero/mean_abs_velocity_input_error_rad_s": float(
                            torch.mean(torch.abs(physical_kinematics.velocity_rad_s - commanded_qd)).item()
                        ),
                        "WingAero/mean_abs_acceleration_input_error_rad_s2": float(
                            torch.mean(torch.abs(selected_qdd - commanded_qdd)).item()
                        ),
                        "WingAero/mean_abs_physx_acceleration_error_rad_s2": float(
                            torch.mean(torch.abs(actual_qdd - commanded_qdd)).item()
                        ),
                    }
                )
        else:
            q = torch.repeat_interleave(self._q_cmd, 2)  # (B,)
            qd = torch.repeat_interleave(self._qd_cmd, 2)
            qdd = torch.repeat_interleave(self._qdd_cmd, 2)
            actual_qdd = qdd.reshape(N_env, 2)
        if store_diagnostics:
            self._debug_last_wing_aero_position_rad.copy_(q.reshape(N_env, 2))
            self._debug_last_wing_aero_velocity_rad_s.copy_(qd.reshape(N_env, 2))
            self._debug_last_wing_aero_acceleration_rad_s2.copy_(qdd.reshape(N_env, 2))
            self._debug_last_wing_actual_acceleration_rad_s2.copy_(actual_qdd)
        w = torch.repeat_interleave(2.0 * torch.pi * self._freq, 2)  # (B,)

        # theta_a (flapping-axis angle relative to the freestream) per env -> per wing.
        # In free-flight we approximate this as the body-frame angle-of-attack based on the velocity vector.
        # This matches the wind-tunnel convention when the vehicle is trimmed (v_z small, pitch≈flight-path angle),
        # and avoids large sign errors during dives/climbs where pitch!=AOA.
        quat_w = self._robot.data.root_quat_w  # (N,4)
        # Isaac/project body data are FLU. Convert the vehicle air-relative
        # velocity explicitly to DeLaurier's internal FRD-like section frame
        # before applying theta_a=atan2(w_D,u_D).
        v_air_delaurier = body_air_velocity_to_delaurier_section_velocity(v_air_b, body_frame="FLU")
        theta_a_env = compute_delaurier_axis_incidence(
            air_velocity_body=v_air_b,
            body_frame="FLU",
            minimum_forward_speed_mps=1.0e-3,
        )
        theta_a = torch.repeat_interleave(theta_a_env, 2)  # (B,)
        theta_bar = theta_a + math.radians(float(self.cfg.delaurier_theta_w_deg))

        # Airspeed approximation: body-forward velocity component clamped (matches wind-tunnel trim scans).
        vx = torch.clamp(v_air_delaurier[:, 0], min=float(self.cfg.delaurier_min_airspeed))
        U = torch.repeat_interleave(vx, 2)  # (B,)

        h = -q.view(B, 1) * y
        hdot = -qd.view(B, 1) * y
        hddot = -qdd.view(B, 1) * y

        dynamic_twist_mode = validate_delaurier_dynamic_twist_mode(self.cfg.dynamic_twist_mode)
        current_phase = torch.repeat_interleave(self._phase, 2)  # (B,), rad
        # The environment uses the instantaneous commanded frequency within a
        # physics step and currently assumes zero phase acceleration.  The pure
        # helper accepts a non-zero value for future variable-frequency inputs.
        current_phase_acceleration = torch.zeros_like(w)
        phase_delaurier, phase_rate_delaurier, phase_acceleration_delaurier = resolve_delaurier_phase(
            current_phase=current_phase,
            current_phase_rate=w,
            current_phase_acceleration=current_phase_acceleration,
            phase_direction=float(self.cfg.dynamic_twist_phase_direction),
            phase_offset_rad=math.radians(float(self.cfg.dynamic_twist_phase_offset_deg)),
        )

        if dynamic_twist_mode in {"disabled", "delaurier_linear_spanwise"}:
            twist_kinematics = compute_delaurier_dynamic_twist(
                strip_span_m=self._wing_geom.x_mid,
                strip_width_m=self._wing_geom.dx,
                semi_span_m=float(self._wing_geom.R),
                mean_pitch_rad=theta_bar,
                tip_twist_amplitude_rad=math.radians(float(self.cfg.dynamic_twist_tip_amplitude_deg)),
                phase_rad=phase_delaurier,
                phase_rate_rad_s=phase_rate_delaurier,
                phase_acceleration_rad_s2=phase_acceleration_delaurier,
                enabled=dynamic_twist_mode == "delaurier_linear_spanwise",
            )
        else:
            # Legacy compatibility only: a clamped, full-span-uniform twist
            # proportional to qd.  This is not DeLaurier's numerical-example
            # linear-spanwise dynamic twist.
            eta_max = torch.deg2rad(
                torch.tensor(float(self.cfg.twist_eta_max_deg), device=self.device, dtype=qd.dtype)
            )
            eta_lim = torch.deg2rad(
                torch.tensor(float(self.cfg.twist_eta_limit_deg), device=self.device, dtype=qd.dtype)
            )
            f_ref = float(self.cfg.twist_f_ref_hz)
            qd_ref = float(self._wing_amp) * (2.0 * math.pi * f_ref)
            left_twist_sign = float(self.cfg.twist_sign_left)
            right_twist_sign = float(self.cfg.twist_sign_right)
            wing_twist_sign = torch.tensor(
                [left_twist_sign, right_twist_sign], device=self.device, dtype=qd.dtype
            ).repeat(N_env)
            qddd = -(w * w) * qd
            legacy_twist = compute_legacy_qd_scaled_twist(
                num_strips=N_strip,
                mean_pitch_rad=theta_bar,
                joint_velocity_rad_s=qd,
                joint_acceleration_rad_s2=qdd,
                joint_jerk_rad_s3=qddd,
                reference_velocity_rad_s=qd_ref,
                maximum_twist_rad=eta_max,
                twist_limit_rad=eta_lim,
                twist_sign=wing_twist_sign,
            )
            twist_kinematics = DeLaurierTwistKinematics(
                theta=legacy_twist.theta,
                theta_dot=legacy_twist.theta_dot,
                theta_ddot=legacy_twist.theta_ddot,
                delta_theta=legacy_twist.delta_theta,
                delta_theta_dot=legacy_twist.delta_theta_dot,
                delta_theta_ddot=legacy_twist.delta_theta_ddot,
                span_fraction=y / float(self._wing_geom.R),
                phase=phase_delaurier.view(B, 1),
                phase_rate=phase_rate_delaurier.view(B, 1),
                phase_acceleration=phase_acceleration_delaurier.view(B, 1),
            )

        theta = twist_kinematics.theta
        thetad = twist_kinematics.theta_dot
        thetadd = twist_kinematics.theta_ddot
        if store_diagnostics:
            self._debug_last_delaurier_twist_kinematics = (
                twist_kinematics if bool(self.cfg.delaurier_store_strip_diagnostics) else None
            )

        omega_ref = w.view(B, 1).expand(B, N_strip)

        moment_mode = str(self.cfg.wing_moment_mode)
        if moment_mode not in {"legacy_fixed_quarter_chord", "strip_integrated"}:
            raise ValueError(
                "cfg.wing_moment_mode must be 'legacy_fixed_quarter_chord' or 'strip_integrated', "
                f"got {moment_mode!r}."
            )
        if moment_mode == "strip_integrated" and bool(self.cfg.delaurier_enable_separation):
            raise ValueError("strip_integrated DeLaurier moment requires delaurier_enable_separation=False.")

        # Wang->link for each wing (repeat per environment). This polar-vector
        # matrix has det=-1 for the mirrored right wing; the physics helper
        # applies the required axial-vector parity for moments.
        A_w2l = self._A_w2l_batch.repeat(N_env, 1, 1)  # (B,3,3)
        q_w_link = self._robot.data.body_quat_w[:, self._wing_body_ids, :].reshape(B, 4)
        p_wing_origin_w = self._robot.data.body_pos_w[:, self._wing_body_ids, :].reshape(B, 3)
        p_base_w = self._robot.data.root_com_pos_w

        if moment_mode == "legacy_fixed_quarter_chord":
            assert self._wing_application_point_link is not None
            force_wang, _legacy_zero_moment_wang, _power_in, _sep_ratio = compute_aero_wrench_delaurier1993(
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
            )
            force_link = torch.bmm(A_w2l, force_wang.unsqueeze(-1)).squeeze(-1)
            wing_application_point_link = self._wing_application_point_link.repeat(N_env, 1)
            moment_link_about_wing_origin = torch.linalg.cross(
                wing_application_point_link,
                force_link,
            )
            if store_diagnostics:
                self._debug_last_delaurier_strip_loads = None
                self._debug_last_delaurier_strip_wrench = None
        else:
            strip_loads = compute_delaurier_strip_loads(
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
                enable_separation=False,
            )
            strip_wrench = integrate_delaurier_strip_wrench(
                strip_loads,
                include_aerodynamic_center_moment=bool(self.cfg.delaurier_include_aerodynamic_center_moment),
                include_apparent_mass_moment=bool(self.cfg.delaurier_include_apparent_mass_moment),
            )
            force_link, moment_link_about_wing_origin = transform_wang_wrench_to_link(
                strip_wrench.force_wang,
                strip_wrench.moment_wang_about_wing_origin,
                A_w2l,
            )
            if store_diagnostics:
                if bool(self.cfg.delaurier_store_strip_diagnostics):
                    self._debug_last_delaurier_strip_loads = strip_loads
                    self._debug_last_delaurier_strip_wrench = strip_wrench
                else:
                    self._debug_last_delaurier_strip_loads = None
                    self._debug_last_delaurier_strip_wrench = None

        force_world = quat_apply(q_w_link, force_link)
        moment_world_about_wing_origin = quat_apply(q_w_link, moment_link_about_wing_origin)
        moment_world_about_base_com = translate_wrench_moment(
            force_world,
            moment_world_about_wing_origin,
            p_wing_origin_w,
            torch.repeat_interleave(p_base_w, 2, dim=0),
        )
        wing_com_position_link = self._robot.data.body_com_pos_b[:, self._wing_body_ids, :]
        moment_link_about_com = translate_wing_root_wrench_to_com_link(
            force_link_n=force_link.view(N_env, 2, 3),
            moment_link_about_wing_origin_nm=moment_link_about_wing_origin.view(N_env, 2, 3),
            wing_com_position_link_m=wing_com_position_link,
        )
        force_world_sum = force_world.view(N_env, 2, 3).sum(dim=1)
        moment_world_sum_about_base_com = moment_world_about_base_com.view(N_env, 2, 3).sum(dim=1)

        # World->body
        F_b = quat_apply_inverse(quat_w, force_world_sum)
        tau_b = quat_apply_inverse(quat_w, moment_world_sum_about_base_com)

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

        return _DeLaurierWingWrenchResult(
            net_force_b_n=F_b,
            net_moment_b_about_base_com_nm=tau_b,
            force_link_n=force_link.view(N_env, 2, 3),
            moment_link_about_com_nm=moment_link_about_com,
        )

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

        # Randomize the world-frame line direction while keeping the vehicle,
        # initial velocity and route geometry mutually aligned.
        n = env_ids.shape[0]
        if self._pure_rl_longitudinal_stage is not None:
            assert self._pure_rl_longitudinal_path is not None
            if self.cfg.pure_rl_eval_longitudinal_task_schedule is not None:
                assert self.cfg.pure_rl_eval_heading_schedule_rad is not None
                assert self.cfg.pure_rl_eval_longitudinal_slope_deg_schedule is not None
                assert self.cfg.pure_rl_eval_entry_length_m_schedule is not None
                assert self.cfg.pure_rl_eval_slope_length_m_schedule is not None
                schedule_length = len(self.cfg.pure_rl_eval_heading_schedule_rad)
                schedule_indices = env_ids.to(dtype=torch.long) % schedule_length
                sampled_path = PureRLLongitudinalPathBatch(
                    task_id=torch.as_tensor(
                        self.cfg.pure_rl_eval_longitudinal_task_schedule,
                        dtype=torch.int64,
                        device=self.device,
                    )[schedule_indices],
                    heading_rad=torch.as_tensor(
                        self.cfg.pure_rl_eval_heading_schedule_rad,
                        dtype=self._height_cmd.dtype,
                        device=self.device,
                    )[schedule_indices],
                    signed_slope_rad=torch.deg2rad(
                        torch.as_tensor(
                            self.cfg.pure_rl_eval_longitudinal_slope_deg_schedule,
                            dtype=self._height_cmd.dtype,
                            device=self.device,
                        )[schedule_indices]
                    ),
                    entry_length_m=torch.as_tensor(
                        self.cfg.pure_rl_eval_entry_length_m_schedule,
                        dtype=self._height_cmd.dtype,
                        device=self.device,
                    )[schedule_indices],
                    slope_length_m=torch.as_tensor(
                        self.cfg.pure_rl_eval_slope_length_m_schedule,
                        dtype=self._height_cmd.dtype,
                        device=self.device,
                    )[schedule_indices],
                    initial_altitude_m=self._height_cmd[env_ids].clone(),
                )
            else:
                sampled_path = sample_longitudinal_path_batch(
                    num_paths=n,
                    stage=self._pure_rl_longitudinal_stage,
                    device=self.device,
                    dtype=self._height_cmd.dtype,
                    initial_altitude_m=self._height_cmd[env_ids],
                )
            write_longitudinal_path_batch_rows_(
                destination=self._pure_rl_longitudinal_path,
                env_ids=env_ids.to(dtype=torch.int64),
                source=sampled_path,
            )
            heading = sampled_path.heading_rad
        elif self.cfg.pure_rl_eval_heading_schedule_rad is not None:
            schedule = torch.as_tensor(
                self.cfg.pure_rl_eval_heading_schedule_rad,
                dtype=torch.float32,
                device=self.device,
            )
            heading = schedule[env_ids.to(dtype=torch.long) % schedule.numel()]
        elif bool(self.cfg.randomize_straight_line_heading):
            heading = (2.0 * torch.rand((n,), device=self.device) - 1.0) * math.pi
        else:
            heading = torch.zeros((n,), device=self.device)
        self._straight_line_heading_rad[env_ids] = heading
        self._straight_line_tangent_w[env_ids, 0] = torch.cos(heading)
        self._straight_line_tangent_w[env_ids, 1] = torch.sin(heading)
        self._straight_line_tangent_w[env_ids, 2] = 0.0
        self._straight_line_normal_w[env_ids, 0] = -torch.sin(heading)
        self._straight_line_normal_w[env_ids, 1] = torch.cos(heading)
        self._straight_line_normal_w[env_ids, 2] = 0.0

        # root state: spawn at local (0,0,height_cmd) relative to env origin, with initial vx ~ vx_cmd
        env_origins = self.scene.env_origins[env_ids]
        pos = env_origins.clone()
        pos[:, 2] += self._height_cmd[env_ids]
        # Isaac uses a right-handed convention: positive rotation about +Y pitches the nose *down*.
        # Users typically specify +pitch as "nose up", so we negate it here to match the wind-tunnel scripts.
        pitch = -torch.deg2rad(torch.full((n,), float(self.cfg.reset_pitch_deg), device=self.device))
        rot = quat_from_euler_xyz(
            roll=torch.zeros_like(pitch),
            pitch=pitch,
            yaw=heading,
        )
        lin_vel = torch.zeros(n, 3, device=self.device)
        reset_forward_speed = self._vx_cmd[env_ids]
        if self.cfg.reset_forward_speed_mps is not None:
            reset_forward_speed = torch.full(
                (n,),
                float(self.cfg.reset_forward_speed_mps),
                device=self.device,
            )
        lin_vel[:, 0:2] = reset_forward_speed.unsqueeze(1) * self._straight_line_tangent_w[env_ids, 0:2]
        ang_vel = torch.zeros(n, 3, device=self.device)
        root_state = torch.cat([pos, rot, lin_vel, ang_vel], dim=1)
        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._spawn_root_state[env_ids] = root_state

        # joints to default and zero velocity
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) == PRESCRIBED_COUPLED_WING_DRIVE:
            assert self._nominal_dof_position_limits_cpu is not None
            env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.int64)
            reset_limits_cpu = build_prescribed_dof_position_limits(
                nominal_limits_rad=self._nominal_dof_position_limits_cpu[env_ids_cpu],
                target_position_rad=torch.full(
                    (n,),
                    self._wing_mid_L,
                    device=self._nominal_dof_position_limits_cpu.device,
                    dtype=self._nominal_dof_position_limits_cpu.dtype,
                ),
                joint_id=int(self._joint_ids[self._IDX_LEFT_WING]),
                half_width_rad=float(self.cfg.prescribed_joint_limit_half_width_rad),
            )
            reset_limits_cpu = build_prescribed_dof_position_limits(
                nominal_limits_rad=reset_limits_cpu,
                target_position_rad=torch.full(
                    (n,),
                    self._wing_mid_R,
                    device=reset_limits_cpu.device,
                    dtype=reset_limits_cpu.dtype,
                ),
                joint_id=int(self._joint_ids[self._IDX_RIGHT_WING]),
                half_width_rad=float(self.cfg.prescribed_joint_limit_half_width_rad),
            )
            self._robot.root_physx_view.set_dof_limits(
                reset_limits_cpu,
                indices=env_ids_cpu.to(dtype=torch.int32),
            )
        f0 = float(self.cfg.reset_flap_hz)
        f0 = max(float(self.cfg.min_flap_hz), min(float(self.cfg.max_flap_hz), f0))
        if self.cfg.pure_rl_eval_flap_phase_schedule_rad is not None:
            schedule = torch.as_tensor(
                self.cfg.pure_rl_eval_flap_phase_schedule_rad,
                dtype=torch.float32,
                device=self.device,
            )
            phase0 = torch.remainder(
                schedule[env_ids.to(dtype=torch.long) % schedule.numel()],
                2.0 * math.pi,
            )
        elif bool(self.cfg.randomize_flap_phase_at_reset):
            phase0 = 2.0 * math.pi * torch.rand((n,), device=self.device)
        else:
            phase0 = torch.zeros((n,), device=self.device)
        jpos = self._default_joint_pos.expand(n, -1).clone()
        # initialize tail joints from mixed reset elevon commands
        action_interface = validate_action_interface(self.cfg.action_interface)
        if action_interface == DIRECT_TAIL_SURFACE_ACTION:
            rudder_lower = self._joint_lower_limits[self._IDX_RUDDER]
            rudder_upper = self._joint_upper_limits[self._IDX_RUDDER]
            l_lower = self._joint_lower_limits[self._IDX_LEFT_TAIL]
            l_upper = self._joint_upper_limits[self._IDX_LEFT_TAIL]
            r_lower = self._joint_lower_limits[self._IDX_RIGHT_TAIL]
            r_upper = self._joint_upper_limits[self._IDX_RIGHT_TAIL]
        else:
            elevon_lim = torch.deg2rad(
                torch.tensor(float(self.cfg.elevon_max_deg), device=self.device)
            )
            rudder_lim = torch.deg2rad(
                torch.tensor(float(self.cfg.rudder_max_deg), device=self.device)
            )
            rudder_lower = torch.maximum(self._joint_lower_limits[self._IDX_RUDDER], -rudder_lim)
            rudder_upper = torch.minimum(self._joint_upper_limits[self._IDX_RUDDER], rudder_lim)
            l_lower = torch.maximum(self._joint_lower_limits[self._IDX_LEFT_TAIL], -elevon_lim)
            l_upper = torch.minimum(self._joint_upper_limits[self._IDX_LEFT_TAIL], elevon_lim)
            r_lower = torch.maximum(self._joint_lower_limits[self._IDX_RIGHT_TAIL], -elevon_lim)
            r_upper = torch.minimum(self._joint_upper_limits[self._IDX_RIGHT_TAIL], elevon_lim)
        rudder0 = torch.full((n,), math.radians(float(self.cfg.reset_rudder_deg)), device=self.device).clamp(
            rudder_lower, rudder_upper
        )
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
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) in {
            SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
            IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
            NATIVE_HOLONOMIC_WING_DRIVE,
        }:
            initial_common_position = self._wing_amp * torch.sin(phase0)
            jpos[:, self._IDX_LEFT_WING] = self._wing_mid_L + initial_common_position
            jpos[:, self._IDX_RIGHT_WING] = self._wing_mid_R - initial_common_position
            initial_common_velocity = self._wing_amp * 2.0 * math.pi * f0 * torch.cos(phase0)
            jvel[:, self._IDX_LEFT_WING] = initial_common_velocity
            jvel[:, self._IDX_RIGHT_WING] = -initial_common_velocity
        self._robot.write_joint_state_to_sim(jpos, jvel, joint_ids=self._joint_ids, env_ids=env_ids)

        # clear actions and phases
        self._actions[env_ids] = 0.0
        self._act_lpf[env_ids] = 0.0
        self._act_cmd[env_ids] = 0.0
        self._phase[env_ids] = phase0
        self._ideal_torque_elapsed_s[env_ids] = 0.0
        # start near trim frequency to avoid immediate drop before the policy stabilizes
        self._freq[env_ids] = f0
        if self._phase_throttle is not None:
            self._phase_throttle[env_ids] = f0 / float(self.cfg.max_flap_hz)
        if self._phase_target_frequency_hz is not None:
            self._phase_target_frequency_hz[env_ids] = f0
        if self._requested_frequency_hz is not None:
            self._requested_frequency_hz[env_ids] = f0
        if self._applied_frequency_hz is not None:
            self._applied_frequency_hz[env_ids] = f0
        if self._frequency_slew_hz_per_s is not None:
            self._frequency_slew_hz_per_s[env_ids] = 0.0
        if self._frequency_governor_limited is not None:
            self._frequency_governor_limited[env_ids] = False
        if bool(self.cfg.frequency_governor_enabled):
            reset_frequency_action = frequency_hz_to_normalized_action(
                torch.full((n,), f0, device=self.device),
                minimum_frequency_hz=float(self.cfg.min_flap_hz),
                maximum_frequency_hz=float(self.cfg.max_flap_hz),
            )
            self._act_lpf[env_ids, 0] = reset_frequency_action
            self._act_cmd[env_ids, 0] = reset_frequency_action
        if self._phase_acceleration_rad_s2 is not None:
            self._phase_acceleration_rad_s2[env_ids] = 0.0
        if self._sinusoidal_phase_drive_state is not None:
            self._sinusoidal_phase_drive_state.phase_rad[env_ids] = phase0
            self._sinusoidal_phase_drive_state.phase_rate_rad_s[env_ids] = (
                2.0 * math.pi * f0
            )
            self._sinusoidal_phase_drive_state.speed_error_integral_rad[env_ids] = 0.0
        if self._ideal_inverse_phase_state is not None:
            self._ideal_inverse_phase_state.phase_rad[env_ids] = phase0
            self._ideal_inverse_phase_state.frequency_hz[env_ids] = f0
        if validate_wing_drive_variant(self.cfg.wing_drive_variant) == NATIVE_HOLONOMIC_WING_DRIVE:
            assert self._native_holonomic is not None
            assert self._ideal_inverse_phase_state is not None
            phase = self._ideal_inverse_phase_state.phase_rad
            frequency = self._ideal_inverse_phase_state.frequency_hz
            target_position = float(self._wing_mid_L) + float(self._wing_amp) * torch.sin(phase)
            target_velocity = (
                float(self._wing_amp) * 2.0 * math.pi * frequency * torch.cos(phase)
            )
            self._native_holonomic.set_targets(
                self._native_holonomic_joint_paths,
                target_position.detach().to(device="cpu", dtype=torch.float64).tolist(),
                target_velocity.detach().to(device="cpu", dtype=torch.float64).tolist(),
            )
        # initialize action history to match the reset frequency so action filtering doesn't create a large transient
        a0 = frequency_hz_to_normalized_action(
            torch.full((n,), f0, device=self.device),
            minimum_frequency_hz=float(self.cfg.min_flap_hz),
            maximum_frequency_hz=float(self.cfg.max_flap_hz),
        )
        self._actions[env_ids, 0] = a0
        self._act_lpf[env_ids, 0] = a0
        self._act_cmd[env_ids, 0] = a0
        # initialize action history for tail channels to avoid large transients
        if action_interface == DIRECT_TAIL_SURFACE_ACTION:
            tail_reset_action = torch.stack(
                (
                    joint_position_to_normalized_action(
                        rudder0,
                        lower_limit_rad=rudder_lower,
                        upper_limit_rad=rudder_upper,
                    ),
                    joint_position_to_normalized_action(
                        left0,
                        lower_limit_rad=l_lower,
                        upper_limit_rad=l_upper,
                    ),
                    joint_position_to_normalized_action(
                        right0,
                        lower_limit_rad=r_lower,
                        upper_limit_rad=r_upper,
                    ),
                ),
                dim=1,
            )
            self._actions[env_ids, 1:4] = tail_reset_action
            self._act_lpf[env_ids, 1:4] = tail_reset_action
            self._act_cmd[env_ids, 1:4] = tail_reset_action
        else:
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
        self._rudder_cmd[env_ids] = rudder0
        self._roll_cmd[env_ids] = 0.5 * (left0 - right0)
        if self._teacher_controller is not None:
            self._teacher_controller.reset(env_ids)
        self._reset_runtime_state_estimation(env_ids)
        self._hist_valid[env_ids] = False
        if self._pure_rl_history_valid is not None:
            self._pure_rl_history_valid[env_ids] = False
        if self._pure_rl_previous_reward_action is not None:
            self._pure_rl_previous_reward_action[env_ids] = self._act_cmd[env_ids]
        self._freeze_steps[env_ids] = int(self.cfg.freeze_steps_after_reset)

    def _query_pure_rl_longitudinal_path(self) -> PureRLLongitudinalPathQuery:
        """Query the active C2 path for every environment."""

        assert self._pure_rl_longitudinal_path is not None
        return query_longitudinal_path(
            path=self._pure_rl_longitudinal_path,
            position_world_m=self._robot.data.root_pos_w - self.scene.env_origins,
            ground_velocity_world_mps=self._robot.data.root_lin_vel_w,
            minimum_preview_speed_mps=float(self.cfg.pure_rl_preview_minimum_speed_mps),
            maximum_preview_speed_mps=float(self.cfg.pure_rl_preview_maximum_speed_mps),
        )

    def _get_pure_rl_observations(self) -> dict[str, Tensor]:
        """Build the normalized 555-value actor observation at policy rate."""

        assert self._pure_rl_history_valid is not None
        assert self._pure_rl_previous_orientation_wxyz is not None
        assert self._pure_rl_sensor_history is not None
        assert self._pure_rl_action_history is not None
        orientation_wxyz = self._robot.data.root_quat_w
        ground_velocity_b = self._robot.data.root_lin_vel_b
        angular_velocity_b = self._robot.data.root_ang_vel_b
        wind_b = quat_apply_inverse(orientation_wxyz, self._wind_w)
        forward_air_velocity_b = ground_velocity_b[:, 0] - wind_b[:, 0]

        previous_orientation = torch.where(
            self._pure_rl_history_valid.unsqueeze(1),
            self._pure_rl_previous_orientation_wxyz,
            orientation_wxyz,
        )
        sensor_frame = build_raw_sensor_frame(
            orientation_world_wxyz=orientation_wxyz,
            ground_velocity_body_mps=ground_velocity_b,
            angular_velocity_body_rad_s=angular_velocity_b,
            forward_air_velocity_body_mps=forward_air_velocity_b,
            actual_flap_frequency_hz=self._freq,
            flap_phase_rad=self._phase,
            previous_orientation_world_wxyz=previous_orientation,
        )

        local_position_w = self._robot.data.root_pos_w - self.scene.env_origins
        if self._pure_rl_longitudinal_stage is not None:
            preview_points_w = self._query_pure_rl_longitudinal_path().preview_points_world_m
        else:
            route_origin_w = torch.zeros_like(local_position_w)
            route_origin_w[:, 2] = self._height_cmd
            route_delta_w = local_position_w - route_origin_w
            closest_progress_m = torch.sum(route_delta_w * self._straight_line_tangent_w, dim=1)
            preview_progress_m, _preview_speed_mps = compute_preview_query_progress_m(
                closest_path_progress_m=closest_progress_m,
                ground_velocity_world_mps=self._robot.data.root_lin_vel_w,
                path_tangent_world=self._straight_line_tangent_w,
                minimum_preview_speed_mps=float(self.cfg.pure_rl_preview_minimum_speed_mps),
                maximum_preview_speed_mps=float(self.cfg.pure_rl_preview_maximum_speed_mps),
            )
            preview_points_w = (
                route_origin_w.unsqueeze(1)
                + preview_progress_m.unsqueeze(2) * self._straight_line_tangent_w.unsqueeze(1)
            )
        preview_points_b = transform_world_preview_points_to_body(
            preview_points_world_m=preview_points_w,
            vehicle_position_world_m=local_position_w,
            orientation_world_wxyz=orientation_wxyz,
        )

        invalid_ids = (~self._pure_rl_history_valid).nonzero(as_tuple=False).squeeze(-1)
        if invalid_ids.numel() > 0:
            self._pure_rl_sensor_history[invalid_ids] = sensor_frame[invalid_ids].unsqueeze(1).expand(
                -1,
                PURE_RL_RAW_OBSERVATION_LAYOUT.sensor_history_steps,
                -1,
            )
            self._pure_rl_action_history[invalid_ids] = self._act_cmd[invalid_ids].unsqueeze(1).expand(
                -1,
                PURE_RL_RAW_OBSERVATION_LAYOUT.action_history_steps,
                -1,
            )
        self._pure_rl_sensor_history.copy_(torch.roll(self._pure_rl_sensor_history, shifts=-1, dims=1))
        self._pure_rl_action_history.copy_(torch.roll(self._pure_rl_action_history, shifts=-1, dims=1))
        self._pure_rl_sensor_history[:, -1, :] = sensor_frame
        self._pure_rl_action_history[:, -1, :] = self._act_cmd
        self._pure_rl_previous_orientation_wxyz.copy_(sensor_frame[:, 0:4])
        self._pure_rl_history_valid[:] = True

        observation = normalize_actor_observation(
            sensor_history=self._pure_rl_sensor_history,
            action_history=self._pure_rl_action_history,
            preview_points_body_m=preview_points_b,
        )
        return {"policy": observation}

    def _get_observations(self) -> dict[str, Tensor]:
        if bool(self.cfg.use_pure_rl_actor_observation):
            return self._get_pure_rl_observations()

        self._refresh_runtime_estimated_state()
        pos_w, lin_vel_b, ang_vel_b, g_b = self._get_policy_observation_state()
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

    def _straight_line_cross_track_m(self, local_position_w: Tensor) -> Tensor:
        """Return signed distance from each environment's randomized line."""

        return torch.sum(local_position_w * self._straight_line_normal_w, dim=1)

    def _get_pure_rl_curriculum1_reward(self) -> Tensor:
        """Compute and expose the approved geometric straight-flight reward."""

        assert self._pure_rl_previous_reward_action is not None
        local_position_w = self._robot.data.root_pos_w - self.scene.env_origins
        ground_velocity_w = self._robot.data.root_lin_vel_w
        roll_rad, pitch_rad, _yaw_rad = euler_xyz_from_quat(self._robot.data.root_quat_w)
        reward_cfg = self.cfg.pure_rl_reward_cfg
        if self._pure_rl_longitudinal_stage is not None:
            query = self._query_pure_rl_longitudinal_path()
            height_error_m = query.height_error_m
            cross_track_error_m = query.cross_track_error_m
            along_track_progress_m = query.horizontal_progress_m
            along_track_velocity_mps = torch.sum(ground_velocity_w * query.tangent_world, dim=1)
            cross_track_velocity_mps = torch.sum(
                ground_velocity_w * query.lateral_normal_world,
                dim=1,
            )
            vertical_velocity_mps = torch.sum(
                ground_velocity_w * query.vertical_normal_world,
                dim=1,
            )
            terms = compute_pure_rl_path_reward_terms(
                cross_track_error_m=cross_track_error_m,
                height_error_m=height_error_m,
                tangent_velocity_mps=along_track_velocity_mps,
                lateral_normal_velocity_mps=cross_track_velocity_mps,
                vertical_normal_velocity_mps=vertical_velocity_mps,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                angular_velocity_body_rad_s=self._robot.data.root_ang_vel_b,
                actual_flap_frequency_hz=self._freq,
                frequency_slew_hz_per_s=self._frequency_slew_hz_per_s,
                applied_action=self._act_cmd,
                previous_applied_action=self._pure_rl_previous_reward_action,
                config=reward_cfg,
            )
        else:
            height_error_m = local_position_w[:, 2] - self._height_cmd
            cross_track_error_m = self._straight_line_cross_track_m(local_position_w)
            along_track_progress_m = torch.sum(
                local_position_w * self._straight_line_tangent_w,
                dim=1,
            )
            along_track_velocity_mps = torch.sum(
                ground_velocity_w * self._straight_line_tangent_w,
                dim=1,
            )
            cross_track_velocity_mps = torch.sum(
                ground_velocity_w * self._straight_line_normal_w,
                dim=1,
            )
            vertical_velocity_mps = ground_velocity_w[:, 2]
            terms = compute_pure_rl_reward_terms(
                cross_track_error_m=cross_track_error_m,
                height_error_m=height_error_m,
                along_track_velocity_mps=along_track_velocity_mps,
                cross_track_velocity_mps=cross_track_velocity_mps,
                vertical_velocity_mps=vertical_velocity_mps,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                angular_velocity_body_rad_s=self._robot.data.root_ang_vel_b,
                actual_flap_frequency_hz=self._freq,
                frequency_slew_hz_per_s=self._frequency_slew_hz_per_s,
                applied_action=self._act_cmd,
                previous_applied_action=self._pure_rl_previous_reward_action,
                config=reward_cfg,
            )
        assert self._eval_pure_rl_cross_track_error_m is not None
        assert self._eval_pure_rl_height_error_m is not None
        assert self._eval_pure_rl_along_track_progress_m is not None
        assert self._eval_pure_rl_along_track_velocity_mps is not None
        assert self._eval_pure_rl_angular_rate_rad_s is not None
        assert self._eval_pure_rl_actual_flap_frequency_hz is not None
        assert self._eval_pure_rl_frequency_limit_active is not None
        assert self._eval_pure_rl_tail_limit_active is not None
        assert self._eval_pure_rl_normalized_action_delta is not None
        assert self._eval_pure_rl_frequency_slew_hz_per_s is not None
        assert self._eval_pure_rl_frequency_governor_limited is not None
        self._eval_pure_rl_cross_track_error_m.copy_(cross_track_error_m)
        self._eval_pure_rl_height_error_m.copy_(height_error_m)
        self._eval_pure_rl_along_track_progress_m.copy_(along_track_progress_m)
        self._eval_pure_rl_along_track_velocity_mps.copy_(along_track_velocity_mps)
        if self._pure_rl_longitudinal_stage is not None:
            assert self._eval_pure_rl_lateral_normal_velocity_mps is not None
            assert self._eval_pure_rl_vertical_normal_velocity_mps is not None
            assert self._eval_pure_rl_active_slope_rad is not None
            assert self._eval_pure_rl_reached_recovery is not None
            self._eval_pure_rl_lateral_normal_velocity_mps.copy_(cross_track_velocity_mps)
            self._eval_pure_rl_vertical_normal_velocity_mps.copy_(vertical_velocity_mps)
            self._eval_pure_rl_active_slope_rad.copy_(query.active_slope_rad)
            self._eval_pure_rl_reached_recovery.copy_(query.reached_recovery)
        self._eval_pure_rl_angular_rate_rad_s.copy_(
            torch.linalg.vector_norm(self._robot.data.root_ang_vel_b, dim=1)
        )
        self._eval_pure_rl_actual_flap_frequency_hz.copy_(self._freq)
        self._eval_pure_rl_frequency_limit_active.copy_(
            self._freq >= 0.95 * float(reward_cfg.maximum_flap_frequency_hz)
        )
        self._eval_pure_rl_tail_limit_active.copy_(
            torch.any(
                torch.abs(self._act_cmd[:, 1:4]) >= float(reward_cfg.tail_action_limit_threshold),
                dim=1,
            )
        )
        self._eval_pure_rl_normalized_action_delta.copy_(
            torch.mean(0.5 * torch.abs(self._act_cmd - self._pure_rl_previous_reward_action), dim=1)
        )
        self._eval_pure_rl_frequency_slew_hz_per_s.copy_(self._frequency_slew_hz_per_s)
        self._eval_pure_rl_frequency_governor_limited.copy_(self._frequency_governor_limited)
        self._pure_rl_previous_reward_action.copy_(self._act_cmd)

        if bool(self.cfg.pure_rl_reward_telemetry_enabled):
            log = self.extras.setdefault("log", {})
            log.update(
                {
                    "PureRLReward/total": terms.total_reward.mean(),
                    "PureRLReward/path": terms.path_reward.mean(),
                    "PureRLReward/progress": terms.progress_reward.mean(),
                    "PureRLReward/velocity": terms.velocity_reward.mean(),
                    "PureRLReward/roll": terms.roll_reward.mean(),
                    "PureRLReward/angular_rate": terms.angular_rate_reward.mean(),
                    "PureRLPenalty/pitch_envelope": terms.pitch_envelope_penalty.mean(),
                    "PureRLPenalty/flap": terms.flap_penalty.mean(),
                    "PureRLPenalty/frequency_slew": terms.frequency_slew_penalty.mean(),
                    "PureRLPenalty/tail_action_delta": terms.tail_action_delta_penalty.mean(),
                    "PureRLPenalty/tail_action_limit": terms.tail_action_limit_penalty.mean(),
                    "PureRLContribution/path": reward_cfg.path_reward_weight * terms.path_reward.mean(),
                    "PureRLContribution/progress": (
                        reward_cfg.progress_reward_weight * terms.progress_reward.mean()
                    ),
                    "PureRLContribution/velocity": (
                        reward_cfg.velocity_reward_weight * terms.velocity_reward.mean()
                    ),
                    "PureRLContribution/roll": reward_cfg.roll_reward_weight * terms.roll_reward.mean(),
                    "PureRLContribution/angular_rate": (
                        reward_cfg.angular_rate_reward_weight * terms.angular_rate_reward.mean()
                    ),
                    "PureRLContribution/pitch_envelope": (
                        -reward_cfg.pitch_envelope_penalty_weight
                        * terms.pitch_envelope_penalty.mean()
                    ),
                    "PureRLContribution/flap": -reward_cfg.flap_penalty_weight * terms.flap_penalty.mean(),
                    "PureRLContribution/frequency_slew": (
                        -reward_cfg.frequency_slew_penalty_weight
                        * terms.frequency_slew_penalty.mean()
                    ),
                    "PureRLContribution/tail_action_delta": (
                        -reward_cfg.tail_action_delta_penalty_weight
                        * terms.tail_action_delta_penalty.mean()
                    ),
                    "PureRLContribution/tail_action_limit": (
                        -reward_cfg.tail_action_limit_penalty_weight
                        * terms.tail_action_limit_penalty.mean()
                    ),
                    "PureRLState/mean_abs_cross_track_error_m": cross_track_error_m.abs().mean(),
                    "PureRLState/mean_abs_height_error_m": height_error_m.abs().mean(),
                    "PureRLState/mean_along_track_velocity_mps": along_track_velocity_mps.mean(),
                    "PureRLState/mean_abs_cross_track_velocity_mps": (
                        cross_track_velocity_mps.abs().mean()
                    ),
                    "PureRLState/mean_abs_vertical_velocity_mps": vertical_velocity_mps.abs().mean(),
                    "PureRLState/mean_abs_roll_rad": roll_rad.abs().mean(),
                    "PureRLState/mean_abs_pitch_rad": pitch_rad.abs().mean(),
                    "PureRLState/mean_actual_flap_frequency_hz": self._freq.mean(),
                    "PureRLState/mean_requested_frequency_hz": self._requested_frequency_hz.mean(),
                    "PureRLState/mean_applied_frequency_hz": self._applied_frequency_hz.mean(),
                    "PureRLState/mean_abs_frequency_slew_hz_per_s": (
                        self._frequency_slew_hz_per_s.abs().mean()
                    ),
                    "PureRLState/frequency_governor_limited_fraction": (
                        self._frequency_governor_limited.float().mean()
                    ),
                }
            )
            if self._pure_rl_longitudinal_stage is not None:
                assert self._pure_rl_longitudinal_path is not None
                assert self._eval_pure_rl_active_slope_rad is not None
                assert self._eval_pure_rl_reached_recovery is not None
                log.update(
                    {
                        "PureRLPath/mean_sampled_task_id": (
                            self._pure_rl_longitudinal_path.task_id.to(torch.float32).mean()
                        ),
                        "PureRLPath/mean_sampled_signed_slope_rad": (
                            self._pure_rl_longitudinal_path.signed_slope_rad.mean()
                        ),
                        "PureRLPath/mean_active_slope_rad": self._eval_pure_rl_active_slope_rad.mean(),
                        "PureRLPath/recovery_reached_fraction": (
                            self._eval_pure_rl_reached_recovery.to(torch.float32).mean()
                        ),
                        "PureRLPath/mean_tangent_velocity_mps": along_track_velocity_mps.mean(),
                        "PureRLPath/mean_abs_lateral_normal_velocity_mps": (
                            cross_track_velocity_mps.abs().mean()
                        ),
                        "PureRLPath/mean_abs_vertical_normal_velocity_mps": (
                            vertical_velocity_mps.abs().mean()
                        ),
                    }
                )
        return terms.total_reward

    def _get_rewards(self) -> Tensor:
        if bool(self.cfg.use_pure_rl_curriculum1_reward):
            return self._get_pure_rl_curriculum1_reward()

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
        cross_track_m = self._straight_line_cross_track_m(pos_w)
        p_y = torch.tanh(torch.abs(cross_track_m) / float(self.cfg.terminate_abs_y))

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
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        if bool(self.cfg.use_pure_rl_curriculum1_reward):
            if self._pure_rl_longitudinal_stage is not None:
                query = self._query_pure_rl_longitudinal_path()
                cross_track_error_m = query.cross_track_error_m
                height_error_m = query.height_error_m
            else:
                cross_track_error_m = self._straight_line_cross_track_m(pos_w)
                height_error_m = height - self._height_cmd
            terms = compute_pure_rl_termination_terms(
                height_m=height,
                cross_track_error_m=cross_track_error_m,
                height_error_m=height_error_m,
                projected_gravity_body=self._robot.data.projected_gravity_b,
                config=self._pure_rl_termination_cfg,
            )
            assert self._eval_pure_rl_tilt_rad is not None
            assert self._eval_pure_rl_ground_termination is not None
            assert self._eval_pure_rl_tilt_termination is not None
            assert self._eval_pure_rl_cross_track_termination is not None
            assert self._eval_pure_rl_height_termination is not None
            self._eval_pure_rl_tilt_rad.copy_(terms.tilt_rad)
            self._eval_pure_rl_ground_termination.copy_(terms.ground)
            self._eval_pure_rl_tilt_termination.copy_(terms.tilt)
            self._eval_pure_rl_cross_track_termination.copy_(terms.cross_track)
            self._eval_pure_rl_height_termination.copy_(terms.height_error)
            if bool(self.cfg.pure_rl_reward_telemetry_enabled):
                log = self.extras.setdefault("log", {})
                log.update(
                    {
                        "PureRLTermination/ground_fraction": terms.ground.to(torch.float32).mean(),
                        "PureRLTermination/tilt_fraction": terms.tilt.to(torch.float32).mean(),
                        "PureRLTermination/cross_track_fraction": (
                            terms.cross_track.to(torch.float32).mean()
                        ),
                        "PureRLTermination/height_error_fraction": (
                            terms.height_error.to(torch.float32).mean()
                        ),
                        "PureRLTermination/terminated_fraction": (
                            terms.terminated.to(torch.float32).mean()
                        ),
                        "PureRLTermination/time_out_fraction": timed_out.to(torch.float32).mean(),
                        "PureRLState/mean_tilt_rad": terms.tilt_rad.mean(),
                    }
                )
            return terms.terminated, timed_out

        fell = height <= float(self.cfg.terminate_ground_height)

        # tilt termination
        g_b = self._robot.data.projected_gravity_b
        tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
        tilt_thr = torch.sin(torch.deg2rad(torch.tensor(self.cfg.terminate_tilt_deg, device=self.device)))
        fell = fell | (tilt > tilt_thr)

        # lateral bound
        cross_track_m = self._straight_line_cross_track_m(pos_w)
        fell = fell | (torch.abs(cross_track_m) > float(self.cfg.terminate_abs_y))

        return fell, timed_out
