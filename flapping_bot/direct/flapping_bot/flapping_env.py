"""Direct RL environment scaffolding for the flapping-wing robot."""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from pathlib import Path

from ...assets import FlappingBotCfg
from ...physics import FlappingQSMCfg, QuasiSteadyWingModel, WingQSMCfg
from ...scenes import FlappingRoomSceneCfg


@configclass
class FlappingBotEnvCfg(DirectRLEnvCfg):
    """Simulation and control configuration for the flapping bot environment."""

    # episode / control props
    episode_length_s = 10.0
    decimation = 4
    action_space = 4
    observation_space = 19
    state_space = 0
    action_scale = 1.0
    hover_height = 0.3

    # UI configuration: disable custom UI window to avoid Manager visualizer warnings
    ui_window_class_type = None

    # physics configuration
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 240.0,
        render_interval=decimation,
        # Use standard gravity in the scene
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
    )

    # scene and assets
    scene: InteractiveSceneCfg = FlappingRoomSceneCfg(num_envs=512, env_spacing=5.0)
    robot: ArticulationCfg = FlappingBotCfg.replace(prim_path="/World/envs/env_.*/Robot")

    # controller details
    controlled_joints: tuple[str, ...] = (
        "left_wing",
        "right_wing",
        "left_tail",
        "right_tail",
        "mid_tail",
    )
    joint_limit_softness: float = 0.98  # shrink hard limits slightly to avoid instability
    terminate_height_bounds: tuple[float, float] = (0.05, 2.0)
    qsm: FlappingQSMCfg = FlappingQSMCfg(
        wings=(
            WingQSMCfg(
                name="left_wing",
                joint_name="left_wing",
                hinge_axis_body=(0.0, 1.0, 0.0),
                lever_arm_body=(0.0, 0.18, 0.02),
                area=0.0075,
                lift_coefficient=1.2,
                drag_coefficient=0.18,
                effective_radius_fraction=0.75,
                hinge_damping=0.015,
            ),
            WingQSMCfg(
                name="right_wing",
                joint_name="right_wing",
                hinge_axis_body=(0.0, -1.0, 0.0),
                lever_arm_body=(0.0, -0.18, 0.02),
                area=0.0075,
                lift_coefficient=1.2,
                drag_coefficient=0.18,
                effective_radius_fraction=0.75,
                hinge_damping=0.015,
            ),
            WingQSMCfg(
                name="left_tail",
                joint_name="left_tail",
                hinge_axis_body=(0.0, 1.0, 0.0),
                lever_arm_body=(-0.01, -0.32, 0.04),
                area=0.0025,
                lift_coefficient=0.8,
                drag_coefficient=0.12,
                effective_radius_fraction=0.6,
                hinge_damping=0.01,
            ),
            WingQSMCfg(
                name="right_tail",
                joint_name="right_tail",
                hinge_axis_body=(0.0, -1.0, 0.0),
                lever_arm_body=(-0.01, -0.32, 0.04),
                area=0.0025,
                lift_coefficient=0.8,
                drag_coefficient=0.12,
                effective_radius_fraction=0.6,
                hinge_damping=0.01,
            ),
        ),
        air_density=1.225,
    )

    # Flapping-only demo: fixed frequency in Hz for both wings (0-5)
    flapping_freq_hz: float = 2.0
    # Whether to read frequency from actions (a0/a1). For this demo keep False.
    use_action_frequency: bool = False

    # Runtime mass override (advanced): apply MassAPI values from config file.
    # Warning: Mutating USD after PhysX views are created can invalidate tensor views in some versions.
    # Keep disabled by default for stability.
    mass_props_enable: bool = False
    mass_props_path: str | None = None  # If None, uses extension default config/mass_props.json


class FlappingBotEnv(DirectRLEnv):
    """Minimal direct RL environment wiring for the flapping-wing platform."""

    cfg: FlappingBotEnvCfg

    def __init__(self, cfg: FlappingBotEnvCfg, render_mode: str | None = None, **kwargs):
        self._num_actuators = len(cfg.controlled_joints)
        self._actions: torch.Tensor | None = None
        self._joint_targets: torch.Tensor | None = None
        self._robot: Articulation | None = None
        self._joint_ids: list[int] = []
        self._joint_lower_limits: torch.Tensor | None = None
        self._joint_upper_limits: torch.Tensor | None = None
        self._joint_mid: torch.Tensor | None = None
        self._joint_half_range: torch.Tensor | None = None
        self._default_joint_pos: torch.Tensor | None = None
        self._qsm_model: QuasiSteadyWingModel | None = None
        self._qsm_joint_indices: list[int] = []
        self._qsm_joint_tensor_idx: torch.Tensor | None = None
        self._qsm_force: torch.Tensor | None = None
        self._qsm_torque: torch.Tensor | None = None

        super().__init__(cfg, render_mode, **kwargs)

        # Resolve joints after scene creation
        joint_ids, joint_names = self._robot.find_joints(self.cfg.controlled_joints, preserve_order=True)
        if len(joint_ids) != self._num_actuators:
            raise RuntimeError(
                f"Expected {self._num_actuators} joints but resolved {len(joint_ids)}: {joint_names}. "
                "Check controlled_joints names against the URDF."
            )
        self._joint_ids = joint_ids

        joint_limits = self._robot.data.joint_pos_limits[0, self._joint_ids].to(device=self.device)
        lower = joint_limits[:, 0]
        upper = joint_limits[:, 1]
        limit_span = upper - lower
        margin = (1.0 - self.cfg.joint_limit_softness) * limit_span * 0.5
        self._joint_lower_limits = lower + margin
        self._joint_upper_limits = upper - margin
        self._joint_mid = 0.5 * (self._joint_lower_limits + self._joint_upper_limits)
        self._joint_half_range = 0.5 * (self._joint_upper_limits - self._joint_lower_limits)
        self._default_joint_pos = self._robot.data.default_joint_pos[0, self._joint_ids].to(device=self.device)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._joint_targets = self._default_joint_pos.expand(self.num_envs, -1).clone()
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

        # Optional: runtime mass override (disabled by default for stability)
        if self.cfg.mass_props_enable:
            try:
                import json
                import omni.usd
                from pxr import UsdPhysics, Gf
                cfg_path = (
                    Path(self.cfg.mass_props_path)
                    if self.cfg.mass_props_path
                    else (Path(__file__).resolve().parents[2] / "config" / "mass_props.json")
                )
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        mass_cfg = json.load(f)
                    stage = omni.usd.get_context().get_stage()
                    robot_prims = sim_utils.find_matching_prims(self._robot.cfg.prim_path)
                    for robot_prim in robot_prims:
                        base = robot_prim.GetPath().pathString
                        for link_name, props in mass_cfg.items():
                            link_path = f"{base}/{link_name}"
                            prim = stage.GetPrimAtPath(link_path)
                            if not prim or not prim.IsValid():
                                continue
                            mass_api = UsdPhysics.MassAPI.Apply(prim)
                            m = float(props.get("mass", 0.0))
                            com = props.get("com", [0.0, 0.0, 0.0])
                            I = props.get("inertia", [0.0, 0.0, 0.0])
                            mass_api.CreateMassAttr().Set(m)
                            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com))
                            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*I))
            except Exception:
                # Non-fatal: keep defaults if override fails
                pass

        if cfg.qsm and cfg.qsm.wings:
            self._qsm_model = QuasiSteadyWingModel(cfg.qsm, self.device)
            self._qsm_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
            self._qsm_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
            # map wings to joint indices
            self._qsm_joint_indices = []
            for wing in self.cfg.qsm.wings:
                try:
                    idx = self.cfg.controlled_joints.index(wing.joint_name)
                except ValueError as exc:
                    raise RuntimeError(
                        f"QSM wing '{wing.name}' references joint '{wing.joint_name}' which is not part of controlled_joints."
                    ) from exc
                self._qsm_joint_indices.append(idx)
            self._qsm_joint_tensor_idx = torch.tensor(self._qsm_joint_indices, dtype=torch.long, device=self.device)
        self._root_id = 0

        # Control indices for convenience (ordered as controlled_joints)
        self._IDX_LEFT_WING = 0
        self._IDX_RIGHT_WING = 1
        self._IDX_LEFT_TAIL = 2
        self._IDX_RIGHT_TAIL = 3
        self._IDX_MID_TAIL = 4

        # Action indices: [freq, tail_pitch, tail_roll, mid_tail]
        self._ACT_IDX_FREQ = 0
        self._ACT_IDX_TAIL_PITCH = 1
        self._ACT_IDX_TAIL_ROLL = 2
        self._ACT_IDX_MID_TAIL = 3

        # Time accumulator for flapping phase (seconds)
        self._time = torch.zeros(self.num_envs, device=self.device)
        # Frequency buffers (Hz)
        self._freq_left = torch.full((self.num_envs,), self.cfg.flapping_freq_hz, device=self.device)
        self._freq_right = torch.full((self.num_envs,), self.cfg.flapping_freq_hz, device=self.device)

    # ---------------------------------------------------------------------
    # Scene / asset setup
    # ---------------------------------------------------------------------
    def _setup_scene(self):
        # Explicitly instantiate the robot (direct workflow pattern)
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Clone environment instances
        self.scene.clone_environments(copy_from_source=False)

        # Do not access joint views here; physics views are ready after __init__ completes

    # ---------------------------------------------------------------------
    # Action processing and application
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # Cache actions in [-1, 1]
        self._actions = actions.clamp(-1.0, 1.0)

        if self.cfg.use_action_frequency:
            # One shared frequency (Hz) in [0, 5] for both wings
            f = 0.5 * (self._actions[:, self._ACT_IDX_FREQ] + 1.0) * 5.0
            self._freq_left = f
            self._freq_right = f
        else:
            # Use fixed frequency from cfg
            self._freq_left.fill_(self.cfg.flapping_freq_hz)
            self._freq_right.fill_(self.cfg.flapping_freq_hz)

        # Tail commands: pitch (same sign), roll (opposite sign)
        tail_max_L = self._joint_upper_limits[self._IDX_LEFT_TAIL]
        tail_max_R = self._joint_upper_limits[self._IDX_RIGHT_TAIL]
        tail_max = torch.minimum(tail_max_L, tail_max_R)
        tail_pitch = self._actions[:, self._ACT_IDX_TAIL_PITCH] * tail_max
        tail_roll = self._actions[:, self._ACT_IDX_TAIL_ROLL] * tail_max
        # Combine: left = pitch + roll, right = pitch - roll
        self._left_tail_cmd = torch.clamp(tail_pitch + tail_roll,
                                          self._joint_lower_limits[self._IDX_LEFT_TAIL],
                                          self._joint_upper_limits[self._IDX_LEFT_TAIL])
        self._right_tail_cmd = torch.clamp(tail_pitch - tail_roll,
                                           self._joint_lower_limits[self._IDX_RIGHT_TAIL],
                                           self._joint_upper_limits[self._IDX_RIGHT_TAIL])
        # Mid tail (single deflection command)
        mid_max = torch.minimum(
            torch.abs(self._joint_lower_limits[self._IDX_MID_TAIL]),
            torch.abs(self._joint_upper_limits[self._IDX_MID_TAIL]),
        )
        self._mid_tail_cmd = torch.clamp(
            self._actions[:, self._ACT_IDX_MID_TAIL] * mid_max,
            self._joint_lower_limits[self._IDX_MID_TAIL],
            self._joint_upper_limits[self._IDX_MID_TAIL],
        )

    def _apply_action(self):
        # Advance time by physics dt
        self._time += self.physics_dt

        # Build joint targets per actuator
        jt = self._joint_targets.clone()

        # Wing sine targets with asymmetric ranges:
        # Left  in [-60°, 0°], Right in [0°, 60°].
        # Use sin(2π f t) mapped to [0,1] then to [lower, upper].
        two_pi = 6.283185307179586
        # Left wing
        lower_L = self._joint_lower_limits[self._IDX_LEFT_WING]
        upper_L = self._joint_upper_limits[self._IDX_LEFT_WING]
        span_L = (upper_L - lower_L) * 0.95
        phase01_L = 0.5 * (torch.sin(two_pi * self._freq_left * self._time) + 1.0)
        jt[:, self._IDX_LEFT_WING] = torch.clamp(lower_L + span_L * phase01_L, lower_L, upper_L)

        # Right wing (mirror: 0..60)
        lower_R = self._joint_lower_limits[self._IDX_RIGHT_WING]
        upper_R = self._joint_upper_limits[self._IDX_RIGHT_WING]
        span_R = (upper_R - lower_R) * 0.95
        phase01_R = 0.5 * (torch.sin(two_pi * self._freq_right * self._time) + 1.0)
        jt[:, self._IDX_RIGHT_WING] = torch.clamp(upper_R - span_R * phase01_R, lower_R, upper_R)

        # Tails
        jt[:, self._IDX_LEFT_TAIL] = self._left_tail_cmd
        jt[:, self._IDX_RIGHT_TAIL] = self._right_tail_cmd
        jt[:, self._IDX_MID_TAIL] = self._mid_tail_cmd

        self._joint_targets = jt
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

        if self._qsm_model is not None and self._qsm_joint_tensor_idx is not None:
            joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
            joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
            qsm_joint_pos = joint_pos[:, self._qsm_joint_tensor_idx]
            qsm_joint_vel = joint_vel[:, self._qsm_joint_tensor_idx]

            forces, torques, hinge_torques = self._qsm_model.compute_forces(
                joint_pos=qsm_joint_pos,
                joint_vel=qsm_joint_vel,
                root_lin_vel=self._robot.data.root_lin_vel_b,
                root_ang_vel=self._robot.data.root_ang_vel_b,
            )
            self._qsm_force[:, 0, :] = forces
            self._qsm_torque[:, 0, :] = torques
            self._robot.set_external_force_and_torque(self._qsm_force, self._qsm_torque, body_ids=self._root_id)

            joint_efforts = torch.zeros_like(joint_pos)
            joint_efforts[:, self._qsm_joint_tensor_idx] = hinge_torques
            self._robot.set_joint_effort_target(joint_efforts, joint_ids=self._joint_ids)

    # ---------------------------------------------------------------------
    # Observations / rewards / terminations
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                joint_pos,
                joint_vel,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        height_error = self._robot.data.root_pos_w[:, 2] - self.cfg.hover_height
        hover_reward = 1.0 - torch.tanh(torch.abs(height_error) / 0.2)
        action_penalty = torch.sum(self._actions**2, dim=1) * 0.01
        reward = hover_reward - action_penalty
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        height = self._robot.data.root_pos_w[:, 2]
        lower, upper = self.cfg.terminate_height_bounds
        fell = torch.logical_or(height < lower, height > upper)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, timed_out

    # ---------------------------------------------------------------------
    # Reset handling
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        self._joint_targets[env_ids] = self._default_joint_pos
        self._actions[env_ids] = 0.0
        self._robot.set_joint_position_target(self._joint_targets[env_ids], joint_ids=self._joint_ids, env_ids=env_ids)
        if self._qsm_force is not None:
            self._qsm_force[env_ids] = 0.0
        if self._qsm_torque is not None:
            self._qsm_torque[env_ids] = 0.0
