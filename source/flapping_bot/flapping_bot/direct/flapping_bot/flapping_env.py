"""Direct RL environment for the flapping-wing robot (cleaned)."""

from __future__ import annotations

from dataclasses import dataclass
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

from ...assets import FlappingBotCfg
from ...physics import FlappingQSMCfg, QuasiSteadyWingModel, WingQSMCfg
from ...scenes import FlappingRoomSceneCfg


@configclass
class FlappingBotEnvCfg(DirectRLEnvCfg):
    # episode / control
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 3
    observation_space: int = 14
    state_space: int = 0
    action_scale: float = 1.0
    hover_height: float = 10.0
    # command targets (added to obs)
    vx_cmd: float = 3.0
    height_cmd: float = 10.0
    randomize_commands: bool = False
    vx_cmd_range: tuple[float, float] = (0.0, 5.0)
    height_cmd_range: tuple[float, float] = (8.0, 12.0)
    min_flap_hz: float = 1.0
    # action filtering (normalized action space [-1, 1])
    act_lpf_tau_s: float = 0.1  # 0: off; first-order low-pass time constant (s)
    act_rate_limit_per_s: float = 2.0  # 0: off; max |delta a| per second in normalized units

    # UI
    ui_window_class_type = None
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        eye=(8.0, 0.0, 4.0),
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
    robot: ArticulationCfg = FlappingBotCfg.replace(prim_path="/World/envs/env_.*/Robot")

    # control
    controlled_joints: Tuple[str, ...] = (
        "left_wing",
        "right_wing",
        "left_tail",
        "right_tail",
    )
    joint_limit_softness: float = 0.98
    terminate_ground_height: float = 0.05
    terminate_tilt_deg: float = 60.0  # terminate when tilt exceeds this (approx roll/pitch limit)

    # QSM (optional; kept minimal and robust to missing joints)
    qsm: FlappingQSMCfg = FlappingQSMCfg(
        wings=(
            WingQSMCfg(
                name="left_wing",
                joint_name="left_wing",
                hinge_axis_body=(1.0, 0.0, 0.0),
                lever_arm_body=(0.0, 0.18, 0.02),
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
                lever_arm_body=(0.0, -0.18, 0.02),
                area=0.165624,
                lift_coefficient=1.2,
                drag_coefficient=0.18,
                effective_radius_fraction=0.75,
                hinge_damping=0.01,
            ),
        ),
        air_density=1.225,
    )

    # flapping frequency control
    flapping_freq_hz: float = 2.0
    use_action_frequency: bool = True


class FlappingBotEnv(DirectRLEnv):
    cfg: FlappingBotEnvCfg

    def __init__(self, cfg: FlappingBotEnvCfg, render_mode: str | None = None, **kwargs):
        # runtime buffers
        self._robot: Articulation | None = None
        self._joint_ids: list[int] = []
        self._resolved_joint_names: list[str] = []
        self._joint_lower_limits: torch.Tensor | None = None
        self._joint_upper_limits: torch.Tensor | None = None
        self._default_joint_pos: torch.Tensor | None = None
        self._joint_targets: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self._act_lpf: torch.Tensor | None = None
        self._act_cmd: torch.Tensor | None = None

        # wing phase and frequency
        self._phase_left: torch.Tensor | None = None
        self._phase_right: torch.Tensor | None = None
        self._freq_left: torch.Tensor | None = None
        self._freq_right: torch.Tensor | None = None

        # tail command buffers
        self._left_tail_cmd: torch.Tensor | None = None
        self._right_tail_cmd: torch.Tensor | None = None
        # command buffers
        self._vx_cmd: torch.Tensor | None = None
        self._height_cmd: torch.Tensor | None = None

        # indices
        self._IDX_LEFT_WING = None
        self._IDX_RIGHT_WING = None
        self._IDX_LEFT_TAIL = None
        self._IDX_RIGHT_TAIL = None

        # QSM
        self._qsm_model: QuasiSteadyWingModel | None = None
        self._qsm_joint_indices: list[int] = []
        self._qsm_joint_tensor_idx: torch.Tensor | None = None

        super().__init__(cfg, render_mode, **kwargs)

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

        # joint targets
        self._joint_targets = self._default_joint_pos.expand(self.num_envs, -1).clone()
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

        # optional QSM (filter by available joints)
        if self.cfg.qsm and getattr(self.cfg.qsm, "wings", None):
            typed = []
            for w in self.cfg.qsm.wings:
                try:
                    typed.append(w if isinstance(w, WingQSMCfg) else WingQSMCfg(**w))
                except Exception:
                    pass
            filtered = [w for w in typed if w.joint_name in self._resolved_joint_names]
            if filtered:
                qsm_cfg = FlappingQSMCfg(wings=tuple(filtered), air_density=getattr(self.cfg.qsm, "air_density", 1.225))
                self._qsm_model = QuasiSteadyWingModel(qsm_cfg, self.device)
                self._qsm_joint_indices = [name_to_idx[w.joint_name] for w in filtered]
                self._qsm_joint_tensor_idx = torch.tensor(self._qsm_joint_indices, dtype=torch.long, device=self.device)

        # control indices
        self._IDX_LEFT_WING = name_to_idx.get("left_wing")
        self._IDX_RIGHT_WING = name_to_idx.get("right_wing")
        self._IDX_LEFT_TAIL = name_to_idx.get("left_tail")
        self._IDX_RIGHT_TAIL = name_to_idx.get("right_tail")

        # wing phases and frequencies
        self._phase_left = torch.zeros(self.num_envs, device=self.device)
        self._phase_right = torch.zeros(self.num_envs, device=self.device)
        self._freq_left = torch.full((self.num_envs,), self.cfg.flapping_freq_hz, device=self.device)
        self._freq_right = torch.full((self.num_envs,), self.cfg.flapping_freq_hz, device=self.device)
        # initialize commands
        self._vx_cmd = torch.full((self.num_envs,), float(self.cfg.vx_cmd), device=self.device)
        self._height_cmd = torch.full((self.num_envs,), float(self.cfg.height_cmd), device=self.device)

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
    def _pre_physics_step(self, actions: torch.Tensor):
        # raw actions in [-1, 1]
        self._actions = actions.clamp(-1.0, 1.0)
        # low-pass filter
        if self.cfg.act_lpf_tau_s > 0.0:
            alpha = float(self.step_dt) / (self.cfg.act_lpf_tau_s + float(self.step_dt))
            self._act_lpf = self._act_lpf + alpha * (self._actions - self._act_lpf)
        else:
            self._act_lpf = self._actions
        # slew-rate limit
        if self.cfg.act_rate_limit_per_s > 0.0:
            max_delta = self.cfg.act_rate_limit_per_s * float(self.step_dt)
            delta = torch.clamp(self._act_lpf - self._act_cmd, min=-max_delta, max=max_delta)
            self._act_cmd = self._act_cmd + delta
        else:
            self._act_cmd = self._act_lpf

        # frequency from action 0 in [0, 5] Hz
        if self.cfg.use_action_frequency:
            f = 0.5 * (self._act_cmd[:, 0] + 1.0) * 5.0
            f = torch.clamp(f, min=self.cfg.min_flap_hz)
            self._freq_left = f
            self._freq_right = f
        else:
            self._freq_left.fill_(self.cfg.flapping_freq_hz)
            self._freq_right.fill_(self.cfg.flapping_freq_hz)

        # tail: a1=俯仰(pitch)，a2=滚转(roll)，耦合成左右尾翼
        tail_lower_target = torch.tensor(-0.5235987756, device=self.device)  # -30 deg
        tail_upper_target = torch.tensor(0.724311, device=self.device)       # 41.5 deg
        # 与各自软限位求交集（安全范围）
        l_lower = torch.maximum(self._joint_lower_limits[self._IDX_LEFT_TAIL], tail_lower_target)
        l_upper = torch.minimum(self._joint_upper_limits[self._IDX_LEFT_TAIL], tail_upper_target)
        r_lower = torch.maximum(self._joint_lower_limits[self._IDX_RIGHT_TAIL], tail_lower_target)
        r_upper = torch.minimum(self._joint_upper_limits[self._IDX_RIGHT_TAIL], tail_upper_target)
        # 左右共同工作区取交集：下界取更大者，上界取更小者
        lower = torch.maximum(l_lower, r_lower)
        upper = torch.minimum(l_upper, r_upper)
        mid = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        # 俯仰/滚转分量（均映射到 [-half, +half]）
        pitch_off = half * self._act_cmd[:, 1]
        roll_off = half * self._act_cmd[:, 2]
        # 左右尾翼：同号俯仰 + 反号滚转
        left_cmd  = torch.clamp(mid + pitch_off + roll_off,  l_lower, l_upper)
        right_cmd = torch.clamp(mid + pitch_off - roll_off,  r_lower, r_upper)
        self._left_tail_cmd = left_cmd
        self._right_tail_cmd = right_cmd

    def _apply_action(self):
        # advance phase
        two_pi = 6.283185307179586
        self._phase_left = (self._phase_left + two_pi * self._freq_left * self.physics_dt) % two_pi
        self._phase_right = (self._phase_right + two_pi * self._freq_right * self.physics_dt) % two_pi

        # 机翼同号：范围目标 [-30deg, 30deg]，并与 URDF 软限位取交集后对称输出
        target_lower = torch.tensor(-0.5235987756, device=self.device)
        target_upper = torch.tensor(0.5235987756, device=self.device)
        l_lower = torch.maximum(self._joint_lower_limits[self._IDX_LEFT_WING], target_lower)
        l_upper = torch.minimum(self._joint_upper_limits[self._IDX_LEFT_WING], target_upper)
        r_lower = torch.maximum(self._joint_lower_limits[self._IDX_RIGHT_WING], target_lower)
        r_upper = torch.minimum(self._joint_upper_limits[self._IDX_RIGHT_WING], target_upper)
        # 统一对称中心与幅度（取保守值）
        mid_L = 0.5 * (l_lower + l_upper)
        mid_R = 0.5 * (r_lower + r_upper)
        amp = torch.minimum(0.5 * (l_upper - l_lower), 0.5 * (r_upper - r_lower))

        s = torch.sin(self._phase_left)
        left_cmd = torch.clamp(mid_L + amp * s, l_lower, l_upper)
        right_cmd = torch.clamp(mid_R + amp * s, r_lower, r_upper)

        jt = self._joint_targets
        jt[:, self._IDX_LEFT_WING] = left_cmd
        jt[:, self._IDX_RIGHT_WING] = right_cmd
        jt[:, self._IDX_LEFT_TAIL] = self._left_tail_cmd
        jt[:, self._IDX_RIGHT_TAIL] = self._right_tail_cmd
        self._robot.set_joint_position_target(jt, joint_ids=self._joint_ids)

        # aerodynamic forces via QSM (sum to base body)
        if self._qsm_model is not None and self._qsm_joint_tensor_idx is not None:
            # gather joint kinematics for wings used by QSM
            jpos_all = self._robot.data.joint_pos[:, self._joint_ids]
            jvel_all = self._robot.data.joint_vel[:, self._joint_ids]
            jpos = jpos_all[:, self._qsm_joint_tensor_idx]
            jvel = jvel_all[:, self._qsm_joint_tensor_idx]
            v_b = self._robot.data.root_lin_vel_b
            w_b = self._robot.data.root_ang_vel_b
            f_b, tau_b, _ = self._qsm_model.compute_forces(jpos, jvel, v_b, w_b)
            f_sum = torch.sum(f_b, dim=1).unsqueeze(1)  # [N,1,3]
            t_sum = torch.sum(tau_b, dim=1).unsqueeze(1)  # [N,1,3]
            # apply at base body (id 0) in body frame
            self._robot.set_external_force_and_torque(forces=f_sum, torques=t_sum, body_ids=[0], is_global=False)

    # ------------------------------------------------------------------
    # Observations / Rewards / Dones
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | list[int]):
        if isinstance(env_ids, list):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # root: z=hover_height, initial forward vx=5 m/s
        n = env_ids.shape[0]
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 2] = self.cfg.hover_height
        rot = torch.zeros(n, 4, device=self.device)
        rot[:, 0] = 1.0
        lin_vel = torch.zeros(n, 3, device=self.device)
        lin_vel[:, 0] = 5.0
        ang_vel = torch.zeros(n, 3, device=self.device)
        root_state = torch.cat([pos, rot, lin_vel, ang_vel], dim=1)
        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # joints to default and zero velocity
        jpos = self._default_joint_pos.expand(n, -1).clone()
        jvel = torch.zeros_like(jpos)
        self._robot.write_joint_state_to_sim(jpos, jvel, joint_ids=self._joint_ids, env_ids=env_ids)

        # clear actions and phases
        self._actions[env_ids] = 0.0
        self._phase_left[env_ids] = 0.0
        self._phase_right[env_ids] = 0.0
        self._freq_left[env_ids] = self.cfg.flapping_freq_hz
        self._freq_right[env_ids] = self.cfg.flapping_freq_hz
        # commands: randomize or set defaults per env
        if self.cfg.randomize_commands:
            vl, vh = self.cfg.vx_cmd_range
            zl, zh = self.cfg.height_cmd_range
            self._vx_cmd[env_ids] = torch.rand_like(self._vx_cmd[env_ids]) * (vh - vl) + vl
            self._height_cmd[env_ids] = torch.rand_like(self._height_cmd[env_ids]) * (zh - zl) + zl
        else:
            self._vx_cmd[env_ids] = float(self.cfg.vx_cmd)
            self._height_cmd[env_ids] = float(self.cfg.height_cmd)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # base kin
        pos_w = self._robot.data.root_pos_w
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        g_b = self._robot.data.projected_gravity_b
        # tail joint pos (if any)
        jpos = self._robot.data.joint_pos[:, self._joint_ids]

        # normalized/scaled observations + command targets
        # remove absolute x,y (invariance), keep height as error to command
        z_err = (pos_w[:, 2] - self._height_cmd) / 20.0
        lin_s = lin_vel_b[:, 0:3] / 10.0
        ang_s = ang_vel_b[:, 0:3] / 10.0
        g_s = g_b[:, 0:3]
        # tail normalized by max magnitude of soft limits
        l_idx, r_idx = self._IDX_LEFT_TAIL, self._IDX_RIGHT_TAIL
        l_den = torch.maximum(self._joint_upper_limits[l_idx].abs(), self._joint_lower_limits[l_idx].abs())
        r_den = torch.maximum(self._joint_upper_limits[r_idx].abs(), self._joint_lower_limits[r_idx].abs())
        tail_s = torch.stack([jpos[:, l_idx] / l_den, jpos[:, r_idx] / r_den], dim=1)
        freq_s = self._freq_left.unsqueeze(1) / 5.0
        # forward velocity tracking: use error (vx - vx_cmd)
        vx_err_s = (lin_vel_b[:, 0] - self._vx_cmd).unsqueeze(1) / 5.0

        obs = torch.cat([z_err.unsqueeze(1), lin_s, ang_s, g_s, tail_s, freq_s, vx_err_s], dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # altitude around hover
        height = self._robot.data.root_pos_w[:, 2]
        height_error = height - self.cfg.hover_height
        r_height = 1.0 - torch.tanh(torch.abs(height_error) / 0.5)

        # attitude stability (upright)
        g_b = self._robot.data.projected_gravity_b
        tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
        r_tilt = 1.0 - torch.tanh(tilt * 2.0)
        ang = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        p_ang = 0.05 * torch.tanh(ang)

        # forward velocity (body-x)
        v_fwd = torch.clamp(self._robot.data.root_lin_vel_b[:, 0], min=0.0)
        r_fwd = torch.tanh(v_fwd / 2.0)

        # action penalty
        p_act = 0.01 * torch.sum(self._act_cmd ** 2, dim=1)

        return 0.5 * r_height + 0.3 * r_tilt + 0.3 * r_fwd - p_act - p_ang

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        height = self._robot.data.root_pos_w[:, 2]
        fell = height <= self.cfg.terminate_ground_height
        # tilt termination using projected gravity in body frame
        g_b = self._robot.data.projected_gravity_b
        tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
        tilt_thr = torch.sin(torch.deg2rad(torch.tensor(self.cfg.terminate_tilt_deg, device=self.device)))
        fell = fell | (tilt > tilt_thr)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, timed_out
