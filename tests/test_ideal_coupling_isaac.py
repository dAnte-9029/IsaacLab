"""Headless PhysX feasibility test for ideal left/right wing coupling."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

from dataclasses import dataclass
import math
from pathlib import Path

import pytest
from pxr import PhysxSchema, UsdPhysics
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import build_simulation_context


_URDF_PATH = Path(__file__).parent / "data" / "ideal_coupled_wings.urdf"
_STROKE_AMPLITUDE_RAD = math.radians(30.0)
_SYNC_ACCEPTANCE_RAD = math.radians(0.1)
_DRIVER_STIFFNESS_NM_PER_RAD = 2_000.0
_DRIVER_DAMPING_NM_S_PER_RAD = 20.0
_DRIVER_EFFORT_LIMIT_NM = 100.0
_MIMIC_NATURAL_FREQUENCY_ATTR = "physxMimicJoint:rotX:naturalFrequency"
_MIMIC_DAMPING_RATIO_ATTR = "physxMimicJoint:rotX:dampingRatio"


@dataclass(frozen=True)
class _CouplingMetrics:
    """Metrics from one coupled-wing PhysX run."""

    frequency_hz: float
    dt_s: float
    asymmetric_right_torque_nm: float
    max_sync_error_rad: float
    rms_sync_error_rad: float
    max_driver_tracking_error_rad: float
    rms_driver_tracking_error_rad: float
    max_driver_effort_nm: float
    max_base_vertical_speed_m_s: float
    max_base_pitch_rate_rad_s: float
    max_base_roll_yaw_rate_rad_s: float


def _make_robot(usd_dir: Path, *, solver_position_iterations: int = 16) -> Articulation:
    """Create a floating two-wing articulation with one driver and one mimic joint."""

    cfg = ArticulationCfg(
        prim_path="/World/IdealCoupledWings",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(_URDF_PATH),
            usd_dir=str(usd_dir),
            usd_file_name="ideal_coupled_wings.usd",
            fix_base=False,
            merge_fixed_joints=True,
            # In Isaac Lab 0.48.6 this flag is forwarded to the importer as
            # parse_mimic=True, preserving the URDF relation as
            # PhysxMimicJointAPI instead of dropping it.
            convert_mimic_joints_to_normal_joints=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            make_instanceable=False,
            force_usd_conversion=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=solver_position_iterations,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                retain_accelerations=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"left_wing": 0.0, "right_wing": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "driver": ImplicitActuatorCfg(
                joint_names_expr=["left_wing"],
                effort_limit_sim=_DRIVER_EFFORT_LIMIT_NM,
                velocity_limit_sim=200.0,
                stiffness=_DRIVER_STIFFNESS_NM_PER_RAD,
                damping=_DRIVER_DAMPING_NM_S_PER_RAD,
            ),
            "passive_mimic": ImplicitActuatorCfg(
                joint_names_expr=["right_wing"],
                effort_limit_sim=_DRIVER_EFFORT_LIMIT_NM,
                velocity_limit_sim=200.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    return Articulation(cfg)


def _mimic_api(robot: Articulation) -> PhysxSchema.PhysxMimicJointAPI:
    candidates = [
        prim
        for prim in robot.stage.Traverse()
        if prim.GetName() == "right_wing" and prim.HasAPI(PhysxSchema.PhysxMimicJointAPI)
    ]
    assert len(candidates) == 1
    return PhysxSchema.PhysxMimicJointAPI(candidates[0], UsdPhysics.Tokens.rotX)


def _set_hard_mimic_constraint(robot: Articulation) -> tuple[float, float]:
    """Disable importer-provided mimic compliance before PhysX initialization."""

    mimic_prim = _mimic_api(robot).GetPrim()
    natural_frequency_attr = mimic_prim.GetAttribute(_MIMIC_NATURAL_FREQUENCY_ATTR)
    damping_ratio_attr = mimic_prim.GetAttribute(_MIMIC_DAMPING_RATIO_ATTR)
    assert natural_frequency_attr.IsValid()
    assert damping_ratio_attr.IsValid()
    imported_values = float(natural_frequency_attr.Get()), float(damping_ratio_attr.Get())
    natural_frequency_attr.Set(0.0)
    damping_ratio_attr.Set(0.0)
    return imported_values


def _run_case(
    robot: Articulation,
    sim,
    *,
    frequency_hz: float,
    dt_s: float,
    asymmetric_right_torque_nm: float,
) -> _CouplingMetrics:
    """Run four wingbeats and evaluate the final two wingbeats."""

    joint_ids, _ = robot.find_joints(["left_wing", "right_wing"], preserve_order=True)
    body_ids, _ = robot.find_bodies(["base_link", "right_wing"], preserve_order=True)
    left_joint_id, right_joint_id = (int(index) for index in joint_ids)
    base_body_id, right_body_id = (int(index) for index in body_ids)

    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
    robot.write_joint_state_to_sim(
        torch.zeros_like(robot.data.default_joint_pos),
        torch.zeros_like(robot.data.default_joint_vel),
    )
    robot.reset()
    sim.forward()
    robot.update(dt_s)

    steps_per_cycle = round(1.0 / (frequency_hz * dt_s))
    total_steps = 4 * steps_per_cycle
    measurement_start = 2 * steps_per_cycle
    sync_errors: list[float] = []
    tracking_errors: list[float] = []
    driver_efforts: list[float] = []
    base_vertical_speeds: list[float] = []
    base_pitch_rates: list[float] = []
    base_roll_yaw_rates: list[float] = []

    zero_force = torch.zeros((1, 2, 3), device=robot.device)
    hinge_load_torque = torch.zeros_like(zero_force)

    for step in range(total_steps):
        time_s = step * dt_s
        phase_rad = 2.0 * math.pi * frequency_hz * time_s
        q_ref_rad = _STROKE_AMPLITUDE_RAD * math.sin(phase_rad)
        qd_ref_rad_s = (
            _STROKE_AMPLITUDE_RAD * 2.0 * math.pi * frequency_hz * math.cos(phase_rad)
        )
        disturbance_nm = asymmetric_right_torque_nm * math.sin(phase_rad)
        hinge_load_torque[:, 0, 0] = -disturbance_nm
        hinge_load_torque[:, 1, 0] = disturbance_nm
        robot.set_joint_position_target(
            torch.tensor([[q_ref_rad]], device=robot.device),
            joint_ids=[left_joint_id],
        )
        robot.set_joint_velocity_target(
            torch.tensor([[qd_ref_rad_s]], device=robot.device),
            joint_ids=[left_joint_id],
        )
        robot.set_external_force_and_torque(
            zero_force,
            hinge_load_torque,
            body_ids=[base_body_id, right_body_id],
            is_global=False,
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt_s)

        if step >= measurement_start:
            q_left = float(robot.data.joint_pos[0, left_joint_id].item())
            q_right = float(robot.data.joint_pos[0, right_joint_id].item())
            sync_errors.append(q_left + q_right)
            tracking_errors.append(q_left - q_ref_rad)
            driver_efforts.append(float(robot.data.applied_torque[0, left_joint_id].item()))
            base_vertical_speeds.append(float(robot.data.body_link_lin_vel_w[0, base_body_id, 2].item()))
            base_pitch_rates.append(float(robot.data.body_link_ang_vel_w[0, base_body_id, 1].item()))
            base_roll_yaw_rates.append(
                max(
                    abs(float(robot.data.body_link_ang_vel_w[0, base_body_id, 0].item())),
                    abs(float(robot.data.body_link_ang_vel_w[0, base_body_id, 2].item())),
                )
            )

    sync_tensor = torch.tensor(sync_errors, dtype=torch.float64)
    tracking_tensor = torch.tensor(tracking_errors, dtype=torch.float64)
    return _CouplingMetrics(
        frequency_hz=frequency_hz,
        dt_s=dt_s,
        asymmetric_right_torque_nm=asymmetric_right_torque_nm,
        max_sync_error_rad=float(torch.max(torch.abs(sync_tensor)).item()),
        rms_sync_error_rad=float(torch.sqrt(torch.mean(sync_tensor.square())).item()),
        max_driver_tracking_error_rad=float(torch.max(torch.abs(tracking_tensor)).item()),
        rms_driver_tracking_error_rad=float(torch.sqrt(torch.mean(tracking_tensor.square())).item()),
        max_driver_effort_nm=max(abs(value) for value in driver_efforts),
        max_base_vertical_speed_m_s=max(abs(value) for value in base_vertical_speeds),
        max_base_pitch_rate_rad_s=max(abs(value) for value in base_pitch_rates),
        max_base_roll_yaw_rate_rad_s=max(base_roll_yaw_rates),
    )


@pytest.mark.isaacsim_ci
def test_urdf_import_preserves_opposite_mimic_relation(tmp_path: Path) -> None:
    """Confirm the importer creates the intended PhysX mimic constraint."""

    with build_simulation_context(device="cpu", auto_add_lighting=False, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None
        robot = _make_robot(tmp_path)
        imported_compliance = _set_hard_mimic_constraint(robot)
        sim.reset()
        robot.update(sim.cfg.dt)

        mimic_api = _mimic_api(robot)
        print(
            "URDF importer mimic compliance:"
            f" natural_frequency={imported_compliance[0]:.6g},"
            f" damping_ratio={imported_compliance[1]:.6g}"
        )
        assert mimic_api.GetGearingAttr().Get() == pytest.approx(1.0)
        assert mimic_api.GetOffsetAttr().Get() == pytest.approx(0.0)
        reference_targets = mimic_api.GetReferenceJointRel().GetTargets()
        assert len(reference_targets) == 1
        assert str(reference_targets[0]).endswith("/left_wing")
        assert imported_compliance[0] > 0.0
        assert imported_compliance[1] > 0.0
        assert mimic_api.GetPrim().GetAttribute(_MIMIC_NATURAL_FREQUENCY_ATTR).Get() == pytest.approx(0.0)
        assert mimic_api.GetPrim().GetAttribute(_MIMIC_DAMPING_RATIO_ATTR).Get() == pytest.approx(0.0)


@pytest.mark.isaacsim_ci
def test_mimic_coupling_maintains_relation_under_asymmetric_hinge_load(tmp_path: Path) -> None:
    """Check synchronization, inertial reaction and time-step convergence."""

    dt_s = 1.0e-3
    with build_simulation_context(
        device="cpu",
        dt=dt_s,
        auto_add_lighting=False,
        gravity_enabled=False,
    ) as sim:
        sim._app_control_on_stop_handle = None
        robot = _make_robot(tmp_path / "dt_1ms")
        _set_hard_mimic_constraint(robot)
        sim.reset()
        robot.update(sim.cfg.dt)
        metrics = [
            _run_case(robot, sim, frequency_hz=2.0, dt_s=dt_s, asymmetric_right_torque_nm=0.0),
            _run_case(robot, sim, frequency_hz=5.0, dt_s=dt_s, asymmetric_right_torque_nm=0.0),
            _run_case(robot, sim, frequency_hz=5.0, dt_s=dt_s, asymmetric_right_torque_nm=0.25),
        ]

    for result in metrics:
        print(result)
        assert result.max_sync_error_rad < _SYNC_ACCEPTANCE_RAD

    unloaded_2hz = metrics[0]
    unloaded_5hz = metrics[1]
    loaded_5hz = metrics[2]
    assert unloaded_2hz.max_base_vertical_speed_m_s > 1.0e-3
    assert unloaded_2hz.max_base_pitch_rate_rad_s > 1.0e-3
    assert unloaded_5hz.max_base_vertical_speed_m_s > unloaded_2hz.max_base_vertical_speed_m_s
    assert unloaded_5hz.max_base_pitch_rate_rad_s > unloaded_2hz.max_base_pitch_rate_rad_s
    assert loaded_5hz.max_sync_error_rad < _SYNC_ACCEPTANCE_RAD

    fine_dt_s = 5.0e-4
    with build_simulation_context(
        device="cpu",
        dt=fine_dt_s,
        auto_add_lighting=False,
        gravity_enabled=False,
    ) as sim:
        sim._app_control_on_stop_handle = None
        robot = _make_robot(tmp_path / "dt_0p5ms")
        _set_hard_mimic_constraint(robot)
        sim.reset()
        robot.update(sim.cfg.dt)
        fine_loaded_5hz = _run_case(
            robot,
            sim,
            frequency_hz=5.0,
            dt_s=fine_dt_s,
            asymmetric_right_torque_nm=0.25,
        )

    print(fine_loaded_5hz)
    assert fine_loaded_5hz.max_sync_error_rad < _SYNC_ACCEPTANCE_RAD
    assert fine_loaded_5hz.max_sync_error_rad <= loaded_5hz.max_sync_error_rad
