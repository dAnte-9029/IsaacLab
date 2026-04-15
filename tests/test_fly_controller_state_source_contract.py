from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_fake_isaaclab_app(monkeypatch) -> None:
    isaaclab_mod = ModuleType("isaaclab")
    app_mod = ModuleType("isaaclab.app")

    class _FakeAppLauncher:
        def __init__(self, *args, **kwargs):
            self.app = None

        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--device", type=str, default="cpu")

    app_mod.AppLauncher = _FakeAppLauncher
    isaaclab_mod.app = app_mod
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_mod)


def _load_script_module(module_name: str, relative_path: str, monkeypatch):
    _install_fake_isaaclab_app(monkeypatch)
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_fly_straight_line_parser_accepts_teacher_state_source_and_imu_source(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_straight_line_module",
        "scripts/flapping_px4/fly_straight_line.py",
        monkeypatch,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fly_straight_line.py",
            "--state_source",
            "compare",
            "--teacher_state_source",
            "truth",
            "--imu_source",
            "isaacsim",
        ],
    )
    args = module._parse_args()
    selection = module._resolve_state_source_selection(args)

    assert args.teacher_state_source == "truth"
    assert args.imu_source == "isaacsim"
    assert selection.controller_state_source == "compare"
    assert selection.teacher_state_source == "truth"
    assert selection.policy_state_source == "estimated"
    assert selection.imu_source == "isaacsim"


def test_fly_straight_line_defaults_teacher_source_from_state_source(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_straight_line_defaults_module",
        "scripts/flapping_px4/fly_straight_line.py",
        monkeypatch,
    )
    monkeypatch.setattr(sys, "argv", ["fly_straight_line.py", "--state_source", "estimated"])
    args = module._parse_args()
    selection = module._resolve_state_source_selection(args)

    assert args.teacher_state_source is None
    assert selection.teacher_state_source == "estimated"
    assert selection.policy_state_source == "estimated"


def test_fly_straight_line_parser_accepts_env_seed(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_straight_line_seed_module",
        "scripts/flapping_px4/fly_straight_line.py",
        monkeypatch,
    )

    monkeypatch.setattr(sys, "argv", ["fly_straight_line.py"])
    default_args = module._parse_args()
    assert default_args.seed is None
    assert default_args.estimator_attitude_correction_mode == "gravity_vector"
    assert default_args.estimator_accel_hard_gate is True
    assert default_args.estimator_accel_gate_low_g == 0.9
    assert default_args.estimator_accel_gate_high_g == 1.1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fly_straight_line.py",
            "--seed",
            "123",
            "--estimator_attitude_correction_mode",
            "euler_lpf",
            "--no-estimator_accel_hard_gate",
            "--estimator_accel_gate_low_g",
            "0.8",
            "--estimator_accel_gate_high_g",
            "1.2",
        ],
    )
    seeded_args = module._parse_args()
    assert seeded_args.seed == 123
    assert seeded_args.estimator_attitude_correction_mode == "euler_lpf"
    assert seeded_args.estimator_accel_hard_gate is False
    assert seeded_args.estimator_accel_gate_low_g == 0.8
    assert seeded_args.estimator_accel_gate_high_g == 1.2


def test_fly_straight_line_parser_accepts_total_mass_override(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_straight_line_mass_override_module",
        "scripts/flapping_px4/fly_straight_line.py",
        monkeypatch,
    )

    monkeypatch.setattr(sys, "argv", ["fly_straight_line.py"])
    default_args = module._parse_args()
    assert default_args.total_mass_kg_override is None

    monkeypatch.setattr(sys, "argv", ["fly_straight_line.py", "--total_mass_kg_override", "0.95"])
    override_args = module._parse_args()
    assert override_args.total_mass_kg_override == 0.95


def test_fly_straight_line_apply_runtime_mass_override_updates_env_cfg() -> None:
    from scripts.flapping_px4 import fly_straight_line as module

    env_cfg = SimpleNamespace(total_mass_kg_override=None)
    args = SimpleNamespace(total_mass_kg_override=0.95)

    module._apply_runtime_mass_override_arg(env_cfg, args)

    assert env_cfg.total_mass_kg_override == 0.95


def test_fly_straight_line_apply_runtime_mass_override_leaves_env_cfg_untouched_when_arg_is_none() -> None:
    from scripts.flapping_px4 import fly_straight_line as module

    env_cfg = SimpleNamespace(total_mass_kg_override=None)
    args = SimpleNamespace(total_mass_kg_override=None)

    module._apply_runtime_mass_override_arg(env_cfg, args)

    assert env_cfg.total_mass_kg_override is None


def test_fly_loiter_parser_accepts_teacher_state_source_and_imu_source(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_loiter_module",
        "scripts/flapping_px4/fly_loiter.py",
        monkeypatch,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fly_loiter.py",
            "--state_source",
            "truth",
            "--teacher_state_source",
            "estimated",
            "--imu_source",
            "isaacsim",
        ],
    )
    args = module._parse_args()
    selection = module._resolve_state_source_selection(args)

    assert args.teacher_state_source == "estimated"
    assert args.imu_source == "isaacsim"
    assert selection.controller_state_source == "truth"
    assert selection.teacher_state_source == "estimated"
    assert selection.policy_state_source == "truth"
    assert selection.imu_source == "isaacsim"


def test_fly_loiter_defaults_teacher_source_from_compare_state_source(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_loiter_defaults_module",
        "scripts/flapping_px4/fly_loiter.py",
        monkeypatch,
    )
    monkeypatch.setattr(sys, "argv", ["fly_loiter.py", "--state_source", "compare"])
    args = module._parse_args()
    selection = module._resolve_state_source_selection(args)

    assert args.teacher_state_source is None
    assert selection.teacher_state_source == "estimated"
    assert selection.policy_state_source == "estimated"


def test_fly_loiter_parser_accepts_env_seed(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_loiter_seed_module",
        "scripts/flapping_px4/fly_loiter.py",
        monkeypatch,
    )

    monkeypatch.setattr(sys, "argv", ["fly_loiter.py"])
    default_args = module._parse_args()
    assert default_args.seed is None
    assert default_args.loiter_radius_m == 40.0
    assert default_args.estimator_attitude_correction_mode == "gravity_vector"
    assert default_args.estimator_accel_hard_gate is True
    assert default_args.estimator_accel_gate_low_g == 0.9
    assert default_args.estimator_accel_gate_high_g == 1.1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fly_loiter.py",
            "--seed",
            "123",
            "--estimator_attitude_correction_mode",
            "euler_lpf",
            "--no-estimator_accel_hard_gate",
            "--estimator_accel_gate_low_g",
            "0.8",
            "--estimator_accel_gate_high_g",
            "1.2",
        ],
    )
    seeded_args = module._parse_args()
    assert seeded_args.seed == 123
    assert seeded_args.estimator_attitude_correction_mode == "euler_lpf"
    assert seeded_args.estimator_accel_hard_gate is False
    assert seeded_args.estimator_accel_gate_low_g == 0.8
    assert seeded_args.estimator_accel_gate_high_g == 1.2


def test_fly_loiter_parser_accepts_total_mass_override(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_loiter_mass_override_module",
        "scripts/flapping_px4/fly_loiter.py",
        monkeypatch,
    )

    monkeypatch.setattr(sys, "argv", ["fly_loiter.py"])
    default_args = module._parse_args()
    assert default_args.total_mass_kg_override is None

    monkeypatch.setattr(sys, "argv", ["fly_loiter.py", "--total_mass_kg_override", "0.95"])
    override_args = module._parse_args()
    assert override_args.total_mass_kg_override == 0.95


def test_fly_loiter_apply_runtime_mass_override_updates_env_cfg() -> None:
    from scripts.flapping_px4 import fly_loiter as module

    env_cfg = SimpleNamespace(total_mass_kg_override=None)
    args = SimpleNamespace(total_mass_kg_override=0.95)

    module._apply_runtime_mass_override_arg(env_cfg, args)

    assert env_cfg.total_mass_kg_override == 0.95


def test_fly_loiter_apply_runtime_mass_override_leaves_env_cfg_untouched_when_arg_is_none() -> None:
    from scripts.flapping_px4 import fly_loiter as module

    env_cfg = SimpleNamespace(total_mass_kg_override=None)
    args = SimpleNamespace(total_mass_kg_override=None)

    module._apply_runtime_mass_override_arg(env_cfg, args)

    assert env_cfg.total_mass_kg_override is None


def test_fly_straight_line_build_imu_measurement_routes_specific_force_to_provider(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_straight_line_imu_module",
        "scripts/flapping_px4/fly_straight_line.py",
        monkeypatch,
    )

    captured = {}

    class RecordingProvider:
        def build_from_truth(self, *, ang_vel_body, specific_force_body):
            captured["ang_vel_body"] = ang_vel_body.clone()
            captured["specific_force_body"] = specific_force_body.clone()
            return SimpleNamespace(gyro_rad_s=ang_vel_body.clone(), accel_mps2=specific_force_body.clone())

    measurement = module._build_imu_measurement(
        RecordingProvider(),
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        vel_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        prev_vel_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        ang_vel_b=torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32),
        dt=0.01,
        quat_apply_inverse_fn=lambda quat, vec: vec,
    )

    assert torch.allclose(captured["ang_vel_body"], torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32))
    assert torch.allclose(captured["specific_force_body"], torch.tensor([[0.0, 0.0, 9.81]], dtype=torch.float32))
    assert torch.allclose(measurement.accel_mps2, torch.tensor([[0.0, 0.0, 9.81]], dtype=torch.float32))


def test_fly_loiter_build_imu_measurement_routes_specific_force_to_provider(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_loiter_imu_module",
        "scripts/flapping_px4/fly_loiter.py",
        monkeypatch,
    )

    captured = {}

    class RecordingProvider:
        def build_from_truth(self, *, ang_vel_body, specific_force_body):
            captured["ang_vel_body"] = ang_vel_body.clone()
            captured["specific_force_body"] = specific_force_body.clone()
            return SimpleNamespace(gyro_rad_s=ang_vel_body.clone(), accel_mps2=specific_force_body.clone())

    measurement = module._build_imu_measurement(
        RecordingProvider(),
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        vel_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        prev_vel_w=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        ang_vel_b=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        dt=0.5,
        quat_apply_inverse_fn=lambda quat, vec: vec,
    )

    assert torch.allclose(captured["ang_vel_body"], torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32))
    assert torch.allclose(captured["specific_force_body"], torch.tensor([[1.0, 0.0, 9.81]], dtype=torch.float32))
    assert torch.allclose(measurement.accel_mps2, torch.tensor([[1.0, 0.0, 9.81]], dtype=torch.float32))


def test_fly_straight_line_create_isaacsim_imu_sensor_uses_first_body_name(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_straight_line_create_sensor_module",
        "scripts/flapping_px4/fly_straight_line.py",
        monkeypatch,
    )
    fake_px4_like = ModuleType("flapping_bot.px4_like")

    class FakeSpec:
        def __init__(self, prim_path, update_period, offset_pos_b=(0.0, 0.0, 0.0)):
            self.prim_path = prim_path
            self.update_period = update_period
            self.offset_pos_b = offset_pos_b

    fake_px4_like.IsaacSimImuSensorSpec = FakeSpec
    fake_px4_like.resolve_base_body_com_offset_b = lambda robot, base_body_ids: tuple(
        float(v) for v in robot.data.body_com_pos_b[0, int(base_body_ids[0]), 0:3].tolist()
    )
    monkeypatch.setitem(sys.modules, "flapping_bot.px4_like", fake_px4_like)

    created = {}

    class FakeSensor:
        def reset(self):
            created["reset"] = True

        def update(self, dt, force_recompute=False):
            created["update"] = (dt, force_recompute)

    class FakeProvider:
        backend_name = "isaacsim"

        def create_sensor(self, spec):
            created["prim_path"] = spec.prim_path
            created["update_period"] = spec.update_period
            created["offset_pos_b"] = spec.offset_pos_b
            return FakeSensor()

    fake_env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _robot=SimpleNamespace(body_names=["base_link", "left_wing"]),
            _base_body_ids=[0],
            cfg=SimpleNamespace(robot=SimpleNamespace(prim_path="/World/envs/env_.*/Robot")),
        )
    )
    fake_env.unwrapped._robot.data = SimpleNamespace(
        body_com_pos_b=torch.tensor([[[0.15, -0.02, 0.03]]], dtype=torch.float32)
    )

    sensor = module._create_isaacsim_imu_sensor(FakeProvider(), fake_env, env_step_dt=0.02)
    assert sensor is not None
    assert created["prim_path"] == "/World/envs/env_.*/Robot/base_link"
    assert created["update_period"] == 0.02
    assert created["offset_pos_b"] == pytest.approx((0.15, -0.02, 0.03))
    assert created["reset"] is True
    assert created["update"] == (0.02, True)


def test_fly_straight_line_create_isaacsim_imu_sensor_initializes_late_sensor(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_straight_line_create_sensor_init_module",
        "scripts/flapping_px4/fly_straight_line.py",
        monkeypatch,
    )
    fake_px4_like = ModuleType("flapping_bot.px4_like")

    class FakeSpec:
        def __init__(self, prim_path, update_period, offset_pos_b=(0.0, 0.0, 0.0)):
            self.prim_path = prim_path
            self.update_period = update_period
            self.offset_pos_b = offset_pos_b

    fake_px4_like.IsaacSimImuSensorSpec = FakeSpec
    fake_px4_like.resolve_base_body_com_offset_b = lambda robot, base_body_ids: tuple(
        float(v) for v in robot.data.body_com_pos_b[0, int(base_body_ids[0]), 0:3].tolist()
    )
    monkeypatch.setitem(sys.modules, "flapping_bot.px4_like", fake_px4_like)

    created = {"initialize_impl": 0, "reset": 0, "update": 0}

    class FakeSensor:
        is_initialized = False

        def _initialize_impl(self):
            created["initialize_impl"] += 1

        def reset(self):
            created["reset"] += 1

        def update(self, dt, force_recompute=False):
            created["update"] += 1

    class FakeProvider:
        backend_name = "isaacsim"

        def create_sensor(self, spec):
            return FakeSensor()

    fake_env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _robot=SimpleNamespace(body_names=["base_link"]),
            _base_body_ids=[0],
            cfg=SimpleNamespace(robot=SimpleNamespace(prim_path="/World/envs/env_.*/Robot")),
        )
    )
    fake_env.unwrapped._robot.data = SimpleNamespace(
        body_com_pos_b=torch.tensor([[[0.05, 0.01, -0.02]]], dtype=torch.float32)
    )

    sensor = module._create_isaacsim_imu_sensor(FakeProvider(), fake_env, env_step_dt=0.02)
    assert sensor is not None
    assert created["initialize_impl"] == 1
    assert sensor._is_initialized is True
    assert created["reset"] == 1
    assert created["update"] == 1


def test_fly_loiter_create_isaacsim_imu_sensor_returns_none_for_non_isaacsim_backend(monkeypatch) -> None:
    module = _load_script_module(
        "test_fly_loiter_create_sensor_module",
        "scripts/flapping_px4/fly_loiter.py",
        monkeypatch,
    )
    fake_env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _robot=SimpleNamespace(body_names=["base_link"]),
            _base_body_ids=[0],
            cfg=SimpleNamespace(robot=SimpleNamespace(prim_path="/World/envs/env_.*/Robot")),
        )
    )
    fake_env.unwrapped._robot.data = SimpleNamespace(
        body_com_pos_b=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    )

    class FakeProvider:
        backend_name = "synthetic"

    assert module._create_isaacsim_imu_sensor(FakeProvider(), fake_env, env_step_dt=0.02) is None
