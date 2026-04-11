import ast
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import gymnasium as gym

from flapping_bot.direct.flapping_bot.path_tracking_env import _compute_path_episode_length_s
from flapping_bot.path_tracking.path_manager import PathManager, PathManagerCfg
from scripts.flapping_px4.fly_path_mission import (
    _configure_env,
    _inject_mission,
    build_path_mission_parser,
    build_phase_mission,
    build_random_mission,
    ensure_episode_horizon,
    finalize_rollout_metrics,
    prepend_task_warmup,
    resolve_rollout_status,
)


def _fly_path_mission_module_ast() -> ast.Module:
    return ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "flapping_px4"
            / "fly_path_mission.py"
        ).read_text()
    )


def test_build_phase_mission_supports_explicit_descent() -> None:
    mission = build_phase_mission("descent_straight")
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=40.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    assert manager.sample(manager.total_length_m)[2] < manager.sample(0.0)[2]


def test_build_phase_mission_multi_segment_contains_turning_and_straight() -> None:
    mission = build_phase_mission("multi_segment")
    kinds = [segment.kind for segment in mission.segments]
    assert "straight" in kinds
    assert "turn" in kinds
    assert "loiter" in kinds


def test_prepend_task_warmup_marks_leading_segment_as_non_scored() -> None:
    mission = prepend_task_warmup(build_phase_mission("level_loiter"), warmup_enabled=True)

    assert [segment.kind for segment in mission.segments[:2]] == ["straight", "loiter"]
    assert mission.segments[0].counts_toward_progress is False
    assert mission.segments[1].counts_toward_progress is True


def test_resolve_rollout_status_treats_near_complete_path_as_success() -> None:
    completed, stop_now, failure_kind = resolve_rollout_status(
        progress_ratio=0.999,
        terminated=True,
        truncated=False,
        completion_threshold=0.995,
    )
    assert completed is True
    assert stop_now is True
    assert failure_kind is None


def test_ensure_episode_horizon_extends_env_timeout() -> None:
    class DummyEnv:
        def __init__(self):
            self.unwrapped = self
            self.cfg = SimpleNamespace(episode_length_s=1.0)

        @property
        def max_episode_length(self) -> int:
            return int(self.cfg.episode_length_s / 0.01)

    env = DummyEnv()
    ensure_episode_horizon(env, effective_steps=250, env_step_dt=0.01)
    assert env.unwrapped.max_episode_length >= 251
    assert env.unwrapped.cfg.episode_length_s >= 2.5


def test_build_random_mission_is_deterministic_for_seed() -> None:
    mission_a = build_random_mission(
        seed=17,
        num_segments_min=2,
        num_segments_max=4,
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
        allow_climb_on_straight=True,
    )
    mission_b = build_random_mission(
        seed=17,
        num_segments_min=2,
        num_segments_max=4,
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
        allow_climb_on_straight=True,
    )
    assert mission_a == mission_b


def test_build_random_mission_respects_enabled_segment_types() -> None:
    mission = build_random_mission(
        seed=3,
        num_segments_min=3,
        num_segments_max=3,
        allow_straight=False,
        allow_turn=True,
        allow_loiter=False,
        allow_climb_on_straight=False,
    )
    assert len(mission.segments) == 3
    assert all(segment.kind == "turn" for segment in mission.segments)


def test_finalize_rollout_metrics_uses_completion_trigger_step_progress() -> None:
    final_progress_ratio, completed_path = finalize_rollout_metrics(
        progress_ratio_hist=[0.2, 0.4, 0.6],
        last_observed_progress_ratio=0.996,
        completed_path=False,
        failure_step=None,
        completion_threshold=0.995,
    )

    assert final_progress_ratio == pytest.approx(0.996)
    assert completed_path is True


def test_path_mission_logger_does_not_call_stateful_teacher_controller_directly() -> None:
    module = _fly_path_mission_module_ast()

    direct_teacher_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_compute_teacher_actions"
    ]

    assert direct_teacher_calls == []


def test_path_mission_parser_accepts_teacher_tecs_overrides() -> None:
    parser = build_path_mission_parser()
    args = parser.parse_args(
        [
            "--phase",
            "level_turn",
            "--teacher_pitch_kp",
            "2.8",
            "--teacher_inner_pitch_ki",
            "1.0",
            "--teacher_tecs_roll_throttle_compensation",
            "2.5",
            "--teacher_tecs_load_factor_clamp_max",
            "3.0",
            "--teacher_tecs_load_factor_pitch_compensation_gain",
            "0.4",
            "--teacher_tecs_bank_aware_speed_scale",
            "0.75",
            "--teacher_tecs_bank_aware_speed_clamp_mps",
            "1.5",
            "--teacher_tecs_bank_aware_min_airspeed_mps",
            "8.0",
            "--teacher_tecs_bank_aware_min_airspeed_scale",
            "1.2",
            "--teacher_tecs_bank_aware_min_airspeed_clamp_mps",
            "2.5",
            "--teacher_tecs_altitude_hold_error_band_m",
            "0.1",
            "--teacher_tecs_altitude_capture_error_m",
            "0.9",
            "--teacher_tecs_altitude_capture_time_const_s",
            "0.65",
            "--teacher_tecs_altitude_error_gain",
            "0.9",
            "--teacher_tecs_pitch_speed_weight",
            "0.55",
            "--teacher_tecs_pitch_speed_weight_capture",
            "0.3",
            "--teacher_tecs_capture_extra_climb_rate_mps",
            "1.0",
            "--teacher_tecs_capture_extra_sink_rate_mps",
            "0.25",
            "--teacher_tecs_pitch_damping_gain",
            "0.1",
            "--no-teacher_use_tecs_load_factor_compensation",
            "--no-teacher_tecs_load_factor_use_roll_sp",
            "--no-teacher_use_tecs_bank_aware_speed_sp",
            "--teacher_use_tecs_bank_aware_min_airspeed",
        ]
    )

    assert args.teacher_pitch_kp == pytest.approx(2.8)
    assert args.teacher_inner_pitch_ki == pytest.approx(1.0)
    assert args.teacher_tecs_roll_throttle_compensation == pytest.approx(2.5)
    assert args.teacher_tecs_load_factor_clamp_max == pytest.approx(3.0)
    assert args.teacher_tecs_load_factor_pitch_compensation_gain == pytest.approx(0.4)
    assert args.teacher_tecs_bank_aware_speed_scale == pytest.approx(0.75)
    assert args.teacher_tecs_bank_aware_speed_clamp_mps == pytest.approx(1.5)
    assert args.teacher_tecs_bank_aware_min_airspeed_mps == pytest.approx(8.0)
    assert args.teacher_tecs_bank_aware_min_airspeed_scale == pytest.approx(1.2)
    assert args.teacher_tecs_bank_aware_min_airspeed_clamp_mps == pytest.approx(2.5)
    assert args.teacher_tecs_altitude_hold_error_band_m == pytest.approx(0.1)
    assert args.teacher_tecs_altitude_capture_error_m == pytest.approx(0.9)
    assert args.teacher_tecs_altitude_capture_time_const_s == pytest.approx(0.65)
    assert args.teacher_tecs_altitude_error_gain == pytest.approx(0.9)
    assert args.teacher_tecs_pitch_speed_weight == pytest.approx(0.55)
    assert args.teacher_tecs_pitch_speed_weight_capture == pytest.approx(0.3)
    assert args.teacher_tecs_capture_extra_climb_rate_mps == pytest.approx(1.0)
    assert args.teacher_tecs_capture_extra_sink_rate_mps == pytest.approx(0.25)
    assert args.teacher_tecs_pitch_damping_gain == pytest.approx(0.1)
    assert args.teacher_use_tecs_load_factor_compensation is False
    assert args.teacher_tecs_load_factor_use_roll_sp is False
    assert args.teacher_use_tecs_bank_aware_speed_sp is False
    assert args.teacher_use_tecs_bank_aware_min_airspeed is True


def test_path_mission_parser_accepts_tail_aero_compatibility_overrides() -> None:
    parser = build_path_mission_parser()
    args = parser.parse_args(
        [
            "--phase",
            "level_turn",
            "--tail_horizontal_tail_incidence_bias_deg",
            "-3.5",
            "--tail_fixed_horizontal_effectiveness",
            "0.9",
            "--tail_elevon_effectiveness",
            "1.4",
            "--tail_elevon_alpha_limit_deg",
            "38.0",
            "--tail_horizontal_tail_q_scale",
            "1.15",
        ]
    )

    assert args.tail_horizontal_tail_incidence_bias_deg == pytest.approx(-3.5)
    assert args.tail_fixed_horizontal_effectiveness == pytest.approx(0.9)
    assert args.tail_elevon_effectiveness == pytest.approx(1.4)
    assert args.tail_elevon_alpha_limit_deg == pytest.approx(38.0)
    assert args.tail_horizontal_tail_q_scale == pytest.approx(1.15)


def test_path_mission_parser_accepts_base_body_com_override_x() -> None:
    parser = build_path_mission_parser()
    args = parser.parse_args(
        [
            "--phase",
            "level_turn",
            "--base_body_com_override_x_m",
            "-0.10",
        ]
    )

    assert args.base_body_com_override_x_m == pytest.approx(-0.10)


def test_path_mission_parser_defaults_base_body_com_override_to_none() -> None:
    parser = build_path_mission_parser()
    args = parser.parse_args(["--phase", "level_turn"])

    assert args.base_body_com_override_x_m is None


def test_path_mission_parser_accepts_reset_trim_overrides() -> None:
    parser = build_path_mission_parser()
    args = parser.parse_args(
        [
            "--phase",
            "level_straight",
            "--reset_pitch_deg",
            "6.5",
            "--reset_flap_hz",
            "3.2",
            "--reset_elevon_pitch_deg",
            "-14.0",
            "--reset_forward_speed_mps",
            "8.1",
        ]
    )

    assert args.reset_pitch_deg == pytest.approx(6.5)
    assert args.reset_flap_hz == pytest.approx(3.2)
    assert args.reset_elevon_pitch_deg == pytest.approx(-14.0)
    assert args.reset_forward_speed_mps == pytest.approx(8.1)


def test_path_mission_parser_accepts_warmup_overrides() -> None:
    parser = build_path_mission_parser()
    args = parser.parse_args(
        [
            "--phase",
            "level_turn",
            "--path_warmup_straight_length_m",
            "25.0",
            "--no-path_warmup_enabled",
        ]
    )

    assert args.path_warmup_straight_length_m == pytest.approx(25.0)
    assert args.path_warmup_enabled is False


def test_configure_env_applies_teacher_tecs_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    env_cfg = SimpleNamespace(
        randomize_commands=True,
        height_cmd=0.0,
        action_space=0,
        act_lpf_tau_s=0.1,
        act_rate_limit_per_s=2.0,
        sim=SimpleNamespace(dt=1.0 / 240.0),
        decimation=2,
        episode_length_s=18.0,
        teacher_guidance_enabled=False,
        teacher_guidance_delta_init=0.2,
        teacher_guidance_delta_final=2.0,
        teacher_guidance_schedule_steps=(0, 1),
        teacher_guidance_schedule_deltas=(0.1, 0.2),
        teacher_guidance_disable_after_steps=100,
        wind_enabled=False,
        randomize_wind=True,
        wind_xy_mps=(0.0, 0.0),
        wind_x_range_mps=(0.0, 0.0),
        wind_y_range_mps=(0.0, 0.0),
        wind_ou_enabled=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_xy_mps=(0.0, 0.0),
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        teacher_use_tecs_load_factor_compensation=False,
        teacher_tecs_roll_throttle_compensation=0.0,
        teacher_tecs_load_factor_clamp_max=2.0,
        teacher_tecs_load_factor_use_roll_sp=True,
        teacher_tecs_load_factor_pitch_compensation_gain=0.75,
        teacher_pitch_kp=2.5,
        teacher_inner_pitch_ki=0.9,
        teacher_use_tecs_bank_aware_speed_sp=False,
        teacher_tecs_bank_aware_speed_scale=0.0,
        teacher_tecs_bank_aware_speed_clamp_mps=0.0,
        teacher_use_tecs_bank_aware_min_airspeed=False,
        teacher_tecs_bank_aware_min_airspeed_mps=0.0,
        teacher_tecs_bank_aware_min_airspeed_scale=0.0,
        teacher_tecs_bank_aware_min_airspeed_clamp_mps=0.0,
        teacher_tecs_altitude_hold_error_band_m=0.25,
        teacher_tecs_altitude_capture_error_m=0.8,
        teacher_tecs_altitude_capture_time_const_s=1.0,
        teacher_tecs_altitude_error_gain=0.55,
        teacher_tecs_pitch_speed_weight=0.8,
        teacher_tecs_pitch_speed_weight_capture=0.35,
        teacher_tecs_capture_extra_climb_rate_mps=0.7,
        teacher_tecs_capture_extra_sink_rate_mps=0.2,
        teacher_tecs_pitch_damping_gain=0.08,
    )

    parse_cfg_mod = ModuleType("isaaclab_tasks.utils.parse_cfg")
    parse_cfg_mod.parse_env_cfg = lambda *args, **kwargs: env_cfg
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", ModuleType("isaaclab_tasks"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", ModuleType("isaaclab_tasks.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils.parse_cfg", parse_cfg_mod)

    captured: dict[str, object] = {}

    def _fake_make(task: str, cfg):
        captured["task"] = task
        captured["cfg"] = cfg
        return "dummy-env"

    monkeypatch.setattr(gym, "make", _fake_make)

    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        device="cpu",
        num_envs=1,
        height_sp=11.0,
        steps=2400,
        episode_length_s=None,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=32.0,
        path_manager_max_flight_path_angle_deg=9.0,
        teacher_use_tecs_load_factor_compensation=True,
        teacher_tecs_roll_throttle_compensation=2.5,
        teacher_tecs_load_factor_clamp_max=3.0,
        teacher_tecs_load_factor_use_roll_sp=False,
        teacher_tecs_load_factor_pitch_compensation_gain=0.4,
        teacher_pitch_kp=2.8,
        teacher_inner_pitch_ki=1.0,
        teacher_use_tecs_bank_aware_speed_sp=True,
        teacher_tecs_bank_aware_speed_scale=0.75,
        teacher_tecs_bank_aware_speed_clamp_mps=1.5,
        teacher_use_tecs_bank_aware_min_airspeed=True,
        teacher_tecs_bank_aware_min_airspeed_mps=8.0,
        teacher_tecs_bank_aware_min_airspeed_scale=1.2,
        teacher_tecs_bank_aware_min_airspeed_clamp_mps=2.5,
        teacher_tecs_altitude_hold_error_band_m=0.1,
        teacher_tecs_altitude_capture_error_m=0.9,
        teacher_tecs_altitude_capture_time_const_s=0.65,
        teacher_tecs_altitude_error_gain=0.9,
        teacher_tecs_pitch_speed_weight=0.55,
        teacher_tecs_pitch_speed_weight_capture=0.3,
        teacher_tecs_capture_extra_climb_rate_mps=1.0,
        teacher_tecs_capture_extra_sink_rate_mps=0.25,
        teacher_tecs_pitch_damping_gain=0.1,
    )

    env, configured_env_cfg, env_step_dt = _configure_env(args)

    assert env == "dummy-env"
    assert captured["task"] == args.task
    assert captured["cfg"] is configured_env_cfg
    assert configured_env_cfg.teacher_use_tecs_load_factor_compensation is True
    assert configured_env_cfg.teacher_tecs_roll_throttle_compensation == pytest.approx(2.5)
    assert configured_env_cfg.teacher_tecs_load_factor_clamp_max == pytest.approx(3.0)
    assert configured_env_cfg.teacher_tecs_load_factor_use_roll_sp is False
    assert configured_env_cfg.teacher_tecs_load_factor_pitch_compensation_gain == pytest.approx(0.4)
    assert configured_env_cfg.teacher_pitch_kp == pytest.approx(2.8)
    assert configured_env_cfg.teacher_inner_pitch_ki == pytest.approx(1.0)
    assert configured_env_cfg.teacher_use_tecs_bank_aware_speed_sp is True
    assert configured_env_cfg.teacher_tecs_bank_aware_speed_scale == pytest.approx(0.75)
    assert configured_env_cfg.teacher_tecs_bank_aware_speed_clamp_mps == pytest.approx(1.5)
    assert configured_env_cfg.teacher_use_tecs_bank_aware_min_airspeed is True
    assert configured_env_cfg.teacher_tecs_bank_aware_min_airspeed_mps == pytest.approx(8.0)
    assert configured_env_cfg.teacher_tecs_bank_aware_min_airspeed_scale == pytest.approx(1.2)
    assert configured_env_cfg.teacher_tecs_bank_aware_min_airspeed_clamp_mps == pytest.approx(2.5)
    assert configured_env_cfg.teacher_tecs_altitude_hold_error_band_m == pytest.approx(0.1)
    assert configured_env_cfg.teacher_tecs_altitude_capture_error_m == pytest.approx(0.9)
    assert configured_env_cfg.teacher_tecs_altitude_capture_time_const_s == pytest.approx(0.65)
    assert configured_env_cfg.teacher_tecs_altitude_error_gain == pytest.approx(0.9)
    assert configured_env_cfg.teacher_tecs_pitch_speed_weight == pytest.approx(0.55)
    assert configured_env_cfg.teacher_tecs_pitch_speed_weight_capture == pytest.approx(0.3)
    assert configured_env_cfg.teacher_tecs_capture_extra_climb_rate_mps == pytest.approx(1.0)
    assert configured_env_cfg.teacher_tecs_capture_extra_sink_rate_mps == pytest.approx(0.25)
    assert configured_env_cfg.teacher_tecs_pitch_damping_gain == pytest.approx(0.1)
    assert configured_env_cfg.path_manager_max_roll_deg == pytest.approx(32.0)
    assert configured_env_cfg.path_manager_max_flight_path_angle_deg == pytest.approx(9.0)
    assert env_step_dt == pytest.approx((1.0 / 240.0) * 2.0)


def test_configure_env_applies_tail_aero_compatibility_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    env_cfg = SimpleNamespace(
        randomize_commands=True,
        height_cmd=0.0,
        action_space=0,
        act_lpf_tau_s=0.1,
        act_rate_limit_per_s=2.0,
        sim=SimpleNamespace(dt=1.0 / 240.0),
        decimation=2,
        episode_length_s=18.0,
        teacher_guidance_enabled=False,
        teacher_guidance_delta_init=0.2,
        teacher_guidance_delta_final=2.0,
        teacher_guidance_schedule_steps=(0, 1),
        teacher_guidance_schedule_deltas=(0.1, 0.2),
        teacher_guidance_disable_after_steps=100,
        wind_enabled=False,
        randomize_wind=True,
        wind_xy_mps=(0.0, 0.0),
        wind_x_range_mps=(0.0, 0.0),
        wind_y_range_mps=(0.0, 0.0),
        wind_ou_enabled=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_xy_mps=(0.0, 0.0),
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        tail_horizontal_tail_incidence_bias_deg=0.0,
        tail_fixed_horizontal_effectiveness=1.0,
        tail_elevon_effectiveness=1.0,
        tail_elevon_alpha_limit_deg=25.0,
        tail_horizontal_tail_q_scale=1.0,
        base_body_com_override_x_m=None,
    )

    parse_cfg_mod = ModuleType("isaaclab_tasks.utils.parse_cfg")
    parse_cfg_mod.parse_env_cfg = lambda *args, **kwargs: env_cfg
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", ModuleType("isaaclab_tasks"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", ModuleType("isaaclab_tasks.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils.parse_cfg", parse_cfg_mod)

    captured: dict[str, object] = {}

    def _fake_make(task: str, cfg):
        captured["task"] = task
        captured["cfg"] = cfg
        return "dummy-env"

    monkeypatch.setattr(gym, "make", _fake_make)

    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        device="cpu",
        num_envs=1,
        height_sp=11.0,
        steps=2400,
        episode_length_s=None,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=32.0,
        path_manager_max_flight_path_angle_deg=9.0,
        teacher_use_tecs_load_factor_compensation=None,
        teacher_tecs_roll_throttle_compensation=None,
        teacher_tecs_load_factor_clamp_max=None,
        teacher_tecs_load_factor_use_roll_sp=None,
        teacher_tecs_load_factor_pitch_compensation_gain=None,
        teacher_pitch_kp=None,
        teacher_inner_pitch_ki=None,
        teacher_use_tecs_bank_aware_speed_sp=None,
        teacher_tecs_bank_aware_speed_scale=None,
        teacher_tecs_bank_aware_speed_clamp_mps=None,
        teacher_use_tecs_bank_aware_min_airspeed=None,
        teacher_tecs_bank_aware_min_airspeed_mps=None,
        teacher_tecs_bank_aware_min_airspeed_scale=None,
        teacher_tecs_bank_aware_min_airspeed_clamp_mps=None,
        tail_horizontal_tail_incidence_bias_deg=-3.5,
        tail_fixed_horizontal_effectiveness=0.9,
        tail_elevon_effectiveness=1.4,
        tail_elevon_alpha_limit_deg=38.0,
        tail_horizontal_tail_q_scale=1.15,
        base_body_com_override_x_m=-0.10,
    )

    env, configured_env_cfg, env_step_dt = _configure_env(args)

    assert env == "dummy-env"
    assert captured["task"] == args.task
    assert captured["cfg"] is configured_env_cfg
    assert configured_env_cfg.tail_horizontal_tail_incidence_bias_deg == pytest.approx(-3.5)
    assert configured_env_cfg.tail_fixed_horizontal_effectiveness == pytest.approx(0.9)
    assert configured_env_cfg.tail_elevon_effectiveness == pytest.approx(1.4)
    assert configured_env_cfg.tail_elevon_alpha_limit_deg == pytest.approx(38.0)
    assert configured_env_cfg.tail_horizontal_tail_q_scale == pytest.approx(1.15)
    assert configured_env_cfg.base_body_com_override_x_m == pytest.approx(-0.10)
    assert env_step_dt == pytest.approx((1.0 / 240.0) * 2.0)


def test_configure_env_preserves_default_base_body_com_without_cli_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_cfg = SimpleNamespace(
        randomize_commands=True,
        height_cmd=0.0,
        action_space=0,
        act_lpf_tau_s=0.1,
        act_rate_limit_per_s=2.0,
        sim=SimpleNamespace(dt=1.0 / 240.0),
        decimation=2,
        episode_length_s=18.0,
        teacher_guidance_enabled=False,
        teacher_guidance_delta_init=0.2,
        teacher_guidance_delta_final=2.0,
        teacher_guidance_schedule_steps=(0, 1),
        teacher_guidance_schedule_deltas=(0.1, 0.2),
        teacher_guidance_disable_after_steps=100,
        wind_enabled=False,
        randomize_wind=True,
        wind_xy_mps=(0.0, 0.0),
        wind_x_range_mps=(0.0, 0.0),
        wind_y_range_mps=(0.0, 0.0),
        wind_ou_enabled=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_xy_mps=(0.0, 0.0),
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        tail_horizontal_tail_incidence_bias_deg=0.0,
        tail_fixed_horizontal_effectiveness=1.0,
        tail_elevon_effectiveness=1.0,
        tail_elevon_alpha_limit_deg=25.0,
        tail_horizontal_tail_q_scale=1.0,
        base_body_com_override_x_m=-0.10,
    )

    parse_cfg_mod = ModuleType("isaaclab_tasks.utils.parse_cfg")
    parse_cfg_mod.parse_env_cfg = lambda *args, **kwargs: env_cfg
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", ModuleType("isaaclab_tasks"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", ModuleType("isaaclab_tasks.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils.parse_cfg", parse_cfg_mod)

    captured: dict[str, object] = {}

    def _fake_make(task: str, cfg):
        captured["task"] = task
        captured["cfg"] = cfg
        return "dummy-env"

    monkeypatch.setattr(gym, "make", _fake_make)

    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        device="cpu",
        num_envs=1,
        height_sp=11.0,
        steps=2400,
        episode_length_s=None,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=32.0,
        path_manager_max_flight_path_angle_deg=9.0,
        teacher_use_tecs_load_factor_compensation=None,
        teacher_tecs_roll_throttle_compensation=None,
        teacher_tecs_load_factor_clamp_max=None,
        teacher_tecs_load_factor_use_roll_sp=None,
        teacher_tecs_load_factor_pitch_compensation_gain=None,
        teacher_pitch_kp=None,
        teacher_inner_pitch_ki=None,
        teacher_use_tecs_bank_aware_speed_sp=None,
        teacher_tecs_bank_aware_speed_scale=None,
        teacher_tecs_bank_aware_speed_clamp_mps=None,
        teacher_use_tecs_bank_aware_min_airspeed=None,
        teacher_tecs_bank_aware_min_airspeed_mps=None,
        teacher_tecs_bank_aware_min_airspeed_scale=None,
        teacher_tecs_bank_aware_min_airspeed_clamp_mps=None,
        tail_horizontal_tail_incidence_bias_deg=None,
        tail_fixed_horizontal_effectiveness=None,
        tail_elevon_effectiveness=None,
        tail_elevon_alpha_limit_deg=None,
        tail_horizontal_tail_q_scale=None,
        base_body_com_override_x_m=None,
    )

    env, configured_env_cfg, env_step_dt = _configure_env(args)

    assert env == "dummy-env"
    assert captured["task"] == args.task
    assert captured["cfg"] is configured_env_cfg
    assert configured_env_cfg.base_body_com_override_x_m == pytest.approx(-0.10)
    assert env_step_dt == pytest.approx((1.0 / 240.0) * 2.0)


def test_configure_env_applies_reset_trim_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    env_cfg = SimpleNamespace(
        randomize_commands=True,
        height_cmd=0.0,
        action_space=0,
        act_lpf_tau_s=0.1,
        act_rate_limit_per_s=2.0,
        sim=SimpleNamespace(dt=1.0 / 240.0),
        decimation=2,
        episode_length_s=18.0,
        teacher_guidance_enabled=False,
        teacher_guidance_delta_init=0.2,
        teacher_guidance_delta_final=2.0,
        teacher_guidance_schedule_steps=(0, 1),
        teacher_guidance_schedule_deltas=(0.1, 0.2),
        teacher_guidance_disable_after_steps=100,
        wind_enabled=False,
        randomize_wind=True,
        wind_xy_mps=(0.0, 0.0),
        wind_x_range_mps=(0.0, 0.0),
        wind_y_range_mps=(0.0, 0.0),
        wind_ou_enabled=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_xy_mps=(0.0, 0.0),
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        reset_pitch_deg=10.0,
        reset_flap_hz=2.5,
        reset_elevon_pitch_deg=-10.0,
        reset_forward_speed_mps=None,
    )

    parse_cfg_mod = ModuleType("isaaclab_tasks.utils.parse_cfg")
    parse_cfg_mod.parse_env_cfg = lambda *args, **kwargs: env_cfg
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", ModuleType("isaaclab_tasks"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", ModuleType("isaaclab_tasks.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils.parse_cfg", parse_cfg_mod)

    captured: dict[str, object] = {}

    def _fake_make(task: str, cfg):
        captured["task"] = task
        captured["cfg"] = cfg
        return "dummy-env"

    monkeypatch.setattr(gym, "make", _fake_make)

    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        device="cpu",
        num_envs=1,
        height_sp=10.0,
        steps=2200,
        episode_length_s=None,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        teacher_use_tecs_load_factor_compensation=None,
        teacher_tecs_roll_throttle_compensation=None,
        teacher_tecs_load_factor_clamp_max=None,
        teacher_tecs_load_factor_use_roll_sp=None,
        teacher_tecs_load_factor_pitch_compensation_gain=None,
        teacher_pitch_kp=None,
        teacher_inner_pitch_ki=None,
        teacher_use_tecs_bank_aware_speed_sp=None,
        teacher_tecs_bank_aware_speed_scale=None,
        teacher_tecs_bank_aware_speed_clamp_mps=None,
        teacher_use_tecs_bank_aware_min_airspeed=None,
        teacher_tecs_bank_aware_min_airspeed_mps=None,
        teacher_tecs_bank_aware_min_airspeed_scale=None,
        teacher_tecs_bank_aware_min_airspeed_clamp_mps=None,
        tail_horizontal_tail_incidence_bias_deg=None,
        tail_fixed_horizontal_effectiveness=None,
        tail_elevon_effectiveness=None,
        tail_elevon_alpha_limit_deg=None,
        tail_horizontal_tail_q_scale=None,
        base_body_com_override_x_m=None,
        reset_pitch_deg=6.5,
        reset_flap_hz=3.2,
        reset_elevon_pitch_deg=-14.0,
        reset_forward_speed_mps=8.1,
    )

    env, configured_env_cfg, env_step_dt = _configure_env(args)

    assert env == "dummy-env"
    assert captured["task"] == args.task
    assert captured["cfg"] is configured_env_cfg
    assert configured_env_cfg.reset_pitch_deg == pytest.approx(6.5)
    assert configured_env_cfg.reset_flap_hz == pytest.approx(3.2)
    assert configured_env_cfg.reset_elevon_pitch_deg == pytest.approx(-14.0)
    assert configured_env_cfg.reset_forward_speed_mps == pytest.approx(8.1)
    assert env_step_dt == pytest.approx((1.0 / 240.0) * 2.0)


def test_inject_mission_recomputes_path_episode_horizon_for_replaced_mission() -> None:
    class DummyTeacherController:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    class DummyEnv:
        def __init__(self) -> None:
            self.unwrapped = self
            self.num_envs = 1
            self.device = torch.device("cpu")
            self.step_dt = 1.0 / 120.0
            self.cfg = SimpleNamespace(
                episode_length_s=18.0,
                freeze_steps_after_reset=240,
                sim=SimpleNamespace(dt=1.0 / 240.0),
                path_episode_speed_ref_mps=6.0,
                path_episode_completion_margin_s=3.0,
                path_episode_max_s=36.0,
            )
            self._height_cmd = torch.tensor([10.0], dtype=torch.float32)
            self._missions = [None]
            self._path_managers = [None]
            self._path_progress_prev_s = torch.zeros(1, dtype=torch.float32)
            self._path_delta_s = torch.zeros(1, dtype=torch.float32)
            self._path_progress_s = torch.zeros(1, dtype=torch.float32)
            self._path_lateral_error_m = torch.zeros(1, dtype=torch.float32)
            self._path_height_error_m = torch.zeros(1, dtype=torch.float32)
            self._path_align_error_rad = torch.zeros(1, dtype=torch.float32)
            self._path_curvature_m_inv = torch.zeros(1, dtype=torch.float32)
            self._path_height_sp_m = torch.zeros(1, dtype=torch.float32)
            self._path_closest_point_xyz = torch.zeros((1, 3), dtype=torch.float32)
            self._path_tangent_xy = torch.zeros((1, 2), dtype=torch.float32)
            self._path_preview_points_xyz = torch.zeros((1, 5, 3), dtype=torch.float32)
            self._path_preview_points_body_xyz = torch.zeros((1, 5, 3), dtype=torch.float32)
            self._path_action_delta = torch.zeros((1, 4), dtype=torch.float32)
            self._path_episode_horizon_steps = torch.tensor([1111], dtype=torch.long)
            self._path_query_dirty = False
            self._teacher_controller = DummyTeacherController()

        def _ensure_path_buffers(self) -> None:
            return

        def _refresh_path_state(self) -> None:
            return

    env = DummyEnv()
    args = SimpleNamespace(
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        straight_length_m=60.0,
        turn_radius_m=20.0,
        loiter_radius_m=20.0,
        turn_sweep_deg=90.0,
        loiter_turns=1.0,
        climb_delta_m=3.0,
        height_sp=10.0,
    )
    mission = build_phase_mission("level_straight")

    manager = _inject_mission(env, args, mission)

    expected_episode_length_s = _compute_path_episode_length_s(
        current_episode_length_s=float(env.cfg.episode_length_s),
        path_total_length_m=float(manager.total_length_m),
        speed_ref_mps=float(env.cfg.path_episode_speed_ref_mps),
        freeze_steps=int(env.cfg.freeze_steps_after_reset),
        sim_dt=float(env.cfg.sim.dt),
        completion_margin_s=float(env.cfg.path_episode_completion_margin_s),
        max_episode_length_s=float(env.cfg.path_episode_max_s),
    )
    expected_horizon_steps = max(int(math.ceil(expected_episode_length_s / env.step_dt)), 1)

    assert int(env._path_episode_horizon_steps[0].item()) == expected_horizon_steps
    assert env._teacher_controller.reset_calls == 1


def test_fly_path_mission_trajectory_rows_include_longitudinal_diagnostics() -> None:
    module = _fly_path_mission_module_ast()

    required_keys = {
        "tecs_pitch_sp_deg",
        "action_elevon_pitch",
        "pitch_deg",
        "tecs_tas",
        "tecs_tas_sp",
    }

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "traj_rows":
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        found_keys = {
            key.value
            for key in node.args[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert required_keys.issubset(found_keys)
        return

    raise AssertionError("traj_rows.append({...}) not found")


def test_fly_path_mission_summary_records_tail_aero_compatibility_and_failure_audit_fields() -> None:
    module = _fly_path_mission_module_ast()

    required_keys = {
        "tail_horizontal_tail_incidence_bias_deg",
        "tail_fixed_horizontal_effectiveness",
        "tail_elevon_effectiveness",
        "tail_elevon_alpha_limit_deg",
        "tail_horizontal_tail_q_scale",
        "base_body_com_override_x_m",
        "runtime_base_body_com_x_m",
        "failure_kind",
        "steps_completed",
    }

    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "summary":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        found_keys = {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert required_keys.issubset(found_keys)
        return

    raise AssertionError("summary = {...} not found")
