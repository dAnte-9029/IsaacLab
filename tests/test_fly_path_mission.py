from types import SimpleNamespace

import pytest
import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import _compute_path_episode_length_s
from flapping_bot.path_tracking.path_manager import PathManager, PathManagerCfg
from scripts.flapping_px4.fly_path_mission import (
    _inject_mission,
    build_phase_mission,
    build_random_mission,
    ensure_episode_horizon,
    finalize_rollout_metrics,
    resolve_rollout_status,
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
    expected_horizon_steps = max(int(expected_episode_length_s / env.step_dt), 1)

    assert int(env._path_episode_horizon_steps[0].item()) == expected_horizon_steps
    assert env._teacher_controller.reset_calls == 1
