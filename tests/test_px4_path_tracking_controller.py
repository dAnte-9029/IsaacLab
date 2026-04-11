import math

import torch

from flapping_bot.px4_like.path_tracking_controller import (
    PX4LikePathTrackingController,
    PX4LikePathTrackingControllerCfg,
)


def _path_query() -> dict[str, torch.Tensor]:
    return {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "curvature_m_inv": torch.zeros((1,)),
        "height_sp_m": torch.tensor([10.0]),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }


def _path_tracking_pitch_measurement_ripple_std(cfg: PX4LikePathTrackingControllerCfg) -> float:
    controller = PX4LikePathTrackingController(cfg, device=torch.device("cpu"))
    dt = float(cfg.control_dt_s)
    samples: list[float] = []

    for step in range(360):
        t = step * dt
        pitch = -math.radians(12.0) + math.radians(4.0) * math.sin(2.0 * math.pi * 5.0 * t)
        pitch_rate = math.radians(4.0) * 2.0 * math.pi * 5.0 * math.cos(2.0 * math.pi * 5.0 * t)
        _, diag = controller.compute_actions_from_query(
            path_query=_path_query(),
            pos_local=torch.tensor([[8.0 * t, 0.0, 10.0]], dtype=torch.float32),
            ground_vel_local=torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float32),
            wind_vel_local=torch.zeros((1, 2), dtype=torch.float32),
            roll=torch.tensor([0.0], dtype=torch.float32),
            pitch=torch.tensor([pitch], dtype=torch.float32),
            yaw=torch.tensor([0.0], dtype=torch.float32),
            ang_vel_body=torch.tensor([[0.0, pitch_rate, 0.0]], dtype=torch.float32),
        )
        if step >= 120:
            samples.append(float(diag["pitch_meas_filt"][0]))

    mean = sum(samples) / len(samples)
    return math.sqrt(sum((sample - mean) ** 2 for sample in samples) / len(samples))


def test_path_tracking_inner_pitch_cycle_mean_filter_rejects_flap_period_ripple() -> None:
    fast_cfg = PX4LikePathTrackingControllerCfg(
        enable_tecs=False,
        inner_pitch_lpf_tau_s=0.02,
        inner_pitch_rate_lpf_tau_s=0.02,
        inner_pitch_cycle_mean_enabled=False,
        inner_pitch_ki=0.0,
    )
    cycle_mean_cfg = PX4LikePathTrackingControllerCfg(
        enable_tecs=False,
        inner_pitch_lpf_tau_s=0.02,
        inner_pitch_rate_lpf_tau_s=0.02,
        inner_pitch_cycle_mean_enabled=True,
        inner_pitch_cycle_mean_tau_s=0.30,
        inner_pitch_rate_cycle_mean_tau_s=0.24,
        inner_pitch_ki=0.0,
    )

    fast_std = _path_tracking_pitch_measurement_ripple_std(fast_cfg)
    cycle_mean_std = _path_tracking_pitch_measurement_ripple_std(cycle_mean_cfg)

    assert cycle_mean_std < 0.35 * fast_std


def test_generic_controller_returns_four_actions():
    controller = PX4LikePathTrackingController(PX4LikePathTrackingControllerCfg(), device=torch.device("cpu"))
    query = {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "curvature_m_inv": torch.zeros((1,)),
        "height_sp_m": torch.tensor([10.0]),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }
    actions, diag = controller.compute_actions_from_query(
        path_query=query,
        pos_local=torch.zeros((1, 3)),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.zeros((1,)),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )
    assert actions.shape == (1, 4)
    assert "course_sp" in diag


def test_generic_controller_uses_query_height_reference():
    controller = PX4LikePathTrackingController(PX4LikePathTrackingControllerCfg(), device=torch.device("cpu"))
    base_query = {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "curvature_m_inv": torch.zeros((1,)),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }
    actions_low, diag_low = controller.compute_actions_from_query(
        path_query={**base_query, "height_sp_m": torch.tensor([0.0])},
        pos_local=torch.zeros((1, 3)),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.zeros((1,)),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )
    actions_high, diag_high = controller.compute_actions_from_query(
        path_query={**base_query, "height_sp_m": torch.tensor([15.0])},
        pos_local=torch.zeros((1, 3)),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.zeros((1,)),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )
    assert torch.allclose(diag_low["height_sp_m"], torch.tensor([0.0]))
    assert torch.allclose(diag_high["height_sp_m"], torch.tensor([15.0]))
    assert not torch.allclose(actions_low, actions_high)


def test_generic_controller_can_start_from_elevon_trim_action():
    controller = PX4LikePathTrackingController(
        PX4LikePathTrackingControllerCfg(initial_elevon_pitch_action=-0.4),
        device=torch.device("cpu"),
    )
    query = {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "curvature_m_inv": torch.zeros((1,)),
        "height_sp_m": torch.tensor([10.0]),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }
    actions, _ = controller.compute_actions_from_query(
        path_query=query,
        pos_local=torch.tensor([[0.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )
    assert torch.allclose(actions[:, 2], torch.tensor([-0.4]), atol=1.0e-3)


def test_generic_controller_increases_tecs_throttle_for_curved_path_load_factor() -> None:
    cfg = PX4LikePathTrackingControllerCfg(
        use_tecs_load_factor_compensation=True,
        tecs_roll_throttle_compensation=0.5,
        tecs_load_factor_clamp_max=2.0,
        tecs_load_factor_use_roll_sp=True,
    )
    straight_controller = PX4LikePathTrackingController(cfg, device=torch.device("cpu"))
    curved_controller = PX4LikePathTrackingController(cfg, device=torch.device("cpu"))
    base_query = {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "height_sp_m": torch.tensor([10.0]),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }

    _, diag_straight = straight_controller.compute_actions_from_query(
        path_query={**base_query, "curvature_m_inv": torch.zeros((1,))},
        pos_local=torch.tensor([[0.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )
    _, diag_curved = curved_controller.compute_actions_from_query(
        path_query={**base_query, "curvature_m_inv": torch.tensor([0.08])},
        pos_local=torch.tensor([[0.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )

    assert float(diag_curved["tecs_load_factor"][0]) > 1.0
    assert float(diag_curved["tecs_load_factor_energy_bias"][0]) >= 0.0
    assert float(diag_curved["tecs_throttle_sp"][0]) >= float(diag_straight["tecs_throttle_sp"][0])


def test_generic_controller_raises_tecs_speed_target_for_curved_path_when_bank_aware_speed_enabled() -> None:
    cfg = PX4LikePathTrackingControllerCfg(
        use_tecs_load_factor_compensation=True,
        tecs_roll_throttle_compensation=30.0,
        tecs_load_factor_clamp_max=2.0,
        tecs_load_factor_use_roll_sp=True,
        use_tecs_bank_aware_speed_sp=True,
        tecs_bank_aware_speed_scale=1.0,
        tecs_bank_aware_speed_clamp_mps=2.0,
        speed_sp_mps=7.0,
    )
    straight_controller = PX4LikePathTrackingController(cfg, device=torch.device("cpu"))
    curved_controller = PX4LikePathTrackingController(cfg, device=torch.device("cpu"))
    base_query = {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "height_sp_m": torch.tensor([10.0]),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }

    _, diag_straight = straight_controller.compute_actions_from_query(
        path_query={**base_query, "curvature_m_inv": torch.zeros((1,))},
        pos_local=torch.tensor([[0.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )
    _, diag_curved = curved_controller.compute_actions_from_query(
        path_query={**base_query, "curvature_m_inv": torch.tensor([0.08])},
        pos_local=torch.tensor([[0.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )

    assert float(diag_curved["tecs_load_factor"][0]) > 1.0
    assert float(diag_curved["tecs_tas_sp"][0]) > float(diag_straight["tecs_tas_sp"][0])


def test_generic_controller_enforces_bank_aware_min_airspeed_floor_for_curved_path() -> None:
    cfg = PX4LikePathTrackingControllerCfg(
        use_tecs_load_factor_compensation=True,
        tecs_roll_throttle_compensation=30.0,
        tecs_load_factor_clamp_max=2.0,
        tecs_load_factor_use_roll_sp=True,
        use_tecs_bank_aware_speed_sp=False,
        use_tecs_bank_aware_min_airspeed=True,
        tecs_bank_aware_min_airspeed_mps=8.0,
        tecs_bank_aware_min_airspeed_scale=1.0,
        tecs_bank_aware_min_airspeed_clamp_mps=2.0,
        speed_sp_mps=7.0,
    )
    straight_controller = PX4LikePathTrackingController(cfg, device=torch.device("cpu"))
    curved_controller = PX4LikePathTrackingController(cfg, device=torch.device("cpu"))
    base_query = {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "height_sp_m": torch.tensor([10.0]),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }

    _, diag_straight = straight_controller.compute_actions_from_query(
        path_query={**base_query, "curvature_m_inv": torch.zeros((1,))},
        pos_local=torch.tensor([[0.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )
    _, diag_curved = curved_controller.compute_actions_from_query(
        path_query={**base_query, "curvature_m_inv": torch.tensor([0.08])},
        pos_local=torch.tensor([[0.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )

    assert float(diag_straight["tecs_tas_sp"][0]) == 8.0
    assert float(diag_curved["tecs_tas_sp"][0]) > float(diag_straight["tecs_tas_sp"][0])
    assert float(diag_curved["tecs_bank_aware_min_airspeed_mps"][0]) > 8.0


def test_path_tracking_controller_passes_capture_pitch_tuning_to_tecs() -> None:
    controller = PX4LikePathTrackingController(
        PX4LikePathTrackingControllerCfg(
            tecs_pitch_sp_filter_tau_capture_s=0.05,
            tecs_pitch_sp_rate_limit_capture_deg_s=60.0,
        ),
        device=torch.device("cpu"),
    )

    assert controller._tecs.cfg.pitch_sp_filter_tau_capture_s == 0.05
    assert controller._tecs.cfg.pitch_sp_rate_limit_capture_deg_s == 60.0


def test_generic_controller_pitch_rate_shaping_makes_initial_recovery_more_aggressive() -> None:
    slow_controller = PX4LikePathTrackingController(
        PX4LikePathTrackingControllerCfg(
            inner_pitch_tc_s=2.0,
            inner_pitch_rate_max_deg_s=20.0,
            inner_elevon_pitch_rate_limit_per_s=0.0,
        ),
        device=torch.device("cpu"),
    )
    fast_controller = PX4LikePathTrackingController(
        PX4LikePathTrackingControllerCfg(
            inner_pitch_tc_s=0.3,
            inner_pitch_rate_max_deg_s=120.0,
            inner_elevon_pitch_rate_limit_per_s=0.0,
        ),
        device=torch.device("cpu"),
    )
    query = {
        "closest_point_xyz": torch.zeros((1, 3)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "curvature_m_inv": torch.tensor([0.08]),
        "height_sp_m": torch.tensor([10.0]),
        "progress_s": torch.zeros((1,)),
        "preview_points_xyz": torch.zeros((1, 5, 3)),
    }
    kwargs = dict(
        path_query=query,
        pos_local=torch.tensor([[0.0, 0.0, 6.0]]),
        ground_vel_local=torch.tensor([[7.0, 0.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.zeros((1,)),
        yaw=torch.zeros((1,)),
        ang_vel_body=torch.zeros((1, 3)),
    )

    slow_actions, slow_diag = slow_controller.compute_actions_from_query(**kwargs)
    fast_actions, fast_diag = fast_controller.compute_actions_from_query(**kwargs)

    assert float(fast_diag["pitch_rate_sp"][0]) < float(slow_diag["pitch_rate_sp"][0])
    assert float(fast_actions[0, 2]) < float(slow_actions[0, 2])
