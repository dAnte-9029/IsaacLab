import math

import torch

from flapping_bot.px4_like.loiter_controller import PX4LikeLoiterController, PX4LikeLoiterControllerCfg


def _loiter_pitch_measurement_ripple_std(cfg: PX4LikeLoiterControllerCfg) -> float:
    controller = PX4LikeLoiterController(cfg, device=torch.device("cpu"))
    dt = float(cfg.control_dt_s)
    samples: list[float] = []

    for step in range(360):
        t = step * dt
        pitch = -math.radians(12.0) + math.radians(4.0) * math.sin(2.0 * math.pi * 5.0 * t)
        pitch_rate = math.radians(4.0) * 2.0 * math.pi * 5.0 * math.cos(2.0 * math.pi * 5.0 * t)
        _, diag = controller.compute_actions(
            pos_local=torch.tensor([[20.0, 0.0, 10.0]], dtype=torch.float32),
            ground_vel_local=torch.tensor([[0.0, 8.0, 0.0]], dtype=torch.float32),
            wind_vel_local=torch.zeros((1, 2), dtype=torch.float32),
            roll=torch.tensor([0.0], dtype=torch.float32),
            pitch=torch.tensor([pitch], dtype=torch.float32),
            yaw=torch.tensor([math.pi / 2.0], dtype=torch.float32),
            ang_vel_body=torch.tensor([[0.0, pitch_rate, 0.0]], dtype=torch.float32),
        )
        if step >= 120:
            samples.append(float(diag["pitch_meas_filt"][0]))

    mean = sum(samples) / len(samples)
    return math.sqrt(sum((sample - mean) ** 2 for sample in samples) / len(samples))


def test_loiter_inner_pitch_cycle_mean_filter_rejects_flap_period_ripple() -> None:
    fast_cfg = PX4LikeLoiterControllerCfg(
        enable_tecs=False,
        inner_pitch_lpf_tau_s=0.02,
        inner_pitch_rate_lpf_tau_s=0.02,
        inner_pitch_cycle_mean_enabled=False,
        inner_pitch_ki=0.0,
    )
    cycle_mean_cfg = PX4LikeLoiterControllerCfg(
        enable_tecs=False,
        inner_pitch_lpf_tau_s=0.02,
        inner_pitch_rate_lpf_tau_s=0.02,
        inner_pitch_cycle_mean_enabled=True,
        inner_pitch_cycle_mean_tau_s=0.30,
        inner_pitch_rate_cycle_mean_tau_s=0.24,
        inner_pitch_ki=0.0,
    )

    fast_std = _loiter_pitch_measurement_ripple_std(fast_cfg)
    cycle_mean_std = _loiter_pitch_measurement_ripple_std(cycle_mean_cfg)

    assert cycle_mean_std < 0.35 * fast_std


def test_loiter_controller_passes_turn_load_factor_into_tecs() -> None:
    controller = PX4LikeLoiterController(
        PX4LikeLoiterControllerCfg(
            use_tecs_load_factor_compensation=True,
            tecs_roll_throttle_compensation=30.0,
            tecs_load_factor_clamp_max=2.0,
            tecs_load_factor_use_roll_sp=True,
            use_tecs_bank_aware_speed_sp=True,
            tecs_bank_aware_speed_scale=1.0,
            tecs_bank_aware_speed_clamp_mps=2.0,
            use_tecs_bank_aware_min_airspeed=True,
            tecs_bank_aware_min_airspeed_mps=8.0,
            tecs_bank_aware_min_airspeed_scale=1.0,
            tecs_bank_aware_min_airspeed_clamp_mps=2.0,
            load_factor_pitch_compensation_gain=0.75,
        ),
        device=torch.device("cpu"),
    )

    _, diag = controller.compute_actions(
        pos_local=torch.tensor([[20.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[0.0, 7.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.tensor([torch.pi / 2.0]),
        ang_vel_body=torch.zeros((1, 3)),
    )

    assert float(diag["roll_sp"][0]) != 0.0
    assert float(diag["tecs_load_factor"][0]) > 1.0
    assert float(diag["tecs_load_factor_pitch_bias"][0]) > 0.0


def test_loiter_controller_uses_actual_roll_for_tecs_when_roll_sp_compensation_disabled() -> None:
    controller = PX4LikeLoiterController(
        PX4LikeLoiterControllerCfg(
            use_tecs_load_factor_compensation=True,
            tecs_roll_throttle_compensation=30.0,
            tecs_load_factor_clamp_max=2.0,
            tecs_load_factor_use_roll_sp=False,
        ),
        device=torch.device("cpu"),
    )

    _, diag = controller.compute_actions(
        pos_local=torch.tensor([[20.0, 0.0, 10.0]]),
        ground_vel_local=torch.tensor([[0.0, 7.0, 0.0]]),
        wind_vel_local=torch.zeros((1, 2)),
        roll=torch.zeros((1,)),
        pitch=torch.tensor([-torch.deg2rad(torch.tensor(10.0)).item()]),
        yaw=torch.tensor([torch.pi / 2.0]),
        ang_vel_body=torch.zeros((1, 3)),
    )

    assert float(diag["roll_sp"][0]) != 0.0
    assert float(diag["tecs_load_factor"][0]) == 1.0
    assert float(diag["tecs_load_factor_energy_bias"][0]) == 0.0
