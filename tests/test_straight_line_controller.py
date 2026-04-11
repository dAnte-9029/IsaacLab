from __future__ import annotations

import math

import torch

from flapping_bot.px4_like.straight_line_controller import (
    PX4LikeStraightLineController,
    PX4LikeStraightLineControllerCfg,
)


def _pitch_measurement_ripple_std(cfg: PX4LikeStraightLineControllerCfg) -> float:
    controller = PX4LikeStraightLineController(cfg, device=torch.device("cpu"))
    dt = float(cfg.control_dt_s)
    samples: list[float] = []

    for step in range(360):
        t = step * dt
        pitch = -math.radians(12.0) + math.radians(4.0) * math.sin(2.0 * math.pi * 5.0 * t)
        pitch_rate = math.radians(4.0) * 2.0 * math.pi * 5.0 * math.cos(2.0 * math.pi * 5.0 * t)
        _, diag = controller.compute_actions(
            pos_local=torch.tensor([[8.0 * t, 0.0, 10.0]], dtype=torch.float32),
            ground_vel_local=torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float32),
            wind_vel_local=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            roll=torch.tensor([0.0], dtype=torch.float32),
            pitch=torch.tensor([pitch], dtype=torch.float32),
            yaw=torch.tensor([0.0], dtype=torch.float32),
            ang_vel_body=torch.tensor([[0.0, pitch_rate, 0.0]], dtype=torch.float32),
        )
        if step >= 120:
            samples.append(float(diag["pitch_meas_filt"][0]))

    mean = sum(samples) / len(samples)
    return math.sqrt(sum((sample - mean) ** 2 for sample in samples) / len(samples))


def test_inner_pitch_cycle_mean_filter_rejects_flap_period_ripple() -> None:
    fast_cfg = PX4LikeStraightLineControllerCfg(
        enable_tecs=False,
        inner_pitch_lpf_tau_s=0.02,
        inner_pitch_rate_lpf_tau_s=0.02,
        inner_pitch_cycle_mean_enabled=False,
        inner_pitch_ki=0.0,
    )
    cycle_mean_cfg = PX4LikeStraightLineControllerCfg(
        enable_tecs=False,
        inner_pitch_lpf_tau_s=0.02,
        inner_pitch_rate_lpf_tau_s=0.02,
        inner_pitch_cycle_mean_enabled=True,
        inner_pitch_cycle_mean_tau_s=0.30,
        inner_pitch_rate_cycle_mean_tau_s=0.24,
        inner_pitch_ki=0.0,
    )

    fast_std = _pitch_measurement_ripple_std(fast_cfg)
    cycle_mean_std = _pitch_measurement_ripple_std(cycle_mean_cfg)

    assert cycle_mean_std < 0.35 * fast_std


def test_default_lateral_guidance_uses_retuned_path_tracking_gains() -> None:
    cfg = PX4LikeStraightLineControllerCfg()

    assert cfg.guidance_period_s == 2.2
    assert cfg.heading_p_gain == 1.8
    assert cfg.roll_kp == 4.5
    assert cfg.roll_kd == 0.85
