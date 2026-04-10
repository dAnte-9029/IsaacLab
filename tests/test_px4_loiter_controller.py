import torch

from flapping_bot.px4_like.loiter_controller import PX4LikeLoiterController, PX4LikeLoiterControllerCfg


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
