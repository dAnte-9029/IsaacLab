import torch

from flapping_bot.px4_like.path_tracking_controller import (
    PX4LikePathTrackingController,
    PX4LikePathTrackingControllerCfg,
)


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
