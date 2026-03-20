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
