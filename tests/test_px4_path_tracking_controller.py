import torch

try:
    from flapping_bot.px4_like.path_tracking_controller import (
        PX4LikePathTrackingController,
        PX4LikePathTrackingControllerCfg,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility import path
    from flapping_bot.flapping_bot.px4_like.path_tracking_controller import (
        PX4LikePathTrackingController,
        PX4LikePathTrackingControllerCfg,
    )


def test_generic_controller_returns_four_actions():
    controller = PX4LikePathTrackingController(PX4LikePathTrackingControllerCfg(), device=torch.device("cpu"))
    query = {
        "closest_point_xy": torch.zeros((1, 2)),
        "tangent_xy": torch.tensor([[1.0, 0.0]]),
        "curvature": torch.zeros((1,)),
        "height_sp_m": torch.tensor([10.0]),
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
