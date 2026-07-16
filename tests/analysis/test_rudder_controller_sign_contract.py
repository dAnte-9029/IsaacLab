from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from flapping_bot.analysis.tail_unit_audit import TailAuditSettings, build_production_tail_cfg
from flapping_bot.physics.tail_aero import TailAeroModel
from flapping_bot.px4_like.loiter_controller import PX4LikeLoiterController, PX4LikeLoiterControllerCfg
from flapping_bot.px4_like.path_tracking_controller import (
    PX4LikePathTrackingController,
    PX4LikePathTrackingControllerCfg,
)
from flapping_bot.px4_like.straight_line_controller import (
    PX4LikeStraightLineController,
    PX4LikeStraightLineControllerCfg,
)


ControllerRunner = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _common_state(yaw_rad: torch.Tensor, yaw_rate_rad_s: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = yaw_rad.shape[0]
    angular_rate = torch.zeros((batch, 3), dtype=torch.float32)
    angular_rate[:, 2] = yaw_rate_rad_s
    return {
        "pos_local": torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32).expand(batch, 3).clone(),
        "ground_vel_local": torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float32).expand(batch, 3).clone(),
        "wind_vel_local": torch.zeros((batch, 2), dtype=torch.float32),
        "roll": torch.zeros(batch, dtype=torch.float32),
        "pitch": torch.zeros(batch, dtype=torch.float32),
        "yaw": yaw_rad,
        "ang_vel_body": angular_rate,
    }


def _run_straight(yaw_rad: torch.Tensor, yaw_rate_rad_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    controller = PX4LikeStraightLineController(
        PX4LikeStraightLineControllerCfg(enable_tecs=False), device=torch.device("cpu")
    )
    actions, diag = controller.compute_actions(**_common_state(yaw_rad, yaw_rate_rad_s))
    return actions[:, 1], diag["course_err"]


def _run_loiter(yaw_rad: torch.Tensor, yaw_rate_rad_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    controller = PX4LikeLoiterController(
        PX4LikeLoiterControllerCfg(
            enable_tecs=False,
            circle_center_xy=(-40.0, 0.0),
            loiter_radius_m=40.0,
        ),
        device=torch.device("cpu"),
    )
    actions, diag = controller.compute_actions(**_common_state(yaw_rad, yaw_rate_rad_s))
    return actions[:, 1], diag["course_err"]


def _run_path(yaw_rad: torch.Tensor, yaw_rate_rad_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    controller = PX4LikePathTrackingController(
        PX4LikePathTrackingControllerCfg(enable_tecs=False), device=torch.device("cpu")
    )
    batch = yaw_rad.shape[0]
    path_query = {
        "closest_point_xyz": torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32)
        .expand(batch, 3)
        .clone(),
        "tangent_xy": torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(batch, 2).clone(),
        "curvature_m_inv": torch.zeros(batch, dtype=torch.float32),
        "height_sp_m": torch.full((batch,), 10.0, dtype=torch.float32),
    }
    actions, diag = controller.compute_actions_from_query(
        path_query=path_query,
        **_common_state(yaw_rad, yaw_rate_rad_s),
    )
    return actions[:, 1], diag["course_err"]


@pytest.fixture(params=[_run_straight, _run_loiter, _run_path], ids=["straight", "loiter", "path"])
def controller_runner(request: pytest.FixtureRequest) -> ControllerRunner:
    return request.param


def _tail_yaw_moment_increment(action_rudder: torch.Tensor) -> torch.Tensor:
    settings = TailAuditSettings()
    model = TailAeroModel(build_production_tail_cfg(settings), "cpu")
    batch = action_rudder.shape[0]
    velocity = torch.zeros((batch + 1, 3), dtype=torch.float64)
    velocity[:, 0] = settings.nominal_airspeed_mps
    rates = torch.zeros_like(velocity)
    com = torch.tensor(settings.base_com_pos_b_m, dtype=torch.float64).view(1, 3).expand(batch + 1, 3)
    zero = torch.zeros(batch + 1, dtype=torch.float64)
    rudder_rad = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            math.radians(settings.rudder_limit_deg) * action_rudder.to(dtype=torch.float64),
        )
    )
    _, moment = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=zero,
        right_elevon_rad=zero,
        rudder_rad=rudder_rad,
        base_com_pos_b=com,
    )
    return moment[1:, 2] - moment[0, 2]


def test_current_proportional_rudder_chain_should_reduce_alignment_error(
    controller_runner: ControllerRunner,
) -> None:
    """Desired contract: the rudder increment must oppose yaw-minus-air-heading error."""

    yaw = torch.deg2rad(torch.tensor([-5.0, 5.0], dtype=torch.float32))
    yaw_rate = torch.zeros(2, dtype=torch.float32)
    action_rudder, course_error = controller_runner(yaw, yaw_rate)
    delta_mz = _tail_yaw_moment_increment(action_rudder)

    assert torch.all(course_error.to(dtype=torch.float64) * delta_mz < 0.0), (
        f"regenerative proportional chain: course_error={course_error.tolist()}, "
        f"action_rudder={action_rudder.tolist()}, delta_mz_Nm={delta_mz.tolist()}"
    )


def test_current_yaw_rate_rudder_chain_is_damping(controller_runner: ControllerRunner) -> None:
    """The existing derivative term should remain opposing for both yaw-rate signs."""

    yaw = torch.zeros(2, dtype=torch.float32)
    yaw_rate = torch.tensor([-0.2, 0.2], dtype=torch.float32)
    action_rudder, course_error = controller_runner(yaw, yaw_rate)
    delta_mz = _tail_yaw_moment_increment(action_rudder)

    torch.testing.assert_close(course_error, torch.zeros_like(course_error), atol=1.0e-7, rtol=0.0)
    assert torch.all(yaw_rate.to(dtype=torch.float64) * delta_mz < 0.0), (
        f"non-damping derivative chain: yaw_rate={yaw_rate.tolist()}, "
        f"action_rudder={action_rudder.tolist()}, delta_mz_Nm={delta_mz.tolist()}"
    )
