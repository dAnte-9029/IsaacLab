from __future__ import annotations

import pytest
import torch

from flapping_bot.physics import build_prescribed_dof_position_limits


def test_prescribed_limits_replace_only_selected_joint() -> None:
    nominal = torch.tensor(
        [
            [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]],
            [[-1.1, 1.1], [-2.1, 2.1], [-3.1, 3.1]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([0.25, -0.4], dtype=torch.float64)

    result = build_prescribed_dof_position_limits(
        nominal_limits_rad=nominal,
        target_position_rad=target,
        joint_id=1,
    )

    torch.testing.assert_close(result[:, 0], nominal[:, 0])
    torch.testing.assert_close(result[:, 2], nominal[:, 2])
    torch.testing.assert_close(result[:, 1, 0], target)
    torch.testing.assert_close(result[:, 1, 1], target)
    torch.testing.assert_close(nominal[:, 1], torch.tensor([[-2.0, 2.0], [-2.1, 2.1]], dtype=torch.float64))


def test_prescribed_limit_band_is_explicit_and_validated() -> None:
    nominal = torch.tensor([[[-1.0, 1.0]]])
    target = torch.tensor([0.2])

    result = build_prescribed_dof_position_limits(
        nominal_limits_rad=nominal,
        target_position_rad=target,
        joint_id=0,
        half_width_rad=0.01,
    )

    torch.testing.assert_close(result, torch.tensor([[[0.19, 0.21]]]))
    with pytest.raises(ValueError, match="nonnegative"):
        build_prescribed_dof_position_limits(
            nominal_limits_rad=nominal,
            target_position_rad=target,
            joint_id=0,
            half_width_rad=-0.01,
        )
