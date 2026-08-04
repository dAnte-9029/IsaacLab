from __future__ import annotations

import pytest
import torch

from flapping_bot.physics import (
    ACTUAL_JOINT_ACCELERATION,
    ACTUAL_MOTION_BASE_EQUIVALENT,
    ACTUAL_PER_WING_LINK,
    COMMANDED_BASE_EQUIVALENT,
    FULL_WING_LINK_WRENCH,
    PRESCRIBED_ACCELERATION,
    PRESCRIBED_PER_WING_LINK,
    SINUSOIDAL_PHASE_PER_WING_LINK,
    WING_LINK_FORCE_ONLY,
    WING_LINK_MOMENT_ONLY,
    ZERO_ACCELERATION,
    map_opposed_joint_states_to_physical_wing_kinematics,
    resolve_wing_aero_acceleration,
    translate_wing_root_wrench_to_com_link,
    validate_wing_aero_acceleration_source,
    validate_wing_aero_coupling_mode,
    validate_wing_link_aero_load_mode,
)


def test_coupling_mode_validation_preserves_explicit_baseline() -> None:
    assert validate_wing_aero_coupling_mode(COMMANDED_BASE_EQUIVALENT) == COMMANDED_BASE_EQUIVALENT
    assert validate_wing_aero_coupling_mode(ACTUAL_MOTION_BASE_EQUIVALENT) == ACTUAL_MOTION_BASE_EQUIVALENT
    assert validate_wing_aero_coupling_mode(ACTUAL_PER_WING_LINK) == ACTUAL_PER_WING_LINK
    assert validate_wing_aero_coupling_mode(PRESCRIBED_PER_WING_LINK) == PRESCRIBED_PER_WING_LINK
    assert (
        validate_wing_aero_coupling_mode(SINUSOIDAL_PHASE_PER_WING_LINK)
        == SINUSOIDAL_PHASE_PER_WING_LINK
    )
    with pytest.raises(ValueError, match="Unsupported wing_aero_coupling_mode"):
        validate_wing_aero_coupling_mode("implicit")


def test_opposed_joint_states_map_to_equal_physical_wing_motion() -> None:
    position = torch.tensor([[0.31, -0.27]], dtype=torch.float64)
    velocity = torch.tensor([[2.4, -2.4]], dtype=torch.float64)
    acceleration = torch.tensor([[-18.0, 18.0]], dtype=torch.float64)

    result = map_opposed_joint_states_to_physical_wing_kinematics(
        joint_position_rad=position,
        joint_velocity_rad_s=velocity,
        joint_acceleration_rad_s2=acceleration,
        left_joint_mid_rad=0.02,
        right_joint_mid_rad=0.02,
    )

    torch.testing.assert_close(
        result.position_rad,
        torch.tensor([[0.29, 0.29]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.velocity_rad_s,
        torch.tensor([[2.4, 2.4]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.acceleration_rad_s2,
        torch.tensor([[-18.0, -18.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.mean(result.position_rad, dim=1),
        torch.tensor([0.29], dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.mean(result.velocity_rad_s, dim=1),
        torch.tensor([2.4], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.position_rad[:, 0] - result.position_rad[:, 1],
        torch.tensor([0.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (ACTUAL_JOINT_ACCELERATION, [[12.0, -8.0]]),
        (PRESCRIBED_ACCELERATION, [[3.0, 4.0]]),
        (ZERO_ACCELERATION, [[0.0, 0.0]]),
    ),
)
def test_wing_aero_acceleration_source_is_explicit(
    source: str,
    expected: list[list[float]],
) -> None:
    actual = torch.tensor([[12.0, -8.0]], dtype=torch.float64)
    prescribed = torch.tensor([[3.0, 4.0]], dtype=torch.float64)

    assert validate_wing_aero_acceleration_source(source) == source
    result = resolve_wing_aero_acceleration(
        source=source,
        actual_acceleration_rad_s2=actual,
        prescribed_acceleration_rad_s2=prescribed,
    )

    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))


def test_wing_aero_acceleration_source_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="wing_aero_acceleration_source"):
        validate_wing_aero_acceleration_source("filtered")


@pytest.mark.parametrize(
    "mode",
    (
        FULL_WING_LINK_WRENCH,
        WING_LINK_FORCE_ONLY,
        WING_LINK_MOMENT_ONLY,
    ),
)
def test_wing_link_aero_load_mode_is_explicit(mode: str) -> None:
    assert validate_wing_link_aero_load_mode(mode) == mode


def test_wing_root_to_com_translation_preserves_moment_about_root() -> None:
    force_link = torch.tensor(
        [[[3.0, -2.0, 5.0], [-4.0, 1.0, 6.0]]],
        dtype=torch.float64,
    )
    moment_about_root = torch.tensor(
        [[[0.2, -0.4, 0.7], [-0.1, 0.3, -0.6]]],
        dtype=torch.float64,
    )
    com_from_root = torch.tensor(
        [[[0.08, 0.21, -0.03], [0.08, -0.21, -0.03]]],
        dtype=torch.float64,
    )

    moment_about_com = translate_wing_root_wrench_to_com_link(
        force_link_n=force_link,
        moment_link_about_wing_origin_nm=moment_about_root,
        wing_com_position_link_m=com_from_root,
    )
    reconstructed_root = moment_about_com + torch.linalg.cross(
        com_from_root,
        force_link,
        dim=-1,
    )

    torch.testing.assert_close(reconstructed_root, moment_about_root)


def test_multibody_coupling_helpers_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 2\)"):
        map_opposed_joint_states_to_physical_wing_kinematics(
            joint_position_rad=torch.zeros(1, 3),
            joint_velocity_rad_s=torch.zeros(1, 3),
            joint_acceleration_rad_s2=torch.zeros(1, 3),
            left_joint_mid_rad=0.0,
            right_joint_mid_rad=0.0,
        )
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 2, 3\)"):
        translate_wing_root_wrench_to_com_link(
            force_link_n=torch.zeros(1, 3),
            moment_link_about_wing_origin_nm=torch.zeros(1, 3),
            wing_com_position_link_m=torch.zeros(1, 3),
        )
