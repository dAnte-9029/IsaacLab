from __future__ import annotations

import math

import pytest

from flapping_bot.physics.measured_mass_properties import (
    BODY_INERTIA_TRIANGLE_TOLERANCE_KG_M2,
    BODY_LINK_MASS_PROPERTIES,
    BODY_MEASURED_U_FRD,
    FRD_TO_FLU,
    LEFT_WING_FROM_RIGHT_WING_LINK_MIRROR,
    LEFT_WING_LINK_MASS_PROPERTIES,
    MEASURED_FLAPPING_BOT_LINK_MASS_PROPERTIES,
    MEASURED_MASS_PROPERTIES_WORKBOOK_NAME,
    MEASURED_MASS_PROPERTIES_WORKBOOK_SHA256,
    RIGHT_WING_LINK_MASS_PROPERTIES,
    RIGHT_WING_MEASURED_W_R0_FRD,
    RigidBodyMassProperties,
    inertia_triangle_margins_kg_m2,
    measured_three_link_total_mass_kg,
    transform_mass_properties,
    validate_mass_properties,
)


def _flatten_matrix(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, ...]:
    return tuple(value for row in matrix for value in row)


def test_measured_values_and_provenance_match_workbook_release() -> None:
    assert MEASURED_MASS_PROPERTIES_WORKBOOK_NAME == "质量 重心位置 惯量.xlsx"
    assert MEASURED_MASS_PROPERTIES_WORKBOOK_SHA256 == (
        "4673fdee7be187278bf6ae9957b78645c781330007a43e20a4f24f5dfd24831b"
    )
    assert BODY_MEASURED_U_FRD.mass_kg == pytest.approx(0.78261)
    assert BODY_MEASURED_U_FRD.com_m == pytest.approx((-0.13103, 0.00625, -0.01500))
    assert tuple(BODY_MEASURED_U_FRD.inertia_kg_m2[index][index] for index in range(3)) == pytest.approx(
        (2.98e-3, 2.316e-2, 1.987e-2)
    )
    assert RIGHT_WING_MEASURED_W_R0_FRD.mass_kg == pytest.approx(0.06077)
    assert RIGHT_WING_MEASURED_W_R0_FRD.com_m == pytest.approx((-0.06040, 0.29394, 0.0))
    assert tuple(RIGHT_WING_MEASURED_W_R0_FRD.inertia_kg_m2[index][index] for index in range(3)) == pytest.approx(
        (2.70e-3, 1.01e-3, 3.71e-3)
    )


def test_frd_to_flu_transforms_body_and_right_wing_com() -> None:
    assert BODY_LINK_MASS_PROPERTIES.frame == "base_link_FLU"
    assert BODY_LINK_MASS_PROPERTIES.com_m == pytest.approx((-0.13103, -0.00625, 0.01500))
    assert RIGHT_WING_LINK_MASS_PROPERTIES.frame == "right_wing_FLU"
    assert RIGHT_WING_LINK_MASS_PROPERTIES.com_m == pytest.approx((-0.06040, -0.29394, 0.0))


def test_frd_to_flu_transforms_full_inertia_tensor_independently() -> None:
    source = RigidBodyMassProperties(
        link_name="fixture",
        frame="fixture_FRD",
        mass_kg=1.0,
        com_m=(1.0, 2.0, 3.0),
        inertia_kg_m2=(
            (4.0, 0.1, 0.2),
            (0.1, 5.0, 0.3),
            (0.2, 0.3, 6.0),
        ),
        source_cells="hand fixture",
    )

    target = transform_mass_properties(
        source,
        rotation_target_from_source=FRD_TO_FLU,
        target_frame="fixture_FLU",
    )

    assert target.com_m == pytest.approx((1.0, -2.0, -3.0))
    assert _flatten_matrix(target.inertia_kg_m2) == pytest.approx(
        _flatten_matrix(
            (
                (4.0, -0.1, -0.2),
                (-0.1, 5.0, 0.3),
                (-0.2, 0.3, 6.0),
            )
        )
    )


def test_left_wing_is_link_frame_mirror_of_right_wing() -> None:
    assert LEFT_WING_LINK_MASS_PROPERTIES.link_name == "left_wing"
    assert LEFT_WING_LINK_MASS_PROPERTIES.frame == "left_wing_FLU"
    assert LEFT_WING_LINK_MASS_PROPERTIES.mass_kg == RIGHT_WING_LINK_MASS_PROPERTIES.mass_kg
    assert LEFT_WING_LINK_MASS_PROPERTIES.com_m == pytest.approx((-0.06040, 0.29394, 0.0))
    assert LEFT_WING_LINK_MASS_PROPERTIES.inertia_kg_m2 == RIGHT_WING_LINK_MASS_PROPERTIES.inertia_kg_m2


def test_left_right_mirror_flips_ixy_and_iyz_but_preserves_ixz() -> None:
    right = RigidBodyMassProperties(
        link_name="right_wing",
        frame="right_wing_FLU",
        mass_kg=1.0,
        com_m=(1.0, -2.0, 3.0),
        inertia_kg_m2=(
            (4.0, 0.1, 0.2),
            (0.1, 5.0, 0.3),
            (0.2, 0.3, 6.0),
        ),
        source_cells="hand fixture",
    )

    left = transform_mass_properties(
        right,
        rotation_target_from_source=LEFT_WING_FROM_RIGHT_WING_LINK_MIRROR,
        target_frame="left_wing_FLU",
        target_link_name="left_wing",
    )

    assert left.com_m == pytest.approx((1.0, 2.0, 3.0))
    assert _flatten_matrix(left.inertia_kg_m2) == pytest.approx(
        _flatten_matrix(
            (
                (4.0, -0.1, 0.2),
                (-0.1, 5.0, -0.3),
                (0.2, -0.3, 6.0),
            )
        )
    )


def test_measured_three_link_mass_is_exact_release_total() -> None:
    assert set(MEASURED_FLAPPING_BOT_LINK_MASS_PROPERTIES) == {"base_link", "left_wing", "right_wing"}
    assert measured_three_link_total_mass_kg() == pytest.approx(0.90415, abs=1.0e-12)


def test_measured_tensors_are_positive_definite_with_documented_body_tolerance() -> None:
    validate_mass_properties(
        BODY_LINK_MASS_PROPERTIES,
        inertia_triangle_tolerance_kg_m2=BODY_INERTIA_TRIANGLE_TOLERANCE_KG_M2,
    )
    validate_mass_properties(LEFT_WING_LINK_MASS_PROPERTIES)
    validate_mass_properties(RIGHT_WING_LINK_MASS_PROPERTIES)

    body_margins = inertia_triangle_margins_kg_m2(BODY_LINK_MASS_PROPERTIES)
    assert body_margins[1] == pytest.approx(-3.10e-4, abs=1.0e-12)
    with pytest.raises(ValueError, match="triangle inequality"):
        validate_mass_properties(BODY_LINK_MASS_PROPERTIES)


def test_validation_rejects_non_positive_definite_inertia() -> None:
    invalid = RigidBodyMassProperties(
        link_name="invalid",
        frame="invalid_FLU",
        mass_kg=1.0,
        com_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        source_cells="hand fixture",
    )

    with pytest.raises(ValueError, match="positive definite"):
        validate_mass_properties(invalid, inertia_triangle_tolerance_kg_m2=math.inf)
