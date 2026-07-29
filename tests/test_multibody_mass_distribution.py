from __future__ import annotations

import pytest
import torch

from flapping_bot.physics import (
    BODY_LINK_MASS_PROPERTIES,
    MEASURED_MULTIBODY_TOTAL_MASS_KG,
    build_measured_wing_multibody_tensors,
    inertia_triangle_margins_kg_m2,
    measured_wing_multibody_runtime_properties,
    project_diagonal_inertia_to_triangle_cone,
)
from flapping_bot.physics.measured_mass_properties import RigidBodyMassProperties


BODY_NAMES = ("base_link", "left_wing", "right_wing", "rudder", "left_tail", "right_tail")


def test_body_inertia_projection_is_nearest_triangle_consistent_diagonal() -> None:
    projected = project_diagonal_inertia_to_triangle_cone(BODY_LINK_MASS_PROPERTIES)
    assert tuple(projected[index][index] for index in range(3)) == pytest.approx(
        (0.003083333333333333, 0.023056666666666667, 0.019973333333333333),
        abs=1.0e-15,
    )
    projected_properties = RigidBodyMassProperties(
        link_name="base_link",
        frame="base_link_FLU",
        mass_kg=BODY_LINK_MASS_PROPERTIES.mass_kg,
        com_m=BODY_LINK_MASS_PROPERTIES.com_m,
        inertia_kg_m2=projected,
        source_cells="runtime projection",
    )
    assert min(inertia_triangle_margins_kg_m2(projected_properties)) == pytest.approx(0.0, abs=1.0e-15)


def test_runtime_partition_preserves_total_and_subtracts_placeholder_mass_from_base() -> None:
    properties = measured_wing_multibody_runtime_properties(placeholder_mass_kg=1.0e-4)
    assert properties["base_link"].mass_kg == pytest.approx(0.78231)
    assert properties["left_wing"].mass_kg == pytest.approx(0.06077)
    assert properties["right_wing"].mass_kg == pytest.approx(0.06077)
    total = sum(properties[name].mass_kg for name in properties) + 3.0e-4
    assert total == pytest.approx(MEASURED_MULTIBODY_TOTAL_MASS_KG, abs=1.0e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_builder_applies_measured_links_and_scales_placeholders(dtype: torch.dtype) -> None:
    masses = torch.tensor([[0.7, 0.2, 0.2, 0.05, 0.1, 0.1]], dtype=dtype).repeat(2, 1)
    inertias = torch.arange(1, 55, dtype=dtype).reshape(1, 6, 9).repeat(2, 1, 1) * 1.0e-4
    coms = torch.zeros((2, 6, 7), dtype=dtype)
    coms[:, :, 3] = 1.0

    new_masses, new_inertias, new_coms = build_measured_wing_multibody_tensors(
        body_names=BODY_NAMES,
        masses_kg=masses,
        inertias_kg_m2=inertias,
        com_poses_link=coms,
    )

    expected_masses = torch.tensor(
        [0.78231, 0.06077, 0.06077, 1.0e-4, 1.0e-4, 1.0e-4],
        dtype=dtype,
    )
    torch.testing.assert_close(new_masses, expected_masses.repeat(2, 1), atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(
        new_masses.sum(dim=1),
        torch.full((2,), MEASURED_MULTIBODY_TOTAL_MASS_KG, dtype=dtype),
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(new_coms[:, 1, :3], torch.tensor([-0.06040, 0.29394, 0.0], dtype=dtype).repeat(2, 1))
    torch.testing.assert_close(
        new_coms[:, 2, :3],
        torch.tensor([-0.06040, -0.29394, 0.0], dtype=dtype).repeat(2, 1),
    )
    torch.testing.assert_close(new_inertias[:, 3, :], inertias[:, 3, :] * (1.0e-4 / 0.05))
    torch.testing.assert_close(new_inertias[:, 4, :], inertias[:, 4, :] * (1.0e-4 / 0.1))
    torch.testing.assert_close(new_inertias[:, 5, :], inertias[:, 5, :] * (1.0e-4 / 0.1))

    assert not torch.equal(new_masses, masses)
    assert not torch.equal(new_inertias, inertias)
    assert not torch.equal(new_coms, coms)
