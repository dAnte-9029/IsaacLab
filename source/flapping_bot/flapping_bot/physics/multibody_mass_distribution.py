"""Runtime mass distribution for the measured-wing PhysX plant.

Mass uses kg, COM vectors use link-local FLU meters, and inertia tensors use
kg m^2 about each link COM in the same link-local frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .measured_mass_properties import (
    BODY_LINK_MASS_PROPERTIES,
    LEFT_WING_LINK_MASS_PROPERTIES,
    Matrix3,
    RIGHT_WING_LINK_MASS_PROPERTIES,
    RigidBodyMassProperties,
    diagonal_inertia_kg_m2,
)

NEAR_SINGLE_RIGID_BODY_PLANT = "near_single_rigid_body"
MEASURED_WING_MULTIBODY_PLANT = "measured_wing_multibody"
FLAPPING_BOT_PLANT_VARIANTS = frozenset(
    {
        NEAR_SINGLE_RIGID_BODY_PLANT,
        MEASURED_WING_MULTIBODY_PLANT,
    }
)

MEASURED_MULTIBODY_TOTAL_MASS_KG = 0.90415
MEASURED_MULTIBODY_PLACEHOLDER_LINK_NAMES = ("left_tail", "right_tail", "rudder")
MEASURED_MULTIBODY_DEFAULT_PLACEHOLDER_MASS_KG = 1.0e-4


@dataclass(frozen=True)
class RuntimeMassProperties:
    """Runtime-safe mass properties for one named PhysX link."""

    link_name: str
    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_kg_m2: Matrix3


def validate_flapping_bot_plant_variant(plant_variant: str) -> str:
    """Validate and return a flapping-bot plant variant name."""

    variant = str(plant_variant)
    if variant not in FLAPPING_BOT_PLANT_VARIANTS:
        expected = ", ".join(sorted(FLAPPING_BOT_PLANT_VARIANTS))
        raise ValueError(f"Unknown flapping-bot plant variant {variant!r}; expected one of: {expected}.")
    return variant


def project_diagonal_inertia_to_triangle_cone(properties: RigidBodyMassProperties) -> Matrix3:
    """Return the nearest Euclidean diagonal satisfying inertia triangles.

    Positive diagonal rigid-body inertia can violate at most one triangle
    inequality. The orthogonal projection shares the violated margin equally
    across the three diagonal entries and leaves valid inputs unchanged.
    """

    diagonal = [float(properties.inertia_kg_m2[index][index]) for index in range(3)]
    if not all(math.isfinite(value) and value > 0.0 for value in diagonal):
        raise ValueError(f"{properties.link_name} inertia diagonal must be finite and positive.")

    margins = (
        diagonal[0] + diagonal[1] - diagonal[2],
        diagonal[0] + diagonal[2] - diagonal[1],
        diagonal[1] + diagonal[2] - diagonal[0],
    )
    violated_index = min(range(3), key=margins.__getitem__)
    violated_margin = margins[violated_index]
    if violated_margin >= 0.0:
        return diagonal_inertia_kg_m2(*diagonal)

    correction = -violated_margin / 3.0
    diagonal[violated_index] -= correction
    for index in range(3):
        if index != violated_index:
            diagonal[index] += correction
    return diagonal_inertia_kg_m2(*diagonal)


def measured_wing_multibody_runtime_properties(
    *,
    placeholder_mass_kg: float = MEASURED_MULTIBODY_DEFAULT_PLACEHOLDER_MASS_KG,
) -> dict[str, RuntimeMassProperties]:
    """Build the explicit body/wing mass partition for the PhysX plant."""

    placeholder_mass = float(placeholder_mass_kg)
    if not math.isfinite(placeholder_mass) or placeholder_mass <= 0.0:
        raise ValueError("placeholder_mass_kg must be finite and positive.")

    placeholder_total = placeholder_mass * len(MEASURED_MULTIBODY_PLACEHOLDER_LINK_NAMES)
    base_mass = BODY_LINK_MASS_PROPERTIES.mass_kg - placeholder_total
    if base_mass <= 0.0:
        raise ValueError("placeholder masses exceed the measured non-wing body mass.")

    properties = {
        "base_link": RuntimeMassProperties(
            link_name="base_link",
            mass_kg=base_mass,
            com_m=BODY_LINK_MASS_PROPERTIES.com_m,
            inertia_kg_m2=project_diagonal_inertia_to_triangle_cone(BODY_LINK_MASS_PROPERTIES),
        ),
        "left_wing": RuntimeMassProperties(
            link_name="left_wing",
            mass_kg=LEFT_WING_LINK_MASS_PROPERTIES.mass_kg,
            com_m=LEFT_WING_LINK_MASS_PROPERTIES.com_m,
            inertia_kg_m2=LEFT_WING_LINK_MASS_PROPERTIES.inertia_kg_m2,
        ),
        "right_wing": RuntimeMassProperties(
            link_name="right_wing",
            mass_kg=RIGHT_WING_LINK_MASS_PROPERTIES.mass_kg,
            com_m=RIGHT_WING_LINK_MASS_PROPERTIES.com_m,
            inertia_kg_m2=RIGHT_WING_LINK_MASS_PROPERTIES.inertia_kg_m2,
        ),
    }
    return properties


def build_measured_wing_multibody_tensors(
    *,
    body_names: Sequence[str],
    masses_kg: torch.Tensor,
    inertias_kg_m2: torch.Tensor,
    com_poses_link: torch.Tensor,
    placeholder_mass_kg: float = MEASURED_MULTIBODY_DEFAULT_PLACEHOLDER_MASS_KG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return PhysX tensors for the measured-wing multibody mass partition.

    Args:
        body_names: Link names matching the second tensor dimension.
        masses_kg: Current masses, shape ``(N, B)``.
        inertias_kg_m2: Current flattened inertia tensors, shape ``(N, B, 9)``.
        com_poses_link: Current COM poses ``(position, quaternion wxyz)``,
            shape ``(N, B, 7)``.
        placeholder_mass_kg: Positive numerical mass assigned to each tail and
            rudder link.

    Returns:
        Cloned ``(masses, inertias, com_poses)`` tensors with the measured
        partition applied.
    """

    num_bodies = len(body_names)
    if masses_kg.ndim != 2 or masses_kg.shape[1] != num_bodies:
        raise ValueError("masses_kg must have shape (N, len(body_names)).")
    if inertias_kg_m2.shape != (*masses_kg.shape, 9):
        raise ValueError("inertias_kg_m2 must have shape (N, B, 9).")
    if com_poses_link.shape != (*masses_kg.shape, 7):
        raise ValueError("com_poses_link must have shape (N, B, 7).")

    required_names = {
        "base_link",
        "left_wing",
        "right_wing",
        *MEASURED_MULTIBODY_PLACEHOLDER_LINK_NAMES,
    }
    name_to_index = {name: index for index, name in enumerate(body_names)}
    missing_names = sorted(required_names.difference(name_to_index))
    if missing_names:
        raise ValueError(f"Measured multibody plant is missing required links: {missing_names}.")

    runtime_properties = measured_wing_multibody_runtime_properties(
        placeholder_mass_kg=placeholder_mass_kg,
    )
    masses = masses_kg.clone()
    inertias = inertias_kg_m2.clone()
    com_poses = com_poses_link.clone()

    for link_name, properties in runtime_properties.items():
        body_id = name_to_index[link_name]
        masses[:, body_id] = properties.mass_kg
        inertia_flat = tuple(value for row in properties.inertia_kg_m2 for value in row)
        inertias[:, body_id, :] = torch.tensor(
            inertia_flat,
            dtype=inertias.dtype,
            device=inertias.device,
        )
        com_poses[:, body_id, :3] = torch.tensor(
            properties.com_m,
            dtype=com_poses.dtype,
            device=com_poses.device,
        )
        com_poses[:, body_id, 3:] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            dtype=com_poses.dtype,
            device=com_poses.device,
        )

    placeholder_mass = float(placeholder_mass_kg)
    for link_name in MEASURED_MULTIBODY_PLACEHOLDER_LINK_NAMES:
        body_id = name_to_index[link_name]
        old_mass = masses_kg[:, body_id]
        if torch.any(old_mass <= 0.0):
            raise ValueError(f"{link_name} source mass must be positive before placeholder scaling.")
        ratio = placeholder_mass / old_mass
        masses[:, body_id] = placeholder_mass
        inertias[:, body_id, :] = inertias_kg_m2[:, body_id, :] * ratio.unsqueeze(-1)

    expected_total = torch.full(
        (masses.shape[0],),
        MEASURED_MULTIBODY_TOTAL_MASS_KG,
        dtype=masses.dtype,
        device=masses.device,
    )
    if not torch.allclose(masses.sum(dim=1), expected_total, atol=1.0e-7, rtol=0.0):
        raise RuntimeError("Measured multibody tensor construction did not preserve the 0.90415 kg total mass.")
    return masses, inertias, com_poses
