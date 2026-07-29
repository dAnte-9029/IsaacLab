"""Measured link mass properties for the PhysX multibody plant.

Mass uses kg, COM vectors use m in the named local frame, and inertia tensors
use kg m^2 about the corresponding link COM in that same frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Mapping

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]

MEASURED_MASS_PROPERTIES_WORKBOOK_NAME = "质量 重心位置 惯量.xlsx"
MEASURED_MASS_PROPERTIES_WORKBOOK_SHA256 = "4673fdee7be187278bf6ae9957b78645c781330007a43e20a4f24f5dfd24831b"
BODY_INERTIA_TRIANGLE_TOLERANCE_KG_M2 = 5.0e-4

FRD_TO_FLU: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)
LEFT_WING_FROM_RIGHT_WING_LINK_MIRROR: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class RigidBodyMassProperties:
    """One rigid body's mass properties about its COM in a named local frame."""

    link_name: str
    frame: str
    mass_kg: float
    com_m: Vector3
    inertia_kg_m2: Matrix3
    source_cells: str


def diagonal_inertia_kg_m2(ixx: float, iyy: float, izz: float) -> Matrix3:
    """Construct a diagonal inertia tensor in kg m^2."""

    return (
        (float(ixx), 0.0, 0.0),
        (0.0, float(iyy), 0.0),
        (0.0, 0.0, float(izz)),
    )


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _matrix_multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matrix_vector_multiply(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _determinant(matrix: Matrix3) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_orthogonal_transform(rotation_target_from_source: Matrix3, *, tolerance: float = 1.0e-12) -> None:
    identity = _matrix_multiply(rotation_target_from_source, _transpose(rotation_target_from_source))
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if not math.isclose(identity[row][column], expected, abs_tol=tolerance):
                raise ValueError("rotation_target_from_source must be orthogonal.")
    if not math.isclose(abs(_determinant(rotation_target_from_source)), 1.0, abs_tol=tolerance):
        raise ValueError("rotation_target_from_source determinant magnitude must be one.")


def transform_mass_properties(
    properties: RigidBodyMassProperties,
    *,
    rotation_target_from_source: Matrix3,
    target_frame: str,
    target_link_name: str | None = None,
    source_cells: str | None = None,
) -> RigidBodyMassProperties:
    """Transform COM and COM-referenced inertia into another orthonormal frame.

    The tensor transformation is ``I_target = R I_source R^T``. Reflections
    are supported for left/right geometric mirroring.
    """

    _validate_orthogonal_transform(rotation_target_from_source)
    inertia_target = _matrix_multiply(
        _matrix_multiply(rotation_target_from_source, properties.inertia_kg_m2),
        _transpose(rotation_target_from_source),
    )
    return replace(
        properties,
        link_name=properties.link_name if target_link_name is None else str(target_link_name),
        frame=str(target_frame),
        com_m=_matrix_vector_multiply(rotation_target_from_source, properties.com_m),
        inertia_kg_m2=inertia_target,
        source_cells=properties.source_cells if source_cells is None else str(source_cells),
    )


def inertia_triangle_margins_kg_m2(properties: RigidBodyMassProperties) -> Vector3:
    """Return ``(Ixx+Iyy-Izz, Ixx+Izz-Iyy, Iyy+Izz-Ixx)`` in kg m^2."""

    ixx = properties.inertia_kg_m2[0][0]
    iyy = properties.inertia_kg_m2[1][1]
    izz = properties.inertia_kg_m2[2][2]
    return ixx + iyy - izz, ixx + izz - iyy, iyy + izz - ixx


def validate_mass_properties(
    properties: RigidBodyMassProperties,
    *,
    inertia_triangle_tolerance_kg_m2: float = 0.0,
    symmetry_tolerance_kg_m2: float = 1.0e-12,
) -> None:
    """Validate finite, symmetric, positive-definite rigid-body properties."""

    if not math.isfinite(properties.mass_kg) or properties.mass_kg <= 0.0:
        raise ValueError(f"{properties.link_name} mass must be finite and positive.")
    values = (*properties.com_m, *(value for row in properties.inertia_kg_m2 for value in row))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{properties.link_name} mass properties must be finite.")
    for row in range(3):
        for column in range(3):
            if not math.isclose(
                properties.inertia_kg_m2[row][column],
                properties.inertia_kg_m2[column][row],
                abs_tol=float(symmetry_tolerance_kg_m2),
            ):
                raise ValueError(f"{properties.link_name} inertia tensor must be symmetric.")

    inertia = properties.inertia_kg_m2
    leading_minor_1 = inertia[0][0]
    leading_minor_2 = inertia[0][0] * inertia[1][1] - inertia[0][1] * inertia[1][0]
    leading_minor_3 = _determinant(inertia)
    if min(leading_minor_1, leading_minor_2, leading_minor_3) <= 0.0:
        raise ValueError(f"{properties.link_name} inertia tensor must be positive definite.")

    tolerance = float(inertia_triangle_tolerance_kg_m2)
    if tolerance < 0.0:
        raise ValueError("inertia_triangle_tolerance_kg_m2 must be non-negative.")
    if min(inertia_triangle_margins_kg_m2(properties)) < -tolerance:
        raise ValueError(f"{properties.link_name} inertia diagonal violates the triangle inequality.")


BODY_MEASURED_U_FRD = RigidBodyMassProperties(
    link_name="base_link",
    frame="U_FRD",
    mass_kg=0.78261,
    com_m=(-0.13103, 0.00625, -0.01500),
    inertia_kg_m2=diagonal_inertia_kg_m2(2.98e-3, 2.316e-2, 1.987e-2),
    source_cells="计算结果!B41:B42,B190",
)

RIGHT_WING_MEASURED_W_R0_FRD = RigidBodyMassProperties(
    link_name="right_wing",
    frame="W_R0_FRD",
    mass_kg=0.06077,
    com_m=(-0.06040, 0.29394, 0.0),
    inertia_kg_m2=diagonal_inertia_kg_m2(2.70e-3, 1.01e-3, 3.71e-3),
    source_cells="计算结果!B43:B46",
)

BODY_LINK_MASS_PROPERTIES = transform_mass_properties(
    BODY_MEASURED_U_FRD,
    rotation_target_from_source=FRD_TO_FLU,
    target_frame="base_link_FLU",
)
RIGHT_WING_LINK_MASS_PROPERTIES = transform_mass_properties(
    RIGHT_WING_MEASURED_W_R0_FRD,
    rotation_target_from_source=FRD_TO_FLU,
    target_frame="right_wing_FLU",
)
LEFT_WING_LINK_MASS_PROPERTIES = transform_mass_properties(
    RIGHT_WING_LINK_MASS_PROPERTIES,
    rotation_target_from_source=LEFT_WING_FROM_RIGHT_WING_LINK_MIRROR,
    target_frame="left_wing_FLU",
    target_link_name="left_wing",
    source_cells="mirrored from 计算结果!B43:B46",
)

MEASURED_FLAPPING_BOT_LINK_MASS_PROPERTIES: Mapping[str, RigidBodyMassProperties] = MappingProxyType(
    {
        "base_link": BODY_LINK_MASS_PROPERTIES,
        "left_wing": LEFT_WING_LINK_MASS_PROPERTIES,
        "right_wing": RIGHT_WING_LINK_MASS_PROPERTIES,
    }
)


def measured_three_link_total_mass_kg() -> float:
    """Return body plus two-wing measured mass in kg."""

    return sum(properties.mass_kg for properties in MEASURED_FLAPPING_BOT_LINK_MASS_PROPERTIES.values())
