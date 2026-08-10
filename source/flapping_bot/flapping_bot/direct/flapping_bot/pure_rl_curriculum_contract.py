"""Backend-independent contracts for measured PureRL curriculum stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class PureRLSharedContract:
    """Semantics that must remain identical across curricula and backends."""

    schema_version: str
    physics_dt_s: float
    policy_decimation: int
    action_interface: str
    tail_aero_deflection_source: str
    action_channels: tuple[str, ...]
    observation_dim: int
    minimum_flap_frequency_hz: float
    maximum_flap_frequency_hz: float
    frequency_governor_rise_hz_per_s: float
    frequency_governor_fall_hz_per_s: float

    @property
    def policy_frequency_hz(self) -> float:
        """Return the policy action frequency implied by physics and decimation."""

        return 1.0 / (self.physics_dt_s * self.policy_decimation)

    @property
    def semantic_signature(self) -> tuple[object, ...]:
        """Return the immutable signature used by stage/backend promotion gates."""

        return (
            self.schema_version,
            self.physics_dt_s,
            self.policy_decimation,
            self.action_interface,
            self.tail_aero_deflection_source,
            self.action_channels,
            self.observation_dim,
            self.minimum_flap_frequency_hz,
            self.maximum_flap_frequency_hz,
            self.frequency_governor_rise_hz_per_s,
            self.frequency_governor_fall_hz_per_s,
        )


@dataclass(frozen=True)
class PureRLCurriculumContract:
    """Task scope and retention requirements for one curriculum stage."""

    stage_id: str
    shared_contract_id: str
    enabled_primitives: tuple[str, ...]
    required_retention_stages: tuple[str, ...]
    wind_enabled: bool


@dataclass(frozen=True)
class PureRLBackendContract:
    """Physics-backend status without changing the policy-facing contract."""

    backend_id: str
    shared_contract_id: str
    device_kind: str
    wing_drive_variant: str
    replicate_physics: bool
    authoritative: bool
    requires_cpu_promotion: bool


PURE_RL_SHARED_CONTRACT = PureRLSharedContract(
    schema_version="measured_pure_rl_shared_v1",
    physics_dt_s=1.0 / 480.0,
    policy_decimation=8,
    action_interface="direct_tail_surface",
    tail_aero_deflection_source="actual_joint_position",
    action_channels=("flap_frequency", "rudder", "left_elevon", "right_elevon"),
    observation_dim=555,
    minimum_flap_frequency_hz=0.0,
    maximum_flap_frequency_hz=5.0,
    frequency_governor_rise_hz_per_s=2.0,
    frequency_governor_fall_hz_per_s=2.0,
)


PURE_RL_CURRICULUM_CONTRACTS: Mapping[str, PureRLCurriculumContract] = MappingProxyType(
    {
        "c1_straight": PureRLCurriculumContract(
            stage_id="c1_straight",
            shared_contract_id=PURE_RL_SHARED_CONTRACT.schema_version,
            enabled_primitives=("straight_level",),
            required_retention_stages=(),
            wind_enabled=False,
        ),
        "c2_longitudinal": PureRLCurriculumContract(
            stage_id="c2_longitudinal",
            shared_contract_id=PURE_RL_SHARED_CONTRACT.schema_version,
            enabled_primitives=("straight_level", "climb", "descent"),
            required_retention_stages=("c1_straight",),
            wind_enabled=False,
        ),
        "c3_lateral_composite": PureRLCurriculumContract(
            stage_id="c3_lateral_composite",
            shared_contract_id=PURE_RL_SHARED_CONTRACT.schema_version,
            enabled_primitives=(
                "straight_level",
                "climb",
                "descent",
                "turn",
                "loiter",
                "composite",
            ),
            required_retention_stages=("c1_straight", "c2_longitudinal"),
            wind_enabled=False,
        ),
    }
)


PURE_RL_BACKEND_CONTRACTS: Mapping[str, PureRLBackendContract] = MappingProxyType(
    {
        "cpu_native_authority": PureRLBackendContract(
            backend_id="cpu_native_authority",
            shared_contract_id=PURE_RL_SHARED_CONTRACT.schema_version,
            device_kind="cpu",
            wing_drive_variant="native_holonomic_drive",
            replicate_physics=False,
            authoritative=True,
            requires_cpu_promotion=False,
        ),
        "gpu_implicit_candidate": PureRLBackendContract(
            backend_id="gpu_implicit_candidate",
            shared_contract_id=PURE_RL_SHARED_CONTRACT.schema_version,
            device_kind="cuda",
            wing_drive_variant="ideal_coupled_drive",
            replicate_physics=True,
            authoritative=False,
            requires_cpu_promotion=True,
        ),
    }
)


def validate_shared_contract(contract: PureRLSharedContract) -> None:
    """Fail when a purported shared contract drifts from the approved semantics."""

    if not math.isfinite(contract.physics_dt_s) or contract.physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be finite and positive")
    if contract.policy_decimation <= 0:
        raise ValueError("policy_decimation must be positive")
    if not math.isclose(contract.policy_frequency_hz, 60.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("policy_frequency_hz must remain 60 Hz")
    if contract.action_interface != "direct_tail_surface" or len(contract.action_channels) != 4:
        raise ValueError("action interface must remain the four-channel direct-tail contract")
    if contract.tail_aero_deflection_source != "actual_joint_position":
        raise ValueError("tail_aero_deflection_source must remain actual_joint_position")
    if contract.observation_dim != 555:
        raise ValueError("observation_dim must remain 555")
    if (
        not math.isfinite(contract.minimum_flap_frequency_hz)
        or not math.isfinite(contract.maximum_flap_frequency_hz)
        or contract.minimum_flap_frequency_hz < 0.0
        or contract.maximum_flap_frequency_hz <= contract.minimum_flap_frequency_hz
    ):
        raise ValueError("frequency bounds must remain finite and ordered")
    for name, value in (
        ("frequency_governor_rise_hz_per_s", contract.frequency_governor_rise_hz_per_s),
        ("frequency_governor_fall_hz_per_s", contract.frequency_governor_fall_hz_per_s),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def validate_contract_graph(
    shared: PureRLSharedContract,
    curricula: Mapping[str, PureRLCurriculumContract],
    backends: Mapping[str, PureRLBackendContract],
) -> None:
    """Validate curriculum/backend ownership of one policy-facing contract."""

    validate_shared_contract(shared)
    for name, stage in curricula.items():
        if stage.stage_id != name:
            raise ValueError(f"Curriculum registry key does not match stage_id: {name}")
        if stage.shared_contract_id != shared.schema_version:
            raise ValueError(f"Curriculum {name} does not reference the shared contract")
    for name, backend in backends.items():
        if backend.backend_id != name:
            raise ValueError(f"Backend registry key does not match backend_id: {name}")
        if backend.shared_contract_id != shared.schema_version:
            raise ValueError(f"Backend {name} does not reference the shared contract")

    authoritative = [backend for backend in backends.values() if backend.authoritative]
    if len(authoritative) != 1:
        raise ValueError("PureRL requires exactly one authoritative backend")
    authority = authoritative[0]
    if authority.device_kind != "cpu" or authority.wing_drive_variant != "native_holonomic_drive":
        raise ValueError("The authoritative backend must remain the CPU native-holonomic plant")
    for backend in backends.values():
        if not backend.authoritative and not backend.requires_cpu_promotion:
            raise ValueError(f"Candidate backend must require CPU promotion: {backend.backend_id}")


validate_shared_contract(PURE_RL_SHARED_CONTRACT)
validate_contract_graph(PURE_RL_SHARED_CONTRACT, PURE_RL_CURRICULUM_CONTRACTS, PURE_RL_BACKEND_CONTRACTS)


__all__ = [
    "PURE_RL_BACKEND_CONTRACTS",
    "PURE_RL_CURRICULUM_CONTRACTS",
    "PURE_RL_SHARED_CONTRACT",
    "PureRLBackendContract",
    "PureRLCurriculumContract",
    "PureRLSharedContract",
    "validate_contract_graph",
    "validate_shared_contract",
]
