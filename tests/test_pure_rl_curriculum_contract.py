from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from flapping_bot.direct.flapping_bot.pure_rl_curriculum_contract import (
    PURE_RL_BACKEND_CONTRACTS,
    PURE_RL_CURRICULUM_CONTRACTS,
    PURE_RL_SHARED_CONTRACT,
    validate_contract_graph,
    validate_shared_contract,
)


def test_shared_contract_freezes_existing_measured_pure_rl_semantics() -> None:
    contract = PURE_RL_SHARED_CONTRACT

    assert contract.schema_version == "measured_pure_rl_shared_v1"
    assert contract.physics_dt_s == pytest.approx(1.0 / 480.0)
    assert contract.policy_decimation == 8
    assert contract.policy_frequency_hz == pytest.approx(60.0)
    assert contract.action_interface == "direct_tail_surface"
    assert contract.action_channels == (
        "flap_frequency",
        "rudder",
        "left_elevon",
        "right_elevon",
    )
    assert contract.observation_dim == 555
    assert contract.minimum_flap_frequency_hz == 0.0
    assert contract.maximum_flap_frequency_hz == 5.0
    assert contract.frequency_governor_rise_hz_per_s == 2.0
    assert contract.frequency_governor_fall_hz_per_s == 2.0


def test_curriculum_contracts_match_the_approved_stage_boundary() -> None:
    c1 = PURE_RL_CURRICULUM_CONTRACTS["c1_straight"]
    c2 = PURE_RL_CURRICULUM_CONTRACTS["c2_longitudinal"]
    c3 = PURE_RL_CURRICULUM_CONTRACTS["c3_lateral_composite"]

    assert c1.enabled_primitives == ("straight_level",)
    assert c1.required_retention_stages == ()
    assert c2.enabled_primitives == ("straight_level", "climb", "descent")
    assert c2.required_retention_stages == ("c1_straight",)
    assert c3.enabled_primitives == (
        "straight_level",
        "climb",
        "descent",
        "turn",
        "loiter",
        "composite",
    )
    assert c3.required_retention_stages == ("c1_straight", "c2_longitudinal")
    assert all(not stage.wind_enabled for stage in (c1, c2, c3))
    assert all(stage.shared_contract_id == PURE_RL_SHARED_CONTRACT.schema_version for stage in (c1, c2, c3))


def test_backend_contracts_keep_cpu_authority_and_gpu_candidate_separate() -> None:
    cpu = PURE_RL_BACKEND_CONTRACTS["cpu_native_authority"]
    gpu = PURE_RL_BACKEND_CONTRACTS["gpu_implicit_candidate"]

    assert cpu.authoritative is True
    assert cpu.device_kind == "cpu"
    assert cpu.wing_drive_variant == "native_holonomic_drive"
    assert cpu.replicate_physics is False
    assert cpu.requires_cpu_promotion is False

    assert gpu.authoritative is False
    assert gpu.device_kind == "cuda"
    assert gpu.wing_drive_variant == "ideal_coupled_drive"
    assert gpu.replicate_physics is True
    assert gpu.requires_cpu_promotion is True
    assert cpu.shared_contract_id == gpu.shared_contract_id == PURE_RL_SHARED_CONTRACT.schema_version


def test_contract_graph_rejects_stage_or_backend_boundary_drift() -> None:
    validate_contract_graph(
        PURE_RL_SHARED_CONTRACT,
        PURE_RL_CURRICULUM_CONTRACTS,
        PURE_RL_BACKEND_CONTRACTS,
    )

    drifted_stages = dict(PURE_RL_CURRICULUM_CONTRACTS)
    drifted_stages["c2_longitudinal"] = replace(
        drifted_stages["c2_longitudinal"], shared_contract_id="different_action_obs_contract"
    )
    with pytest.raises(ValueError, match="shared contract"):
        validate_contract_graph(PURE_RL_SHARED_CONTRACT, drifted_stages, PURE_RL_BACKEND_CONTRACTS)

    drifted_backends = dict(PURE_RL_BACKEND_CONTRACTS)
    drifted_backends["gpu_implicit_candidate"] = replace(
        drifted_backends["gpu_implicit_candidate"], authoritative=True, requires_cpu_promotion=False
    )
    with pytest.raises(ValueError, match="exactly one authoritative"):
        validate_contract_graph(PURE_RL_SHARED_CONTRACT, PURE_RL_CURRICULUM_CONTRACTS, drifted_backends)


def test_shared_contract_validator_fails_on_semantic_drift() -> None:
    validate_shared_contract(PURE_RL_SHARED_CONTRACT)

    with pytest.raises(ValueError, match="observation_dim"):
        validate_shared_contract(replace(PURE_RL_SHARED_CONTRACT, observation_dim=554))
    with pytest.raises(ValueError, match="policy_frequency_hz"):
        validate_shared_contract(replace(PURE_RL_SHARED_CONTRACT, policy_decimation=4))
    with pytest.raises(ValueError, match="frequency bounds"):
        validate_shared_contract(replace(PURE_RL_SHARED_CONTRACT, maximum_flap_frequency_hz=0.0))


def test_measured_environment_defaults_reference_the_shared_contract() -> None:
    env_path = (
        Path(__file__).resolve().parents[1]
        / "source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py"
    )
    module = ast.parse(env_path.read_text(encoding="utf-8"))
    cfg = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg"
    )
    expected_attributes = {
        "decimation": "policy_decimation",
        "observation_space": "observation_dim",
        "min_flap_hz": "minimum_flap_frequency_hz",
        "max_flap_hz": "maximum_flap_frequency_hz",
        "frequency_governor_maximum_rise_rate_hz_per_s": "frequency_governor_rise_hz_per_s",
        "frequency_governor_maximum_fall_rate_hz_per_s": "frequency_governor_fall_hz_per_s",
    }
    assignments = {
        node.target.id: node.value
        for node in cfg.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    for field_name, attribute_name in expected_attributes.items():
        value = assignments[field_name]
        assert isinstance(value, ast.Attribute)
        assert isinstance(value.value, ast.Name)
        assert value.value.id == "PURE_RL_SHARED_CONTRACT"
        assert value.attr == attribute_name
