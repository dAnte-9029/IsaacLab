from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/action_contract.py"
)
SPEC = importlib.util.spec_from_file_location("flapping_action_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
action_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(action_contract)


def test_zero_to_five_hz_action_mapping_and_inverse_round_trip() -> None:
    actions = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)

    frequencies = action_contract.normalized_action_to_frequency_hz(
        actions,
        minimum_frequency_hz=0.0,
        maximum_frequency_hz=5.0,
    )
    recovered_actions = action_contract.frequency_hz_to_normalized_action(
        frequencies,
        minimum_frequency_hz=0.0,
        maximum_frequency_hz=5.0,
    )

    assert frequencies.tolist() == pytest.approx([0.0, 2.5, 5.0])
    assert torch.equal(recovered_actions, actions)
    assert frequencies.dtype == actions.dtype
    assert frequencies.device == actions.device


def test_frequency_action_mapping_clamps_inputs_and_rejects_invalid_bounds() -> None:
    actions = torch.tensor([-2.0, 2.0])

    frequencies = action_contract.normalized_action_to_frequency_hz(
        actions,
        minimum_frequency_hz=0.0,
        maximum_frequency_hz=5.0,
    )

    assert frequencies.tolist() == pytest.approx([0.0, 5.0])
    with pytest.raises(ValueError, match="maximum_frequency_hz"):
        action_contract.normalized_action_to_frequency_hz(
            actions,
            minimum_frequency_hz=5.0,
            maximum_frequency_hz=5.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        action_contract.normalized_action_to_frequency_hz(
            actions,
            minimum_frequency_hz=-1.0,
            maximum_frequency_hz=5.0,
        )


def test_frequency_governor_applies_asymmetric_physical_slew_without_overshoot() -> None:
    requested = torch.tensor([4.0, 0.0, 2.51, 2.49], dtype=torch.float64)
    previous = torch.full((4,), 2.5, dtype=torch.float64)

    step = action_contract.apply_frequency_slew_governor(
        requested,
        previous_frequency_hz=previous,
        policy_step_dt_s=0.02,
        maximum_rise_rate_hz_per_s=3.0,
        maximum_fall_rate_hz_per_s=1.0,
    )

    torch.testing.assert_close(
        step.applied_frequency_hz,
        torch.tensor([2.56, 2.48, 2.51, 2.49], dtype=torch.float64),
    )
    torch.testing.assert_close(
        step.slew_hz_per_s,
        torch.tensor([3.0, -1.0, 0.5, -0.5], dtype=torch.float64),
    )
    assert step.limited.tolist() == [True, True, False, False]
    assert step.requested_frequency_hz.data_ptr() == requested.data_ptr()
    assert step.applied_frequency_hz.dtype == requested.dtype
    assert step.applied_frequency_hz.device == requested.device


def test_frequency_governor_preserves_batches_and_fails_closed_on_invalid_inputs() -> None:
    requested = torch.tensor([[1.0, 4.0], [2.0, 3.0]], dtype=torch.float32)
    previous = torch.tensor([[1.0, 3.0], [2.5, 3.0]], dtype=torch.float32)

    step = action_contract.apply_frequency_slew_governor(
        requested,
        previous_frequency_hz=previous,
        policy_step_dt_s=0.1,
        maximum_rise_rate_hz_per_s=2.0,
        maximum_fall_rate_hz_per_s=4.0,
    )

    assert step.applied_frequency_hz.shape == requested.shape
    torch.testing.assert_close(
        step.applied_frequency_hz,
        torch.tensor([[1.0, 3.2], [2.1, 3.0]], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="same shape"):
        action_contract.apply_frequency_slew_governor(
            requested,
            previous_frequency_hz=torch.zeros(4),
            policy_step_dt_s=0.1,
            maximum_rise_rate_hz_per_s=2.0,
            maximum_fall_rate_hz_per_s=2.0,
        )
    with pytest.raises(ValueError, match="policy_step_dt_s"):
        action_contract.apply_frequency_slew_governor(
            requested,
            previous_frequency_hz=previous,
            policy_step_dt_s=0.0,
            maximum_rise_rate_hz_per_s=2.0,
            maximum_fall_rate_hz_per_s=2.0,
        )
    with pytest.raises(ValueError, match="rate"):
        action_contract.apply_frequency_slew_governor(
            requested,
            previous_frequency_hz=previous,
            policy_step_dt_s=0.1,
            maximum_rise_rate_hz_per_s=-1.0,
            maximum_fall_rate_hz_per_s=2.0,
        )


def test_direct_joint_action_uses_zero_center_and_asymmetric_limits() -> None:
    actions = torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0], dtype=torch.float64)

    positions = action_contract.normalized_action_to_joint_position(
        actions,
        lower_limit_rad=-0.6,
        upper_limit_rad=0.8,
    )
    recovered = action_contract.joint_position_to_normalized_action(
        positions,
        lower_limit_rad=-0.6,
        upper_limit_rad=0.8,
    )

    assert positions.tolist() == pytest.approx([-0.6, -0.3, 0.0, 0.2, 0.8])
    torch.testing.assert_close(recovered, actions)
    assert positions.dtype == actions.dtype
    assert positions.device == actions.device


def test_direct_joint_action_supports_batched_runtime_limits_and_clamps() -> None:
    actions = torch.tensor([[-2.0, 2.0, 0.5]])
    lower = torch.tensor([-0.7, -0.6, -0.5])
    upper = torch.tensor([0.4, 0.5, 0.6])

    positions = action_contract.normalized_action_to_joint_position(
        actions,
        lower_limit_rad=lower,
        upper_limit_rad=upper,
    )

    torch.testing.assert_close(positions, torch.tensor([[-0.7, 0.5, 0.3]]))
    with pytest.raises(ValueError, match="strictly contain zero"):
        action_contract.normalized_action_to_joint_position(
            actions,
            lower_limit_rad=0.0,
            upper_limit_rad=1.0,
        )


def test_action_and_tail_aero_interface_validators_fail_closed() -> None:
    assert (
        action_contract.validate_action_interface("direct_tail_surface")
        == action_contract.DIRECT_TAIL_SURFACE_ACTION
    )
    assert (
        action_contract.validate_tail_aero_deflection_source("actual_joint_position")
        == action_contract.ACTUAL_JOINT_TAIL_AERO_DEFLECTION
    )
    with pytest.raises(ValueError, match="action_interface"):
        action_contract.validate_action_interface("unknown")
    with pytest.raises(ValueError, match="tail_aero_deflection_source"):
        action_contract.validate_tail_aero_deflection_source("unknown")
