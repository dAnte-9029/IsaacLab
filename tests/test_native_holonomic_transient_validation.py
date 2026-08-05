from __future__ import annotations

import pytest

from flapping_bot.analysis.native_holonomic_transient_validation import (
    native_transient_frequency_targets_hz,
)


def test_transient_frequency_profiles_have_explicit_endpoints() -> None:
    assert native_transient_frequency_targets_hz(0.0) == pytest.approx((2.0, 2.0, 2.0))
    assert native_transient_frequency_targets_hz(0.25) == pytest.approx((2.75, 5.0, 3.5))
    assert native_transient_frequency_targets_hz(0.5) == pytest.approx((3.5, 5.0, 5.0))
    assert native_transient_frequency_targets_hz(1.0) == pytest.approx((5.0, 5.0, 2.0))
    assert native_transient_frequency_targets_hz(1.25) == pytest.approx((5.0, 5.0, 3.5))


def test_transient_frequency_profiles_reject_negative_time() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        native_transient_frequency_targets_hz(-1.0e-6)
