from __future__ import annotations

from pathlib import Path


STRAIGHT_FLIGHT_ENV = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "straight_flight_env.py"
)


def test_delaurier_theta_a_uses_frd_positive_down_alpha_convention() -> None:
    source = STRAIGHT_FLIGHT_ENV.read_text(encoding="utf-8")

    assert "theta_a_env = torch.atan2(v_air_b[:, 2], vx_b)" in source
    assert "theta_a_env = torch.atan2(-v_air_b[:, 2], vx_b)" not in source
