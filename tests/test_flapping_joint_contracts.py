from __future__ import annotations

import ast
from pathlib import Path

import torch

from flapping_bot.direct.flapping_bot.startup_phase import map_symmetric_flap_coordinate_to_joint_space


STRAIGHT_FLIGHT_ENV_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "straight_flight_env.py"
)

FLAPPING_ENV_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "flapping_env.py"
)


def _class_assignments(path: Path, class_name: str) -> dict[str, object]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            assignments: dict[str, object] = {}
            for stmt in node.body:
                if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                    continue
                try:
                    assignments[stmt.target.id] = ast.literal_eval(stmt.value)
                except Exception:
                    continue
            return assignments
    raise AssertionError(f"Class {class_name} not found in {path}")


def _method_body(path: Path, method_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker = f"def {method_name}"
    start = text.index(marker)
    next_def = text.find("\n    def ", start + len(marker))
    if next_def == -1:
        next_def = len(text)
    return text[start:next_def]


def test_straight_flight_env_declares_rudder_in_controlled_joints() -> None:
    assignments = _class_assignments(STRAIGHT_FLIGHT_ENV_FILE, "FlappingBotStraightFlightEnvCfg")
    assert assignments["controlled_joints"] == ("left_wing", "right_wing", "rudder", "left_tail", "right_tail")


def test_legacy_flapping_env_declares_rudder_in_controlled_joints() -> None:
    assignments = _class_assignments(FLAPPING_ENV_FILE, "FlappingBotEnvCfg")
    assert assignments["controlled_joints"] == ("left_wing", "right_wing", "rudder", "left_tail", "right_tail")


def test_straight_flight_env_writes_rudder_target_to_joint_buffer() -> None:
    body = _method_body(STRAIGHT_FLIGHT_ENV_FILE, "_apply_action")
    assert "jt[:, self._IDX_RUDDER] = self._rudder_cmd" in body


def test_straight_flight_env_disables_virtual_roll_surrogate_by_default() -> None:
    assignments = _class_assignments(STRAIGHT_FLIGHT_ENV_FILE, "FlappingBotStraightFlightEnvCfg")
    assert assignments["virtual_roll_moment_gain"] == 0.0
    assert assignments["virtual_roll_moment_damping"] == 0.0


def test_straight_flight_env_reset_initializes_rudder_joint() -> None:
    body = _method_body(STRAIGHT_FLIGHT_ENV_FILE, "_reset_idx")
    assert "jpos[:, self._IDX_RUDDER] = rudder0" in body


def test_symmetric_flap_coordinate_maps_to_opposite_urdf_joint_signs() -> None:
    flap_position = torch.tensor([0.3, -0.2], dtype=torch.float64)
    flap_velocity = torch.tensor([-1.4, 0.8], dtype=torch.float64)
    left_position, right_position, left_velocity, right_velocity = map_symmetric_flap_coordinate_to_joint_space(
        flap_position_rad=flap_position,
        flap_velocity_rad_s=flap_velocity,
        left_joint_mid_rad=0.0,
        right_joint_mid_rad=0.0,
    )
    torch.testing.assert_close(left_position, flap_position, atol=0.0, rtol=0.0)
    torch.testing.assert_close(right_position, -flap_position, atol=0.0, rtol=0.0)
    torch.testing.assert_close(left_velocity, flap_velocity, atol=0.0, rtol=0.0)
    torch.testing.assert_close(right_velocity, -flap_velocity, atol=0.0, rtol=0.0)
