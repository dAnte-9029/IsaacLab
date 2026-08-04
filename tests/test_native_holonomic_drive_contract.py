"""Pure contract checks for the native holonomic wing-drive variant."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from flapping_bot.physics import NATIVE_HOLONOMIC_PER_WING_LINK, validate_wing_aero_coupling_mode

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASSET_PATH = _REPO_ROOT / "source/flapping_bot/flapping_bot/assets/ideal_coupled_drive.py"
_HELPER_PATH = _REPO_ROOT / "source/flapping_bot/flapping_bot/assets/native_holonomic_drive.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("native_holonomic_drive_contract", _HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_holonomic_variant_is_explicit_and_passive() -> None:
    tree = ast.parse(_ASSET_PATH.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert ast.literal_eval(assignments["NATIVE_HOLONOMIC_WING_DRIVE"]) == "native_holonomic_drive"
    wing_variants = assignments["WING_DRIVE_VARIANTS"]
    assert isinstance(wing_variants, ast.Call)
    assert any(
        isinstance(node, ast.Name) and node.id == "NATIVE_HOLONOMIC_WING_DRIVE"
        for node in ast.walk(wing_variants)
    )
    native_cfg = assignments["NativeHolonomicCoupledFlappingBotCfg"]
    assert isinstance(native_cfg, ast.Call)
    assert isinstance(native_cfg.func, ast.Attribute)
    assert isinstance(native_cfg.func.value, ast.Name)
    assert native_cfg.func.value.id == "PrescribedCoupledFlappingBotCfg"
    assert ast.literal_eval(assignments["NATIVE_HOLONOMIC_SOLVER_POSITION_ITERATIONS"]) == 16
    assert ast.literal_eval(assignments["NATIVE_HOLONOMIC_SOLVER_VELOCITY_ITERATIONS"]) == 4
    assert validate_wing_aero_coupling_mode(NATIVE_HOLONOMIC_PER_WING_LINK) == NATIVE_HOLONOMIC_PER_WING_LINK


def test_native_extension_launch_contract_resolves_in_repository() -> None:
    helper = _load_helper()
    extension_root = helper.native_holonomic_extension_root()
    assert extension_root.is_dir()
    assert (extension_root / "config/extension.toml").is_file()
    kit_args = helper.native_holonomic_kit_args()
    assert str(extension_root.parent) in kit_args
    assert f"--enable {helper.NATIVE_HOLONOMIC_EXTENSION_ID}" in kit_args


def test_native_joint_paths_are_external_to_replicated_articulation() -> None:
    helper = _load_helper()
    paths = helper.cloned_native_holonomic_joint_paths(
        ["/World/envs/env_0", "/World/envs/env_1"]
    )
    assert paths == [
        "/World/flapping_bot_constraints/env_0/flapping_wing_trajectory",
        "/World/flapping_bot_constraints/env_1/flapping_wing_trajectory",
    ]
