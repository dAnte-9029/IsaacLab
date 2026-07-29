"""Asset configuration entry points for the flapping bot extension."""

from .flapping_bot_cfg import FlappingBotCfg
from .ideal_coupled_drive import (
    IDEAL_COUPLED_WING_DRIVE,
    KINEMATIC_WING_OVERRIDE,
    WING_DRIVE_VARIANTS,
    IdealCoupledFlappingBotCfg,
    apply_hard_opposed_wing_mimic,
    validate_wing_drive_variant,
)

__all__ = [
    "FlappingBotCfg",
    "IdealCoupledFlappingBotCfg",
    "KINEMATIC_WING_OVERRIDE",
    "IDEAL_COUPLED_WING_DRIVE",
    "WING_DRIVE_VARIANTS",
    "validate_wing_drive_variant",
    "apply_hard_opposed_wing_mimic",
]
