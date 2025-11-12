"""Physics utility modules for the flapping bot (e.g., aerodynamics)."""

from .qsm import FlappingQSMCfg, QuasiSteadyWingModel, WingQSMCfg

__all__ = ["WingQSMCfg", "FlappingQSMCfg", "QuasiSteadyWingModel"]
