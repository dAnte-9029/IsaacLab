"""Physics utility modules for the flapping bot (e.g., aerodynamics)."""

from .qsm import FlappingQSMCfg, QuasiSteadyWingModel, WingQSMCfg
from .qsm_wang2016 import (
    WingGeometry,
    compute_aero_wrench,
    cp_location_rotation,
    eta_shape_piecewise,
    rotation_matrix_i_from_c,
    transform_wrench_c_to_world,
)
from .virtual_twist import VirtualTwistCfg, VirtualTwistState, solve_quasi_static_eta_tip, step_virtual_twist

__all__ = [
    "WingQSMCfg",
    "FlappingQSMCfg",
    "QuasiSteadyWingModel",
    "WingGeometry",
    "compute_aero_wrench",
    "cp_location_rotation",
    "eta_shape_piecewise",
    "rotation_matrix_i_from_c",
    "transform_wrench_c_to_world",
    "VirtualTwistCfg",
    "VirtualTwistState",
    "solve_quasi_static_eta_tip",
    "step_virtual_twist",
]
