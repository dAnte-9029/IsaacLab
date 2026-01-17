"""Physics utility modules for the flapping bot (e.g., aerodynamics)."""

from .qsm import FlappingQSMCfg, QuasiSteadyWingModel, WingQSMCfg
from .qsm_wang2016 import (
    WingGeometry,
    compute_aero_wrench,
    compute_aero_wrench_from_omega_alpha,
    cp_location_rotation,
    eta_shape_piecewise,
    rotation_matrix_i_from_c,
    transform_wrench_c_to_world,
)
from .qsm_delaurier1993 import DeLaurierParams, compute_aero_wrench_delaurier1993
from .virtual_twist import VirtualTwistCfg, VirtualTwistState, solve_quasi_static_eta_tip, step_virtual_twist

__all__ = [
    "WingQSMCfg",
    "FlappingQSMCfg",
    "QuasiSteadyWingModel",
    "WingGeometry",
    "compute_aero_wrench",
    "compute_aero_wrench_from_omega_alpha",
    "cp_location_rotation",
    "eta_shape_piecewise",
    "rotation_matrix_i_from_c",
    "transform_wrench_c_to_world",
    "DeLaurierParams",
    "compute_aero_wrench_delaurier1993",
    "VirtualTwistCfg",
    "VirtualTwistState",
    "solve_quasi_static_eta_tip",
    "step_virtual_twist",
]
