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
from .qsm_delaurier1993 import (
    DeLaurierParams,
    DeLaurierStripLoads,
    DeLaurierStripWrench,
    compute_aero_wrench_delaurier1993,
    compute_delaurier_strip_loads,
    integrate_delaurier_strip_wrench,
    transform_wang_wrench_to_link,
    translate_wrench_moment,
)
from .delaurier_twist import (
    DELAURIER_DYNAMIC_TWIST_MODES,
    DeLaurierTwistKinematics,
    compute_delaurier_dynamic_twist,
    resolve_delaurier_phase,
    validate_delaurier_dynamic_twist_mode,
)
from .tail_aero import TailAeroCfg, TailAeroModel, TailSurfaceCfg
from .tail_geometry import PlaceholderValue, TailGeometry, TailSurfaceGeometry, load_tail_geometry_from_urdf
from .wing_equivalent_ac import compute_area_weighted_quarter_chord_link_points
from .wing_geom_csv import WingGeomCsvInfo, build_wing_geometry_from_csv, infer_span_from_x_mid, load_wing_geom_csv
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
    "DeLaurierStripLoads",
    "DeLaurierStripWrench",
    "compute_aero_wrench_delaurier1993",
    "compute_delaurier_strip_loads",
    "integrate_delaurier_strip_wrench",
    "transform_wang_wrench_to_link",
    "translate_wrench_moment",
    "DELAURIER_DYNAMIC_TWIST_MODES",
    "DeLaurierTwistKinematics",
    "compute_delaurier_dynamic_twist",
    "resolve_delaurier_phase",
    "validate_delaurier_dynamic_twist_mode",
    "TailSurfaceCfg",
    "TailAeroCfg",
    "TailAeroModel",
    "PlaceholderValue",
    "TailSurfaceGeometry",
    "TailGeometry",
    "load_tail_geometry_from_urdf",
    "compute_area_weighted_quarter_chord_link_points",
    "WingGeomCsvInfo",
    "load_wing_geom_csv",
    "infer_span_from_x_mid",
    "build_wing_geometry_from_csv",
    "VirtualTwistCfg",
    "VirtualTwistState",
    "solve_quasi_static_eta_tip",
    "step_virtual_twist",
]
