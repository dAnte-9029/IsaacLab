"""Read-only section wrench decomposition for prescribed ULLT motion."""

from __future__ import annotations

import torch

from .ullt_state_space import ULLTGeometry, ULLTLoads, ULLTMotion


def section_load_diagnostics(
    geometry: ULLTGeometry,
    motion: ULLTMotion,
    loads_at_unit_density: ULLTLoads,
    density_kg_m3: torch.Tensor,
    reference_flu_m: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return section FRD wrenches (B,N,6), ordered Fx,Fy,Fz,Mx,My,Mz.

    Forces are N, moments N m about the supplied CG, originally in FLU. Density
    is (B,1); input loads must have been evaluated with rho=1 kg/m^3. Positive
    moments obey the right hand rule. Circulatory/profile forces act at 1/4
    chord; added mass acts at 3/4 chord and includes its free couple.

    Only circulatory + added_mass + profile_drag are additive. The flow-parallel
    circulatory projection is a SUBCOMPONENT, not another drag force to add.
    profile_drag_per_cd is an exact derivative of the existing profile formula,
    not an assumed Cd, fitted drag, or an independently evaluated flow solution.
    """
    scale = density_kg_m3[..., None]
    sign = geometry.chord_m.new_tensor([1, -1, -1])
    r_quarter = geometry.quarter_chord_w_m - reference_flu_m
    r_three = geometry.three_quarter_chord_w_m - reference_flu_m

    def wrench(force: torch.Tensor, arm: torch.Tensor, couple: torch.Tensor | None = None) -> torch.Tensor:
        moment = torch.linalg.cross(arm, force)
        if couple is not None:
            moment = moment + couple
        return torch.cat((force * scale * sign, moment * scale * sign), -1)

    fc = loads_at_unit_density.circulatory_force_w_n
    fa = loads_at_unit_density.added_mass_force_w_n
    fp = loads_at_unit_density.profile_force_w_n
    values = {
        "circulatory": wrench(fc, r_quarter),
        "added_mass": wrench(fa, r_three, loads_at_unit_density.added_mass_couple_three_quarter_w_nm),
        "profile_drag": wrench(fp, r_quarter),
    }
    values["total"] = sum(values.values())
    flow = motion.flow_three_quarter_w_mps
    perpendicular = flow - (flow * geometry.span_w).sum(-1, keepdim=True) * geometry.span_w
    speed = torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True)
    if bool((speed <= 0).any()):
        raise ValueError("Undefined local flow direction")
    direction = perpendicular / speed
    values["circulatory_flow_parallel"] = wrench((fc * direction).sum(-1, keepdim=True) * direction, r_quarter)
    unit_drag = 0.5 * geometry.chord_m[..., None] * geometry.width_m[..., None] * speed.square() * direction
    values["profile_drag_per_cd"] = wrench(unit_drag, r_quarter)
    values["added_mass_free_couple"] = loads_at_unit_density.added_mass_couple_three_quarter_w_nm * scale * sign
    values["dynamic_pressure_pa"] = (0.5 * scale * speed.square()).squeeze(-1)
    values["effective_area_m2"] = geometry.chord_m * geometry.width_m
    values["alpha_deg"] = torch.rad2deg(loads_at_unit_density.geometric_incidence_rad)
    return values
