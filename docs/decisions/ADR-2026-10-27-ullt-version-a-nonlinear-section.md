# ADR: ULLT version A, nonlinear equivalent-normalwash section closure

- Status: Explicitly authorized by the user for implementation and a bounded offline experiment.
- Scope: Standalone prescribed-motion ULLT; no RL/default DeLaurier/assets changes.

## Sources and coefficient provenance

Local PDFs under `docs/papers/` were read directly:

- Izraelevitz 2017, DOI 10.2514/1.J055144: original state-space circulation/lift
  kernels and normalwash formulation; retained, not replaced by a new wake model.
- Sugar-Gabor 2018, DOI 10.1017/aer.2018.57, Eqs. (10)--(12): solve a nonlinear
  circulation residual using section aerodynamics and an analytic Jacobian.
  This version does not reproduce its vortex-ring wake/time-discretization equations.
- Wang, Goosen & van Keulen 2016, DOI 10.1017/jfm.2016.413, Eqs. (2.6), (2.9),
  (2.10): separated flat-plate angular shapes Cl=A sin(2 alpha),
  Cd=2 A sin²(alpha). Only these shapes are borrowed. Its whole-wing/finite-AR
  amplitude formula is not reused inside an already coupled lifting line.
- Bakhtiari et al., DOI 10.1177/0954410018771175 (local PDF copyright/acceptance
  2018; cited by the user as 2019), Ruiz et al. 2022, and Yang et al. 2025:
  flexible-wing/high-angle and LEV effects motivate limits, not a claim that a
  static polar reproduces their separated-flow or vortex models. Yang's explicit
  PLEV and Ruiz's CFD-derived memory effects are absent here.

Attached Cl=Cla sin(alpha), with Cla=the existing configurable lift slope
(default 2*pi/rad), exactly represents the existing normalwash law. It has the
same small-angle slope as thin-airfoil theory; using Cla*alpha at finite angle
would silently change the current model. Blend attached and separated Cl using
quintic smoothstep in absolute incidence. Cd=existing Cd0 plus the same blend
times 2 A sin²(alpha). The polar is symmetric and C2, and Cd is nonnegative.

Two fixed, **unmeasured** hypotheses are selected before the flight comparison:

| Name | Blend starts | Blend ends | A | Normal-force scale 2A |
|---|---:|---:|---:|---:|
| early | 15 deg | 30 deg | 1.0 | 2.0 |
| late | 20 deg | 40 deg | 1.2 | 2.4 |

Transition limits locate a hypothetical attached-to-separated change; they are
not identified stall angles. A is the separated branch's peak Cl at 45 deg,
not necessarily the blended curve's global peak. Their numerical values are
engineering sensitivity priors, not coefficients extracted from the real fabric
wing, NACA data, or fitted to these logs. The finite-AR effects are left to ULLT.
No additional rotational lift, leading-edge-vortex lift or added mass is added.

## Nonlinear residual and state consistency

At fixed wake state x, let Veff=Vincident+W Gamma, projected onto the local
chord/normal plane. Define u=-Veff.chord, n=Veff.normal, alpha=atan2(n,u),
V=sqrt(u²+n²), and **ue=V Cl(alpha)/Cla**. Below the blend start, ue=n exactly.

Solve per section:

`R(Gamma)=2 Gamma/(c Cla) - gamma0*ue(Gamma) - history_gamma(x) = 0`.

The implicit residual is solved in `_solve`, not by attenuating final loads.
Use the original linear solution and the steady circulation implied by the supplied
wake state as deterministic predictors; choose the smaller residual. The latter
preserves a steady root already encoded in the state. Use an analytic Newton
Jacobian, squared-L2 residual merit and per-environment backtracking; accept only
positive effective chordwise flow. Up to three fixed numerical restarts use the
no-induced-flow nonlinear prediction, half the linear prediction, then zero.
These initializations are independent of log load errors and do not alter the
polar, residual equation or acceptance tolerance. Each attempt has at most 16
updates and 10 backtracking trials; failure after this bounded budget raises.
Setting `nonlinear_max_restarts=0` disables restarts. Convergence uses velocity residual
units with explicit absolute/relative tolerances and a 32*machine-epsilon speed
floor for floating-point feasibility. Iteration/backtracking limits are finite;
failure raises an exception with residuals and iteration counts. There is no
fallback to linear, no clipping of failed circulation and no fitted damping.
Post-stall roots may not be unique; convergence alone is not physical validation.
The synthetic AR=3, 40-deg steady case exposed root sensitivity (including dtype
sensitivity). Successful residuals do not certify a unique smooth spanwise root;
no span smoothing or artificial circulation clipping is introduced. Explicit
failure is still possible, particularly for quasi-steady transition-region roots.

Use **the same ue** in circulation response, lift response and all main state
derivatives. The induced velocity remains W Gamma. The extra two-dimensional
wake velocity uses `lift_response-ue`, so instantaneous polar attenuation is not
counted again as a fictitious induced velocity. Steady-state initialization also
solves the nonlinear quasi-steady closure before initializing the memory states.

The physical effective normalwash and the equivalent forcing are separately
reported. Diagnostics expose effective angle, static Cl/Cd, speed ratio, residual,
tolerance, iteration count and optional accumulated statistics across RK stages.

## Engineering approximations and force/moment closure

Version A deliberately retains the original prescribed incident perpendicular
speed U in the convective rate 2U/c and the lift prefactor. Thus steady
circulatory magnitude is `0.5*rho*c*U*V*Cl`, rather than replacing the original
model by `0.5*rho*c*V²*Cl`. This choice preserves the original low-angle backend,
including induced flow. Report V/U to quantify this mixed-speed approximation.
It is not a fully rederived nonlinear unsteady lifting-line theory.

Retain the original baseline Cd0 force. Add only the blended separated Cd
increment, with magnitude `0.5*rho*c*width*U*V*(Cd-Cd0)`, directed along the
physical effective incident flow in the section plane. This matches the mixed
speed scale of the circulatory closure. Do not add a separate induced-drag term:
circulatory force already follows the induced/unsteady flow direction.

Both circulatory and profile loads retain the existing quarter-chord application
point. Added mass, its 3/4-chord free couple and its kinematics are unchanged.
Separated center-of-pressure migration and new section pitching coefficients are
not modeled; moment results are therefore especially provisional. Static polar
bounds do not prohibit transient overshoot from the inherited linear kernels.
No separation-history state, LEV shedding/wake capture or dynamic stall is solved.

## Configuration, experiment and validation

Explicit `section_model='nonlinear_flat_plate'` enables this path. Default remains
`linear`; quasi_steady/one_state/four_state are supported. Nonlinear six_state is
rejected without changing its original linear geometry restriction.

Keep twist fixed at (35 deg,-0.5,0 deg). Reuse the exact three training logs and
windows, comparing the two hypotheses against twisted linear, original untwisted
linear and frozen DeLaurier. Do not fit twist, drag or phase offsets, access test,
run PPO or broaden the search. Measure equal-row per-flight RMSE, complete-cycle
mean, signed peaks/peak-to-peak and first-harmonic phase, section maps and runtime.

Require disabled/low-angle regression, polar smoothness/symmetry/bounds,
independent Jacobian checks, finite batch solves, explicit failure tests,
steady-state consistency, force/moment closure and all relevant existing tests.
Preserve all prior uncommitted diagnostics. Reconsider assumptions with measured
polars/deformation, solver multiplicity/failures, large speed-ratio differences,
or independent evidence that section moments/dynamic stall dominate.

## Alternatives and consequences

- Final-force attenuation was rejected because it leaves the original circulation,
  induction and wake forcing inconsistent with the new section law.
- Replacing every velocity factor by Veff would change the attached-flow baseline;
  retain the mixed U/V convention and report its size instead.
- A new LEV/dynamic-stall state or measured airfoil polar is outside version A.
- A global uniqueness guarantee, continuation in prescribed motion, or an explicit
  root-history state may be needed if branch switching dominates. Bounded numerical
  restarts are not a physical model of separation hysteresis.
- The replay's `ullt_circulatory_*` columns now exclude profile drag explicitly;
  `ullt_profile_drag_*` is added. Archived zero-Cd outputs remain unchanged.
