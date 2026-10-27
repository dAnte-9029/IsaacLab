# ADR: Small prescribed-twist ULLT flight screen

- Status: Accepted by user request to try the existing law on a few logs and parameter combinations.
- Date: 2026-10-27
- Supersedes only the deferred-twist/no-tuning restriction for this offline screen.

## Context and decision

Apply the existing Quadratic-2 kinematic prior to ULLT-4:
`theta=A_tip*((1-kappa)*s+kappa*s^2)*cos(phi_C-psi)`.
This equals `-A_tip*g(s)*sin(phi_D-psi)` with `phi_D=phi_C-pi/2`.
Positive theta pitches the chord nose-up about negative neutral LE span.
No additional uniform whole-wing pitch, static offset, stiffness or damping is added.

The straight leading-edge tube stays on its URDF-root/flapping trajectory.
Quarter-chord endpoints and control points rotate around that tube at their own
root-local stations. Adjacent panel endpoints remain shared. Reproject the
physical chord onto the plane perpendicular to the deformed lifting line,
preserving the original swept-section convention. Analytic first and second
derivatives include this projection and the moving orthonormal section frame.
Compose twist, flapping and body motion including transport/Coriolis terms.
Moment remains about the same logged neutral CG, with the existing quarter-chord
circulatory and three-quarter-chord added-mass force/free-couple closure.
This varying projected-section representation is an approximation, not a fabric
membrane or structural deformation solve; changing section dimensions may need
a more complete lifting-surface/added-mass treatment later.

Use three train flights selected by low/median/high input frequency among
eligible flights, independent of wrench residuals. Retain the original two
12-second input-selected windows per flight and 2-second wake warmup.
Six fixed candidates `(A_deg,kappa,psi_deg)`:
`(20,0,-25), (35,-0.5,-25), (40,0,-25), (35,-0.5,0),
(35,-0.5,-45), (35,0,-25)`, plus zero-twist ULLT and frozen DeLaurier.
No validation/test tuning or test access. The existing registry loader verifies
and loads train/validation inputs; only the selected train logs are replayed/scored.
Use identical finite rows across candidates, report any lost coverage and failures.
Retain per-flight wrench RMSE, complete-cycle mean/amplitude/phase, transitions,
local incidence, exact parameter/configuration/source hashes and plots.

## Alternatives and consequences

- Merely subtract twist from angle of attack: rejected because moving-point
  acceleration, moment arms and added-mass terms would be inconsistent.
- Fit a passive structural model: deferred; structural parameters are unmeasured.
- Change the formal model/default parameters: outside scope.

Results are an exploratory comparison of wing predictions to reconstructed
whole-aircraft loads, not identification of measured wing deformation, an
independent validation score, or evidence of generalization to other flights.
Original ULLT, DeLaurier, RL defaults and historical artifacts remain available.

## Validation and reconsideration

Require exact zero-amplitude replay parity, independent LE geometry checks,
time finite differences for point velocity/acceleration and frame rotation,
mirror load symmetry and existing ULLT/replay regressions. Recheck one promising
candidate at doubled span resolution. Revisit with measured twist, large
resolution sensitivity, reverse-flow rejection, or independent flight evidence.
