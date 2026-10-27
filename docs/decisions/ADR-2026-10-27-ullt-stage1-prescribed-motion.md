# ADR: ULLT stage 1 with prescribed motion

- Status: Accepted for stage 1 by the user's instruction to implement the proposed first stage
- Date: 2026-10-27
- Depends on: `ADR-2026-10-27-ullt-urdf-root-local-csv-geometry.md`

## Context and decision

Implement a standalone, explicitly selected Izraelevitz 2017 backend under
project physics, plus reproducible prescribed-motion validation. Preserve all
existing environment defaults and DeLaurier code. No RL or free-body PhysX
coupling is part of this stage.

Use thin flat attached-flow sections, lift slope 2*pi/rad, zero camber/zero-lift
offset, zero profile drag for paper reproduction, and the paper's flat-plate
added-mass matrix with anti-Munk correction. These are analytical priors, not
identified fabric-wing properties. Added-mass aspect-ratio factors are explicit
inputs (paper examples: 0.85 at AR=3, 0.95 at AR=6); do not fit them to flight.

Provide 1-state, 4-state (Section II.C, without II.D correction), and 6-state
models, plus a quasi-steady reference. Six-state correction is initially gated
to parallel, coplanar, straight unswept lifting lines; reject unsupported
geometry explicitly. Prescribed flapping/nonplanar cases use 1/4 states.
This gate is an implementation boundary, not a claim that all other geometry
is outside the paper's method. Track its extension as future work.

State is explicit per environment and section. Evaluate loads without advancing
state; integrate with stage-consistent prescribed inputs and RK4 substeps.
Keep zero-wake and steady-wake initialization distinct. Reject vanishing or
reversed chordwise flow instead of hiding it behind the old airspeed clamp.

## Frames and force closure

Use a common inertial frame, SI units and explicit batch/section dimensions.
Section chord-forward, span-positive and upward-normal axes form a right-handed
frame. Span-positive follows the lifting line from right to left for a FLU wing.
Incident flow is air velocity minus section velocity. Positive circulation is
defined to generate positive-normal lift; its bound-vortex orientation is
opposite span-positive. Thus lift direction is span cross incident total flow.

Circulatory loads act at quarter chord. Added mass uses the paper's
three-quarter-chord origin, with local axes chord-aft, negative span, normal.
Transform its complete force/couple to the declared inertial moment reference.
Distinguish fixed-frame acceleration from the derivative of local velocity
components. Do not add DeLaurier apparent-mass or Oswald drag on top.

## Reproduction conventions and manuscript ambiguities

Use Table 1 coefficients as printed, including rounded circulation coefficients;
normalize their steady value using `response(0)=1+sum(A)`. Share the first two
circulation poles with lift using Eq. (22).

In the supplied accepted manuscript Eq. (28) repeats the `w1` state index;
implement the second independent pole as `w2`. The sign before the convolution
in Eq. (68) is inconsistent with integration by parts of Eq. (67). Use the
positive convolution of the response derivative, consistent with Eqs. (27),
(28), (69), and exact steady cancellation of W0/W1/W2. Verify independently.

Appendix B does not specify the numerical fit grid/weights for C1/C2. Record
a uniform least-squares fit over dimensionless travel 0--15, 301 samples,
using the analytic half-loop kernel. This is a numerical reproduction choice;
report fit residuals and do not claim the authors' exact unpublished table.

## Alternatives and consequences

- Mutate the live environment immediately: deferred until independent load
  tests and free-body added-mass coupling are validated.
- Treat geometry-specific six-state correction as universally valid: rejected.
- Tune drag or twist until trajectories look plausible: rejected.

The selectable API and CLI are independent physics entry points, not a new
registered RL environment. Paper-scenario generation, analytical verification,
and agreement with published UVLM/experiment curves are separate evidence.
Do not claim the latter without an independent reference dataset.

## Validation and reconsideration

Test analytic 2D step response, independent harmonic response, steady finite-wing
limits, vortex signs, added mass, moment translation, rigid-frame covariance,
batch reset, dtype/device behavior, convergence and unsupported-input rejection.
Run existing DeLaurier regressions. Record limitations and promotion gates in
the stage handoff. Revisit the model with force data, structural measurements,
new geometry, or explicit authorization for free-body integration.
