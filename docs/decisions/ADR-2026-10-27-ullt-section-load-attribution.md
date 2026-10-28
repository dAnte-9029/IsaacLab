# ADR: Phase/span-resolved ULLT load attribution

- Status: User-authorized diagnostic only, 2026-10-27.
- Context: A per-time maximum incidence cannot establish high-incidence area,
  load burden, or error dominance. Whole-aircraft reconstructed labels are not
  measured wing loads.

## Decision and scope

Reuse the three training logs and six windows from the fixed twist screen.
Diagnose zero twist and the already-screened `(35 deg,-0.5,0 deg)` candidate;
do not fit new parameters, select new logs, read sealed tests, or promote a model.
An optional read-only replay observer saves scored per-section loads. The force
equations, wake integration, original log/prior values and defaults stay unchanged.

Record signed geometric incidence from prescribed local 3/4-chord incident
velocity (before induced flow), root-local span station, wing side, phase,
reference CSV planform area, effective projected model area, local dynamic
pressure, three forces and three moments about the same neutral CG.
Use reference CSV area for physical area fractions; exclude the central gap.
Current projected model area is separate and not interpreted as measured fabric area.

Partition loads into circulatory, added mass (force lever arm plus free couple),
and the existing profile drag. Their signed section sums must reconstruct the
backend force and moment. Separately expose the circulatory projection parallel
to prescribed incident flow as a nested diagnostic, not an additional drag term
or uniquely identified induced drag. The current profile Cd is zero. Report
the exact existing-model derivative per unit Cd, optionally scaled by 0.01 for
readability; do not assume a real coefficient or fit one to the residual.

Use absolute-incidence bins [0,15), [15,25), [25,35), [35,45), [45,60), [60,180]
degrees and thresholds >25, >35, >45, >60 degrees. These are descriptive bins,
not universal stall boundaries. Phase bins are 10 degrees; retain separate wings,
each flight and equal-flight aggregate tables. Continuous section data remain
available so the binning does not erase time/position evidence.

Report signed mean and RMS of group resultants and absolute section-load shares.
The latter denominator sums absolute per-section contributions, preventing
near-zero net loads from creating arbitrary percentages. High-angle region and
added-mass labels overlap: they are a two-way decomposition, not independent
causal percentages. Store their intersections and exclusive incidence bands.

For each masked/component load C, report residual alignment mean((prediction-log)*C)
and RMS(prediction-log-C) with the original wake and kinematics frozen. These are
algebraic diagnostics, NOT a nonlinear rerun removing sections, a physical
correction, causal attribution, or additive shares of the squared error. Squared
error changes have cross terms and cannot be added across mechanisms.

## Alternatives, limits and reconsideration

Do not allocate the whole-aircraft residual to strips as if local force labels
existed. Do not infer missing drag or actual deformation from leftover residual.
Zero-to-prescribed-twist differences only quantify sensitivity to that assumed
kinematics. Added-mass load terms share acceleration inputs with reconstructed
labels; this does not validate free-body added-mass coupling. Tail/body forces,
wind, inertia/derivative labels, separation and deformation remain confounded.

## Validation

Check force/moment closure including free couples and frame signs, the unit-Cd
derivative against two actual backend evaluations, area-weighted statistics and
cancellation handling against independent examples, and observer/no-observer
replay parity. Preserve original artifacts and hashes. Perform one bounded
10/20-panel sensitivity check for area/load fractions before calling the apparent
spatial concentration robust. Reconsider numerical resolution or the next
physical model only after these diagnostics, without modifying defaults here.
