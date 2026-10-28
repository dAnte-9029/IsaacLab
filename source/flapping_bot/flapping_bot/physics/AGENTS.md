# Aerodynamics and Plant Physics

For new force/moment outputs, make frame, SI units, sign, application/reference points, tensor shape, and modeling assumptions clear. When changing sectional loads, consider the corresponding resultant and moment consistently, including moment arms about the stated reference.

Treat measured or published quantities, analytical priors, tuned coefficients, closure assumptions, and numerical stabilizations as different kinds of evidence. A physically plausible trajectory or smaller RMSE alone does not identify aerodynamic coefficients.

The ULLT, DeLaurier, and other backends may be redesigned or replaced for a requested experiment; a pre-existing ADR or separate authorization phase is **not** a prerequisite for exploratory research. Do not secretly overwrite the only frozen comparison baseline. Explicitly state when a theory's geometry, incidence, Reynolds-number, or nonlinearity assumptions may not hold for the bird-scale wing.

Focus numerical checks on actual risks of the change: finite values, force/moment consistency, phase/sign/frame transforms, convergence, or handling near singular conditions where relevant. Do not mandate every analytic, symmetry, regression, and precision test for every modification. Parity with the old model applies to behavior-preserving work, not a genuinely new physical model.

Physics calculations must not silently depend on controller rewards or retune aircraft parameters to make a task succeed. Coupled model/control research is allowed when the coupling is explicit in the approved design.
