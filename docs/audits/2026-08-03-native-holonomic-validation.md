# Native holonomic measured-wing validation

## Scope

This audit validates the project-local `FlappingWingTrajectoryJoint` after its
integration with the measured-wing PhysX plant. It covers fixed-root trajectory
tracking, independent environments, frequency transitions, DeLaurier per-wing
loads, solver/time-step sensitivity, free-body momentum behavior, reset, CPU/GPU
execution and extension lifecycle. It does not validate the existing flight
controller or promote the plant to the repository default.

The acceptance gate for the mechanism coordinates is

\[
\max |q_L-q_{\mathrm{ref}}|<0.1^\circ,
\qquad
\max |(q_L-q_{L,\mathrm{mid}})+(q_R-q_{R,\mathrm{mid}})|<0.1^\circ.
\]

## Validation-boundary correction

The first aerodynamic sweep manually wrote an 8 m/s wind after reset while
leaving `wind_enabled=False`. `_pre_physics_step()` correctly cleared that wind,
so DeLaurier used its 0.5 m/s minimum-speed clamp. Those aerodynamic numbers,
including a reported 306 N startup peak, are invalid and are not used below.

The runner now configures deterministic wind through
`wind_enabled=True` and `wind_xy_mps=(-airspeed_mps, 0)`. The corrected test also
sets wind in the configuration before environment construction. The result
schema was advanced to `native_holonomic_frequency_v3`.

A second validation-harness defect was then isolated with a known 1 N force.
The runner had set `retain_accelerations=True`, which carries applied forces
between physics steps. Since the environment also submitted a new aerodynamic
wrench every step, the PhysX load grew as `F, 2F, 3F, ...`. With
`retain_accelerations=False`, the 0.2 s known-load test recovered a maximum
linear impulse residual of `1.89e-7 kg m/s` and maximum angular impulse
residual of `9.73e-5 kg m^2/s`. All earlier aerodynamic matrices using the
retained-force configuration are superseded by the results below.

## Fixed-root CPU results

The no-aerodynamics two-environment gate at 2 and 5 Hz and 1/480 s passed after
an explicit reset. The recovered mean frequencies were 2.0 and 4.9999995 Hz,
the maximum trajectory error was 0.06291 degree and the maximum left/right
synchronization error was 1.71e-6 degree.

The transition sequence 0 -> 2 -> 5 -> 3 -> 0 Hz preserves the unwrapped phase
without jumps. At 1/480 s its maximum trajectory error was 0.06249 degree and
its maximum synchronization error was 1.71e-6 degree.

With corrected 8 m/s relative flow, non-retained per-step wrenches and the
final 16 position / 4 velocity iterations, the time-step matrix was:

| Physics step | 2 Hz | 3 Hz | 4 Hz | 5 Hz | Gate |
|---|---:|---:|---:|---:|---|
| 1/240 s | 0.04006 deg | 0.09004 deg | 0.15992 deg | 0.24970 deg | fail at 4/5 Hz |
| 1/480 s | 0.01002 deg | 0.02252 deg | 0.04001 deg | 0.06249 deg | pass |
| 1/1000 s | 0.00232 deg | 0.00519 deg | 0.00923 deg | 0.01440 deg | pass |

The left/right synchronization error remained below `1e-6 deg` in every case.
The 1/240 s result is sufficient at 2/3 Hz but not at 4/5 Hz. The 1/480 s step
is the coarsest tested step that passes the complete frequency gate.

The earlier retained-force matrix appeared to require 32 position iterations,
but that solver ablation is invalid because it compensated for a force that
grew every step. Under corrected force semantics, 5 Hz at 1/480 s passes with
16/4 at `0.06250 deg`, indistinguishable from 32/4 at `0.06293 deg`. The native
variant therefore uses the lower-cost 16/4 configuration; other plant variants
retain their settings.

The final 16/4, 1/480 s, 8 m/s matrix passed:

| Frequency | Maximum trajectory error | Maximum synchronization error | Steady peak single-wing force |
|---:|---:|---:|---:|
| 2 Hz | 0.01002 deg | 8.54e-7 deg | 9.7700 N |
| 3 Hz | 0.02252 deg | 0 deg | 12.9446 N |
| 4 Hz | 0.04001 deg | 1.07e-7 deg | 16.2167 N |
| 5 Hz | 0.06249 deg | 1.71e-6 deg | 19.6947 N |

## Free-body no-external-force gate

Gravity and aerodynamics were disabled, the base was released and two
environments ramped independently to 2 and 5 Hz. The ideal mechanism is an
internal constraint, so total system linear and angular momentum should remain
constant up to integration error.

| Physics step | Max tracking | Max sync | Max COM drift | Max linear-momentum drift | Max angular-momentum drift |
|---|---:|---:|---:|---:|---:|
| 1/480 s | 0.05244 deg | 3.42e-6 deg | 1.14e-5 m | 3.76e-6 kg m/s | 2.35e-4 kg m^2/s |
| 1/1000 s | 0.01381 deg | 1.71e-6 deg | 1.26e-5 m | 3.03e-6 kg m/s | 1.68e-4 kg m^2/s |

Both cases remained finite and below the 0.1 degree mechanism gates. The
angular-momentum residual decreases with the smaller step and remains below the
current 5e-4 kg m^2/s numerical gate.

## Free-body aerodynamic closure

The base was released with gravity, tail aerodynamics and fuselage drag
disabled. A deterministic 8 m/s relative flow was applied at 2 and 5 Hz for
1 s. Acceptance required trajectory and synchronization errors below
`0.1 deg`, relative linear impulse-momentum error below `1e-3`, and relative
angular impulse-momentum error below 5 percent.

| Frequency | Max tracking | Max sync | Max wing force | Max root speed | Linear closure | Angular closure |
|---:|---:|---:|---:|---:|---:|---:|
| 2 Hz | 0.02094 deg | 4.27e-7 deg | 9.6634 N | 5.0876 m/s | 3.80e-6 | 2.87% |
| 5 Hz | 0.05285 deg | 1.71e-6 deg | 26.7184 N | 7.4708 m/s | 3.81e-6 | 2.46% |

The maximum absolute linear residuals were `1.64e-5` and
`2.32e-5 kg m/s`; maximum angular residuals were `0.00965` and
`0.01415 kg m^2/s`. A 1/1000 s comparison reduced the 2/5 Hz trajectory errors
to `0.00487/0.01229 deg` but left the relative angular closure near 2.4--2.9
percent. This supports treating the remaining angular discrepancy as the
observed constrained-articulation closure floor over the tested steps rather
than a divergent time-step error.

## Uncontrolled-flight smoke

A separate 1 s smoke used a released base, gravity, still air, 8 m/s initial
forward speed, 4 Hz flapping, neutral control surfaces, and the wing and tail
aerodynamic plant. Fuselage drag remained at the project default of zero rather
than introducing an unmeasured coefficient. This is a numerical boundedness
check, not an attitude-stability or controller test. It passed with maximum trajectory
error `0.03336 deg`, synchronization error `1.71e-6 deg`, root linear speed
`11.2767 m/s`, root angular speed `3.5236 rad/s`, single-wing force `10.9078 N`,
and maximum one-step root-velocity change `0.19111`.

## GPU and lifecycle findings

The current custom `PxConstraint` cannot be added to a PhysX scene configured
for direct GPU access. PhysX reports that a non-GPU-compatible constraint is
not allowed and recommends a D6 joint. The environment now rejects
`native_holonomic_drive` with a CUDA simulation device before scene creation.
The native plant is therefore CPU-only; GPU training or large direct-GPU
rollouts cannot use this implementation.

Reset with a nonzero 2 Hz state, two independent target frequencies, repeated
phase transitions and fresh-process startup/shutdown pass. An attempted second
full environment construction in the same Kit process after detach/close hung
during scene creation and was stopped. This does not yet isolate the custom
extension from Isaac Lab's process-global simulation-context lifecycle, so
same-process stage reload remains unresolved.

## Conclusion and next gate

The CPU native holonomic mechanism is accepted for the explicit measured-wing
plant at 1/480 s with 16/4 solver iterations. It is closer to the intended
mechanism than the effort-level alternatives because PhysX solves trajectory
reaction, wing inertia and external wing load in one constraint solve while
the wing links retain measured mass and inertia.

The fixed-root, free-body momentum, free-body aerodynamic closure and bounded
uncontrolled-flight subsystem gates are complete. This is numerical plant
integration evidence, not aerodynamic-model validation, controller evaluation
or real-flight validation. Default promotion now requires an explicit runtime
decision accepting the CPU-only custom constraint, disabled physics
replication and unresolved same-process stage reload. A controller baseline
must be rebuilt only after that plant-selection decision.

For command-line automation, read `all_cases_accepted` from the emitted JSON or
invoke the worker with the activated Conda Python directly. The repository
`isaaclab.sh -p` wrapper currently discards the child Python exit status after
the command returns, even though the worker itself returns 2 on a failed gate.
