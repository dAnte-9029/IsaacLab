# PhysX measured-wing inertial validation against Amini SFWM

Date: 2026-07-29
Branch: `feat/physx-wing-multibody`
Input commit: `2c388bf6`

## Objective

Test whether the measured-wing PhysX articulation reproduces the dominant
inertial coupling predicted by Amini et al. (2020), without adding the
analytical term to the simulated plant. The benchmark disables gravity and
aerodynamics and commands a symmetric 30 degree sine stroke at 2, 3, 4 and
5 Hz.

This is a numerical-physics validation under the paper's symmetric
longitudinal assumptions. It is not motor identification, aerodynamic
validation or real-flight validation.

## Analytical reference and frame mapping

`flapping_bot.physics.sfwm_inertial_reference` implements Amini Eqs. (16)-(18):

```text
a_fz = rho*l_y*(gamma_ddot*cos(gamma) - gamma_dot^2*sin(gamma))

alpha_fy = -2*l_y*(l_x + r_p_x)*m_w/I
           *(gamma_ddot*cos(gamma) - gamma_dot^2*sin(gamma))
```

The analytical reference is evaluated from the actual PhysX joint position,
velocity and acceleration. It is never applied as a force or acceleration.
The paper uses body FRD; the repository uses body FLU, so vertical and pitch
components both change sign. The paper point `O` is derived as the body-fixed
point coincident with the measured body-plus-two-wing COM at the neutral
dihedral:

| Quantity | Value |
|---|---:|
| Total three-body mass | 0.90415 kg |
| Wing mass fraction `rho` | 0.1344246 |
| Neutral paper angle `gamma_mean` | -0.019391 rad |
| Point `O` in base FLU | (-0.1215356, 0, 0.0122175) m |
| Amini constant pitch inertia | 0.0256486 kg m^2 |

The negative neutral angle follows from the measured neutral wing COM lying
below the hinge in body FLU.

## Benchmark construction

`flapping_bot.analysis.physx_multibody_inertial_benchmark` instantiates the
formal `IdealCoupledFlappingBotCfg`, applies the accepted measured link
properties, disables gravity, and applies no external wrench. One left-wing
position/velocity drive commands

```text
q_ref = 30 deg * sin(2*pi*f*t)
```

while the hard PhysX mimic relation makes the right wing passive and opposed.
Each case runs eight cycles and analyzes the last four. Harmonic amplitudes
and phases are least-squares fits that include a linear drift term and the
second and third harmonics. Total linear and angular momentum include all
PhysX links and are evaluated about the instantaneous system COM.

## Results

The full 12-case matrix completed. The ranges below span 2-5 Hz.

| Step | Max sync error | Max position tracking error | Linear momentum residual | Angular momentum residual |
|---:|---:|---:|---:|---:|
| 1/240 s | 0.00180-0.01094 deg | 0.0471-0.2851 deg | 1.50e-6-2.90e-6 | 1.38e-4-7.54e-4 |
| 1/480 s | 0.00087-0.00523 deg | 0.0172-0.0631 deg | 9.08e-7-3.24e-6 | 1.19e-4-6.47e-4 |
| 1/1000 s | 0.00033-0.00382 deg | 0.0171-0.0654 deg | 5.20e-6-7.55e-6 | 8.54e-5-5.98e-4 |

The momentum residuals are the maximum total-system momentum norm divided by
the maximum sum of individual-body momentum magnitudes. They measure
cancellation of internal motion, not error relative to a nonzero external
reference.

At the finest step, the actual joint fundamental follows the commanded
trajectory closely:

| Frequency | `q` amplitude ratio | `q` phase error | `qdot` amplitude ratio | `qddot` amplitude ratio | `qddot` phase error |
|---:|---:|---:|---:|---:|---:|
| 2 Hz | 1.00029 | -0.0025 deg | 1.00030 | 1.00030 | -0.441 deg |
| 3 Hz | 1.00064 | -0.0081 deg | 1.00066 | 1.00066 | -0.666 deg |
| 4 Hz | 1.00108 | -0.0183 deg | 1.00113 | 1.00112 | -0.911 deg |
| 5 Hz | 1.00159 | -0.0339 deg | 1.00159 | 1.00158 | -1.096 deg |

The instantaneous Isaac Lab finite-difference `joint_acc` trace becomes noisy
at 1 ms, but its fitted wingbeat fundamental remains accurate. Consequently,
the maximum pointwise acceleration tracking error is a numerical diagnostic,
not a reliable model-acceptance metric by itself.

### Frequency-squared scaling

| Step | PhysX vertical exponent | PhysX pitch exponent | `amplitude/f^2` CV |
|---:|---:|---:|---:|
| 1/240 s | 1.99346 | 1.99431 | 0.20-0.23% |
| 1/480 s | 1.99847 | 1.99868 | 0.047-0.054% |
| 1/1000 s | 2.00132 | 2.00129 | 0.045-0.046% |

All log-log fits have `R^2 > 0.9999992`. The simulated inertial response
therefore follows the expected frequency-squared law over 2-5 Hz.

### PhysX versus Amini

| Step | Vertical amplitude ratio | Vertical phase difference | Pitch amplitude ratio | Pitch phase difference |
|---:|---:|---:|---:|---:|
| 1/240 s | 1.00183-1.00200 | -0.020 to -0.007 deg | 0.96700-0.96796 | 0.233-0.281 deg |
| 1/480 s | 1.00181-1.00184 | -0.009 to -0.003 deg | 0.96679-0.96701 | 0.152-0.240 deg |
| 1/1000 s | 1.00180-1.00183 | -0.004 to -0.001 deg | 0.96668-0.96671 | 0.108-0.218 deg |

The vertical dominant harmonic agrees within 0.20 percent. The PhysX pitch
amplitude is consistently about 3.3 percent lower than the reduced model,
with less than 0.3 degree phase difference. A stable amplitude offset is
expected because PhysX retains pose-varying full-link inertia and all links,
whereas Amini uses a constant symmetric three-body pitch inertia.

### Time-step convergence

Relative to 1/1000 s, the 1/480 s amplitude error remains below 0.37 percent
for all three tracked signals, with acceleration phase errors below
1.28 degrees. At 1/240 s, the worst case is the 5 Hz vertical response:
0.89 percent amplitude error and 3.72 degrees phase error. Thus 1/240 s
preserves amplitude and synchronization well but has a visible high-frequency
phase discretization error; 1/480 s is the stronger validation step.

## Conclusions

The benchmark supports four bounded conclusions:

1. the ideal coupling solver maintains `q_left + q_right = 0` well below the
   existing 0.1 degree gate;
2. internal wing motion conserves whole-system linear and angular momentum to
   small numerical residuals;
3. the body inertial response follows the expected frequency-squared scaling;
4. PhysX naturally reproduces the dominant Amini vertical and pitch inertial
   harmonics without applying an analytical correction term.

Together these results show that the new plant is a functioning measured-wing
multibody realization of the paper's dominant inertial mechanism. They do not
show that the chosen drive represents the real motor and gearbox.

## Limitations and next boundary

- `robot.data.applied_torque` varies strongly with simulation step
  (the 5 Hz maximum is 142.4 N m at 1/240 s and 35.8 N m at 1 ms). It contains
  the sampled numerical drive contribution but does not isolate PhysX mimic
  constraint reaction or identify real shaft torque. It must not be used as a
  motor-load result.
- The analytical model contains the measured body and two wings; PhysX also
  contains minimum-mass tail and rudder placeholders.
- Aerodynamic forces, gravity, asymmetric loading, gearbox loss, compliance,
  backlash and motor limits are outside this test.
- The formal aerodynamic path still uses commanded wing kinematics and applies
  the equivalent wing wrench to `base_link`. Actual joint kinematics and
  per-wing link wrenches are required before aerodynamic joint-load claims.

## Reproduction

The active conda editable installation points at another checkout, so the
current worktree source must be placed first on `PYTHONPATH`:

```bash
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/flapping_bot${PYTHONPATH:+:$PYTHONPATH}"
TERM=xterm python -u scripts/flapping_px4/validate_multibody_inertia.py \
  --headless \
  --output-dir /tmp/physx_multibody_inertia_full_20260729
```

The command wrote `summary.json` and `cases.csv`. The runner now rejects a
`flapping_bot` import resolved outside the current checkout and records the
resolved module path in provenance.
