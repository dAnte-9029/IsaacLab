# DeLaurier Calibration Protocol for Effective-Wrench Comparison

Date: 2026-05-13

## Purpose

This note fixes the intended scope of the DeLaurier calibration baselines used in the paper comparison. The goal is not to turn the DeLaurier model into a high-capacity learned model. The goal is to give the literature-based physics baseline a transparent and limited opportunity to adapt to the specific vehicle before comparing it with the learned effective-wrench estimator.

All calibration must use only training flight logs. Validation logs may be used for model selection or regularization choices. Held-out test logs must be used only for final reporting.

The target for all baselines is the same body-frame effective wrench:

```text
w_b = [F_x, F_y, F_z, M_x, M_y, M_z]
```

## Baseline Ladder

The comparison should use a ladder of increasing adaptation:

```text
Uncalibrated DeLaurier
Gain-calibrated DeLaurier
Physically calibrated DeLaurier
Learned effective wrench
```

This ordering keeps the paper interpretation clear. If the learned model only beats the uncalibrated baseline, the result may mostly reflect parameter mismatch. If it also beats the calibrated baselines on held-out logs, the result supports the claim that real-flight effective wrench contains state-dependent, phase-dependent, or actuator-history-dependent effects beyond simple physics-model calibration.

## Gain-Calibrated DeLaurier

The gain-calibrated baseline is an output-level calibration. First, run the DeLaurier model on the recorded states and controls from the training logs and collect its body-frame wrench prediction:

```text
d_t = [F_x, F_y, F_z, M_x, M_y, M_z]_DeLaurier
y_t = [F_x, F_y, F_z, M_x, M_y, M_z]_label
```

Then fit one scalar gain per wrench component:

```text
y_hat_{t,j} = s_j d_{t,j}
```

where `j` indexes the six wrench channels. Each gain is fitted on training logs by least squares:

```text
s_j = argmin_s sum_t (s d_{t,j} - y_{t,j})^2
```

An optional bias term may be tested on validation logs:

```text
y_hat_{t,j} = s_j d_{t,j} + b_j
```

However, the preferred first report is gain-only because it is easier to interpret physically as an amplitude correction. Bias terms can absorb trim offsets, but they are less physically clean and should be reported separately if used.

This baseline answers a narrow question: can the mismatch between DeLaurier and real-flight effective wrench be explained mostly by per-axis amplitude error?

## Physically Calibrated DeLaurier

The physically calibrated baseline adjusts a small set of interpretable parameters inside or around the DeLaurier-based simulation model. The parameter set is intentionally limited to eight parameters:

```text
theta_phys = [
  wing_normal_force_scale,
  wing_chordwise_force_scale,
  delaurier_theta_w_deg,
  twist_eta_max_deg,
  delaurier_induced_drag_efficiency,
  fuselage_drag_cda,
  tail_lift_scale,
  phase_delay_s
]
```

These parameters are grouped by physical role.

### Aerodynamic Magnitude

`wing_normal_force_scale` scales the wing normal-force contribution. It captures systematic uncertainty in effective wing area, flexible-wing loading, local separation, Reynolds-number effects, and other factors that change the amplitude of the force normal to the wing surface. It primarily affects vertical force and pitch-related moments.

Suggested bounded range:

```text
0.5 to 1.8
```

`wing_chordwise_force_scale` scales the wing chordwise-force contribution. It captures uncertainty in thrust and drag generation, including leading-edge suction efficiency, skin friction, camber-related drag, and chordwise effects of flexible wing motion. It primarily affects streamwise force.

Suggested bounded range:

```text
0.2 to 2.0
```

### Geometry and Compliance

`delaurier_theta_w_deg` is the effective mean wing incidence or pitch offset used by the DeLaurier backend. It accounts for small errors in wing installation angle, mean twist, body-frame alignment, and attitude zeroing. It changes effective angle of attack rather than simply scaling the output.

Suggested bounded range:

```text
-8 deg to +8 deg
```

`twist_eta_max_deg` is the prescribed flexible-wing twist amplitude proxy. The current implementation uses a velocity-scaled prescribed twist model. Calibrating this parameter accounts for uncertainty in passive wing torsion and its effect on instantaneous angle of attack.

Suggested bounded range:

```text
0 deg to 25 deg
```

### Missing Drag

`delaurier_induced_drag_efficiency` controls the finite-wing induced-drag correction. It accounts for the fact that the strip-theory implementation does not fully represent finite-span induced drag in free flight.

Suggested bounded range:

```text
0.3 to 1.2
```

`fuselage_drag_cda` is the equivalent parasite-drag area for the body and non-wing structures. It accounts for drag from the fuselage, tail boom, motor, sensors, and other components not represented by the wing strip-theory model.

Suggested bounded range:

```text
0.0001 to 0.02 m^2
```

### Control and Timing

`tail_lift_scale` scales the low-order tail/elevon/rudder aerodynamic effectiveness. It accounts for uncertainty in tail lift slope, servo effectiveness, surface installation angle, and local flow at the tail.

Suggested bounded range:

```text
0.3 to 2.0
```

`phase_delay_s` shifts the flapping phase or wing kinematic input used by the DeLaurier prediction. It accounts for actuator delay, sensor timestamp mismatch, encoder-to-force delay, and phase-alignment uncertainty. This is especially important for wingbeat-scale wrench prediction.

Suggested bounded range:

```text
-0.04 s to +0.04 s
```

## Calibration Objective

The physically calibrated parameters should be fitted by minimizing normalized wrench error on training logs:

```text
min_theta sum_t || W (w_Del(x_t, u_t; theta) - y_t) ||_2^2 + R(theta)
```

where `W` normalizes each channel, for example by the training-set standard deviation of that wrench channel. `R(theta)` is an optional weak regularization term that penalizes large movement from nominal values. The objective should not be tuned on test logs.

Recommended optimizer for the first implementation:

```text
Powell or Nelder-Mead with bounded parameters
```

If the objective is noisy or slow, use random search followed by local optimization.

## What Not to Calibrate in the First Version

Do not initially calibrate the full set of DeLaurier formula coefficients such as `alpha0_rad`, `eta_s`, `cd_cf`, `alpha_stall_max_rad`, and `c_mac`. Although these parameters are part of the underlying aerodynamic model, tuning many of them together makes the baseline harder to interpret and risks turning it into an opaque fitted model.

The first physically calibrated baseline should focus on platform-adaptation parameters: force scale, drag, trim/incidence, flexible twist, tail effectiveness, and phase alignment.

## Reporting Rules

Report all calibration results on held-out logs. Training-log performance may be used as a diagnostic but should not be the main paper result.

Report the final calibrated parameter values. This is important for interpretability. If a parameter hits its bound, mention it and treat it as a diagnostic that the model class may be missing a relevant effect.

Do not claim that the learned model invalidates DeLaurier theory. The correct interpretation is narrower:

```text
The calibrated DeLaurier baseline tests how far a compact, physically interpretable adaptation of the analytical model can go. Any remaining improvement from the learned effective-wrench estimator indicates effects in the held-out real-flight data that are not captured by global scaling, drag correction, trim adjustment, flexible-twist proxy, tail effectiveness scaling, or phase alignment alone.
```

## Paper Wording

Suggested methods wording:

```text
To avoid comparing a data-trained estimator against a completely unadapted physics model, we also evaluate calibrated DeLaurier baselines. The gain-calibrated baseline fits one scalar gain per body-frame wrench component using only training logs. The physically calibrated baseline fits a small set of eight interpretable parameters associated with aerodynamic magnitude, wing incidence and compliance, missing drag, tail effectiveness, and flapping phase alignment. The calibrated parameters are selected to account for platform-specific uncertainty without freely refitting the analytical model. All calibrated baselines are evaluated on the same held-out logs as the learned model.
```

Suggested discussion wording:

```text
The calibrated DeLaurier results should be interpreted as a strong physics baseline rather than as a separate learned model. If the learned estimator outperforms both gain-calibrated and physically calibrated DeLaurier baselines, the improvement cannot be attributed only to global amplitude errors, missing parasite drag, trim bias, or phase misalignment. It instead suggests that the real-flight effective wrench contains state-dependent or history-dependent residual effects that are better captured by the learned temporal estimator.
```
