# PureRL Curriculum 3 Spatial Path Design

- Status: Approved
- Date: 2026-08-11
- Scope: no-wind lateral and composite path curriculum on the measured CPU-native PureRL plant

## Objective

Curriculum 3 extends a promoted C2c policy from straight level, climb, and descent paths to randomized lateral
and coupled three-dimensional paths. The policy must learn stable turns, transitions between different path
primitives, and finally simultaneous turn and climb or descent. It is not given a target speed, target roll, or
target pitch.

The shared `measured_pure_rl_shared_v1` contract remains unchanged: four direct action channels, actual tail-joint
deflection for tail aerodynamics, a normalized 555-value actor observation, 60 Hz policy rate, 480 Hz physics,
the 0--5 Hz flap-frequency command with a 2 Hz/s governor, and the CPU-native holonomic plant as the sole training
and promotion authority. Wind remains disabled.

## Evidence and safety boundary

The real-flight trajectory artifact contains 138 mission segments: 68 straight-level, 21 waypoint-turn, and
49 loiter segments. Only explicit loiter radii are identifiable, spanning 45--80 m; generic waypoint-turn radius
is not identifiable. The real logs contain mixed flight quality and therefore do not establish a recoverable roll
limit. The PX4 configuration uses a 35 degree fixed-wing roll limit, and the operator reports that larger real
roll excursions were not recoverable in the relevant flights.

C3 therefore uses the following approved boundary:

- geometry-induced roll budget: 20 degrees;
- soft actual-roll penalty begins at 25 degrees;
- hard actual-roll termination: 35 degrees;
- extreme-attitude recovery is deferred to a separate future curriculum.

The generator uses 12 m/s as a conservative geometry guard speed. This is a feasibility calculation scale, not a
commanded or rewarded speed. At 12 m/s, radii of approximately 60, 49, and 45 m correspond to 14, 17, and
18 degrees of steady coordinated-turn bank demand, respectively.

## Alternatives considered

### Straight and circular-arc primitives

Reusing the existing path manager would minimize initial implementation work, but its fixed templates and abrupt
curvature boundaries do not provide the required randomized composite paths or a direct coupled-severity bound.

### Random waypoints with a spline

Random waypoint splines produce diverse-looking paths, but curvature and slope bounds are indirect. Rejection,
self-intersection handling, projection behavior, and deterministic evaluation coverage would be harder to audit.

### Constraint-driven arc-length generator

The approved approach samples smooth curvature and flight-path-angle profiles as functions of arc length. It
directly exposes the physical difficulty variables, supports deterministic batched generation, and produces fixed
evaluation cases without changing the policy interface.

## Generator architecture

The generator samples horizontal curvature `kappa(s)` and flight-path angle `gamma(s)`, then integrates

```text
dpsi/ds = kappa(s)
dx/ds = cos(psi) * cos(gamma)
dy/ds = sin(psi) * cos(gamma)
dz/ds = sin(gamma)
```

Every episode independently samples initial heading, turn direction, event type, event amplitude, event length,
transition length, and initial flap phase. A seed determines a reproducible sequence rather than a fixed direction
or trajectory.

Curvature and slope transitions use the quintic smoothstep

```text
h(u) = 10*u^3 - 15*u^4 + 6*u^5
```

so the value and first two derivatives are continuous at event boundaries. Each path begins with zero curvature
and zero slope. Entry and exit transitions are 8--12 m long. Invalid samples are rejected when they self-intersect,
approach a nonadjacent path branch too closely, leave the configured altitude envelope, or violate the stage's
roll, slope, or coupled-severity bound.

The environment stores a batched Tensor centerline table at 0.25 m arc-length spacing for approximately 300 m.
This covers a 20 s episode at the 12 m/s preview guard plus the final preview distance while keeping per-environment
geometry compact enough for the 256-environment CPU-native default.

## Difficulty stages

### C3a: isolated level turn

C3a first isolates lateral control. A current-stage path is a level straight entry, smooth turn entry, one
constant-curvature turn, smooth turn exit, and level recovery.

- geometry roll: 8--14 degrees;
- approximate radius at the guard speed: 60--105 m;
- heading change: 20--50 degrees;
- left and right turns: equal probability;
- episode duration: 20 s;
- sampling: 15% C1, 25% C2c climb/descent, and 60% C3a.

### C3b: sequential lateral and spatial events

C3b adds two or three events per path without allowing curvature and slope to be nonzero simultaneously. Current
tasks include same-direction turns, S-turns, turn then climb or descent, climb or descent then turn, and sustained
loiter.

- geometry roll: 10--17 degrees;
- approximate radius at the guard speed: 49--83 m;
- loiter radius: 50--80 m;
- heading change per finite turn: 20--60 degrees;
- climb and descent angle: 4--12 degrees, matching C2c v2;
- sampling: 15% C1, 20% C2c, 15% C3a, and 50% C3b.

### C3c: coupled three-dimensional events

C3c samples two to four events and requires at least one event with nonzero curvature and nonzero slope.
When a sampled path contains multiple vertical events, their climb and descent directions alternate to bound the
net altitude excursion within the 10 m reset altitude.

- coupled-event geometry roll: 6--17 degrees;
- coupled-event climb or descent magnitude: 3--10 degrees;
- sampling: 15% C1, 20% C2c, 15% earlier C3, and 50% C3c.

Simultaneous events satisfy

```text
(phi_geometry / 20 deg)^2 + (abs(gamma) / 10 deg)^2 <= 1
```

so the generator cannot demand maximum lateral and vertical difficulty at the same time.
The deterministic C3c gate uses one coupled turn and climb or descent event followed by one level turn. This keeps
the two-event retention requirement while avoiding an artificial double altitude excursion from two same-sign
vertical events.

## Projection and path frame

The closest point is an orthogonal projection onto local polyline segments near the previous path progress. The
search allows approximately 2 m of backward progress to represent small reversals and projection noise, while a
bounded local window prevents jumping to another branch. The generator also rejects true self-intersections.

The route-relative frame remains the stable C2 heading/slope frame rather than a raw Frenet normal:

```text
t_hat = [cos(gamma)*cos(psi), cos(gamma)*sin(psi), sin(gamma)]
n_h   = [-sin(psi), cos(psi), 0]
n_v   = [-sin(gamma)*cos(psi), -sin(gamma)*sin(psi), cos(gamma)]
```

Position error is the displacement from the projected centerline point resolved along `n_h` and `n_v`. Velocity
is resolved along all three axes. This preserves the approved reward meaning: reward path-tangent progress and
penalize path-normal velocity without specifying a target speed.

## Preview and observation contract

The actor continues to receive five body-frame relative preview points at 0.12, 0.24, 0.36, 0.48, and 0.60 s.
The query progress is

```text
s_i = s_closest + clamp(v_tangent, 1 m/s, 12 m/s) * preview_time_i
```

No curvature, target attitude, or new task identifier is added to the actor observation. The preview geometry is
the only direct route description, so the normalized actor observation remains exactly 555 values.

## Reward and termination

The existing path-position, tangent-progress, path-normal velocity, angular-rate, pitch-envelope, flap-frequency,
frequency-slew, tail-action-change, and tail-limit terms remain unchanged.

The C1/C2 roll reward is centered on zero roll and would incorrectly penalize necessary turn bank. C3 therefore
uses a curvature-conditioned roll term:

- on zero-curvature C1/C2 rehearsal geometry, retain the existing zero-roll stability reward;
- smoothly fade that reward out during turn entry;
- on an active turn, do not reward a target roll and do not penalize roll at or below 25 degrees;
- between 25 and 35 degrees, apply a quadratic soft penalty;
- at or above 35 degrees, terminate the episode with a distinct roll-limit cause.

The policy must infer the required bank from preview geometry and path-tracking consequences. It is not given a
coordinated-turn attitude command.

## Deterministic evaluation grids

Training samples continuous random trajectories. Promotion uses fixed cases held out from training randomness.

| Stage | Cartesian grid | Cases |
| --- | --- | ---: |
| C3a | 3 geometry-roll levels x 2 directions x 4 headings x 4 flap phases | 96 |
| C3b | 3 lateral-only templates at 16 cases plus 4 vertical templates at 2 slopes x 16 cases | 176 |
| C3c | 3 coupled severities x 4 lateral/vertical sign pairs x 4 headings x 2 flap phases | 96 |

The seven C3b templates are same-direction turns, S-turns, turn then climb, turn then descent, climb then turn,
descent then turn, and sustained loiter. Each vertical template is evaluated at 8 and 12 degrees. C3c covers
left/right crossed with climb/descent at `(roll, slope)` severities `(8,4)`, `(12,7)`, and `(6,9)` degrees. Reports remain split by
direction, vertical sign, event family, and coupled severity.

## Promotion gates

Every current-stage grid must satisfy:

- overall survival rate at least 95%;
- all-event completion rate at least 95%;
- overall success rate at least 90%;
- mean absolute horizontal-normal and vertical-normal position errors each no greater than 0.5 m;
- P95 absolute horizontal-normal and vertical-normal errors each no greater than 1.5 m;
- reverse-motion fraction no greater than 1%;
- P95 absolute roll no greater than 25 degrees;
- zero 35-degree roll-limit terminations;
- finite state, action, reward, and telemetry values.

C3a additionally requires at least 90% success separately for left and right turns. C3b requires at least 87.5%
success for each of its seven 16-case template families. C3c requires at least 90% success separately for left,
right, climb, descent, and each of the four lateral/vertical sign pairs.

Success means that the episode survived, completed its planned events, and ended within the existing path-error
termination envelope. A timeout without event completion is not a success.

At the 256-environment default, promotion evidence begins after 50 PPO iterations and repeats every 25 iterations,
preserving the approved 614,400-transition minimum and 307,200-transition evidence interval. Promotion requires
two adjacent checkpoints to pass the current grid and all required retention suites.

## Rehearsal and retention

The required evaluation matrix is:

| Training stage | Required retained evaluations |
| --- | --- |
| C3a | C1 and C2c |
| C3b | C1, C2c, and C3a |
| C3c | C1, C2c, C3a, and C3b |

Every earlier stage must still pass its frozen promotion gate. Its success rate may not fall by more than five
percentage points from the promoted source checkpoint. C1 also retains its existing score and error-relative
gate. Missing, duplicate, ambiguous, or non-finite evidence fails closed. A current-stage pass with any retained
stage failure is labeled as forgetting and cannot be promoted.

## Implementation and validation boundary

The implementation adds one project-local Tensor spatial-path module, one C3 evaluation module, one C3 promotion
module, C3 environment/task registration, and focused tests. Existing evaluation and training launchers receive
only the integration needed to recognize C3 stages and evidence. No upstream Isaac Lab code is modified.

Validation must cover deterministic and randomized sampling, geometric bounds, transition continuity, rejection
conditions, projection locality, preview compatibility, zero-curvature C1/C2 equivalence, roll reward and
termination boundaries, fixed-grid completeness, retention, partial reset, device/dtype preservation, and a small
fresh-process CPU-native runtime smoke. PPO training is not part of the implementation gate.

## Non-goals

- no target speed, roll, pitch, or low-level attitude controller;
- no action, observation-dimension, normalization, policy-rate, or physics-step change;
- no plant, aerodynamic, frequency-governor, or PPO-network change;
- no wind or dynamics randomization;
- no GPU implicit-drive revival;
- no extreme-attitude recovery curriculum;
- no reward retuning beyond the approved C3 roll semantics;
- no C3 training before C2c is promoted.

## Reconsideration triggers

Revisit the geometry bounds only if C3a fails after verified code/runtime gates and multiple adequately trained
CPU-native seeds, or if real-flight evidence identifies a narrower recoverable-roll envelope. Revisit the preview
contract only if diagnostics show that the fixed 0.6 s horizon cannot disambiguate valid C3 paths. Do not weaken
retention or safety gates merely to promote a checkpoint.
