# ADR: Paired global-yaw consistency experiment for PureRL C3b

- Date: 2026-08-31
- Status: Experimental

## Context

The accepted split-frequency C3b actor retains the frozen C1, C2c, and C3a skills, but the
matched transition diagnostic still shows heading-dependent failures. In particular, the
same body-relative C3b template can pass from global heading 0 and fail from heading 180,
even though the path errors at reset are equivalent. The 555-value actor observation contains
30 world-frame attitude quaternions, while its velocities, rates, action history, and body-frame
path preview are already invariant to a common world-yaw rotation. The actor can therefore
fit different action mappings to physically equivalent global headings.

## Decision

Run a single-variable comparison from the fixed-mixture C3b `model_100.pt` source. Keep the
adaptive sampler, split actor, reward, task-aware PPO weights, warm-start guard, seed, number
of environments, minibatches, and iteration budget identical to the completed adaptive run.
Add only a paired actor consistency loss with coefficient 0.05.

For each PPO minibatch observation, sample one yaw delta uniformly from `[-pi, pi]` and left
multiply all 30 world-frame attitude quaternions by that same yaw rotation. Do not change the
body-frame kinematics, flap state, action history, or body-frame path preview. The auxiliary
target is the original actor mean action, so the loss enforces

`actor(global_yaw(observation)) ~= actor(observation)`.

Use RSL-RL's existing mirror-loss hook with data augmentation disabled. This keeps the PPO and
critic objectives on the original on-policy batch. The new auxiliary-loss path is disabled by
default and is mutually exclusive with actor policy distillation because both use the same hook.

## Alternatives considered

- Increase heading randomization or failed-heading sampling. This may improve coverage, but it
  does not directly constrain equivalent headings to use the same action and confounds the first
  causal test with another sampler change.
- Canonicalize the quaternion observation into a heading-relative frame. This removes the
  nuisance coordinate by construction, but changes the deployed observation contract and makes
  warm-start behavior less controlled.
- Introduce a yaw-equivariant network. This gives a stronger architectural guarantee, but is a
  larger intervention than needed for the first benefit measurement.

## Consequences and validation

Baseline behavior is unchanged when the coefficient is zero. The experiment must pass unit tests
for the quaternion transform, auxiliary actor targets, launcher provenance, and a short CPU-native
PhysX smoke before formal training. Benefit is determined only by the frozen C1, C2c, C3a, and C3b
evaluation grids, with special attention to the matched heading-0 versus heading-180 transition
cases. Training reward alone does not establish improvement or promotion.
