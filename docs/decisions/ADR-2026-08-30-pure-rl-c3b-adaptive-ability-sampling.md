# ADR-2026-08-30: Fine-grained adaptive ability sampling for PureRL C3b

- Status: Experimental
- Date: 2026-08-30
- Scope: C3b reset sampling on the measured CPU-native plant

## Context

The fixed-mixture C3b run used the promoted split-frequency C3a actor and the frozen
`15/20/15/50` C1/C2c/C3a/C3b task-aware PPO objective. Fresh-process evaluation found no
all-suite passing checkpoint. `model_100.pt` retained C1, C2c and C3a and was the best C3b
candidate, while its remaining C3b failures were concentrated in templates 5 and 7 with
strong positive climb. Later checkpoints also lost C2c retention.

The previous C3a retention-aware scheduler showed that bounded adaptive allocation can
recover some old-task performance, but its task-family success signal did not align closely
enough with the frozen `+12 deg` gate. Repeating that coarse scheduler would allocate many
samples to already-solved cases.

## Decision

Run one controlled 101-iteration C3b continuation initialized weights-only from the fixed-mixture
C3b `model_100.pt`, with a fresh optimizer and the existing split-frequency actor.

The task-aware PPO objective remains fixed at `15/20/15/50`. Reset sampling keeps the total
simple-task mass at 50 percent and the C3b mass at 50 percent. Within the simple half, completed
on-policy C1 survival/tail-limit, C2c strong-climb recovery and C3a event success can reallocate
the C1/C2c/C3a reset probabilities subject to `10/15/10` percent floors. Within C3b, separate
signals adapt:

- the probability of templates 5 and 7 versus the other five C3b templates;
- the probability of sampling `10--12 deg` climb within templates 5 and 7.

The scheduler retains the existing EMA, minimum completed-episode count, update interval and
bounded probability-step rules. Frozen evaluation grids do not feed the scheduler.

## Alternatives considered

- Keep the fixed mixture and manually oversample templates 5/7. This is simpler but cannot react
  if C2c, C3a or another C3b bucket becomes the weakest ability.
- Adapt both reset probabilities and PPO task weights. This changes the optimization objective
  and sampling distribution together, preventing a clean attribution.
- Start C3c coupled training immediately. This adds a new task before C3b has a promotable
  adjacent checkpoint pair and would not isolate the value of adaptive sampling.

## Consequences

- The plant, reward, observation, action governor, split actor and frozen evaluation contracts
  are unchanged.
- More samples improve coverage of weak abilities, while fixed task-aware PPO weights preserve
  the declared multi-task objective.
- Online scheduler telemetry remains diagnostic and cannot promote a checkpoint.
- C3c remains deferred until this bounded C3b experiment is evaluated.

## Assumptions

- Templates 5/7 and their `10--12 deg` climb band are learnable because the promoted primitive
  policies already pass the corresponding isolated abilities.
- Completed training episodes provide a useful but noisy control signal when smoothed and bounded.
- A 101-iteration run is sufficient to produce the frozen `50/75/100` checkpoint sequence.

## Validation requirements

- Pure path tests must verify explicit weak-template and strong-climb probabilities.
- Scheduler tests must verify the 50 percent simple-task total, probability floors and bounded
  changes.
- Launcher tests must freeze source checkpoint, split actor, fresh optimizer, task weights,
  iteration count and train-only execution.
- A fresh CPU-native smoke must instantiate C3b, execute an adaptive update and enter PPO without
  non-finite observations or rewards.
- Final evidence must evaluate checkpoints 50, 75 and 100 in separate fresh processes on C3b v3,
  C3a, C2c and C1. Promotion still requires an adjacent all-suite passing pair.

## Reconsideration triggers

Reconsider this design if the weak-template probability saturates without improving template 5/7,
if C1/C2c/C3a regress despite their floors, or if the adaptive distribution oscillates between
successive update windows. In those cases, diagnose gradient interference or add primitive-only
teacher consolidation rather than increasing scheduler aggressiveness.
