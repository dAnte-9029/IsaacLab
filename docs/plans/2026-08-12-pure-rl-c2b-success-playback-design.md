# PureRL C2b Successful Playback Design

## Objective

Record the promoted C2b `model_475.pt` on one deterministic positive six-degree climb case and produce a GIF,
MP4, and machine-readable manifest. The recording must show the actual longitudinal target path and vehicle trail,
not the C1 horizontal route.

## Design

Extend the existing `play_pure_rl_checkpoint.py` entry point with an explicit task argument and C2 longitudinal
case arguments. Preserve its current measured C1 task as the default. For C2b, configure one fixed evaluation row:
heading 0 degrees, flap phase 0 degrees, task ID 1, signed slope +6 degrees, 17.5 m entry, and 25 m slope segment.

The playback will obtain target points from the environment's generated longitudinal path and render them as
small target markers. The existing blue vehicle trail remains. It will continue to enforce nonblank rendered
frames and finite metrics, and additionally require a C2 recording to survive the episode and reach the recovery
segment. Generated media remains under the ignored training run log directory.

## Validation

Pure helper tests will first fail for missing C2 playback configuration and success predicates, then pass after
the minimal implementation. Runtime validation must create all three artifacts, pass the nonblank-frame gate,
show motion across sampled frames, and record a nonterminated recovery-reaching C2b result in the manifest.

## Scope

Only playback scripts, playback helper tests, and this documentation are in scope. Environment dynamics, plant,
reward, termination, checkpoint weights, training configuration, and task registration are unchanged.
