# C4 frequency/tail channel replacement experiment

User-authorized diagnostic, base `41ea8565`, uncommitted worktree.
No training, reward, plant, preview-default or checkpoint changes.

`evaluate_continuous_curve.py --tail-checkpoint` selects a second strictly loaded
split actor with matching architecture and observation/action contract. Each step
both actors see the same current closed-loop observation, including the history of
combined executed commands. Channel 0 comes from `--checkpoint`, channels 1..3 from
`--tail-checkpoint`. Component raw actions and combined action are saved in NPZ;
launch metadata identifies both weights. Without the option, inference is unchanged.

Four predeclared conditions (frequency, tail): (25,25), (50,50), (25,50), (50,25).
Use all 96 existing development recipes and 1.2 s preview; holdout remains sealed.
Pure-model conditions also use the two-actor code path to check identity controls.
Successful frequency replacement implicates frequency-dependent behavior; successful
tail replacement implicates tail behavior. Neither result alone proves the specific
PPO/reward cause. Hybrid policies may be outside either actor's training distribution.
Compare signed/absolute tracking errors, speed, altitude, roll and actuator limiting;
reject a claimed recovery that merely trades away other tracking requirements.

Validation: four evaluator unit tests passed, including identity composition,
nonmutation and invalid device/dtype/shape contract handling (CPU dtype/shape and
nonfinite cases exercised; CUDA not exercised). `bash -n` and `git diff --check`
passed. Native smoke and full evaluations were submitted to background execution;
results are pending and no success is claimed.

Outputs and reproduction pipeline:
`logs/continuous_curve/20260911_c4_channel_swap/run.sh`.
Service: `c4-channel-swap-20260911.service`.
Pipeline first requires completed six-case/eight-step smoke, then requires completed
worker status and complete evaluation summary for each full condition. Output paths
must be new. No monitoring requested; inspect durable results in a subsequent task.
