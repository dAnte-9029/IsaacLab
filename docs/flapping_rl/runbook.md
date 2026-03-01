# FlappingBot Straight-Flight RL Runbook

This runbook documents the intended workflow for training a policy that can fly straight in still air.

## Task IDs

- `Isaac-FlappingBot-StraightFlight-Simple-Direct-v0`
- `Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0`

## Smoke Test (Recommended Before Training)

```bash
./isaaclab.sh -p scripts/flapping_rl/smoke_straight_flight_env.py \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --num_envs 16 --steps 240 \
  --headless
```

## Open-Loop Debug (Sanity-Check Trim / Sign Conventions)

```bash
./isaaclab.sh -p scripts/flapping_rl/open_loop_straight_flight.py \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --steps 1200 --num_envs 1 \
  --f-hz 3.8 --elevator-deg -10 --rudder-deg 0 \
  --headless
```

## Single-GPU Training

Stage A (fast wings):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 2000 \
  --seed 0
```

Stage B (DeLaurier wings):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 3000 \
  --seed 0
```

## Dual-GPU Training (Ubuntu)

```bash
chmod +x scripts/reinforcement_learning/rsl_rl/train_flapping_straightflight_dual_gpu.sh
./scripts/reinforcement_learning/rsl_rl/train_flapping_straightflight_dual_gpu.sh \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 2000
```

## Playing a Checkpoint

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --headless \
  --checkpoint <PATH_TO_CHECKPOINT.pt>
```

## Evaluating a Checkpoint (Finite Episodes)

```bash
./isaaclab.sh -p scripts/flapping_rl/eval_straight_flight_checkpoint.py \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --checkpoint <PATH_TO_CHECKPOINT.pt> \
  --episodes 10 --num_envs 1 \
  --headless
```

Note: the evaluator will, by default, load the saved `params/env.yaml` and `params/agent.yaml` from the run
directory (unless you pass `--no_saved_cfg`) so results stay comparable even if you later edit defaults.

## Watch a Run Directory and Auto-Evaluate New Checkpoints

```bash
./isaaclab.sh -p scripts/flapping_rl/watch_and_eval.py \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --log_dir logs/rsl_rl/flapping_bot_straight_flight/<RUN_DIR> \
  --episodes 5 --num_envs 1 \
  --poll_s 60 \
  --headless
```

## Train + Watch in One Command (Train on GPU0, Eval on GPU1)

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
  --run-name sf_long \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 512 \
  --max-iterations 2000 \
  --save-interval 100 \
  --episodes 5 --poll-s 120 \
  --headless
```

## Common Failure Modes

- `ModuleNotFoundError: flapping_bot`
  - Run via `./isaaclab.sh -p ...` so the extension path is set up consistently.
- `No controlled joints resolved`
  - The environment expects joints: `left_wing`, `right_wing`, `left_tail`, `right_tail`.
- NaNs in observations/rewards
  - First run the smoke test script and reduce `--num_envs` to reproduce quickly.
