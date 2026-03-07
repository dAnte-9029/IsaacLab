# Flapping Bot RL Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a staged reinforcement-learning stack that starts from the current PX4-inspired teacher-guided straight-flight baseline and ends with a robust pure-RL policy family that can handle straight flight, loiter, and later general mission-level tasks for the DeLaurier flapping-wing vehicle.

**Architecture:** Keep a single core environment family under `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`, and grow capability by adding task wrappers, curriculum switches, observation modes, and evaluation suites instead of forking many unrelated environments. Use the existing PX4-like controller as a teacher / safety prior in the early phases, then progressively weaken and remove that prior while keeping the same action interface and standardized checkpoint evaluation flow.

**Tech Stack:** IsaacLab direct RL environments, Isaac Sim, RSL-RL PPO, DeLaurier wing aerodynamics, existing PX4-like control code, custom evaluation scripts in `scripts/flapping_rl/`, pytest for logic regressions.

---

## Scope and Assumptions

This roadmap is based on the current agreed direction:

- Use `Isaac-FlappingBot-StraightFlight-DeLaurier-TeacherRL-Direct-v0` as the RL starting point.
- Keep the real-vehicle-like action interface: throttle / rudder / elevon pitch / elevon roll.
- Train with true-state observations first.
- Keep wind in training from day one, but do **not** expose wind truth in observations.
- Use the existing PX4-like straight-flight controller as the teacher prior, not as the final deployed policy.
- Progress task difficulty in this order:
  1. straight flight,
  2. loiter,
  3. multi-task mission behaviors,
  4. estimated-state / noisy-sensor robustness,
  5. pure-RL deployment candidate.

---

## Recommended Development Path

### Option A: Teacher-Guided PPO First, Then Pure RL (**Recommended**)

Use the current teacher-guided action envelope, wind curriculum, and watch-and-eval loop to obtain a stable straight-flight RL baseline. Once checkpoint selection is reliable and policy quality is repeatable, shrink teacher dependence and convert the same task to pure RL.

**Why this is recommended:**
- Lowest risk from the current repo state.
- Reuses the good straight-flight baseline that already exists.
- Makes debugging easier because environment, controller, and evaluation are already wired.
- Avoids mixing sensor-noise, multi-tasking, and teacher-removal all at once.

### Option B: Pure RL on Straight Flight Immediately

Disable teacher guidance and train directly with only reward shaping and curriculum.

**Trade-offs:**
- Cleaner conceptually.
- More likely to waste compute on unstable exploration.
- Harder to tell whether failure is from reward design, action space, or insufficient prior.

### Option C: Behavior Cloning / Distillation Before PPO

Collect trajectories from the PX4-like teacher, behavior-clone a policy, then fine-tune with PPO.

**Trade-offs:**
- Could reduce PPO warm-up time.
- Adds data-generation, BC training, and policy handoff complexity.
- Premature until the current teacher-guided PPO path becomes the bottleneck.

**Chosen path:** Option A.

---

## Phase Plan

### Phase 0: Freeze and Validate the Current RL Baseline

**Objective:** Make the current straight-flight teacher-guided pipeline reproducible and easy to inspect.

**Files:**
- Validate: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Validate: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py`
- Validate: `scripts/flapping_rl/train_and_watch.py`
- Validate: `scripts/flapping_rl/watch_and_eval.py`
- Validate: `scripts/flapping_rl/eval_straight_flight_checkpoint.py`
- Validate: `tests/test_rl_teacher_guidance.py`
- Validate: `tests/test_train_and_watch.py`

**Deliverables:**
- One documented “known-good” straight-flight run.
- One selected best checkpoint path.
- One short command set for training, watching, evaluating, and visualizing.

**Steps:**
1. Run the existing smoke workflow and confirm checkpoint scoring consistency.
2. Run a medium training job and record the best checkpoint by `suite` score.
3. Write a short README-style usage note under `docs/flapping_rl/` or update an existing doc.
4. Add a small helper script or doc snippet for “show best checkpoint in Isaac Sim”.
5. Commit only reproducibility and usability changes.

**Tests / Verification:**
- `./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_rl_teacher_guidance.py tests/test_px4_like_guidance.py`
- `python -m py_compile scripts/flapping_rl/train_and_watch.py scripts/flapping_rl/watch_and_eval.py`

---

### Phase 1: Make “Best Checkpoint” a First-Class Artifact

**Objective:** Stop relying on “last checkpoint” and make model selection explicit.

**Files:**
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Create: `scripts/flapping_rl/select_best_checkpoint.py`
- Test: `tests/test_train_and_watch.py`
- Create or modify: `tests/test_watch_and_eval.py`

**Deliverables:**
- Automatic best-checkpoint selection.
- Optional `best_model.pt` symlink or copied artifact.
- A single command to evaluate or visualize the best model.

**Steps:**
1. Write a failing test for selecting the best `suite` row from `summary.csv`.
2. Add logic to parse the evaluation table and pick the best checkpoint.
3. Optionally write `best_model.pt` or `best_checkpoint.txt` into the run directory.
4. Add a small CLI that prints the best checkpoint for a run.
5. Verify no regression in watcher behavior.
6. Commit.

**Tests / Verification:**
- `./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_watch_and_eval.py`

---

### Phase 2: Straight Flight Teacher-to-RL Transfer

**Objective:** Turn the current “teacher-guided but good” policy into a policy that remains good when teacher support weakens.

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/rl_training_utils.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py`
- Create: `tests/test_teacher_schedule.py`

**Key design changes:**
- Separate the teacher schedule into explicit stages:
  - warm start,
  - medium envelope,
  - weak teacher,
  - teacher off.
- Keep reward task-based; do **not** reward matching teacher actions.
- Log how much the policy actually deviates from teacher over time.

**Steps:**
1. Write failing tests for teacher schedule helpers.
2. Add config knobs for multi-stage teacher weakening.
3. Add logs for effective teacher usage over training.
4. Add at least one env config for “weak teacher” and one for “teacher off”.
5. Run medium training to compare `suite` score and termination rate across stages.
6. Commit.

**Success criteria:**
- A weak-teacher policy reaches similar or better `suite` score than the current teacher-heavy baseline.
- A teacher-off continuation run does not immediately collapse.

---

### Phase 3: Straight Flight Wind Robustness Curriculum

**Objective:** Make straight flight robust to a meaningful range of crosswind and time-varying gusts.

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Create: `tests/test_wind_curriculum.py`
- Create: `docs/flapping_rl/wind_eval_protocol.md`

**Key design changes:**
- Keep wind hidden from observations.
- Expand the evaluation suite to multiple wind bins, not just one steady and one OU case.
- Introduce curriculum stops based on completed training stages, not arbitrary one-shot jumps.

**Steps:**
1. Write failing tests for wind curriculum helper behavior.
2. Add named train/eval wind profiles: calm, mild crosswind, strong crosswind, mild OU, strong OU.
3. Add an aggregate score that penalizes collapse in any one case.
4. Run comparative training jobs and select the most stable schedule.
5. Commit.

**Success criteria:**
- Best checkpoint completes all evaluation cases with low termination rate.
- Policy quality is not only good in calm air.

---

### Phase 4: Mission Task Expansion — Loiter First

**Objective:** Reuse the straight-flight policy stack and extend it to `loiter` without breaking straight-flight.

**Files:**
- Create: `source/flapping_bot/flapping_bot/direct/flapping_bot/loiter_env.py` or a task-mode extension in the same env file
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`
- Create: `scripts/flapping_rl/eval_loiter_checkpoint.py`
- Create: `scripts/flapping_rl/watch_and_eval_loiter.py`
- Create: `tests/test_loiter_task_cfg.py`

**Design choice:**
- Prefer sharing the same dynamics and action interface.
- Change only command generation, reward, and evaluation metrics.
- Keep straight-flight and loiter as separate task IDs.

**Steps:**
1. Freeze the straight-flight baseline before touching loiter.
2. Write the loiter task config and registration.
3. Add loiter-specific reward and metrics: radius error, orbit completion, altitude, termination rate.
4. Add loiter watch-and-eval flow mirroring straight-flight.
5. Start with teacher-guided loiter if needed.
6. Commit.

**Success criteria:**
- Straight-flight remains reproducible.
- Loiter can be trained and evaluated with the same tooling style.

---

### Phase 5: Multi-Task RL (Straight + Loiter)

**Objective:** Train one policy family that can switch tasks by command / task token instead of maintaining isolated policies forever.

**Files:**
- Modify: shared env/task config files under `source/flapping_bot/flapping_bot/direct/flapping_bot/`
- Modify: PPO config under `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/`
- Create: task-conditioned evaluation scripts under `scripts/flapping_rl/`
- Create: `tests/test_task_conditioning.py`

**Design choice:**
- Add an explicit task identifier or command mode to observations.
- Keep action space unchanged.
- Standardize evaluation by per-task metrics plus weighted aggregate score.

**Steps:**
1. Write failing tests for task-conditioning utilities.
2. Add task token / command-mode observation.
3. Add mixed-task episode sampling.
4. Add multi-task evaluation summary.
5. Compare single-task vs multi-task degradation.
6. Commit.

**Success criteria:**
- One policy handles at least straight-flight and loiter with acceptable degradation.

---

### Phase 6: Switch from True State to Estimated State

**Objective:** Replace truth observations with the sensor-estimator stack in a controlled way.

**Files:**
- Modify: estimator / observation code under `source/flapping_bot/flapping_bot/`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Create: `tests/test_estimator_observation_mode.py`
- Create: `docs/flapping_rl/estimation_transition.md`

**Design choice:**
- Keep a config switch between truth and estimated observations.
- Train first with truth only, then mixed truth/estimate curriculum, then estimate only.

**Steps:**
1. Add observation-mode switch tests.
2. Add per-step logging of truth-vs-estimate error.
3. Add mixed observation curriculum if needed.
4. Evaluate whether the same policy can survive the switch.
5. If not, fine-tune from the truth-trained checkpoint.
6. Commit.

**Success criteria:**
- Estimated-state policy remains task-capable with small degradation.

---

### Phase 7: Toward Pure-RL Deployment Candidate

**Objective:** Produce a candidate policy that does not rely on teacher actions during execution.

**Files:**
- Modify: env configs in `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: task registration if new pure-RL task IDs are needed
- Create: deployment notes in `docs/flapping_rl/deployment_candidate.md`

**Steps:**
1. Start from the best teacher-guided checkpoint.
2. Resume training with teacher disabled or near-disabled.
3. Keep the same evaluation suite and compare against the teacher-guided baseline.
4. Only promote a policy if it beats or matches the baseline in task metrics.
5. Commit promoted config / notes.

**Success criteria:**
- Pure-RL checkpoint is selected as best by the same evaluation metric, not by manual inspection.

---

## Cross-Cutting Engineering Rules

### Rule 1: Never trust the last checkpoint

Always select by `suite` metrics from `eval/summary.csv`, not by the final training iteration.

### Rule 2: One new difficulty axis at a time

Do not introduce teacher fadeout, stronger wind, sensor noise, and multi-tasking all in the same experiment.

### Rule 3: Keep action interface stable

Do not change the deployed action semantics unless there is a compelling real-vehicle reason.

### Rule 4: Use the same evaluator shape for every task

Every task should have:
- checkpoint evaluator,
- watch-and-eval script,
- per-case metrics,
- aggregate summary row.

### Rule 5: Promote only reproducible artifacts

A “good policy” is not a screenshot; it is:
- a named run directory,
- a selected best checkpoint,
- a saved metric table,
- a replay / visualization command.

---

## Immediate Next 3 Tasks (Recommended Order)

### Task 1: Productize the current straight-flight baseline

**Files:**
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Create: `scripts/flapping_rl/select_best_checkpoint.py`

**Why now:** Current policy quality is already usable; the highest value is making best-model selection explicit and repeatable.

### Task 2: Add straight-flight teacher fadeout configs

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py`
- Create: `tests/test_teacher_schedule.py`

**Why now:** This is the shortest path from “good teacher-guided RL” to “good RL with weaker teacher”.

### Task 3: Build loiter as the second RL task

**Files:**
- Create: loiter env/task/eval files under `source/flapping_bot/flapping_bot/direct/flapping_bot/` and `scripts/flapping_rl/`
- Create: loiter tests

**Why now:** It grows mission capability without yet entangling sensor realism.

---

## Suggested Commands for the Current Stage

**Medium training with evaluation:**
```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-TeacherRL-Direct-v0 \
  --run-name medium_teacher_rl \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 16 \
  --max-iterations 60 \
  --save-interval 10 \
  --episodes 3 \
  --poll-s 5 \
  --run-dir-timeout-s 120 \
  --headless
```

**Visualize a selected checkpoint:**
```bash
./isaaclab.sh -p scripts/flapping_rl/eval_straight_flight_checkpoint.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-TeacherRL-Direct-v0 \
  --checkpoint logs/rsl_rl/flapping_bot_straight_flight/<RUN>/model_<N>.pt \
  --episodes 1 \
  --num_envs 1 \
  --device cuda:0
```

---

## Exit Criteria for “Straight-Flight RL Complete”

Do **not** declare straight-flight RL complete until all of the following are true:

- Best-checkpoint selection is automated.
- A policy survives calm, steady crosswind, and OU crosswind evaluation cases.
- The chosen policy is not teacher-heavy in practice, or there is a clear handoff plan to a weak-teacher / no-teacher continuation.
- Visualization command and evaluator outputs are documented.
- At least one follow-up task path is open: loiter or estimated-state transfer.

