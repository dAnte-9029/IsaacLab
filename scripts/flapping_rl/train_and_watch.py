"""Launch training with optional concurrent checkpoint evaluation.

Motivation: for long unattended runs, it's useful to continuously evaluate new
`model_*.pt` checkpoints and append metrics to `eval/summary.csv` while training
keeps running.

Typical usage:
  # Train on GPU0, evaluate on GPU1
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-TeacherRL-Direct-v0 \
    --run-name sf_long \
    --train-device cuda:0 \
    --eval-device cuda:1 \
    --num-envs 512 \
    --max-iterations 2000 \
    --save-interval 100 \
    --episodes 5 --poll-s 120 \
    --headless

  # Canonical CPU-native measured-wing plant (sequential evaluation by default)
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0 \
    --run-name native_cpu_pure_rl \
    --headless

  # Controlled C3a retention Phase A (task-aware PPO + actor distillation 0.05)
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
    --run-name pure_rl_c3a_task_aware_distill005_seed0_101iter \
    --c3a-retention-phase-a \
    --load_run <PROMOTED_C2C_RUN> \
    --checkpoint model_550.pt \
    --source-checkpoint-path <PROMOTED_C2C_MODEL_550> \
    --headless

  # C1-initialized C1+C2c+C3a joint multi-task experiment
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
    --run-name pure_rl_c3a_joint_from_c1_seed0_201iter \
    --c3a-joint-from-c1 \
    --load_run <PROMOTED_C1_RUN> \
    --checkpoint model_1300.pt \
    --source-stage c1_straight \
    --source-checkpoint-path <PROMOTED_C1_MODEL_1300> \
    --headless

  # Same joint recipe with pre-governor requested-frequency smoothness
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
    --run-name pure_rl_c3a_joint_reqfreqsmooth005_seed0_201iter \
    --c3a-joint-requested-frequency-smoothness \
    --source-checkpoint-path <PROMOTED_C1_MODEL_1300> \
    --headless

  # Same joint recipe with L1 total variation on the pre-governor frequency request
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
    --run-name pure_rl_c3a_joint_reqfreqtv010_seed0_201iter \
    --c3a-joint-requested-frequency-total-variation \
    --source-checkpoint-path <PROMOTED_C1_MODEL_1300> \
    --headless

  # Same joint recipe with a penalty on frequency requests rejected by the governor
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
    --run-name pure_rl_c3a_joint_reqappliedgap005_seed0_201iter \
    --c3a-joint-requested-applied-frequency-gap \
    --source-checkpoint-path <PROMOTED_C1_MODEL_1300> \
    --headless

  # Independent slow frequency and full-observation tail actor trunks
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
    --run-name pure_rl_c3a_split_frequency_actor_seed0_201iter \
    --c3a-split-frequency-actor \
    --source-checkpoint-path <JOINT_MODEL_200> \
    --headless

Add `--concurrent-eval` only when the training and watcher processes are
intentionally allowed to share or use independently assigned compute resources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from checkpoint_selection import refresh_best_checkpoint_artifacts, select_best_checkpoint_row
from eval_suites import get_eval_suite_choices
from pure_rl_eval_common import (
    MEASURED_PURE_RL_TASK_ID,
    PURE_RL_CURRICULUM1_EVAL_CONTRACT,
    PURE_RL_CURRICULUM1_EVAL_SUITE,
    curriculum_stage_for_task,
    is_measured_pure_rl_task,
    longitudinal_stage_for_task,
    spatial_stage_for_task,
)
from pure_rl_longitudinal_eval import LONGITUDINAL_EVAL_CONTRACTS, build_longitudinal_evaluation_grid
from pure_rl_spatial_eval import SPATIAL_EVAL_CONTRACTS, build_spatial_evaluation_grid


_NATIVE_HOLONOMIC_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"
_MEASURED_PURE_RL_TASK_ID = MEASURED_PURE_RL_TASK_ID
_DEFAULT_NUM_ENVS = 512
_DEFAULT_MAX_ITERATIONS = 2000
_DEFAULT_SAVE_INTERVAL = 100
_MEASURED_PURE_RL_DEFAULT_NUM_ENVS = 256
_MEASURED_PURE_RL_DEFAULT_MAX_ITERATIONS = 500
_MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL = 25
_MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES = 16
_C3A_TASK_ID = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0"
_C3B_TASK_ID = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3b-Direct-v0"
_C3C_TASK_ID = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3c-Direct-v0"
_C3A_RETENTION_PHASE_A_NUM_ENVS = 256
_C3A_RETENTION_PHASE_A_MAX_ITERATIONS = 101
_C3A_RETENTION_PHASE_A_SAVE_INTERVAL = 25
_C3A_RETENTION_PHASE_A_NUM_MINI_BATCHES = 16
_C3A_RETENTION_PHASE_A_STRONG_CLIMB_PROBABILITY = 0.5
_C3A_RETENTION_PHASE_A_DISTILLATION_COEFFICIENT = 0.05
_C3A_JOINT_FROM_C1_MAX_ITERATIONS = 201
_C3A_JOINT_FROM_C1_STRONG_CLIMB_PROBABILITY = 0.5
_C3A_JOINT_FROM_C1_SOURCE_STAGE = "c1_straight"
_C3A_JOINT_FROM_C1_SOURCE_RUN = "2026-08-07_05-05-44_curriculum1_overnight_seed2"
_C3A_JOINT_FROM_C1_CHECKPOINT = "model_1300.pt"
_C3A_JOINT_REQUESTED_FREQUENCY_SQUARED_DELTA_PENALTY_WEIGHT = 0.05
_C3A_JOINT_REQUESTED_FREQUENCY_TOTAL_VARIATION_PENALTY_WEIGHT = 0.10
_C3A_JOINT_REQUESTED_APPLIED_FREQUENCY_GAP_PENALTY_WEIGHT = 0.05
_C3A_BASE_POLICY_HIDDEN_DIMS = (256, 128)
_C3A_LARGE_ACTOR_HIDDEN_DIMS = (512, 256)
_C3A_SPLIT_FREQUENCY_ACTOR_SOURCE_STAGE = "c3a_joint"
_C3A_SPLIT_FREQUENCY_ACTOR_SOURCE_RUN = (
    "2026-08-28_15-56-43_pure_rl_c3a_joint_reqappliedgap005_seed0_201iter"
)
_C3A_SPLIT_FREQUENCY_ACTOR_CHECKPOINT = "model_200.pt"
_C3A_SPLIT_FREQUENCY_ACTOR_CLASS_NAME = "PureRLSplitActorCritic"
_C3A_SPLIT_FREQUENCY_ACTOR_ROUTE = "c3a_split_frequency_actor_v1"
_C3B_SPLIT_FREQUENCY_ACTOR_SOURCE_STAGE = "c3a"
_C3B_SPLIT_FREQUENCY_ACTOR_SOURCE_RUN = (
    "2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter"
)
_C3B_SPLIT_FREQUENCY_ACTOR_CHECKPOINT = "model_200.pt"
_C3B_SPLIT_FREQUENCY_ACTOR_ROUTE = "c3b_split_frequency_actor_v2"
_C3B_SPLIT_FREQUENCY_ACTOR_MAX_ITERATIONS = 251
_C3B_ADAPTIVE_SAMPLING_SOURCE_STAGE = "c3b"
_C3B_ADAPTIVE_SAMPLING_SOURCE_RUN = (
    "2026-08-29_15-01-17_pure_rl_c3b_split_frequency_actor_v2_seed0_251iter"
)
_C3B_ADAPTIVE_SAMPLING_CHECKPOINT = "model_100.pt"
_C3B_ADAPTIVE_SAMPLING_ROUTE = "c3b_adaptive_sampling_from_model100_v1"
_C3B_ADAPTIVE_SAMPLING_MAX_ITERATIONS = 101
_C3B_YAW_CONSISTENCY_COEFFICIENT = 0.05
_C3B_YAW_CONSISTENCY_ROUTE = "c3b_adaptive_sampling_yaw_consistency_from_model100_v1"
_C3B_HEADING_CANONICAL_OBSERVATION_ROUTE = (
    "c3b_adaptive_sampling_heading_canonical_observation_from_model100_v1"
)
_C3C_JOINT_SOURCE_STAGE = "c3b"
_C3C_JOINT_SOURCE_RUN = "2026-08-31_17-32-50_pure_rl_c3b_headingcanonical_from100_seed0_101iter"
_C3C_JOINT_CHECKPOINT = "model_75.pt"
_C3C_JOINT_ROUTE = "c3c_joint_heading_canonical_from_promoted_c3b_v1"
_C3C_JOINT_MAX_ITERATIONS = 101
_C3_FULL_JOINT_SOURCE_STAGE = "c2c"
_C3_FULL_JOINT_SOURCE_RUN = "2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550"
_C3_FULL_JOINT_CHECKPOINT = "model_550.pt"
_C3_FULL_JOINT_ROUTE = "c3_full_joint_heading_canonical_from_promoted_c2c_v1"
_C3_FULL_JOINT_MAX_ITERATIONS = 301
_C3_FULL_JOINT_TASK_WEIGHTS = (0.15, 0.20, 0.15, 0.25, 0.25)


def _resolve_eval_suite(task: str, eval_suite: str) -> str:
    if eval_suite == "straight_standard" and is_measured_pure_rl_task(task):
        longitudinal_stage_id = longitudinal_stage_for_task(task)
        if longitudinal_stage_id is not None:
            return LONGITUDINAL_EVAL_CONTRACTS[longitudinal_stage_id]
        spatial_stage_id = spatial_stage_for_task(task)
        if spatial_stage_id is not None:
            return SPATIAL_EVAL_CONTRACTS[spatial_stage_id]
        return PURE_RL_CURRICULUM1_EVAL_SUITE
    if eval_suite == "straight_standard" and "PathTracking" in str(task):
        if "Primitive" in str(task):
            return "path_tracking_estimated_primitives_nowind_v1"
        return "path_tracking_estimated_nowind_v1"
    return str(eval_suite)


def _resolve_eval_shape(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve task-aware watcher defaults while preserving explicit overrides."""

    stage_id = longitudinal_stage_for_task(args.task)
    spatial_stage_id = spatial_stage_for_task(args.task)
    resolved_suite = _resolve_eval_suite(args.task, str(args.eval_suite))
    if stage_id is not None and resolved_suite == LONGITUDINAL_EVAL_CONTRACTS[stage_id]:
        default_count = len(build_longitudinal_evaluation_grid(stage_id))
    elif spatial_stage_id is not None and resolved_suite == SPATIAL_EVAL_CONTRACTS[spatial_stage_id]:
        default_count = len(build_spatial_evaluation_grid(spatial_stage_id))
    elif is_measured_pure_rl_task(args.task) and resolved_suite == PURE_RL_CURRICULUM1_EVAL_SUITE:
        default_count = 16
    else:
        default_count = None
    requested_num_envs = getattr(args, "eval_num_envs", None)
    requested_episodes = getattr(args, "episodes", None)
    num_envs = default_count if requested_num_envs is None and default_count is not None else (
        1 if requested_num_envs is None else int(requested_num_envs)
    )
    episodes = default_count if requested_episodes is None and default_count is not None else (
        5 if requested_episodes is None else int(requested_episodes)
    )
    if num_envs <= 0 or episodes <= 0:
        raise ValueError("Evaluation environment and episode counts must be positive.")
    return num_envs, episodes


def _should_apply_estimated_teacher_defaults(task: str) -> bool:
    task_name = str(task)
    return "FlappingBot" in task_name and "RL" in task_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with optional concurrent checkpoint evaluation.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help=(
            "Training environment count. Defaults to 256 for the CPU-native MeasuredPureRL task "
            "and 512 for other tasks."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Defaults to 500 for MeasuredPureRL tasks and 2000 for other tasks.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Defaults to 25 for MeasuredPureRL tasks and 100 for other tasks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mass-kg-override",
        type=float,
        default=None,
        help="Optional total vehicle mass override passed as total_mass_kg_override=<value>.",
    )
    parser.add_argument("--train-device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="cuda:1")
    parser.add_argument(
        "--native-cpu",
        action="store_true",
        help=(
            "Run the simulator and policy on CPU and load the canonical native holonomic constraint extension. "
            "This is selected automatically for the MeasuredPureRL task. The extension is incompatible with "
            "direct-GPU PhysX."
        ),
    )
    parser.add_argument(
        "--native-extension-parent",
        type=Path,
        default=None,
        help="Parent directory containing the omni.flapping_bot.holonomic_constraint extension.",
    )
    parser.add_argument(
        "--agent-device",
        type=str,
        default=None,
        help="Optional RSL-RL policy/optimizer device override; --native-cpu defaults this to cpu.",
    )
    parser.add_argument(
        "--agent-num-mini-batches",
        type=int,
        default=None,
        help="Positive PPO minibatch count. Defaults to 16 for MeasuredPureRL tasks.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Run training without the concurrent checkpoint watcher or final evaluation. This is already the "
            "default for measured PureRL tasks and remains available as an explicit compatibility flag."
        ),
    )
    parser.add_argument(
        "--concurrent-eval",
        action="store_true",
        help=(
            "Run the checkpoint watcher concurrently with training. Measured PureRL tasks require this explicit "
            "opt-in; other tasks retain their concurrent-evaluation default."
        ),
    )
    parser.add_argument(
        "--disable-kit-fs-watcher",
        action="store_true",
        help="Disable Kit extension filesystem watching for non-interactive runs.",
    )
    parser.add_argument(
        "--freeze-steps-after-reset",
        type=int,
        default=None,
        help=(
            "Optional environment reset-freeze override. MeasuredPureRL now defaults to zero in its task config."
        ),
    )
    parser.add_argument(
        "--c2c-strong-climb-probability",
        type=float,
        default=None,
        help="Optional C3 rehearsal quota for C2c climbs in the configured strong-slope band.",
    )
    parser.add_argument(
        "--c2c-recycle-on-recovery",
        action="store_true",
        help="End successful C2c rehearsal episodes as timeouts immediately after recovery begins.",
    )
    parser.add_argument(
        "--adaptive-task-sampling",
        action="store_true",
        help=(
            "Adapt the C1/C2c/C3a environment mix from completed-episode retention signals; "
            "C3b additionally adapts weak-template and strong-climb coverage while keeping its "
            "50 percent complex-task share fixed."
        ),
    )
    parser.add_argument(
        "--actor-distillation-coefficient",
        type=float,
        default=0.0,
        help=(
            "Frozen source-actor MSE coefficient on C1/C2c rehearsal observations. "
            "Currently supported only for a weights-only C3a warm start."
        ),
    )
    parser.add_argument(
        "--c3a-retention-phase-a",
        action="store_true",
        help=(
            "Apply the controlled C3a task-aware PPO plus actor-distillation-0.05 recipe: "
            "weights-only C2c warm start, seed 0, 256 environments, 16 mini-batches, "
            "101 iterations, 25-iteration checkpoint cadence, strong-climb probability 0.5, "
            "and train-only execution. Conflicting overrides fail closed."
        ),
    )
    parser.add_argument(
        "--c3a-large-actor",
        action="store_true",
        help=(
            "Use the method-2 C3a actor capacity experiment: actor hidden dimensions [512, 256], "
            "critic unchanged, and function-preserving Net2Wider initialization from the [256, 128] "
            "C2c source actor. Requires a weights-only C3a warm start."
        ),
    )
    parser.add_argument(
        "--c3a-joint-from-c1",
        action="store_true",
        help=(
            "Run the accepted C1-initialized C1+C2c+C3a joint multi-task experiment: "
            "weights-only promoted-C1 warm start, seed 0, 256 environments, 16 mini-batches, "
            "201 iterations, 25-iteration checkpoint cadence, strong-climb probability 0.5, "
            "task-aware PPO with the registered 15/35/50 weights, and no distillation, "
            "adaptive sampling, or wider actor. Conflicting overrides fail closed."
        ),
    )
    parser.add_argument(
        "--c3a-joint-requested-frequency-smoothness",
        action="store_true",
        help=(
            "Run the C1-initialized joint recipe with one additional controlled variable: "
            "a 0.05 penalty on the pre-governor requested frequency-action delta. "
            "The original --c3a-joint-from-c1 route remains unchanged."
        ),
    )
    parser.add_argument(
        "--c3a-joint-requested-frequency-total-variation",
        action="store_true",
        help=(
            "Run the C1-initialized joint recipe with one additional controlled variable: "
            "a 0.10 L1 total-variation penalty on the pre-governor requested frequency action. "
            "The original joint and squared-smoothness routes remain unchanged."
        ),
    )
    parser.add_argument(
        "--c3a-joint-requested-applied-frequency-gap",
        action="store_true",
        help=(
            "Run the C1-initialized joint recipe with one additional controlled variable: "
            "a 0.05 penalty on the absolute gap between the clipped frequency request and "
            "the governor-applied frequency action. Existing joint reward routes remain unchanged."
        ),
    )
    parser.add_argument(
        "--c3a-split-frequency-actor",
        action="store_true",
        help=(
            "Warm-start two independent [256, 128] actor trunks from the passing joint model_200: "
            "frequency uses a cycle-averaged phase-fixed 555-value observation, tail uses the full "
            "observation, and the existing governor-gap joint recipe remains otherwise unchanged."
        ),
    )
    parser.add_argument(
        "--c3b-split-frequency-actor",
        action="store_true",
        help=(
            "Run the accepted C3b joint recipe from the promoted C3a split actor: "
            "fresh optimizer, 15/20/15/50 task weights, bounded warm start, seed 0, "
            "256 environments, 16 mini-batches, 251 iterations, and 25-iteration saves."
        ),
    )
    parser.add_argument(
        "--c3b-adaptive-sampling",
        action="store_true",
        help=(
            "Run the controlled C3b adaptive-sampling continuation from the best fixed-mixture "
            "split actor model_100: fresh optimizer, fixed 15/20/15/50 task-aware PPO objective, "
            "bounded simple-task reallocation, adaptive weak-template/strong-climb coverage, "
            "101 iterations, and train-only execution."
        ),
    )
    parser.add_argument(
        "--c3b-yaw-consistency",
        action="store_true",
        help=(
            "Run the matched C3b adaptive continuation with one additional variable: "
            "a 0.05 actor MSE loss between each observation and a paired global-yaw rotation."
        ),
    )
    parser.add_argument(
        "--c3b-heading-canonical-observation",
        action="store_true",
        help=(
            "Run the matched C3b adaptive continuation with one observation change: "
            "express all actor attitude quaternions relative to the episode route heading."
        ),
    )
    parser.add_argument(
        "--c3c-joint-from-promoted-c3b",
        action="store_true",
        help=(
            "Run the bounded C3c joint recipe from promoted heading-canonical C3b model_75: "
            "fresh optimizer, registered 15/20/15/50 cumulative sampling, split actor, "
            "heading-canonical observation, task-aware PPO, 101 iterations, and 25-iteration saves."
        ),
    )
    parser.add_argument(
        "--c3-full-joint-from-c2c",
        action="store_true",
        help=(
            "Train the complete C3 task union directly from promoted C2c: fixed "
            "15/20/15/25/25 C1/C2c/C3a/C3b/C3c sampling, all C3b templates, "
            "coupled C3c paths, split actor, heading-canonical observation, fresh optimizer, "
            "301 iterations, and no adaptive sampling or distillation."
        ),
    )
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        default=None,
        help="Defaults to 16 for the MeasuredPureRL fixed grid and 1 for other tasks.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Defaults to 16 for the MeasuredPureRL fixed grid and 5 for other tasks.",
    )
    parser.add_argument("--poll-s", type=float, default=120.0)
    parser.add_argument("--run-dir-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--eval-suite",
        type=str,
        default="straight_standard",
        choices=get_eval_suite_choices(),
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from a previous run/checkpoint.")
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="Warm-start policy weights from a checkpoint without restoring optimizer or iteration state.",
    )
    parser.add_argument("--load_run", type=str, default=None, help="Existing run directory name used for resume.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or regex used for resume.")
    parser.add_argument(
        "--source-stage",
        type=str,
        default=None,
        help="Required C1/C2 source stage recorded for a longitudinal warm start.",
    )
    parser.add_argument(
        "--source-checkpoint-path",
        type=Path,
        default=None,
        help="Required exact source checkpoint path used for C2 provenance.",
    )
    parser.add_argument(
        "--portable-root-base",
        type=Path,
        default=None,
        help="Base portable root used to isolate IsaacSim cache/state for the train and watcher child processes.",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _set_preset_value(
    args: argparse.Namespace,
    *,
    name: str,
    expected: object,
    option: str,
    preset_option: str,
) -> None:
    configured = getattr(args, name, None)
    if configured is not None and configured != expected:
        raise ValueError(f"{preset_option} requires {option}={expected}; received {configured}.")
    setattr(args, name, expected)


def _apply_c3a_retention_phase_a_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the frozen single-variable Phase-A training recipe."""

    if not bool(getattr(args, "c3a_retention_phase_a", False)):
        return args
    if str(getattr(args, "task", "")) != _C3A_TASK_ID:
        raise ValueError("--c3a-retention-phase-a requires the measured CPU-native C3a task.")
    for name, option in (
        ("resume", "--resume"),
        ("concurrent_eval", "--concurrent-eval"),
        ("adaptive_task_sampling", "--adaptive-task-sampling"),
        ("c2c_recycle_on_recovery", "--c2c-recycle-on-recovery"),
    ):
        if bool(getattr(args, name, False)):
            raise ValueError(f"--c3a-retention-phase-a cannot be combined with {option}.")

    _set_preset_value(
        args,
        name="num_envs",
        expected=_C3A_RETENTION_PHASE_A_NUM_ENVS,
        option="--num-envs",
        preset_option="--c3a-retention-phase-a",
    )
    _set_preset_value(
        args,
        name="max_iterations",
        expected=_C3A_RETENTION_PHASE_A_MAX_ITERATIONS,
        option="--max-iterations",
        preset_option="--c3a-retention-phase-a",
    )
    _set_preset_value(
        args,
        name="save_interval",
        expected=_C3A_RETENTION_PHASE_A_SAVE_INTERVAL,
        option="--save-interval",
        preset_option="--c3a-retention-phase-a",
    )
    _set_preset_value(
        args,
        name="seed",
        expected=0,
        option="--seed",
        preset_option="--c3a-retention-phase-a",
    )
    _set_preset_value(
        args,
        name="agent_num_mini_batches",
        expected=_C3A_RETENTION_PHASE_A_NUM_MINI_BATCHES,
        option="--agent-num-mini-batches",
        preset_option="--c3a-retention-phase-a",
    )
    _set_preset_value(
        args,
        name="c2c_strong_climb_probability",
        expected=_C3A_RETENTION_PHASE_A_STRONG_CLIMB_PROBABILITY,
        option="--c2c-strong-climb-probability",
        preset_option="--c3a-retention-phase-a",
    )
    configured_coefficient = float(getattr(args, "actor_distillation_coefficient", 0.0))
    if configured_coefficient not in (0.0, _C3A_RETENTION_PHASE_A_DISTILLATION_COEFFICIENT):
        raise ValueError(
            "--c3a-retention-phase-a requires --actor-distillation-coefficient=0.05."
        )
    args.actor_distillation_coefficient = _C3A_RETENTION_PHASE_A_DISTILLATION_COEFFICIENT

    source_stage = str(getattr(args, "source_stage", "") or "").strip()
    if source_stage not in ("", "c2c"):
        raise ValueError("--c3a-retention-phase-a requires --source-stage=c2c.")
    checkpoint = str(getattr(args, "checkpoint", "") or "").strip()
    if checkpoint not in ("", "model_550.pt"):
        raise ValueError("--c3a-retention-phase-a requires --checkpoint=model_550.pt.")

    args.source_stage = "c2c"
    args.checkpoint = "model_550.pt"
    args.load_weights_only = True
    args.native_cpu = True
    _set_preset_value(
        args,
        name="agent_device",
        expected="cpu",
        option="--agent-device",
        preset_option="--c3a-retention-phase-a",
    )
    args.train_only = True
    args.concurrent_eval = False
    return args


def _apply_c3a_joint_from_c1_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the accepted C1-initialized C3a joint multi-task recipe."""

    joint_from_c1 = bool(getattr(args, "c3a_joint_from_c1", False))
    requested_frequency_smoothness = bool(
        getattr(args, "c3a_joint_requested_frequency_smoothness", False)
    )
    requested_frequency_total_variation = bool(
        getattr(args, "c3a_joint_requested_frequency_total_variation", False)
    )
    requested_applied_frequency_gap = bool(
        getattr(args, "c3a_joint_requested_applied_frequency_gap", False)
    )
    selected_routes = sum(
        (
            joint_from_c1,
            requested_frequency_smoothness,
            requested_frequency_total_variation,
            requested_applied_frequency_gap,
        )
    )
    if selected_routes == 0:
        return args
    if selected_routes > 1:
        raise ValueError(
            "Select only one of --c3a-joint-from-c1, "
            "--c3a-joint-requested-frequency-smoothness, and "
            "--c3a-joint-requested-frequency-total-variation, and "
            "--c3a-joint-requested-applied-frequency-gap."
        )
    if requested_applied_frequency_gap:
        preset_option = "--c3a-joint-requested-applied-frequency-gap"
    elif requested_frequency_total_variation:
        preset_option = "--c3a-joint-requested-frequency-total-variation"
    elif requested_frequency_smoothness:
        preset_option = "--c3a-joint-requested-frequency-smoothness"
    else:
        preset_option = "--c3a-joint-from-c1"
    args.c3a_joint_from_c1 = True
    if str(getattr(args, "task", "")) != _C3A_TASK_ID:
        raise ValueError(f"{preset_option} requires the measured CPU-native C3a task.")
    for name, option in (
        ("c3a_retention_phase_a", "--c3a-retention-phase-a"),
        ("c3a_large_actor", "--c3a-large-actor"),
        ("c3a_split_frequency_actor", "--c3a-split-frequency-actor"),
        ("c3b_adaptive_sampling", "--c3b-adaptive-sampling"),
        ("c3b_yaw_consistency", "--c3b-yaw-consistency"),
        ("c3b_heading_canonical_observation", "--c3b-heading-canonical-observation"),
        ("resume", "--resume"),
        ("concurrent_eval", "--concurrent-eval"),
        ("adaptive_task_sampling", "--adaptive-task-sampling"),
        ("c2c_recycle_on_recovery", "--c2c-recycle-on-recovery"),
    ):
        if bool(getattr(args, name, False)):
            raise ValueError(f"{preset_option} cannot be combined with {option}.")

    _set_preset_value(
        args,
        name="num_envs",
        expected=_MEASURED_PURE_RL_DEFAULT_NUM_ENVS,
        option="--num-envs",
        preset_option=preset_option,
    )
    _set_preset_value(
        args,
        name="max_iterations",
        expected=_C3A_JOINT_FROM_C1_MAX_ITERATIONS,
        option="--max-iterations",
        preset_option=preset_option,
    )
    _set_preset_value(
        args,
        name="save_interval",
        expected=_MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL,
        option="--save-interval",
        preset_option=preset_option,
    )
    _set_preset_value(args, name="seed", expected=0, option="--seed", preset_option=preset_option)
    _set_preset_value(
        args,
        name="agent_num_mini_batches",
        expected=_MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES,
        option="--agent-num-mini-batches",
        preset_option=preset_option,
    )
    _set_preset_value(
        args,
        name="c2c_strong_climb_probability",
        expected=_C3A_JOINT_FROM_C1_STRONG_CLIMB_PROBABILITY,
        option="--c2c-strong-climb-probability",
        preset_option=preset_option,
    )
    _set_preset_value(
        args,
        name="agent_device",
        expected="cpu",
        option="--agent-device",
        preset_option=preset_option,
    )

    if float(getattr(args, "actor_distillation_coefficient", 0.0)) != 0.0:
        raise ValueError(f"{preset_option} requires --actor-distillation-coefficient=0.")
    source_stage = str(getattr(args, "source_stage", "") or "").strip()
    if source_stage not in ("", _C3A_JOINT_FROM_C1_SOURCE_STAGE):
        raise ValueError(
            f"{preset_option} requires --source-stage={_C3A_JOINT_FROM_C1_SOURCE_STAGE}."
        )
    checkpoint = str(getattr(args, "checkpoint", "") or "").strip()
    if checkpoint not in ("", _C3A_JOINT_FROM_C1_CHECKPOINT):
        raise ValueError(
            f"{preset_option} requires --checkpoint={_C3A_JOINT_FROM_C1_CHECKPOINT}."
        )
    load_run = str(getattr(args, "load_run", "") or "").strip()
    if load_run not in ("", _C3A_JOINT_FROM_C1_SOURCE_RUN):
        raise ValueError(f"{preset_option} requires --load_run={_C3A_JOINT_FROM_C1_SOURCE_RUN}.")

    args.source_stage = _C3A_JOINT_FROM_C1_SOURCE_STAGE
    args.load_run = _C3A_JOINT_FROM_C1_SOURCE_RUN
    args.checkpoint = _C3A_JOINT_FROM_C1_CHECKPOINT
    args.load_weights_only = True
    args.native_cpu = True
    args.train_only = True
    args.concurrent_eval = False
    return args


def _apply_c3a_split_frequency_actor_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the controlled independent frequency/tail actor experiment."""

    if not bool(getattr(args, "c3a_split_frequency_actor", False)):
        return args
    preset_option = "--c3a-split-frequency-actor"
    if str(getattr(args, "task", "")) != _C3A_TASK_ID:
        raise ValueError(f"{preset_option} requires the measured CPU-native C3a task.")
    for name, option in (
        ("c3a_retention_phase_a", "--c3a-retention-phase-a"),
        ("c3a_large_actor", "--c3a-large-actor"),
        ("c3a_joint_from_c1", "--c3a-joint-from-c1"),
        ("c3a_joint_requested_frequency_smoothness", "--c3a-joint-requested-frequency-smoothness"),
        ("c3a_joint_requested_frequency_total_variation", "--c3a-joint-requested-frequency-total-variation"),
        ("c3a_joint_requested_applied_frequency_gap", "--c3a-joint-requested-applied-frequency-gap"),
        ("resume", "--resume"),
        ("concurrent_eval", "--concurrent-eval"),
        ("adaptive_task_sampling", "--adaptive-task-sampling"),
        ("c2c_recycle_on_recovery", "--c2c-recycle-on-recovery"),
    ):
        if bool(getattr(args, name, False)):
            raise ValueError(f"{preset_option} cannot be combined with {option}.")

    for name, expected, option in (
        ("num_envs", _MEASURED_PURE_RL_DEFAULT_NUM_ENVS, "--num-envs"),
        ("max_iterations", _C3A_JOINT_FROM_C1_MAX_ITERATIONS, "--max-iterations"),
        ("save_interval", _MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL, "--save-interval"),
        ("seed", 0, "--seed"),
        ("agent_num_mini_batches", _MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES, "--agent-num-mini-batches"),
        ("c2c_strong_climb_probability", _C3A_JOINT_FROM_C1_STRONG_CLIMB_PROBABILITY, "--c2c-strong-climb-probability"),
        ("agent_device", "cpu", "--agent-device"),
    ):
        _set_preset_value(
            args,
            name=name,
            expected=expected,
            option=option,
            preset_option=preset_option,
        )

    if float(getattr(args, "actor_distillation_coefficient", 0.0)) != 0.0:
        raise ValueError(f"{preset_option} requires --actor-distillation-coefficient=0.")
    for name, expected, option in (
        ("source_stage", _C3A_SPLIT_FREQUENCY_ACTOR_SOURCE_STAGE, "--source-stage"),
        ("load_run", _C3A_SPLIT_FREQUENCY_ACTOR_SOURCE_RUN, "--load_run"),
        ("checkpoint", _C3A_SPLIT_FREQUENCY_ACTOR_CHECKPOINT, "--checkpoint"),
    ):
        configured = str(getattr(args, name, "") or "").strip()
        if configured not in ("", expected):
            raise ValueError(f"{preset_option} requires {option}={expected}.")
        setattr(args, name, expected)

    args.load_weights_only = True
    args.native_cpu = True
    args.train_only = True
    args.concurrent_eval = False
    return args


def _apply_c3b_split_frequency_actor_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the frozen C3b split-actor joint training recipe."""

    if not bool(getattr(args, "c3b_split_frequency_actor", False)):
        return args
    preset_option = "--c3b-split-frequency-actor"
    if str(getattr(args, "task", "")) != _C3B_TASK_ID:
        raise ValueError(f"{preset_option} requires the measured CPU-native C3b task.")
    for name, option in (
        ("c3a_retention_phase_a", "--c3a-retention-phase-a"),
        ("c3a_large_actor", "--c3a-large-actor"),
        ("c3a_joint_from_c1", "--c3a-joint-from-c1"),
        ("c3a_joint_requested_frequency_smoothness", "--c3a-joint-requested-frequency-smoothness"),
        ("c3a_joint_requested_frequency_total_variation", "--c3a-joint-requested-frequency-total-variation"),
        ("c3a_joint_requested_applied_frequency_gap", "--c3a-joint-requested-applied-frequency-gap"),
        ("c3a_split_frequency_actor", "--c3a-split-frequency-actor"),
        ("c3b_adaptive_sampling", "--c3b-adaptive-sampling"),
        ("c3b_yaw_consistency", "--c3b-yaw-consistency"),
        ("c3b_heading_canonical_observation", "--c3b-heading-canonical-observation"),
        ("resume", "--resume"),
        ("concurrent_eval", "--concurrent-eval"),
        ("adaptive_task_sampling", "--adaptive-task-sampling"),
        ("c2c_recycle_on_recovery", "--c2c-recycle-on-recovery"),
    ):
        if bool(getattr(args, name, False)):
            raise ValueError(f"{preset_option} cannot be combined with {option}.")

    for name, expected, option in (
        ("num_envs", _MEASURED_PURE_RL_DEFAULT_NUM_ENVS, "--num-envs"),
        ("max_iterations", _C3B_SPLIT_FREQUENCY_ACTOR_MAX_ITERATIONS, "--max-iterations"),
        ("save_interval", _MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL, "--save-interval"),
        ("seed", 0, "--seed"),
        ("agent_num_mini_batches", _MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES, "--agent-num-mini-batches"),
        ("c2c_strong_climb_probability", 0.5, "--c2c-strong-climb-probability"),
        ("agent_device", "cpu", "--agent-device"),
    ):
        _set_preset_value(
            args,
            name=name,
            expected=expected,
            option=option,
            preset_option=preset_option,
        )
    if float(getattr(args, "actor_distillation_coefficient", 0.0)) != 0.0:
        raise ValueError(f"{preset_option} requires --actor-distillation-coefficient=0.")
    for name, expected, option in (
        ("source_stage", _C3B_SPLIT_FREQUENCY_ACTOR_SOURCE_STAGE, "--source-stage"),
        ("load_run", _C3B_SPLIT_FREQUENCY_ACTOR_SOURCE_RUN, "--load_run"),
        ("checkpoint", _C3B_SPLIT_FREQUENCY_ACTOR_CHECKPOINT, "--checkpoint"),
    ):
        configured = str(getattr(args, name, "") or "").strip()
        if configured not in ("", expected):
            raise ValueError(f"{preset_option} requires {option}={expected}.")
        setattr(args, name, expected)

    args.load_weights_only = True
    args.native_cpu = True
    args.train_only = True
    args.concurrent_eval = False
    return args


def _apply_c3b_adaptive_sampling_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the controlled C3b fine-grained adaptive-sampling continuation."""

    adaptive_sampling = bool(getattr(args, "c3b_adaptive_sampling", False))
    yaw_consistency = bool(getattr(args, "c3b_yaw_consistency", False))
    heading_canonical = bool(getattr(args, "c3b_heading_canonical_observation", False))
    selected = sum((adaptive_sampling, yaw_consistency, heading_canonical))
    if selected == 0:
        return args
    if selected > 1:
        raise ValueError(
            "Select only one C3b adaptive continuation preset: sampling, yaw consistency, "
            "or heading-canonical observation."
        )
    if heading_canonical:
        preset_option = "--c3b-heading-canonical-observation"
    elif yaw_consistency:
        preset_option = "--c3b-yaw-consistency"
    else:
        preset_option = "--c3b-adaptive-sampling"
    if str(getattr(args, "task", "")) != _C3B_TASK_ID:
        raise ValueError(f"{preset_option} requires the measured CPU-native C3b task.")
    for name, option in (
        ("c3a_retention_phase_a", "--c3a-retention-phase-a"),
        ("c3a_large_actor", "--c3a-large-actor"),
        ("c3a_joint_from_c1", "--c3a-joint-from-c1"),
        ("c3a_joint_requested_frequency_smoothness", "--c3a-joint-requested-frequency-smoothness"),
        ("c3a_joint_requested_frequency_total_variation", "--c3a-joint-requested-frequency-total-variation"),
        ("c3a_joint_requested_applied_frequency_gap", "--c3a-joint-requested-applied-frequency-gap"),
        ("c3a_split_frequency_actor", "--c3a-split-frequency-actor"),
        ("c3b_split_frequency_actor", "--c3b-split-frequency-actor"),
        ("resume", "--resume"),
        ("concurrent_eval", "--concurrent-eval"),
        ("adaptive_task_sampling", "--adaptive-task-sampling"),
        ("c2c_recycle_on_recovery", "--c2c-recycle-on-recovery"),
    ):
        if bool(getattr(args, name, False)):
            raise ValueError(f"{preset_option} cannot be combined with {option}.")

    for name, expected, option in (
        ("num_envs", _MEASURED_PURE_RL_DEFAULT_NUM_ENVS, "--num-envs"),
        ("max_iterations", _C3B_ADAPTIVE_SAMPLING_MAX_ITERATIONS, "--max-iterations"),
        ("save_interval", _MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL, "--save-interval"),
        ("seed", 0, "--seed"),
        ("agent_num_mini_batches", _MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES, "--agent-num-mini-batches"),
        ("c2c_strong_climb_probability", 0.5, "--c2c-strong-climb-probability"),
        ("agent_device", "cpu", "--agent-device"),
    ):
        _set_preset_value(
            args,
            name=name,
            expected=expected,
            option=option,
            preset_option=preset_option,
        )
    if float(getattr(args, "actor_distillation_coefficient", 0.0)) != 0.0:
        raise ValueError(f"{preset_option} requires --actor-distillation-coefficient=0.")
    for name, expected, option in (
        ("source_stage", _C3B_ADAPTIVE_SAMPLING_SOURCE_STAGE, "--source-stage"),
        ("load_run", _C3B_ADAPTIVE_SAMPLING_SOURCE_RUN, "--load_run"),
        ("checkpoint", _C3B_ADAPTIVE_SAMPLING_CHECKPOINT, "--checkpoint"),
    ):
        configured = str(getattr(args, name, "") or "").strip()
        if configured not in ("", expected):
            raise ValueError(f"{preset_option} requires {option}={expected}.")
        setattr(args, name, expected)

    args.adaptive_task_sampling = True
    args.load_weights_only = True
    args.native_cpu = True
    args.train_only = True
    args.concurrent_eval = False
    return args


def _apply_c3c_joint_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the bounded C3c joint continuation from promoted C3b."""

    if not bool(getattr(args, "c3c_joint_from_promoted_c3b", False)):
        return args
    preset_option = "--c3c-joint-from-promoted-c3b"
    if str(getattr(args, "task", "")) != _C3C_TASK_ID:
        raise ValueError(f"{preset_option} requires the measured CPU-native C3c task.")
    for name, option in (
        ("c3a_retention_phase_a", "--c3a-retention-phase-a"),
        ("c3a_large_actor", "--c3a-large-actor"),
        ("c3a_joint_from_c1", "--c3a-joint-from-c1"),
        ("c3a_split_frequency_actor", "--c3a-split-frequency-actor"),
        ("c3b_split_frequency_actor", "--c3b-split-frequency-actor"),
        ("c3b_adaptive_sampling", "--c3b-adaptive-sampling"),
        ("c3b_yaw_consistency", "--c3b-yaw-consistency"),
        ("c3b_heading_canonical_observation", "--c3b-heading-canonical-observation"),
        ("resume", "--resume"),
        ("concurrent_eval", "--concurrent-eval"),
        ("adaptive_task_sampling", "--adaptive-task-sampling"),
        ("c2c_recycle_on_recovery", "--c2c-recycle-on-recovery"),
    ):
        if bool(getattr(args, name, False)):
            raise ValueError(f"{preset_option} cannot be combined with {option}.")
    for name, expected, option in (
        ("num_envs", _MEASURED_PURE_RL_DEFAULT_NUM_ENVS, "--num-envs"),
        ("max_iterations", _C3C_JOINT_MAX_ITERATIONS, "--max-iterations"),
        ("save_interval", _MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL, "--save-interval"),
        ("seed", 0, "--seed"),
        ("agent_num_mini_batches", _MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES, "--agent-num-mini-batches"),
        ("c2c_strong_climb_probability", 0.5, "--c2c-strong-climb-probability"),
        ("agent_device", "cpu", "--agent-device"),
    ):
        _set_preset_value(
            args,
            name=name,
            expected=expected,
            option=option,
            preset_option=preset_option,
        )
    if float(getattr(args, "actor_distillation_coefficient", 0.0)) != 0.0:
        raise ValueError(f"{preset_option} requires --actor-distillation-coefficient=0.")
    for name, expected, option in (
        ("source_stage", _C3C_JOINT_SOURCE_STAGE, "--source-stage"),
        ("load_run", _C3C_JOINT_SOURCE_RUN, "--load_run"),
        ("checkpoint", _C3C_JOINT_CHECKPOINT, "--checkpoint"),
    ):
        configured = str(getattr(args, name, "") or "").strip()
        if configured not in ("", expected):
            raise ValueError(f"{preset_option} requires {option}={expected}.")
        setattr(args, name, expected)
    args.load_weights_only = True
    args.native_cpu = True
    args.train_only = True
    args.concurrent_eval = False
    return args


def _apply_c3_full_joint_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply complete C3 joint training from the promoted C2c actor."""

    if not bool(getattr(args, "c3_full_joint_from_c2c", False)):
        return args
    preset_option = "--c3-full-joint-from-c2c"
    if str(getattr(args, "task", "")) != _C3C_TASK_ID:
        raise ValueError(f"{preset_option} requires the measured CPU-native C3c task.")
    for name, option in (
        ("c3a_retention_phase_a", "--c3a-retention-phase-a"),
        ("c3a_large_actor", "--c3a-large-actor"),
        ("c3a_joint_from_c1", "--c3a-joint-from-c1"),
        ("c3a_split_frequency_actor", "--c3a-split-frequency-actor"),
        ("c3b_split_frequency_actor", "--c3b-split-frequency-actor"),
        ("c3b_adaptive_sampling", "--c3b-adaptive-sampling"),
        ("c3b_yaw_consistency", "--c3b-yaw-consistency"),
        ("c3b_heading_canonical_observation", "--c3b-heading-canonical-observation"),
        ("c3c_joint_from_promoted_c3b", "--c3c-joint-from-promoted-c3b"),
        ("resume", "--resume"),
        ("concurrent_eval", "--concurrent-eval"),
        ("adaptive_task_sampling", "--adaptive-task-sampling"),
        ("c2c_recycle_on_recovery", "--c2c-recycle-on-recovery"),
    ):
        if bool(getattr(args, name, False)):
            raise ValueError(f"{preset_option} cannot be combined with {option}.")
    for name, expected, option in (
        ("num_envs", _MEASURED_PURE_RL_DEFAULT_NUM_ENVS, "--num-envs"),
        ("max_iterations", _C3_FULL_JOINT_MAX_ITERATIONS, "--max-iterations"),
        ("save_interval", _MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL, "--save-interval"),
        ("seed", 0, "--seed"),
        ("agent_num_mini_batches", _MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES, "--agent-num-mini-batches"),
        ("c2c_strong_climb_probability", 0.5, "--c2c-strong-climb-probability"),
        ("agent_device", "cpu", "--agent-device"),
    ):
        _set_preset_value(
            args,
            name=name,
            expected=expected,
            option=option,
            preset_option=preset_option,
        )
    if float(getattr(args, "actor_distillation_coefficient", 0.0)) != 0.0:
        raise ValueError(f"{preset_option} requires --actor-distillation-coefficient=0.")
    for name, expected, option in (
        ("source_stage", _C3_FULL_JOINT_SOURCE_STAGE, "--source-stage"),
        ("load_run", _C3_FULL_JOINT_SOURCE_RUN, "--load_run"),
        ("checkpoint", _C3_FULL_JOINT_CHECKPOINT, "--checkpoint"),
    ):
        configured = str(getattr(args, name, "") or "").strip()
        if configured not in ("", expected):
            raise ValueError(f"{preset_option} requires {option}={expected}.")
        setattr(args, name, expected)
    args.load_weights_only = True
    args.native_cpu = True
    args.train_only = True
    args.concurrent_eval = False
    return args


def _extract_ckpt_index(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _load_completed_checkpoints(summary_csv: Path) -> set[str]:
    completed: set[str] = set()
    if not summary_csv.is_file():
        return completed

    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint = row.get("checkpoint")
            case_name = row.get("case")
            if checkpoint and case_name == "suite":
                completed.add(str(Path(checkpoint).expanduser().resolve()))
    return completed


def _latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=_extract_ckpt_index)
    if not ckpts:
        return None
    return ckpts[-1].resolve()


def _start_output_forwarder(
    stream: TextIO,
    *,
    sink: Callable[[str], None] | None = None,
) -> threading.Thread:
    """Drain a child stream in the background to prevent pipe backpressure."""

    if sink is None:

        def sink(line: str) -> None:
            sys.stdout.write(line)
            sys.stdout.flush()

    def _forward() -> None:
        for line in stream:
            sink(line)

    thread = threading.Thread(target=_forward, name="train-output-forwarder", daemon=True)
    thread.start()
    return thread


def _wait_for_run_dir(run_dir: Path, train: subprocess.Popen, *, timeout_s: float, poll_s: float = 0.5) -> None:
    t0 = time.time()
    while not run_dir.is_dir():
        if train.poll() is not None:
            raise RuntimeError("Training exited before creating a run directory.")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for run dir: {run_dir}")
        time.sleep(poll_s)


def _needs_final_eval(run_dir: Path) -> bool:
    latest_ckpt = _latest_checkpoint(run_dir)
    if latest_ckpt is None:
        return False
    completed = _load_completed_checkpoints(run_dir / "eval" / "summary.csv")
    return str(latest_ckpt) not in completed


def _safe_terminate(p: subprocess.Popen, timeout_s: float = 10.0):
    if p.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
        else:
            p.send_signal(signal.SIGINT)
        p.wait(timeout=timeout_s)
        return
    except Exception:
        pass

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        else:
            p.terminate()
        p.wait(timeout=timeout_s)
        return
    except Exception:
        pass

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        else:
            p.kill()
        p.wait(timeout=timeout_s)
    except Exception:
        p.kill()


def _run_final_eval_once(watch_cmd: list[str], *, child_env: dict[str, str] | None = None) -> int:
    final_cmd = list(watch_cmd)
    if "--once" not in final_cmd:
        final_cmd.append("--once")
    print("[INFO] Running final one-shot evaluation:", flush=True)
    print(" ", " ".join(final_cmd), flush=True)
    return subprocess.call(final_cmd, env=child_env)


def _resolve_portable_root_base(args: argparse.Namespace) -> Path:
    configured = getattr(args, "portable_root_base", None)
    if configured is not None:
        return Path(configured)
    return Path("logs/portable/train_and_watch") / f"{args.run_name}_seed{args.seed}_pid{os.getpid()}"


def _portable_root_for_role(args: argparse.Namespace, role: str) -> Path:
    return _resolve_portable_root_base(args) / role


def _portable_kit_args(args: argparse.Namespace, role: str) -> str:
    return f"--portable-root {_portable_root_for_role(args, role)}"


def _native_cpu_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "native_cpu", False)) or is_measured_pure_rl_task(
        str(getattr(args, "task", ""))
    )


def _watcher_enabled(args: argparse.Namespace) -> bool:
    train_only = bool(getattr(args, "train_only", False))
    concurrent_eval = bool(getattr(args, "concurrent_eval", False))
    if train_only and concurrent_eval:
        raise ValueError("--train-only and --concurrent-eval cannot both be enabled.")
    if train_only:
        return False
    if concurrent_eval:
        return True
    return not is_measured_pure_rl_task(str(getattr(args, "task", "")))


def _resolved_train_num_envs(args: argparse.Namespace) -> int:
    configured = getattr(args, "num_envs", None)
    if configured is None:
        configured = (
            _MEASURED_PURE_RL_DEFAULT_NUM_ENVS
            if is_measured_pure_rl_task(str(getattr(args, "task", "")))
            else _DEFAULT_NUM_ENVS
        )
    value = int(configured)
    if value <= 0:
        raise ValueError("--num-envs must be positive.")
    return value


def _resolved_max_iterations(args: argparse.Namespace) -> int:
    configured = getattr(args, "max_iterations", None)
    if configured is None:
        configured = (
            _MEASURED_PURE_RL_DEFAULT_MAX_ITERATIONS
            if is_measured_pure_rl_task(str(getattr(args, "task", "")))
            else _DEFAULT_MAX_ITERATIONS
        )
    value = int(configured)
    if value <= 0:
        raise ValueError("--max-iterations must be positive.")
    return value


def _resolved_save_interval(args: argparse.Namespace) -> int:
    configured = getattr(args, "save_interval", None)
    if configured is None:
        configured = (
            _MEASURED_PURE_RL_DEFAULT_SAVE_INTERVAL
            if is_measured_pure_rl_task(str(getattr(args, "task", "")))
            else _DEFAULT_SAVE_INTERVAL
        )
    value = int(configured)
    if value <= 0:
        raise ValueError("--save-interval must be positive.")
    return value


def _resolved_agent_num_mini_batches(args: argparse.Namespace) -> int | None:
    configured = getattr(args, "agent_num_mini_batches", None)
    if configured is None and is_measured_pure_rl_task(str(getattr(args, "task", ""))):
        configured = _MEASURED_PURE_RL_DEFAULT_NUM_MINI_BATCHES
    if configured is None:
        return None
    value = int(configured)
    if value <= 0:
        raise ValueError("--agent-num-mini-batches must be positive.")
    return value


def _native_extension_parent(args: argparse.Namespace) -> Path:
    configured = getattr(args, "native_extension_parent", None)
    if configured is not None:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "source/flapping_bot/native_extensions").resolve()


def _validate_native_extension(args: argparse.Namespace) -> None:
    if not _native_cpu_enabled(args):
        return
    extension_parent = _native_extension_parent(args)
    extension_binary = (
        extension_parent
        / _NATIVE_HOLONOMIC_EXTENSION_ID
        / "omni/flapping_bot/holonomic_constraint/_native.so"
    )
    if not extension_binary.is_file():
        raise FileNotFoundError(
            f"Native extension binary does not exist: {extension_binary}. "
            "Build it with scripts/flapping_px4/build_holonomic_constraint_extension.sh."
        )


def _child_entrypoint(args: argparse.Namespace, script: str) -> list[str]:
    if _native_cpu_enabled(args):
        return [sys.executable, script]
    return ["./isaaclab.sh", "-p", script]


def _native_asset_source() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    ).resolve()


def _native_asset_cache(args: argparse.Namespace, role: str = "train") -> Path:
    return (_portable_root_for_role(args, role) / "generated_assets/flap_robot_552").resolve()


def _child_process_env(args: argparse.Namespace) -> dict[str, str]:
    child_env = os.environ.copy()
    if not _native_cpu_enabled(args):
        return child_env
    repo_root = Path(__file__).resolve().parents[2]
    local_sources = [
        str((repo_root / "source/flapping_bot").resolve()),
        str((repo_root / "source/isaaclab_assets").resolve()),
        str((repo_root / "source/isaaclab_tasks").resolve()),
    ]
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        local_sources.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(local_sources)
    return child_env


def _kit_args(args: argparse.Namespace, role: str) -> str:
    kit_args = _portable_kit_args(args, role)
    if bool(getattr(args, "disable_kit_fs_watcher", False)):
        kit_args += " --/apps/extensions/fsWatcherEnabled=false"
    if _native_cpu_enabled(args):
        repo_source = (Path(__file__).resolve().parents[2] / "source").resolve()
        kit_args += (
            f" --ext-folder {repo_source}"
            f" --ext-folder {_native_extension_parent(args)}"
            f" --enable {_NATIVE_HOLONOMIC_EXTENSION_ID}"
        )
    return kit_args


def _sim_device(args: argparse.Namespace, role: str) -> str:
    if _native_cpu_enabled(args):
        return "cpu"
    return str(getattr(args, f"{role}_device"))


def _build_train_cmd(args: argparse.Namespace) -> list[str]:
    if bool(args.resume) and bool(getattr(args, "load_weights_only", False)):
        raise ValueError("--resume and --load_weights_only cannot both be enabled.")
    large_actor_enabled = bool(getattr(args, "c3a_large_actor", False))
    if large_actor_enabled:
        if spatial_stage_for_task(str(args.task)) != "c3a":
            raise ValueError("--c3a-large-actor is currently supported only for C3a training.")
        if not bool(getattr(args, "load_weights_only", False)):
            raise ValueError("--c3a-large-actor requires --load_weights_only from the promoted C2c actor.")

    train_cmd = [
        *_child_entrypoint(args, "scripts/reinforcement_learning/rsl_rl/train.py"),
        "--task",
        args.task,
        "--device",
        _sim_device(args, "train"),
        "--num_envs",
        str(_resolved_train_num_envs(args)),
        "--max_iterations",
        str(_resolved_max_iterations(args)),
        "--seed",
        str(args.seed),
        f"agent.run_name={args.run_name}",
        f"agent.save_interval={_resolved_save_interval(args)}",
    ]
    if bool(args.resume):
        train_cmd.append("--resume")
    if bool(getattr(args, "load_weights_only", False)):
        train_cmd.append("--load_weights_only")
    if args.load_run is not None:
        train_cmd.extend(["--load_run", str(args.load_run)])
    if args.checkpoint is not None:
        train_cmd.extend(["--checkpoint", str(args.checkpoint)])
    train_cmd.extend(["--kit_args", _kit_args(args, "train")])
    agent_device = getattr(args, "agent_device", None)
    if _native_cpu_enabled(args):
        agent_device = agent_device or "cpu"
    if agent_device is not None:
        train_cmd.append(f"agent.device={agent_device}")
    agent_num_mini_batches = _resolved_agent_num_mini_batches(args)
    if agent_num_mini_batches is not None:
        train_cmd.append(f"agent.algorithm.num_mini_batches={agent_num_mini_batches}")
    freeze_steps_after_reset = getattr(args, "freeze_steps_after_reset", None)
    if freeze_steps_after_reset is not None:
        if int(freeze_steps_after_reset) < 0:
            raise ValueError("--freeze-steps-after-reset must be non-negative.")
        train_cmd.append(f"env.freeze_steps_after_reset={int(freeze_steps_after_reset)}")
    strong_climb_probability = getattr(args, "c2c_strong_climb_probability", None)
    if strong_climb_probability is not None:
        probability = float(strong_climb_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("--c2c-strong-climb-probability must lie in [0, 1].")
        train_cmd.append(f"env.pure_rl_c2c_strong_climb_probability={probability}")
    if bool(getattr(args, "c2c_recycle_on_recovery", False)):
        train_cmd.append("env.pure_rl_c2c_recycle_on_recovery=true")
    if bool(getattr(args, "adaptive_task_sampling", False)):
        if spatial_stage_for_task(str(args.task)) not in {"c3a", "c3b"}:
            raise ValueError("Adaptive task sampling is supported only for C3a or C3b training.")
        if strong_climb_probability is None:
            raise ValueError(
                "Adaptive task sampling requires an explicit --c2c-strong-climb-probability."
            )
        train_cmd.append("env.pure_rl_adaptive_task_sampling_enabled=true")
    actor_distillation_coefficient = float(getattr(args, "actor_distillation_coefficient", 0.0))
    if not math.isfinite(actor_distillation_coefficient) or actor_distillation_coefficient < 0.0:
        raise ValueError("--actor-distillation-coefficient must be finite and non-negative.")
    if actor_distillation_coefficient > 0.0:
        if spatial_stage_for_task(str(args.task)) != "c3a":
            raise ValueError("Actor policy distillation is currently supported only for C3a training.")
        if not bool(getattr(args, "load_weights_only", False)):
            raise ValueError("Actor policy distillation requires --load_weights_only to define the frozen teacher.")
        train_cmd.append(
            f"env.pure_rl_actor_distillation_coefficient={actor_distillation_coefficient}"
        )
    if bool(getattr(args, "c3b_yaw_consistency", False)):
        if spatial_stage_for_task(str(args.task)) != "c3b":
            raise ValueError("--c3b-yaw-consistency is supported only for C3b training.")
        if not bool(getattr(args, "load_weights_only", False)):
            raise ValueError("--c3b-yaw-consistency requires a weights-only warm start.")
        train_cmd.append(
            "env.pure_rl_actor_yaw_consistency_coefficient="
            f"{_C3B_YAW_CONSISTENCY_COEFFICIENT}"
        )
    heading_canonical_observation = bool(
        getattr(args, "c3b_heading_canonical_observation", False)
    ) or bool(getattr(args, "c3c_joint_from_promoted_c3b", False)) or bool(
        getattr(args, "c3_full_joint_from_c2c", False)
    )
    if heading_canonical_observation:
        if spatial_stage_for_task(str(args.task)) not in {"c3b", "c3c"}:
            raise ValueError(
                "Heading-canonical observation is supported only for C3b/C3c training."
            )
        if not bool(getattr(args, "load_weights_only", False)):
            raise ValueError(
                "Heading-canonical observation requires a weights-only warm start."
            )
        train_cmd.append("env.pure_rl_heading_canonical_observation=true")
    if large_actor_enabled:
        hidden_dims = ",".join(str(value) for value in _C3A_LARGE_ACTOR_HIDDEN_DIMS)
        train_cmd.append(f"agent.policy.actor_hidden_dims=[{hidden_dims}]")
    if bool(getattr(args, "c3a_joint_from_c1", False)):
        hidden_dims = ",".join(str(value) for value in _C3A_BASE_POLICY_HIDDEN_DIMS)
        train_cmd.extend(
            [
                f"agent.policy.actor_hidden_dims=[{hidden_dims}]",
                f"agent.policy.critic_hidden_dims=[{hidden_dims}]",
            ]
        )
    if bool(getattr(args, "c3a_joint_requested_frequency_smoothness", False)):
        train_cmd.append(
            "env.pure_rl_reward_cfg.requested_frequency_action_delta_penalty_weight="
            f"{_C3A_JOINT_REQUESTED_FREQUENCY_SQUARED_DELTA_PENALTY_WEIGHT}"
        )
    if bool(getattr(args, "c3a_joint_requested_frequency_total_variation", False)):
        train_cmd.extend(
            [
                "env.pure_rl_reward_cfg.requested_frequency_action_delta_penalty_mode=absolute",
                "env.pure_rl_reward_cfg.requested_frequency_action_delta_penalty_weight="
                f"{_C3A_JOINT_REQUESTED_FREQUENCY_TOTAL_VARIATION_PENALTY_WEIGHT}",
            ]
        )
    if bool(getattr(args, "c3a_joint_requested_applied_frequency_gap", False)):
        train_cmd.append(
            "env.pure_rl_reward_cfg.requested_applied_frequency_action_gap_penalty_weight="
            f"{_C3A_JOINT_REQUESTED_APPLIED_FREQUENCY_GAP_PENALTY_WEIGHT}"
        )
    if bool(getattr(args, "c3_full_joint_from_c2c", False)):
        task_weights = ",".join(str(value) for value in _C3_FULL_JOINT_TASK_WEIGHTS)
        train_cmd.extend(
            [
                "env.pure_rl_full_c3_joint_training_enabled=true",
                f"env.pure_rl_task_aware_ppo_task_weights=[{task_weights}]",
            ]
        )
    if bool(getattr(args, "c3a_split_frequency_actor", False)) or bool(
        getattr(args, "c3b_split_frequency_actor", False)
    ) or bool(getattr(args, "c3b_adaptive_sampling", False)) or bool(
        getattr(args, "c3b_yaw_consistency", False)
    ) or bool(
        getattr(args, "c3b_heading_canonical_observation", False)
    ) or bool(
        getattr(args, "c3c_joint_from_promoted_c3b", False)
    ) or bool(
        getattr(args, "c3_full_joint_from_c2c", False)
    ):
        hidden_dims = ",".join(str(value) for value in _C3A_BASE_POLICY_HIDDEN_DIMS)
        train_cmd.extend(
            [
                f"agent.policy.class_name={_C3A_SPLIT_FREQUENCY_ACTOR_CLASS_NAME}",
                f"agent.policy.actor_hidden_dims=[{hidden_dims}]",
                f"agent.policy.critic_hidden_dims=[{hidden_dims}]",
                "env.pure_rl_reward_cfg.requested_applied_frequency_action_gap_penalty_weight="
                f"{_C3A_JOINT_REQUESTED_APPLIED_FREQUENCY_GAP_PENALTY_WEIGHT}",
            ]
        )
    if _native_cpu_enabled(args):
        train_cmd.extend(
            [
                f"env.robot.spawn.asset_path={_native_asset_source()}",
                f"env.robot.spawn.usd_dir={_native_asset_cache(args)}",
            ]
        )
    if _should_apply_estimated_teacher_defaults(args.task):
        train_cmd.extend(
            [
                "env.teacher_state_source=estimated",
                "env.policy_state_source=estimated",
                "env.imu_source=synthetic",
            ]
        )
    mass_kg_override = getattr(args, "mass_kg_override", None)
    if mass_kg_override is not None:
        train_cmd.append(f"env.total_mass_kg_override={float(mass_kg_override)}")
    if args.headless:
        train_cmd.append("--headless")
    return train_cmd


def _build_watch_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    eval_num_envs, episodes = _resolve_eval_shape(args)
    watch_cmd = [
        *_child_entrypoint(args, "scripts/flapping_rl/watch_and_eval.py"),
        "--task",
        args.task,
        "--log_dir",
        str(run_dir),
        "--device",
        _sim_device(args, "eval"),
        "--episodes",
        str(episodes),
        "--num_envs",
        str(eval_num_envs),
        "--poll_s",
        str(args.poll_s),
        "--eval_suite",
        _resolve_eval_suite(args.task, str(args.eval_suite)),
    ]
    if is_measured_pure_rl_task(args.task):
        watch_cmd.append("--no_saved_cfg")
    watch_cmd.extend(["--kit_args", _kit_args(args, "watch")])
    if _native_cpu_enabled(args):
        watch_cmd.extend(
            [
                "--robot-asset-path",
                str(_native_asset_source()),
                "--robot-usd-dir",
                str(_native_asset_cache(args, "watch")),
            ]
        )
    if bool(getattr(args, "c3a_split_frequency_actor", False)) or bool(
        getattr(args, "c3b_split_frequency_actor", False)
    ) or bool(getattr(args, "c3b_adaptive_sampling", False)) or bool(
        getattr(args, "c3b_yaw_consistency", False)
    ) or bool(
        getattr(args, "c3b_heading_canonical_observation", False)
    ) or bool(
        getattr(args, "c3c_joint_from_promoted_c3b", False)
    ) or bool(
        getattr(args, "c3_full_joint_from_c2c", False)
    ):
        watch_cmd.append("--pure-rl-split-frequency-actor")
    if bool(getattr(args, "c3b_heading_canonical_observation", False)) or bool(
        getattr(args, "c3c_joint_from_promoted_c3b", False)
    ) or bool(
        getattr(args, "c3_full_joint_from_c2c", False)
    ):
        watch_cmd.append("--pure-rl-heading-canonical-observation")
    if args.headless:
        watch_cmd.append("--headless")
    return watch_cmd


def _build_curriculum_source_metadata(args: argparse.Namespace) -> dict[str, str] | None:
    """Validate cross-stage warm-start provenance; same-stage resumes keep their run metadata."""

    target_stage = curriculum_stage_for_task(str(getattr(args, "task", "")))
    if target_stage is None:
        return None
    if bool(getattr(args, "resume", False)):
        return None
    expected_source_stage = {
        "c2a": "c1_straight",
        "c2b": "c2a",
        "c2c": "c2b",
        "c3a": "c2c",
        "c3b": "c3a",
        "c3c": "c3b",
    }[target_stage]
    joint_from_c1 = bool(getattr(args, "c3a_joint_from_c1", False))
    split_frequency_actor = bool(getattr(args, "c3a_split_frequency_actor", False))
    c3b_split_frequency_actor = bool(getattr(args, "c3b_split_frequency_actor", False))
    c3b_adaptive_sampling = bool(getattr(args, "c3b_adaptive_sampling", False))
    c3b_yaw_consistency = bool(getattr(args, "c3b_yaw_consistency", False))
    c3b_heading_canonical = bool(
        getattr(args, "c3b_heading_canonical_observation", False)
    )
    c3c_joint = bool(getattr(args, "c3c_joint_from_promoted_c3b", False))
    c3_full_joint = bool(getattr(args, "c3_full_joint_from_c2c", False))
    if c3_full_joint:
        if target_stage != "c3c":
            raise ValueError("Full C3 joint training is valid only for target stage c3c.")
        expected_source_stage = _C3_FULL_JOINT_SOURCE_STAGE
    elif c3c_joint:
        if target_stage != "c3c":
            raise ValueError("C3c joint continuation is valid only for target stage c3c.")
        expected_source_stage = _C3C_JOINT_SOURCE_STAGE
    elif c3b_adaptive_sampling or c3b_yaw_consistency or c3b_heading_canonical:
        if target_stage != "c3b":
            raise ValueError("C3b adaptive continuations are valid only for target stage c3b.")
        expected_source_stage = _C3B_ADAPTIVE_SAMPLING_SOURCE_STAGE
    elif c3b_split_frequency_actor:
        if target_stage != "c3b":
            raise ValueError("--c3b-split-frequency-actor is valid only for target stage c3b.")
        expected_source_stage = _C3B_SPLIT_FREQUENCY_ACTOR_SOURCE_STAGE
    elif split_frequency_actor:
        if target_stage != "c3a":
            raise ValueError("--c3a-split-frequency-actor is valid only for target stage c3a.")
        expected_source_stage = _C3A_SPLIT_FREQUENCY_ACTOR_SOURCE_STAGE
    elif joint_from_c1:
        if target_stage != "c3a":
            raise ValueError("--c3a-joint-from-c1 is valid only for target stage c3a.")
        expected_source_stage = _C3A_JOINT_FROM_C1_SOURCE_STAGE
    source_stage = str(getattr(args, "source_stage", "") or "").strip()
    if source_stage != expected_source_stage:
        raise ValueError(
            f"Target stage {target_stage} must use source stage {expected_source_stage}; "
            f"received {source_stage or '<missing>'}."
        )
    configured_path = getattr(args, "source_checkpoint_path", None)
    if configured_path is None:
        raise ValueError("Curriculum training requires --source-checkpoint-path for provenance.")
    checkpoint_path = Path(configured_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {checkpoint_path}")
    if split_frequency_actor and (
        checkpoint_path.name != _C3A_SPLIT_FREQUENCY_ACTOR_CHECKPOINT
        or checkpoint_path.parent.name != _C3A_SPLIT_FREQUENCY_ACTOR_SOURCE_RUN
    ):
        raise ValueError(
            "--c3a-split-frequency-actor source path must resolve to the passing joint model_200.pt."
        )
    if c3b_split_frequency_actor and (
        checkpoint_path.name != _C3B_SPLIT_FREQUENCY_ACTOR_CHECKPOINT
        or checkpoint_path.parent.name != _C3B_SPLIT_FREQUENCY_ACTOR_SOURCE_RUN
    ):
        raise ValueError(
            "--c3b-split-frequency-actor source path must resolve to the promoted C3a split model_200.pt."
        )
    if (c3b_adaptive_sampling or c3b_yaw_consistency or c3b_heading_canonical) and (
        checkpoint_path.name != _C3B_ADAPTIVE_SAMPLING_CHECKPOINT
        or checkpoint_path.parent.name != _C3B_ADAPTIVE_SAMPLING_SOURCE_RUN
    ):
        raise ValueError(
            "C3b adaptive continuation source must resolve to the fixed-mixture C3b model_100.pt."
        )
    if c3c_joint and (
        checkpoint_path.name != _C3C_JOINT_CHECKPOINT
        or checkpoint_path.parent.name != _C3C_JOINT_SOURCE_RUN
    ):
        raise ValueError(
            "C3c joint source must resolve to the promoted heading-canonical C3b model_75.pt."
        )
    if c3_full_joint and (
        checkpoint_path.name != _C3_FULL_JOINT_CHECKPOINT
        or checkpoint_path.parent.name != _C3_FULL_JOINT_SOURCE_RUN
    ):
        raise ValueError(
            "Full C3 joint source must resolve to the promoted C2c model_550.pt."
        )
    if joint_from_c1 and (
        checkpoint_path.name != _C3A_JOINT_FROM_C1_CHECKPOINT
        or checkpoint_path.parent.name != _C3A_JOINT_FROM_C1_SOURCE_RUN
    ):
        raise ValueError(
            "--c3a-joint-from-c1 source path must resolve to the selected C1 run and model_1300.pt."
        )
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "target_stage": target_stage,
        "source_stage": source_stage,
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": digest.hexdigest(),
    }
    if c3_full_joint:
        metadata["curriculum_route"] = _C3_FULL_JOINT_ROUTE
        metadata["policy_class_name"] = _C3A_SPLIT_FREQUENCY_ACTOR_CLASS_NAME
        metadata["heading_canonical_observation"] = "true"
        metadata["task_probabilities"] = ",".join(
            str(value) for value in _C3_FULL_JOINT_TASK_WEIGHTS
        )
    elif c3c_joint:
        metadata["curriculum_route"] = _C3C_JOINT_ROUTE
        metadata["policy_class_name"] = _C3A_SPLIT_FREQUENCY_ACTOR_CLASS_NAME
        metadata["heading_canonical_observation"] = "true"
    elif c3b_adaptive_sampling or c3b_yaw_consistency or c3b_heading_canonical:
        if c3b_heading_canonical:
            metadata["curriculum_route"] = _C3B_HEADING_CANONICAL_OBSERVATION_ROUTE
        elif c3b_yaw_consistency:
            metadata["curriculum_route"] = _C3B_YAW_CONSISTENCY_ROUTE
        else:
            metadata["curriculum_route"] = _C3B_ADAPTIVE_SAMPLING_ROUTE
        metadata["policy_class_name"] = _C3A_SPLIT_FREQUENCY_ACTOR_CLASS_NAME
    elif c3b_split_frequency_actor:
        metadata["curriculum_route"] = _C3B_SPLIT_FREQUENCY_ACTOR_ROUTE
        metadata["policy_class_name"] = _C3A_SPLIT_FREQUENCY_ACTOR_CLASS_NAME
    elif split_frequency_actor:
        metadata["curriculum_route"] = _C3A_SPLIT_FREQUENCY_ACTOR_ROUTE
        metadata["policy_class_name"] = _C3A_SPLIT_FREQUENCY_ACTOR_CLASS_NAME
    elif joint_from_c1:
        if bool(getattr(args, "c3a_joint_requested_applied_frequency_gap", False)):
            metadata["curriculum_route"] = (
                "c3a_joint_from_c1_requested_applied_frequency_gap_v1"
            )
        elif bool(getattr(args, "c3a_joint_requested_frequency_total_variation", False)):
            metadata["curriculum_route"] = (
                "c3a_joint_from_c1_requested_frequency_total_variation_v1"
            )
        elif bool(getattr(args, "c3a_joint_requested_frequency_smoothness", False)):
            metadata["curriculum_route"] = (
                "c3a_joint_from_c1_requested_frequency_smoothness_v1"
            )
        else:
            metadata["curriculum_route"] = "c3a_joint_from_c1_v1"
    return metadata


def main():
    args = _apply_c3a_retention_phase_a_preset(_parse_args())
    args = _apply_c3a_joint_from_c1_preset(args)
    args = _apply_c3a_split_frequency_actor_preset(args)
    args = _apply_c3b_split_frequency_actor_preset(args)
    args = _apply_c3b_adaptive_sampling_preset(args)
    args = _apply_c3c_joint_preset(args)
    args = _apply_c3_full_joint_preset(args)
    curriculum_source_metadata = _build_curriculum_source_metadata(args)
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    args.portable_root_base = _resolve_portable_root_base(args)
    _validate_native_extension(args)
    _portable_root_for_role(args, "train").mkdir(parents=True, exist_ok=True)
    if _watcher_enabled(args):
        _portable_root_for_role(args, "watch").mkdir(parents=True, exist_ok=True)
    if _native_cpu_enabled(args):
        if not _native_asset_source().is_file():
            raise FileNotFoundError(f"Native PureRL URDF does not exist: {_native_asset_source()}")
        _native_asset_cache(args).mkdir(parents=True, exist_ok=True)
        if _watcher_enabled(args):
            _native_asset_cache(args, "watch").mkdir(parents=True, exist_ok=True)

    train_cmd = _build_train_cmd(args)
    child_env = _child_process_env(args)

    print("[INFO] Launching training:")
    print(" ", " ".join(train_cmd), flush=True)

    train = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=child_env,
    )

    log_root: Path | None = None
    timestamp: str | None = None
    run_dir: Path | None = None
    watch_cmd: list[str] | None = None
    final_eval_needed = False
    rc = 1
    try:
        assert train.stdout is not None
        for line in train.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            if log_root is None:
                match = re.search(r"Logging experiment in directory: (.+)$", line.strip())
                if match:
                    log_root = Path(match.group(1)).expanduser().resolve()
            if timestamp is None:
                match = re.search(r"Exact experiment name requested from command line: (\S+)$", line.strip())
                if match:
                    timestamp = match.group(1)
            if log_root is not None and timestamp is not None:
                break

        if log_root is None or timestamp is None:
            raise RuntimeError("Failed to parse log directory from training output.")

        run_dir = (log_root / f"{timestamp}_{args.run_name}").resolve()
        output_thread = _start_output_forwarder(train.stdout)
        _wait_for_run_dir(run_dir, train, timeout_s=float(args.run_dir_timeout_s))
        if curriculum_source_metadata is not None:
            (run_dir / "curriculum_source.json").write_text(
                json.dumps(curriculum_source_metadata, indent=2) + "\n",
                encoding="utf-8",
            )

        if _watcher_enabled(args):
            watch_cmd = _build_watch_cmd(args, run_dir)
            print("[INFO] Launching watcher:")
            print(" ", " ".join(watch_cmd), flush=True)
            watcher = subprocess.Popen(watch_cmd, start_new_session=True, env=child_env)
        else:
            print("[INFO] Train-only mode: checkpoint watcher disabled.", flush=True)

        rc = train.wait()
        output_thread.join(timeout=5.0)
        print(f"[INFO] Training finished with return code: {rc}", flush=True)
        final_eval_needed = rc == 0 and run_dir is not None and watch_cmd is not None
    except KeyboardInterrupt:
        print("[WARN] KeyboardInterrupt: stopping processes...", flush=True)
        rc = 130
    finally:
        try:
            if "watcher" in locals():
                _safe_terminate(watcher)
        except Exception:
            pass

        if final_eval_needed and watch_cmd is not None and run_dir is not None:
            if _needs_final_eval(run_dir):
                final_eval_rc = _run_final_eval_once(watch_cmd, child_env=child_env)
                if final_eval_rc != 0:
                    print(f"[WARN] Final one-shot evaluation exited with code: {final_eval_rc}", flush=True)
                    rc = final_eval_rc if rc == 0 else rc
            else:
                print("[INFO] Latest checkpoint already has a suite row; skipping final one-shot evaluation.", flush=True)

            stage_id = longitudinal_stage_for_task(args.task)
            spatial_stage_id = spatial_stage_for_task(args.task)
            evaluation_contract = (
                LONGITUDINAL_EVAL_CONTRACTS[stage_id]
                if stage_id is not None
                else (
                    SPATIAL_EVAL_CONTRACTS[spatial_stage_id]
                    if spatial_stage_id is not None
                    else (PURE_RL_CURRICULUM1_EVAL_CONTRACT if is_measured_pure_rl_task(args.task) else None)
                )
            )
            best_row = refresh_best_checkpoint_artifacts(
                run_dir,
                evaluation_contract=evaluation_contract,
            )
            if best_row is None:
                best_row = select_best_checkpoint_row(
                    run_dir / "eval" / "summary.csv",
                    evaluation_contract=evaluation_contract,
                )
            if best_row is not None:
                print(
                    "[INFO] Current best checkpoint:",
                    {
                        "checkpoint": best_row["checkpoint"],
                        "score": best_row["score"],
                        "ckpt_index": best_row.get("ckpt_index"),
                    },
                    flush=True,
                )

        _safe_terminate(train)
        if "output_thread" in locals():
            output_thread.join(timeout=5.0)

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
