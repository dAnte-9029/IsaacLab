"""Phase-isolated frequency branch for the measured PureRL actor."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP

Tensor = torch.Tensor

PURE_RL_POLICY_RATE_HZ = 60.0
PURE_RL_SENSOR_HISTORY_STEPS = 30
PURE_RL_SENSOR_FRAME_DIM = 14
PURE_RL_ACTION_HISTORY_STEPS = 30
PURE_RL_ACTION_DIM = 4
PURE_RL_PREVIEW_DIM = 15
PURE_RL_OBSERVATION_DIM = 555
PURE_RL_ACTUAL_FREQUENCY_INDEX = 11
PURE_RL_PHASE_SIN_INDEX = 12
PURE_RL_PHASE_COS_INDEX = 13
PURE_RL_MINIMUM_FREQUENCY_HZ = 0.0
PURE_RL_MAXIMUM_FREQUENCY_HZ = 5.0


def build_cycle_averaged_frequency_observation(observation: Tensor) -> Tensor:
    """Return a 555-value slow observation for the flap-frequency branch.

    The most recent actual frequency selects a 12--30 step window, equivalent
    to one wingbeat at the 60 Hz policy rate over the accepted 2--5 Hz range.
    Sensor and applied-action histories are averaged over that window and
    repeated to retain the frozen observation dimension. Quaternion averages
    are renormalized and phase is replaced by the fixed valid pair ``(0, 1)``.
    The path preview remains unchanged.
    """

    if observation.ndim != 2 or observation.shape[1] != PURE_RL_OBSERVATION_DIM:
        raise ValueError(
            f"PureRL split actor requires observation shape (N, {PURE_RL_OBSERVATION_DIM})."
        )
    if not torch.is_floating_point(observation):
        raise TypeError("PureRL split actor observations must use a floating dtype.")

    sensor_dim = PURE_RL_SENSOR_HISTORY_STEPS * PURE_RL_SENSOR_FRAME_DIM
    action_dim = PURE_RL_ACTION_HISTORY_STEPS * PURE_RL_ACTION_DIM
    sensor_history = observation[:, :sensor_dim].reshape(
        -1,
        PURE_RL_SENSOR_HISTORY_STEPS,
        PURE_RL_SENSOR_FRAME_DIM,
    )
    action_history = observation[:, sensor_dim : sensor_dim + action_dim].reshape(
        -1,
        PURE_RL_ACTION_HISTORY_STEPS,
        PURE_RL_ACTION_DIM,
    )
    preview = observation[:, sensor_dim + action_dim :]

    normalized_frequency = sensor_history[:, -1, PURE_RL_ACTUAL_FREQUENCY_INDEX]
    frequency_hz = (
        0.5
        * (normalized_frequency + 1.0)
        * (PURE_RL_MAXIMUM_FREQUENCY_HZ - PURE_RL_MINIMUM_FREQUENCY_HZ)
        + PURE_RL_MINIMUM_FREQUENCY_HZ
    )
    minimum_window_frequency_hz = PURE_RL_POLICY_RATE_HZ / PURE_RL_SENSOR_HISTORY_STEPS
    cycle_steps = torch.round(
        PURE_RL_POLICY_RATE_HZ / torch.clamp(frequency_hz, min=minimum_window_frequency_hz)
    ).to(dtype=torch.long)
    cycle_steps = torch.clamp(
        cycle_steps,
        min=int(PURE_RL_POLICY_RATE_HZ / PURE_RL_MAXIMUM_FREQUENCY_HZ),
        max=PURE_RL_SENSOR_HISTORY_STEPS,
    )

    history_age = torch.arange(
        PURE_RL_SENSOR_HISTORY_STEPS - 1,
        -1,
        -1,
        device=observation.device,
    )
    window_mask = (history_age.unsqueeze(0) < cycle_steps.unsqueeze(1)).to(
        dtype=observation.dtype
    )
    divisor = cycle_steps.to(dtype=observation.dtype).unsqueeze(1)
    mean_sensor = torch.sum(sensor_history * window_mask.unsqueeze(2), dim=1) / divisor
    mean_action = torch.sum(action_history * window_mask.unsqueeze(2), dim=1) / divisor

    mean_quaternion = mean_sensor[:, 0:4]
    epsilon = torch.finfo(observation.dtype).eps
    mean_quaternion = mean_quaternion / torch.clamp(
        torch.linalg.vector_norm(mean_quaternion, dim=1, keepdim=True),
        min=epsilon,
    )
    phase_sin = torch.zeros_like(mean_sensor[:, PURE_RL_PHASE_SIN_INDEX : PURE_RL_PHASE_SIN_INDEX + 1])
    phase_cos = torch.ones_like(mean_sensor[:, PURE_RL_PHASE_COS_INDEX : PURE_RL_PHASE_COS_INDEX + 1])
    mean_sensor = torch.cat(
        (
            mean_quaternion,
            mean_sensor[:, 4:PURE_RL_PHASE_SIN_INDEX],
            phase_sin,
            phase_cos,
        ),
        dim=1,
    )

    repeated_sensor = mean_sensor.unsqueeze(1).expand(
        -1,
        PURE_RL_SENSOR_HISTORY_STEPS,
        -1,
    )
    repeated_action = mean_action.unsqueeze(1).expand(
        -1,
        PURE_RL_ACTION_HISTORY_STEPS,
        -1,
    )
    return torch.cat(
        (
            repeated_sensor.reshape(observation.shape[0], -1),
            repeated_action.reshape(observation.shape[0], -1),
            preview,
        ),
        dim=1,
    )


class PureRLSplitActor(nn.Module):
    """Independent slow frequency and full-observation tail actor trunks."""

    is_pure_rl_split_frequency_actor = True

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        if int(input_dim) != PURE_RL_OBSERVATION_DIM:
            raise ValueError(
                f"PureRL split actor requires {PURE_RL_OBSERVATION_DIM} actor observations."
            )
        resolved_hidden_dims = tuple(int(value) for value in hidden_dims)
        if not resolved_hidden_dims or any(value <= 0 for value in resolved_hidden_dims):
            raise ValueError("PureRL split actor hidden dimensions must be positive.")
        self.frequency_actor = MLP(
            PURE_RL_OBSERVATION_DIM,
            1,
            resolved_hidden_dims,
            activation,
        )
        self.tail_actor = MLP(
            PURE_RL_OBSERVATION_DIM,
            3,
            resolved_hidden_dims,
            activation,
        )

    def forward(self, observation: Tensor) -> Tensor:
        """Return actions ordered as frequency, rudder, left and right tail."""

        slow_observation = build_cycle_averaged_frequency_observation(observation)
        frequency_action = self.frequency_actor(slow_observation)
        tail_actions = self.tail_actor(observation)
        return torch.cat((frequency_action, tail_actions), dim=-1)


class PureRLSplitActorCritic(ActorCritic):
    """RSL-RL actor-critic with separate frequency and tail actor trunks."""

    def __init__(
        self,
        obs: object,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
        state_dependent_std: bool = False,
        **kwargs: object,
    ) -> None:
        if int(num_actions) != PURE_RL_ACTION_DIM:
            raise ValueError("PureRL split actor requires exactly four actions.")
        if bool(state_dependent_std):
            raise ValueError("PureRL split actor requires state-independent action noise.")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            state_dependent_std=False,
            **kwargs,
        )
        num_actor_obs = sum(int(obs[group].shape[-1]) for group in obs_groups["policy"])
        self.actor = PureRLSplitActor(
            num_actor_obs,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        print(f"PureRL split actor: {self.actor}")


def register_pure_rl_split_actor_critic() -> None:
    """Register the project-local policy class in RSL-RL's runner namespace."""

    import rsl_rl.runners.on_policy_runner as runner_module

    existing = getattr(runner_module, "PureRLSplitActorCritic", None)
    if existing is not None and existing is not PureRLSplitActorCritic:
        raise RuntimeError("RSL-RL already defines a different PureRLSplitActorCritic class.")
    runner_module.PureRLSplitActorCritic = PureRLSplitActorCritic


__all__ = [
    "PURE_RL_OBSERVATION_DIM",
    "PureRLSplitActor",
    "PureRLSplitActorCritic",
    "build_cycle_averaged_frequency_observation",
    "register_pure_rl_split_actor_critic",
]
