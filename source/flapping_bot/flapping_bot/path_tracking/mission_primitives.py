from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class MissionSegment:
    """A single mission segment primitive."""

    kind: str
    altitude_changes: bool = False
    altitude_direction: int = 0


@dataclass(frozen=True)
class Mission:
    """A sampled mission consisting of segment primitives."""

    segments: list[MissionSegment]


@dataclass(frozen=True)
class MissionGeneratorCfg:
    """Sampling configuration for random missions."""

    seed: int = 0
    num_segments_min: int = 2
    num_segments_max: int = 4
    allow_straight: bool = True
    allow_turn: bool = True
    allow_loiter: bool = True
    allow_climb_on_straight: bool = True


def sample_mission(cfg: MissionGeneratorCfg) -> Mission:
    """Sample a deterministic mission from enabled primitives.

    Args:
        cfg: Mission sampling configuration.

    Returns:
        A mission with sampled segment sequence.
    """
    if cfg.num_segments_min <= 0:
        raise ValueError("num_segments_min must be positive")
    if cfg.num_segments_min > cfg.num_segments_max:
        raise ValueError("num_segments_min must be <= num_segments_max")

    enabled_kinds: list[str] = []
    if cfg.allow_straight:
        enabled_kinds.append("straight")
    if cfg.allow_turn:
        enabled_kinds.append("turn")
    if cfg.allow_loiter:
        enabled_kinds.append("loiter")
    if not enabled_kinds:
        raise ValueError("at least one segment type must be enabled")

    rng = random.Random(cfg.seed)
    num_segments = rng.randint(cfg.num_segments_min, cfg.num_segments_max)
    segments: list[MissionSegment] = []
    for _ in range(num_segments):
        kind = rng.choice(enabled_kinds)
        altitude_changes = kind == "straight" and cfg.allow_climb_on_straight and bool(rng.getrandbits(1))
        segments.append(MissionSegment(kind=kind, altitude_changes=altitude_changes))
    return Mission(segments=segments)
