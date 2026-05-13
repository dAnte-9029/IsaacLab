from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class MissionSegment:
    """A single mission segment primitive."""

    kind: str
    altitude_changes: bool = False
    altitude_direction: int = 0
    counts_toward_progress: bool = True
    length_m: float | None = None


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
    straight_weight: float = 1.0
    turn_weight: float = 1.0
    loiter_weight: float = 1.0


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

    weighted_kinds: list[tuple[str, float]] = []
    for kind, enabled, weight in (
        ("straight", bool(cfg.allow_straight), float(cfg.straight_weight)),
        ("turn", bool(cfg.allow_turn), float(cfg.turn_weight)),
        ("loiter", bool(cfg.allow_loiter), float(cfg.loiter_weight)),
    ):
        if not enabled:
            continue
        if weight < 0.0:
            raise ValueError(f"{kind}_weight must be non-negative")
        if weight > 0.0:
            weighted_kinds.append((kind, weight))
    if not weighted_kinds:
        raise ValueError("at least one enabled segment type must have a positive sampling weight")

    rng = random.Random(cfg.seed)
    num_segments = rng.randint(cfg.num_segments_min, cfg.num_segments_max)
    segments: list[MissionSegment] = []
    enabled_kinds = [kind for kind, _ in weighted_kinds]
    enabled_weights = [weight for _, weight in weighted_kinds]
    for _ in range(num_segments):
        kind = str(rng.choices(enabled_kinds, weights=enabled_weights, k=1)[0])
        altitude_changes = kind == "straight" and cfg.allow_climb_on_straight and bool(rng.getrandbits(1))
        segments.append(MissionSegment(kind=kind, altitude_changes=altitude_changes))
    return Mission(segments=segments)
