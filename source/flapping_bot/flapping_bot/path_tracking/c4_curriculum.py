"""Bounded training-only C4 sampling with explicit competence and regression history."""

from dataclasses import asdict, dataclass, replace
import math

import torch

from flapping_bot.path_tracking.continuous_curve import (
    ContinuousCurveSamplingConfig, ContinuousCurveSpec, sample_accepted_continuous_curve_specs,
)

C4_CURRICULUM_BIN_OFFSET = 100  # Diagnostic namespace; disjoint from C3 hard-slice IDs.


@dataclass(frozen=True)
class C4CurriculumConfig:
    """Pilot settings inside the existing mild candidate envelope."""

    window: int = 32
    minimum_episodes: int = 8
    coverage_fraction: float = 0.4
    maximum_probability_change: float = 0.05
    initial_probabilities: tuple[float, ...] = (0.55, 0.20, 0.20, 0.05)

    def validate(self) -> None:
        if not 1 <= self.minimum_episodes <= self.window:
            raise ValueError("Invalid curriculum episode windows")
        if not 0 < self.coverage_fraction <= 1 or not 0 < self.maximum_probability_change <= 1:
            raise ValueError("Invalid curriculum probability controls")
        if (len(self.initial_probabilities) != 4
                or any(not math.isfinite(p) or p <= 0 for p in self.initial_probabilities)
                or not math.isclose(sum(self.initial_probabilities), 1.0)):
            raise ValueError("Four positive normalized initial probabilities required")


class C4Curriculum:
    """Four-bin self-paced sampler; never reads evaluation cases or changes physics."""

    def __init__(self, *, seed: int, config: C4CurriculumConfig = C4CurriculumConfig()) -> None:
        config.validate()
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.tensor(config.initial_probabilities, dtype=torch.float64)
        self.probabilities = config.coverage_fraction / 4 + (1 - config.coverage_fraction) * initial
        self.history: list[list[float]] = [[] for _ in range(4)]
        self.counts = [0] * 4
        self.updates = 0
        self.last_update_count = 0

    def observe(self, bin_id: int, score: float) -> None:
        """Record one complete training episode, with failures included."""
        if not 0 <= bin_id < 4 or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid curriculum episode")
        self.history[bin_id].append(float(score))
        self.history[bin_id] = self.history[bin_id][-2 * self.config.window:]
        self.counts[bin_id] += 1

    def update(self) -> dict:
        """Bound distribution movement; confidence requires completed episodes per bin."""
        utility, competence, regression = [], [], []
        for history in self.history:
            recent = history[-self.config.window:]
            current = sum(recent) / len(recent) if recent else 0.0
            older = history[:-self.config.window]
            previous = sum(older) / len(older) if older else current
            competence.append(current)
            regression.append(max(0.0, previous - current))
            # Highest frontier weight at intermediate competence; persistent zero
            # competence receives coverage, not an unbounded failure priority.
            utility.append(0.05 + 4 * current * (1 - current) + abs(current - previous))
        ready = all(len(h) >= self.config.minimum_episodes for h in self.history)
        if ready and sum(self.counts) > self.last_update_count:
            score = torch.tensor(utility, dtype=torch.float64)
            target = self.config.coverage_fraction / 4 + (1 - self.config.coverage_fraction) * score / score.sum()
            difference = target - self.probabilities
            scale = min(1.0, self.config.maximum_probability_change / max(float(difference.abs().max()), 1e-12))
            self.probabilities += scale * difference
            self.updates += 1
            self.last_update_count = sum(self.counts)
        return dict(probabilities=self.probabilities.tolist(), competence=competence,
                    regression=regression, counts=self.counts.copy(), ready=ready, updates=self.updates)

    def sample(self, num_paths: int, *, generator: torch.Generator | None = None,
               device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
               initial_altitude_m: float | torch.Tensor = 10.0,
               config: ContinuousCurveSamplingConfig = ContinuousCurveSamplingConfig(),
               diagnostics: dict[str, int] | None = None) -> tuple[tuple[ContinuousCurveSpec, ...], torch.Tensor]:
        """Draw bins first, then reject within each bin using a private CPU RNG."""
        del generator
        if torch.device(device).type != "cpu":
            raise ValueError("C4 curriculum pilot requires CPU-native execution")
        bins = torch.multinomial(self.probabilities, num_paths, replacement=True, generator=self.generator)
        altitude = torch.as_tensor(initial_altitude_m, device=device).expand(num_paths)
        specs: list[ContinuousCurveSpec | None] = [None] * num_paths
        for bin_id in range(4):
            rows = (bins == bin_id).nonzero().flatten()
            if not rows.numel():
                continue
            def half(interval: tuple[float, float], upper: bool) -> tuple[float, float]:
                mid = sum(interval) / 2
                return (mid, interval[1]) if upper else (interval[0], mid)
            region = replace(config,
                             heading_amplitude_range=half(config.heading_amplitude_range, bool(bin_id // 2)),
                             slope_amplitude_deg_range=half(config.slope_amplitude_deg_range, bool(bin_id % 2)))
            sampled = sample_accepted_continuous_curve_specs(
                len(rows), generator=self.generator, device=device, dtype=dtype,
                initial_altitude_m=altitude[rows], config=region, diagnostics=diagnostics)
            for row, spec in zip(rows.tolist(), sampled, strict=True):
                specs[row] = spec
        assert all(spec is not None for spec in specs)
        return tuple(specs), bins.to(device=device)  # type: ignore[arg-type]

    def state_dict(self) -> dict:
        """Serialize sampler progress; simulation episodes themselves are not restored."""
        return dict(config=asdict(self.config), probabilities=self.probabilities.clone(),
                    history=[h.copy() for h in self.history], counts=self.counts.copy(),
                    updates=self.updates, last_update_count=self.last_update_count, rng=self.generator.get_state())

    def load_state_dict(self, state: dict) -> None:
        """Reject a resume under a different sampling contract."""
        if state["config"] != asdict(self.config):
            raise ValueError("Curriculum resume configuration differs")
        probabilities = state["probabilities"]
        if (probabilities.shape != (4,) or not torch.isfinite(probabilities).all()
                or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0))
                or (probabilities < self.config.coverage_fraction / 4 - 1e-10).any()):
            raise ValueError("Invalid saved curriculum probabilities")
        self.probabilities = probabilities.clone()
        self.history = [list(h) for h in state["history"]]
        self.counts = list(state["counts"])
        self.updates = int(state["updates"])
        self.last_update_count = int(state.get("last_update_count", 0))
        self.generator.set_state(state["rng"])
