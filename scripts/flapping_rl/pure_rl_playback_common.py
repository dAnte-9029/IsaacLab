"""Pure helpers for reproducible PureRL checkpoint playback artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


PLAYBACK_SCHEMA_VERSION = "pure_rl_playback_v1"
MEASURED_PURE_RL_TASK_ID = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"
MEASURED_PURE_RL_C2B_TASK_ID = (
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2b-Direct-v0"
)


@dataclass(frozen=True)
class SuccessfulCheckpoint:
    """A success-gated checkpoint selected from one training run."""

    run_dir: Path
    checkpoint: Path
    selection_row: dict[str, object]


@dataclass(frozen=True)
class RouteVisualGeometry:
    """Pose and dimensions of a visual-only straight route segment."""

    midpoint_w: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    size_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class PlaybackCase:
    """Validated deterministic condition for one PureRL playback."""

    task: str
    stage_id: str | None
    heading_deg: float
    flap_phase_deg: float
    longitudinal_task_id: int | None = None
    longitudinal_task: str | None = None
    longitudinal_slope_deg: float | None = None
    longitudinal_entry_length_m: float | None = None
    longitudinal_slope_length_m: float | None = None


def resolve_playback_case(
    *,
    task: str,
    heading_deg: float,
    flap_phase_deg: float,
    longitudinal_slope_deg: float,
    longitudinal_entry_length_m: float,
    longitudinal_slope_length_m: float,
) -> PlaybackCase:
    """Resolve the supported C1 or C2b deterministic playback condition."""

    heading = float(heading_deg)
    phase = float(flap_phase_deg)
    values = (
        heading,
        phase,
        float(longitudinal_slope_deg),
        float(longitudinal_entry_length_m),
        float(longitudinal_slope_length_m),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Playback condition values must be finite.")
    if task == MEASURED_PURE_RL_TASK_ID:
        return PlaybackCase(task=task, stage_id=None, heading_deg=heading, flap_phase_deg=phase)
    if task != MEASURED_PURE_RL_C2B_TASK_ID:
        raise ValueError(f"Unsupported PureRL playback task: {task}")

    slope = float(longitudinal_slope_deg)
    if abs(slope) < 2.0 or abs(slope) > 6.0:
        raise ValueError("C2b playback slope magnitude must be within 2--6 degrees.")
    entry_length = float(longitudinal_entry_length_m)
    slope_length = float(longitudinal_slope_length_m)
    if entry_length <= 0.0 or slope_length <= 0.0:
        raise ValueError("C2b playback path lengths must be positive.")
    task_id = 1 if slope > 0.0 else 2
    return PlaybackCase(
        task=task,
        stage_id="c2b",
        heading_deg=heading,
        flap_phase_deg=phase,
        longitudinal_task_id=task_id,
        longitudinal_task="climb" if slope > 0.0 else "descent",
        longitudinal_slope_deg=slope,
        longitudinal_entry_length_m=entry_length,
        longitudinal_slope_length_m=slope_length,
    )


def playback_case_succeeded(
    *,
    stage_id: str | None,
    terminated: bool,
    c1_success_gate_passed: bool,
    recovery_reached: bool,
) -> bool:
    """Apply the stage-appropriate success requirement to one playback episode."""

    if stage_id is None:
        return bool(c1_success_gate_passed)
    if stage_id == "c2b":
        return not bool(terminated) and bool(recovery_reached)
    raise ValueError(f"Unsupported playback stage: {stage_id}")


def resolve_successful_checkpoint(run_dir: Path) -> SuccessfulCheckpoint:
    """Resolve the frozen best checkpoint and require its success gate."""

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise NotADirectoryError(resolved_run_dir)
    selection_path = resolved_run_dir / "eval" / "best_checkpoint.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text())
    if int(float(selection.get("success_gate_passed", 0))) != 1:
        raise ValueError(f"Best checkpoint has not passed the success gate: {selection_path}")

    checkpoint_value = selection.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError(f"Missing checkpoint path in: {selection_path}")
    checkpoint = Path(checkpoint_value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = resolved_run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != resolved_run_dir:
        raise ValueError(f"Selected checkpoint is outside its run directory: {checkpoint}")
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return SuccessfulCheckpoint(
        run_dir=resolved_run_dir,
        checkpoint=checkpoint,
        selection_row=dict(selection),
    )


def resolve_explicit_checkpoint(run_dir: Path, checkpoint: Path) -> SuccessfulCheckpoint:
    """Resolve an explicitly approved checkpoint within its training run."""

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise NotADirectoryError(resolved_run_dir)
    resolved_checkpoint = Path(checkpoint).expanduser()
    if not resolved_checkpoint.is_absolute():
        resolved_checkpoint = resolved_run_dir / resolved_checkpoint
    resolved_checkpoint = resolved_checkpoint.resolve()
    if resolved_checkpoint.parent != resolved_run_dir:
        raise ValueError(f"Explicit checkpoint is outside its run directory: {resolved_checkpoint}")
    if resolved_checkpoint.suffix != ".pt" or not resolved_checkpoint.is_file():
        raise FileNotFoundError(resolved_checkpoint)
    stem_parts = resolved_checkpoint.stem.split("_", maxsplit=1)
    ckpt_index = int(stem_parts[1]) if len(stem_parts) == 2 and stem_parts[1].isdigit() else -1
    return SuccessfulCheckpoint(
        run_dir=resolved_run_dir,
        checkpoint=resolved_checkpoint,
        selection_row={
            "checkpoint": str(resolved_checkpoint),
            "ckpt_index": ckpt_index,
            "selection_source": "explicit_promoted_checkpoint",
        },
    )


def compute_route_visual_geometry(
    *,
    env_origin_w: Sequence[float],
    target_height_m: float,
    heading_rad: float,
    behind_m: float,
    ahead_m: float,
    width_m: float,
    thickness_m: float,
    vertical_offset_m: float,
) -> RouteVisualGeometry:
    """Compute a yaw-aligned visual route segment in world coordinates."""

    if len(env_origin_w) != 3:
        raise ValueError("env_origin_w must contain exactly three values.")
    if behind_m < 0.0 or ahead_m <= 0.0:
        raise ValueError("Route extents must satisfy behind >= 0 and ahead > 0.")
    if width_m <= 0.0 or thickness_m <= 0.0:
        raise ValueError("Route width and thickness must be positive.")
    length_m = float(behind_m) + float(ahead_m)
    midpoint_progress_m = 0.5 * (float(ahead_m) - float(behind_m))
    tangent_x = math.cos(float(heading_rad))
    tangent_y = math.sin(float(heading_rad))
    half_yaw = 0.5 * float(heading_rad)
    return RouteVisualGeometry(
        midpoint_w=(
            float(env_origin_w[0]) + midpoint_progress_m * tangent_x,
            float(env_origin_w[1]) + midpoint_progress_m * tangent_y,
            float(env_origin_w[2]) + float(target_height_m) + float(vertical_offset_m),
        ),
        orientation_wxyz=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        size_xyz_m=(length_m, float(width_m), float(thickness_m)),
    )


def build_gif_command(
    *,
    ffmpeg_executable: str,
    input_mp4: Path,
    output_gif: Path,
    fps: int,
    width_px: int,
) -> list[str]:
    """Build a palette-based ffmpeg GIF conversion command."""

    if fps <= 0 or width_px <= 0:
        raise ValueError("GIF fps and width must be positive.")
    filter_graph = (
        f"[0:v]fps={int(fps)},scale={int(width_px)}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    return [
        str(ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(Path(input_mp4)),
        "-filter_complex",
        filter_graph,
        "-loop",
        "0",
        str(Path(output_gif)),
    ]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_new_playback_outputs(output_dir: Path, name: str) -> tuple[Path, Path, Path]:
    """Resolve output paths and fail rather than overwrite prior evidence."""

    if not name or Path(name).name != name:
        raise ValueError("Playback name must be a non-empty filename-safe component.")
    output_dir = Path(output_dir).expanduser().resolve()
    mp4_path = output_dir / f"{name}.mp4"
    gif_path = output_dir / f"{name}.gif"
    manifest_path = output_dir / f"{name}.json"
    existing = [path for path in (mp4_path, gif_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Playback outputs already exist: {existing}")
    return mp4_path, gif_path, manifest_path


def numeric_metrics_are_finite(row: Mapping[str, object], keys: Sequence[str]) -> bool:
    """Return whether named metrics exist and are finite numeric values."""

    return all(key in row and math.isfinite(float(row[key])) for key in keys)
