from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/pure_rl_playback_common.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_playback_common", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
playback = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = playback
SPEC.loader.exec_module(playback)


def _write_selection(run_dir: Path, checkpoint: Path, *, passed: int) -> None:
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "eval" / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "success_gate_passed": str(passed),
                "score": "98.0",
            }
        )
    )


def test_resolve_successful_checkpoint_requires_in_run_success_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_100.pt"
    checkpoint.write_bytes(b"weights")
    _write_selection(run_dir, checkpoint, passed=1)

    selected = playback.resolve_successful_checkpoint(run_dir)

    assert selected.run_dir == run_dir.resolve()
    assert selected.checkpoint == checkpoint.resolve()
    assert selected.selection_row["success_gate_passed"] == "1"


def test_resolve_successful_checkpoint_rejects_failed_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_0.pt"
    checkpoint.write_bytes(b"weights")
    _write_selection(run_dir, checkpoint, passed=0)

    with pytest.raises(ValueError, match="has not passed"):
        playback.resolve_successful_checkpoint(run_dir)


def test_resolve_successful_checkpoint_rejects_path_outside_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"weights")
    _write_selection(run_dir, checkpoint, passed=1)

    with pytest.raises(ValueError, match="outside its run directory"):
        playback.resolve_successful_checkpoint(run_dir)


def test_resolve_explicit_checkpoint_inside_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_475.pt"
    checkpoint.write_bytes(b"weights")

    selected = playback.resolve_explicit_checkpoint(run_dir, checkpoint)

    assert selected.run_dir == run_dir.resolve()
    assert selected.checkpoint == checkpoint.resolve()
    assert selected.selection_row == {
        "checkpoint": str(checkpoint.resolve()),
        "ckpt_index": 475,
        "selection_source": "explicit_promoted_checkpoint",
    }


def test_resolve_explicit_checkpoint_rejects_path_outside_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = tmp_path / "model_475.pt"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(ValueError, match="outside its run directory"):
        playback.resolve_explicit_checkpoint(run_dir, checkpoint)


def test_route_visual_geometry_rotates_and_offsets_segment() -> None:
    geometry = playback.compute_route_visual_geometry(
        env_origin_w=(10.0, 20.0, 0.0),
        target_height_m=10.0,
        heading_rad=0.5 * math.pi,
        behind_m=10.0,
        ahead_m=100.0,
        width_m=0.08,
        thickness_m=0.025,
        vertical_offset_m=-0.18,
    )

    assert geometry.midpoint_w == pytest.approx((10.0, 65.0, 9.82))
    assert geometry.orientation_wxyz == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    assert geometry.size_xyz_m == pytest.approx((110.0, 0.08, 0.025))


def test_gif_command_uses_palette_and_refuses_invalid_dimensions(tmp_path: Path) -> None:
    command = playback.build_gif_command(
        ffmpeg_executable="ffmpeg",
        input_mp4=tmp_path / "input.mp4",
        output_gif=tmp_path / "output.gif",
        fps=15,
        width_px=640,
    )

    assert command[0] == "ffmpeg"
    assert "palettegen" in command[command.index("-filter_complex") + 1]
    assert "paletteuse" in command[command.index("-filter_complex") + 1]
    with pytest.raises(ValueError, match="positive"):
        playback.build_gif_command(
            ffmpeg_executable="ffmpeg",
            input_mp4=tmp_path / "input.mp4",
            output_gif=tmp_path / "output.gif",
            fps=0,
            width_px=640,
        )


def test_output_contract_refuses_overwrite_and_hashes_checkpoint(tmp_path: Path) -> None:
    mp4, gif, manifest = playback.validate_new_playback_outputs(tmp_path, "example")
    assert (mp4.name, gif.name, manifest.name) == ("example.mp4", "example.gif", "example.json")
    mp4.write_bytes(b"video")
    with pytest.raises(FileExistsError, match="already exist"):
        playback.validate_new_playback_outputs(tmp_path, "example")
    assert playback.sha256_file(mp4) == "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc"


def test_resolve_c2b_climb_playback_case() -> None:
    case = playback.resolve_playback_case(
        task="Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2b-Direct-v0",
        heading_deg=0.0,
        flap_phase_deg=0.0,
        longitudinal_slope_deg=6.0,
        longitudinal_entry_length_m=17.5,
        longitudinal_slope_length_m=25.0,
    )

    assert case.stage_id == "c2b"
    assert case.longitudinal_task_id == 1
    assert case.longitudinal_task == "climb"
    assert case.longitudinal_slope_deg == 6.0
    assert case.longitudinal_entry_length_m == 17.5
    assert case.longitudinal_slope_length_m == 25.0


@pytest.mark.parametrize(
    ("terminated", "recovery_reached", "expected"),
    ((False, True, True), (True, True, False), (False, False, False)),
)
def test_c2_playback_success_requires_survival_and_recovery(
    terminated: bool,
    recovery_reached: bool,
    expected: bool,
) -> None:
    assert playback.playback_case_succeeded(
        stage_id="c2b",
        terminated=terminated,
        c1_success_gate_passed=False,
        recovery_reached=recovery_reached,
    ) is expected


def test_c1_playback_success_preserves_existing_gate() -> None:
    assert playback.playback_case_succeeded(
        stage_id=None,
        terminated=False,
        c1_success_gate_passed=True,
        recovery_reached=False,
    ) is True
