from pathlib import Path
from types import SimpleNamespace

from scripts.flapping_px4.run_reset_trim_sweep import build_rollout_command, build_sweep_parser


def test_build_sweep_parser_accepts_reset_trim_ranges() -> None:
    args = build_sweep_parser().parse_args(
        [
            "--reset_pitch_deg_values",
            "6.0",
            "8.0",
            "--reset_flap_hz_values",
            "2.8",
            "3.2",
            "--reset_elevon_pitch_deg_values",
            "-14.0",
            "-10.0",
        ]
    )

    assert args.reset_pitch_deg_values == [6.0, 8.0]
    assert args.reset_flap_hz_values == [2.8, 3.2]
    assert args.reset_elevon_pitch_deg_values == [-14.0, -10.0]


def test_build_rollout_command_includes_reset_trim_overrides() -> None:
    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        num_envs=1,
        height_sp=10.0,
        steps=2200,
        metrics_warmup_s=3.0,
        straight_length_m=60.0,
        turn_radius_m=20.0,
        loiter_radius_m=20.0,
        turn_sweep_deg=90.0,
        loiter_turns=1.0,
        climb_delta_m=3.0,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        print_every=0,
        headless=True,
    )

    cmd = build_rollout_command(
        args,
        out_dir=Path("logs/trim"),
        reset_pitch_deg=6.5,
        reset_flap_hz=3.2,
        reset_elevon_pitch_deg=-14.0,
    )

    assert "--phase" in cmd
    assert cmd[cmd.index("--phase") + 1] == "level_straight"
    assert cmd[cmd.index("--reset_pitch_deg") + 1] == "6.5"
    assert cmd[cmd.index("--reset_flap_hz") + 1] == "3.2"
    assert cmd[cmd.index("--reset_elevon_pitch_deg") + 1] == "-14.0"
    assert "--headless" in cmd
