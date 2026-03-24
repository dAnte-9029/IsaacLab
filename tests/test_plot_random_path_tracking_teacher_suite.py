from pathlib import Path

from scripts.flapping_px4.plot_random_path_tracking_teacher_suite import (
    build_segment_label,
    discover_suite_episodes,
    infer_grid_shape,
)


def test_build_segment_label_uses_compact_codes() -> None:
    assert build_segment_label(["straight", "loiter", "turn"]) == "S-L-T"


def test_infer_grid_shape_prefers_requested_column_count() -> None:
    assert infer_grid_shape(num_items=50, cols=10) == (5, 10)
    assert infer_grid_shape(num_items=7, cols=4) == (2, 4)


def test_discover_suite_episodes_reads_episode_manifest(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    episode_run_dir = suite_dir / "episodes" / "random_seed_000001" / "20260321_000000"
    episode_run_dir.mkdir(parents=True)
    (episode_run_dir / "trajectory_env0.csv").write_text("t,x,y,z,reference_z\n0,0,0,10,10\n")
    (episode_run_dir / "reference_path.csv").write_text("progress_s,x_ref,y_ref,z_ref\n0,0,0,10\n")
    (episode_run_dir / "summary.json").write_text(
        '{"mission_seed": 1, "mission_segments": ["loiter", "straight"], "completed_path": true}'
    )
    (suite_dir / "episodes.csv").write_text(
        "\n".join(
            [
                "completed_path,mission_seed,run_dir,summary_path,traj_path",
                f"True,1,{episode_run_dir},{episode_run_dir / 'summary.json'},{episode_run_dir / 'trajectory_env0.csv'}",
            ]
        )
        + "\n"
    )

    episodes = discover_suite_episodes(suite_dir)

    assert len(episodes) == 1
    assert episodes[0].mission_seed == 1
    assert episodes[0].segment_label == "L-S"
    assert episodes[0].run_dir == episode_run_dir
