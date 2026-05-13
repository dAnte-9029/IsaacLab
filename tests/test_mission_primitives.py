from flapping_bot.path_tracking.mission_primitives import MissionGeneratorCfg, sample_mission


def test_sample_mission_returns_allowed_segments():
    cfg = MissionGeneratorCfg(
        seed=7,
        num_segments_min=2,
        num_segments_max=4,
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
        allow_climb_on_straight=True,
    )
    mission = sample_mission(cfg)
    assert 2 <= len(mission.segments) <= 4
    assert all(seg.kind in {"straight", "turn", "loiter"} for seg in mission.segments)
    assert all(not seg.altitude_changes for seg in mission.segments if seg.kind in {"turn", "loiter"})


def test_sample_mission_excludes_zero_weight_primitives() -> None:
    cfg = MissionGeneratorCfg(
        seed=11,
        num_segments_min=8,
        num_segments_max=8,
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
        allow_climb_on_straight=False,
        straight_weight=0.0,
        turn_weight=1.0,
        loiter_weight=1.0,
    )

    mission = sample_mission(cfg)

    assert len(mission.segments) == 8
    assert all(seg.kind in {"turn", "loiter"} for seg in mission.segments)
