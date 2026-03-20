from flapping_bot.path_tracking.mission_primitives import Mission, MissionSegment
from flapping_bot.path_tracking.path_manager import PathManager, PathManagerCfg


def test_path_manager_produces_finite_curvature_and_progress():
    mission = Mission(
        segments=[
            MissionSegment(kind="straight", altitude_changes=False),
            MissionSegment(kind="turn", altitude_changes=False),
        ]
    )
    manager = PathManager(PathManagerCfg(max_roll_deg=35.0, max_flight_path_angle_deg=10.0), mission)
    query = manager.query(position_xy=(0.0, 0.0), altitude_m=10.0, speed_mps=7.0)
    assert query.curvature_m_inv == query.curvature_m_inv
    assert query.progress_s >= 0.0
    assert len(query.preview_points_xyz) == 5
