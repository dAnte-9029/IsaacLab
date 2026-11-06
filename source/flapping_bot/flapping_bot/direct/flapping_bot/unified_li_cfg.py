"""L/I-specific finite-horizon initial conditions; identical within backend across samplers."""
from isaaclab.utils import configclass
from flapping_bot.physics.control_aero_3d import ControlAero3DConfig
from flapping_bot.analysis.unified_ppo_contract import CANDIDATES
from .unified_trajectory_env import UnifiedTrajectoryEnvCfg


@configclass
class UnifiedTrajectoryL3DEnvCfg(UnifiedTrajectoryEnvCfg):
    control_aero_3d: ControlAero3DConfig = ControlAero3DConfig(backend='L3D',twist_amplitude_deg=35.,sections_per_wing=10)
    reset_forward_speed_mps: float = CANDIDATES['L3D'][0]
    reset_pitch_deg: float = CANDIDATES['L3D'][1]
    reset_flap_hz: float = CANDIDATES['L3D'][2]
    reset_elevon_pitch_deg: float = CANDIDATES['L3D'][3]
    reset_rudder_deg: float = CANDIDATES['L3D'][4]
    reset_elevon_roll_deg: float = CANDIDATES['L3D'][5]
    randomize_straight_line_heading: bool = False
    randomize_flap_phase_at_reset: bool = False
    teacher_guidance_enabled: bool = False
    freeze_steps_after_reset: int = 0
    wind_enabled: bool = False
    wind_ou_enabled: bool = False


@configclass
class UnifiedTrajectoryI3DEnvCfg(UnifiedTrajectoryL3DEnvCfg):
    control_aero_3d: ControlAero3DConfig = ControlAero3DConfig(backend='I3D',twist_amplitude_deg=35.,sections_per_wing=10)
    reset_forward_speed_mps: float = CANDIDATES['I3D'][0]
    reset_pitch_deg: float = CANDIDATES['I3D'][1]
    reset_flap_hz: float = CANDIDATES['I3D'][2]
    reset_elevon_pitch_deg: float = CANDIDATES['I3D'][3]
    reset_rudder_deg: float = CANDIDATES['I3D'][4]
    reset_elevon_roll_deg: float = CANDIDATES['I3D'][5]
