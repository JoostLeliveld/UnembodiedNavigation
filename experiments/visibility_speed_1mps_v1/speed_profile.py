"""Shared speed and preview contract for this separate commissioning pilot."""
from pathlib import Path
import json,hashlib,math
PROFILE_PATH=Path(__file__).with_suffix('.json')
PROFILE=json.loads(PROFILE_PATH.read_text())
V_MAX=PROFILE['max_linear_speed_m_s']
assert math.isfinite(V_MAX) and V_MAX==1.0
LOOKAHEAD=V_MAX*PROFILE['tracker_lookahead_s']
LOCAL_GOAL_AHEAD=V_MAX*PROFILE['local_goal_preview_s']
HANDOFF_FORWARD=V_MAX*PROFILE['handoff_forward_window_s']
def validate_config(cfg):
    if cfg.get('speed_profile')!=PROFILE:raise ValueError('Missing or inconsistent shared 1 m/s speed profile')
def profile_hash():return hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
