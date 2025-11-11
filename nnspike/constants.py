"""Constants and enumerations used throughout the nnspike package.

This module defines constants for robot control, camera parameters, and behavior modes
used by the LEGO SPIKE robot for line following and navigation tasks.
"""

from enum import Enum
from typing import TypedDict

# Camera and Robot Geometry Constants
CAMERA_HEIGHT = 0.20  # Camera height above ground in meters
CAMERA_FOCAL_LENGTH_PIXELS = 640  # Approximate focal length in pixels
WHEELBASE = 0.11  # Distance between wheels in meters
OFFSET_Y = 470  # 0.20 meters to the robot

# Scale for relative position in motor control
RELATIVE_POSITION_SCALE = 80000


# Behavior Mode
class Mode(Enum):
    # Manual Control Modes
    MOVE_FORWARD = 100
    MOVE_FORWARD_LEFT = 110
    MOVE_FORWARD_RIGHT = 120
    TURN_LEFT = 130
    TURN_RIGHT = 140
    MOVE_BACKWARD = 150
    PAUSE = 160
    REMOTE_CONTROL = 999


class PhaseConfig(TypedDict):
    """Type definition for phase configuration."""

    initiated: bool
    base_speed: int
    pid_params: tuple[float, float, float]
    roi: tuple[int, int, int, int]
    method_name: str
