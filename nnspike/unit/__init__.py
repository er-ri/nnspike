"""Hardware control and management units for LEGO SPIKE robot.

This module provides classes and utilities for controlling and managing the
LEGO SPIKE robot hardware, including communication, mode management, and
real-time data streaming.

Modules:
    action_chain: Sequential action execution for robot behaviors
    etrobot: Main robot control interface for LEGO SPIKE hardware
    mode_manager: Robot operation mode management and switching
    spike_status: Robot status monitoring and reporting
    webcam_video_stream: Real-time video streaming and processing

Example:
    Basic robot initialization and control:

        from nnspike.unit import ETRobot, ModeManager

        # Initialize robot
        robot = ETRobot()
        mode_manager = ModeManager(robot)

        # Start line following mode
        mode_manager.set_mode("line_follow")
"""

from .action_chain import ActionChain
from .etrobot import ETRobot
from .mode_manager import ModeManager
from .spike_status import SpikeStatus
from .webcam_video_stream import WebcamVideoStream

__all__ = ["ActionChain", "ETRobot", "ModeManager", "WebcamVideoStream", "SpikeStatus"]
