"""Hardware control and management units for LEGO SPIKE robot.

This module provides classes and utilities for controlling and managing the
LEGO SPIKE robot hardware, including communication, and
real-time data streaming.

Modules:
    etrobot: Main robot control interface for LEGO SPIKE hardware
    spike_status: Robot status monitoring and reporting
    webcam_video_stream: Real-time video streaming and processing
"""

from .etrobot import ETRobot
from .spike_status import SpikeStatus
from .webcam_video_stream import WebcamVideoStream

__all__ = [
    "ETRobot",
    "WebcamVideoStream",
    "SpikeStatus",
]
