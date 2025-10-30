"""Utility functions and algorithms for robot control and image processing.

This module provides essential utility functions for image processing, control
algorithms, and data recording used throughout the nnspike package for robot
navigation and line following.

Modules:
    control: Computer vision algorithms for line detection and navigation
    image: Image processing and visualization utilities
    pid: PID controller implementation for smooth robot movement
    recorder: Data recording utilities for training data collection

Example:
    Using image processing and control utilities:

        from nnspike.utils import find_line_edges_at_y, PIDController

        # Detect line edges in an image
        left_edge, right_edge = find_line_edges_at_y(image, y_position=100)

        # Create PID controller for steering
        pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        steering = pid.update(error)
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .control import (
    calculate_attitude_angle,
    find_bottle_center,
    find_bullseye,
    find_gate_virtual_line,
    find_line_edges_at_y,
)
from .image import draw_driving_info, normalize_image
from .pid import PIDController
from .recorder import SensorRecorder

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "find_line_edges_at_y",
    "find_gate_virtual_line",
    "find_bottle_center",
    "find_bullseye",
    "calculate_attitude_angle",
    "normalize_image",
    "draw_driving_info",
    "PIDController",
    "SensorRecorder",
]
