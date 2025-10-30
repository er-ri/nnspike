#!/usr/bin/env python3
"""Neural Network Spike Robot Control - Simplified Speed Control.

This script controls a line-following robot using neural network predictions
with a simplified speed control algorithm for easier tuning:

Speed Tuning Parameters:
- BASE_SPEED: Base speed for straight lines (start here)

PID Tuning Parameters:
- Kp, Ki, Kd: Standard PID parameters for steering correction
"""

import argparse
import pickle
import socket
import struct
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import cv2
import numpy as np

from nnspike.constants import (
    CAMERA_FOCAL_LENGTH_PIXELS,
    CAMERA_HEIGHT,
    OFFSET_Y,
    PhaseConfig,
)
from nnspike.unit import ETRobot, WebcamVideoStream
from nnspike.utils import (
    PIDController,
    SensorRecorder,
    calculate_attitude_angle,
    draw_driving_info,
)

# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 45  # Base speed for straight lines (adjust this first)
HOST_IP_ADDRESS = (
    "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to
)


def initialize_phase(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to automatically initialize a phase if it hasn't been initiated yet."""

    @wraps(func)
    def wrapper(action_chain: ActionChain, *args: Any, **kwargs: Any) -> Any:
        phase_conf = action_chain.phases_mapper[action_chain.active_phase]

        if phase_conf["initiated"] is False:
            # Capture starting encoder position for distance-based logic
            action_chain.start_position = (
                action_chain.et.retrieve_motors_relative_position()
            )

            # Update PID controller with phase-specific parameters if provided
            pid_params = phase_conf["pid_params"]
            action_chain.pid = PIDController(
                kp=pid_params[0],
                ki=pid_params[1],
                kd=pid_params[2],
                setpoint=0,
                output_limits=(-100, 100),
            )

            phase_conf["initiated"] = True
        return func(action_chain, *args, **kwargs)

    return wrapper


class ActionChain:
    """A class to manage a sequence of actions for an ETRobot.

    This class allows you to define a chain of actions, each consisting of
    setting left and right motor speeds for a specified duration.
    """

    def __init__(self, et: ETRobot, initial_phase: int, course: str) -> None:
        """Initialize the ActionChain with an ETRobot instance and initial phase."""
        self.et = et
        self.course = course
        self.active_phase = initial_phase

        # Action with specific positioning
        self.start_position = 0
        self.current_position = 0
        self.mark_position = 0

        # fmt: off
        self.phases_mapper: dict[int, PhaseConfig] = {
            # phase_index: configuration
                0: {
                    "initiated": False,
                    "base_speed": 80,
                    "pid_params": (50.0, 0.0, 2.0),
                    "roi": (0, 0, 640, 480),
                    "method_name": "perform_phase0"
                },  # Phase 0
                1: {
                    "initiated": False,
                    "base_speed": 80,
                    "pid_params": (50.0, 0.0, 2.0),
                    "roi": (0, 0, 640, 480),
                    "method_name": "perform_phase1"
                },  # Phase 1
        }
        # fmt: on

        pid_params = self.phases_mapper[initial_phase]["pid_params"]
        self.pid = PIDController(
            kp=pid_params[0],
            ki=pid_params[1],
            kd=pid_params[2],
            setpoint=0,
            output_limits=(-100, 100),
        )

    def reset(self) -> None:
        """Reset the action chain by setting the start position to 0."""
        self.start_position = 0
        self.current_position = 0
        self.mark_position = 0

    def get_moved_distance(self) -> int:
        """Get the distance moved since the start of the current phase.

        Returns:
            int: The distance moved since the start of the current phase.
        """
        if self.start_position == 0:
            return 0
        self.current_position = self.et.retrieve_motors_relative_position()
        return self.current_position - self.start_position

    def perform_action_chain(self, image: np.ndarray) -> tuple[int, int]:
        """Execute the current phase and return its (target_x, speed) result."""
        if self.active_phase not in self.phases_mapper:
            raise ValueError(
                f"phase '{self.active_phase}' not defined in phases_mapper."
            )

        phase_conf = self.phases_mapper[self.active_phase]
        method_name = str(phase_conf["method_name"])

        method = getattr(self, method_name, None)
        if not callable(method):
            raise ValueError(f"Method '{method_name}' not found in ActionChain.")

        result = method(image=image)
        # Ensure result is a tuple of expected structure
        return result  # type: ignore[no-any-return]

    def __wrap_pid_calculation(self, target_x: float) -> tuple[int, int]:
        x1, y1, x2, y2 = self.phases_mapper[self.active_phase]["roi"]
        roi_center_x = (x1 + x2) / 2
        offset_pixels = target_x - roi_center_x
        theta = calculate_attitude_angle(
            offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS
        )
        steering_correction = self.pid.update(theta)
        left_speed = int(
            self.phases_mapper[self.active_phase]["base_speed"] - steering_correction
        )
        right_speed = int(
            self.phases_mapper[self.active_phase]["base_speed"] + steering_correction
        )
        return left_speed, right_speed

    @initialize_phase
    def perform_phase0(self, image: np.ndarray) -> tuple[int, int]:
        print("Performing phase 0")

        target_x = 0.0
        left_speed, right_speed = self.__wrap_pid_calculation(target_x=target_x)

        if 1 == 1:  # Placeholder for actual condition
            print("Switching to phase 1")
            self.active_phase = 1
            return 0, 0  # Stop before switching

        return left_speed, right_speed

    @initialize_phase
    def perform_phase1(self, image: np.ndarray) -> tuple[int, int]:
        print("Performing phase 1")

        if 2 == 2:  # Placeholder for actual condition
            print("Switching to phase 2")
            self.active_phase = 2
            return 0, 0  # Stop before switching

        return 0, 0


def main(
    model_path: str,
    record_sensor_data: bool = False,
    save_camera_video: bool = False,
    send_video_stream: bool = False,
    course: str = "left",
) -> None:
    # Generate timestamp for consistent naming if recording is enabled
    timestamp = (
        time.strftime("%Y%m%d%H%M%S", time.localtime())
        if (record_sensor_data or save_camera_video)
        else ""
    )

    vs = WebcamVideoStream(
        src=0,
        save_video=save_camera_video,
        save_path=f"storage/videos/{timestamp}_picamera.avi",
    )
    vs.start()

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=timestamp)
        sensor_recorder.start_recording()

    client_socket = None
    if send_video_stream:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((HOST_IP_ADDRESS, 8485))
            print(f"Connected to host PC at {HOST_IP_ADDRESS}:8485 for video streaming")
        except Exception as e:
            print(f"Warning: Could not connect to host PC for video streaming: {e}")
            client_socket = None  # Initialization

    et = ETRobot()
    action_chain = ActionChain(et, initial_phase=0, course=course)

    time.sleep(0.5)

    et.set_motor_relative_position(left_positon=0, right_position=0)

    try:
        while et.is_running == True:
            ret, frame = vs.read()
            if not ret or frame is None:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Execute phase and get control outputs
            left_speed, right_speed = action_chain.perform_action_chain(image=frame)

            et.set_motor_speed(left_speed=int(left_speed), right_speed=int(right_speed))

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(
                    et.get_spike_status()
                )  # Send driving information for the real-time inspection

            if send_video_stream and client_socket is not None:
                info: dict[str, Any] = {
                    "offset_y": OFFSET_Y,
                    "text": {
                        "left_speed": int(left_speed),
                        "right_speed": int(right_speed),
                    },
                }

                gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                gray = draw_driving_info(
                    gray,
                    info,
                    action_chain.phases_mapper[action_chain.active_phase]["roi"],
                )

                try:
                    ret, buffer = cv2.imencode(".jpg", gray)
                    img_encoded = buffer.tobytes()
                    data = pickle.dumps(img_encoded)
                    client_socket.sendall(struct.pack("L", len(data)) + data)
                except Exception as e:
                    print(f"Socket error: {e}")
                    break

    finally:
        vs.stop()
        et.stop()

        # Clean up socket connection if it was used
        if send_video_stream and client_socket is not None:
            client_socket.close()

        # Clean up sensor recorder if it was used
        if record_sensor_data and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the robot with optional sensor recording and video saving",
        epilog='Example usage: python run.py --model-path "./storage/models/model_left.pt"',
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--record-sensor", action="store_true", help="Record sensor data to file"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save camera video to file"
    )
    parser.add_argument(
        "--send-video", action="store_true", help="Send video stream to host PC"
    )
    parser.add_argument(
        "--course",
        choices=["left", "right"],
        default="left",
        help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: left)",
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
        course=args.course,
    )
