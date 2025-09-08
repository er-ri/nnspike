#!/usr/bin/env python3
import argparse
import os

# Platform-specific imports for keyboard input (Raspberry Pi only)
import select
import sys
import termios
import time
import tty

import cv2

from nnspike.constants import Mode
from nnspike.unit import ETRobot
from nnspike.utils.recorder import SensorRecorder

# User defined constants
BASE_SPEED = 55
SMOOTH_TURN_SPEED = 40  # For smoother turning


class KeyboardController:
    def __init__(self) -> None:
        self.running = True
        self.current_key = None

        # Save terminal settings for Unix-like systems
        self.old_settings = termios.tcgetattr(sys.stdin)  # type: ignore
        tty.setraw(sys.stdin.fileno())  # type: ignore

    def get_key(self) -> str | None:
        """Get a single keypress"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1).lower()
            return key
        return None

    def cleanup(self) -> None:
        """Restore terminal settings"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)  # type: ignore


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Remote Control Car with optional recording features"
    )
    parser.add_argument(
        "--record-sensor", action="store_true", help="Enable sensor data recording"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Enable video recording"
    )
    args = parser.parse_args()

    # Generate timestamp for consistent naming if recording is enabled
    timestamp = (
        time.strftime("%Y%m%d%H%M%S", time.localtime())
        if (args.record_sensor or args.save_video)
        else None
    )

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if args.record_sensor:
        sensor_recorder = SensorRecorder(timestamp=timestamp)
        sensor_recorder.start_recording()

    # Initialize video writer conditionally
    video_writer = None
    video_filename = None
    cap = None
    if args.save_video:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
        video_filename = f"storage/videos/{timestamp}_rc_control.avi"

        # Ensure the directory exists
        os.makedirs("storage/videos", exist_ok=True)

        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=25,
            frameSize=(640, 480),
        )

    # Initialize robot and keyboard controller
    et = ETRobot()
    keyboard = KeyboardController()

    print("Remote Control Car Initialized!")
    if args.record_sensor and sensor_recorder is not None:
        print(
            f"Sensor recording: ENABLED (saving to: {sensor_recorder.get_filename()})"
        )
    if args.save_video and video_filename is not None:
        print(f"Video recording: ENABLED (saving to: {video_filename})")
    print("Controls:")
    print("  w - Move Forward")
    print("  s - Move Backward")
    print("  a - Smooth Turn Left")
    print("  d - Smooth Turn Right")
    print("  o - Complete Obstacle Avoidance Maneuver (returns to original heading)")
    print("  b - Stop/Brake")
    print("  q - Quit")
    print("Press any key to start...")

    # Wait for initial keypress
    while True:
        key = keyboard.get_key()
        if key:
            break
        time.sleep(0.01)

    print("Remote control active!")

    try:
        while keyboard.running:
            # Capture video frame if video recording is enabled
            frame = None
            if args.save_video and cap is not None:
                ret, frame = cap.read()
                if ret and video_writer is not None:
                    video_writer.write(frame)

            # Log sensor data if sensor recording is enabled
            if args.record_sensor and sensor_recorder is not None:
                sensor_recorder.log_frame_data(
                    et.get_spike_status(), Mode.REMOTE_CONTROL
                )

            # Get keyboard input
            key = keyboard.get_key()

            if key:
                if key == "q":
                    print("Quitting...")
                    break
                elif key == "w":
                    # Move forward
                    print("Moving forward")
                    et.set_motor_speed(left_speed=BASE_SPEED, right_speed=BASE_SPEED)
                elif key == "s":
                    # Move backward
                    print("Moving backward")
                    et.set_motor_speed(left_speed=-BASE_SPEED, right_speed=-BASE_SPEED)
                elif key == "a":
                    # Smooth turn left
                    print("Turning left")
                    et.set_motor_speed(
                        left_speed=SMOOTH_TURN_SPEED, right_speed=BASE_SPEED
                    )
                elif key == "d":  # Smooth turn right
                    print("Turning right")
                    et.set_motor_speed(
                        left_speed=BASE_SPEED, right_speed=SMOOTH_TURN_SPEED
                    )
                elif key == "b":
                    # brake
                    print("Stopping")
                    et.brake()
                else:  # Unknown key - brake for safety
                    et.brake()

            # Small delay to prevent excessive CPU usage while maintaining responsiveness
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping robot...")

    finally:
        print("Cleaning up resources...")

        # Stop the robot
        et.stop()

        # Clean up keyboard controller
        keyboard.cleanup()

        # Clean up video recording if it was enabled
        if args.save_video:
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()
                print(f"Video saved to: {video_filename}")

        # Clean up sensor recorder if it was enabled
        if args.record_sensor and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Sensor data saved to: {sensor_recorder.get_filename()}")
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")

        print("Cleanup completed.")


if __name__ == "__main__":
    main()
