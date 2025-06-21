#!/usr/bin/env python3
import sys
import cv2
import time
import argparse
from nnspike.unit import ETRobot
from nnspike.utils import SensorRecorder

# Platform-specific imports for keyboard input
try:
    import msvcrt  # Windows

    WINDOWS = True
except ImportError:
    import select
    import tty
    import termios

    WINDOWS = False

# User defined constants
BASE_POWER = 50
TURN_POWER = 40


class KeyboardController:
    def __init__(self):
        self.running = True
        self.current_key = None

        if not WINDOWS:
            # Save terminal settings for Unix-like systems
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())

    def get_key(self):
        """Get a single keypress"""
        if WINDOWS:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8").lower()
                return key
        else:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1).lower()
                return key
        return None

    def cleanup(self):
        """Restore terminal settings"""
        if not WINDOWS:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main(record_sensor_data=False, save_camera_video=False):
    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = (
        time.strftime("%Y%m%d%H%M%S", time.localtime())
        if (record_sensor_data or save_camera_video)
        else None
    )

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()

    # Initialize video writer conditionally
    video_writer = None
    cap = None
    if save_camera_video:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(640, 480),
        )

    # Initialize robot and keyboard controller
    et = ETRobot()
    keyboard = KeyboardController()

    print("Remote Control Car Initialized!")
    print("Controls:")
    print("  w - Move Forward")
    print("  s - Move Backward")
    print("  a - Turn Left")
    print("  d - Turn Right")
    print("  b - Stop/Brake")
    print("  q - Quit")
    print("\nPress any key to start...")

    # Wait for initial keypress
    while True:
        key = keyboard.get_key()
        if key:
            break
        time.sleep(0.01)

    print("Remote control active!")

    try:
        while keyboard.running:
            # Get camera frame if video recording is enabled
            if save_camera_video and cap is not None:
                ret, frame = cap.read()
                if ret and video_writer is not None:
                    video_writer.write(frame)

            # Get keyboard input
            key = keyboard.get_key()

            if key:
                if key == "q":
                    print("\nQuitting...")
                    break
                elif key == "w":
                    # Move forward
                    print("Moving forward")
                    et.set_motor_forward_power(
                        left_power=BASE_POWER, right_power=BASE_POWER
                    )
                elif key == "s":
                    # Move backward
                    print("Moving backward")
                    et.set_motor_backward_power(
                        left_power=BASE_POWER, right_power=BASE_POWER
                    )
                elif key == "a":
                    # Turn left
                    print("Turning left")
                    et.set_motor_forward_power(left_power=0, right_power=TURN_POWER)
                elif key == "d":
                    # Turn right
                    print("Turning right")
                    et.set_motor_forward_power(left_power=TURN_POWER, right_power=0)
                elif key == "b":
                    # brake
                    print("Stopping")
                    et.brake()
                else:
                    # Unknown key - brake for safety
                    et.brake()

            # Log sensor data if recording is enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status())

            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping robot...")

    finally:
        print("Cleaning up resources...")

        # Stop the robot
        et.stop()

        # Clean up keyboard controller
        keyboard.cleanup()

        # Clean up camera and video writer
        if cap is not None:
            cap.release()
        if save_camera_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")

        # Clean up sensor recorder
        if record_sensor_data and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")

        print("Cleanup completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run remote control car with keyboard controls"
    )
    parser.add_argument(
        "--record-sensor", action="store_true", help="Record sensor data to file"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save camera video to file"
    )

    args = parser.parse_args()

    main(record_sensor_data=args.record_sensor, save_camera_video=args.save_video)
