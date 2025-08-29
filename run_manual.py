#!/usr/bin/env python3
"""
OpenCV-Based Line Following Robot Control

This script controls a line-following robot using OpenCV for image processing
instead of neural network predictions. It uses the find_line_edges_at_y function
to detect the line centroid and follows it using PID control.

Speed Tuning Parameters:
- BASE_SPEED: Base speed for straight lines (start here)

PID Tuning Parameters:
- Kp, Ki, Kd: Standard PID parameters for steering correction
  * Kp (Proportional): Controls immediate response to error
    - Too high: Causes zigzag/oscillation
    - Too low: Slow response, may not follow turns
    - Start with: 10-20 for line following
  * Ki (Integral): Eliminates steady-state error
    - Too high: Causes instability and overshoot
    - Too low: Robot may drift to one side
    - Start with: 0.1-1.0
  * Kd (Derivative): Smooths out rapid changes
    - Too high: Sensitive to noise, erratic behavior
    - Too low: May overshoot on turns
    - Start with: 2-10
"""
import argparse
import math
import pickle

# Platform-specific imports for keyboard input (Raspberry Pi only)
import select
import socket
import struct
import sys
import termios
import time
import tty
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from nnspike.constants import CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, RELATIVE_POSITION_SCALE, ROI_CNN, Mode
from nnspike.unit import ETRobot
from nnspike.unit.action_chain import ActionChain
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info, find_line_edges_at_y
from scripts.utils import model_inference, process_image

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest for OpenCV processing


# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 45  # Base speed for straight lines (adjust this first)
TURN_SPEED = 30  # Speed for turns (adjust this if needed)

# Socket connection settings
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


class KeyboardController:
    def __init__(self):
        self.running = True
        self.current_key = None

        # Save terminal settings for Unix-like systems
        self.old_settings = termios.tcgetattr(sys.stdin)  # type: ignore
        tty.setraw(sys.stdin.fileno())  # type: ignore

    def get_key(self):
        """Get a single keypress from standard input.

        Returns:
            str or None: The key pressed, or None if no key was pressed.
        """
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1).lower()
            return key
        return None

    def cleanup(self):
        """Restore terminal settings to their original state.

        This method should be called before exiting to ensure the terminal
        is returned to a usable state.
        """
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)  # type: ignore


def main(
    record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="left", initial_mode=None, model_path=None
):
    # Initialize model
    session = ort.InferenceSession(model_path)

    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime()) if (record_sensor_data or save_camera_video) else None

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()  # Initialize video writer conditionally
    video_writer = None

    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(640, 480),
        )  # Socket connection for sending camera capture (only if enabled)
    client_socket = None

    if send_video_stream:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((HOST_IP_ADDRESS, 8485))
            print(f"Connected to host PC at {HOST_IP_ADDRESS}:8485 for video streaming")
        except Exception as e:
            print(f"Warning: Could not connect to host PC for video streaming: {e}")
            client_socket = None

    # Initialize edge following preference based on the course parameter
    et = ETRobot()
    action_chain = ActionChain(et=et, course=course)

    # Set initial mode based on parameter
    mode = initial_mode

    # Initialize robot, PID controller, and keyboard controller
    keyboard = KeyboardController()
    pid = PIDController(
        Kp=50,  # Reduced from 50 to minimize zigzag behavior
        Ki=0,  # Small integral term to eliminate steady-state error
        Kd=5,  # Derivative term to smooth out rapid changes
        setpoint=0,
        output_limits=(
            -BASE_SPEED,
            BASE_SPEED,
        ),  # Direct radian limits for steering correction
    )

    et.set_motor_relative_position(left_positon=0, right_position=0)
    et.move_arm(0, 0.5)

    try:
        while et.is_running and keyboard.running:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)

            # Check for keyboard input to change behavior mode
            key = keyboard.get_key()
            if key == "\x1b":  # ESC key to quit
                print("Quitting...")
                keyboard.running = False
                break
            elif key == "t":
                mode = Mode.FOLLOW_LEFT_EDGE
                print("Switched to following: left edge")
            elif key == "y":
                mode = Mode.FOLLOW_RIGHT_EDGE
                print("Switched to following: right edge")
            elif key == "2":
                mode = Mode.AVOID_OBSTACLE
                print("Switched to obstacle avoidance mode")
            elif key == "3":
                mode = Mode.CARRY_BOTTLE_PHASE1
                print("Switched to bottle carrying 1 mode")
            elif key == "4":
                mode = Mode.CARRY_BOTTLE_PHASE2
                print("Switched to bottle carrying 2 mode")
            elif key == "b":
                mode = Mode.PAUSE
                print("Pausing robot")
            elif key == "w":
                mode = Mode.MOVE_FORWARD
                print("Moving forward")
            elif key == "q":
                mode = Mode.MOVE_FORWARD_LEFT
                print("Moving forward left")
            elif key == "e":
                mode = Mode.MOVE_FORWARD_RIGHT
                print("Moving forward right")
            elif key == "s":
                mode = Mode.MOVE_BACKWARD
                print("Moving backward")
            elif key == "a":
                mode = Mode.TURN_LEFT
                print("Turning left")
            elif key == "d":
                mode = Mode.TURN_RIGHT
                print("Turning right")

            target_x = None  # Default target x position
            speed = None  # Initialize speed variable
            max_contour = None  # Initialize contour variable

            # Initialize variables that might not be set in all code paths
            theta = 0.0
            steering_correction = 0.0
            mx = 0.0
            my = 0.0

            left_speed, right_speed = 0, 0

            motors_relative_position = et.retrieve_motors_relative_position()
            scaled_relative_position = motors_relative_position / RELATIVE_POSITION_SCALE
            image = cv2.resize(frame, (200, 66))  # Resize to model input size
            image = image.astype(np.float32) / 255.0  # Normalize
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            # Prepare relative position
            relative_pos = np.array([[scaled_relative_position]], dtype=np.float32)

            # Run inference
            inputs = {"image": image, "relative_position": relative_pos}
            outputs = session.run(None, inputs)

            predicted_x = x1 + (outputs[0][0] * (x2 - x1))

            match mode:
                case Mode.FOLLOW_LEFT_EDGE:
                    target_x, _, mode = action_chain.follow_left_edge(image=frame, predicted_x=predicted_x)
                case Mode.FOLLOW_RIGHT_EDGE:
                    target_x, _, mode = action_chain.follow_right_edge(image=frame, predicted_x=predicted_x)
                case Mode.AVOID_OBSTACLE:
                    _, speed, mode = action_chain.avoid_obstacle(init_flag=True)
                # Manual Control Modes
                case Mode.MOVE_FORWARD:
                    speed = (BASE_SPEED, BASE_SPEED)
                case Mode.MOVE_FORWARD_LEFT:
                    speed = (int(BASE_SPEED * 0.8), BASE_SPEED)  # Slightly reduce left wheel speed
                case Mode.MOVE_FORWARD_RIGHT:
                    speed = (BASE_SPEED, int(BASE_SPEED * 0.8))  # Slightly reduce right wheel speed
                case Mode.TURN_LEFT:
                    speed = (0, TURN_SPEED)  # Left wheel stopped, right wheel moving forward
                case Mode.TURN_RIGHT:
                    speed = (TURN_SPEED, 0)  # Right wheel stopped, left wheel moving forward
                case Mode.MOVE_BACKWARD:
                    speed = (-BASE_SPEED, -BASE_SPEED)  # Both wheels moving backward
                case Mode.PAUSE:
                    speed = (0, 0)  # Stop both wheels

            if target_x is not None:
                # Calculate position relative to ROI
                mx = target_x - x1  # Relative to ROI
                my = OFFSET_Y - y1  # Relative to ROI

                # Calculate offset from ROI center
                roi_center_x = (x2 - x1) // 2
                offset_pixels = mx - roi_center_x

                # Create a simple contour for visualization (approximate target point)
                max_contour = np.array([[[mx, my]]], dtype=np.int32)

                theta = calculate_attitude_angle(
                    offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS
                )  # Use simplified speed control
                current_base_speed = BASE_SPEED

                steering_correction = pid.update(theta)

                # Apply simple differential steering
                left_speed = int(current_base_speed - steering_correction)
                right_speed = int(current_base_speed + steering_correction)
            elif speed is not None:
                left_speed, right_speed = speed
            else:
                raise ValueError("No valid target_x or speed provided for motor control")

            # Clamp speed values to valid range (-100, 100)
            left_speed = int(max(-100, min(100, left_speed)))
            right_speed = int(max(-100, min(100, right_speed)))

            print(f"Relative position is: {et.retrieve_motors_relative_position()}")

            et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status(), mode)

            if send_video_stream and client_socket is not None:
                status = et.get_spike_status()
                left_pos = status.motors["A"].relative_position
                right_pos = status.motors["B"].relative_position

                info: dict[str, Any] = {}
                info["target_x"], info["offset_y"] = x1 + mx, y1 + my
                info["text"] = {
                    "mode": mode.name,
                    "distance_sensor": status.sensors.distance,
                    "reflective_sensor": status.sensors.color.reflected,
                    "left_relative_position": left_pos,
                    "right_relative_position": right_pos,
                }

                # Create visualization frame
                gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                gray = draw_driving_info(gray, info, (x1, y1, x2, y2))
                # Draw contour on the visualization if found
                if max_contour is not None:
                    # Adjust contour coordinates to full frame
                    adjusted_contour = max_contour + np.array([x1, y1])
                    cv2.drawContours(gray, [adjusted_contour], -1, (255, 255, 255), 2)  # Draw centroid
                    cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)

                try:
                    ret, buffer = cv2.imencode(".jpg", gray)
                    img_encoded = buffer.tobytes()
                    data = pickle.dumps(img_encoded)
                    client_socket.sendall(struct.pack("L", len(data)) + data)
                except Exception as e:
                    print(f"Socket error: {e}")
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        et.stop()
        cap.release()

        # Close socket connection if it was opened
        if client_socket is not None:
            client_socket.close()

        # Clean up video writer if it was used
        if save_camera_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")

        # Clean up sensor recorder if it was used
        if record_sensor_data and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")
        # Restore terminal settings on exit
        keyboard.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the OpenCV-based line following robot with optional sensor recording and video saving"
    )
    parser.add_argument("--record-sensor", action="store_true", help="Record sensor data to file")
    parser.add_argument("--save-video", action="store_true", help="Save camera video to file")
    parser.add_argument("--send-video", action="store_true", help="Send video stream to host PC")
    parser.add_argument(
        "--course",
        choices=["left", "right"],
        default="left",
        help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: left)",
    )
    parser.add_argument(
        "--initial-mode",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help=(
            "Initial mode to start with: 0=left_edge, 1=right_edge, 2=avoid_obstacle, "
            "3=head_bottle1, 4=carry_bottle1, 5=head_bottle2"
        ),
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file to load")

    args = parser.parse_args()

    # Convert number mode to Mode enum
    mode_mapping = {
        0: Mode.FOLLOW_LEFT_EDGE,
        1: Mode.FOLLOW_RIGHT_EDGE,
        2: Mode.AVOID_OBSTACLE,
        16: Mode.PAUSE,
    }

    initial_mode = mode_mapping.get(args.initial_mode) if args.initial_mode else None

    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print(f"course: {args.course}")
    print(f"Initial mode: {args.initial_mode if args.initial_mode else 'Default (based on course)'}")
    print(f"Video streaming to host PC: {'Enabled' if args.send_video else 'Disabled'}")
    print("Controls:")
    print("  'a' - Follow left edge")
    print("  'd' - Follow right edge")
    print("  'q' - Quit")
    print("Press Ctrl+C to stop")

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
        course=args.course,
        initial_mode=initial_mode,
        model_path=args.model_path,
    )
