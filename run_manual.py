#!/usr/bin/env python3
"""
OpenCV-Based Line Following Robot Control

This script controls a line-following robot using OpenCV for image processing
instead of neural network predictions. It uses the get_line_edges_at_y function
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

import cv2
import numpy as np

from nnspike.constants import CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, ROI_CNN, Mode
from nnspike.unit import ETRobot
from nnspike.unit.action_chain import ActionChain
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info, get_line_edges_at_y, get_virtual_line_edges_at_y, find_bottle_center, find_blue_target_center

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest for OpenCV processing

# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 45  # Base speed for straight lines (adjust this first)

# Socket connection settings
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 25)
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
        """Get a single keypress"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1).lower()
            return key
        return None

    def cleanup(self):
        """Restore terminal settings"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)  # type: ignore


def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="left", initial_mode=None):
    pre_target_x = None  # GATE_PASS用の前回値
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
    action_chain = ActionChain(et, course)

    # Set initial mode to PAUSE (initial_mode/course-based logic is disabled)
    # if initial_mode:
    #     mode = initial_mode
    # else:
    #     mode = Mode.FOLLOW_LEFT_EDGE if course == "left" else Mode.FOLLOW_RIGHT_EDGE
    mode = Mode.PAUSE

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

    et.move_arm(1, 1.0)  # アームを上げる（1: up）
    et.move_arm(0, 1.0)  # アームを下げる（0: down）
    et.move_arm(2, 0.5)  # アームを止める
    et.set_motor_relative_position(left_positon=0, right_position=0)

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
            if key == "q":  # 'q' key to quit
                print("Quitting...")
                keyboard.running = False
                break
            elif key == "a":
                mode = Mode.FOLLOW_LEFT_EDGE
                print("Switched to following: left edge")
            elif key == "d":
                mode = Mode.FOLLOW_RIGHT_EDGE
                print("Switched to following: right edge")
            elif key == "r":
                mode = Mode.TURN_RIGHT
                print("Switched to turn right mode")
            elif key == "l":
                mode = Mode.TURN_LEFT
                print("Switched to turn left mode")
            elif key == "f":
                mode = Mode.FORWARD
                print("Switched to forward mode")
            elif key == "b":
                mode = Mode.BACKWARD
                print("Switched to backward mode")
            elif key == "g":
                mode = Mode.GATE_PASS
                print("Switched to gate pass mode")
            elif key == "e":
                mode = Mode.EYE_BLUE
                print("Switched to blue eyes mode")
            elif key == "2":
                mode = Mode.AVOID_OBSTACLE
                print("Switched to obstacle avoidance mode")
            elif key == "3":
                mode = Mode.HEAD_BOTTLE1
                print("Switched to heading bottle 1 mode")
            elif key == "4":
                mode = Mode.CARRY_BOTTLE1
                print("Switched to bottle carrying 1 mode")
            elif key == "5":
                mode = Mode.HEAD_BOTTLE2
                print("Switched to heading bottle 2 mode")
            elif key == "6":
                mode = Mode.CARRY_BOTTLE2
                print("Switched to bottle carrying 2 mode")
            elif key == "7":
                mode = Mode.HEAD_GOAL
                print("Switched to heading goal mode")
            elif key == "8":
                mode = Mode.PAUSE
                print("Pausing robot")
            # GATE_PASS: ゲートを潜る（仮実装: 直進）
            # EYE_BLUE: ブルーアイズを目標に動作（仮実装: 青重心に向かう）

            target_x = None  # Default target x position
            left_speed, right_speed = None, None  # Initialize speeds



            match mode:
                case Mode.FOLLOW_LEFT_EDGE:
                    _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    target_x = right_x
                case Mode.FOLLOW_RIGHT_EDGE:
                    left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    target_x = left_x
                case Mode.AVOID_OBSTACLE:
                    _, (left_speed, right_speed), mode = action_chain.avoid_obstacle()
                case Mode.TURN_LEFT:
                    _, (left_speed, right_speed), mode = action_chain.trun_left()
                case Mode.TURN_RIGHT:
                    _, (left_speed, right_speed), mode = action_chain.trun_right()
                case Mode.HEAD_BOTTLE1:
                    target_x, mode = action_chain.heading_bottle1(frame)
                case Mode.CARRY_BOTTLE1:
                    target_x, mode = action_chain.carry_bottle1(frame)
                case Mode.HEAD_BOTTLE2:
                    target_x, mode = action_chain.heading_bottle2(frame)
                case Mode.CARRY_BOTTLE2:
                    target_x, mode = action_chain.carry_bottle2(frame)
                case Mode.HEAD_GOAL:
                    target_x, mode = action_chain.heading_goal(frame)
                case Mode.FORWARD:
                    # 画像全体の黄色重心・赤色重心に向かって進む（find_bottle_center使用）
                    yellow_cx, _, yellow_pixel_count = find_bottle_center(frame, color="yellow")
                    red_cx, _, red_pixel_count = find_bottle_center(frame, color="red")
                    blue_cx, _, blue_pixel_count = find_bottle_center(frame, color="blue")
                    if yellow_pixel_count > 14000:
                        mode = Mode.AVOID_OBSTACLE
                        print("Avoiding obstacle (auto FORWARD)...")
                        target_x = (x1 + x2) // 2
                    elif yellow_pixel_count > 4000:
                        if yellow_cx is not None:
                            target_x = yellow_cx[0]  # X座標のみを取得
                        else:
                            target_x = (x1 + x2) // 2
                    elif red_pixel_count > 4000:
                        if red_cx is not None:
                            target_x = red_cx[0]  # X座標のみを取得
                        else:
                            target_x = (x1 + x2) // 2
                    elif blue_pixel_count > 4000:
                        if blue_cx is not None:
                            target_x = blue_cx[0]  # X座標のみを取得
                        else:
                            target_x = (x1 + x2) // 2
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.GATE_PASS:
                    # ゲートを潜る: 仮想ラインエッジを使う
                    target_x = get_virtual_line_edges_at_y(frame, OFFSET_Y, previous_center_x=pre_target_x)
                    if target_x is not None:
                        pre_target_x = target_x
                    elif pre_target_x is None:
                        pre_target_x = (x1 + x2) // 2  # 初期値
                case Mode.EYE_BLUE:
                    # ブルーアイズ（青重心）に向かう: find_blue_target_centerを使用
                    center, area, blue_pixel_count = find_blue_target_center(frame)
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.BACKWARD:
                    left_speed = -BASE_SPEED
                    right_speed = -BASE_SPEED
                case Mode.PAUSE:
                    left_speed, right_speed = 0, 0
                case _:
                    # Default to center if invalid edge specified
                    target_x = (x1 + x2) // 2

            if target_x is not None:
                # Calculate position relative to ROI
                mx = target_x - x1  # Relative to ROI
                my = OFFSET_Y - y1  # Relative to ROI

                # Calculate offset from ROI center
                roi_center_x = (x2 - x1) // 2
                offset_pixels = mx - roi_center_x

                # Create a simple contour for visualization (approximate target point)
                max_contour = np.array([[[mx, my]]], dtype=np.int32)

                theta = calculate_attitude_angle(offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS)  # Use simplified speed control
                current_base_speed = BASE_SPEED

                steering_correction = pid.update(theta)

                # Apply simple differential steering
                left_speed = current_base_speed - steering_correction
                right_speed = current_base_speed + steering_correction

                # Clamp speed values to valid range
                left_speed = int(max(0, min(100, left_speed)))
                right_speed = int(max(0, min(100, right_speed)))

            # Temporarily set Heading Gate mode
            if mode == Mode.PAUSE:
                et.brake()
            else:
                et.set_motor_speed(
                    left_speed=left_speed,
                    right_speed=right_speed,
                )

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status(), mode)

            if send_video_stream and client_socket is not None:
                status = et.get_spike_status()
                left_pos = status.motors["A"].relative_position
                right_pos = status.motors["B"].relative_position

                info = dict()
                info["target_x"], info["offset_y"] = x1 + mx, y1 + my
                info["text"] = {
                    "mode": mode.name,
                    "left_relative_position": left_pos,
                    "right_relative_position": right_pos,
                    "theta_deg": round(math.degrees(theta), 2),
                    "steering_correction": round(steering_correction, 2),
                    "left_speed": left_speed,
                    "right_speed": right_speed,
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
    parser = argparse.ArgumentParser(description="Run the OpenCV-based line following robot with optional sensor recording and video saving")
    parser.add_argument("--record-sensor", action="store_true", help="Record sensor data to file")
    parser.add_argument("--save-video", action="store_true", help="Save camera video to file")
    parser.add_argument("--send-video", action="store_true", help="Send video stream to host PC")
    parser.add_argument("--course", choices=["left", "right"], default="left", help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: left)")
    parser.add_argument(
        "--initial-mode",
        choices=["left_edge", "right_edge", "obstacle_avoidance", "heading_bottle1", "bottle_carrying1", "heading_bottle2", "bottle_carrying2", "heading_goal", "pause"],
        help="Initial mode to start with (overrides initial-course if specified)",
    )

    args = parser.parse_args()

    # Convert string mode to Mode enum
    mode_mapping = {
        "left_edge": Mode.FOLLOW_LEFT_EDGE,
        "right_edge": Mode.FOLLOW_RIGHT_EDGE,
        "obstacle_avoidance": Mode.AVOID_OBSTACLE,
        "heading_bottle1": Mode.HEAD_BOTTLE1,
        "bottle_carrying1": Mode.CARRY_BOTTLE1,
        "heading_bottle2": Mode.HEAD_BOTTLE2,
        "bottle_carrying2": Mode.CARRY_BOTTLE2,
        "heading_goal": Mode.HEAD_GOAL,
        "pause": Mode.PAUSE,
        "turn_left": Mode.TURN_LEFT,
        "turn_right": Mode.TURN_RIGHT,
        "forward": Mode.FORWARD,
        "backward": Mode.BACKWARD,
        "gate_pass": Mode.GATE_PASS,
        "eye_blue": Mode.EYE_BLUE,
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
    )
