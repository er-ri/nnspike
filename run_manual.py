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
from nnspike.unit.actions import avoid_obstacle, turn_right, turn_left
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info, get_line_edges_at_y, get_virtual_line_edges_at_y
from nnspike.utils import find_bottle_center_with_yellow_count, find_bottle_center_with_red_count, find_bottle_center_with_blue_count

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


def set_action_mode_defaults(x1, x2, y1, y2):
    """
    アクションモード用のダミー値を設定する共通関数
    """
    target_x = None  # 通常制御は行わない
    # 可視化用ダミー値
    mx = (x2 - x1) // 2
    my = (y2 - y1) // 2
    offset_pixels = 0
    max_contour = None
    # アクションモード用のダミー値
    theta = 0.0
    steering_correction = 0.0
    
    return target_x, mx, my, offset_pixels, max_contour, theta, steering_correction


def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False, initial_course="left"):
    # Initialize edge following preference based on the initial_course parameter
    #mode = Mode.LEFT_EDGE_FOLLOWING if initial_course == "left" else Mode.RIGHT_EDGE_FOLLOWING
    mode = Mode.PAUSE  # 最初はpause状態で開始

    # 障害物回避用の状態管理
    action_state = None  # None:通常, dict:アクション実行中
    previous_mode = None
    pre_target_x = None  # 前回のtarget_xを保持（軌道安定性のため）
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
        )

    # Socket connection for sending camera capture (only if enabled)
    client_socket = None
    if send_video_stream:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((HOST_IP_ADDRESS, 8485))
            print(f"Connected to host PC at {HOST_IP_ADDRESS}:8485 for video streaming")
        except Exception as e:
            print(f"Warning: Could not connect to host PC for video streaming: {e}")
            client_socket = None

    # Initialize robot, PID controller, and keyboard controller
    et = ETRobot()
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

    #et.move_arm(1, 1.0)  # アームを上げる
    #et.move_arm(0, 1.0)  # アームを下げる
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
            elif key == "a":  # 'a' key for left
                mode = Mode.LEFT_EDGE_FOLLOWING
                print("Switched to following: left edge")
            elif key == "d":  # 'd' key for right
                mode = Mode.RIGHT_EDGE_FOLLOWING
                print("Switched to following: right edge")
            elif key == "f":  # 'f' key for moving forward
                mode = Mode.FORWARD
                print("Switched to forward mode")
            elif key == "b":  # 'b' key for moving backward
                mode = Mode.BACKWARD
                print("Switched to backward mode")
            elif key == "p":  # 'p' key for pause
                mode = Mode.PAUSE
                print("Switched to pause mode")
            elif key == "o":  # 'o' key to avoid obstacle
                if action_state is None:
                    previous_mode = mode  # Save current mode
                    mode = Mode.OBSTACLE_AVOIDANCE
                    action_state = None  # avoid_obstacle_stepの初期化
                    print("Avoiding obstacle...")
                continue
            elif key == "g":  # 'g' key for red bottle to gate mode
                mode = Mode.RED_BOTTLE_TO_GATE
                print("Switched to red bottle to gate mode")
                continue
            elif key == "r":  # 'r' key to turn right
                if action_state is None:
                    previous_mode = mode  # Save current mode
                    mode = Mode.TURN_RIGHT
                    action_state = None  # turn_right_stepの初期化
                    print("Turning right...")
                continue
            elif key == "l":  # 'l' key to turn left
                if action_state is None:
                    previous_mode = mode  # Save current mode
                    mode = Mode.TURN_LEFT
                    action_state = None  # turn_left_stepの初期化
                    print("Turning left...")
                continue

            match mode:
                case Mode.OBSTACLE_AVOIDANCE:
                    # 障害物回避モード: 1フレーム分の指示を取得
                    action_state, left_speed, right_speed, finished = avoid_obstacle(action_state, et, frame)
                    if finished:
                        mode = previous_mode
                        action_state = None
                    target_x, mx, my, offset_pixels, max_contour, theta, steering_correction = set_action_mode_defaults(x1, x2, y1, y2)
                case Mode.TURN_RIGHT:
                    # 右旋回モード: 1フレーム分の指示を取得
                    action_state, left_speed, right_speed, finished = turn_right(action_state, et, frame)
                    if finished:
                        mode = previous_mode
                        action_state = None
                    target_x, mx, my, offset_pixels, max_contour, theta, steering_correction = set_action_mode_defaults(x1, x2, y1, y2)
                case Mode.TURN_LEFT:
                    # 左旋回モード: 1フレーム分の指示を取得
                    action_state, left_speed, right_speed, finished = turn_left(action_state, et, frame)
                    if finished:
                        mode = previous_mode
                        action_state = None
                    target_x, mx, my, offset_pixels, max_contour, theta, steering_correction = set_action_mode_defaults(x1, x2, y1, y2)
                case Mode.LEFT_EDGE_FOLLOWING:
                    left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    target_x = left_x
                case Mode.RIGHT_EDGE_FOLLOWING:
                    _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    target_x = right_x
                case Mode.BOTTLE_CARRYING:
                    print("Not implemented: Bottle carrying mode")
                    target_x = (x1 + x2) // 2
                case Mode.RED_BOTTLE_TO_GATE:
                    # 仮想ラインエッジを使用して赤ボトルからゲートへ走行
                    # OFFSET_Yを使用
                    target_x = get_virtual_line_edges_at_y(frame, OFFSET_Y, previous_center_x=pre_target_x)
                    # このモード内で前回のtarget_xを更新（軌道安定性のため）
                    if target_x is not None:
                        pre_target_x = target_x
                    elif pre_target_x is None:
                        pre_target_x = (x1 + x2) // 2  # 初期値設定
                case Mode.FORWARD:
                    # 画像全体の黄色重心に向かって進む
                    yellow_cx, _, yellow_pixel_count = find_bottle_center_with_yellow_count(frame)
                    red_cx, _, red_pixel_count = find_bottle_center_with_red_count(frame)
                    if yellow_pixel_count > 14000:
                        if action_state is None:
                            previous_mode = mode
                            mode = Mode.OBSTACLE_AVOIDANCE
                            action_state = None
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
                    else:
                        target_x = (x1 + x2) // 2
                case _:
                    # Default to center if invalid edge specified
                    target_x = (x1 + x2) // 2

            # OBSTACLE_AVOIDANCE、TURN_RIGHT、TURN_LEFT以外のときは通常の制御値計算
            if mode not in [Mode.OBSTACLE_AVOIDANCE, Mode.TURN_RIGHT, Mode.TURN_LEFT]:
                if target_x is not None:
                    # Calculate position relative to ROI
                    mx = target_x - x1  # Relative to ROI
                    my = OFFSET_Y - y1  # Relative to ROI

                    # Calculate offset from ROI center
                    roi_center_x = (x2 - x1) // 2
                    offset_pixels = mx - roi_center_x

                    # Create a simple contour for visualization (approximate target point)
                    max_contour = np.array([[[mx, my]]], dtype=np.int32)
                else:
                    # No line detected, use center values
                    mx = (x2 - x1) // 2
                    my = (y2 - y1) // 2
                    offset_pixels = 0
                    max_contour = None  # Calculate attitude angle using camera geometry

                theta = calculate_attitude_angle(offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS)  # Use simplified speed control
                current_base_speed = BASE_SPEED

                steering_correction = pid.update(theta)

                # Apply simple differential steering
                left_speed = current_base_speed - steering_correction
                right_speed = current_base_speed + steering_correction

                # Clamp speed values to valid range
                left_speed = int(max(0, min(100, left_speed)))
                right_speed = int(max(0, min(100, right_speed)))

            # モーター制御の集約: BACKWARD以外はforwardで統一
            if mode == Mode.BACKWARD:
                et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
            elif mode == Mode.PAUSE:
                et.brake()
            else:
                et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status(), mode)

            status = et.get_spike_status()
            left_pos = status.motors["A"].relative_position
            right_pos = status.motors["B"].relative_position

            info = dict()
            info["target_x"], info["offset_y"] = x1 + int(mx), y1 + int(my)
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
                cv2.circle(gray, (int(x1 + int(mx)), int(y1 + int(my))), 5, (255, 255, 255), -1)
            if send_video_stream and client_socket is not None:
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
    parser.add_argument(
        "--initial-course", choices=["left", "right"], default="left", help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: left)"
    )

    args = parser.parse_args()

    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print(f"Initial course: Following {args.initial_course} edge")
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
        initial_course=args.initial_course,
    )
