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
import time

import termios
import tty

import cv2
import numpy as np
import torch

from nnspike.constants import CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, RELATIVE_POSITION_SCALE, ROI_CNN, Mode
from nnspike.models import NvidiaModel
from nnspike.unit import ETRobot
from nnspike.unit.action_chain import ActionChain
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info, get_line_edges_at_y, get_virtual_line_edges_at_y, find_bottle_center, find_blue_target_center
from scripts.utils import process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="left", model_path=None):
    def nvidia_model_predict(frame, left_pos, right_pos, model):
        """NVIDIAモデルによる予測を行う。"""
        try:
            roi_area = process_image(image=frame.copy(), device=device, roi=(x1, y1, x2, y2))
            rel_pos_a = left_pos if left_pos is not None else 0
            rel_pos_b = right_pos if right_pos is not None else 0
            relative_pos_value = abs(rel_pos_a) + abs(rel_pos_b)
            relative_position = abs(relative_pos_value / RELATIVE_POSITION_SCALE) if relative_pos_value is not None else 0.0
            relative_position = torch.tensor(relative_position, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(roi_area, relative_position)
            prob, mode = torch.max(outputs[0], dim=1)
            nvidia_prob = round(prob[0].item(), 2)
            nvidia_mode_prediction = mode.item()
            nvidia_prediction = x1 + (outputs[1][0][0] * (x2 - x1)).detach().item()
            return nvidia_prediction, nvidia_mode_prediction, nvidia_prob
        except Exception as e:
            print(f"NVIDIA model prediction error: {e}")
            return None, None, None

    def unpack_action_result(result, default_mode=Mode.PAUSE):
        # Noneや不正な戻り値も吸収して安全にアンパック
        if result is None:
            return None, (0, 0), default_mode
        if len(result) == 3:
            target_x, speeds, mode = result
            if speeds is None:
                speeds = (0, 0)
            if mode is None:
                mode = default_mode
            return target_x, speeds, mode
        return None, (0, 0), default_mode

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
        # フレームサイズがNoneや不正な場合はデフォルト(640,480)を使う
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not frame_width or not frame_height:
            frame_width, frame_height = 640, 480
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(frame_width, frame_height),
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

    # Initialize NVIDIA model if enabled
    model = None
    if model_path:
        print(f"Loading NVIDIA model from: {model_path}")
        model = NvidiaModel()
        try:
            state_dict = torch.load(model_path, map_location=device)
            model_state = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered_state_dict)
            model.load_state_dict(model_state)
            model.eval()
            print("NVIDIA model loaded successfully (filtered)")
        except Exception as e:
            print(f"Error loading NVIDIA model: {e}")
            model = None

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

    #et.move_arm(1, 1.0)  # アームを上げる（1: up）
    #et.move_arm(0, 1.0)  # アームを下げる（0: down）
    #et.move_arm(2, 0.5)  # アームを止める
    et.set_motor_relative_position(left_positon=0, right_position=0)

    try:
        while et.is_running and keyboard.running:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # 毎ループ1回だけstatusを取得
            status = et.get_spike_status()
            left_pos = status.motors["A"].relative_position
            right_pos = status.motors["B"].relative_position

            # NVIDIAモデル予測用変数の初期化
            nvidia_prediction = None
            nvidia_mode_prediction = None
            nvidia_prob = None

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(status, mode)

            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)

            # Send video stream and driving info if enabled (must be after frame, target_x, etc. are set)
            if send_video_stream and client_socket is not None:

                info = dict()
                # target_x, offset_yがNoneでない場合のみint変換（通常はint変換で例外は起きない想定）
                if target_x is not None and offset_y is not None:
                    info["target_x"] = int(target_x) if isinstance(target_x, (int, float)) else None
                    info["offset_y"] = int(offset_y) if isinstance(offset_y, (int, float)) else None
                else:
                    info["target_x"], info["offset_y"] = None, None
                # draw_driving_infoでint()エラーを防ぐため、Noneなら0に変換
                info["target_x"] = info["target_x"] if info["target_x"] is not None else 0
                info["offset_y"] = info["offset_y"] if info["offset_y"] is not None else 0
                info["text"] = {
                    "mode": mode.name,
                    "left_relative_position": int(left_pos) if left_pos is not None else 0,
                    "right_relative_position": int(right_pos) if right_pos is not None else 0,
                    "theta_deg": round(math.degrees(theta), 2) if theta is not None else 0,
                    "steering_correction": round(steering_correction, 2) if steering_correction is not None else 0,
                    "left_speed": int(left_speed) if left_speed is not None else 0,
                    "right_speed": int(right_speed) if right_speed is not None else 0,
                }
                
                # Add NVIDIA model info if available
                # if use_nvidia_model and nvidia_prediction is not None:
                #     info["text"]["nvidia_x"] = round(nvidia_prediction, 2)
                #     info["text"]["nvidia_mode"] = nvidia_mode_prediction
                #     info["text"]["nvidia_prob"] = nvidia_prob

                # Create visualization frame
                gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                gray = draw_driving_info(gray, info, (x1, y1, x2, y2))
                # Draw contour on the visualization if found
                if max_contour is not None and mx is not None and my is not None:
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
            elif key == "j":
                mode = Mode.SMALL_TURN_LEFT
                print("Switched to small turn left mode")
            elif key == "k":
                mode = Mode.SMALL_TURN_RIGHT
                print("Switched to small turn right mode")
            elif key == "i":
                mode = Mode.TURN_LEFT_RELATIVE
                print("Switched to turn left (relative) mode")
            elif key == "o":
                mode = Mode.TURN_RIGHT_RELATIVE
                print("Switched to turn right (relative) mode")
            elif key == "b":
                mode = Mode.BACKWARD
                print("Switched to backward mode")
            elif key == "g":
                mode = Mode.GATE_PASS
                print("Switched to gate pass mode")
            elif key == "e":
                mode = Mode.EYE_BLUE
                print("Switched to blue eyes mode")
            elif key == "u":
                mode = Mode.BLUE_BOTTLE_CATCH
                print("Switched to blue bottle catch mode")
            elif key == "2":
                mode = Mode.AVOID_OBSTACLE
                print("Switched to obstacle avoidance mode")
            elif key == "3":
                mode = Mode.CARRY_BOTTLE1
                print("Switched to bottle carrying 1 mode")
            elif key == "4":
                mode = Mode.BACK_AND_TURN1
                print("Switched to back and turn 1 mode")
            elif key == "5":
                mode = Mode.CARRY_BOTTLE2
                print("Switched to bottle carrying 2 mode")
            elif key == "6":
                mode = Mode.BACK_AND_TURN2
                print("Switched to back and turn 2 mode")
            elif key == "7":
                mode = Mode.HEAD_GOAL
                print("Switched to heading goal mode")
            elif key == "8" or key == "p":
                mode = Mode.PAUSE
                print("Pausing robot")
            elif key == "n":
                if model is not None:
                    mode = Mode.NVIDIA_FOLLOW
                    print("Switched to NVIDIA model following mode")
                else:
                    print("NVIDIA model not available")
            elif key == "t":
                mode = Mode.TURN_AT_END
                print("Switched to Turn at the end mode")

            target_x = None  # Default target x position
            offset_y = None  # target_y相当も初期化
            left_speed, right_speed = None, None  # Initialize speeds

            match mode:
                case Mode.TURN_LEFT_RELATIVE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_left_relative(frame))
                case Mode.TURN_RIGHT_RELATIVE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_right_relative(frame))
                case Mode.TURN_AT_END:
                    target_x, (left_speed, right_speed), ret_mode = unpack_action_result(action_chain.turn_at_end(frame))
                    if ret_mode == Mode.FOLLOW_RIGHT_EDGE:
                        mode = Mode.FOLLOW_RIGHT_EDGE
                case Mode.BLUE_BOTTLE_CATCH:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.blue_bottle_catch(frame))
                case Mode.FOLLOW_LEFT_EDGE:
                    left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    if left_x is not None:
                        target_x = left_x
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.FOLLOW_RIGHT_EDGE:
                    yellow_result = find_bottle_center(frame, color="yellow")
                    if yellow_result is not None:
                        if len(yellow_result) == 3:
                            yellow_cx, _, yellow_pixel_count = yellow_result
                        else:
                            yellow_cx, yellow_pixel_count = None, 0
                    else:
                        yellow_cx, yellow_pixel_count = None, 0
                    # シンプルに右モーターの相対位置はright_posを使う
                    _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    if yellow_pixel_count > 14000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        mode = Mode.AVOID_OBSTACLE
                        print("Avoiding obstacle (auto FOLLOW_RIGHT_EDGE)...")
                        target_x = (x1 + x2) // 2
                    elif yellow_pixel_count > 3000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        target_x = yellow_cx[0]  # X座標のみを取得
                    elif right_x is not None:
                        target_x = right_x
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.AVOID_OBSTACLE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.avoid_obstacle())
                case Mode.TURN_LEFT:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.trun_left())
                case Mode.SMALL_TURN_LEFT:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.small_turn_left())
                case Mode.TURN_RIGHT:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.trun_right())
                case Mode.SMALL_TURN_RIGHT:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.small_turn_right())
                case Mode.CARRY_BOTTLE1:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.carry_bottle1_relative(frame))
                case Mode.BACK_AND_TURN1:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.back_and_turn1_relative(frame))
                    # 後退フェーズ（両輪とも正方向速度）の場合はset_motor_backward_speedを使う
                    if left_speed == BASE_SPEED and right_speed == BASE_SPEED:
                        et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                        continue  # 以降のset_motor_speed処理をスキップ
                case Mode.CARRY_BOTTLE2:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.carry_bottle2_relative(frame))
                case Mode.BACK_AND_TURN2:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.back_and_turn2_relative(frame))
                    # 後退フェーズ（両輪とも正方向速度）の場合はset_motor_backward_speedを使う
                    if left_speed == BASE_SPEED and right_speed == BASE_SPEED:
                        et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                        continue  # 以降のset_motor_speed処理をスキップ
                case Mode.HEAD_GOAL:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.heading_goal_relative(frame))
                case Mode.FORWARD:
                    # 赤色重心に向かって進む（find_bottle_center使用）。イエロー・ブルー検知は行わない。
                    red_cx, _, red_pixel_count = find_bottle_center(frame, color="red")
                    if red_pixel_count > 3000:
                        if red_cx is not None:
                            target_x = red_cx[0]  # X座標のみを取得
                        else:
                            target_x = (x1 + x2) // 2
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.GATE_PASS:
                    # ゲートを潜る: 仮想ラインエッジを使う
                    temp_x = get_virtual_line_edges_at_y(frame, OFFSET_Y, previous_center_x=pre_target_x)
                    if temp_x is not None:
                        target_x = temp_x
                        pre_target_x = temp_x
                    elif pre_target_x is not None:
                        target_x = pre_target_x
                    else:
                        target_x = (x1 + x2) // 2
                        pre_target_x = target_x
                case Mode.EYE_BLUE:
                    # ブルーアイズ（青重心）に向かう: find_blue_target_centerを使用
                    center, area, blue_pixel_count = find_blue_target_center(frame)
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.BACKWARD:
                    # set_motor_backward_speedで後退（左右は入れ替えない）
                    left_speed = BASE_SPEED
                    right_speed = BASE_SPEED
                    et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                    continue  # 以降のset_motor_speed処理をスキップ
                case Mode.PAUSE:
                    left_speed, right_speed = 0, 0
                case Mode.NVIDIA_FOLLOW:
                    # NVIDIAモデル予測をこのcase内で実行（right_pos >= 7000の場合のみ）
                    if model is not None and right_pos is not None and abs(right_pos) >= 7000:
                        nvidia_prediction, nvidia_mode_prediction, nvidia_prob = nvidia_model_predict(frame, left_pos, right_pos, model)
                    # NVIDIA_FOLLOWの処理をaction_chainに委譲
                    target_x, speeds, mode = action_chain.nvidia_follow(frame, nvidia_mode_prediction)
                    if speeds is not None:
                        left_speed, right_speed = speeds
                case _:
                    # Default to center if invalid edge specified
                    target_x = (x1 + x2) // 2

            # target_xがNoneのときのみ可視化・送信用変数をリセット
            if target_x is None:
                mx = my = theta = steering_correction = None
                max_contour = None
            else:
                # Calculate position relative to ROI
                mx = target_x - x1  # Relative to ROI
                my = OFFSET_Y - y1  # Relative to ROI
                offset_y = y1 + my  # offset_yを明示的にセット

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

            # left_speed/right_speedがNoneなら0に（int()前に必ず実施）
            if left_speed is None:
                left_speed = 0
            if right_speed is None:
                right_speed = 0
            # Clamp speed values to valid range (必ずset_motor_speed前に実施)
            left_speed = int(max(0, min(100, left_speed)))
            right_speed = int(max(0, min(100, right_speed)))

            # Temporarily set Heading Gate mode
            if mode == Mode.PAUSE:
                et.brake()
            else:
                et.set_motor_forward_speed(
                    left_speed=left_speed,
                    right_speed=right_speed,
                )

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
    parser.add_argument("--model-path", help="Path to the trained NVIDIA model file (enables NVIDIA model)")

    args = parser.parse_args()
    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print(f"course: {args.course}")
    print(f"NVIDIA model: {'Enabled' if args.model_path else 'Disabled'}")
    if args.model_path:
        print(f"Model path: {args.model_path}")
    print(f"Video streaming to host PC: {'Enabled' if args.send_video else 'Disabled'}")
    print("Controls:")
    print("  'a' - Follow left edge")
    print("  'd' - Follow right edge")
    print("  'l' - Turn left")
    print("  'r' - Turn right")
    print("  'k' - Small turn left")
    print("  'j' - Small turn right")
    print("  'f' - Forward")
    if args.model_path:
        print("  'n' - NVIDIA model following")
    print("  'q' - Quit")
    print("Press Ctrl+C to stop")

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
        course=args.course,
        model_path=args.model_path,
    )
