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
import nnspike

from nnspike.constants import CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, RELATIVE_POSITION_SCALE, ROI_CNN, Mode, NUM_MODES
from nnspike.unit import ETRobot
from nnspike.unit.action_chain import ActionChain
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info, get_line_edges_at_y, find_bottle_center, find_blue_target_center, get_virtual_line_target_x
from scripts.utils import process_image, model_inference, load_optimized_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable QNNPACK for optimal performance on ARM processors (Raspberry Pi)
torch.backends.quantized.engine = "qnnpack"


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


def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="right", course_type="upper", model_path=None):
    def nvidia_model_predict(model, et: ETRobot):
        """NVIDIAモデルによる予測を行う。ノートブックテスト結果を反映した安定版"""
        try:
            roi_area = process_image(image=model_input_frame, device=device, roi=(x1, y1, x2, y2))
            
            # ETRobot.retrieve_motors_relative_position()は1つの値（int）を返す
            # 両モーターの絶対値の合計: motor_a_position + motor_b_position
            relative_position = et.retrieve_motors_relative_position()
            scaled_relative_position = relative_position / RELATIVE_POSITION_SCALE
            tensor_relative_position = torch.tensor([scaled_relative_position], dtype=torch.float32).unsqueeze(0).to(device)
            
            # model_inference関数を使用（torch.no_grad()は関数内で実行される）
            nvidia_prediction, (nvidia_mode_prediction, nvidia_prob) = model_inference(model, roi_area, tensor_relative_position)
            
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
    video_filename = None  # Initialize to avoid UnboundLocalError
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
        )

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
    action_chain = ActionChain(et, course, course_type)

    # Initialize NVIDIA model if enabled
    model = None
    if model_path:
        print(f"Loading NVIDIA model from: {model_path}")
        try:
            model = load_optimized_model(model_path, device)
            model.eval()
            print("SUCCESS: NVIDIA model loaded successfully")
            total_params = sum(p.numel() for p in model.parameters())
            # 実際のクラス数をmode_classifier.weight.shape[0]から取得
            if hasattr(model, 'mode_classifier') and hasattr(model.mode_classifier, 'weight'):
                actual_classes = model.mode_classifier.weight.shape[0]
            else:
                actual_classes = 'Unknown'
            print(f"Model parameters: {total_params:,}, Classes: {actual_classes}")
        except Exception as e:
            print(f"ERROR: Error loading NVIDIA model: {e}")
            print("Continuing without NVIDIA model...")
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

    # フレームカウンターとモデル予測結果を保持する変数
    frame_counter = 0
    last_nvidia_prediction = None
    last_nvidia_mode_prediction = None
    last_nvidia_prob = None

    #et.move_arm(1, 1.0)  # アームを上げる（1: up）
    #et.move_arm(0, 1.0)  # アームを下げる（0: down）
    #et.move_arm(2, 0.5)  # アームを止める
    et.set_motor_relative_position(left_positon=0, right_position=0)

    # --- ここから未定義エラー防止のための宣言（関数スコープ） ---
    target_x = None
    offset_y = None
    theta = None
    steering_correction = None
    left_speed = None
    right_speed = None
    mx = None
    my = None
    max_contour = None
    # --- ここまで ---

    try:
        while et.is_running and keyboard.running:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # course leftのときはmodel入力用画像を左右反転
            model_input_frame = frame.copy()
            if course == "left":
                model_input_frame = cv2.flip(model_input_frame, 1)
            # 毎ループ1回だけstatusを取得
            status = et.get_spike_status()
            left_pos = status.motors["A"].relative_position
            right_pos = status.motors["B"].relative_position

            # NVIDIAモデル予測をright_posが22000以下の時のみ実行（フレームスキップで負荷軽減）
            nvidia_prediction = None
            nvidia_mode_prediction = None
            nvidia_prob = None
            frame_counter += 1
            
            # 3フレームに1回だけモデル予測を実行（負荷軽減）
            if model is not None and (right_pos is None or abs(right_pos) <= 22000) and frame_counter % 3 == 0:
                last_nvidia_prediction, last_nvidia_mode_prediction, last_nvidia_prob = nvidia_model_predict(model, et)
            
            # 最新の予測結果を使用
            nvidia_prediction = last_nvidia_prediction
            nvidia_mode_prediction = last_nvidia_mode_prediction
            nvidia_prob = last_nvidia_prob

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(status, mode)

            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                if course == "left":
                    video_writer.write(model_input_frame)
                else:
                    video_writer.write(frame)

            # Send video stream and driving info if enabled (must be after frame, target_x, etc. are set)
            if send_video_stream and client_socket is not None:

                info = dict()
                # target_x, offset_yがNoneの場合は0にして送信（video/可視化側でNoneを扱わない）
                safe_target_x = int(target_x) if isinstance(target_x, (int, float)) and target_x is not None else 0
                safe_offset_y = int(offset_y) if isinstance(offset_y, (int, float)) and offset_y is not None else 0
                info["target_x"] = safe_target_x
                info["offset_y"] = safe_offset_y
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

            # --- ここから未定義エラー防止のための初期化 ---
            target_x = None
            offset_y = None
            theta = None
            steering_correction = None
            left_speed = None
            right_speed = None
            mx = None
            my = None
            max_contour = None
            # --- ここまで ---

            match mode:
                case Mode.TURN_LEFT_RELATIVE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_left_relative(frame))
                case Mode.TURN_RIGHT_RELATIVE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_right_relative(frame))
                case Mode.BLUE_BOTTLE_CATCH:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.blue_bottle_catch(frame))
                case Mode.FOLLOW_LEFT_EDGE:
                    yellow_cx, _, yellow_pixel_count = find_bottle_center(frame, color="yellow")
                    left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    # left_posが7000を超えたらNVIDIA_FOLLOWに切り替え
                    if left_pos is not None and abs(left_pos) >= 7000:
                        if model is not None:
                            mode = Mode.NVIDIA_FOLLOW
                            print("Switched to NVIDIA_FOLLOW mode")
                            # NVIDIA_FOLLOWの処理は次のループで実行される
                        else:
                            print("NVIDIA model not available, continuing with FOLLOW_LEFT_EDGE")
                    elif yellow_pixel_count > 16000 and yellow_cx is not None and left_pos is not None and abs(left_pos) < 7000:
                        mode = Mode.AVOID_OBSTACLE
                        target_x = (x1 + x2) // 2
                    elif yellow_pixel_count > 3000 and yellow_cx is not None and left_pos is not None and abs(left_pos) < 7000:
                        target_x = yellow_cx[0]  # X座標のみを取得
                    elif left_x is not None:
                        target_x = left_x
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.FOLLOW_RIGHT_EDGE:
                    yellow_cx, _, yellow_pixel_count = find_bottle_center(frame, color="yellow")
                    # シンプルに右モーターの相対位置はright_posを使う
                    _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    
                    # right_posが7000を超えたらNVIDIA_FOLLOWに切り替え
                    if right_pos is not None and abs(right_pos) >= 7000:
                        if model is not None:
                            mode = Mode.NVIDIA_FOLLOW
                            print("Switched to NVIDIA_FOLLOW mode")
                            # NVIDIA_FOLLOWの処理は次のループで実行される
                        else:
                            print("NVIDIA model not available, continuing with FOLLOW_RIGHT_EDGE")
                    elif yellow_pixel_count > 16000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        mode = Mode.AVOID_OBSTACLE
                        target_x = (x1 + x2) // 2
                    elif yellow_pixel_count > 3000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        target_x = yellow_cx[0]  # X座標のみを取得
                    elif right_x is not None:
                        target_x = right_x
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.AVOID_OBSTACLE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.avoid_obstacle_relative(frame))
                case Mode.TURN_LEFT:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_left())
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
                    temp_x = get_virtual_line_target_x(frame, previous_center_x=pre_target_x)
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
                    center, area, blue_pixel_count = find_blue_target_center(frame, gray_ellipse_enable=False)
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
                    # courseによってCARRY_BOTTLE1への切り替え判定を分岐
                    if course == "left":
                        if left_pos is not None and abs(left_pos) > 22000:
                            mode = Mode.CARRY_BOTTLE1
                            print(f"Left position {abs(left_pos)} > 22000, switching to CARRY_BOTTLE1")
                        else:
                            # courseによって左右判定を反転
                            # abs(left_pos)が16000～19000のときは強制的にleft_xをtarget_xにする
                            if left_pos is not None and 16000 <= abs(left_pos) < 19000:
                                left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                                target_x = left_x if left_x is not None else (x1 + x2) // 2
                            # abs(left_pos)が19000～21000のときは強制的にright_xをtarget_xにする
                            elif left_pos is not None and 19000 <= abs(left_pos) <= 21000:
                                _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                                target_x = right_x if right_x is not None else (x1 + x2) // 2
                            # abs(left_pos)が21000より大きい場合は左
                            elif left_pos is not None and abs(left_pos) > 21000:
                                left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                                target_x = left_x if left_x is not None else (x1 + x2) // 2
                            elif nvidia_mode_prediction == Mode.FOLLOW_LEFT_EDGE.value:  # 左エッジ
                                _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                                target_x = right_x if right_x is not None else (x1 + x2) // 2
                            else:  # 左モード以外はすべて右エッジ
                                left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                                target_x = left_x if left_x is not None else (x1 + x2) // 2
                    else:
                        if right_pos is not None and abs(right_pos) > 22000:
                            mode = Mode.CARRY_BOTTLE1
                            print(f"Right position {abs(right_pos)} > 22000, switching to CARRY_BOTTLE1")
                        else:
                            # ...existing code...
                            if nvidia_mode_prediction == Mode.FOLLOW_LEFT_EDGE.value:  # 左エッジ
                                left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                                target_x = left_x if left_x is not None else (x1 + x2) // 2
                            else:  # 左モード以外はすべて右エッジ
                                _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                                target_x = right_x if right_x is not None else (x1 + x2) // 2
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
            if video_filename:
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
    parser.add_argument("--course", choices=["left", "right"], default="right", help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: right)")
    parser.add_argument("--course-type", choices=["upper", "lower"], default="upper", help="Course type: 'upper' or 'lower' (default: upper)")
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
        course_type=args.course_type,
        model_path=args.model_path,
    )
