#!/usr/bin/env python3
"""
OpenCV-Based Line Following Robot Control

このスクリプトはOpenCVによる画像処理でライン追従ロボットを制御します。
get_line_edges_at_y関数でライン重心を検出し、PID制御で追従します。

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
import socket
import struct
import sys
import time

from nnspike.unit import ETRobot, ActionChain, KeyboardController

import cv2
import numpy as np
from nnspike.constants import BASE_SPEED, HIGH_SPEED_BASE, CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, ROI_CNN, Mode, ROI_COLOER
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info, get_line_edges_at_y, find_bottle_center, find_blue_target_center, get_virtual_line_target_x, get_offset_pixels

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest for OpenCV processing

# Socket connection settings
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
# StateFlagsクラス（バックアップより）
class StateFlags:

    def __init__(self):
        self.yellow_blocked = False
        self.force_sensor_switched = False
        self.first_key_used = False
        self.force_sensor_mode_switch_enabled = False
    def enable_force_sensor_mode_switch(self):
        self.force_sensor_mode_switch_enabled = True
    def disable_force_sensor_mode_switch(self):
        self.force_sensor_mode_switch_enabled = False
    def is_force_sensor_mode_switch_enabled(self):
        return self.force_sensor_mode_switch_enabled
    def set_first_key_used(self, value: bool):
        self.first_key_used = value
    def is_first_key_used(self):
        return self.first_key_used
    def set_yellow_blocked(self, value: bool):
        self.yellow_blocked = value
    def is_yellow_blocked(self):
        return self.yellow_blocked
    def set_force_sensor_switched(self, value: bool):
        self.force_sensor_switched = value
    def is_force_sensor_switched(self):
        return self.force_sensor_switched

def wait_for_start(et, keyboard, state_flags):
    """
    起動時にフォースセンサーの接続状態を確認し、
    フォースセンサーまたは有効なモードキーでロボットをスタートさせる。
    終了時は first_key（最初に押されたキーまたはforceセンサー）を返す。
    """
    try:
        status_init = et.get_spike_status()
        force_val_init = getattr(status_init.sensors, "force", None)
        if force_val_init is None:
            time.sleep(1)
            status_init = et.get_spike_status()
            force_val_init = getattr(status_init.sensors, "force", None)
        if force_val_init is not None:
            print("Force sensor is active. You can press it anytime to switch edge-following mode.")
            print("\r", end="")
            sys.stdout.flush()
        else:
            print("Force sensor is NOT detected. Please check connection.")
            print("\r", end="")
            sys.stdout.flush()
    except Exception:
        print("Force sensor check failed. Please check hardware.")

    print("Press the force sensor or any mode key to start...")
    started = False
    first_key = None
    try:
        while not started and keyboard.running:
            status = et.get_spike_status()
            force_val = getattr(status.sensors, "force", None)
            key = keyboard.get_key()
            # forceセンサー押下でスタート
            if (force_val is not None and force_val > 0):
                print("Start!")
                started = True
                first_key = "__force__"  # forceセンサーでスタートした場合はダミー値をセット
            # 有効なモードキーでスタート
            elif key is not None and keyboard.is_mode_key(key):
                print("Start!")
                first_key = key
                started = True
            if not keyboard.running:
                print("Quitting before start. Exiting...")
                et.stop()
                keyboard.cleanup()
                return None
            time.sleep(0.02)
        # スタート決定後に一度だけモード切替有効化/無効化を判定
        if started:
            if first_key == "__force__":
                state_flags.enable_force_sensor_mode_switch()
            else:
                state_flags.disable_force_sensor_mode_switch()
    # KeyboardInterrupt例外処理を削除
    return first_key

def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="right", course_type="upper", manual_mode=False):
    def reset_frame_vars():
        return None, None, None, None, None, None, None, None, None

    def unpack_action_result(result, default_mode=Mode.PAUSE):
        # Noneや不正な戻り値も吸収して安全にアンパック
        if result is None:
            return None, (0, 0, 0), default_mode
        if len(result) == 3:
            target_x, speeds, mode = result
            # speedsが2要素なら3要素化（currentはBASE_SPEED）
            if speeds is None:
                speeds = (0, 0, 0)
            elif len(speeds) == 2:
                speeds = (speeds[0], speeds[1], BASE_SPEED)
            elif len(speeds) == 3:
                left, right, current = speeds
                if current is None or current == 0:
                    current = BASE_SPEED
                speeds = (left, right, current)
            if mode is None:
                mode = default_mode
            return target_x, speeds, mode
        return None, (0, 0, 0), default_mode

    state_flags = StateFlags()
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

    et.set_motor_relative_position(left_positon=0, right_position=0)

    first_key = wait_for_start(et, keyboard, state_flags)
    if first_key is None:
        cap.release()
        return

    # --- ここから未定義エラー防止のための宣言（関数スコープ） ---
    target_x, offset_y, theta, steering_correction, left_speed, right_speed, mx, my, max_contour = reset_frame_vars()
    pre_target_x = (x1 + x2) // 2  # GATE_PASS用の前回値
    # --- ここまで ---

    try:
        while et.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # 毎ループ1回だけstatusを取得
            status = et.get_spike_status()
            left_raw = status.motors["A"].relative_position
            right_raw = status.motors["B"].relative_position
            left_pos = left_raw[0] if isinstance(left_raw, tuple) else left_raw
            right_pos = right_raw[0] if isinstance(right_raw, tuple) else right_raw

            # forceセンサーでスタートした場合のみ、forceセンサーによるモード切替を有効化（フラグ廃止のため直接判定）
            if state_flags.is_force_sensor_mode_switch_enabled():
                if not state_flags.is_force_sensor_switched() and status.sensors.force is not None and status.sensors.force > 0:
                    mode = Mode.HIGH_SPEED_AVOID
                    print("\nForce sensor pressed: Switched to HIGH_SPEED_AVOID mode")
                    state_flags.set_force_sensor_switched(True)

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(status, mode)

            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)

            # Send video stream and driving info if enabled (must be after frame, target_x, etc. are set)
            if send_video_stream and client_socket is not None:
                info = dict()
                mx = target_x - x1 if target_x is not None else None  # Relative to ROI
                my = OFFSET_Y - y1 if target_x is not None else None  # Relative to ROI
                offset_y = y1 + my if my is not None else None  # offset_yを明示的にセット
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

                # Create visualization frame
                max_contour = np.array([[[mx, my]]], dtype=np.int32) if mx is not None and my is not None else None
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

            # manual_mode時はfirst_keyを優先して使い、消費後はget_key()に切り替える
            if manual_mode:
                if not state_flags.is_first_key_used():
                    key = first_key
                    state_flags.set_first_key_used(True)
                else:
                    key = keyboard.get_key()
                mode_result, msg = keyboard.get_mode_from_key(key)
                if mode_result == "quit":
                    print(msg)
                    keyboard.running = False
                    break
                elif mode_result is not None:
                    mode = mode_result
                    print(msg)
            else:
                if not state_flags.is_first_key_used():
                    state_flags.set_first_key_used(True)

            # --- ここから未定義エラー防止のための初期化 ---
            target_x, offset_y, theta, steering_correction, left_speed, right_speed, mx, my, max_contour = reset_frame_vars()
            # --- ここまで ---
            # --- ここから速度基準値初期化 ---
            current_base_speed = BASE_SPEED
            # --- ここまで速度基準値初期化 ---

            match mode:
                case Mode.DOUBLE_LOOP:
                    # 設定を元に戻す
                    pid.Kp = 50
                    pid.Ki = 0
                    pid.Kd = 5
                    pid.output_limits = (-BASE_SPEED, BASE_SPEED)
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.execute_double_loop(frame))
                case Mode.TURN_LEFT_RELATIVE:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.turn_left_relative(frame))
                case Mode.TURN_RIGHT_RELATIVE:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.turn_right_relative(frame))
                case Mode.BLUE_BOTTLE_CATCH:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.blue_bottle_catch(frame))
                case Mode.FOLLOW_LEFT_EDGE:
                    yellow_cx, _, yellow_pixel_count = find_bottle_center(frame, "yellow", roi=ROI_COLOER)
                    # left_posが7000を超えたらNVIDIA_FOLLOWに切り替え
                    if left_pos is not None and abs(left_pos) >= 7000:
                        mode = Mode.DOUBLE_LOOP
                        target_x = (x1 + x2) // 2
                        print("Switched to DOUBLE_LOOP mode (left_pos >= 7000)")
                    elif yellow_pixel_count > 18000 and yellow_cx is not None and left_pos is not None and abs(left_pos) < 7000:
                        mode = Mode.AVOID_OBSTACLE
                        target_x = (x1 + x2) // 2
                        state_flags.set_yellow_blocked(True)
                    elif not state_flags.is_yellow_blocked() and yellow_pixel_count > 3000 and yellow_cx is not None and left_pos is not None and abs(left_pos) < 7000:
                        target_x = yellow_cx[0]  # X座標のみを取得
                    else:
                        target_x = action_chain.get_target_x_by_course(frame, OFFSET_Y, course)
                case Mode.FOLLOW_RIGHT_EDGE:
                    yellow_cx, _, yellow_pixel_count = find_bottle_center(frame, "yellow", roi=ROI_COLOER)
                    # シンプルに右モーターの相対位置はright_posを使う
                    # right_posが7000を超えたらDOUBLE_LOOPに切り替え
                    if right_pos is not None and abs(right_pos) >= 7000:
                        mode = Mode.DOUBLE_LOOP
                        target_x = (x1 + x2) // 2
                        print("Switched to DOUBLE_LOOP mode (left_pos >= 7000)")
                    elif yellow_pixel_count > 18000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        mode = Mode.AVOID_OBSTACLE
                        target_x = (x1 + x2) // 2
                        state_flags.set_yellow_blocked(True)
                    elif not state_flags.is_yellow_blocked() and yellow_pixel_count > 3000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        target_x = yellow_cx[0]  # X座標のみを取得
                    else:
                        target_x = action_chain.get_target_x_by_course(frame, OFFSET_Y, course)
                case Mode.AVOID_OBSTACLE:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.avoid_obstacle_relative(frame))
                case Mode.SMALL_TURN_LEFT:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.small_turn_left())
                case Mode.HIGH_SPEED_AVOID:
                    pid.Kp = 5
                    pid.Ki = 0
                    pid.Kd = 5
                    pid.output_limits = (-8, 8)  # さらに狭く
                    target_x, (left_speed, right_speed, current_base_speed), mode = unpack_action_result(action_chain.high_speed_avoid(frame))
                case Mode.HIGH_SPEED:
                    # ハイスピードモード（右エッジ追従＋高速）
                    target_x = action_chain.get_target_x_by_course(frame, OFFSET_Y, course)
                    current_base_speed = HIGH_SPEED_BASE
                case Mode.SMALL_TURN_RIGHT:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.small_turn_right())
                case Mode.CARRY_BOTTLE1:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.carry_bottle1_relative(frame))
                case Mode.BACK_AND_TURN1:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.back_and_turn1_relative(frame))
                    # 後退フェーズ（両輪とも正方向速度）の場合はset_motor_backward_speedを使う
                    if left_speed == BASE_SPEED and right_speed == BASE_SPEED:
                        et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                        continue  # 以降のset_motor_speed処理をスキップ
                case Mode.CARRY_BOTTLE2:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.carry_bottle2_relative(frame))
                case Mode.BACK_AND_TURN2:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.back_and_turn2_relative(frame))
                    # 後退フェーズ（両輪とも正方向速度）の場合はset_motor_backward_speedを使う
                    if left_speed == BASE_SPEED and right_speed == BASE_SPEED:
                        et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                        continue  # 以降のset_motor_speed処理をスキップ
                case Mode.HEAD_GOAL:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.heading_goal_relative(frame))
                case Mode.FORWARD:
                    # 赤色重心に向かって進む（find_bottle_center使用）。イエロー・ブルー検知は行わない。
                    red_cx, _, red_pixel_count = find_bottle_center(frame, "red", roi=ROI_COLOER)
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
                # NVIDIA_FOLLOWモード分岐を一時的に削除
                case _:
                    # Default to center if invalid edge specified
                    target_x = (x1 + x2) // 2

            # target_xがNoneのときのみ可視化・送信用変数をリセット
            if target_x is None:
                theta = steering_correction = None
            else:
                offset_pixels = get_offset_pixels(target_x, ROI_CNN)
                theta = calculate_attitude_angle(offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS)
                steering_correction = pid.update(theta)
                left_speed = current_base_speed - steering_correction
                right_speed = current_base_speed + steering_correction

            # left_speed/right_speedがNoneなら0に（int()前に必ず実施）
            left_speed = 0 if left_speed is None else left_speed
            right_speed = 0 if right_speed is None else right_speed
            # Clamp speed values to valid range（上限255、0未満は0に）
            left_speed = int(max(0, min(255, left_speed)))
            right_speed = int(max(0, min(255, right_speed)))

            # Temporarily set Heading Gate mode
            if mode == Mode.PAUSE:
                et.brake()
            else:
                et.set_motor_forward_speed(
                    left_speed=left_speed,
                    right_speed=right_speed,
                )

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
    parser.add_argument("--manual", action="store_true", help="Enable manual key input control mode")
    args = parser.parse_args()
    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print(f"course: {args.course}")
    print(f"Video streaming to host PC: {'Enabled' if args.send_video else 'Disabled'}")
    print("Controls:")
    print("  'a' - Follow left edge")
    print("  'd' - Follow right edge")
    print("  'f' - Forward")
    print("  'b' - Backward")
    print("  'q' - Quit")
    print("Press Ctrl+C to stop")

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
        course=args.course,
        course_type=args.course_type,
        manual_mode=args.manual,
    )
