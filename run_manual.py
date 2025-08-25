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
 # import select（未使用のため削除）
import socket
import struct
 # import sys（未使用のため削除）
import time

import cv2
import numpy as np
import nnspike
from nnspike.unit import ETRobot, ActionChain, WebcamVideoStream, KeyboardController
from nnspike.constants import CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, ROI_CNN, Mode
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info, get_line_edges_at_y, find_bottle_center, find_blue_target_center, get_virtual_line_target_x

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest for OpenCV processing

# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 45  # Base speed for straight lines (adjust this first)

# Socket connection settings
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to

# Camera/video setup (WebcamVideoStreamで一元管理)
## TIMESTAMP生成はmain関数内に集約

class StateFlags:
    def __init__(self):
        self.yellow_blocked = False
        self.force_sensor_switched = False
    def set_yellow_blocked(self, value: bool):
        self.yellow_blocked = value
    def is_yellow_blocked(self):
        return self.yellow_blocked
    def set_force_sensor_switched(self, value: bool):
        self.force_sensor_switched = value
    def is_force_sensor_switched(self):
        return self.force_sensor_switched

def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="right", course_type="upper"):

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

    # TIMESTAMP生成（video/sensor記録時のみ）
    TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime()) if (record_sensor_data or save_camera_video) else ""
    vs = WebcamVideoStream(src=0, save_video=save_camera_video, timestamp=TIMESTAMP, resolution=(640, 480))
    vs.start()

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()  # Initialize video writer conditionally

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
    state_flags = StateFlags()
    keyboard = KeyboardController()
    pid = PIDController(
        Kp=50,
        Ki=0,
        Kd=5,
        setpoint=0,
        output_limits=(-BASE_SPEED, BASE_SPEED),
    )
    et.set_motor_relative_position(left_positon=0, right_position=0)

    # --- フォースセンサー起動時チェック（初期化後1秒待機して再取得、表示は1回のみ） ---
    try:
        status_init = et.get_spike_status()
        force_val_init = getattr(status_init.sensors, "force", None)
        if force_val_init is None:
            time.sleep(1)
            status_init = et.get_spike_status()
            force_val_init = getattr(status_init.sensors, "force", None)
        if force_val_init is not None:
            print("Force sensor is active. You can press it anytime to switch edge-following mode.\n")
        else:
            print("Force sensor is NOT detected. Please check connection.\n")
    except Exception:
        print("Force sensor check failed. Please check hardware.")

    print()
    print("Press the force sensor or any mode key to start...")
    started = False
    # 有効なモードキーは keycontrol.py の get_mode_from_key で判定
    try:
        while not started and keyboard.running:
            status = et.get_spike_status()
            force_val = getattr(status.sensors, "force", None)
            key = keyboard.get_key()
            if (force_val is not None and force_val > 0) or (key is not None and keyboard.get_mode_from_key(key, Mode.PAUSE) != Mode.PAUSE):
                print("Start!")
                started = True
            if not keyboard.running:
                print("Quitting before start. Exiting...")
                et.stop()
                vs.stop()
                keyboard.cleanup()
                return
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Interrupted before start. Exiting...")
        et.stop()
        vs.stop()
        keyboard.cleanup()
        return

    # 変数初期化
    target_x = None
    offset_y = None
    theta = None
    steering_correction = None
    left_speed = None
    right_speed = None
    mx = None
    my = None
    max_contour = None
    pre_target_x = (x1 + x2) // 2

    try:
        while et.is_running and keyboard.running:
            ret, frame = vs.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            # 毎ループ1回だけstatusを取得
            status = et.get_spike_status()
            left_pos = status.motors["A"].relative_position
            right_pos = status.motors["B"].relative_position

            # --- フォースセンサー押下でエッジ追従モード切替（1回のみ） ---
            if not state_flags.is_force_sensor_switched() and status.sensors.force is not None and status.sensors.force > 0:
                if course == "right":
                    mode = Mode.FOLLOW_RIGHT_EDGE
                    print("\nForce sensor pressed: Switched to FOLLOW_RIGHT_EDGE mode")
                else:
                    mode = Mode.FOLLOW_LEFT_EDGE
                    print("\nForce sensor pressed: Switched to FOLLOW_LEFT_EDGE mode")
                state_flags.set_force_sensor_switched(True)

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(status, mode)

            # Save video frame if enabled (WebcamVideoStreamで管理)
            # vs.write(frame) は未実装または不要

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

            key = keyboard.get_key()
            mode = keyboard.get_mode_from_key(key, mode)
            if not keyboard.running:
                break

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
                case Mode.DOUBLE_LOOP:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.execute_double_loop(frame))
                case Mode.TURN_LEFT_RELATIVE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_left_relative(frame))
                case Mode.TURN_RIGHT_RELATIVE:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_right_relative(frame))
                case Mode.BLUE_BOTTLE_CATCH:
                    target_x, (left_speed, right_speed), mode = unpack_action_result(action_chain.blue_bottle_catch(frame))
                case Mode.FOLLOW_LEFT_EDGE:
                    yellow_cx, _, yellow_pixel_count = find_bottle_center(image=frame, color="yellow")
                    left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
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
                    elif left_x is not None:
                        target_x = left_x
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.FOLLOW_RIGHT_EDGE:
                    yellow_cx, _, yellow_pixel_count = find_bottle_center(image=frame, color="yellow")
                    # シンプルに右モーターの相対位置はright_posを使う
                    _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    # right_posが7000を超えたらDOUBLE_LOOPに切り替え
                    if right_pos is not None and abs(right_pos) >= 7000:
                        mode = Mode.DOUBLE_LOOP
                        target_x = (x1 + x2) // 2
                        print("Switched to DOUBLE_LOOP mode (right_pos >= 7000)")
                    elif yellow_pixel_count > 18000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        mode = Mode.AVOID_OBSTACLE
                        target_x = (x1 + x2) // 2
                        state_flags.set_yellow_blocked(True)
                    elif not state_flags.is_yellow_blocked() and yellow_pixel_count > 3000 and yellow_cx is not None and right_pos is not None and abs(right_pos) < 7000:
                        target_x = yellow_cx[0]  # X座標のみを取得
                    elif right_x is not None:
                        target_x = right_x
                    else:
                        target_x = (x1 + x2) // 2
                case Mode.AVOID_OBSTACLE:
                    #_, (left_speed, right_speed), mode = unpack_action_result(action_chain.avoid_obstacle_relative(frame))
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.avoid_obstacle(frame))
                case Mode.TURN_LEFT:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.turn_left())
                case Mode.SMALL_TURN_LEFT:
                    _, (left_speed, right_speed), mode = unpack_action_result(action_chain.small_turn_left())
                case Mode.HIGH_SPEED:
                    # ハイスピードモード（右エッジ追従＋高速）
                    _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    target_x = right_x if right_x is not None else (x1 + x2) // 2
                    current_base_speed = 255
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
                    red_cx, _, red_pixel_count = find_bottle_center(image=frame, color="red")
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

                theta = calculate_attitude_angle(offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS)
                # HIGH_SPEEDモード以外はBASE_SPEEDを使用
                if mode == Mode.HIGH_SPEED:
                    current_base_speed = 100
                else:
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
            # Clamp speed values to valid range（上限255、0未満は0に）
            left_speed = int(max(0, min(255, left_speed)))
            right_speed = int(max(0, min(255, right_speed)))

            et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        et.stop()
        vs.stop()

        # Close socket connection if it was opened
        if client_socket is not None:
            client_socket.close()

    # 動画保存はWebcamVideoStreamで自動管理される

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

    args = parser.parse_args()
    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print(f"course: {args.course}")
    print(f"Video streaming to host PC: {'Enabled' if args.send_video else 'Disabled'}")
    print("Controls:")
    print("  'a' - Follow left edge")
    print("  'd' - Follow right edge")
    print("  'l' - Turn left")
    print("  'r' - Turn right")
    print("  'k' - Small turn left")
    print("  'j' - Small turn right")
    print("  'f' - Forward")
    print("  'q' - Quit")
    print("Press Ctrl+C to stop")

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
        course=args.course,
        course_type=args.course_type,
    )
