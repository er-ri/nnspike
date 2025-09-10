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
from nnspike.constants import BASE_SPEED, HIGH_SPEED_BASE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, OFFSET_Y, ROI_CNN, Mode, ROI_COLOR
from nnspike.utils import PIDController, SensorRecorder, draw_driving_info, get_line_edges_at_y, find_bottle_center, find_blue_target_center, get_virtual_line_target_x, get_offset_pixels

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest for OpenCV processing

# Socket connection settings
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to

# Camera setup with venue-safe settings (no auto-adjustments)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

def handle_status_and_video(frame, status, mode, target_x, theta, steering_correction, left_speed, right_speed,
                           record_sensor_data, sensor_recorder, send_video_stream, client_socket, 
                           save_camera_video, video_writer):
    """status取得・センサー記録・動画送信処理"""
    if record_sensor_data and sensor_recorder is not None:
        sensor_recorder.log_frame_data(status, mode)
    
    if send_video_stream and client_socket is not None:
        left_pos = status.motors["A"].relative_position
        right_pos = status.motors["B"].relative_position
        mx = target_x - x1 if target_x is not None else None
        my = OFFSET_Y - y1 if target_x is not None else None
        offset_y = y1 + my if my is not None else None
        
        info = {
            "target_x": int(target_x) if isinstance(target_x, (int, float)) and target_x is not None else 0,
            "offset_y": int(offset_y) if isinstance(offset_y, (int, float)) and offset_y is not None else 0,
            "text": {
                "mode": mode.name,
                "left_relative_position": int(left_pos) if left_pos is not None else 0,
                "right_relative_position": int(right_pos) if right_pos is not None else 0,
                "theta_deg": round(math.degrees(theta), 2) if theta is not None else 0,
                "steering_correction": round(steering_correction, 2) if steering_correction is not None else 0,
                "left_speed": int(left_speed) if left_speed is not None else 0,
                "right_speed": int(right_speed) if right_speed is not None else 0,
            }
        }

        max_contour = np.array([[[mx, my]]], dtype=np.int32) if mx is not None and my is not None else None
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        gray = draw_driving_info(gray, info, (x1, y1, x2, y2))
        
        if max_contour is not None and mx is not None and my is not None:
            adjusted_contour = max_contour + np.array([x1, y1])
            cv2.drawContours(gray, [adjusted_contour], -1, (255, 255, 255), 2)
            cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)

        try:
            ret, buffer = cv2.imencode(".jpg", gray)
            img_encoded = buffer.tobytes()
            data = pickle.dumps(img_encoded)
            client_socket.sendall(struct.pack("L", len(data)) + data)
        except Exception as e:
            print(f"Socket error: {e}")
            # エラーでもループを継続
    
    if save_camera_video and video_writer is not None:
        video_writer.write(frame)
    
    return True
    
# StateFlagsクラス（バックアップより）
class StateFlags:

    def __init__(self):
        self._yellow_blocked = False
        self._first_key_used = False
        self._force_sensor_mode_switch_enabled = False

    @property
    def yellow_blocked(self):
        return self._yellow_blocked
    @yellow_blocked.setter
    def yellow_blocked(self, value: bool):
        self._yellow_blocked = value

    @property
    def first_key_used(self):
        return self._first_key_used
    @first_key_used.setter
    def first_key_used(self, value: bool):
        self._first_key_used = value

    @property
    def force_sensor_mode_switch_enabled(self):
        return self._force_sensor_mode_switch_enabled
    @force_sensor_mode_switch_enabled.setter
    def force_sensor_mode_switch_enabled(self, value: bool):
        self._force_sensor_mode_switch_enabled = value

    def enable_force_sensor_mode_switch(self):
        self.force_sensor_mode_switch_enabled = True
    def disable_force_sensor_mode_switch(self):
        self.force_sensor_mode_switch_enabled = False

def wait_for_start(et, keyboard, state_flags, manual_mode=False):
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
    while not started and keyboard.running:
        # 1秒間で最大50回リトライ
        for _ in range(50):
            status = et.get_spike_status()
            force_val = getattr(status.sensors, "force", None)
            key = keyboard.get_key()
            # forceセンサー押下でスタート
            if (force_val is not None and force_val > 0):
                print("Start!")
                started = True
                first_key = "__force__"  # forceセンサーでスタートした場合はダミー値をセット
                break
            # manual_mode時のみ有効なモードキーでスタート
            elif manual_mode and key is not None and keyboard.is_mode_key(key):
                print("Start!")
                first_key = key
                started = True
                break
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

    return first_key

def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="right", course_type="upper", manual_mode=False):
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

    def handle_debug_output(loop_start, loop_end, debug_state, min_interval=0.05):
        """debug出力処理（遅延なし）"""
        loop_elapsed = loop_end - loop_start
        debug_state['counter'] += 1
        # 必要なら経過時間のみ出力（sleepなし）
        elapsed_ms = int(loop_elapsed * 1000)
        # print(f"[DEBUG] loop={debug_state['counter']} time={elapsed_ms}ms")

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
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=CAMERA_FPS,
            frameSize=(CAMERA_WIDTH, CAMERA_HEIGHT),
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
    # まずPIDインスタンスを生成
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
    # その後でActionChainに渡す
    action_chain = ActionChain(et, course, course_type, pid=pid)

    # Initialize robot, keyboard controller
    keyboard = KeyboardController()

    et.set_motor_relative_position(left_positon=0, right_position=0)

    # --- カメラウォームアップ（初回タイムラグ対策） ---
    for _ in range(10):
        cap.read()
    # --- スタート待ち ---
    first_key = wait_for_start(et, keyboard, state_flags, manual_mode=manual_mode)
    if first_key is None:
        cap.release()
        return

    # wait_for_start()の後にmodeの初期値を決定
    if state_flags.force_sensor_mode_switch_enabled:
        mode = Mode.HIGH_SPEED_AVOID
    else:
        mode = Mode.PAUSE

    # --- 変数初期化 ---
    target_x = theta = steering_correction = left_speed = right_speed = None
    center_x = (x1 + x2) // 2  # ROI中心X座標（複数箇所で使用）
    pre_target_x = center_x  # GATE_PASS用の前回値

    # 毎回判定する必要のないフラグを事前計算
    need_status = (record_sensor_data and sensor_recorder is not None) or (send_video_stream and client_socket is not None)

    # debug状態を辞書で管理（エレガントな状態管理）
    debug_state = {
        'counter': 0,
        'last_print': time.time()
    }
    try:
        while et.is_running:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # status取得・センサー記録・動画送信処理
            if need_status:
                status = et.get_spike_status()
                handle_status_and_video(frame, status, mode, target_x, theta, steering_correction, left_speed, right_speed,
                                      record_sensor_data, sensor_recorder, send_video_stream, client_socket,
                                      save_camera_video, video_writer)

            # キー処理とモード切替（統合版）
            if manual_mode and not state_flags.first_key_used:
                key = first_key
                state_flags.first_key_used = True
            else:
                key = keyboard.get_key()
                if not manual_mode and not state_flags.first_key_used:
                    state_flags.first_key_used = True
            
            mode_result, msg = keyboard.get_mode_from_key(key)
            if mode_result == "quit":
                print(msg)
                keyboard.running = False
                break
            elif manual_mode and mode_result is not None:
                mode = mode_result
                print(msg)

            # --- 変数初期化 ---
            target_x = theta = steering_correction = left_speed = right_speed = None
            current_base_speed = BASE_SPEED

            match mode:
                case Mode.DOUBLE_LOOP:
                    # 設定を元に戻す
                    # pid.Kp = 50
                    # pid.Ki = 0
                    # pid.Kd = 5
                    # pid.output_limits = (-BASE_SPEED, BASE_SPEED)
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.execute_double_loop(frame))
                case Mode.TURN_LEFT_RELATIVE:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.turn_left_relative(frame))
                case Mode.TURN_RIGHT_RELATIVE:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.turn_right_relative(frame))
                case Mode.BLUE_BOTTLE_CATCH:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.blue_bottle_catch(frame))
                case Mode.FOLLOW_LEFT_EDGE:
                    # 単純な左エッジトレースのみ（course='left'を明示的に指定）
                    target_x = action_chain.get_target_x_by_course(frame, OFFSET_Y, 'left')
                case Mode.FOLLOW_RIGHT_EDGE:
                    # 単純な右エッジトレースのみ（course='right'を明示的に指定）
                    target_x = action_chain.get_target_x_by_course(frame, OFFSET_Y, 'right')
                case Mode.AVOID_OBSTACLE:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.avoid_obstacle_relative(frame))
                case Mode.SMALL_TURN_LEFT:
                    _, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.small_turn_left())
                case Mode.HIGH_SPEED_AVOID:
                    # pid.Kp = 1.0  # 🏆 36回段階テスト結果：バランス0.624で最適（効率0.543 + 制御力0.590）
                    # pid.Ki = 0
                    # pid.Kd = 0.3  # 安定した微分制御で自然安定性向上
                    # pid.output_limits = (-4, 4)  # テスト結果による最適制御範囲
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
                        # debug出力処理
                        loop_end = time.time()
                        handle_debug_output(loop_start, loop_end, debug_state)
                        continue  # 以降のset_motor_speed処理をスキップ
                case Mode.CARRY_BOTTLE2:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.carry_bottle2_relative(frame))
                case Mode.BACK_AND_TURN2:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.back_and_turn2_relative(frame))
                    # 後退フェーズ（両輪とも正方向速度）の場合はset_motor_backward_speedを使う
                    if left_speed == BASE_SPEED and right_speed == BASE_SPEED:
                        et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                        # debug出力処理
                        loop_end = time.time()
                        handle_debug_output(loop_start, loop_end, debug_state)
                        continue  # 以降のset_motor_speed処理をスキップ
                case Mode.HEAD_GOAL:
                    target_x, (left_speed, right_speed, _), mode = unpack_action_result(action_chain.heading_goal_relative(frame))
                case Mode.FORWARD:
                    # 赤色重心に向かって進む（find_bottle_center使用）。イエロー・ブルー検知は行わない。
                    red_cx, _, red_pixel_count = find_bottle_center(frame, "red", roi=ROI_COLOR)
                    if red_pixel_count > 3000:
                        if red_cx is not None:
                            target_x = red_cx[0]  # X座標のみを取得
                        else:
                            target_x = center_x
                    else:
                        target_x = center_x
                case Mode.GATE_PASS:
                    # ゲートを潜る: 仮想ラインエッジを使う
                    temp_x = get_virtual_line_target_x(frame, previous_center_x=pre_target_x)
                    if temp_x is not None:
                        target_x = temp_x
                        pre_target_x = temp_x
                    elif pre_target_x is not None:
                        target_x = pre_target_x
                    else:
                        target_x = center_x
                        pre_target_x = target_x
                case Mode.EYE_BLUE:
                    # ブルーアイズ（青重心）に向かう: find_blue_target_centerを使用
                    center, area, blue_pixel_count = find_blue_target_center(frame)
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = center_x
                case Mode.BACKWARD:
                    # set_motor_backward_speedで後退（左右は入れ替えない）
                    left_speed = BASE_SPEED
                    right_speed = BASE_SPEED
                    et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                    # debug出力処理
                    loop_end = time.time()
                    handle_debug_output(loop_start, loop_end, debug_state)
                    continue  # 以降のset_motor_speed処理をスキップ
                case Mode.PAUSE:
                    left_speed, right_speed = 0, 0
                case _:
                    # Default to center if invalid edge specified
                    target_x = center_x

            # PID制御処理（target_xが設定されている場合）
            if target_x is not None:
                offset_pixels = get_offset_pixels(target_x, ROI_CNN)
                theta = math.atan2(offset_pixels, CAMERA_WIDTH)  # 簡素化: 直接計算
                steering_correction = pid.update(theta)
                left_speed = current_base_speed - steering_correction
                right_speed = current_base_speed + steering_correction

            # left_speed/right_speedがNoneなら0に（int()前に必ず実施）
            left_speed = 0 if left_speed is None else left_speed
            right_speed = 0 if right_speed is None else right_speed
            # Clamp speed values to valid range（上限255、0未満は0に）
            left_speed = int(max(0, min(255, left_speed)))
            right_speed = int(max(0, min(255, right_speed)))

            # ハイスピードアボイド時のみパワー差を4以内に制限
            if mode == Mode.HIGH_SPEED_AVOID:
                diff = left_speed - right_speed
                if abs(diff) > 4:
                    if diff > 0:
                        left_speed = right_speed + 4
                    else:
                        right_speed = left_speed + 4

            # Temporarily set Heading Gate mode
            if mode == Mode.PAUSE:
                et.brake()
            else:
                et.set_motor_forward_speed(
                    left_speed=left_speed,
                    right_speed=right_speed,
                )

            # --- ループ周期制限とdebug出力（最後） ---
            loop_end = time.time()
            handle_debug_output(loop_start, loop_end, debug_state)

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

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
        course=args.course,
        course_type=args.course_type,
        manual_mode=args.manual,
    )
