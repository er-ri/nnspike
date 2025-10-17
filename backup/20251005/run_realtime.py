#!/usr/bin/env python3

import argparse
from email.mime import base
import sys
import time
import math

from nnspike.unit import Video, ETRobot, KeyboardController
from nnspike.unit.fast_lap_chain import FastLapChain
from nnspike.unit.action_chain import ActionChain

import cv2
import numpy as np
from nnspike.constants import BASE_SPEED, HIGH_SPEED_BASE, CAMERA_WIDTH, CAMERA_HEIGHT, OFFSET_Y, ROI_CNN, Mode, ROI_COLOR
from nnspike.utils import PIDController, SensorRecorder, draw_driving_info, get_line_edges_at_y, find_bottle_center, find_blue_target_center, get_virtual_line_target_x, get_offset_pixels
import threading

def handle_status_and_video(frame, status, mode, left_speed, right_speed,
                           record_sensor_data, sensor_recorder, save_camera_video, video_writer):
    """status取得・センサー記録・動画送信処理"""
    if record_sensor_data and sensor_recorder is not None:
        sensor_recorder.log_frame_data(status, mode, left_speed, right_speed)
    if save_camera_video and video_writer is not None:
        video_writer.write(frame)
    
    return True
    
# StateFlagsクラス（バックアップより）
class StateFlags:

    def __init__(self):
        self._first_key_used = False
        self._force_sensor_mode_switch_enabled = False

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
    force_baseline = None
    try:
        status_init = et.get_spike_status()
        force_baseline = getattr(status_init.sensors, "force", None)
        if force_baseline is None:
            time.sleep(1)
            status_init = et.get_spike_status()
            force_baseline = getattr(status_init.sensors, "force", None)
        if force_baseline is not None:
            print(f"Force sensor is active. Baseline: {force_baseline}. Press to start.")
            print("\r", end="")
            sys.stdout.flush()
        else:
            print("Force sensor is NOT detected. Please check connection.")
            print("\r", end="")
            sys.stdout.flush()
    except Exception:
        print("Force sensor check failed. Please check hardware.")

    print("Press the force sensor (change >100) or any mode key to start...")
    started = False
    first_key = None
    while not started and keyboard.running:
        # 1秒間で最大50回リトライ
        for _ in range(50):
            status = et.get_spike_status()
            force_val = getattr(status.sensors, "force", None)
            key = keyboard.get_key()
            # forceセンサー押下でスタート（初期値から絶対値100以上変動）
            if (force_val is not None and force_baseline is not None and abs(force_val - force_baseline) >= 100):
                print(f"Start! (force changed: {force_baseline} -> {force_val})")
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

def main(record_sensor_data=False, save_camera_video=False, course="right", course_type="upper", manual_mode=False, use_video=False):
    def unpack_action_result(result, default_mode=Mode.PAUSE):
        # Noneや不正な戻り値も吸収して安全にアンパック
        if result is None:
            return (0, 0), default_mode
        if len(result) == 2:
            speeds, mode = result
            if speeds is None:
                speeds = (0, 0)
            elif len(speeds) == 2:
                left, right = speeds
                left = 0 if left is None else left
                right = 0 if right is None else right
                speeds = (left, right)
            if mode is None:
                mode = default_mode
            return speeds, mode
        return (0, 0), default_mode

    def handle_debug_output(loop_start, loop_end, debug_state, min_interval=0.017):
        debug_state['counter'] += 1
        # --- min_interval引数で周期調整＋デバッグ出力 ---
        dt = loop_end - loop_start
        sleep_sec = min_interval - dt if dt < min_interval else 0
        # sleep_ms = sleep_sec * 1000  # デバッグ用途のみなのでコメントアウト
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        # total_ms = (time.time() - loop_start) * 1000  # デバッグ用途のみなのでコメントアウト
        # print(f"[DEBUG] dt={dt*1000:.2f}ms, sleep={sleep_ms:.2f}ms, total={total_ms:.2f}ms")  # デバッグ出力のみコメントアウト

    state_flags = StateFlags()
    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime()) if (record_sensor_data or save_camera_video) else None

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()

    video_writer = None
    video_filename = None
    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(CAMERA_WIDTH, CAMERA_HEIGHT),
        )

    # Initialize edge following preference based on the course parameter
    et = ETRobot()

    pid = PIDController(
        Kp=50,
        Ki=0,
        Kd=5,
        setpoint=0,
        output_limits=(-BASE_SPEED, BASE_SPEED),
    )

    # Initialize robot, keyboard controller
    keyboard = KeyboardController()

    et.set_motor_relative_position(left_position=0, right_position=0)

    # FastLapChainインスタンス生成
    fast_lap_chain = FastLapChain(et, course)
    action_chain = ActionChain(et, course, course_type, pid=pid)

    # --- 変数初期化 ---
    left_speed = None
    right_speed = None
    dummy_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    # 毎回判定する必要のないフラグを事前計算
    need_status = (record_sensor_data and sensor_recorder is not None)

    # カメラ起動条件をuse_cameraまたはuse_videoどちらかTrueで判定
    video = None
    if use_video:
        video = Video()
        video.warmup()
        actual_fps = video.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera actual FPS: {actual_fps}")
    # --- スタート待ち ---
    first_key = wait_for_start(et, keyboard, state_flags, manual_mode=manual_mode)
    if first_key is None:
        if video is not None:
            video.release()
        return

    # wait_for_start()の後にmodeの初期値を決定
    if state_flags.force_sensor_mode_switch_enabled:
        mode = Mode.FAST_LAP
    else:
        mode = Mode.PAUSE

    # debug状態を辞書で管理（エレガントな状態管理）
    debug_state = {
        'counter': 0,
        'last_print': time.time()
    }

    try:
        while et.is_running:
            loop_start = time.time()
            if video is not None:
                frame = video.get_frame()
                if save_camera_video and video_writer is not None and frame is not None and isinstance(frame, np.ndarray):
                    video_writer.write(frame)
            else:
                frame = dummy_frame

            # status取得・センサー記録（メインスレッドで直接処理）
            if need_status:
                status = et.get_spike_status()
                safe_left_speed = left_speed if left_speed is not None else 0
                safe_right_speed = right_speed if right_speed is not None else 0
                if sensor_recorder is not None:
                    sensor_recorder.log_frame_data(status, mode, safe_left_speed, safe_right_speed)

            # キー処理とモード切替（統合版）
            if manual_mode and not state_flags.first_key_used:
                key = first_key
                state_flags.first_key_used = True
            else:
                key = keyboard.get_key()
                if not manual_mode and not state_flags.first_key_used:
                    state_flags.first_key_used = True

            result = keyboard.get_mode_from_key(key)
            if result is not None:
                mode_result, msg = result
            else:
                mode_result, msg = None, None
            if mode_result == "quit":
                print(msg)
                keyboard.running = False
                break
            elif manual_mode and mode_result is not None:
                mode = mode_result
                print(msg)

            # モード分岐（match-case構文）
            match mode:
                case Mode.FAST_LAP:
                    (left_speed, right_speed), mode = unpack_action_result(fast_lap_chain.fast_lap(frame))
                    et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.FORWARD:
                    left_speed = BASE_SPEED
                    right_speed = BASE_SPEED
                    et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.BACKWARD:
                    left_speed = BASE_SPEED
                    right_speed = BASE_SPEED
                    et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.TURN_LEFT_YAW:
                    (left_speed, right_speed), mode = unpack_action_result(fast_lap_chain.turn_left_yaw(frame))
                    et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.TURN_RIGHT_YAW:
                    (left_speed, right_speed), mode = unpack_action_result(fast_lap_chain.turn_right_yaw(frame))
                    et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.DOUBLE_LOOP:
                    (left_speed, right_speed), mode = unpack_action_result(action_chain.execute_double_loop(frame))
                    et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.CARRY_BOTTLE1:
                    (left_speed, right_speed), mode = unpack_action_result(action_chain.carry_bottle1_relative(frame))
                    et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.BACK_AND_TURN1:
                    (left_speed, right_speed), mode = unpack_action_result(action_chain.back_and_turn1_relative(frame))
                    if left_speed == BASE_SPEED and right_speed == BASE_SPEED:
                        et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                    else:
                        et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.CARRY_BOTTLE2:
                    (left_speed, right_speed), mode = unpack_action_result(action_chain.carry_bottle2_relative(frame))
                    et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.BACK_AND_TURN2:
                    (left_speed, right_speed), mode = unpack_action_result(action_chain.back_and_turn2_relative(frame))
                    if left_speed == BASE_SPEED and right_speed == BASE_SPEED:
                        et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
                    else:
                        et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.HEAD_GOAL:
                    (left_speed, right_speed), mode = unpack_action_result(action_chain.heading_goal_relative(frame))
                    et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.EYE_BLUE:
                    (left_speed, right_speed), mode = unpack_action_result(action_chain.eye_blue(frame))
                    et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)
                case Mode.PAUSE:
                    left_speed, right_speed = 0, 0
                    et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)

            # --- ループ周期・フレーム取得周期デバッグ出力（詳細&sleep調整） ---
            loop_end = time.time()
            handle_debug_output(loop_start, loop_end, debug_state, min_interval=0.017)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        et.stop()
        if video is not None:
            video.release()

        # 録画クリーンアップ
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
    parser.add_argument("--course", choices=["left", "right"], default="right", help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: right)")
    parser.add_argument("--course-type", choices=["upper", "lower"], default="upper", help="Course type: 'upper' or 'lower' (default: upper)")
    parser.add_argument("--manual", action="store_true", help="Enable manual key input control mode")
    parser.add_argument("--use-video", dest="use_video", action="store_true", default=True, help="Enable camera at startup (default: True)")
    parser.add_argument("--no-use-video", dest="use_video", action="store_false", help="Disable camera at startup")
    args = parser.parse_args()
    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print(f"course: {args.course}")
    print("Controls:")
    print("  'a' - Follow left edge")
    print("  'd' - Follow right edge")
    print("  'f' - Forward")
    print("  'b' - Backward")
    print("  'q' - Quit")

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        course=args.course,
        course_type=args.course_type,
        manual_mode=args.manual,
        use_video=args.use_video,
    )
