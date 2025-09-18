#!/usr/bin/env python3

import argparse
from email.mime import base
import sys
import time

from nnspike.unit import ETRobot, KeyboardController

import cv2
import numpy as np
from nnspike.constants import BASE_SPEED, HIGH_SPEED_BASE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, OFFSET_Y, ROI_CNN, Mode, ROI_COLOR
from nnspike.utils import PIDController, SensorRecorder, draw_driving_info, get_line_edges_at_y, find_bottle_center, find_blue_target_center, get_virtual_line_target_x, get_offset_pixels
import threading

class Video:
    def __init__(self, mode='realtime'):
        """
        mode: 'realtime'（遅延最小化・最新フレーム優先） or 'continuous'（フレーム連続性重視）
        fps/buffer_sizeは内部定数で管理
        """
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self._FPS_REALTIME = 30
        self._FPS_CONTINUOUS = 25
        self.mode = mode
        self._BUFFER_REALTIME = 1
        self._BUFFER_CONTINUOUS = 4
        self.fps = self._FPS_REALTIME if mode == 'realtime' else self._FPS_CONTINUOUS
        buffer_size = self._BUFFER_REALTIME if mode == 'realtime' else self._BUFFER_CONTINUOUS
        self.cap = cv2.VideoCapture(0)  # USBカメラ前提で0固定
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        # continuous用フレームキュー
        self._frame_queue = []
        self.last_update_time = None
        self.update_count = 0
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def set_mode(self, mode):
        """
        Videoモード（'realtime' or 'continuous'）を切り替え、バッファサイズ・fpsも自動調整
        """
        self.mode = mode
        if mode == 'realtime':
            self.fps = self._FPS_REALTIME
            buffer_size = self._BUFFER_REALTIME
        else:
            self.fps = self._FPS_CONTINUOUS
            buffer_size = self._BUFFER_CONTINUOUS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    def warmup(self, count=10):
        """
        カメラウォームアップ用: 指定回数read()を呼ぶ
        """
        for _ in range(count):
            self.read()

    def _update(self):
        prev_time = None
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
                now = time.time()
                self.last_update_time = now
                self.update_count += 1
                # continuousモード時のみキューに追加
                if self.mode == 'continuous':
                    if frame is not None:
                        self._frame_queue.append((ret, frame.copy()))
                        # バッファサイズ超過時は古いものから捨てる
                        while len(self._frame_queue) > self._BUFFER_CONTINUOUS:
                            self._frame_queue.pop(0)
                else:
                    self._frame_queue.clear()
                # デバッグ出力: 差分msのみ
                if prev_time is not None:
                    diff_ms = int((now - prev_time) * 1000)
                else:
                    diff_ms = 0
                # デバッグ出力削除
                prev_time = now
            time.sleep(0.001)  # 軽いウェイトでCPU負荷抑制

    def read(self):
        """
        - realtime: 最新フレームのみ返す
        - continuous: 未読フレームを順次返す（最大4枚まで蓄積、乖離防止）
        """
        with self.lock:
            if self.mode == 'continuous' and self._frame_queue:
                ret, frame = self._frame_queue.pop(0)
                return ret, frame.copy() if frame is not None else (False, None)
            else:
                return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.thread.join()
        # self.cap.release()

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

def main(record_sensor_data=False, save_camera_video=False, course="right", course_type="upper", manual_mode=False):

    def handle_debug_output(loop_start, loop_end, debug_state, min_interval=0.03):
        """debug出力処理＋ループ周期50msに制御"""
        loop_elapsed = loop_end - loop_start
        debug_state['counter'] += 1
        # 前回ループ終了時刻との差分のみ表示
        prev_end = debug_state.get('last_print', None)
        now = loop_end
        if prev_end is not None:
            diff_ms = int((now - prev_end) * 1000)
        else:
            diff_ms = 0
        debug_state['last_print'] = now
        # min_interval周期制御（1ループmin_interval未満ならsleepで調整）
        sleep_time = min_interval - loop_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    # デバッグ出力削除

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

    # Initialize edge following preference based on the course parameter
    et = ETRobot()

    # Initialize robot, keyboard controller
    keyboard = KeyboardController()

    et.set_motor_relative_position(left_positon=0, right_position=0)

    # --- Videoクラスでカメラ起動・ウォームアップ（カメラ停止中） ---
    # video = Video()
    # video.warmup()
    # --- スタート待ち ---
    first_key = wait_for_start(et, keyboard, state_flags, manual_mode=manual_mode)
    if first_key is None:
        # if 'video' is used, release it
        # video.release()
        return

    # wait_for_start()の後にmodeの初期値を決定
    if state_flags.force_sensor_mode_switch_enabled:
        mode = Mode.TEST
    else:
        mode = Mode.PAUSE

    # --- 変数初期化 ---
    left_speed = None
    right_speed = None
    turn_left_started = False
    turn_right_started = False
    yaw_start_left = None
    yaw_start_right = None

    # 毎回判定する必要のないフラグを事前計算
    need_status = (record_sensor_data and sensor_recorder is not None)

    # debug状態を辞書で管理（エレガントな状態管理）
    debug_state = {
        'counter': 0,
        'last_print': time.time()
    }
    try:
        turn_left_started = False
        yaw_start = None
        while et.is_running:
            loop_start = time.time()
            # カメラ停止中のためframe取得・処理は省略
            # ret, frame = video.read()
            # if not ret:
            #     print("Can't receive frame (stream end?). Exiting ...")
            #     break

            # status取得・センサー記録・動画送信処理
            if need_status:
                status = et.get_spike_status()
                handle_status_and_video(None, status, mode, left_speed, right_speed,
                                      record_sensor_data, sensor_recorder, save_camera_video, video_writer)

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

            # モード分岐
            if mode == Mode.FORWARD:
                # シンプルに直進のみ
                left_speed = BASE_SPEED
                right_speed = BASE_SPEED
                et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
            elif mode == Mode.BACKWARD:
                # set_motor_backward_speedで後退（左右は入れ替えない）
                left_speed = BASE_SPEED
                right_speed = BASE_SPEED
                et.set_motor_backward_speed(left_speed=left_speed, right_speed=right_speed)
            elif mode == Mode.TURN_LEFT:
                # --- ヨー角判定ロジック（元の形式） ---
                status = et.get_spike_status()
                print(f"DEBUG yaw_pitch_roll (LEFT): {status.sensors.yaw_pitch_roll}")
                yaw = status.sensors.yaw_pitch_roll.get('yaw') if status.sensors.yaw_pitch_roll else 0.0
                if not turn_left_started:
                    yaw_start_left = yaw
                    turn_left_started = True
                left_speed = 0
                right_speed = BASE_SPEED
                et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)
                print(f"[TURN_LEFT] yaw={yaw:.2f}, yaw_start={yaw_start_left:.2f}, diff={yaw - (yaw_start_left if yaw_start_left is not None else 0.0):.2f}")
                if (yaw - (yaw_start_left if yaw_start_left is not None else 0.0)) <= -90.0:
                    print(f"[TURN_LEFT] reached -90 deg and stopped | yaw={yaw:.2f}")
                    et.set_motor_forward_speed(left_speed=0, right_speed=0)
                    mode = Mode.PAUSE
                    turn_left_started = False
            elif mode == Mode.TURN_RIGHT:
                # --- ヨー角判定ロジック（元の形式） ---
                status = et.get_spike_status()
                print(f"DEBUG yaw_pitch_roll (RIGHT): {status.sensors.yaw_pitch_roll}")
                yaw = status.sensors.yaw_pitch_roll.get('yaw') if status.sensors.yaw_pitch_roll else 0.0
                if not turn_right_started:
                    yaw_start_right = yaw
                    turn_right_started = True
                left_speed = BASE_SPEED
                right_speed = 0
                et.set_motor_speed(left_speed=left_speed, right_speed=right_speed)
                print(f"[TURN_RIGHT] yaw={yaw:.2f}, yaw_start={yaw_start_right:.2f}, diff={yaw - (yaw_start_right if yaw_start_right is not None else 0.0):.2f}")
                if (yaw - (yaw_start_right if yaw_start_right is not None else 0.0)) >= 90.0:
                    print(f"[TURN_RIGHT] reached +90 deg and stopped | yaw={yaw:.2f}")
                    et.set_motor_forward_speed(left_speed=0, right_speed=0)
                    mode = Mode.PAUSE
                    turn_right_started = False
            elif mode == Mode.TEST:
                # ロール補正のみでベーススピード走行
                # left_speed, right_speed = et.calc_max_speed_with_roll_control(BASE_SPEED)
                left_speed, right_speed = 0, 0  # 仮対応
                et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
            elif mode == Mode.PAUSE:
                left_speed, right_speed = 0, 0
                et.set_motor_forward_speed(left_speed=left_speed, right_speed=right_speed)
                # 急停止フラグが残っていればゼロ速度出力＆フラグクリア
                if turn_left_started == 'reverse_stop_left':
                    turn_left_started = False
                if turn_right_started == 'reverse_stop_right':
                    turn_right_started = False

            # --- ループ周期制限とdebug出力（最後） ---
            loop_end = time.time()
            handle_debug_output(loop_start, loop_end, debug_state)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        et.stop()
        # if 'video' is used, release it
        # video.release()

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
    parser.add_argument("--course", choices=["left", "right"], default="right", help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: right)")
    parser.add_argument("--course-type", choices=["upper", "lower"], default="upper", help="Course type: 'upper' or 'lower' (default: upper)")
    parser.add_argument("--manual", action="store_true", help="Enable manual key input control mode")
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
    )
